"""Stage 1 with BPR — LightGCN unfrozen, FACE-style joint training.

This is the FACE-style variant of Stage 1: the LightGCN backbone is
unfrozen and trained jointly with the InATTo tokenizer, descriptor, and
alignment modules via a single combined loss

    L = λ_bpr · L_BPR
      + λ_recon · L_recon
      + λ_Q · L_Q
      + λ_align · L_align
      + λ_ui · L_ui
      + λ_rate · L_rate

The motivation (see results/wo_bpr_baseline.md): without BPR, the
codebook collapses to a generic abstract cluster (food-themed words
on Toys) because the frozen CF embeddings don't share dataset-aware
semantics with the codebook. With BPR continuing from the pretrained
LightGCN checkpoint, CF embeddings are pulled toward task-relevant
positions and the codebook follows via the alignment loss.

Usage:
    pixi run -- python scripts/08a_train_stage1_bpr.py --dataset toys --cuda 2

Saves:
    checkpoints/inatto/inatto-<ds>-<seed>.stage1bpr{tag_suffix}.pth
    checkpoints/inatto/inatto-<ds>-<seed>.stage1bpr{tag_suffix}.latest.pth
    data/<ds>/identifier_cache.<seed>.bpr.pkl
    results/inatto-<ds>-<seed>.stage1bpr{tag_suffix}.log.json
"""

from __future__ import annotations
import argparse
import json
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.e2e_model import InATToE2E, PHASE_WEIGHTS
from inatto.backbone.lightgcn import LightGCN
from trainer.phase_scheduler import PhaseSchedule
from data_utils.seq_loader import InATToSeqDataset, make_collate
from data_utils.adj_builder import build_torch_adj
from generative.identifier_extractor import extract_item_identifiers


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


@torch.no_grad()
def extract_user_identifiers(model, n_users, device, batch_size=512):
    model.eval()
    out: dict[int, list[int]] = {}
    lengths = np.zeros(n_users, dtype=np.int32)
    ids_all = torch.arange(n_users, dtype=torch.long, device=device)
    for s in tqdm(range(0, n_users, batch_size), desc="extract user identifiers"):
        batch_ids = ids_all[s : s + batch_size]
        tok_out = model.tokenizer(batch_ids, mode="user", hard_mask=True)
        codes = tok_out["codes"]; masks = tok_out["depth_mask"]
        for j in range(codes.shape[0]):
            seq = model.id_builder.user_token_ids(codes[j], masks[j])
            uid = int(batch_ids[j].item())
            out[uid] = seq; lengths[uid] = len(seq)
    return out, lengths


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--data_root",  type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--llm_path",   type=Path, default=Path(__file__).resolve().parent.parent / "LLMs/all-MiniLM-L6-v2")
    p.add_argument("--t5_path",    type=Path, default=Path(__file__).resolve().parent.parent / "LLMs/t5-small")
    p.add_argument("--coca_path",  type=Path, default=Path(__file__).resolve().parent.parent / "assets/word_frequency_list_60000_English.xlsx")
    p.add_argument("--ckpt_dir",   type=Path, default=Path(__file__).resolve().parent.parent / "checkpoints/inatto")
    p.add_argument("--lgn_ckpt_dir", type=Path, default=Path(__file__).resolve().parent.parent / "checkpoints/lightgcn")
    p.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--batch_size", type=int, default=512)
    p.add_argument("--lr_tokenizer", type=float, default=1e-3)
    p.add_argument("--lr_lightgcn",  type=float, default=1e-3,
                   help="LR for the LightGCN backbone (FACE uses 1e-3).")
    p.add_argument("--total_epochs", type=int, default=20)
    p.add_argument("--history_max_len", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_aspects", type=int, default=8)
    p.add_argument("--L_max", type=int, default=4)
    p.add_argument("--bpr_weight", type=float, default=1.0,
                   help="λ_BPR weight in the joint loss (FACE uses 1.0).")
    p.add_argument("--reg_weight", type=float, default=1e-4,
                   help="L2 weight decay on LightGCN embeddings.")
    p.add_argument("--rebuild_neighbors_every", type=int, default=5,
                   help="Recompute SATP k-NN this many epochs apart (O(N^2), "
                        "so don't do it every epoch).")
    p.add_argument("--resume", action="store_true")
    # ---- Ablations (for codeword-collapse diagnostics) ----
    p.add_argument("--ablate_adaptive_target", action="store_true",
                   help="Disable rho-weighted target (h_adp = h_raw, FACE convention).")
    p.add_argument("--ablate_satp", action="store_true",
                   help="Disable SATP neighbor aggregation (h_hat = h_txt).")
    p.add_argument("--tag", type=str, default="",
                   help="Optional suffix on checkpoint/log/cache filenames.")
    p.add_argument("--itm_text_embeds_file", type=str, default=None,
                   help="Override the default itm_text_embeds.pkl filename "
                        "(e.g. itm_text_embeds.s1.pkl for sparsity ablation).")
    p.add_argument("--itm_rho_file", type=str, default=None,
                   help="Override the default itm_rho.pkl filename.")
    p.add_argument("--itm_emb_np_file", type=str, default=None,
                   help="Override the default itm_emb_np.pkl filename "
                        "(GPT-profile MiniLM embedding; used as alignment target h_raw).")
    p.add_argument("--user_align_target", type=str, default="self",
                   choices=["self", "u2i"],
                   help="L_align_user target. 'self' (default) uses the "
                        "user's own GPT-profile h_raw (self-loop, since the "
                        "user profile is generated by summarizing the user's "
                        "history); 'u2i' uses the mean h_raw_item over the "
                        "user's training history (DAS-style cross alignment).")
    p.add_argument("--ssw_weight", type=float, default=0.0,
                   help="λ_SSW: >0 enables S2WTM-style spherical sliced Wasserstein "
                        "regularisation on z_aspect to prevent codeword cluster collapse.")
    p.add_argument("--ssw_n_projections", type=int, default=50,
                   help="Number of random spherical projections used by SSW.")
    p.add_argument("--use_vrvq_mask", action="store_true",
                   help="Use VRVQ Eq.7 smooth surrogate for depth_mask STE "
                        "(non-vanishing gradient → fixes phi saturation).")
    p.add_argument("--vrvq_alpha", type=float, default=4.0,
                   help="Slope parameter alpha for VRVQ surrogate.")
    p.add_argument("--use_gumbel_mask", action="store_true",
                   help="★ Gumbel-noise depth mask (training-only stochastic "
                        "STE, anti-saturation, encourages variable depth spread).")
    p.add_argument("--gumbel_tau", type=float, default=1.0,
                   help="Temperature τ for Gumbel-sigmoid depth mask.")
    p.add_argument("--gumbel_v2", action="store_true",
                   help="★ Forward-deterministic Gumbel STE (★ v2): hard mask "
                        "has no noise (train/test identical forward); only the "
                        "smooth gradient surrogate is Gumbel-perturbed.")
    # ---- IAHQ / Hierarchical RQ (Phase 2A) ----
    p.add_argument("--use_hrq", action="store_true",
                   help="Replace flat RQ with hierarchical (tree) RQ using "
                        "the AHC tree over codebook MiniLM embeddings.")
    p.add_argument("--hrq_tree_pkl", type=str,
                   default="assets/codebook_ahc_tree.pkl",
                   help="Path to AHC-tree pickle "
                        "(see scripts/11_build_codebook_ahc_tree.py).")
    p.add_argument("--hrq_parent_constraint", action="store_true",
                   help="Restrict level-l candidates to children(c_{l-1}).")
    p.add_argument("--phi_init_bias", type=float, default=0.0,
                   help="Bias of the Ep MLP's final linear. 0 → phi_init=0.5 "
                        "(spec default). 5 → phi_init≈0.99 (force-deep start).")
    # ---- IAHQ Phase 2E: ELCRec prototypes + C-TF-IDF representative ----
    p.add_argument("--hrq_elcrec_proto", action="store_true",
                   help="Use trainable per-level ELCRec cluster prototypes "
                        "(ParameterList) instead of frozen W_c(mean) anchors. "
                        "Single W_c (SimVQ) preserved on the vocabulary side.")
    p.add_argument("--hrq_ctfidf_repr", action="store_true",
                   help="Use C-TF-IDF representative leaf (semantic discriminator) "
                        "instead of nearest-center leaf.")
    p.add_argument("--sep_weight", type=float, default=0.0,
                   help="Loss weight for the ELCRec separation regularizer.")
    p.add_argument("--phi_anchor_w", type=float, default=0.0,
                   help="★ φ δ-anchor loss weight.")
    # ---- Tree-guide reg (LETTER-style pull-push on CF-supervised hierarchy) ----
    p.add_argument("--tree_reg_w", type=float, default=0.0,
                   help="Loss weight for tree-guide pull-push regulariser "
                        "(disabled when 0). LETTER's diversity reg generalised "
                        "to a CF-supervised semantic hierarchy.")
    p.add_argument("--tree_pkl", type=str,
                   default="assets/codebook_ahc_tree_cf.pkl",
                   help="Path to AHC tree pickle (level-1 cluster_ids used). "
                        "Prefer the CF-supervised version "
                        "(scripts/12_build_cf_aware_tree.py output).")
    # ---- Stage-1 loss ablations (paper §4) ----
    # Each flag zeros out a specific term in the warmup-phase loss to
    # isolate its contribution.  Use one at a time.
    p.add_argument("--ablate_ui", action="store_true",
                   help="Zero out L_ui (cross-side user↔item InfoNCE).")
    p.add_argument("--ablate_align", action="store_true",
                   help="Zero out L_align (descriptor↔raw InfoNCE).")
    p.add_argument("--rate_weight", type=float, default=None,
                   help="Override lambda_rate (default uses PHASE_WEIGHTS warmup: "
                        "0.0005). VRVQ recommends ~0.05-0.1 to suppress phi saturation.")
    args = p.parse_args()

    _set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    tag_suffix = f".{args.tag}" if args.tag else ""
    print(f"[args] {vars(args)}")
    print(f"[device] {device}  tag_suffix='{tag_suffix}'")

    ddir = args.data_root / args.dataset
    print(f"[data] loading buffers from {ddir}")
    itm_text_embeds_file = getattr(args, "itm_text_embeds_file", None) or "itm_text_embeds.pkl"
    itm_rho_file        = getattr(args, "itm_rho_file", None) or "itm_rho.pkl"
    itm_emb_np_file     = getattr(args, "itm_emb_np_file", None) or "itm_emb_np.pkl"
    print(f"[data] itm_text_embeds = {itm_text_embeds_file}")
    print(f"[data] itm_rho         = {itm_rho_file}")
    print(f"[data] itm_emb_np      = {itm_emb_np_file}")
    h_txt_item = torch.tensor(_load(ddir / itm_text_embeds_file)).float()
    n_items = h_txt_item.shape[0]
    itm_rho = torch.tensor(_load(ddir / itm_rho_file)).float()
    usr_rho = torch.tensor(_load(ddir / "usr_rho.pkl")).float()
    itm_emb_np = _load(ddir / itm_emb_np_file)
    usr_emb_np = _load(ddir / "usr_emb_np.pkl")
    h_raw_item = torch.tensor(np.asarray(itm_emb_np)).float()
    h_raw_user = torch.tensor(np.asarray(usr_emb_np)).float()
    h_txt_user = h_raw_user.clone()
    n_users = h_raw_user.shape[0]

    # ---- Optional u2i user-side alignment target ----
    # Replaces self-loop L_align_user (target = h_raw_user) with a target
    # built from the mean of the h_raw_item over each user's training history.
    # This injects collaborative signal into the user codebook (DAS-style u2i)
    # and breaks the user self-loop that would otherwise just re-encode the
    # GPT profile (which is already a summary of the same history).
    h_align_target_user = None
    if getattr(args, "user_align_target", "self") == "u2i":
        import torch.nn.functional as F
        user_train_history = _load(ddir / "user_train_history.pkl")
        print(f"[u2i] computing user alignment target from history "
              f"(mean h_raw_item over {n_users} users) ...")
        h_align_target_user = h_raw_user.clone()
        n_fallback = 0
        for uid, hist in enumerate(user_train_history):
            if hist and len(hist) > 0:
                hist_t = torch.tensor(hist, dtype=torch.long)
                h_align_target_user[uid] = h_raw_item[hist_t].mean(dim=0)
            else:
                # fallback: keep self-loop target
                n_fallback += 1
        h_align_target_user = F.normalize(h_align_target_user, p=2, dim=1)
        print(f"[u2i] done. fallback (empty history) users: {n_fallback}/{n_users}")

    # ---- Load pretrained LightGCN backbone (TRAINABLE in this script) ----
    lgn_path = args.lgn_ckpt_dir / f"lightgcn-{args.dataset}-{args.seed}.pth"
    print(f"[backbone] loading {lgn_path}  (TRAINABLE — BPR-Stage-1 mode)")
    state = torch.load(lgn_path, map_location="cpu", weights_only=False)
    embedding_size = int(state["embedding_size"])
    layer_num = int(state.get("layer_num", 3))
    lightgcn = LightGCN(n_users=n_users, n_items=n_items,
                         embedding_size=embedding_size, layer_num=layer_num)
    with torch.no_grad():
        lightgcn.user_embeds.data.copy_(state["user_embeds"].float())
        lightgcn.item_embeds.data.copy_(state["item_embeds"].float())
    lightgcn = lightgcn.to(device)

    # Build the normalised user-item adjacency for GCN propagation.
    trn_mat = _load(ddir / "trn_mat.pkl")
    adj = build_torch_adj(trn_mat, n_users, n_items, device)
    print(f"[adj] built, nnz={adj._nnz()}")

    # Snapshot pretrained CF embeddings to seed the InATTo buffers.
    with torch.no_grad():
        u0, i0 = lightgcn.propagate(adj)

    # ---- Build InATToE2E (initial buffers = current LightGCN propagation) ----
    cfg = dict(history_max_len=args.history_max_len,
                n_aspects=args.n_aspects, L_max=args.L_max,
                ablate_adaptive_target=args.ablate_adaptive_target,
                ablate_satp=args.ablate_satp,
                ssw_weight=args.ssw_weight,
                ssw_n_projections=args.ssw_n_projections,
                use_vrvq_mask=args.use_vrvq_mask,
                vrvq_alpha=args.vrvq_alpha,
                use_gumbel_mask=args.use_gumbel_mask,
                gumbel_tau=args.gumbel_tau,
                gumbel_v2=args.gumbel_v2,
                use_hrq=args.use_hrq,
                hrq_tree_pkl=args.hrq_tree_pkl,
                hrq_parent_constraint=args.hrq_parent_constraint,
                hrq_elcrec_proto=args.hrq_elcrec_proto,
                hrq_ctfidf_repr=args.hrq_ctfidf_repr,
                sep_weight=args.sep_weight,
                phi_anchor_w=args.phi_anchor_w,
                phi_init_bias=args.phi_init_bias,
                tree_reg_w=args.tree_reg_w,
                tree_pkl=args.tree_pkl)
    model = InATToE2E(
        user_buffers=(u0.detach(), h_txt_user, usr_rho, h_raw_user),
        item_buffers=(i0.detach(), h_txt_item, itm_rho, h_raw_item),
        llm_path=args.llm_path, t5_path=args.t5_path,
        coca_path=args.coca_path, cfg=cfg,
        h_align_target_user=h_align_target_user,
    ).to(device)
    print(f"[model] codebook V={model.codebook.V}  n_aspects={model.cfg['n_aspects']}  "
          f"L_max={model.cfg['L_max']}  d_t5={model.bridge.d_t5}")

    # ---- Data loaders (reuse the seq dataset; sample negatives in-batch) ----
    train_ds = InATToSeqDataset(args.data_root, args.dataset, "train",
                                 history_max_len=args.history_max_len)
    val_ds   = InATToSeqDataset(args.data_root, args.dataset, "val",
                                 history_max_len=args.history_max_len)
    coll = make_collate(args.history_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=coll, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=coll, pin_memory=True)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # ---- Optimizer (LightGCN + tokenizer + alignment as one trainable set, T5 frozen) ----
    # Freeze T5 (Stage 1 doesn't train it).
    for p in model.bridge.t5.parameters(): p.requires_grad = False
    # Descriptor MiniLM is always frozen by construction.

    optim = torch.optim.AdamW([
        {"params": lightgcn.parameters(),                 "lr": args.lr_lightgcn,
         "weight_decay": 2 * args.reg_weight},
        {"params": (list(model.tokenizer.parameters())
                    + list(model.align_user.parameters())
                    + list(model.align_item.parameters())),
         "lr": args.lr_tokenizer, "weight_decay": 0.0},
    ])

    # ---- Resume ----
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage1bpr{tag_suffix}.latest.pth"
    log_path    = args.results_dir / f"inatto-{args.dataset}-{args.seed}.stage1bpr{tag_suffix}.log.json"
    history: list[dict] = []
    start_epoch = 0
    if args.resume and latest_ckpt.exists():
        print(f"[resume] {latest_ckpt}")
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        lightgcn.load_state_dict(ck["lightgcn_state"])
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        history = ck.get("history", [])
        start_epoch = ck.get("next_epoch", len(history))
        print(f"[resume] from epoch {start_epoch}")

    # ---- Train loop ----
    w = dict(PHASE_WEIGHTS["warmup"])   # copy so we can override lambda_rate
    if args.rate_weight is not None:
        print(f"[rate] overriding λ_rate {w['rate']} -> {args.rate_weight} (VRVQ-strength)")
        w["rate"] = args.rate_weight
    for epoch in range(start_epoch, args.total_epochs):
        t0 = time.time()
        model.train(); lightgcn.train()

        # Refresh SATP neighbors periodically (k-NN recomputed on current z).
        if (epoch % max(1, args.rebuild_neighbors_every)) == 0:
            with torch.no_grad():
                u_final, i_final = lightgcn.propagate(adj)
                model.tokenizer.refresh_z('user', u_final, rebuild_neighbors=True)
                model.tokenizer.refresh_z('item', i_final, rebuild_neighbors=True)

        # Accumulators
        agg = {k: 0.0 for k in
               ["L_total", "L_bpr", "L_recon", "L_Q", "L_align", "L_ui", "L_rate", "L_ssw"]}
        agg["phi_user_mean"] = 0.0; agg["phi_item_mean"] = 0.0
        n_batches = 0

        for uids, tgts, hist, valid in tqdm(train_loader, desc=f"ep {epoch}", leave=False):
            uids  = uids.to(device, non_blocking=True)
            tgts  = tgts.to(device, non_blocking=True)
            hist  = hist.to(device, non_blocking=True)
            valid = valid.to(device, non_blocking=True)

            # GCN propagation (full users + items).
            u_final, i_final = lightgcn.propagate(adj)

            # Bind live CF embeddings into the InATTo tokenizer (autograd OK).
            model.tokenizer.refresh_z('user', u_final, rebuild_neighbors=False)
            model.tokenizer.refresh_z('item', i_final, rebuild_neighbors=False)

            # BPR: negative item per anchor sampled uniformly (drop the rare
            # accidental neg==pos collisions implicitly — they're harmless on
            # average).
            neg = torch.randint(0, n_items, (uids.shape[0],), device=device)
            anc_e = u_final[uids]; pos_e = i_final[tgts]; neg_e = i_final[neg]
            L_bpr = LightGCN.bpr_loss(anc_e, pos_e, neg_e)

            # InATTo forward (warmup phase weights: gen=0; tokenizer/align trained).
            # The forward path itself adds ssw_weight·L_ssw into L_total, but we
            # rebuild L_total here so the BPR term is included on top.
            out = model(uids, tgts, hist, valid, phase='warmup')
            L_recon = out["L_recon"]; L_Q = out["L_Q"]
            L_align = out["L_align"]; L_ui = out["L_ui"]; L_rate = out["L_rate"]
            L_ssw   = out["L_ssw"]

            # Ablation overrides (paper §4 loss-term ablation table)
            align_w = 0.0 if args.ablate_align else w["align"]
            ui_w    = 0.0 if args.ablate_ui    else w["ui"]
            L_total = (args.bpr_weight * L_bpr
                       + w["recon"] * L_recon
                       + w["Q"]     * L_Q
                       + align_w    * L_align
                       + ui_w       * L_ui
                       + w["rate"]  * L_rate
                       + args.ssw_weight * L_ssw)

            optim.zero_grad()
            L_total.backward()
            optim.step()

            for k, v in [("L_total", L_total), ("L_bpr", L_bpr), ("L_recon", L_recon),
                         ("L_Q", L_Q), ("L_align", L_align), ("L_ui", L_ui),
                         ("L_rate", L_rate), ("L_ssw", L_ssw)]:
                agg[k] += float(v.item())
            agg["phi_user_mean"] += float(out["phi_user"].mean().item())
            agg["phi_item_mean"] += float(out["phi_item"].mean().item())
            n_batches += 1

        for k in agg: agg[k] /= max(1, n_batches)
        log = {"epoch": epoch, "epoch_time": time.time() - t0, **agg}
        print(f"[ep {epoch:>2}/{args.total_epochs}] "
              f"L={log['L_total']:.3f}  bpr={log['L_bpr']:.3f}  recon={log['L_recon']:.4f}  "
              f"Q={log['L_Q']:.3f}  align={log['L_align']:.3f}  ui={log['L_ui']:.3f}  "
              f"rate={log['L_rate']:.3f}  ssw={log['L_ssw']:.4f}  phi_u={log['phi_user_mean']:.3f}  "
              f"phi_i={log['phi_item_mean']:.3f}  ({log['epoch_time']:.1f}s)")
        history.append(log)
        Path(log_path).write_text(json.dumps(history, indent=2))

        torch.save({
            "lightgcn_state": lightgcn.state_dict(),
            "model_state":    model.state_dict(),
            "optim_state":    optim.state_dict(),
            "history":        history,
            "next_epoch":     epoch + 1,
        }, latest_ckpt)

    # ---- Final ckpt ----
    ckpt = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage1bpr{tag_suffix}.pth"
    torch.save({
        "lightgcn_state": lightgcn.state_dict(),
        "model_state":    model.state_dict(),
        "cfg":            cfg,
        "history":        history,
        "dataset":        args.dataset,
        "seed":           args.seed,
        "stage":          "1bpr",
    }, ckpt)
    print(f"[done] stage-1bpr checkpoint -> {ckpt}")

    # ---- Extract identifiers ----
    # Rebuild neighbors one last time on the final LightGCN.
    with torch.no_grad():
        u_final, i_final = lightgcn.propagate(adj)
        model.tokenizer.refresh_z('user', u_final, rebuild_neighbors=True)
        model.tokenizer.refresh_z('item', i_final, rebuild_neighbors=True)

    print("[extract] item identifiers ...")
    item_ids, item_lens = extract_item_identifiers(model, n_items, device, batch_size=512)
    print(f"  items={len(item_ids)}  min={item_lens.min()}  max={item_lens.max()}  "
          f"mean={item_lens.mean():.2f}")

    print("[extract] user identifiers ...")
    user_ids, user_lens = extract_user_identifiers(model, n_users, device, batch_size=512)
    print(f"  users={len(user_ids)}  min={user_lens.min()}  max={user_lens.max()}  "
          f"mean={user_lens.mean():.2f}")

    cache_path = ddir / f"identifier_cache.{args.seed}.bpr{tag_suffix}.pkl"
    with cache_path.open("wb") as f:
        pickle.dump({
            "item": item_ids,
            "user": user_ids,
            "item_lengths": item_lens,
            "user_lengths": user_lens,
            "cfg": {
                "n_aspects":  args.n_aspects,
                "L_max":      args.L_max,
                "V":          model.codebook.V,
                "d_t5":       model.bridge.d_t5,
                "special_ids":model.bridge.special_ids,
                "code_to_t5": model.bridge.code_to_t5.cpu().tolist(),
                "variant":    "bpr_stage1",
            },
        }, f)
    print(f"[saved] identifier cache -> {cache_path}")


if __name__ == "__main__":
    main()
