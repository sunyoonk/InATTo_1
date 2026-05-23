"""Analyse φ distribution from a trained Stage-1 tokenizer.

The IAHQ hypothesis is that hierarchical structure should make per-item
gain curves item-dependent, so φ should *spread* (not just have a low
mean): sparse items should land at low φ (shallow), rich items at high
φ (deep).

We:
  1. Load a Stage-1 checkpoint (with or without HRQ).
  2. Run a forward pass on every user / every item to dump φ per side.
  3. Report:
       - φ histogram (means / percentiles / std)
       - φ vs raw text word count (Spearman correlation)
       - φ split by sparsity quartile (sparse Q1 vs rich Q4 mean φ)
       - φ vs ρ (text entropy) and vs δ (CF-text gap)

Usage:
  pixi run -- python scripts/12_analyse_phi_spread.py \
      --dataset toys --cuda 2 \
      --tokenizer_ckpt checkpoints/inatto/inatto-toys-2023.stage1bpr.ssw01.hrq.pth \
      --use_hrq --hrq_tree_pkl assets/codebook_ahc_tree.pkl
"""

from __future__ import annotations
import argparse
import pickle
import re
import sys
from pathlib import Path
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.e2e_model import InATToE2E


_TITLE_RX    = re.compile(r"title\s*:\s*([^;]+)", re.IGNORECASE)
_CATEGORY_RX = re.compile(r"categor(?:y|ies)\s*:\s*([^;]+)", re.IGNORECASE)


def raw_word_count(text: str) -> int:
    if not text:
        return 0
    title = _TITLE_RX.search(text)
    cats  = _CATEGORY_RX.search(text)
    parts = []
    if title: parts.append(title.group(1).strip())
    if cats:  parts.append(cats.group(1).strip())
    return sum(len(p.split()) for p in parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--data_root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--tokenizer_ckpt", type=Path, required=True)
    p.add_argument("--llm_path", type=str,
                    default="LLMs/all-MiniLM-L6-v2")
    p.add_argument("--t5_path", type=str,
                    default="LLMs/t5-small")
    p.add_argument("--coca_path", type=str,
                    default="assets/word_frequency_list_60000_English.xlsx")
    p.add_argument("--use_hrq", action="store_true")
    p.add_argument("--hrq_tree_pkl", type=str,
                    default="assets/codebook_ahc_tree.pkl")
    p.add_argument("--batch", type=int, default=512)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    ddir = args.data_root / args.dataset

    print(f"[load] data from {ddir}")
    with open(ddir / "itm_text.pkl", "rb") as f: itm_text = pickle.load(f)
    with open(ddir / "itm_text_embeds.pkl", "rb") as f: h_txt_item = torch.tensor(pickle.load(f), dtype=torch.float32)
    with open(ddir / "itm_emb_np.pkl", "rb") as f: h_raw_item = torch.tensor(pickle.load(f), dtype=torch.float32)
    with open(ddir / "itm_rho.pkl", "rb") as f: itm_rho = torch.tensor(pickle.load(f), dtype=torch.float32)
    with open(ddir / "usr_emb_np.pkl", "rb") as f:
        # user profile embeddings (target for alignment); we also need a CF embedding
        h_raw_user = torch.tensor(pickle.load(f), dtype=torch.float32)
    # user text embeddings: usr_prf doesn't have MiniLM embeds for user_text the
    # same way; reuse h_raw_user as h_txt_user (matches Stage-1 training setup
    # where SATP for user uses profile embeddings).
    h_txt_user = h_raw_user.clone()
    with open(ddir / "usr_rho.pkl", "rb") as f:
        usr_rho = torch.tensor(pickle.load(f), dtype=torch.float32)

    # CF embeddings (LightGCN propagated, frozen at the time of ckpt save).
    # The Stage-1 ckpt contains the LightGCN state, so we propagate once.
    ck = torch.load(args.tokenizer_ckpt, map_location=device, weights_only=False)
    # Build the model exactly like 08a does
    n_users = h_raw_user.shape[0]
    n_items = h_raw_item.shape[0]
    cfg = ck.get("cfg", {})
    if not cfg:
        cfg = dict(history_max_len=50, n_aspects=8, L_max=4,
                    ablate_adaptive_target=True, ablate_satp=False,
                    ssw_weight=0.1, ssw_n_projections=50,
                    use_vrvq_mask=False, vrvq_alpha=4.0,
                    use_hrq=args.use_hrq, hrq_tree_pkl=args.hrq_tree_pkl,
                    hrq_parent_constraint=False,
                    transformer_layers=1, transformer_heads=1,
                    transformer_dropout=0.0,
                    ablate_ep_closed_form=False, ablate_no_rho=False,
                    ablate_no_delta=False, ablate_user_fixed_depth=False,
                    L_align_w=0.5, L_ui_w=0.1, L_recon_w=1.0,
                    L_Q_w=1.0, L_rate_w=0.0005, K_neighbors=10,
                    d_aspect=256, phi_hidden=64, alpha_ste=2.0,
                    commit_beta=0.25)

    # Initial z buffers — placeholder at LightGCN's d_cf=256 dim (replaced
    # by a real propagation after model load).
    d_cf = 256
    z_user_init = torch.zeros(n_users, d_cf)
    z_item_init = torch.zeros(n_items, d_cf)

    model = InATToE2E(
        user_buffers=(z_user_init, h_txt_user, usr_rho, h_raw_user),
        item_buffers=(z_item_init, h_txt_item, itm_rho, h_raw_item),
        llm_path=args.llm_path, t5_path=args.t5_path,
        coca_path=args.coca_path, cfg=cfg,
    ).to(device)
    missing, unexpected = model.load_state_dict(ck["model_state"], strict=False)
    if missing:
        print(f"  [warn] missing keys: {len(missing)}  (e.g., {missing[:3]})")
    if unexpected:
        print(f"  [warn] unexpected keys: {len(unexpected)}  (e.g., {unexpected[:3]})")

    # Run a one-shot LightGCN propagation to set real z buffers.
    with torch.no_grad():
        model.backbone.eval()
        zall = model.backbone.propagate()
        z_user = zall[: n_users]
        z_item = zall[n_users:]
        model.tokenizer.refresh_z("user", z_user, rebuild_neighbors=False)
        model.tokenizer.refresh_z("item", z_item, rebuild_neighbors=False)

    # ---- Dump per-item phi (using model.tokenizer.forward) ----
    model.eval()
    phis_item = []
    rhos      = []
    deltas    = []
    with torch.no_grad():
        for s in range(0, n_items, args.batch):
            ids = torch.arange(s, min(s + args.batch, n_items), device=device)
            out = model.tokenizer(ids, mode="item")
            phis_item.append(out["signals"]["phi"].cpu())
            rhos.append(out["signals"]["rho"].cpu())
            deltas.append(out["signals"]["delta"].cpu())
    phi_item  = torch.cat(phis_item, dim=0)              # (n_items, n_aspects)
    rho_item  = torch.cat(rhos, dim=0)                    # (n_items,)
    delta_item = torch.cat(deltas, dim=0)                 # (n_items,)

    # Per-item: mean phi across aspects (a single "depth budget" per item)
    phi_per_item = phi_item.mean(dim=-1).numpy()          # (n_items,)
    rho_arr   = rho_item.numpy()
    delta_arr = delta_item.numpy()

    # ---- Raw word count from itm_text ----
    wc = np.array([raw_word_count(itm_text.get(i, "")) for i in range(n_items)])

    print()
    print("=" * 80)
    print(f"=== φ_item per-item summary (mean over {phi_item.size(-1)} aspects) ===")
    p = phi_per_item
    print(f"  mean    = {p.mean():.4f}")
    print(f"  std     = {p.std():.4f}    ★ key 'spread' signal")
    print(f"  min,max = {p.min():.4f} , {p.max():.4f}")
    print(f"  10/25/50/75/90 pct = "
           f"{np.quantile(p, 0.10):.4f} / {np.quantile(p, 0.25):.4f} / "
           f"{np.quantile(p, 0.50):.4f} / {np.quantile(p, 0.75):.4f} / "
           f"{np.quantile(p, 0.90):.4f}")

    # Histogram bins
    bins = np.linspace(0, 1, 11)
    hist, _ = np.histogram(p, bins=bins)
    print(f"  histogram (bins 0.0-1.0 step 0.1):")
    for b in range(10):
        bar = "█" * int(hist[b] / max(hist) * 50)
        print(f"    [{bins[b]:.1f},{bins[b+1]:.1f})  {hist[b]:>5}  {bar}")

    # ---- Spread by sparsity-quartile (word count) ----
    qs = np.quantile(wc, [0.25, 0.50, 0.75])
    grp = np.digitize(wc, qs)              # 0,1,2,3
    print()
    print(f"=== φ_item by raw-text quartile (Q1=sparse, Q4=rich) ===")
    print(f"{'Quartile':<10}{'n_items':>10}{'wc range':>14}{'φ_mean':>10}{'φ_std':>10}")
    for q in range(4):
        sel = grp == q
        if sel.sum() == 0: continue
        lo = wc[sel].min(); hi = wc[sel].max()
        print(f"  Q{q+1:<7}{int(sel.sum()):>10}"
               f"  [{lo:>3}-{hi:>3}]"
               f"{p[sel].mean():>10.4f}{p[sel].std():>10.4f}")

    # Spearman corr (we'll just rank-correlate manually)
    def spearman(x, y):
        from scipy.stats import spearmanr
        return spearmanr(x, y).correlation
    try:
        sp_wc    = spearman(p, wc)
        sp_rho   = spearman(p, rho_arr)
        sp_delta = spearman(p, delta_arr)
        print()
        print(f"=== rank correlations (Spearman) ===")
        print(f"  φ vs raw-text-word-count : {sp_wc:+.4f}    (★ positive ⇒ rich = deeper)")
        print(f"  φ vs ρ (text entropy)    : {sp_rho:+.4f}")
        print(f"  φ vs δ (CF-text gap)     : {sp_delta:+.4f}")
    except ImportError:
        pass


if __name__ == "__main__":
    main()
