"""FACE baseline (C-option) driver — 3-phase end-to-end training using the
FACE original tokenizer inside our T5 pipeline.

Mirrors ``08_train_e2e.py`` but uses ``FACEE2E`` instead of ``InATToE2E``.
The CF backbone is loaded *frozen* from the LightGCN checkpoint produced
by ``07_train_lightgcn.py`` (same checkpoint InATTo uses), so BPR loss
is intentionally dropped — only L_gen / L_recon / L_Q / L_align contribute.

Usage:
    pixi run -- python scripts/08_train_face.py --dataset toys --cuda 2

Saves:
    checkpoints/face/face-<dataset>-<seed>.pth
    results/face-<dataset>-<seed>.log.json
"""

from __future__ import annotations
import argparse
import pickle
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.face_e2e_model import FACEE2E
from trainer.phase_scheduler import PhaseSchedule
from trainer.trainer_e2e import E2ETrainer
from data_utils.seq_loader import InATToSeqDataset, make_collate


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--data_root",  type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--t5_path",    type=Path,
                   default=Path(__file__).resolve().parent.parent / "LLMs/t5-small")
    p.add_argument("--face_root",  type=Path,
                   default=Path(__file__).resolve().parents[2] / "FACE")
    p.add_argument("--ckpt_dir",   type=Path,
                   default=Path(__file__).resolve().parent.parent / "checkpoints/face")
    p.add_argument("--lgn_ckpt_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "checkpoints/lightgcn")
    p.add_argument("--results_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--batch_size", type=int, default=128,
                   help="Default 128 avoids OOM at the joint phase when T5 "
                        "becomes trainable. Increase if your GPU has headroom.")
    p.add_argument("--lr_tokenizer", type=float, default=1e-3)
    p.add_argument("--lr_t5", type=float, default=1e-4)
    p.add_argument("--warmup_end", type=int, default=20)
    p.add_argument("--joint_end", type=int, default=60)
    p.add_argument("--total_epochs", type=int, default=80)
    p.add_argument("--history_max_len", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--n_aspects", type=int, default=8,
                   help="FACE original 'word_num' (8 in the FACE paper).")
    p.add_argument("--d_aspect", type=int, default=256,
                   help="FACE original 'word_dim'.")
    p.add_argument("--align_temperature", type=float, default=0.02,
                   help="FACE original InfoNCE temperature (loss_utils.py:5).")
    p.add_argument("--resume", action="store_true",
                   help="Resume from the latest per-epoch checkpoint if present.")
    args = p.parse_args()

    _set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[args] {vars(args)}")
    print(f"[device] {device}")

    # ---- Load buffers ----
    ddir = args.data_root / args.dataset
    print(f"[data] loading buffers from {ddir}")
    h_txt_item = torch.tensor(_load(ddir / "itm_text_embeds.pkl")).float()
    n_items = h_txt_item.shape[0]
    itm_rho = torch.tensor(_load(ddir / "itm_rho.pkl")).float()
    usr_rho = torch.tensor(_load(ddir / "usr_rho.pkl")).float()
    itm_emb_np = _load(ddir / "itm_emb_np.pkl")
    usr_emb_np = _load(ddir / "usr_emb_np.pkl")
    h_raw_item = torch.tensor(np.asarray(itm_emb_np)).float()
    h_raw_user = torch.tensor(np.asarray(usr_emb_np)).float()
    h_txt_user = h_raw_user.clone()
    n_users = h_raw_user.shape[0]

    # ---- Load frozen LightGCN CF backbone ----
    lgn_path = args.lgn_ckpt_dir / f"lightgcn-{args.dataset}-{args.seed}.pth"
    print(f"[backbone] loading {lgn_path}  (frozen)")
    state = torch.load(lgn_path, map_location="cpu", weights_only=False)
    z_user = state["user_embeds"].float()
    z_item = state["item_embeds"].float()
    assert z_user.shape == (n_users, state["embedding_size"]), \
        f"user_embeds shape {z_user.shape}"
    assert z_item.shape == (n_items, state["embedding_size"])

    # ---- Build FACE-E2E ----
    cfg = dict(
        history_max_len=args.history_max_len,
        n_aspects=args.n_aspects,
        d_aspect=args.d_aspect,
        align_temperature=args.align_temperature,
        llm_name="miniLM",
    )
    model = FACEE2E(
        user_buffers=(z_user, h_txt_user, usr_rho, h_raw_user),
        item_buffers=(z_item, h_txt_item, itm_rho, h_raw_item),
        t5_path=args.t5_path,
        dataset_name=args.dataset,
        face_root=args.face_root,
        cfg=cfg,
    )
    print(f"[model] FACE codebook V={model.tokenizer.V}  "
          f"n_aspects={model.tokenizer.n_aspects}  "
          f"d_aspect={model.tokenizer.d_aspect}  "
          f"d_t5={model.bridge.d_t5}")

    # ---- Data loaders ----
    train_ds = InATToSeqDataset(args.data_root, args.dataset, "train",
                                 history_max_len=args.history_max_len)
    val_ds   = InATToSeqDataset(args.data_root, args.dataset, "val",
                                 history_max_len=args.history_max_len)
    coll = make_collate(args.history_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=coll,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=coll,
                               pin_memory=True)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}")

    # ---- Train ----
    schedule = PhaseSchedule(
        warmup_end=args.warmup_end, joint_end=args.joint_end,
        total_epochs=args.total_epochs,
    )
    trainer = E2ETrainer(
        model, schedule,
        lr_tokenizer=args.lr_tokenizer, lr_t5=args.lr_t5,
        device=device,
    )

    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / f"face-{args.dataset}-{args.seed}.log.json"

    # Per-epoch rolling checkpoint so OOM/crashes don't lose progress.
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = args.ckpt_dir / f"face-{args.dataset}-{args.seed}.latest.pth"
    resume_from = latest_ckpt if args.resume and latest_ckpt.exists() else None
    history = trainer.fit(
        train_loader, val_loader,
        log_path=log_path,
        ckpt_save_path=latest_ckpt,
        save_every=1,
        resume_from=resume_from,
    )

    # ---- Final checkpoint ----
    ckpt = args.ckpt_dir / f"face-{args.dataset}-{args.seed}.pth"
    torch.save({
        "model_state": model.state_dict(),
        "cfg": cfg,
        "history": history,
        "dataset": args.dataset,
        "seed": args.seed,
    }, ckpt)
    print(f"[done] checkpoint -> {ckpt}")


if __name__ == "__main__":
    main()
