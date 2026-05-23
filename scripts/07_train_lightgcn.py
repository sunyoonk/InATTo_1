"""Phase 0c — train LightGCN backbone on one dataset.

Produces:
    checkpoints/lightgcn/lightgcn-<dataset>-<seed>.pth

Holds:
    'user_embeds', 'item_embeds', 'n_users', 'n_items',
    'embedding_size', 'layer_num', 'seed', 'best_val_recall20', 'epoch'

Usage:
    pixi run -- python scripts/07_train_lightgcn.py --dataset beauty --cuda 2
    pixi run -- python scripts/07_train_lightgcn.py --dataset toys   --cuda 2
"""

from __future__ import annotations
import argparse
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.adj_builder import build_torch_adj
from data_utils.bpr_dataset import BPRTripletDataset, collate_bpr
from inatto.backbone.lightgcn import LightGCN


# ---------- Eval helpers ----------------------------------------------------

def _make_trn_mask(trn_csr):
    """Returns a function batch_users -> torch [B, n_items] mask of trained items."""
    def mask_fn(users_np):
        B = len(users_np)
        out = np.ones((B, trn_csr.shape[1]), dtype=np.float32)
        for r, u in enumerate(users_np):
            start, end = trn_csr.indptr[u], trn_csr.indptr[u + 1]
            out[r, trn_csr.indices[start:end]] = 0.0
        return out
    return mask_fn


@torch.no_grad()
def evaluate(
    model: LightGCN,
    adj: torch.Tensor,
    trn_csr,
    eval_coo,
    ks: list[int],
    device: torch.device,
    batch_size: int = 1024,
) -> dict[str, float]:
    model.eval()
    u_final, i_final = model.propagate(adj)
    mask_fn = _make_trn_mask(trn_csr)

    # Group eval targets by user
    coo = eval_coo.tocoo()
    targets: dict[int, list[int]] = {}
    for u, i in zip(coo.row, coo.col):
        targets.setdefault(int(u), []).append(int(i))

    users = sorted(targets.keys())
    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    for s in range(0, len(users), batch_size):
        batch_users = users[s : s + batch_size]
        u_np = np.asarray(batch_users, dtype=np.int64)
        scores = u_final[u_np] @ i_final.t()
        mask = torch.from_numpy(mask_fn(u_np)).to(device)
        scores = scores * mask - 1e9 * (1 - mask)
        max_k = max(ks)
        _, topk = torch.topk(scores, max_k, dim=1)
        topk = topk.cpu().numpy()
        for r, u in enumerate(batch_users):
            t = set(targets[u])
            preds = topk[r]
            hits = np.array([1.0 if int(p) in t else 0.0 for p in preds])
            for k in ks:
                rec_k = hits[:k].sum() / len(t)
                # IDCG with all hits at top (relevant items can be > k)
                rels = hits[:k]
                gains = (2 ** rels - 1)
                discounts = 1.0 / np.log2(np.arange(2, k + 2))
                dcg = (gains * discounts).sum()
                ideal_hits = min(len(t), k)
                ideal = ((2 ** np.ones(ideal_hits) - 1) * (1.0 / np.log2(np.arange(2, ideal_hits + 2)))).sum() \
                        if ideal_hits > 0 else 1.0
                ndcg_k = dcg / ideal if ideal > 0 else 0.0
                recalls[k].append(rec_k)
                ndcgs[k].append(ndcg_k)

    return {
        **{f"recall@{k}": float(np.mean(recalls[k])) for k in ks},
        **{f"ndcg@{k}":   float(np.mean(ndcgs[k]))   for k in ks},
    }


# ---------- Main ------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--ckpt_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "checkpoints/lightgcn")
    p.add_argument("--dataset", type=str, required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--embedding_size", type=int, default=256)
    p.add_argument("--layer_num", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--reg_weight", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=4096)
    p.add_argument("--epochs", type=int, default=500)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--eval_every", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    args = p.parse_args()

    # Seed
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    torch.backends.cudnn.deterministic = True

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}  dataset={args.dataset}  seed={args.seed}")

    # ---- Data ----
    ddir = args.data_root / args.dataset
    with (ddir / "trn_mat.pkl").open("rb") as f:
        trn = pickle.load(f)
    with (ddir / "val_mat.pkl").open("rb") as f:
        val = pickle.load(f)
    with (ddir / "tst_mat.pkl").open("rb") as f:
        tst = pickle.load(f)
    n_users, n_items = trn.shape
    print(f"n_users={n_users}  n_items={n_items}  n_train={trn.nnz}")
    trn_csr = trn.tocsr()

    adj = build_torch_adj(trn, n_users, n_items, device)

    train_ds = BPRTripletDataset(trn, n_items, seed=args.seed)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        num_workers=args.num_workers, collate_fn=collate_bpr, pin_memory=True,
    )

    # ---- Model ----
    model = LightGCN(n_users, n_items, args.embedding_size, args.layer_num).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=args.lr,
                           weight_decay=2 * args.reg_weight)
    print(f"params={sum(p.numel() for p in model.parameters()):,}")

    # ---- Train loop ----
    best = {"recall@20": -1.0}
    best_epoch = -1
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_path = args.ckpt_dir / f"lightgcn-{args.dataset}-{args.seed}.pth"

    for epoch in range(args.epochs):
        model.train()
        train_ds.resample()
        epoch_loss = 0.0
        n_batches = 0
        t0 = time.time()
        for u, pi, ni in train_loader:
            u = u.to(device, non_blocking=True)
            pi = pi.to(device, non_blocking=True)
            ni = ni.to(device, non_blocking=True)
            u_e, i_e = model.propagate(adj)
            loss = LightGCN.bpr_loss(u_e[u], i_e[pi], i_e[ni])
            opt.zero_grad()
            loss.backward()
            opt.step()
            epoch_loss += loss.item()
            n_batches += 1
        epoch_loss /= max(n_batches, 1)
        dt = time.time() - t0

        # Eval
        if (epoch + 1) % args.eval_every == 0:
            val_metrics = evaluate(model, adj, trn_csr, val, [5, 10, 20], device)
            print(f"[{args.dataset:<6}] ep {epoch:>3}  loss {epoch_loss:.4f}  "
                  f"R@5 {val_metrics['recall@5']:.4f}  R@10 {val_metrics['recall@10']:.4f}  "
                  f"R@20 {val_metrics['recall@20']:.4f}  N@10 {val_metrics['ndcg@10']:.4f}  "
                  f"({dt:.1f}s)")
            improved = val_metrics["recall@20"] > best["recall@20"]
            if improved:
                best = val_metrics
                best_epoch = epoch
                torch.save({
                    "user_embeds": model.user_embeds.detach().cpu(),
                    "item_embeds": model.item_embeds.detach().cpu(),
                    "n_users": n_users,
                    "n_items": n_items,
                    "embedding_size": args.embedding_size,
                    "layer_num": args.layer_num,
                    "seed": args.seed,
                    "epoch": epoch,
                    "val": best,
                }, ckpt_path)
            elif epoch - best_epoch >= args.patience:
                print(f"early stop at ep {epoch} (best ep {best_epoch})")
                break

    # Final test eval (load best)
    state = torch.load(ckpt_path, map_location=device)
    model.user_embeds.data.copy_(state["user_embeds"].to(device))
    model.item_embeds.data.copy_(state["item_embeds"].to(device))
    test_metrics = evaluate(model, adj, trn_csr, tst, [5, 10, 20], device)
    print(f"\n[{args.dataset}] best val ep {best_epoch}  val={best}")
    print(f"[{args.dataset}] test metrics: {test_metrics}")

    # Save metrics summary alongside the ckpt
    import json
    (args.ckpt_dir / f"lightgcn-{args.dataset}-{args.seed}.json").write_text(
        json.dumps({"val_best": best, "test": test_metrics, "best_epoch": best_epoch},
                   indent=2)
    )


if __name__ == "__main__":
    main()
