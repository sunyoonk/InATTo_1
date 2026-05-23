"""Per-sparsity-group evaluation for InATTo + RPG Stage 2.

The "sparsity" of an item is the *raw text* word count of its
title + categories (the original RLMRec metadata, before any of our
internal signals ρ, δ, φ). This keeps the grouping independent of the
model under test — avoiding the circular-argument trap of grouping by
the very signal that our model conditions on.

For each test user (one target item per user, leave-one-out):
  1. classify the user by the sparsity group of their target item
     (Sparse = bottom 25%, Medium = middle 50%, Rich = top 25%);
  2. record per-user R@5, R@10, N@5, N@10 from the *full-ranking*
     InATTo eval (METHOD §3.6);
  3. aggregate group means and run a one-sided Welch's t-test of the
     "Sparse better than Rich" hypothesis on per-user R@5.

If TIGER/LETTER per-user predictions are available we add their rows to
the table; otherwise we log a note that only published paper numbers
exist as a baseline (and those are overall, not per-group).

The model checkpoint and identifier cache are *not* modified — this is
pure inference on a saved best.pth.

Usage:
    pixi run -- python scripts/10_sparsity_group_eval.py \\
        --dataset toys --cuda 0 \\
        --cache_suffix bpr.ssw01 \\
        --ckpt checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.wd05.best.pth

Smoke-test (first 100 test users):
    ... --max_users 100
"""

from __future__ import annotations
import argparse
import importlib.util
import json
import math
import pickle
import re
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.seq_loader import InATToSeqDataset

_RPG_PATH = Path(__file__).resolve().parent / "08c_train_stage2_rpg.py"
_spec = importlib.util.spec_from_file_location("rpg", str(_RPG_PATH))
_rpg  = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_rpg)


# ----------------------------------------------------------------------------
# Sparsity score = raw text word count (independent of our internal signals)
# ----------------------------------------------------------------------------

_TITLE_RX     = re.compile(r"title\s*:\s*([^;]+)", re.IGNORECASE)
_CATEGORY_RX  = re.compile(r"categor(?:y|ies)\s*:\s*([^;]+)", re.IGNORECASE)


def raw_word_count(text: str) -> int:
    """Word count from title + categories (only). Brand / description /
    price / salesrank are excluded so that "sparse" tracks the strictly
    *user-visible* surface signal."""
    if not text:
        return 0
    title    = _TITLE_RX.search(text)
    cats     = _CATEGORY_RX.search(text)
    chunks   = []
    if title: chunks.append(title.group(1).strip())
    if cats:  chunks.append(cats.group(1).strip())
    return sum(len(c.split()) for c in chunks)


def quantile_groups_from_text(itm_text: dict[int, str],
                                n_items: int,
                                n_groups: int = 3) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Returns (group_id_per_item, boundaries, word_count_per_item)."""
    wc = np.zeros(n_items, dtype=np.int32)
    for iid, txt in itm_text.items():
        if 0 <= iid < n_items:
            wc[iid] = raw_word_count(str(txt))
    qs = np.linspace(0, 1, n_groups + 1)
    boundaries = np.quantile(wc, qs)
    # digitize on internal cutpoints
    gid = np.clip(np.digitize(wc, boundaries[1:-1]), 0, n_groups - 1)
    return gid, boundaries, wc


# ----------------------------------------------------------------------------
# Per-user metric collector
# ----------------------------------------------------------------------------

@torch.no_grad()
def per_user_metrics(model, loader: DataLoader, device: torch.device,
                       Ks: tuple = (5, 10, 20), max_users: int | None = None
                       ) -> dict:
    """Run full-ranking InATTo eval, return a dict of arrays indexed by
    test-user position (in loader order):
      {'target_iid': (N,), 'R@5': (N,), 'R@10': (N,), 'N@5': (N,), ...}
    """
    model.eval()
    Kmax = max(Ks)
    out_target = []
    out_hits   = {K: [] for K in Ks}
    out_ndcg   = {K: [] for K in Ks}

    n_seen = 0
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        scores = model.rank_all_items(batch)                  # (B, n_items)
        # Mask history items (already-seen) and pad item
        for b in range(batch["input_ids"].size(0)):
            seen = batch["input_ids"][b][batch["attention_mask"][b].bool()]
            scores[b, seen] = -float("inf")
            scores[b, 0]    = -float("inf")
        topk = scores.topk(Kmax, dim=-1).indices.cpu().numpy()
        labels = batch["labels"].cpu().numpy()
        for b, label in enumerate(labels):
            out_target.append(int(label))
            for K in Ks:
                top_K = topk[b, :K]
                hit = int(label in top_K)
                out_hits[K].append(hit)
                if hit:
                    pos = int(np.where(top_K == label)[0][0])
                    out_ndcg[K].append(1.0 / math.log2(pos + 2))
                else:
                    out_ndcg[K].append(0.0)
            n_seen += 1
            if max_users is not None and n_seen >= max_users:
                break
        if max_users is not None and n_seen >= max_users:
            break
    return {
        "target_iid": np.array(out_target, dtype=np.int64),
        **{f"R@{K}": np.array(out_hits[K], dtype=np.float32) for K in Ks},
        **{f"N@{K}": np.array(out_ndcg[K], dtype=np.float32) for K in Ks},
    }


# ----------------------------------------------------------------------------
# Aggregation + t-test
# ----------------------------------------------------------------------------

def aggregate_per_group(metrics: dict, item_group: np.ndarray, n_groups: int,
                          Ks: tuple = (5, 10, 20)) -> dict:
    """Group means + per-group user counts."""
    targets = metrics["target_iid"]
    groups  = item_group[targets]
    rows = []
    for g in range(n_groups):
        sel = groups == g
        n   = int(sel.sum())
        row = {"group": g, "n_users": n}
        for K in Ks:
            row[f"R@{K}"] = float(metrics[f"R@{K}"][sel].mean()) if n else 0.0
            row[f"N@{K}"] = float(metrics[f"N@{K}"][sel].mean()) if n else 0.0
        rows.append(row)
    return {"groups": rows, "group_id_per_user": groups}


def welch_ttest_one_sided(a: np.ndarray, b: np.ndarray) -> tuple[float, float]:
    """Welch's t-test, one-sided H1: mean(a) > mean(b). Returns (t, p)."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    na, nb = len(a), len(b)
    if na < 2 or nb < 2:
        return float("nan"), float("nan")
    ma, mb = a.mean(), b.mean()
    va, vb = a.var(ddof=1), b.var(ddof=1)
    se = math.sqrt(va / na + vb / nb)
    if se == 0:
        return float("nan"), float("nan")
    t = (ma - mb) / se
    # Welch-Satterthwaite df
    df_num = (va / na + vb / nb) ** 2
    df_den = (va ** 2) / (na ** 2 * (na - 1)) + (vb ** 2) / (nb ** 2 * (nb - 1))
    df = df_num / df_den if df_den > 0 else float("inf")
    # One-sided p-value from t-distribution via SciPy if available, else normal approx
    try:
        from scipy.stats import t as t_dist
        p = 1.0 - t_dist.cdf(t, df)
    except Exception:
        # normal approximation
        p = 0.5 * math.erfc(t / math.sqrt(2))
    return float(t), float(p)


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--data_root", type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--cache_suffix", type=str, default="bpr.ssw01")
    p.add_argument("--ckpt", type=Path, required=True,
                    help="Path to a Stage-2 RPG best.pth.")
    p.add_argument("--depth_weighted_pool", action="store_true")
    p.add_argument("--n_groups", type=int, default=3,
                    help="3 = sparse / medium / rich; 4 = quartiles.")
    p.add_argument("--history_max_len", type=int, default=50)
    p.add_argument("--eval_batch", type=int, default=64)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--results_dir", type=Path,
                    default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--method_label", type=str, default="InATTo")
    p.add_argument("--output_filename", type=str, default=None,
                    help="Override the auto-generated json name (e.g. "
                         "'sparsity_toys_2023.json').")
    p.add_argument("--max_users", type=int, default=None,
                    help="For smoke tests — eval only first N test users.")
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    ddir = args.data_root / args.dataset

    # ---- Load identifier cache + raw item text (for sparsity grouping) ----
    cache_path = ddir / f"identifier_cache.{args.seed}.{args.cache_suffix}.pkl"
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    with open(ddir / "itm_text.pkl", "rb") as f:
        itm_text = pickle.load(f)

    n_items   = int(cache["cfg"]["V"]) and (max(cache["item"].keys()) + 1)
    group_id_per_item, bounds, wc = quantile_groups_from_text(
        itm_text, n_items, n_groups=args.n_groups
    )
    group_labels = (["sparse", "medium", "rich"] if args.n_groups == 3
                     else [f"q{g+1}" for g in range(args.n_groups)])

    print(f"[sparsity] criterion = raw text word count (title + categories)")
    print(f"  total items : {n_items}")
    print(f"  word count  : min={wc.min()} max={wc.max()} mean={wc.mean():.1f} median={np.median(wc):.0f}")
    print(f"  boundaries  : {[float(b) for b in bounds]}")
    for g in range(args.n_groups):
        n_g = int((group_id_per_item == g).sum())
        print(f"  {group_labels[g]:<8} ({n_g:>5} items)  word-count range "
              f"[{bounds[g]:.0f}, {bounds[g+1]:.0f}]")

    # ---- Build grid + model + load ckpt ----
    n_aspects = int(cache["cfg"]["n_aspects"])
    L_max     = int(cache["cfg"]["L_max"])
    V         = int(cache["cfg"]["V"])
    grid, mask = _rpg.parse_cache_to_grid(cache, n_aspects, L_max)

    model = _rpg.InATToRPG(
        item_id2tokens=grid, item_id2mask=mask,
        n_aspects=n_aspects, L_max=L_max, V=V,
        n_embd=256, n_layer=2, n_head=4, n_inner=1024,
        dropout=0.5, max_seq_len=args.history_max_len, temperature=0.05,
        depth_weighted_pool=args.depth_weighted_pool,
    ).to(device)
    ck = torch.load(args.ckpt, map_location=device, weights_only=False)
    model.load_state_dict(ck["model_state"])
    print(f"\n[ckpt] {args.ckpt.name}  ep{ck.get('epoch', '?')}  "
          f"val_best_R@5={ck.get('best_R5', '?')}")

    # ---- Test loader ----
    test_base = InATToSeqDataset(args.data_root, args.dataset, "test",
                                   history_max_len=args.history_max_len)
    test_ds   = _rpg.HistoryItemDataset(test_base, args.history_max_len)
    test_loader = DataLoader(
        test_ds, batch_size=args.eval_batch, shuffle=False,
        num_workers=args.num_workers, collate_fn=_rpg.make_collate(args.history_max_len),
        pin_memory=True,
    )
    print(f"[data] test_users={len(test_ds)}"
           + (f"  (limited to first {args.max_users} for smoke test)" if args.max_users else ""))

    # ---- Per-user metrics (full ranking) ----
    metrics = per_user_metrics(model, test_loader, device,
                                 Ks=(5, 10, 20), max_users=args.max_users)
    n_users = len(metrics["target_iid"])
    print(f"\n[eval] n_users measured = {n_users}")

    # ---- Aggregate per group ----
    agg = aggregate_per_group(metrics, group_id_per_item, args.n_groups,
                                Ks=(5, 10, 20))
    rows = agg["groups"]

    print()
    print("=" * 96)
    print(f"=== InATTo per-sparsity-group ({args.dataset}) ===")
    print(f"{'Group':<8}{'n_users':>9}{'R@5':>9}{'R@10':>9}{'R@20':>9}{'N@5':>9}{'N@10':>9}{'N@20':>9}")
    print("-" * 96)
    for r in rows:
        print(f"{group_labels[r['group']]:<8}{r['n_users']:>9}"
              f"{r['R@5']:>9.4f}{r['R@10']:>9.4f}{r['R@20']:>9.4f}"
              f"{r['N@5']:>9.4f}{r['N@10']:>9.4f}{r['N@20']:>9.4f}")
    print("=" * 96)

    # Trend: Sparse → Medium → Rich  (R@5 slope)
    rs = [rows[g]["R@5"] for g in range(args.n_groups)]
    print(f"\n[trend] R@5 over (sparse → rich): {[f'{x:.4f}' for x in rs]}")
    if args.n_groups == 3:
        print(f"        slope(sparse-rich) = {rs[0] - rs[-1]:+.4f}  "
               f"(positive ⇒ InATTo helps sparse-text items more)")

    # ---- One-sided t-test: sparse R@5 > rich R@5 ----
    sparse_sel = agg["group_id_per_user"] == 0
    rich_sel   = agg["group_id_per_user"] == (args.n_groups - 1)
    t, pval = welch_ttest_one_sided(metrics["R@5"][sparse_sel],
                                       metrics["R@5"][rich_sel])
    print(f"\n[t-test] Welch one-sided, H1: R@5(sparse) > R@5(rich)")
    print(f"         t = {t:.4f}   p = {pval:.4g}")

    # ---- Warn if too few sparse users ----
    n_sparse = int(sparse_sel.sum())
    if n_sparse < 200:
        print(f"\n[WARN] sparse-group user count = {n_sparse} (<200): "
              f"per-group means may be noisy.")

    # ---- TIGER/LETTER per-user predictions? ----
    print(f"\n[baselines] looking for TIGER/LETTER per-user predictions...")
    tiger_letter_search_paths = [
        Path(__file__).resolve().parent.parent / "TIGER",
        Path(__file__).resolve().parent.parent / "LETTER",
        Path(__file__).resolve().parent.parent / ".." / "letter",
        Path(__file__).resolve().parent.parent / ".." / "TIGER",
    ]
    found = False
    for p_dir in tiger_letter_search_paths:
        if p_dir.exists():
            print(f"   found:   {p_dir}")
            found = True
    if not found:
        print("   none in repo. We will report only InATTo's per-group numbers; "
              "TIGER/LETTER baselines are paper-overall numbers, not per-group.")

    # ---- Save ----
    args.results_dir.mkdir(parents=True, exist_ok=True)
    fname = args.output_filename or (
        f"sparsity_{args.dataset}_{args.seed}"
        + ("_smoke" if args.max_users else "") + ".json"
    )
    out_path = args.results_dir / fname
    payload = {
        "dataset": args.dataset, "seed": args.seed,
        "method": args.method_label,
        "ckpt": str(args.ckpt),
        "cache_suffix": args.cache_suffix,
        "depth_weighted_pool": bool(args.depth_weighted_pool),
        "criterion": "raw_text_word_count(title+categories)",
        "n_groups": args.n_groups,
        "group_labels": group_labels,
        "boundaries": [float(b) for b in bounds],
        "items_per_group": [int((group_id_per_item == g).sum())
                              for g in range(args.n_groups)],
        "n_users_per_group": [r["n_users"] for r in rows],
        "n_users_total": n_users,
        "max_users_for_smoke": args.max_users,
        "per_group": rows,
        "t_test_sparse_vs_rich_R5": {"t": t, "p_one_sided": pval,
                                        "n_sparse": int(sparse_sel.sum()),
                                        "n_rich":   int(rich_sel.sum())},
        # Per-user raw arrays (small enough — float32). Saved for downstream
        # paired tests against TIGER/LETTER when those become available.
        "per_user": {
            "target_iid": metrics["target_iid"].tolist(),
            **{m: metrics[m].tolist()
                for m in ("R@5", "R@10", "R@20", "N@5", "N@10", "N@20")},
            "sparsity_group": agg["group_id_per_user"].tolist(),
        },
    }
    out_path.write_text(json.dumps(payload, indent=2))
    print(f"\n[saved] {out_path}")


if __name__ == "__main__":
    main()
