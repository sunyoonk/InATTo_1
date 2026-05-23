"""Eval pipeline — trie-constrained beam search → R/NDCG @ K (spec §5.6 + §11).

Pipeline:
    1. Extract per-item identifiers from the trained tokenizer (hard mask).
    2. Build trie over all training items (ignore items with empty history).
    3. For each test (user, target) pair: build T5 input embeddings via the
       IdentifierBuilder (USER + HIST), run trie-constrained beam search,
       get top-K items.
    4. Compute Recall@K, NDCG@K for K in {5, 10, 20}.
"""

from __future__ import annotations
from typing import Iterable, Sequence

import math
import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from .trie import ItemTrie
from .beam_search import trie_beam_search


@torch.no_grad()
def evaluate(
    model,                            # InATToE2E
    loader: DataLoader,
    trie: ItemTrie,
    ks: Sequence[int] = (5, 10, 20),
    beam_width: int = 50,
    device: torch.device | None = None,
) -> dict[str, float]:
    model.eval()
    device = device or next(model.parameters()).device
    eoi_id = model.bridge.special_ids["<EOI>"]

    recalls = {k: [] for k in ks}
    ndcgs = {k: [] for k in ks}
    n_skipped = 0
    for uids, tgts, hist, valid in tqdm(loader, desc="eval"):
        uids = uids.to(device); tgts = tgts.to(device)
        hist = hist.to(device); valid = valid.to(device)

        # Build T5 input (no target needed here; we generate it).
        # Tokenize user + history first.
        user_out = model.tokenizer(uids, mode="user", hard_mask=True)
        flat_hist = hist.reshape(-1)
        hist_out = model.tokenizer(flat_hist, mode="item", hard_mask=True)
        B, T = hist.shape
        n = model.cfg["n_aspects"]; L = model.cfg["L_max"]
        hist_codes = hist_out["codes"].reshape(B, T, n, L)
        hist_masks = hist_out["depth_mask"].reshape(B, T, n, L)
        inputs_embeds, attn = model.id_builder.build_inputs(
            user_codes=user_out["codes"],
            user_mask=user_out["depth_mask"],
            history_codes=hist_codes,
            history_masks=hist_masks,
            history_valid=valid,
        )

        topk_lists = trie_beam_search(
            model.bridge.t5, inputs_embeds, attn, trie,
            eoi_token_id=eoi_id,
            beam_width=beam_width,
            num_return_sequences=max(ks),
        )
        for b in range(B):
            target = int(tgts[b].item())
            preds = topk_lists[b]
            if not preds:
                n_skipped += 1
                continue
            for k in ks:
                hits = [1 if p == target else 0 for p in preds[:k]]
                rec_k = sum(hits)                                   # 1 if hit in top-k
                discounts = [1.0 / math.log2(i + 2) for i in range(k)]
                dcg = sum(h * d for h, d in zip(hits, discounts))
                ideal = 1.0  # single relevant item
                recalls[k].append(rec_k)
                ndcgs[k].append(dcg / ideal)

    metrics = {}
    for k in ks:
        metrics[f"recall@{k}"] = float(np.mean(recalls[k])) if recalls[k] else 0.0
        metrics[f"ndcg@{k}"]   = float(np.mean(ndcgs[k]))   if ndcgs[k]   else 0.0
    metrics["n_skipped"] = n_skipped
    return metrics
