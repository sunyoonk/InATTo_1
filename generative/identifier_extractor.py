"""Phase 2 / Eval prep — extract per-item identifiers from a trained tokenizer.

Runs the trained InATToE2E in eval mode over all items, collects each
item's (codes, depth_mask) under hard masking, and converts to a flat
T5-token-id identifier sequence via the model's IdentifierBuilder.

Output:
    item_identifiers : dict[int, list[int]]    item_id -> T5 token id list
    item_lengths     : np.ndarray [n_items]     per-item identifier length

This is consumed by ``generative/trie.py`` to build the constrained trie.
"""

from __future__ import annotations
from typing import Iterable

import numpy as np
import torch
from tqdm import tqdm


@torch.no_grad()
def extract_item_identifiers(
    model,                  # InATToE2E
    n_items: int,
    device: torch.device,
    batch_size: int = 512,
) -> tuple[dict[int, list[int]], np.ndarray]:
    model.eval()
    out: dict[int, list[int]] = {}
    lengths = np.zeros(n_items, dtype=np.int32)
    ids_all = torch.arange(n_items, dtype=torch.long, device=device)
    for s in tqdm(range(0, n_items, batch_size), desc="extract identifiers"):
        batch_ids = ids_all[s : s + batch_size]
        tok_out = model.tokenizer(batch_ids, mode="item", hard_mask=True)
        codes = tok_out["codes"]                 # (B, n, L_max)
        masks = tok_out["depth_mask"]            # (B, n, L_max)
        for j in range(codes.shape[0]):
            seq = model.id_builder.item_token_ids(codes[j], masks[j])
            iid = int(batch_ids[j].item())
            out[iid] = seq
            lengths[iid] = len(seq)
    return out, lengths
