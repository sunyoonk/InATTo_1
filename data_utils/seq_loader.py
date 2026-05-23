"""Sequence-based training/eval dataset for InATTo E2E.

Each user has a chronological item sequence
    [i_1, i_2, ..., i_{n-2}, i_val, i_test]

We construct samples as follows:

Train  : every (history, target) pair within the train portion.
         For user with train length T_u >= 2:
             for j in range(1, T_u):
                 history = items[:j]  (truncated to history_max_len)
                 target  = items[j]

Val    : history = train items (all of them, truncated)
         target  = i_val

Test   : history = train items + i_val (all, truncated)
         target  = i_test
"""

from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset


def _load_pickle(p: str | Path):
    with open(p, "rb") as f:
        return pickle.load(f)


class InATToSeqDataset(Dataset):
    """One sample = (user_id, target_item_id, history_ids, history_valid)."""

    def __init__(
        self,
        data_root: str | Path,
        dataset_name: str,
        split: str,                  # 'train' / 'val' / 'test'
        history_max_len: int = 10,
    ):
        self.root = Path(data_root) / dataset_name
        self.split = split
        self.history_max_len = int(history_max_len)
        train_history = _load_pickle(self.root / "user_train_history.pkl")

        # Materialize sample list as (user_id, target_id, history_list)
        self.samples: list[tuple[int, int, list[int]]] = []
        if split == "train":
            for uid, hist in enumerate(train_history):
                if hist is None or len(hist) < 2:
                    continue
                for j in range(1, len(hist)):
                    history = hist[:j]
                    target = hist[j]
                    self.samples.append((uid, target, history))
        elif split == "val":
            val_coo = _load_pickle(self.root / "val_mat.pkl").tocoo()
            user_to_target: dict[int, int] = {int(u): int(i)
                                              for u, i in zip(val_coo.row, val_coo.col)}
            for uid, hist in enumerate(train_history):
                if uid not in user_to_target or not hist:
                    continue
                self.samples.append((uid, user_to_target[uid], hist))
        elif split == "test":
            val_coo = _load_pickle(self.root / "val_mat.pkl").tocoo()
            tst_coo = _load_pickle(self.root / "tst_mat.pkl").tocoo()
            user_to_val: dict[int, int] = {int(u): int(i)
                                           for u, i in zip(val_coo.row, val_coo.col)}
            user_to_tst: dict[int, int] = {int(u): int(i)
                                           for u, i in zip(tst_coo.row, tst_coo.col)}
            for uid, hist in enumerate(train_history):
                if uid not in user_to_tst or not hist:
                    continue
                history = (hist + [user_to_val[uid]]) if uid in user_to_val else hist
                self.samples.append((uid, user_to_tst[uid], history))
        else:
            raise ValueError(f"unknown split {split!r}")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        uid, target, history = self.samples[idx]
        hist = history[-self.history_max_len :]
        return uid, target, hist


def collate(batch, history_max_len: int):
    """Pad histories to history_max_len with a sentinel (0) + valid mask."""
    B = len(batch)
    uids = torch.tensor([b[0] for b in batch], dtype=torch.long)
    tgts = torch.tensor([b[1] for b in batch], dtype=torch.long)
    hist = torch.zeros(B, history_max_len, dtype=torch.long)
    valid = torch.zeros(B, history_max_len, dtype=torch.long)
    for b, (_, _, h) in enumerate(batch):
        L = min(len(h), history_max_len)
        # Right-align so the most recent items end at index history_max_len-1
        hist[b, history_max_len - L:] = torch.tensor(h[-L:], dtype=torch.long)
        valid[b, history_max_len - L:] = 1
    return uids, tgts, hist, valid


def make_collate(history_max_len: int):
    def _c(batch):
        return collate(batch, history_max_len)
    return _c
