"""BPR triplet sampler (user, pos_item, neg_item) for LightGCN training."""

from __future__ import annotations
import numpy as np
import torch
from torch.utils.data import Dataset


class BPRTripletDataset(Dataset):
    """Sample one negative per (user, pos_item) interaction each epoch.

    Reuses the train coo_matrix indices. Negatives are sampled lazily by
    calling resample() once per epoch (or per batch via __getitem__).
    """

    def __init__(self, trn_mat, n_items: int, seed: int = 0):
        coo = trn_mat.tocoo()
        self.users = coo.row.astype(np.int64)
        self.pos_items = coo.col.astype(np.int64)
        self.n_items = int(n_items)
        # Per-user set of positives for negative rejection sampling.
        self.user_pos: dict[int, set[int]] = {}
        for u, i in zip(self.users, self.pos_items):
            self.user_pos.setdefault(int(u), set()).add(int(i))
        self.rng = np.random.default_rng(seed)
        self.neg_items = np.zeros_like(self.users)
        self.resample()

    def resample(self) -> None:
        # Vectorized rejection sampling: draw and resample collisions.
        cand = self.rng.integers(0, self.n_items, size=len(self.users))
        for idx in range(len(cand)):
            u = int(self.users[idx])
            j = int(cand[idx])
            pos = self.user_pos[u]
            while j in pos:
                j = int(self.rng.integers(0, self.n_items))
            cand[idx] = j
        self.neg_items = cand

    def __len__(self) -> int:
        return len(self.users)

    def __getitem__(self, idx):
        return (
            int(self.users[idx]),
            int(self.pos_items[idx]),
            int(self.neg_items[idx]),
        )


def collate_bpr(batch):
    u = torch.tensor([b[0] for b in batch], dtype=torch.long)
    p = torch.tensor([b[1] for b in batch], dtype=torch.long)
    n = torch.tensor([b[2] for b in batch], dtype=torch.long)
    return u, p, n
