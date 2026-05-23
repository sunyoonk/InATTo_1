"""Build the normalized user-item bipartite adjacency for LightGCN.

Given a [n_users, n_items] interaction matrix R (scipy.sparse.coo):

    A = [[ 0    R  ],
         [ R.T  0  ]]                   shape (n+m) x (n+m)

    A_norm = D^{-1/2}  A  D^{-1/2}      symmetric normalization
                                         no self-loops

Returned as a torch.sparse_coo_tensor on the chosen device.
"""

from __future__ import annotations
import numpy as np
import scipy.sparse as sp
import torch


def make_bipartite_adj(trn_mat, n_users: int, n_items: int) -> sp.coo_matrix:
    """Construct the symmetric bipartite adjacency from trn_mat (scipy sparse)."""
    R = trn_mat.tocsr()
    R = (R != 0).astype(np.float32)
    top_zero = sp.csr_matrix((n_users, n_users), dtype=np.float32)
    bot_zero = sp.csr_matrix((n_items, n_items), dtype=np.float32)
    top = sp.hstack([top_zero, R])
    bot = sp.hstack([R.T, bot_zero])
    A = sp.vstack([top, bot]).tocoo()
    return A


def normalize_adj(A: sp.coo_matrix) -> sp.coo_matrix:
    """Symmetric normalization D^{-1/2} A D^{-1/2}."""
    A = A.tocsr()
    deg = np.asarray(A.sum(axis=1)).flatten()
    d_inv_sqrt = np.power(deg, -0.5, where=(deg > 0))
    d_inv_sqrt[~np.isfinite(d_inv_sqrt)] = 0.0
    D = sp.diags(d_inv_sqrt)
    A_norm = D @ A @ D
    return A_norm.tocoo()


def coo_to_torch_sparse(A: sp.coo_matrix, device: torch.device) -> torch.Tensor:
    idx = torch.from_numpy(np.vstack([A.row, A.col]).astype(np.int64))
    val = torch.from_numpy(A.data.astype(np.float32))
    shape = torch.Size(A.shape)
    return torch.sparse_coo_tensor(idx, val, shape, device=device).coalesce()


def build_torch_adj(trn_mat, n_users: int, n_items: int, device: torch.device) -> torch.Tensor:
    A = make_bipartite_adj(trn_mat, n_users, n_items)
    A_norm = normalize_adj(A)
    return coo_to_torch_sparse(A_norm, device)
