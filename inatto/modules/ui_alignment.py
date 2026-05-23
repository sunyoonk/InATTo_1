"""User-Item joint alignment — Align3GR-inspired (spec §3.2.3 / §4.4).

Forms InfoNCE pairs (u, i+) with N in-batch negatives:

    z_hat_u = sum_k sum_l m_u_kl * c_u_kl       (B, n, d_aspect) -> sum over k,l
    z_hat_i = sum_k sum_l m_i_kl * c_i_kl

    sim(u, i_j) = cos(z_hat_u_flat, z_hat_i_j_flat) / tau

    L_ui = -1/|B| * sum_u log( exp(sim(u, i+)) / sum_j exp(sim(u, i_j)) )

We flatten aspect dim before cosine, so the contrastive head is just
linear in (n * d_aspect)-space. This aligns user and item identifiers
in the shared codebook space.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class UIAlignment(nn.Module):
    def __init__(self, tau: float = 0.07):
        super().__init__()
        self.tau = float(tau)

    def forward(
        self,
        z_hat_u: torch.Tensor,      # (B, n, d_aspect)
        z_hat_i_pos: torch.Tensor,  # (B, n, d_aspect)
        z_hat_i_neg: torch.Tensor | None = None,  # (B, N, n, d_aspect) or None
    ) -> torch.Tensor:
        B = z_hat_u.shape[0]
        u_flat = z_hat_u.reshape(B, -1)                              # (B, n*d_a)
        i_flat = z_hat_i_pos.reshape(B, -1)                          # (B, n*d_a)
        u_n = F.normalize(u_flat, p=2, dim=-1)
        i_n = F.normalize(i_flat, p=2, dim=-1)

        if z_hat_i_neg is None:
            # In-batch negatives: every other item in the batch
            logits = u_n @ i_n.t() / self.tau                         # (B, B)
            log_p = logits.diag() - torch.logsumexp(logits, dim=1)
            return -log_p.mean()
        else:
            N = z_hat_i_neg.shape[1]
            neg_flat = z_hat_i_neg.reshape(B, N, -1)
            neg_n = F.normalize(neg_flat, p=2, dim=-1)
            # Stack pos + neg: (B, 1+N, n*d_a)
            cand = torch.cat([i_n.unsqueeze(1), neg_n], dim=1)
            # (B, 1+N)
            logits = (u_n.unsqueeze(1) * cand).sum(dim=-1) / self.tau
            log_p = logits[:, 0] - torch.logsumexp(logits, dim=1)
            return -log_p.mean()
