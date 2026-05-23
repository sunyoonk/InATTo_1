"""Reliability — global δ and per-aspect δ_k (spec §3.1.1).

    h_txt_CF = W_proj * h_hat_txt                     in R^{d_cf}
    r_ortho  = z - <z, h_txt_CF>/||h_txt_CF||^2 h_txt_CF   (Gram-Schmidt)
    delta    = ||r_ortho||_2

Per-aspect projection (NEW in InATTo, spec §3.1.1):
    r_ortho_k = W_k * r_ortho                         in R^{d_aspect}
    delta_k   = ||r_ortho_k||_2

The W_k matrices come from the MultiProjector E1 (Phase 1). This
module accepts them as a parameter at call time so the two share
weights without duplicating storage.

Only trainable parameter here: W_proj (d_cf × d_txt).
"""

from __future__ import annotations
import torch
import torch.nn as nn


class Reliability(nn.Module):
    def __init__(self, d_txt: int, d_cf: int, eps: float = 1e-8):
        super().__init__()
        self.d_txt = int(d_txt)
        self.d_cf = int(d_cf)
        self.eps = float(eps)
        self.W_proj = nn.Linear(d_txt, d_cf, bias=False)

    def forward(
        self,
        h_hat_txt: torch.Tensor,   # (B, d_txt)
        z: torch.Tensor,           # (B, d_cf)
        W_k: torch.Tensor,         # (n, d_aspect, d_cf)  — from MultiProjector
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_txt_CF, r_ortho, delta, delta_k).

        Shapes:
          h_txt_CF : (B, d_cf)
          r_ortho  : (B, d_cf)
          delta    : (B,)
          delta_k  : (B, n_aspects)
        """
        h_cf = self.W_proj(h_hat_txt)                              # (B, d_cf)
        denom = (h_cf * h_cf).sum(dim=1, keepdim=True).clamp_min(self.eps)
        coef = (z * h_cf).sum(dim=1, keepdim=True) / denom         # (B, 1)
        r_ortho = z - coef * h_cf                                  # (B, d_cf)
        delta = r_ortho.norm(p=2, dim=1)                           # (B,)
        # Per-aspect: project r_ortho through each W_k -> norm
        r_ortho_k = torch.einsum("naj,bj->bna", W_k, r_ortho)      # (B, n, d_aspect)
        delta_k = r_ortho_k.norm(p=2, dim=-1)                       # (B, n)
        return h_cf, r_ortho, delta, delta_k
