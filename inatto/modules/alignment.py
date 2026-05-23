"""Information-Aware Alignment — adaptive target + in-batch InfoNCE
(spec §3.2 / §4 of the design).

Adaptive target (uses rho):
    h_align = W_align * h_hat_txt
    h_adp   = rho * h_raw + (1 - rho) * h_align

In-batch InfoNCE:
    sim(i, j) = cos(h_d_i, h_adp_j) / tau
    L_align   = -1/|B| * sum_i log( exp(sim(i,i)) / sum_j exp(sim(i,j)) )

Operates symmetrically on (user, item) — call once per side.
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


class Alignment(nn.Module):
    def __init__(self, d_txt: int, d_llm: int, tau: float = 0.07):
        super().__init__()
        self.d_txt = int(d_txt)
        self.d_llm = int(d_llm)
        self.tau = float(tau)
        self.W_align = nn.Linear(d_txt, d_llm, bias=False)

    def adaptive_target(
        self,
        h_raw: torch.Tensor,        # (B, d_llm)
        h_hat_txt: torch.Tensor,    # (B, d_txt)
        rho: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        h_align = self.W_align(h_hat_txt)                            # (B, d_llm)
        rho_e = rho.unsqueeze(1)                                     # (B, 1)
        return rho_e * h_raw + (1.0 - rho_e) * h_align

    def forward(
        self,
        h_d: torch.Tensor,          # (B, d_llm)  descriptor embedding (normalized)
        h_raw: torch.Tensor,        # (B, d_llm)
        h_hat_txt: torch.Tensor,    # (B, d_txt)
        rho: torch.Tensor,          # (B,)
    ) -> torch.Tensor:
        h_adp = self.adaptive_target(h_raw, h_hat_txt, rho)
        h_d_n = F.normalize(h_d, p=2, dim=-1)
        h_adp_n = F.normalize(h_adp, p=2, dim=-1)
        logits = h_d_n @ h_adp_n.t() / self.tau                       # (B, B)
        log_p = logits.diag() - torch.logsumexp(logits, dim=1)
        return -log_p.mean()
