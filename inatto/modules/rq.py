"""Residual Vector Quantization with cross-aspect codeword exclusion (spec §3.3.3).

For each item (or user) with disentangled aspect features z_aspect (B, n, d_aspect):
    r_k^(0) = z_k
    used_codes_in_item = set()      # cross-aspect collection
    for l = 1..L_max:
        r_tilde = r_k^(l-1) / ||r_k^(l-1)||
        valid = V \ used_codes_in_item
        c_k^(l) = argmin_{v in valid} ||r_tilde - c_tilde_v||^2
        used_codes_in_item.add(c_k^(l))
        r_k^(l) = r_k^(l-1) - c_v[c_k^(l)]

Reconstruction (masked), with FACE-style STE so the encoder still
receives gradient through z_hat even though codeword selection is
discrete:
    z_hat_k     = sum_l m_k^(l) * c_v[c_k^(l)]        # value-correct
    z_hat_st_k  = z_k + (z_hat_k - z_k).detach()      # STE: value=z_hat, grad=z_k

Two-sided VQ loss (matches FACE's 0.75 / 0.25 weighting):
    codebook side:  || c - stop_grad(z) ||^2      # trains W_c
    commit side:    beta * || z - stop_grad(c) ||^2  # trains encoder
    L_Q = (codebook + commit) over active levels only

The cross-aspect exclusion is the key novelty of this RQ vs FACE:
within one item, each aspect picks a distinct codeword, encouraging
diverse semantic facets per item.
"""

from __future__ import annotations
import torch
import torch.nn as nn


class ResidualQuantizer(nn.Module):
    """RQ with depth mask + cross-aspect exclusion.

    Stateless module — codebook and mask are passed in each forward.

    Parameters
    ----------
    L_max : int
        Maximum RQ depth.
    commit_beta : float
        Weight on the commit-side VQ loss (default 0.25; FACE uses 0.25).
    """

    def __init__(self, L_max: int, commit_beta: float = 0.25,
                 full_codebook_rate: float = 0.25):
        super().__init__()
        self.L_max = int(L_max)
        self.commit_beta = float(commit_beta)
        # VRVQ-style codebook warmup: during training, force a fraction of
        # the batch to use depth_mask = 1 everywhere so codebook learning
        # gets a steady full-depth signal regardless of φ's current state.
        # See VRVQ/models/quantize.py:408,414 ``mask_imp[n_imps+n_dropout:] = 1.0``.
        # 0.25 matches VRVQ's vrvq_a2.yml default (the original audio paper).
        self.full_codebook_rate = float(full_codebook_rate)

    def forward(
        self,
        z_aspect: torch.Tensor,        # (B, n, d_aspect)  E2 output
        C: torch.Tensor,               # (V, d_aspect)     projected codebook
        depth_mask: torch.Tensor,      # (B, n, L_max)     STE mask
    ) -> dict:
        """Run RQ over aspects, with cross-aspect exclusion per item.

        Returns dict with:
            codes       (B, n, L_max)  long, codeword indices (active or pad)
            c_levels    (B, n, L_max, d_aspect)  selected codeword vectors
            z_hat       (B, n, d_aspect)  masked sum of selected codewords
            L_Q         scalar          masked VQ commitment loss
        """
        B, n, d_a = z_aspect.shape
        V = C.shape[0]
        device = z_aspect.device

        # ---- VRVQ codebook warmup ----
        # In training, override depth_mask = ones for the last n_full samples
        # so codebook losses (and reconstruction) see all L_max levels on
        # them. This decouples codebook learning from φ's current trajectory
        # and prevents the depth-budget vs codebook race we observed.
        if self.training and self.full_codebook_rate > 0:
            n_full = int(B * self.full_codebook_rate)
            if n_full > 0:
                depth_mask = depth_mask.clone()
                depth_mask[B - n_full:] = 1.0

        # Precompute normalized codebook for cosine-based argmin.
        C_norm = C / C.norm(p=2, dim=1, keepdim=True).clamp_min(1e-8)   # (V, d_a)

        codes = torch.full((B, n, self.L_max), -1, dtype=torch.long, device=device)
        c_levels = torch.zeros(B, n, self.L_max, d_a, device=device, dtype=z_aspect.dtype)
        L_Q_sum = z_aspect.new_zeros(())
        n_active = z_aspect.new_zeros(())

        # Used-mask per item, cross-aspect.
        used = torch.zeros(B, V, dtype=torch.bool, device=device)
        r = z_aspect.clone()                              # (B, n, d_a)

        for l in range(self.L_max):
            r_norm = r / r.norm(p=2, dim=-1, keepdim=True).clamp_min(1e-8)   # (B, n, d_a)
            # ||r̃ - c̃||^2 = 2 - 2 r̃ . c̃; argmin is argmax of inner prod.
            # Inner product: (B, n, V)
            sim = torch.einsum("bna,va->bnv", r_norm, C_norm)
            for k in range(n):
                masked = sim[:, k, :].clone()
                masked = masked.masked_fill(used, float("-inf"))    # exclude used
                idx = masked.argmax(dim=-1)                          # (B,)
                codes[:, k, l] = idx
                used.scatter_(1, idx.unsqueeze(1), True)
                c_l = C[idx]                                          # (B, d_a)
                c_levels[:, k, l] = c_l
                # Mask-aware residual update — only subtract if this level is active.
                m = depth_mask[:, k, l].unsqueeze(-1)                # (B, 1)
                r_k_prev = r[:, k, :]
                z_k = z_aspect[:, k, :]                              # (B, d_a)
                # Two-sided VQ loss (active levels only).
                #   codebook side: || c - sg(z) ||^2   -> trains W_c
                #   commit  side: || z - sg(c) ||^2   -> trains encoder
                codebook_term = (c_l - z_k.detach()).pow(2).sum(dim=-1)
                commit_term   = (z_k - c_l.detach()).pow(2).sum(dim=-1)
                # Detach the depth_mask in L_Q so codebook learning is
                # decoupled from φ learning — VRVQ original convention
                # (VRVQ/models/quantize.py:422-423: ``mask_imp.detach()``).
                # φ is trained via L_recon (which uses the un-detached mask)
                # and L_rate; commit/codebook commit term only trains W_c.
                L_Q_sum = L_Q_sum + (m.detach().squeeze(-1) *
                                     (codebook_term + self.commit_beta * commit_term)).sum()
                n_active = n_active + m.detach().sum()
                # Update residual: subtract only when active. Pad (m=0) keeps r unchanged.
                r = r.clone()
                r[:, k, :] = r_k_prev - m * c_l

        # Masked reconstruction.
        # depth_mask (B, n, L_max) -> (B, n, L_max, 1)
        z_hat = (depth_mask.unsqueeze(-1) * c_levels).sum(dim=2)     # (B, n, d_a)
        # STE: forward value = z_hat (codeword sum), backward = z_aspect's grad.
        z_hat_st = z_aspect + (z_hat - z_aspect).detach()

        # Per-spec: L_Q averaged over (B, n, active levels).
        L_Q = L_Q_sum / n_active.clamp_min(1.0)
        return {
            "codes": codes,
            "c_levels": c_levels,
            "z_hat": z_hat,           # exact codeword sum (no STE)
            "z_hat_st": z_hat_st,     # STE-attached: use this for downstream losses
            "L_Q": L_Q,
        }
