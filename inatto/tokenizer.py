"""InATTo Tokenizer (Module 1, end-to-end).

Wires together the SATP / Reliability / DualBranchEncoder / depth_mask /
Codebook / ResidualQuantizer pipeline. Mode-aware: a single instance
serves both user and item modes, sharing all trainable parameters; only
the SATP buffers (z, h_txt, rho, nn_idx) differ per mode.

Forward(ids, mode) returns a dict consumed by the alignment modules and
the loss aggregator.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.satp import SATP
from .modules.reliability import Reliability
from .modules.encoder import DualBranchEncoder
from .modules.depth_mask import (depth_mask_ste, depth_mask_hard,
                                  depth_mask_ste_vrvq, depth_mask_gumbel_ste,
                                  depth_mask_gumbel_ste_v2)
from .modules.codebook import Codebook
from .modules.rq import ResidualQuantizer
from .modules.hrq import HierarchicalRQ


class InATToTokenizer(nn.Module):
    """Mode-aware shared tokenizer.

    Shared parameters (single set, trained by user-side AND item-side
    forwards via parameter sharing):
        - Reliability.W_proj
        - DualBranchEncoder (E1.W, E1.b, E2.transformer, Ep.MLP)
        - Codebook.W_c

    Per-mode buffers (frozen, set at construction):
        - satp_user  : SATP(z_user, h_txt_user, rho_user)
        - satp_item  : SATP(z_item, h_txt_item, rho_item)
    """

    def __init__(
        self,
        z_user: torch.Tensor,         # (n_users, d_cf)
        h_txt_user: torch.Tensor,     # (n_users, d_txt)
        rho_user: torch.Tensor,       # (n_users,)
        z_item: torch.Tensor,         # (n_items, d_cf)
        h_txt_item: torch.Tensor,     # (n_items, d_txt)
        rho_item: torch.Tensor,       # (n_items,)
        codebook: Codebook,           # shared codebook
        n_aspects: int = 16,
        d_aspect: int = 256,
        L_max: int = 4,
        K_neighbors: int = 10,
        phi_hidden: int = 64,
        alpha_ste: float = 5.0,
        commit_beta: float = 0.25,
        transformer_layers: int = 1,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.0,
        # ---- ablation toggles ----
        ablate_satp: bool = False,
        ablate_variable_depth: bool = False,
        ablate_ep_closed_form: bool = False,
        ablate_no_rho: bool = False,
        ablate_no_delta: bool = False,
        ablate_user_fixed_depth: bool = False,
        use_vrvq_mask: bool = False,
        vrvq_alpha: float = 4.0,
        use_gumbel_mask: bool = False,
        gumbel_tau: float = 1.0,
        gumbel_v2: bool = False,
        use_hrq: bool = False,
        hrq_tree_pkl: str | None = None,
        hrq_parent_constraint: bool = False,
        hrq_elcrec_proto: bool = False,
        hrq_ctfidf_repr: bool = True,
        phi_init_bias: float = 0.0,
    ):
        super().__init__()
        d_cf  = int(z_user.shape[1]);  assert z_item.shape[1] == d_cf
        d_txt = int(h_txt_user.shape[1]); assert h_txt_item.shape[1] == d_txt

        self.n_aspects = int(n_aspects)
        self.d_aspect = int(d_aspect)
        self.L_max = int(L_max)
        self.alpha_ste = float(alpha_ste)
        self.ablate_satp = bool(ablate_satp)
        self.ablate_variable_depth = bool(ablate_variable_depth)
        # If True, the user-side tokenizer always uses depth_mask = 1
        # (item side keeps its variable depth). Used to test whether the
        # user-side variable depth contributes — addresses reviewer
        # concern about narrow ρ_u distribution.
        self.ablate_user_fixed_depth = bool(ablate_user_fixed_depth)
        # VRVQ Eq.7 smooth surrogate (paper-strength gradient for the
        # importance MLP — solves phi saturation by giving non-vanishing
        # gradient over the entire [0, L_max] range).
        self.use_vrvq_mask = bool(use_vrvq_mask)
        self.vrvq_alpha    = float(vrvq_alpha)
        self.use_gumbel_mask = bool(use_gumbel_mask)
        self.gumbel_tau      = float(gumbel_tau)
        self.gumbel_v2       = bool(gumbel_v2)

        # Per-mode caches.
        self.satp_user = SATP(z_user, h_txt_user, rho_user, K=K_neighbors)
        self.satp_item = SATP(z_item, h_txt_item, rho_item, K=K_neighbors)

        # Shared trainables.
        self.reliability = Reliability(d_txt, d_cf)
        self.encoder = DualBranchEncoder(
            d_cf, n_aspects, d_aspect,
            phi_hidden=phi_hidden,
            transformer_layers=transformer_layers,
            transformer_heads=transformer_heads,
            transformer_dropout=transformer_dropout,
            ep_closed_form=bool(ablate_ep_closed_form),
            ablate_no_rho=bool(ablate_no_rho),
            ablate_no_delta=bool(ablate_no_delta),
            phi_init_bias=float(phi_init_bias),
        )
        self.codebook = codebook
        self.use_hrq = bool(use_hrq)
        if self.use_hrq:
            assert hrq_tree_pkl is not None
            self.rq = HierarchicalRQ(
                raw=codebook.codebook_raw,
                tree_pkl=hrq_tree_pkl,
                d_aspect=d_aspect,
                L_max=L_max,
                parent_constraint=bool(hrq_parent_constraint),
                commit_beta=commit_beta,
                use_elcrec_proto=bool(hrq_elcrec_proto),
                use_ctfidf_repr=bool(hrq_ctfidf_repr),
            )
        else:
            self.rq = ResidualQuantizer(L_max=L_max, commit_beta=commit_beta)

    # ------------------------------------------------------------------
    def _satp_for(self, mode: str) -> SATP:
        if mode == "user":
            return self.satp_user
        if mode == "item":
            return self.satp_item
        raise ValueError(f"mode must be 'user' or 'item', got {mode!r}")

    def refresh_z(self, mode: str, z_new: torch.Tensor,
                  rebuild_neighbors: bool = False) -> None:
        """Re-bind the cached CF embeddings for the given mode.

        NOT under @torch.no_grad so the live LightGCN propagation tensor
        keeps its autograd graph (BPR-Stage-1 mode).
        """
        self._satp_for(mode).refresh_z(z_new, rebuild_neighbors=rebuild_neighbors)

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, mode: str = "item",
                hard_mask: bool = False) -> dict:
        """Run the full tokenization pipeline for a batch.

        Parameters
        ----------
        ids : LongTensor (B,)   per-mode entity indices
        mode : 'user' | 'item'
        hard_mask : bool        if True, use the hard (non-STE) depth mask
                                — for inference / identifier extraction.

        Returns
        -------
        dict with keys:
            codes        (B, n, L_max)  long  — selected codeword indices
            depth_mask   (B, n, L_max)  {0,1} — active levels
            z_aspect     (B, n, d_aspect)     — E2 output
            z_hat        (B, n, d_aspect)     — masked sum (no STE)
            z_hat_st     (B, n, d_aspect)     — STE-attached version
            c_levels     (B, n, L_max, d_aspect)
            signals      dict with rho, delta, delta_k, phi, h_hat_txt,
                         h_cf, r_ortho, z (CF embed lookup)
            losses       dict with L_recon, L_Q, L_rate
        """
        # ---- Step 1: signal extraction ----
        h_hat, rho, z = self._satp_for(mode)(ids, ablate=self.ablate_satp)
        h_cf, r_ortho, delta, delta_k = self.reliability(
            h_hat, z, self.encoder.e1.W
        )                                                         # (B, d_cf), (B, d_cf), (B,), (B, n)

        # ---- Step 2: dual-branch encoder ----
        e, z_aspect, phi = self.encoder(z, rho, delta_k)         # (B, n, d_aspect) * 2, (B, n)

        # ---- Step 3: depth mask (STE or hard, or all-ones for ablation) ----
        if self.ablate_variable_depth:
            # All levels always active for every aspect (fixed length).
            m = phi.new_ones(phi.shape[0], self.n_aspects, self.L_max)
        elif self.ablate_user_fixed_depth and mode == "user":
            # User-side only: force fixed L_max depth. Tests whether the
            # narrow ρ_u distribution makes user variable depth meaningless.
            m = phi.new_ones(phi.shape[0], self.n_aspects, self.L_max)
        elif hard_mask:
            m = depth_mask_hard(phi, self.L_max)
        elif self.use_gumbel_mask:
            if self.gumbel_v2:
                m = depth_mask_gumbel_ste_v2(phi, self.L_max, tau=self.gumbel_tau)
            else:
                m = depth_mask_gumbel_ste(phi, self.L_max, tau=self.gumbel_tau)
        elif self.use_vrvq_mask:
            m = depth_mask_ste_vrvq(phi, self.L_max, alpha=self.vrvq_alpha)
        else:
            m = depth_mask_ste(phi, self.L_max, alpha_ste=self.alpha_ste)

        # ---- Step 4: RQ ----
        if self.use_hrq:
            rq_out = self.rq(z_aspect, m)
        else:
            C = self.codebook.codebook()
            rq_out = self.rq(z_aspect, C, m)

        # ---- Step 5: CF-space decoder (VRVQ + FACE style) ----
        # Decoder receives ``z_hat`` (no STE), so gradients from L_recon flow
        # through the depth_mask back into the importance subnet φ — this is
        # the only training signal for φ in VRVQ's design (cf.
        # VRVQ/models/quantize.py:421 where ``z_q = sum(z_q_is * mask_imp)``
        # uses the un-detached mask, while commit/codebook losses detach it).
        decoded = self.encoder.decode(rq_out["z_hat"])              # (B, d_cf)
        L_recon = F.mse_loss(decoded, z.detach())

        # ---- L_rate over the importance-masked batch portion only ----
        # VRVQ convention (quantize.py:428, train.py:313):
        #   imp_map_out = imp_map[:n_imps]
        #   rate_loss = imp_map_out.mean()
        # The full_codebook portion's φ is irrelevant to depth budget (those
        # samples force mask=1), so averaging over them dilutes / contaminates
        # the budget signal. We compute φ.mean() only over the first n_imps
        # samples to match VRVQ exactly.
        if self.training and self.rq.full_codebook_rate > 0:
            B = phi.shape[0]
            n_full = int(B * self.rq.full_codebook_rate)
            n_imps = B - n_full
            L_rate = phi[:n_imps].mean() if n_imps > 0 else phi.mean()
        else:
            L_rate = phi.mean()

        return {
            "codes":      rq_out["codes"],
            "depth_mask": m,
            "z_aspect":   z_aspect,
            "z_hat":      rq_out["z_hat"],
            "z_hat_st":   rq_out["z_hat_st"],
            "c_levels":   rq_out["c_levels"],
            "signals": {
                "rho": rho, "delta": delta, "delta_k": delta_k, "phi": phi,
                "h_hat_txt": h_hat, "h_cf": h_cf, "r_ortho": r_ortho,
                "z": z, "e": e,
            },
            "losses": {
                "L_recon": L_recon,
                "L_Q":     rq_out["L_Q"],
                "L_rate":  L_rate,
                # ELCRec separation regularizer (0 unless HRQ + proto enabled)
                "L_sep":   rq_out.get("L_sep", z.new_zeros(())),
            },
        }
