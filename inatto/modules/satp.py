"""SATP — Sparse-Aware Text Propagation (spec §3.1.1).

Pipeline (per item or per user, batch B):
    z       : (B, d_cf)     CF embedding from frozen LightGCN
    h_txt   : (B, d_txt)    raw MiniLM-encoded text embedding
    rho     : (B,)          precomputed normalized token entropy
    nn_idx  : (B, K)        top-K CF neighbor indices in the same mode-space

    Attention over neighbors:
        a_{*j} = softmax(z_*^T z_j / sqrt(d_cf))
        h_bar  = sum_j a_{*j} * h_txt_{nn_j}
        h_hat  = rho * h_txt + (1 - rho) * h_bar

All static tensors (z, h_txt, rho, nn_idx) are registered as buffers
once at init; forward only does scatter/gather + softmax.

User-mode vs item-mode is just which set of (z, h_txt, rho, nn_idx) is
loaded — the math is identical. So a single SATP instance is created
per "mode" (user or item) and called with the per-mode batch of ids.
"""

from __future__ import annotations
import math
import torch
import torch.nn as nn


class SATP(nn.Module):
    """Cached SATP for a single mode (user or item).

    Parameters
    ----------
    z_frozen : Tensor [N, d_cf]   pretrained CF embedding (frozen)
    h_txt    : Tensor [N, d_txt]  cached MiniLM raw text embedding
    rho      : Tensor [N]         cached normalized token entropy
    K        : int                top-K CF neighbors per row
    """

    def __init__(
        self,
        z_frozen: torch.Tensor,
        h_txt: torch.Tensor,
        rho: torch.Tensor,
        K: int = 10,
    ):
        super().__init__()
        assert z_frozen.shape[0] == h_txt.shape[0] == rho.shape[0], \
            "row counts must match across z, h_txt, rho"
        self.N = int(z_frozen.shape[0])
        self.d_cf = int(z_frozen.shape[1])
        self.d_txt = int(h_txt.shape[1])
        self.K = int(K)

        # z is a regular attribute (not buffer) so it can be re-assigned
        # to the live LightGCN propagation output each batch with the
        # autograd chain preserved (BPR-Stage-1 mode). For the original
        # frozen-LightGCN path nothing changes — we just hold a detached
        # tensor here.
        self.z = z_frozen.detach().clone()
        self.register_buffer("h_txt", h_txt.detach().clone())
        self.register_buffer("rho",   rho.detach().clone())
        self.register_buffer("nn_idx", self._build_neighbors(self.z))

    @torch.no_grad()
    def _build_neighbors(self, z: torch.Tensor) -> torch.Tensor:
        sim = z @ z.t()                                    # (N, N)
        sim.fill_diagonal_(float("-inf"))
        _, nn_idx = torch.topk(sim, self.K, dim=1)
        return nn_idx

    def refresh_z(self, z_new: torch.Tensor, rebuild_neighbors: bool = False) -> None:
        """Re-bind self.z. By default we do NOT detach so callers can hand
        in a live tensor from LightGCN propagation; gradients then flow
        through SATP -> Reliability -> RQ all the way back to LightGCN.

        Neighbors are kept stale by default (top-K does not need to change
        each batch and rebuild is O(N^2)). Set ``rebuild_neighbors=True``
        once per epoch (or after a checkpoint swap) for fresh neighbors.
        """
        assert z_new.shape == self.z.shape, f"shape mismatch: {z_new.shape}"
        # Keep dtype/device aligned with the buffers above.
        if z_new.dtype != self.h_txt.dtype:
            z_new = z_new.to(self.h_txt.dtype)
        if z_new.device != self.h_txt.device:
            z_new = z_new.to(self.h_txt.device)
        self.z = z_new                       # ← retain grad if z_new requires grad
        if rebuild_neighbors:
            with torch.no_grad():
                self.nn_idx.copy_(self._build_neighbors(self.z.detach()))

    def forward(self, ids: torch.Tensor, ablate: bool = False
                 ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (h_hat, rho_b, z_b).

        ids   : (B,) long
        h_hat : (B, d_txt)
        rho_b : (B,)
        z_b   : (B, d_cf)   convenience (saves a re-lookup downstream)

        ablate : if True, bypass neighbor aggregation and return
                 ``h_hat = h_txt`` directly (ablation: "w/o SATP").
        """
        z_b = self.z[ids]                                    # (B, d_cf)
        h_b = self.h_txt[ids]                                # (B, d_txt)
        rho_b = self.rho[ids]                                # (B,)

        if ablate:
            return h_b, rho_b, z_b

        nbr = self.nn_idx[ids]                               # (B, K)
        z_nbr = self.z[nbr]                                  # (B, K, d_cf)
        h_nbr = self.h_txt[nbr]                              # (B, K, d_txt)

        # Eq 3: attention weights
        scores = torch.einsum("bd,bkd->bk", z_b, z_nbr) / math.sqrt(self.d_cf)
        attn = torch.softmax(scores, dim=1)                  # (B, K)

        # Eq 2: aggregate
        h_bar = torch.einsum("bk,bkd->bd", attn, h_nbr)      # (B, d_txt)

        # Eq 4: rho-weighted interpolation
        rho_e = rho_b.unsqueeze(1)                           # (B, 1)
        h_hat = rho_e * h_b + (1.0 - rho_e) * h_bar          # (B, d_txt)
        return h_hat, rho_b, z_b
