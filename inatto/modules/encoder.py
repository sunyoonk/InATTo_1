"""Dual-branch encoder (Phase 1 core novelty, spec §3.2).

Pipeline:
    z (B, d_cf)
       ↓  E1: MultiProjector  (n orthogonal W_k)
    e (B, n, d_aspect)
       ├──→ E2: DisentangledTransformer  →  z_aspect (B, n, d_aspect)
       └──→ Ep: ImportanceSubnetwork(e, rho, delta_k)  →  phi (B, n)

E2 and Ep both consume **pre-transformer e**, not z_aspect. That's the
dual-branch instantiation of VRVQ's design (importance and latent
predicted from a shared lower encoder).
"""

from __future__ import annotations
import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# E1 — Multi-projector
# ---------------------------------------------------------------------------

class MultiProjector(nn.Module):
    """n orthogonally-initialized linear maps R^{d_cf} -> R^{d_aspect},
    plus an inverse linear map for FACE-style CF-space reconstruction.

    The forward maps are stored as a single (n, d_aspect, d_cf) parameter
    so forward is one einsum. ``get_W_k(k)`` returns a view of W[k] for
    use elsewhere (Reliability per-aspect δ_{i,k} projection).

    The ``reverse_linear`` head (introduced for the CF-space decoder, like
    FACE's ``LinearLayer.reverse_linear``) maps the flattened multi-aspect
    representation back to CF space, ``R^{n*d_aspect} -> R^{d_cf}``.
    """

    def __init__(self, d_cf: int, n_aspects: int, d_aspect: int):
        super().__init__()
        self.d_cf = int(d_cf)
        self.n_aspects = int(n_aspects)
        self.d_aspect = int(d_aspect)
        # Param stored as (n, d_aspect, d_cf) for einsum convenience.
        W = torch.empty(self.n_aspects, self.d_aspect, self.d_cf)
        for k in range(self.n_aspects):
            nn.init.orthogonal_(W[k])
        b = torch.zeros(self.n_aspects, self.d_aspect)
        self.W = nn.Parameter(W)
        self.b = nn.Parameter(b)
        # FACE-style reverse linear: (n * d_aspect) -> d_cf
        self.reverse_linear = nn.Linear(self.n_aspects * self.d_aspect,
                                         self.d_cf)

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        """z: (B, d_cf)  ->  e: (B, n, d_aspect)."""
        # einsum 'naj,bj->bna'  where n=aspects, a=d_aspect, j=d_cf
        return torch.einsum("naj,bj->bna", self.W, z) + self.b

    def get_W_k(self, k: int) -> torch.Tensor:
        return self.W[k]

    def reverse(self, e: torch.Tensor) -> torch.Tensor:
        """e: (B, n, d_aspect)  ->  z_decoded: (B, d_cf).

        Mirrors FACE's ``LinearLayer.reverse``: flatten the per-aspect
        representation and apply a single linear head trained jointly.
        """
        return self.reverse_linear(e.reshape(e.shape[0], -1))


# ---------------------------------------------------------------------------
# E2 — Disentangled Transformer (latent encoder branch)
# ---------------------------------------------------------------------------

class DisentangledTransformer(nn.Module):
    """One-layer one-head Transformer over the n aspect tokens.

    Aspect-wise self-attention (FACE-style). batch_first=True so the
    input shape is (B, n, d_aspect).
    """

    def __init__(self, d_aspect: int, num_layers: int = 1, n_heads: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        layer = nn.TransformerEncoderLayer(
            d_model=d_aspect,
            nhead=n_heads,
            dim_feedforward=d_aspect * 4,
            dropout=dropout,
            batch_first=True,
        )
        self.encoder = nn.TransformerEncoder(layer, num_layers=num_layers)

    def forward(self, e: torch.Tensor) -> torch.Tensor:
        """e: (B, n, d_aspect)  ->  z_aspect: same shape."""
        return self.encoder(e)


# ---------------------------------------------------------------------------
# Ep — Importance Subnetwork (NEW)
# ---------------------------------------------------------------------------

class ImportanceSubnetwork(nn.Module):
    """Per-aspect importance phi_k = sigmoid(MLP_phi([e_k; rho; tilde_delta_k])).

    Shared MLP parameters across aspects (only the per-aspect inputs differ).
    Input is the **pre-transformer** e (consistent with spec §3.2.3).

    Delta is standardized within each forward call to stabilize the input
    distribution (batch-level z-score with min-eps guard).

    Supports ablation ``closed_form=True``: bypass the MLP and use the
    closed-form formula ``phi_k = rho * sigmoid(-delta_k)`` instead
    (per-aspect, no learned parameters).
    """

    def __init__(self, d_aspect: int, hidden: int = 64,
                 closed_form: bool = False,
                 ablate_no_rho: bool = False,
                 ablate_no_delta: bool = False,
                 phi_init_bias: float = 0.0):
        super().__init__()
        in_dim = d_aspect + 2   # [e_k, rho, tilde_delta_k]
        self.closed_form = bool(closed_form)
        # Ablation toggles for paper §4: zero out ρ or δ in the Ep input
        # to isolate each signal's contribution to φ. Defaults are False.
        self.ablate_no_rho = bool(ablate_no_rho)
        self.ablate_no_delta = bool(ablate_no_delta)
        # Always build the MLP so checkpoints saved with one config remain
        # compatible if we toggle the flag; the parameters just stop being
        # used when closed_form=True.
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, hidden),
            nn.ReLU(inplace=True),
            nn.Linear(hidden, 1),
        )
        # Last-linear init. With phi_init_bias=0 we get sigmoid(0)=0.5 (spec
        # default). With phi_init_bias=5 we get sigmoid(5)≈0.993 so every
        # item starts at *near-full* depth and the importance MLP learns
        # to *halt* sparse items instead of *deepening* rich ones. This
        # flips the regime so gain-saturation can't keep phi stuck low
        # (IAHQ Phase 2D).
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.constant_(self.mlp[-1].bias, float(phi_init_bias))

    def forward(self, e: torch.Tensor, rho: torch.Tensor,
                delta_k: torch.Tensor) -> torch.Tensor:
        """
        e        : (B, n, d_aspect)  pre-transformer aspect features
        rho      : (B,)              text density scalar per item
        delta_k  : (B, n)             per-aspect reliability magnitudes

        Returns phi : (B, n) in (0, 1)
        """
        # Standardize delta within batch (across both B and n dims).
        flat = delta_k.reshape(-1)
        mu = flat.mean()
        std = flat.std().clamp_min(1e-6)
        tilde = (delta_k - mu) / std                       # (B, n)

        # ---- Ablation: replace ρ/δ_k with a uninformative constant ----
        # ρ default = its dataset mean (~0.4) so the magnitude is realistic;
        # δ_k default = 0 (the batch-standardized mean).
        if self.ablate_no_rho:
            rho = torch.full_like(rho, 0.4)
        if self.ablate_no_delta:
            tilde = torch.zeros_like(tilde)

        if self.closed_form:
            # Closed-form ablation: phi = rho * sigmoid(-delta_k_standardized)
            rho_b = rho.unsqueeze(1)                       # (B, 1)
            return rho_b * torch.sigmoid(-tilde)

        B, n, d = e.shape
        rho_e = rho.unsqueeze(1).expand(B, n).unsqueeze(-1)    # (B, n, 1)
        tilde_e = tilde.unsqueeze(-1)                          # (B, n, 1)
        u = torch.cat([e, rho_e, tilde_e], dim=-1)             # (B, n, d+2)

        phi_logit = self.mlp(u).squeeze(-1)                    # (B, n)
        return torch.sigmoid(phi_logit)


# ---------------------------------------------------------------------------
# Dual-branch encoder (top-level)
# ---------------------------------------------------------------------------

class CFSpaceDecoder(nn.Module):
    """FACE-style decoder: aspect-space transformer + linear back to CF space.

    Matches ``FACE/encoder/FACE.py::FACE.transformer_decoder`` followed by
    ``LinearLayer.reverse``: a single-layer Transformer over the n aspects,
    then a linear head flattening the aspect dimension to ``d_cf``. The
    ``MultiProjector`` instance (i.e. E1) is passed in so the reverse
    linear head lives alongside the forward projection, exactly like FACE
    keeps both inside one ``LinearLayer``.
    """

    def __init__(
        self,
        e1: "MultiProjector",
        d_aspect: int,
        transformer_layers: int = 1,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.0,
    ):
        super().__init__()
        # Separate Transformer instance from E2 (mirrors FACE: encoder and
        # decoder share the same architecture but distinct parameters).
        self.transformer = DisentangledTransformer(
            d_aspect, transformer_layers, transformer_heads, transformer_dropout,
        )
        self.e1 = e1                    # for ``.reverse(...)`` callthrough

    def forward(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, n, d_aspect)  ->  decoded: (B, d_cf)."""
        h = self.transformer(z_q)
        return self.e1.reverse(h)


class DualBranchEncoder(nn.Module):
    def __init__(
        self,
        d_cf: int,
        n_aspects: int = 16,
        d_aspect: int = 256,
        phi_hidden: int = 64,
        transformer_layers: int = 1,
        transformer_heads: int = 1,
        transformer_dropout: float = 0.0,
        ep_closed_form: bool = False,
        ablate_no_rho: bool = False,
        ablate_no_delta: bool = False,
        phi_init_bias: float = 0.0,
    ):
        super().__init__()
        self.e1 = MultiProjector(d_cf, n_aspects, d_aspect)
        self.e2 = DisentangledTransformer(d_aspect, transformer_layers,
                                          transformer_heads, transformer_dropout)
        self.ep = ImportanceSubnetwork(d_aspect, phi_hidden,
                                        closed_form=ep_closed_form,
                                        ablate_no_rho=ablate_no_rho,
                                        ablate_no_delta=ablate_no_delta,
                                        phi_init_bias=phi_init_bias)
        # FACE-style decoder back to CF space. Owns its own Transformer
        # instance but shares the E1 linear head via ``e1.reverse_linear``.
        self.decoder = CFSpaceDecoder(
            self.e1, d_aspect,
            transformer_layers, transformer_heads, transformer_dropout,
        )

    def forward(
        self,
        z: torch.Tensor,        # (B, d_cf)
        rho: torch.Tensor,      # (B,)
        delta_k: torch.Tensor,  # (B, n)
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Returns (e, z_aspect, phi).
        Shapes: e, z_aspect : (B, n, d_aspect);  phi : (B, n).
        """
        e = self.e1(z)
        z_aspect = self.e2(e)
        phi = self.ep(e, rho, delta_k)
        return e, z_aspect, phi

    def decode(self, z_q: torch.Tensor) -> torch.Tensor:
        """z_q: (B, n, d_aspect)  ->  decoded CF: (B, d_cf)."""
        return self.decoder(z_q)
