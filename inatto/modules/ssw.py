"""SSW — Spherical Sliced Wasserstein (S2WTM exact, S^{d-1} vs Uniform).

Direct port of the SSW² loss used in S2WTM (Adhya & Sanyal, 2024):
`octis/models/spherical_SWTM/models/wae_sp.py:sp_swd_unif_loss`
and `:w2_unif_circle`.

Algorithm:
    For X ⊂ S^{d-1} and M projections:
      1) Z ~ N(0, I) of shape (M, d, 2);   U_m, _ = QR(Z_m)  ⇒ U ∈ V_{d,2}
      2) Project each x ∈ X onto each 2-plane spanned by U_m :
             x_plane = U_m^T · x  ∈ R^2
      3) Normalise to S^1:   x_circle = x_plane / ||x_plane||
      4) Convert to circular coordinate in [0, 1]:
             u = (atan2(-y, -x) + π) / (2π)
      5) Closed-form W₂² vs Uniform([0, 1]) (which is S^1 after the wrap):
             W₂² = E[u²] - (E[u])² + Σ_i (n+1-2i)/n² · u_(i)  + 1/12
         (S2WTM `w2_unif_circle`, exact 1-D Wasserstein-2 to uniform on a
         circle — no Monte-Carlo target sampling needed).

SSW²(X, U_{S^{d-1}}) = mean over M projections of the above W₂².

Inputs are L2-normalised inside; callers can pass un-normalised vectors.
"""

from __future__ import annotations
import math
import torch
import torch.nn.functional as F


def _w2_uniform_circle(u_values: torch.Tensor) -> torch.Tensor:
    """Closed-form W_2^2 between u_values and Uniform([0,1]) on a circle.

    Mirrors S2WTM's `w2_unif_circle` exactly:
        cpt1   = E[u²]
        x_mean = E[u]
        cpt2   = Σ_i (n-1-2i)/n²   ·   sorted u_i              (i=0..n-1)
        return cpt1 - x_mean² + cpt2 + 1/12

    u_values : (..., n) — angles in [0, 1] (= angle / 2π).
    Returns  : (...,)   — W₂² per slice.
    """
    n = u_values.shape[-1]
    u_sorted, _ = torch.sort(u_values, dim=-1)
    cpt1 = (u_sorted ** 2).mean(dim=-1)
    x_mean = u_sorted.mean(dim=-1)
    ns_n2 = torch.arange(n - 1, -n, -2, dtype=u_values.dtype,
                          device=u_values.device) / (n ** 2)
    cpt2 = (ns_n2 * u_sorted).sum(dim=-1)
    return cpt1 - x_mean ** 2 + cpt2 + 1.0 / 12.0


def ssw(
    x: torch.Tensor,
    *,
    n_projections: int = 50,
    n_uniform: int | None = None,
    target: torch.Tensor | None = None,
) -> torch.Tensor:
    """SSW² between empirical(x) on S^{d-1} and Uniform(S^{d-1}).

    Parameters
    ----------
    x : (N, d) tensor — L2-normalised internally to land on S^{d-1}.
    n_projections : M random 2-frame projections via QR (Stiefel V_{d,2}).
    n_uniform, target : kept for API compatibility with the previous
                       directional implementation; *unused* by the S2WTM
                       closed-form path because the target distribution
                       (Uniform on S^1) is integrated analytically.

    Returns
    -------
    scalar tensor — average W₂² over the M circular slices.
    """
    # ``n_uniform`` / ``target`` retained for backward compat — closed-form
    # target needs neither.
    del n_uniform, target

    N, d = x.shape
    x_unit = F.normalize(x, p=2, dim=-1)

    # Stiefel V_{d,2} via QR of an i.i.d. Gaussian — yields uniform 2-frames.
    Z = torch.randn(n_projections, d, 2, device=x.device, dtype=x.dtype)
    U, _ = torch.linalg.qr(Z)                                       # (M, d, 2)

    # Project each sample onto each 2-plane:  U_m^T · x_n  ∈ R^2.
    #   (M, 2, d) @ (N, d, 1)   broadcasting → (M, N, 2)
    Xps = torch.matmul(U.transpose(1, 2).unsqueeze(1),
                        x_unit.unsqueeze(-1)).reshape(n_projections, N, 2)

    # Project plane points onto S^1 then convert to circular coordinate.
    Xps = F.normalize(Xps, p=2, dim=-1)
    angles = (torch.atan2(-Xps[..., 1], -Xps[..., 0]) + math.pi) / (2 * math.pi)
    return _w2_uniform_circle(angles).mean()


def ssw_per_level(z_levels: torch.Tensor, mask: torch.Tensor,
                  *, n_projections: int = 50) -> torch.Tensor:
    """Depth-aware SSW (planned §2 — accumulated over the active levels of
    every aspect).

    Parameters
    ----------
    z_levels : (B, n_aspects, L_max, d) residual representations per level.
    mask     : (B, n_aspects, L_max) {0,1} depth mask.
    """
    B, n_aspects, L_max, d = z_levels.shape
    z_flat = z_levels.reshape(-1, d)                                # (B*n*L, d)
    m_flat = mask.reshape(-1)                                       # (B*n*L,)
    active = m_flat > 0.5
    if active.sum() == 0:
        return z_levels.new_zeros(())
    return ssw(z_flat[active], n_projections=n_projections)
