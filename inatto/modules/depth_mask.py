"""Depth mask — variable-depth STE (spec §3.3.1, VRVQ Eq.7 surrogate).

Hard mask (forward):
    L_k = max(1, floor(phi_k * L_max))
    m_k^(l) = 1 if l <= L_k else 0
    (Always m_k^(1) = 1 — every aspect contributes at least one codeword.)

Smooth surrogate (backward), VRVQ-style sigmoid:
    tilde_m_k^(l) = sigmoid(alpha_ste * (phi_k * L_max - l + 0.5))

STE:
    m_ste = tilde_m + (m_hard - tilde_m).detach()
"""

from __future__ import annotations
import torch


def depth_mask_ste(phi: torch.Tensor, L_max: int,
                   alpha_ste: float = 5.0) -> torch.Tensor:
    """phi: (B, n)  ->  m_ste: (B, n, L_max).

    Forward = hard binary mask. Backward = smooth sigmoid surrogate
    (VRVQ Eq.7 simplified). Level 1 is always active.
    """
    B, n = phi.shape
    device, dtype = phi.device, phi.dtype
    L_max = int(L_max)

    # Scaled importance s = phi * L_max, shape (B, n, 1)
    s = (phi * L_max).clamp(min=0.0).unsqueeze(-1)
    # Level indices l = 1, 2, ..., L_max
    ls = torch.arange(1, L_max + 1, dtype=dtype, device=device).view(1, 1, L_max)

    # Smooth surrogate: sigmoid(alpha * (s - l + 0.5))
    tilde_m = torch.sigmoid(alpha_ste * (s - ls + 0.5))

    # Hard mask
    m_hard = (s >= ls).to(dtype)

    # Ensure level 1 (index 0) is always 1 (both hard and smooth)
    m_hard[..., 0] = 1.0
    tilde_m_replaced = torch.cat(
        [torch.ones(B, n, 1, dtype=dtype, device=device), tilde_m[..., 1:]], dim=-1
    )

    m_ste = tilde_m_replaced + (m_hard - tilde_m_replaced).detach()
    return m_ste


def depth_mask_gumbel_ste(phi: torch.Tensor, L_max: int,
                            tau: float = 1.0) -> torch.Tensor:
    """★ Gumbel-noise-injected depth mask (★ training stochastic, anti-saturation).

    Motivation:
      With α_ste alone, when φ → 1 (saturation, observed in practice),
      every item gets the *same* mask [1,1,1,1]: variable depth collapses.
      Gumbel noise injects per-level randomness at every batch so that
      even saturated φ produces a distribution of masks — encouraging
      the codebook to learn for *different* mask configurations and the
      importance MLP to find a wider φ spread that survives the noise.

    Formula:
        s = φ * L_max
        g_l = -log(-log(uniform))                       (Gumbel noise)
        m_smooth[l] = sigmoid((s - l + 0.5 + g_l) / τ)
        m_hard[l]   = (s + g_l >= l)
        m_ste = m_smooth + (m_hard - m_smooth).detach()
        m_ste[..., 0] = 1.0                              (level 1 always on)

    At inference (model.eval()), noise is zeroed → returns the hard
    deterministic mask, identical to ``depth_mask_hard``.

    phi : (B, n)
    returns : (B, n, L_max)
    """
    import math
    B, n = phi.shape
    device, dtype = phi.device, phi.dtype
    L_max = int(L_max)

    s = (phi * L_max).clamp(min=0.0).unsqueeze(-1)                # (B, n, 1)
    ls = torch.arange(1, L_max + 1, dtype=dtype, device=device).view(1, 1, L_max)

    if torch.is_grad_enabled():
        # Training-time Gumbel noise. eps clamp avoids log(0).
        u = torch.rand(B, n, L_max, dtype=dtype, device=device).clamp(min=1e-9, max=1 - 1e-9)
        g = -torch.log(-torch.log(u))
    else:
        g = torch.zeros(B, n, L_max, dtype=dtype, device=device)

    # Smooth surrogate with noise (forward + backward)
    logits = (s + g - ls + 0.5) / float(tau)
    tilde_m = torch.sigmoid(logits)

    # Hard mask: also includes noise — so the cache built at extract time
    # is *deterministic* (g=0 because we run under no_grad/eval). Training
    # forward sees stochastic hard masks via STE.
    m_hard = ((s + g) >= ls).to(dtype)

    # Level 1 always on.
    m_hard[..., 0] = 1.0
    tilde_m_replaced = torch.cat(
        [torch.ones(B, n, 1, dtype=dtype, device=device), tilde_m[..., 1:]],
        dim=-1,
    )

    return tilde_m_replaced + (m_hard - tilde_m_replaced).detach()


def depth_mask_gumbel_ste_v2(phi: torch.Tensor, L_max: int,
                              tau: float = 1.0) -> torch.Tensor:
    """★ V2: Forward-deterministic, gradient-noisy Gumbel STE.

    Fixes the train/test distribution shift of v1 (where the hard mask
    used in the forward pass also included Gumbel noise, causing the
    inference deterministic mask to differ from the training forward).

    Forward (★ both train & inference): standard hard threshold
        m_hard^(l) = 𝟙[φ·L_max ≥ l]                ★ no noise

    Backward (training only): Gumbel-perturbed sigmoid surrogate
        g_l    ∼ Gumbel(0, 1)                       (training-only)
        m̃^(l)  = sigmoid((s + g_l − l + 0.5) / τ)
        m      = m̃ + (m_hard − m̃).detach()         standard STE

    Effect:
      - Train/test forward identical (★ no distribution shift)
      - Training backward sees noisy gradients (★ regularizes the
        importance MLP, encourages φ spread)
      - Level 1 always active
    """
    import math
    B, n = phi.shape
    device, dtype = phi.device, phi.dtype
    L_max = int(L_max)

    s = (phi * L_max).clamp(min=0.0).unsqueeze(-1)                # (B, n, 1)
    ls = torch.arange(1, L_max + 1, dtype=dtype, device=device).view(1, 1, L_max)

    if torch.is_grad_enabled():
        u = torch.rand(B, n, L_max, dtype=dtype, device=device).clamp(min=1e-9, max=1 - 1e-9)
        g = -torch.log(-torch.log(u))
    else:
        g = torch.zeros(B, n, L_max, dtype=dtype, device=device)

    # ★ Hard mask: deterministic (no noise) — train/test forward identical
    m_hard = (s >= ls).to(dtype)

    # Smooth surrogate with Gumbel noise (for gradient diversity)
    m_smooth = torch.sigmoid((s + g - ls + 0.5) / float(tau))

    # Level 1 always on
    m_hard[..., 0] = 1.0
    m_smooth_replaced = torch.cat(
        [torch.ones(B, n, 1, dtype=dtype, device=device), m_smooth[..., 1:]],
        dim=-1,
    )

    return m_smooth_replaced + (m_hard - m_smooth_replaced).detach()


def depth_mask_hard(phi: torch.Tensor, L_max: int) -> torch.Tensor:
    """Pure binary mask (no gradient), for inference / identifier extraction."""
    B, n = phi.shape
    device, dtype = phi.device, phi.dtype
    L_max = int(L_max)
    s = (phi * L_max).clamp(min=0.0).unsqueeze(-1)
    ls = torch.arange(1, L_max + 1, dtype=dtype, device=device).view(1, 1, L_max)
    m = (s >= ls).to(dtype)
    m[..., 0] = 1.0
    return m


def _log_cosh(x: torch.Tensor) -> torch.Tensor:
    """Numerically stable log(cosh(x)) = |x| + log(1 + exp(-2|x|)) - log(2)."""
    import math
    return x.abs() + torch.nn.functional.softplus(-2.0 * x.abs()) - math.log(2.0)


def depth_mask_ste_vrvq(phi: torch.Tensor, L_max: int,
                          alpha: float = 4.0) -> torch.Tensor:
    """VRVQ Eq.7 smooth surrogate (i2m, importance-to-mask).

    For each aspect with importance phi in [0, 1], the scaled score
    s = phi * L_max selects which residual levels are active. The mask
    at level k uses the exact VRVQ formula

        f_alpha^k(s) = (1/(2*alpha)) *
                       log( cosh(alpha*(s-k)) / cosh(alpha*(-s+k+1)) ) + 1/2

    which is differentiable everywhere and has *non-vanishing* gradient
    over the entire transition region (unlike a sigmoid that saturates
    far from the boundary). This was the key gradient-estimation fix
    introduced in VRVQ (arXiv 2410.06016, Sec. 3.2) — it lets the
    importance MLP learn a meaningful spread instead of collapsing to
    phi → 1 once recon errors are small.

    We keep STE semantics: forward returns the hard binary mask, backward
    propagates through the smooth surrogate.

    phi : (B, n)
    returns : (B, n, L_max)
    """
    import math
    B, n = phi.shape
    device, dtype = phi.device, phi.dtype
    L_max = int(L_max)

    s = (phi * L_max).clamp(min=0.0).unsqueeze(-1)                    # (B, n, 1)
    ks = torch.arange(L_max, dtype=dtype, device=device).view(1, 1, L_max)  # 0..L_max-1

    # VRVQ formula (Eq. 7). Use log_cosh for numerical stability.
    arg1 = alpha * (s - ks)
    arg2 = alpha * (-s + ks + 1.0)
    f = (_log_cosh(arg1) - _log_cosh(arg2)) / (2.0 * alpha) + 0.5
    tilde_m = f.clamp(0.0, 1.0)                                       # smooth ∈ [0, 1]

    # Hard mask (forward): m_k = 1 if s >= k+1, else 0; level 0 always active.
    ls = ks + 1.0                                                      # 1..L_max
    m_hard = (s >= ls).to(dtype)
    m_hard[..., 0] = 1.0
    tilde_m_replaced = torch.cat(
        [torch.ones(B, n, 1, dtype=dtype, device=device), tilde_m[..., 1:]],
        dim=-1,
    )

    return tilde_m_replaced + (m_hard - tilde_m_replaced).detach()
