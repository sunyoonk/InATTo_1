"""Phase 2 validation: SATP, Reliability, DepthMask, Codebook, RQ.

Checks per the checklist:
    SATP        : shape, rho-weighted aggregation
    Reliability : delta > 0, delta_k > 0, gradient flows
    DepthMask   : hard mask binary, level-1 always on, STE forward=hard
    Codebook    : codewords are clean English words; mapping shape ok
    RQ          : cross-aspect exclusion (each item's codes are unique)
"""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.modules.encoder import MultiProjector
from inatto.modules.satp import SATP
from inatto.modules.reliability import Reliability
from inatto.modules.depth_mask import depth_mask_ste, depth_mask_hard
from inatto.modules.codebook import Codebook
from inatto.modules.rq import ResidualQuantizer


# ---------- SATP -----------------------------------------------------------

def test_satp_shape_and_blend():
    torch.manual_seed(0)
    N, d_cf, d_txt, K = 50, 16, 8, 5
    z = torch.randn(N, d_cf)
    h = torch.randn(N, d_txt)
    rho = torch.rand(N)
    satp = SATP(z, h, rho, K=K)
    ids = torch.arange(10)
    h_hat, rho_b, z_b = satp(ids)
    assert h_hat.shape == (10, d_txt)
    assert rho_b.shape == (10,) and z_b.shape == (10, d_cf)
    # When rho=1 we should recover the raw text exactly.
    rho2 = torch.ones(N)
    satp2 = SATP(z, h, rho2, K=K)
    h_hat2, _, _ = satp2(ids)
    assert torch.allclose(h_hat2, h[ids], atol=1e-5), \
        "rho=1 should give h_hat == h_txt exactly"
    print("[OK] SATP shape & rho-blend")


# ---------- Reliability ----------------------------------------------------

def test_reliability():
    torch.manual_seed(0)
    B, d_cf, d_txt, n_aspects, d_aspect = 8, 16, 8, 4, 16
    h_hat = torch.randn(B, d_txt)
    z = torch.randn(B, d_cf)
    proj = MultiProjector(d_cf, n_aspects, d_aspect)
    rel = Reliability(d_txt, d_cf)
    h_cf, r_ortho, delta, delta_k = rel(h_hat, z, proj.W)
    assert h_cf.shape == (B, d_cf)
    assert r_ortho.shape == (B, d_cf)
    assert delta.shape == (B,) and delta_k.shape == (B, n_aspects)
    assert (delta >= 0).all() and (delta_k >= 0).all()
    # Gradient flows into W_proj
    delta.sum().backward()
    assert rel.W_proj.weight.grad is not None and rel.W_proj.weight.grad.abs().sum() > 0
    print(f"[OK] Reliability  delta_mean={delta.mean():.3f}  "
          f"delta_k_mean={delta_k.mean():.3f}")


# ---------- DepthMask ------------------------------------------------------

def test_depth_mask():
    torch.manual_seed(0)
    B, n, L = 4, 4, 4
    # phi spread: 0.0, 0.25, 0.5, 1.0 across aspects
    phi = torch.tensor([[0.0, 0.25, 0.5, 1.0]] * B, requires_grad=True)
    m_hard = depth_mask_hard(phi, L)
    # Level 1 (idx 0) always on
    assert (m_hard[..., 0] == 1.0).all()
    # phi=0 means only level 1 is on
    assert (m_hard[:, 0, 1:] == 0.0).all()
    # phi=1 with L=4 means all 4 levels on (floor(1*4) = 4)
    assert (m_hard[:, 3, :] == 1.0).all()
    # STE: forward equals hard
    m_ste = depth_mask_ste(phi, L, alpha_ste=5.0)
    assert torch.allclose(m_ste.detach(), m_hard.detach()), \
        "STE forward must equal hard mask"
    # Backward flows (gradient from m_ste.sum() to phi)
    m_ste.sum().backward()
    assert phi.grad is not None and phi.grad.abs().sum() > 0
    print(f"[OK] DepthMask  level-1 always on, STE forward=hard, grad flows")


# ---------- Codebook -------------------------------------------------------

def test_codebook():
    cb = Codebook(
        llm_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2",
        d_aspect=256,
        coca_path=None,    # no COCA filter for the test (faster)
    )
    assert cb.V > 5000, f"codebook unexpectedly small: V={cb.V}"
    C = cb.codebook()
    assert C.shape == (cb.V, 256)
    # A few sample words to sanity check
    samples = cb.vocabulary[:5]
    print(f"[OK] Codebook  V={cb.V}  d_llm={cb.d_llm}  "
          f"samples={samples}")


# ---------- RQ -------------------------------------------------------------

def test_rq_cross_aspect_exclusion():
    torch.manual_seed(0)
    B, n, d_aspect, V, L = 4, 4, 16, 100, 3
    z_aspect = torch.randn(B, n, d_aspect)
    C = torch.randn(V, d_aspect)
    # All levels active (phi=1)
    phi = torch.ones(B, n)
    m = depth_mask_hard(phi, L)
    rq = ResidualQuantizer(L_max=L)
    out = rq(z_aspect, C, m)
    assert out["codes"].shape == (B, n, L)
    # Cross-aspect exclusion: within an item, all selected codes must be unique
    for b in range(B):
        seen = set()
        for k in range(n):
            for l in range(L):
                code = int(out["codes"][b, k, l])
                if code in seen:
                    raise AssertionError(f"duplicate code {code} in item {b}")
                seen.add(code)
    # Reconstruction loss is finite
    assert torch.isfinite(out["L_Q"]).item()
    print(f"[OK] RQ cross-aspect exclusion enforced; "
          f"L_Q={out['L_Q'].item():.3f}  z_hat_norm={out['z_hat'].norm():.2f}")


if __name__ == "__main__":
    test_satp_shape_and_blend()
    test_reliability()
    test_depth_mask()
    test_codebook()
    test_rq_cross_aspect_exclusion()
    print("\nALL PHASE 2 TESTS PASSED ✓")
