"""Phase 1 validation tests for the dual-branch encoder (spec §3.2).

Checks (per checklist):
    1. Forward pass shapes
       input (B, d_cf)  →  z_aspect (B, 16, 256), phi (B, 16)
    2. phi.mean() ≈ 0.5 at init  (Ep is a randomly-initialized MLP, sigmoid)
    3. W_k orthogonal: ||W_k @ W_k.T - I|| < 0.01  (when d_aspect == d_cf)
    4. Backward: all parameters receive non-zero gradients

Run with:
    pixi run -- python -m tests.test_encoder
or:
    pixi run -- pytest -xvs tests/test_encoder.py
"""

from __future__ import annotations
import sys
from pathlib import Path

# Allow running as a script
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.modules.encoder import DualBranchEncoder, MultiProjector


def _make_inputs(B: int, d_cf: int, n: int):
    torch.manual_seed(0)
    z = torch.randn(B, d_cf)
    rho = torch.rand(B)
    delta_k = torch.rand(B, n)
    return z, rho, delta_k


def test_forward_shapes():
    enc = DualBranchEncoder(d_cf=256, n_aspects=16, d_aspect=256)
    z, rho, dk = _make_inputs(B=8, d_cf=256, n=16)
    e, z_aspect, phi = enc(z, rho, dk)
    assert e.shape == (8, 16, 256), e.shape
    assert z_aspect.shape == (8, 16, 256), z_aspect.shape
    assert phi.shape == (8, 16), phi.shape
    print("[OK] forward shapes")


def test_phi_init_distribution():
    enc = DualBranchEncoder(d_cf=256, n_aspects=16, d_aspect=256)
    z, rho, dk = _make_inputs(B=256, d_cf=256, n=16)
    with torch.no_grad():
        _, _, phi = enc(z, rho, dk)
    mean = phi.mean().item()
    print(f"[OK] phi at init: mean={mean:.3f}  std={phi.std():.3f}  "
          f"min={phi.min():.3f}  max={phi.max():.3f}")
    assert 0.40 < mean < 0.60, f"phi.mean() = {mean:.3f} not near 0.5"


def test_W_k_orthogonal():
    enc = MultiProjector(d_cf=256, n_aspects=16, d_aspect=256)
    n = enc.n_aspects
    eye = torch.eye(enc.d_aspect)
    for k in range(n):
        W_k = enc.get_W_k(k)
        err = (W_k @ W_k.T - eye).norm().item()
        assert err < 1e-3, f"aspect {k}: ||W_k W_k^T - I|| = {err:.3e}"
    print(f"[OK] W_k orthogonal for all {n} aspects (max ||W W^T - I|| < 1e-3)")


def test_gradient_flow():
    enc = DualBranchEncoder(d_cf=256, n_aspects=16, d_aspect=256)
    z, rho, dk = _make_inputs(B=16, d_cf=256, n=16)
    e, z_aspect, phi = enc(z, rho, dk)
    # Push a loss that depends on all three outputs.
    loss = z_aspect.pow(2).mean() + phi.mean() + e.abs().mean()
    loss.backward()
    no_grad = []
    for name, p in enc.named_parameters():
        if p.grad is None or p.grad.abs().sum() == 0:
            no_grad.append(name)
    assert not no_grad, f"params with no/zero grad: {no_grad}"
    print(f"[OK] gradients flow to all {sum(1 for _ in enc.parameters())} param tensors")


if __name__ == "__main__":
    test_forward_shapes()
    test_phi_init_distribution()
    test_W_k_orthogonal()
    test_gradient_flow()
    print("\nALL ENCODER TESTS PASSED ✓")
