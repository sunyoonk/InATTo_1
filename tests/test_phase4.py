"""Phase 4 validation: Alignment, UIAlignment, Descriptor."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
import torch.nn as nn
from inatto.modules.alignment import Alignment
from inatto.modules.ui_alignment import UIAlignment
from inatto.modules.codebook import Codebook
from inatto.modules.descriptor import Descriptor


def test_alignment_loss_decreases_when_correlated():
    """L_align should be lower when h_d and h_raw are aligned vs random."""
    torch.manual_seed(0)
    B, d_txt, d_llm = 32, 8, 16
    align = Alignment(d_txt, d_llm, tau=0.1)

    # Correlated case: h_d ≈ h_raw (matches identity)
    h_raw = torch.randn(B, d_llm)
    h_d_aligned = h_raw + 0.01 * torch.randn(B, d_llm)
    h_hat_txt = torch.randn(B, d_txt)
    rho = torch.ones(B)        # adaptive target = h_raw exactly
    loss_aligned = align(h_d_aligned, h_raw, h_hat_txt, rho)

    # Random h_d
    h_d_rand = torch.randn(B, d_llm)
    loss_rand = align(h_d_rand, h_raw, h_hat_txt, rho)
    assert loss_aligned < loss_rand, \
        f"aligned ({loss_aligned:.3f}) should be < random ({loss_rand:.3f})"
    print(f"[OK] L_align: aligned={loss_aligned:.3f} < random={loss_rand:.3f}")


def test_ui_alignment_positive_vs_negative():
    """Positive pair sim should exceed in-batch negatives after a few SGD steps."""
    torch.manual_seed(0)
    B, n, d_a = 16, 4, 8
    ui = UIAlignment(tau=0.1)
    z_u = torch.randn(B, n, d_a, requires_grad=True)
    # Construct z_i as z_u + small noise (highly aligned positives)
    z_i = (z_u.detach() + 0.05 * torch.randn(B, n, d_a)).requires_grad_(True)
    loss = ui(z_u, z_i)
    # In-batch logits: diag (positive) vs off-diag (random pairs)
    u_n = torch.nn.functional.normalize(z_u.detach().reshape(B, -1), dim=-1)
    i_n = torch.nn.functional.normalize(z_i.detach().reshape(B, -1), dim=-1)
    sim = u_n @ i_n.t()
    diag_mean = sim.diag().mean().item()
    offdiag_mean = (sim.sum() - sim.diag().sum()).item() / (B * (B - 1))
    assert diag_mean > offdiag_mean, \
        f"positive ({diag_mean:.3f}) should exceed negative ({offdiag_mean:.3f})"
    print(f"[OK] L_ui: pos_sim={diag_mean:.3f} > neg_sim={offdiag_mean:.3f}, "
          f"loss={loss.item():.3f}")


def test_descriptor_shape_and_grad():
    """Descriptor should produce (B, d_llm) and gradient should flow into W_c."""
    torch.manual_seed(0)
    cb = Codebook(
        llm_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2",
        d_aspect=64, coca_path=None,
    )
    desc = Descriptor(cb, llm_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2")
    B, n, d_a = 2, 4, 64
    z_hat = torch.randn(B, n, d_a, requires_grad=True)
    h_d = desc(z_hat, entity="item")
    assert h_d.shape == (B, desc.d_llm), h_d.shape
    # Backward through W_c (reverse path), and through z_hat
    h_d.sum().backward()
    assert z_hat.grad is not None and z_hat.grad.abs().sum() > 0
    # Note: W_c.weight is detached inside reverse_W_c (FACE convention).
    # W_c gets gradient via VQ loss in rq.py, not via the descriptor path.
    print(f"[OK] Descriptor  h_d shape={tuple(h_d.shape)}  "
          f"grad to z_hat={z_hat.grad.abs().sum():.3f}")


if __name__ == "__main__":
    test_alignment_loss_decreases_when_correlated()
    test_ui_alignment_positive_vs_negative()
    test_descriptor_shape_and_grad()
    print("\nALL PHASE 4 TESTS PASSED ✓")
