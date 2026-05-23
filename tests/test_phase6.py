"""Phase 6: end-to-end model — forward on synthetic batch + phase-aware grad."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.e2e_model import InATToE2E


LLM_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2"
T5_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/t5-small"


def _build_model():
    torch.manual_seed(0)
    n_user, n_item = 30, 25
    d_cf, d_txt, d_llm = 16, 8, 384      # d_llm = MiniLM hidden = 384
    z_u = torch.randn(n_user, d_cf); h_u = torch.randn(n_user, d_txt)
    rho_u = torch.rand(n_user); hrw_u = torch.randn(n_user, d_llm)
    z_i = torch.randn(n_item, d_cf); h_i = torch.randn(n_item, d_txt)
    rho_i = torch.rand(n_item); hrw_i = torch.randn(n_item, d_llm)

    cfg = dict(n_aspects=4, d_aspect=32, L_max=3, K_neighbors=5, phi_hidden=16,
               history_max_len=4)
    return InATToE2E(
        user_buffers=(z_u, h_u, rho_u, hrw_u),
        item_buffers=(z_i, h_i, rho_i, hrw_i),
        llm_path=LLM_PATH,
        t5_path=T5_PATH,
        coca_path=None,    # skip COCA filtering for speed
        cfg=cfg,
    )


def test_forward_synthetic_all_finite():
    m = _build_model()
    B, T = 3, 4
    user_ids = torch.randint(0, 30, (B,))
    tgt_ids = torch.randint(0, 25, (B,))
    hist_ids = torch.randint(0, 25, (B, T))
    valid = torch.tensor([[1, 1, 1, 1], [1, 1, 1, 0], [1, 1, 0, 0]])

    for phase in ("warmup", "joint", "refinement"):
        m.set_phase(phase)
        out = m(user_ids, tgt_ids, hist_ids, valid, phase=phase)
        for k in ("L_total", "L_gen", "L_recon", "L_Q", "L_align", "L_ui", "L_rate"):
            v = out[k]
            assert torch.isfinite(v).item(), f"{phase} {k} = {v.item()} not finite"
        print(f"[OK] {phase}  L_total={out['L_total'].item():.3f}  "
              f"gen={out['L_gen'].item():.3f}  recon={out['L_recon'].item():.3f}  "
              f"align={out['L_align'].item():.3f}  ui={out['L_ui'].item():.3f}")


def test_phase_aware_freezing():
    m = _build_model()
    B, T = 2, 3
    user_ids = torch.randint(0, 30, (B,))
    tgt_ids = torch.randint(0, 25, (B,))
    hist_ids = torch.randint(0, 25, (B, T))
    valid = torch.ones(B, T, dtype=torch.long)

    def grad_groups(m):
        tok_train = any(p.requires_grad for p in m.tokenizer.parameters())
        t5_train  = any(p.requires_grad for p in m.bridge.t5.parameters())
        algn_train = any(p.requires_grad for p in m.align_user.parameters())
        desc_train = any(p.requires_grad for p in m.descriptor.llm.parameters())
        return dict(tok=tok_train, t5=t5_train, alg=algn_train, descLLM=desc_train)

    # warmup: tok+alignment train, T5 frozen, descriptor frozen
    m.set_phase("warmup")
    g = grad_groups(m); print(f"[warmup] {g}")
    assert g["tok"] and g["alg"] and not g["t5"] and not g["descLLM"]

    m.set_phase("joint")
    g = grad_groups(m); print(f"[joint]  {g}")
    assert g["tok"] and g["alg"] and g["t5"] and not g["descLLM"]

    m.set_phase("refinement")
    g = grad_groups(m); print(f"[refine] {g}")
    assert not g["tok"] and not g["alg"] and g["t5"] and not g["descLLM"]

    # Backward in joint and verify grads land in the right buckets
    m.set_phase("joint")
    out = m(user_ids, tgt_ids, hist_ids, valid, phase="joint")
    out["L_total"].backward()
    has_grad = lambda mod: any(
        (p.grad is not None and p.grad.abs().sum() > 0)
        for p in mod.parameters() if p.requires_grad
    )
    assert has_grad(m.tokenizer), "tokenizer should have grad in joint"
    assert has_grad(m.align_user), "align_user should have grad in joint"
    assert has_grad(m.bridge.t5), "T5 should have grad in joint"
    print("[OK] grads land in tokenizer, align, and T5 during joint")


if __name__ == "__main__":
    test_forward_synthetic_all_finite()
    test_phase_aware_freezing()
    print("\nALL PHASE 6 TESTS PASSED ✓")
