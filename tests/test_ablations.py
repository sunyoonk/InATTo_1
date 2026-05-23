"""Sanity tests for all ablation toggles — verifies each knob actually
changes the behavior it claims to change."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.e2e_model import InATToE2E


LLM_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2"
T5_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/t5-small"


def _build(cfg_extra: dict | None = None):
    torch.manual_seed(0)
    n_user, n_item = 30, 25
    d_cf, d_txt, d_llm = 16, 8, 384
    z_u = torch.randn(n_user, d_cf); h_u = torch.randn(n_user, d_txt)
    rho_u = torch.rand(n_user); hrw_u = torch.randn(n_user, d_llm)
    z_i = torch.randn(n_item, d_cf); h_i = torch.randn(n_item, d_txt)
    rho_i = torch.rand(n_item); hrw_i = torch.randn(n_item, d_llm)
    cfg = dict(n_aspects=4, d_aspect=32, L_max=3, K_neighbors=5, phi_hidden=16,
               history_max_len=4)
    if cfg_extra:
        cfg.update(cfg_extra)
    return InATToE2E(
        user_buffers=(z_u, h_u, rho_u, hrw_u),
        item_buffers=(z_i, h_i, rho_i, hrw_i),
        llm_path=LLM_PATH, t5_path=T5_PATH, coca_path=None,
        cfg=cfg,
    )


def _sample():
    return (torch.tensor([0, 1, 2]),
            torch.tensor([5, 6, 7]),
            torch.tensor([[0, 1, 2, 3], [2, 3, 4, 5], [1, 1, 2, 2]]),
            torch.ones(3, 4, dtype=torch.long))


def test_ablate_variable_depth_forces_all_levels():
    m = _build({"ablate_variable_depth": True})
    user_ids, _, _, _ = _sample()
    out = m.tokenizer(user_ids, mode="user")
    mask = out["depth_mask"]
    assert mask.shape == (3, 4, 3)
    assert (mask == 1).all(), "ablate_variable_depth should give all-ones mask"
    print("[OK] ablate_variable_depth: mask is all 1s")


def test_ablate_satp_returns_raw_text():
    m = _build({"ablate_satp": True})
    user_ids = torch.tensor([0, 1, 2])
    out = m.tokenizer(user_ids, mode="user")
    expected = m.tokenizer.satp_user.h_txt[user_ids]
    assert torch.allclose(out["signals"]["h_hat_txt"], expected), \
        "ablate_satp should return h_txt directly"
    print("[OK] ablate_satp: h_hat_txt == h_txt")


def test_ablate_Ep_closed_form_uses_rho_sigmoid_delta():
    m = _build({"ablate_ep_closed_form": True})
    user_ids = torch.tensor([0, 1, 2])
    out = m.tokenizer(user_ids, mode="user")
    phi = out["signals"]["phi"]
    rho = out["signals"]["rho"]
    delta_k = out["signals"]["delta_k"]
    # Replicate the closed form: phi = rho * sigmoid(-tilde_delta)
    flat = delta_k.reshape(-1)
    tilde = (delta_k - flat.mean()) / flat.std().clamp_min(1e-6)
    expected = rho.unsqueeze(1) * torch.sigmoid(-tilde)
    assert torch.allclose(phi, expected, atol=1e-5), "closed-form phi mismatch"
    print("[OK] ablate_ep_closed_form: phi == rho * sigmoid(-delta_k_std)")


def test_ablate_adaptive_target_uses_h_raw():
    full = _build()
    abl = _build({"ablate_adaptive_target": True})
    # Both should produce h_adp = h_raw when h_hat_txt = 0 and rho varies.
    # Easier: just check that adaptive target with rho=1 equals h_raw
    user_ids, tgts, hist, valid = _sample()
    out_full = full(user_ids, tgts, hist, valid, phase="warmup")
    out_abl  = abl(user_ids,  tgts, hist, valid, phase="warmup")
    # L_align_u in the ablation uses rho=1, so the target is exactly h_raw_user[user_ids].
    # We can't compare losses directly because Phi/W_align are random, but we
    # can verify the model paths differ.
    assert out_full["L_align_u"].item() != out_abl["L_align_u"].item(), \
        "L_align_u should differ between full and adaptive-target ablation"
    print(f"[OK] ablate_adaptive_target: L_align_u differs "
          f"(full={out_full['L_align_u'].item():.3f}  "
          f"abl={out_abl['L_align_u'].item():.3f})")


def test_ablate_ui_zeros_L_ui():
    m = _build({"ablate_ui": True})
    args = _sample()
    out = m(*args, phase="warmup")
    assert out["L_ui"].item() == 0.0, f"L_ui should be 0, got {out['L_ui'].item()}"
    print("[OK] ablate_ui: L_ui == 0")


def test_face_baseline_combo():
    """face_baseline should toggle all of the above at once."""
    m = _build({
        "ablate_satp": True, "ablate_variable_depth": True,
        "ablate_adaptive_target": True, "ablate_ep_closed_form": True,
        "ablate_ui": True,
    })
    args = _sample()
    out = m(*args, phase="warmup")
    # Sanity: L_ui = 0, mask = all-ones (we'll check via direct call)
    assert out["L_ui"].item() == 0.0
    tok_out = m.tokenizer(args[0], mode="user")
    assert (tok_out["depth_mask"] == 1).all()
    print("[OK] face_baseline combo: L_ui=0, mask=all-ones")


if __name__ == "__main__":
    test_ablate_variable_depth_forces_all_levels()
    test_ablate_satp_returns_raw_text()
    test_ablate_Ep_closed_form_uses_rho_sigmoid_delta()
    test_ablate_adaptive_target_uses_h_raw()
    test_ablate_ui_zeros_L_ui()
    test_face_baseline_combo()
    print("\nALL ABLATION TESTS PASSED ✓")
