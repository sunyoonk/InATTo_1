"""Phase 7 dry-run: 5-epoch training loop over a tiny synthetic dataset."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from inatto.e2e_model import InATToE2E
from trainer.phase_scheduler import PhaseSchedule
from trainer.trainer_e2e import E2ETrainer


LLM_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2"
T5_PATH = "/home/koohy/cikm/InATTo/InATTo_impl/LLMs/t5-small"


def _build_components():
    torch.manual_seed(0); np.random.seed(0)
    n_user, n_item = 40, 50
    d_cf, d_txt, d_llm = 16, 8, 384
    z_u = torch.randn(n_user, d_cf); h_u = torch.randn(n_user, d_txt)
    rho_u = torch.rand(n_user); hrw_u = torch.randn(n_user, d_llm)
    z_i = torch.randn(n_item, d_cf); h_i = torch.randn(n_item, d_txt)
    rho_i = torch.rand(n_item); hrw_i = torch.randn(n_item, d_llm)
    cfg = dict(n_aspects=4, d_aspect=32, L_max=3, K_neighbors=5, phi_hidden=16,
               history_max_len=4)
    model = InATToE2E(
        user_buffers=(z_u, h_u, rho_u, hrw_u),
        item_buffers=(z_i, h_i, rho_i, hrw_i),
        llm_path=LLM_PATH, t5_path=T5_PATH, coca_path=None,
        cfg=cfg,
    )
    # Build a tiny synthetic dataset (N samples)
    N = 64
    uids = torch.randint(0, n_user, (N,))
    tgts = torch.randint(0, n_item, (N,))
    hist = torch.randint(0, n_item, (N, 4))
    valid = torch.ones(N, 4, dtype=torch.long)
    ds = TensorDataset(uids, tgts, hist, valid)
    loader = DataLoader(ds, batch_size=8, shuffle=True)
    return model, loader


def test_dry_run_5_epochs():
    model, loader = _build_components()
    schedule = PhaseSchedule(warmup_end=2, joint_end=4, total_epochs=5)
    trainer = E2ETrainer(model, schedule, lr_tokenizer=1e-3, lr_t5=1e-4,
                          device="cuda" if torch.cuda.is_available() else "cpu")
    history = trainer.fit(loader, val_loader=loader)
    assert len(history) == 5
    # Phase transitions correct
    assert history[0]["phase"] == "warmup"
    assert history[2]["phase"] == "joint"
    assert history[4]["phase"] == "refinement"
    # Losses are all finite
    for h in history:
        for k in ("L_total", "L_gen", "L_recon", "L_Q", "L_align", "L_ui", "L_rate"):
            assert np.isfinite(h[k]), f"{h['phase']} {k} = {h[k]}"
    # During refinement, tokenizer should be frozen — at least we can check
    # the model says so internally
    model.set_phase("refinement")
    assert not any(p.requires_grad for p in model.tokenizer.parameters())
    assert all(p.requires_grad for p in model.bridge.t5.parameters())
    print("[OK] dry-run completed across all 3 phases with finite losses")


if __name__ == "__main__":
    test_dry_run_5_epochs()
    print("\nPHASE 7 DRY-RUN PASSED ✓")
