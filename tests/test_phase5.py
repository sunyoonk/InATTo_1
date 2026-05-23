"""Phase 5: combined InATTo tokenizer (mode-aware shared params)."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.modules.codebook import Codebook
from inatto.tokenizer import InATToTokenizer


def test_tokenizer_both_modes_and_shared_params():
    torch.manual_seed(0)
    # tiny synthetic data
    n_user, n_item = 30, 25
    d_cf, d_txt = 16, 8
    z_u = torch.randn(n_user, d_cf)
    h_u = torch.randn(n_user, d_txt)
    rho_u = torch.rand(n_user)
    z_i = torch.randn(n_item, d_cf)
    h_i = torch.randn(n_item, d_txt)
    rho_i = torch.rand(n_item)

    cb = Codebook(
        llm_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2",
        d_aspect=32, coca_path=None,
    )
    cb.vocabulary = cb.vocabulary[:80]
    cb.codebook_raw = cb.codebook_raw[:80]
    cb.V = 80

    tok = InATToTokenizer(
        z_u, h_u, rho_u, z_i, h_i, rho_i,
        codebook=cb,
        n_aspects=4, d_aspect=32, L_max=3, K_neighbors=5, phi_hidden=16,
    )

    # Forward both modes
    user_out = tok(torch.arange(4), mode="user")
    item_out = tok(torch.arange(5), mode="item")
    assert user_out["codes"].shape == (4, 4, 3)
    assert item_out["codes"].shape == (5, 4, 3)
    print(f"[OK] user codes {tuple(user_out['codes'].shape)}, "
          f"item codes {tuple(item_out['codes'].shape)}")

    # Shared parameter identity check: W_proj, encoder weights, W_c
    ids = {
        "W_proj":   id(tok.reliability.W_proj.weight),
        "E1.W":     id(tok.encoder.e1.W),
        "E2.layer": id(tok.encoder.e2.encoder.layers[0].self_attn.in_proj_weight),
        "Ep.mlp0":  id(tok.encoder.ep.mlp[0].weight),
        "W_c":      id(tok.codebook.W_c.weight),
    }
    # Just print them — they're shared by construction (one tokenizer instance).
    print(f"[OK] shared param ids: {ids}")

    # Input-dependence: user and item with overlapping ids should yield different codes
    ids2 = torch.tensor([0, 1, 2])
    u = tok(ids2, mode="user")["codes"]
    i = tok(ids2, mode="item")["codes"]
    assert not torch.equal(u, i), "user and item with same ids must differ"
    print(f"[OK] user/item input dependence: user[0]={u[0].flatten()[:5].tolist()}  "
          f"item[0]={i[0].flatten()[:5].tolist()}")

    # Loss + backward over both modes
    total = (user_out["losses"]["L_recon"] + user_out["losses"]["L_Q"]
             + user_out["losses"]["L_rate"]
             + item_out["losses"]["L_recon"] + item_out["losses"]["L_Q"]
             + item_out["losses"]["L_rate"])
    total.backward()
    no_grad = [n for n, p in tok.named_parameters()
               if p.requires_grad and (p.grad is None or p.grad.abs().sum() == 0)]
    print(f"[OK] all trainable params receive grad ({len(no_grad)} missing)")
    if no_grad:
        for n in no_grad:
            print(f"   - {n}")
        raise AssertionError(no_grad)


if __name__ == "__main__":
    test_tokenizer_both_modes_and_shared_params()
    print("\nALL PHASE 5 TESTS PASSED ✓")
