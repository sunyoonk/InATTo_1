"""Phase 3 validation: STE bridge + Identifier builder."""

from __future__ import annotations
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import torch
from inatto.modules.codebook import Codebook
from inatto.modules.ste_bridge import STEBridge
from inatto.modules.id_builder import IdentifierBuilder


def _build_components(V: int = 200, d_aspect: int = 256, L: int = 4, n: int = 4):
    # A tiny codebook for fast tests — overwrite Codebook's vocab by hand.
    torch.manual_seed(0)
    cb = Codebook(
        llm_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/all-MiniLM-L6-v2",
        d_aspect=d_aspect, coca_path=None,
    )
    # Limit to V words for the test (avoid resizing T5 by full ~9000)
    cb.vocabulary = cb.vocabulary[:V]
    cb.codebook_raw = cb.codebook_raw[:V]
    cb.token_ids = cb.token_ids[:V]
    cb.V = V

    bridge = STEBridge(
        t5_path="/home/koohy/cikm/InATTo/InATTo_impl/LLMs/t5-small",
        vocabulary=cb.vocabulary,
    )
    builder = IdentifierBuilder(bridge, L_max=L, n_aspects=n, history_max_len=5)
    return cb, bridge, builder


def test_ste_bridge_forward_and_backward():
    cb, bridge, builder = _build_components(V=200)
    V = bridge.V
    d_t5 = bridge.d_t5

    # Hard lookup: shape and value match.
    code_idx = torch.tensor([0, 5, 10])
    hard = bridge(code_idx, soft_assignment=None, training=False)
    assert hard.shape == (3, d_t5)

    # Soft (STE): forward value must equal hard
    soft = torch.zeros(3, V, requires_grad=True)
    soft.data[0, 0] = 1; soft.data[1, 5] = 1; soft.data[2, 10] = 1
    soft = torch.softmax(soft * 10, dim=-1)
    out = bridge(code_idx, soft_assignment=soft, training=True)
    assert out.shape == (3, d_t5)
    assert torch.allclose(out.detach(), hard.detach(), atol=1e-5), \
        "STE forward must equal hard lookup"
    # Backward through soft
    out.sum().backward()
    # T5 embed table should receive gradient via soft pathway
    grad_norm = bridge.t5.get_input_embeddings().weight.grad.abs().sum()
    assert grad_norm > 0, "T5 embedding should receive gradient via soft path"
    print(f"[OK] STE bridge: forward==hard, soft backward grad_norm={grad_norm:.2f}")


def test_identifier_length():
    cb, bridge, builder = _build_components(V=200, L=4, n=16)
    # With phi=1 for all aspects and L_max=4: identifier should be 16*4 + 16 + 1 = 81
    # With phi=0: identifier should be 16*1 + 16 + 1 = 33
    codes = torch.randint(0, 200, (1, 16, 4))
    mask_all = torch.ones(1, 16, 4)
    mask_min = torch.zeros(1, 16, 4); mask_min[:, :, 0] = 1
    seq_max = builder.item_token_ids(codes[0], mask_all[0])
    seq_min = builder.item_token_ids(codes[0], mask_min[0])
    assert len(seq_max) == 81, f"max len = {len(seq_max)} != 81"
    assert len(seq_min) == 33, f"min len = {len(seq_min)} != 33"
    print(f"[OK] Identifier length: min={len(seq_min)}, max={len(seq_max)} (matches spec [33, 81])")


def test_build_inputs_and_targets():
    cb, bridge, builder = _build_components(V=200, L=3, n=4)
    B, T = 2, 3
    V = bridge.V
    user_codes = torch.randint(0, V, (B, 4, 3))
    user_mask = torch.ones(B, 4, 3); user_mask[:, :, 2] = 0   # drop deepest level
    history_codes = torch.randint(0, V, (B, T, 4, 3))
    history_masks = torch.ones(B, T, 4, 3)

    inputs, attn = builder.build_inputs(user_codes, user_mask, history_codes, history_masks)
    assert inputs.ndim == 3 and inputs.shape[0] == B and inputs.shape[2] == bridge.d_t5
    assert attn.shape == (B, inputs.shape[1])
    print(f"[OK] build_inputs  shape={tuple(inputs.shape)}  mask_sum_per_row={attn.sum(1).tolist()}")

    labels, lab_attn = builder.build_targets(user_codes, user_mask)
    assert labels.shape == (B, lab_attn.shape[1])
    print(f"[OK] build_targets shape={tuple(labels.shape)}  mask_sum={lab_attn.sum(1).tolist()}")


if __name__ == "__main__":
    test_ste_bridge_forward_and_backward()
    test_identifier_length()
    test_build_inputs_and_targets()
    print("\nALL PHASE 3 TESTS PASSED ✓")
