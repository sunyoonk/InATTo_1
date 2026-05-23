"""Identifier sequence builder (spec §3.3.5 + §5.1).

Per-item identifier layout (length sum(L_k) + n + 1):
    aspect 1: c_1^(1) c_1^(2) ... c_1^(L_1) EOA
    aspect 2: c_2^(1) c_2^(2) ... c_2^(L_2) EOA
    ...
    aspect n: c_n^(1) c_n^(2) ... c_n^(L_n) EOA
    EOI

Per-user identifier (no EOI; wrapped by USER_BOS/USER_EOI in T5 input):
    aspect 1: c_1^(1) ... EOA
    ...
    aspect n: c_n^(1) ... EOA

Full T5 input (one sample):
    [USER_BOS] <user identifier> [USER_EOI] [HIST_BOS] <item_1 identifier> ... <item_t identifier> [HIST_EOI]

T5 target:
    <next item identifier>     (ends in EOI)

This module produces:
    inputs_embeds      (B, seq_len, d_t5)         continuous, via STE bridge
    inputs_attn_mask   (B, seq_len)               1=real / 0=pad
    target_ids         (B, target_len)            discrete T5 token IDs
    target_attn_mask   (B, target_len)

T5's CrossEntropy loss handles target as discrete ids (no STE needed there).
"""

from __future__ import annotations
from typing import Iterable, Sequence

import torch
import torch.nn as nn

from .ste_bridge import STEBridge


class IdentifierBuilder(nn.Module):
    """Assemble T5 input/target sequences for a batch.

    Parameters
    ----------
    bridge : STEBridge
        Provides codebook -> T5 token mapping + STE-attached embeddings.
    L_max : int
        Max RQ depth (each aspect has at most L_max codewords).
    n_aspects : int
    history_max_len : int
        Truncate user history to the last N items.
    """

    def __init__(self, bridge: STEBridge, L_max: int, n_aspects: int,
                 history_max_len: int = 10):
        super().__init__()
        self.bridge = bridge
        self.L_max = int(L_max)
        self.n_aspects = int(n_aspects)
        self.history_max_len = int(history_max_len)

    # ------------------------------------------------------------------
    # Length math
    # ------------------------------------------------------------------
    def item_token_len(self, depth_mask: torch.Tensor) -> torch.Tensor:
        """depth_mask (B, n, L) -> (B,) length of each item's identifier,
        counting active codewords + n EOAs + 1 EOI."""
        active_codes = depth_mask.sum(dim=(1, 2))                    # (B,)
        return active_codes + self.n_aspects + 1

    def user_token_len(self, depth_mask: torch.Tensor) -> torch.Tensor:
        """User identifier length: codes + n EOAs (no EOI)."""
        return depth_mask.sum(dim=(1, 2)) + self.n_aspects

    # ------------------------------------------------------------------
    # Building one identifier's token id sequence (for target side)
    # ------------------------------------------------------------------
    def item_token_ids(
        self,
        codes: torch.Tensor,          # (n, L_max) long
        depth_mask: torch.Tensor,     # (n, L_max) {0,1}
    ) -> list[int]:
        """Return T5 token id sequence for one item (variable length)."""
        eoa = self.bridge.special_ids["<EOA>"]
        eoi = self.bridge.special_ids["<EOI>"]
        seq: list[int] = []
        for k in range(self.n_aspects):
            for l in range(self.L_max):
                if depth_mask[k, l].item() > 0.5:
                    code = int(codes[k, l].item())
                    seq.append(int(self.bridge.code_to_t5[code].item()))
            seq.append(eoa)
        seq.append(eoi)
        return seq

    def user_token_ids(
        self,
        codes: torch.Tensor,
        depth_mask: torch.Tensor,
    ) -> list[int]:
        eoa = self.bridge.special_ids["<EOA>"]
        seq: list[int] = []
        for k in range(self.n_aspects):
            for l in range(self.L_max):
                if depth_mask[k, l].item() > 0.5:
                    code = int(codes[k, l].item())
                    seq.append(int(self.bridge.code_to_t5[code].item()))
            seq.append(eoa)
        return seq

    # ------------------------------------------------------------------
    # Building one identifier's *embedding* sequence (for input side, STE)
    # ------------------------------------------------------------------
    def _item_embeds(
        self,
        codes: torch.Tensor,                       # (n, L_max)
        depth_mask: torch.Tensor,                  # (n, L_max)
        soft_assignment: torch.Tensor | None,      # (n, L_max, V) or None
        include_final_eoi: bool,
    ) -> torch.Tensor:
        """Return embedding tensor (seq_len, d_t5) for one item or user."""
        eoa_emb = self.bridge.special_embed("<EOA>")              # (d_t5,)
        eoi_emb = self.bridge.special_embed("<EOI>") if include_final_eoi else None
        d_t5 = eoa_emb.shape[0]

        chunks: list[torch.Tensor] = []
        for k in range(self.n_aspects):
            for l in range(self.L_max):
                if depth_mask[k, l].item() <= 0.5:
                    continue
                code = codes[k, l].view(())
                soft = (soft_assignment[k, l] if soft_assignment is not None else None)
                emb = self.bridge(code, soft_assignment=soft)     # (d_t5,)
                chunks.append(emb.view(1, d_t5))
            chunks.append(eoa_emb.view(1, d_t5))
        if include_final_eoi:
            chunks.append(eoi_emb.view(1, d_t5))
        return torch.cat(chunks, dim=0) if chunks else eoa_emb.new_zeros(0, d_t5)

    # ------------------------------------------------------------------
    # Batch assembly
    # ------------------------------------------------------------------
    def build_inputs(
        self,
        user_codes: torch.Tensor,                   # (B, n, L_max)
        user_mask: torch.Tensor,                    # (B, n, L_max)
        history_codes: torch.Tensor,                # (B, T, n, L_max)
        history_masks: torch.Tensor,                # (B, T, n, L_max)
        user_soft: torch.Tensor | None = None,      # (B, n, L_max, V)
        history_soft: torch.Tensor | None = None,   # (B, T, n, L_max, V)
        history_valid: torch.Tensor | None = None,  # (B, T) bool — pad mask for short histories
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Build (inputs_embeds, attn_mask) for a batch."""
        B, T = history_codes.shape[0], history_codes.shape[1]
        device = self.bridge.t5_embed_table().device
        d_t5 = self.bridge.d_t5

        ub_emb = self.bridge.special_embed("<USER_BOS>").view(1, d_t5)
        ue_emb = self.bridge.special_embed("<USER_EOI>").view(1, d_t5)
        hb_emb = self.bridge.special_embed("<HIST_BOS>").view(1, d_t5)
        he_emb = self.bridge.special_embed("<HIST_EOI>").view(1, d_t5)

        seq_list: list[torch.Tensor] = []
        max_len = 0
        for b in range(B):
            u_emb = self._item_embeds(
                user_codes[b], user_mask[b],
                user_soft[b] if user_soft is not None else None,
                include_final_eoi=False,
            )
            block = [ub_emb, u_emb, ue_emb, hb_emb]
            t_eff = T if history_valid is None else int(history_valid[b].sum().item())
            for t in range(t_eff):
                it_emb = self._item_embeds(
                    history_codes[b, t], history_masks[b, t],
                    history_soft[b, t] if history_soft is not None else None,
                    include_final_eoi=True,
                )
                block.append(it_emb)
            block.append(he_emb)
            seq = torch.cat(block, dim=0)
            seq_list.append(seq)
            max_len = max(max_len, seq.shape[0])

        # Right-pad with zeros, build attention mask.
        inputs = torch.zeros(B, max_len, d_t5, device=device, dtype=seq_list[0].dtype)
        attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, seq in enumerate(seq_list):
            inputs[b, : seq.shape[0]] = seq
            attn[b, : seq.shape[0]] = 1
        return inputs, attn

    def build_targets(
        self,
        target_codes: torch.Tensor,                 # (B, n, L_max)
        target_masks: torch.Tensor,                 # (B, n, L_max)
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return (labels, attn_mask). Padding uses -100 (ignored by T5 CE)."""
        device = self.bridge.t5_embed_table().device
        B = target_codes.shape[0]
        seqs = [self.item_token_ids(target_codes[b], target_masks[b]) for b in range(B)]
        max_len = max(len(s) for s in seqs)
        labels = torch.full((B, max_len), -100, dtype=torch.long, device=device)
        attn = torch.zeros(B, max_len, dtype=torch.long, device=device)
        for b, s in enumerate(seqs):
            labels[b, : len(s)] = torch.tensor(s, dtype=torch.long, device=device)
            attn[b, : len(s)] = 1
        return labels, attn
