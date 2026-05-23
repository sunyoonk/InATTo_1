"""STE Bridge — discrete codeword → T5 embedding with straight-through gradient
(spec §5.4).

Operates by extending the T5 tokenizer/embedding table once with:
    - 6 special tokens (USER_BOS, USER_EOI, HIST_BOS, HIST_EOI, EOA, EOI)
    - V codebook words (one new token per codebook entry)

After extension, every codeword corresponds to exactly one T5 token id, so
the discrete forward path is a direct embedding lookup. The soft path,
when provided, weights every codeword's T5 embedding by a soft assignment
over the codebook and STE-replaces the value:

    soft_emb = soft_assignment @ T5_embed[code_to_t5_id]   # (B, ..., d_t5)
    out = soft_emb + (hard_emb - soft_emb).detach()
            └───── value at forward = hard_emb ─────┘
            gradient flows back through soft_emb only

This makes T5's L_gen gradient flow back through soft_assignment (and
therefore through the codebook projection W_c and z_aspect → soft scores).
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
from transformers import T5ForConditionalGeneration, T5Tokenizer


SPECIAL_TOKENS = [
    "<USER_BOS>", "<USER_EOI>", "<HIST_BOS>", "<HIST_EOI>",
    "<EOA>", "<EOI>",
]


def _codeword_to_t5_token(word: str) -> str:
    """Prefix each codebook word so it becomes a single new token in T5 vocab.

    Without a prefix many words would collide with existing T5 SentencePiece
    tokens, splitting them into multiple subwords (e.g. "thyroid" -> ["thy",
    "roid"]). A unique prefix guarantees a 1:1 codeword <-> token mapping.
    """
    return f"<cw_{word}>"


class STEBridge(nn.Module):
    """Codeword index -> T5 input embedding.

    Parameters
    ----------
    t5_path : str | Path
        Path to a local T5 checkpoint dir (e.g., ``LLMs/t5-small``).
    vocabulary : list[str]
        Codebook words from ``inatto.modules.codebook.Codebook.vocabulary``.

    Attributes
    ----------
    t5 : T5ForConditionalGeneration
        The (resized) T5 model. The embedding table grows by
        ``len(SPECIAL_TOKENS) + len(vocabulary)`` rows.
    tokenizer : T5Tokenizer
        Resized tokenizer with the new tokens. Use this when serializing.
    code_to_t5 : LongTensor [V]
        Codebook idx -> T5 token id.
    special_ids : dict[str, int]
        Special token name -> T5 token id.
    """

    def __init__(self, t5_path: str | Path, vocabulary: list[str]):
        super().__init__()
        t5_path = str(t5_path)
        self.tokenizer = T5Tokenizer.from_pretrained(t5_path)
        self.t5 = T5ForConditionalGeneration.from_pretrained(t5_path)

        # Add specials + codebook tokens. additional_special_tokens groups
        # them; we still use add_tokens for the codebook words so they don't
        # appear in the "special" set (they should behave like normal tokens).
        n_special = self.tokenizer.add_special_tokens(
            {"additional_special_tokens": SPECIAL_TOKENS}
        )
        cw_tokens = [_codeword_to_t5_token(w) for w in vocabulary]
        n_added_cw = self.tokenizer.add_tokens(cw_tokens, special_tokens=False)
        self.t5.resize_token_embeddings(len(self.tokenizer))

        # Cache mappings.
        self.special_ids: dict[str, int] = {
            s: self.tokenizer.convert_tokens_to_ids(s) for s in SPECIAL_TOKENS
        }
        code_ids = self.tokenizer.convert_tokens_to_ids(cw_tokens)
        self.register_buffer(
            "code_to_t5", torch.tensor(code_ids, dtype=torch.long)
        )

        # Sanity checks
        assert len(self.code_to_t5) == len(vocabulary)
        assert all(i >= 0 for i in self.code_to_t5.tolist())
        self.V = len(vocabulary)
        self.d_t5 = int(self.t5.get_input_embeddings().weight.shape[1])

    # ------------------------------------------------------------------
    def t5_embed_table(self) -> torch.Tensor:
        """Live view of the T5 input embedding table (V_t5, d_t5)."""
        return self.t5.get_input_embeddings().weight

    def codeword_t5_emb(self) -> torch.Tensor:
        """Per-codeword T5 embedding lookup, (V, d_t5).

        Returns a differentiable view — gradient flows back into the
        T5 embedding table (which is what gets trained by L_gen).
        """
        return self.t5_embed_table()[self.code_to_t5]

    # ------------------------------------------------------------------
    def forward(
        self,
        code_idx: torch.Tensor,                  # (..., ) long
        soft_assignment: torch.Tensor | None = None,  # (..., V)
        training: bool | None = None,
    ) -> torch.Tensor:
        """Returns embedding tensor (..., d_t5).

        If `soft_assignment` is None or model is in eval mode, returns the
        hard lookup. Otherwise applies STE so the forward value equals
        the hard lookup but gradient flows through the soft side.
        """
        if training is None:
            training = self.training

        t5_ids = self.code_to_t5[code_idx]                           # (...,)
        hard = self.t5_embed_table()[t5_ids]                         # (..., d_t5)

        if not training or soft_assignment is None:
            return hard

        # soft_assignment: (..., V) -> (..., d_t5)
        soft = torch.matmul(soft_assignment, self.codeword_t5_emb())
        # STE: forward value = hard, gradient flows through soft.
        return soft + (hard - soft).detach()

    # ------------------------------------------------------------------
    def special_embed(self, name: str) -> torch.Tensor:
        """Embedding (d_t5,) of a special token by name."""
        return self.t5_embed_table()[self.special_ids[name]]
