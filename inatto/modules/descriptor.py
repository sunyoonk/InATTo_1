"""Descriptor generator — frozen LLM (MiniLM) encodes the codeword sequence
into a single sentence embedding h_d (spec §3.2 / §4.1 of the design).

Two pathways:

  (a) Text path  [explanation, eval, identifier visualization]:
        words = [vocabulary[code_idx] for active levels]
        prompt = "This <entity> can be described by these aspects: " +
                 ", ".join(words)
        h_d = MiniLM.encode(prompt)
      Non-differentiable wrt codebook (discrete words).

  (b) Inputs-embeds path  [training, FACE-style]:
        - For each active codeword, take the d_aspect vector c_k^(l)
        - Map back to MiniLM-space via reverse W_c (pseudo-inverse)
        - Build a soft-token sequence: [CLS] prompt_tokens [SEP] w1 , w2 , ... [SEP]
        - Pass through MiniLM in `inputs_embeds` mode → sentence embedding
      Differentiable wrt W_c (and hence wrt z_aspect).

The training pipeline uses (b). (a) is provided for eval / debug.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel

from .codebook import Codebook


ENTITY_WORD = {
    "user": "user",
    "item": "item",
    "beauty": "product",
    "toys": "toy",
    "sports": "product",
    "yelp": "restaurant",
}


class Descriptor(nn.Module):
    """Differentiable descriptor encoder.

    Parameters
    ----------
    codebook : Codebook
        Owns the W_c projection (so we can invert it for the inputs-embeds path).
    llm_path : str | Path
        Where to load MiniLM (same path used by codebook).
    """

    def __init__(self, codebook: Codebook, llm_path: str | Path):
        super().__init__()
        llm_path = str(llm_path)
        self.codebook = codebook
        self.tokenizer = AutoTokenizer.from_pretrained(llm_path)
        self.llm = AutoModel.from_pretrained(llm_path)
        for p in self.llm.parameters():
            p.requires_grad = False
        self.llm.eval()
        self.d_llm = int(self.llm.config.hidden_size)

        # Cache static prompt prefix tokens per entity type at first use.
        self._prompt_cache: dict[str, torch.Tensor] = {}

    @torch.no_grad()
    def _prompt_embedding(self, entity: str) -> torch.Tensor:
        """Return prompt prefix embedding (T_p, d_llm) for the given entity."""
        if entity in self._prompt_cache:
            return self._prompt_cache[entity]
        word = ENTITY_WORD.get(entity, "item")
        prompt = f"This {word} can be described by these aspects:"
        ids = self.tokenizer(prompt, return_tensors="pt",
                              add_special_tokens=False)["input_ids"][0]
        emb = self.llm.get_input_embeddings()(ids.to(self.llm.device)).detach()
        self._prompt_cache[entity] = emb
        return emb

    @torch.no_grad()
    def _comma_embedding(self) -> torch.Tensor:
        if "," in self._prompt_cache:
            return self._prompt_cache[","]
        ids = self.tokenizer(",", return_tensors="pt",
                              add_special_tokens=False)["input_ids"][0]
        emb = self.llm.get_input_embeddings()(ids.to(self.llm.device)).detach()
        self._prompt_cache[","] = emb
        return emb

    def _add_special(self, x: torch.Tensor) -> torch.Tensor:
        """Prepend [CLS] and append [SEP] embeddings to a sequence (B, T, d)."""
        cls_id = self.tokenizer.cls_token_id
        sep_id = self.tokenizer.sep_token_id
        cls_emb = self.llm.get_input_embeddings().weight[cls_id]
        sep_emb = self.llm.get_input_embeddings().weight[sep_id]
        B = x.shape[0]
        cls = cls_emb.view(1, 1, -1).expand(B, 1, -1)
        sep = sep_emb.view(1, 1, -1).expand(B, 1, -1)
        return torch.cat([cls, x, sep], dim=1)

    @staticmethod
    def _mean_pool(model_output, mask):
        last = model_output[0]
        m = mask.unsqueeze(-1).expand(last.size()).float()
        return (last * m).sum(1) / m.sum(1).clamp_min(1e-9)

    # ------------------------------------------------------------------
    # Differentiable inputs-embeds path (used during training)
    # ------------------------------------------------------------------
    def forward(
        self,
        c_levels: torch.Tensor,     # (B, K, L_max, d_aspect)  per-level codeword vectors
        depth_mask: torch.Tensor,   # (B, K, L_max)            ∈ {0,1}, active codeword mask
        entity: str,
    ) -> torch.Tensor:
        """Encode each sample's active codewords into h_d (B, d_llm) with
        a variable-length soft-token sequence.

        Aligned with the identifier convention (id_builder.item_token_ids):
        all active levels of every aspect are emitted as soft tokens, and
        inactive codewords are removed from MiniLM's attention via
        ``attention_mask`` — so the effective sequence length matches the
        identifier length L_i = Σ_k Σ_l m_{k,l}.

        Steps (per sample, batched):
          1. words_embed = reverse_W_c(c_levels)              (B, K, L, d_llm)
             flatten level→sequence:                          (B, K*L, d_llm)
          2. Prepend prompt prefix
          3. Wrap with [CLS] / [SEP]
          4. attention_mask = [1 ... 1 | depth_mask_flat | 1]
          5. MiniLM(inputs_embeds=...) → mean-pool over attention_mask
             → L2 normalize → (B, d_llm)
        """
        B, K, L, d_a = c_levels.shape
        # (1) reverse W_c per codeword.
        # ★ Detach c_levels so the alignment loss does NOT train W_c. The
        # codebook projection W_c is meant to be learned only by L_recon
        # (FACE / SimVQ design); letting alignment backprop into W_c
        # through K*L = 32 codeword positions amplifies the gradient ~4×
        # vs the fixed-K (z_hat sum) descriptor and causes codebook
        # collapse (L_Q -> 0). We keep gradient flow only to the depth_mask
        # (via mean-pool denominator), training phi but leaving W_c untouched.
        words = self.codebook.reverse_W_c(c_levels.detach().reshape(B * K * L, d_a))
        words = words.reshape(B, K * L, -1)                                # (B, K*L, d_llm)
        codeword_mask = depth_mask.reshape(B, K * L).to(words.dtype)       # (B, K*L)

        # (2) Prepend prompt prefix (always attended)
        prompt = self._prompt_embedding(entity).unsqueeze(0).expand(B, -1, -1)  # (B, T_p, d_llm)
        T_p = prompt.shape[1]
        seq = torch.cat([prompt, words], dim=1)                            # (B, T_p+K*L, d_llm)

        # (3) [CLS] ... [SEP] (special tokens always attended)
        seq = self._add_special(seq)                                       # (B, 1+T_p+K*L+1, d_llm)

        # (4) attention_mask: 1 for CLS, prompt, SEP; depth_mask for codeword positions
        ones_prompt = codeword_mask.new_ones(B, T_p)
        ones_special = codeword_mask.new_ones(B, 1)
        attn = torch.cat([ones_special, ones_prompt, codeword_mask, ones_special],
                          dim=1).long()                                    # (B, 1+T_p+K*L+1)

        # (5) MiniLM
        model_out = self.llm(inputs_embeds=seq, attention_mask=attn)
        emb = self._mean_pool(model_out, attn)
        return F.normalize(emb, p=2, dim=-1)
