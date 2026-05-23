"""End-to-end FACE baseline (C-option) — FACE's VQ-RAF tokenizer plugged
into our T5 generative pipeline.

Differences vs ``InATToE2E``:
    * tokenizer  -> ``FACETokenizer`` (FACE original, wrapped)
    * descriptor -> FACE's own ``get_collaborative_representations``
                     (already inside FACETokenizer)
    * alignment  -> vanilla InfoNCE (FACE's ``cal_align_loss``)
    * dropped    -> SATP, Reliability, Importance phi, UI alignment, rate loss
    * kept (same as InATTo) -> STE bridge, IdentifierBuilder, T5, 3-phase schedule

C-option recap: CF backbone is the *frozen* LightGCN checkpoint (passed in
via ``z_user`` / ``z_item``), so BPR loss is not used here. The only
trained components are the FACE tokenizer (encoder + transformer + VQ +
decoder), the FACE codebook mapping linear, and the T5 generative head.
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.ste_bridge import STEBridge
from .modules.id_builder import IdentifierBuilder
from .face_tokenizer import FACETokenizer


# Phase-dependent loss weights (mirrors InATToE2E but ui/rate are gone).
PHASE_WEIGHTS = {
    "warmup":     dict(gen=0.0, recon=1.0, Q=1.0,  align=0.1),
    "joint":      dict(gen=1.0, recon=0.5, Q=0.5,  align=0.1),
    "refinement": dict(gen=1.0, recon=0.0, Q=0.0,  align=0.0),
}


def _info_nce(a: torch.Tensor, b: torch.Tensor, temperature: float = 0.02
              ) -> torch.Tensor:
    """FACE's `cal_align_loss` (FACE/encoder/models/loss_utils.py:5).

    Symmetric InfoNCE across the batch: positive pair is (a_i, b_i),
    negatives are b_{j != i}. Both inputs already row-normalised.
    """
    a = F.normalize(a, p=2, dim=-1)
    b = F.normalize(b, p=2, dim=-1)
    sim = (a @ b.t()) / temperature
    labels = torch.arange(sim.size(0), device=sim.device)
    return F.cross_entropy(sim, labels)


class FACEE2E(nn.Module):
    """End-to-end FACE baseline matching the InATTo evaluation pipeline."""

    DEFAULT_CFG = dict(
        n_aspects=8,              # FACE original default (word_num)
        d_aspect=256,             # FACE original default (word_dim)
        history_max_len=10,
        align_temperature=0.02,   # FACE original (loss_utils.py:5)
        llm_name="miniLM",
    )

    def __init__(
        self,
        user_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        item_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        t5_path: str | Path,
        dataset_name: str,
        face_root: str | Path | None = None,
        cfg: dict | None = None,
    ):
        super().__init__()
        cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.cfg = cfg
        self.dataset_name = str(dataset_name)

        z_u, h_txt_u, rho_u, h_raw_u = user_buffers
        z_i, h_txt_i, rho_i, h_raw_i = item_buffers
        n_users = int(z_u.shape[0]); n_items = int(z_i.shape[0])
        d_llm = int(h_raw_u.shape[1])

        # ---- Tokenizer (FACE original, wrapped) ----
        self.tokenizer = FACETokenizer(
            z_user=z_u, z_item=z_i,
            h_txt_user=h_txt_u, h_txt_item=h_txt_i,
            rho_user=rho_u, rho_item=rho_i,
            dataset_name=self.dataset_name,
            n_aspects=cfg["n_aspects"],
            d_aspect=cfg["d_aspect"],
            llm_name=cfg["llm_name"],
            face_root=face_root,
        )

        # Profile target buffers for alignment (same as InATToE2E).
        assert h_raw_u.shape == (n_users, d_llm)
        assert h_raw_i.shape == (n_items, d_llm)
        self.register_buffer("h_raw_user", h_raw_u.detach().float().clone())
        self.register_buffer("h_raw_item", h_raw_i.detach().float().clone())

        # ---- Trainer-compatibility shims ----
        # trainer_e2e.py groups params as ``tokenizer + align_user + align_item``
        # and logs ``codebook.V``. FACE has no trainable W_align (it uses a
        # parameter-free InfoNCE) and the codebook lives inside the FACE
        # quantizer, so we expose empty placeholders here for API parity.
        self.align_user = nn.ParameterList()
        self.align_item = nn.ParameterList()

        class _CodebookShim:
            def __init__(self, V): self.V = V
        self.codebook = _CodebookShim(self.tokenizer.V)

        # ---- STE bridge + ID builder + T5 ----
        # FACE's `vocabulary` (V words from BERT vocab ∩ COCA60000) is what
        # we extend T5's tokenizer with — same mechanism as InATTo, just a
        # different filtered word list.
        self.bridge = STEBridge(t5_path=t5_path,
                                vocabulary=self.tokenizer.vocabulary)
        self.id_builder = IdentifierBuilder(
            self.bridge,
            L_max=self.tokenizer.L_max,            # FACE: 1
            n_aspects=cfg["n_aspects"],
            history_max_len=cfg["history_max_len"],
        )

    # ----------------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        assert phase in PHASE_WEIGHTS, f"unknown phase {phase!r}"
        tok_train = phase in ("warmup", "joint")
        t5_train  = phase in ("joint", "refinement")

        # FACE tokenizer (encoder, transformer, VQ, decoder, codebook mapping).
        for p in self.tokenizer.parameters():
            p.requires_grad = tok_train
        # FACE's MiniLM embedding model is frozen by construction.
        for p in self.tokenizer.face.quantizer.embedding_model.parameters():
            p.requires_grad = False
        # T5.
        for p in self.bridge.t5.parameters():
            p.requires_grad = t5_train

    # ----------------------------------------------------------------------
    def forward(
        self,
        user_ids: torch.Tensor,         # (B,)
        target_item_ids: torch.Tensor,  # (B,)
        history_ids: torch.Tensor,      # (B, T)
        history_valid: torch.Tensor,    # (B, T) bool
        phase: str = "joint",
    ) -> dict:
        w = PHASE_WEIGHTS[phase]
        B, T = history_ids.shape

        # ---- Tokenize user, target item, history ----
        user_out = self.tokenizer(user_ids, mode="user")
        item_out = self.tokenizer(target_item_ids, mode="item")
        flat_hist = history_ids.reshape(-1)
        hist_out = self.tokenizer(flat_hist, mode="item")
        n = self.cfg["n_aspects"]
        L = self.tokenizer.L_max
        hist_codes = hist_out["codes"].reshape(B, T, n, L)
        hist_mask  = hist_out["depth_mask"].reshape(B, T, n, L)

        # ---- Reconstruction + VQ losses (user + target item) ----
        L_recon = user_out["losses"]["L_recon"] + item_out["losses"]["L_recon"]
        L_Q     = user_out["losses"]["L_Q"]     + item_out["losses"]["L_Q"]

        # ---- Alignment (FACE original: collaborative repr vs profile MiniLM) ----
        # FACE's collaborative repr is computed via the codebook reverse
        # mapping + a prompt; the wrapper exposes it as one call.
        h_d_user = self.tokenizer.collaborative_representations(
            user_out["z_hat_st"], kind="user")
        h_d_item = self.tokenizer.collaborative_representations(
            item_out["z_hat_st"], kind="item")
        h_raw_u  = self.h_raw_user[user_ids]
        h_raw_i  = self.h_raw_item[target_item_ids]
        L_align = (_info_nce(h_d_user, h_raw_u, self.cfg["align_temperature"])
                   + _info_nce(h_d_item, h_raw_i, self.cfg["align_temperature"]))

        # ---- T5 generative loss (skip in warmup) ----
        if phase == "warmup":
            L_gen = user_out["z_aspect"].new_zeros(())
        else:
            inputs_embeds, attn = self.id_builder.build_inputs(
                user_codes=user_out["codes"],
                user_mask=user_out["depth_mask"],
                history_codes=hist_codes,
                history_masks=hist_mask,
                history_valid=history_valid,
            )
            labels, _ = self.id_builder.build_targets(
                target_codes=item_out["codes"],
                target_masks=item_out["depth_mask"],
            )
            t5_out = self.bridge.t5(
                inputs_embeds=inputs_embeds,
                attention_mask=attn,
                labels=labels,
            )
            L_gen = t5_out.loss

        # ---- Total ----
        L_total = (
            w["gen"]   * L_gen
            + w["recon"] * L_recon
            + w["Q"]     * L_Q
            + w["align"] * L_align
        )

        # Match InATToE2E's returned-key set so the existing trainer logger
        # works unchanged. UI / rate are zeroed out.
        zero = L_total.new_zeros(())
        return {
            "L_total":   L_total,
            "L_gen":     L_gen,
            "L_recon":   L_recon,
            "L_Q":       L_Q,
            "L_align":   L_align,
            "L_align_u": L_align / 2,
            "L_align_i": L_align / 2,
            "L_ui":      zero,
            "L_rate":    zero,
            "phi_user":  user_out["signals"]["phi"],
            "phi_item":  item_out["signals"]["phi"],
            "user_codes": user_out["codes"],
            "item_codes": item_out["codes"],
        }
