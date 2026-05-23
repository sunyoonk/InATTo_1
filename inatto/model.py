"""InATTo model — tokenizer + T5 + alignment, jointly trainable
(spec §5 + §6).

Composition:
    - InATToTokenizer (Module 1)
        * shared params, mode-aware
        * applied to: user, target item, history items
    - Descriptor x 2 (user-side, item-side)
        * shared frozen MiniLM; user/item differ only in the prompt prefix
    - Alignment x 2 (user-side, item-side)
        * separate W_align per side
    - UIAlignment
        * cross-side InfoNCE
    - STEBridge + IdentifierBuilder
        * shared T5
    - T5 generative head

Forward returns total loss + dict of components for logging.

Three-phase schedule (per spec §6.2 / §6.3):
    warmup     T5 frozen,         lambda_gen=0,  tokenizer/alignment train
    joint      everything trains, lambda_gen=1.0
    refinement tokenizer/alignment frozen, T5 trains, lambda_gen=1.0 (others 0)
"""

from __future__ import annotations
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

from .modules.codebook import Codebook
from .modules.alignment import Alignment
from .modules.ui_alignment import UIAlignment
from .modules.descriptor import Descriptor
from .modules.ste_bridge import STEBridge
from .modules.id_builder import IdentifierBuilder
from .tokenizer import InATToTokenizer


# Phase-dependent weights & freeze policy.
# align=0.5 (was 0.1): the previous run finished with align ≈ 6.8, barely
# below random InfoNCE baseline log(512)=6.23 → codewords ended up
# semantically unrelated to item categories (toy items got "economic /
# medical / military" codewords). 5× weight pulls codewords toward
# semantically meaningful words. λ_rate=0.0005 retained (winning value
# from sweep #2 — phi multi-modal spread, no collapse).
PHASE_WEIGHTS = {
    "warmup":     dict(gen=0.0, recon=1.0, Q=1.0,  align=0.5, ui=0.1, rate=0.0005),
    "joint":      dict(gen=1.0, recon=0.5, Q=0.5,  align=0.5, ui=0.1, rate=0.0005),
    "refinement": dict(gen=1.0, recon=0.0, Q=0.0,  align=0.0, ui=0.0, rate=0.0),
}


class InATToE2E(nn.Module):
    """End-to-end InATTo for generative recommendation.

    Parameters
    ----------
    user_buffers : tuple of 4 tensors
        (z_user, h_txt_user, rho_user, h_raw_user)
        Shapes  (n_users, d_cf), (n_users, d_txt), (n_users,), (n_users, d_llm)
    item_buffers : tuple of 4 tensors
        Same structure as user_buffers but for items.
    llm_path : str | Path
        MiniLM directory (for codebook + descriptor).
    t5_path : str | Path
        T5-small directory.
    coca_path : str | Path | None
        COCA xlsx for codebook filtering.
    cfg : dict | None
        Hyperparameters; defaults follow spec §7.
    """

    DEFAULT_CFG = dict(
        # 8 aspects (matches FACE word_num=8). With L_max=4 variable depth,
        # identifier length is 8 (all phi=0) to 32+separators (~50 max),
        # comparable to GRAM (5 codes per item) and avoiding the 33-81 length
        # explosion we had with n_aspects=16.
        n_aspects=8,
        d_aspect=256,
        L_max=4,
        K_neighbors=10,
        phi_hidden=64,
        # VRVQ ``conf/vrvq/vrvq_a2.yml`` uses imp2mask_alpha = 2.0. We saw φ
        # collapse with alpha=1.0 because the softer surrogate combined with
        # full_codebook_rate=0.5 cut φ's effective gradient too much. With
        # alpha=2 + full_codebook_rate=0.25 + rate-on-imp-batch-only, the
        # signal balance matches VRVQ exactly.
        alpha_ste=2.0,
        commit_beta=0.25,
        transformer_layers=1,
        transformer_heads=1,
        transformer_dropout=0.0,
        tau_align=0.07,
        tau_ui=0.07,
        history_max_len=10,
        # ---- ablation toggles ----
        ablate_satp=False,                # h_hat = h_txt (no neighbor aggregation)
        ablate_variable_depth=False,      # all aspects use L_max codewords
        ablate_ep_closed_form=False,      # phi = rho * sigmoid(-delta_k_std)
        ablate_adaptive_target=False,     # h_adp = h_raw   (skip rho mixing)
        ablate_ui=False,                  # L_ui contribution = 0
        ablate_no_rho=False,              # zero out ρ in Ep input
        ablate_no_delta=False,            # zero out δ_k in Ep input
        ablate_user_fixed_depth=False,    # user side: depth_mask = ones (item still variable)
        # ---- SSW (S2WTM-style spherical sliced Wasserstein) ----
        # If > 0, adds L_ssw to the total loss, computed on z_aspect (the
        # pre-quantization aspect representation). Prevents aggregated
        # posterior collapse by encouraging z_aspect to spread uniformly
        # on S^{d-1}. See inatto/modules/ssw.py.
        ssw_weight=0.0,
        ssw_n_projections=50,
    )

    def __init__(
        self,
        user_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        item_buffers: tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor],
        llm_path: str | Path,
        t5_path: str | Path,
        coca_path: str | Path | None = None,
        cfg: dict | None = None,
        h_align_target_user: torch.Tensor | None = None,
    ):
        super().__init__()
        cfg = {**self.DEFAULT_CFG, **(cfg or {})}
        self.cfg = cfg

        z_u, h_txt_u, rho_u, h_raw_u = user_buffers
        z_i, h_txt_i, rho_i, h_raw_i = item_buffers

        d_cf = int(z_u.shape[1])
        d_txt = int(h_txt_u.shape[1])

        # ---- Codebook ----
        self.codebook = Codebook(llm_path=llm_path, d_aspect=cfg["d_aspect"],
                                 coca_path=coca_path)
        d_llm = self.codebook.d_llm

        # Save raw semantic targets as buffers
        assert h_raw_u.shape == (z_u.shape[0], d_llm), \
            f"h_raw_user must be ({z_u.shape[0]}, {d_llm}), got {tuple(h_raw_u.shape)}"
        assert h_raw_i.shape == (z_i.shape[0], d_llm)
        self.register_buffer("h_raw_user", h_raw_u.detach().clone())
        self.register_buffer("h_raw_item", h_raw_i.detach().clone())

        # ---- u2i alignment target for user side (optional) ----
        # If provided, this overrides the self-loop h_raw_user as L_align_user
        # target. Pre-computed externally as the (L2-normalized) mean of the
        # h_raw_item over each user's training history — injecting genuine
        # collaborative signal (DAS-style u2i) into the user codebook.
        if h_align_target_user is not None:
            assert h_align_target_user.shape == (z_u.shape[0], d_llm)
            self.register_buffer("h_align_target_user",
                                 h_align_target_user.detach().clone())
            self._use_u2i_user_align = True
        else:
            self._use_u2i_user_align = False

        # ---- Tree-guide cluster ids (multi-level, optional) ----
        # When ``tree_reg_w > 0``, load the pre-computed AHC tree and
        # register *all internal* levels (L1, L2, L3) as separate
        # buffers. The leaf level (each leaf is its own cluster) is
        # skipped because its CE target is trivial. The multi-level
        # contrastive loss enforces hierarchical organization
        # coarse-to-fine on the trainable W_c.
        tree_pkl = cfg.get("tree_pkl", None)
        self._tree_levels: list[str] = []      # names of registered buffers
        if float(cfg.get("tree_reg_w", 0.0)) > 0 and tree_pkl:
            import pickle as _pkl
            with open(tree_pkl, "rb") as _f:
                _tree = _pkl.load(_f)
            n_levels_in_tree = len(_tree["cluster_ids"])
            # All non-leaf levels (last is identity, skip).
            for l, cid_l in enumerate(_tree["cluster_ids"][:-1]):
                cid_t = torch.as_tensor(cid_l, dtype=torch.long)
                assert cid_t.shape[0] == self.codebook.V, (
                    f"tree level-{l} cluster_ids size ({cid_t.shape[0]}) != "
                    f"codebook V ({self.codebook.V})")
                buf_name = f"tree_cluster_ids_l{l}"
                self.register_buffer(buf_name, cid_t)
                self._tree_levels.append(buf_name)
            # back-compat alias for any code path still looking at the L1 buffer
            if self._tree_levels:
                self.tree_cluster_ids = getattr(self, self._tree_levels[0])
            else:
                self.tree_cluster_ids = None
        else:
            self.tree_cluster_ids = None

        # ---- Tokenizer (Module 1) ----
        self.tokenizer = InATToTokenizer(
            z_u, h_txt_u, rho_u, z_i, h_txt_i, rho_i,
            codebook=self.codebook,
            n_aspects=cfg["n_aspects"],
            d_aspect=cfg["d_aspect"],
            L_max=cfg["L_max"],
            K_neighbors=cfg["K_neighbors"],
            phi_hidden=cfg["phi_hidden"],
            alpha_ste=cfg["alpha_ste"],
            commit_beta=cfg["commit_beta"],
            transformer_layers=cfg["transformer_layers"],
            transformer_heads=cfg["transformer_heads"],
            transformer_dropout=cfg["transformer_dropout"],
            ablate_satp=cfg["ablate_satp"],
            ablate_variable_depth=cfg["ablate_variable_depth"],
            ablate_ep_closed_form=cfg["ablate_ep_closed_form"],
            ablate_no_rho=cfg["ablate_no_rho"],
            ablate_no_delta=cfg["ablate_no_delta"],
            ablate_user_fixed_depth=cfg["ablate_user_fixed_depth"],
            use_vrvq_mask=cfg.get("use_vrvq_mask", False),
            vrvq_alpha=cfg.get("vrvq_alpha", 4.0),
            use_gumbel_mask=cfg.get("use_gumbel_mask", False),
            gumbel_tau=cfg.get("gumbel_tau", 1.0),
            gumbel_v2=cfg.get("gumbel_v2", False),
            use_hrq=cfg.get("use_hrq", False),
            hrq_tree_pkl=cfg.get("hrq_tree_pkl", None),
            hrq_parent_constraint=cfg.get("hrq_parent_constraint", False),
            hrq_elcrec_proto=cfg.get("hrq_elcrec_proto", False),
            hrq_ctfidf_repr=cfg.get("hrq_ctfidf_repr", True),
            phi_init_bias=cfg.get("phi_init_bias", 0.0),
        )

        # ---- Descriptor (Module 2 part 1) ----
        self.descriptor = Descriptor(self.codebook, llm_path=llm_path)
        # Descriptor's LLM is frozen — registered separately so we can skip
        # its params from optimization easily.

        # ---- Alignment (Module 2 part 2) ----
        self.align_user = Alignment(d_txt, d_llm, tau=cfg["tau_align"])
        self.align_item = Alignment(d_txt, d_llm, tau=cfg["tau_align"])
        self.ui_align = UIAlignment(tau=cfg["tau_ui"])

        # ---- STE bridge + ID builder + T5 ----
        self.bridge = STEBridge(t5_path=t5_path, vocabulary=self.codebook.vocabulary)
        self.id_builder = IdentifierBuilder(
            self.bridge,
            L_max=cfg["L_max"],
            n_aspects=cfg["n_aspects"],
            history_max_len=cfg["history_max_len"],
        )

    # ----------------------------------------------------------------------
    # Phase scheduling: freeze / unfreeze parameters per phase.
    # ----------------------------------------------------------------------
    def set_phase(self, phase: str) -> None:
        assert phase in PHASE_WEIGHTS, f"unknown phase {phase!r}"
        tok_train = phase in ("warmup", "joint")
        t5_train = phase in ("joint", "refinement")
        align_train = phase in ("warmup", "joint")

        # Tokenizer params (encoder, reliability, codebook.W_c)
        for p in self.tokenizer.parameters():
            p.requires_grad = tok_train
        # Alignment params (W_align user/item)
        for p in self.align_user.parameters(): p.requires_grad = align_train
        for p in self.align_item.parameters(): p.requires_grad = align_train
        # T5 params (only the T5 backbone, not the descriptor MiniLM)
        for p in self.bridge.t5.parameters():
            p.requires_grad = t5_train
        # Descriptor MiniLM stays frozen always.
        for p in self.descriptor.llm.parameters():
            p.requires_grad = False

    # ----------------------------------------------------------------------
    def _tree_reg_loss(self) -> torch.Tensor:
        """Multi-level LETTER-style contrastive tree-guide on the codebook.
        Returns 0 when tree-reg is disabled (no buffer registered).
        Levels with weights = [1.0, 0.5, 0.25] from coarse to fine.
        """
        if not getattr(self, "_tree_levels", None):
            return self.h_raw_item.new_zeros(())
        from .modules.tree_reg import tree_pull_push_loss
        codebook_emb = self.codebook.codebook()        # (V, d_aspect)
        cluster_ids_list = [getattr(self, n) for n in self._tree_levels]
        return tree_pull_push_loss(
            codebook_emb, cluster_ids_list,
            temperature=float(self.cfg.get("tree_reg_temperature", 0.5)),
            level_weights=None,                          # → default [1, 0.5, 0.25]
        )

    # ----------------------------------------------------------------------
    # Forward
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

        # ---- Tokenize user (mode='user') ----
        user_out = self.tokenizer(user_ids, mode="user")

        # ---- Tokenize target item (mode='item') ----
        item_out = self.tokenizer(target_item_ids, mode="item")

        # ---- Tokenize history items (flat, mode='item', no loss contribution) ----
        flat_hist = history_ids.reshape(-1)
        hist_out = self.tokenizer(flat_hist, mode="item")
        hist_codes = hist_out["codes"].reshape(B, T, self.cfg["n_aspects"],
                                                 self.cfg["L_max"])
        hist_mask = hist_out["depth_mask"].reshape(B, T, self.cfg["n_aspects"],
                                                   self.cfg["L_max"])

        # ---- Module 1 losses (user + target item only) ----
        L_recon = user_out["losses"]["L_recon"] + item_out["losses"]["L_recon"]
        L_Q     = user_out["losses"]["L_Q"]     + item_out["losses"]["L_Q"]
        L_rate  = user_out["losses"]["L_rate"]  + item_out["losses"]["L_rate"]
        # ELCRec prototype separation (0 unless HRQ+protos active)
        L_sep   = user_out["losses"].get("L_sep", user_out["z_aspect"].new_zeros(())) \
                  + item_out["losses"].get("L_sep", item_out["z_aspect"].new_zeros(()))

        # ---- φ δ-anchor (★ CF as supervision for variable depth) ----
        # Our prior sparsity analysis found φ–δ Spearman = -0.59:
        # the model already learns "CF–text agreement (small δ) ⇒ go
        # deeper". This loss makes that operational by explicitly
        # anchoring φ to sigmoid(-δ̃) per aspect, *forcing* item-level
        # phi spread so depth_mask becomes variable across items.
        def _phi_anchor(out):
            phi    = out["signals"]["phi"]              # (B, n)
            delta  = out["signals"]["delta_k"]          # (B, n)  per-aspect
            # Standardise δ across the batch (matching how Ep does it).
            d = delta.detach()
            mu, sd = d.mean(), d.std().clamp_min(1e-6)
            dnorm  = (delta.detach() - mu) / sd
            # δ small ⇒ deep (phi high) — sign follows the natural correlation.
            target = torch.sigmoid(-dnorm)
            return ((phi - target) ** 2).mean()
        L_phi_anchor = _phi_anchor(user_out) + _phi_anchor(item_out)

        # ---- Descriptor + Alignment ----
        # Variable-length descriptor: pass per-level codewords + depth_mask so
        # the alignment sequence length matches the identifier length L_i.
        h_d_user = self.descriptor(user_out["c_levels"], user_out["depth_mask"], entity="user")
        h_d_item = self.descriptor(item_out["c_levels"], item_out["depth_mask"], entity="item")
        # User alignment target — u2i (mean of interacted items' h_raw) if
        # h_align_target_user was provided, else self-loop (h_raw_user).
        # Breaking the self-loop injects collaborative signal into the user
        # codebook (DAS-style), making the user side genuinely cross-aligned.
        if self._use_u2i_user_align:
            h_raw_u = self.h_align_target_user[user_ids]
        else:
            h_raw_u = self.h_raw_user[user_ids]
        h_raw_i = self.h_raw_item[target_item_ids]
        # Ablation: w/o adaptive target -> use rho=1 so h_adp = h_raw exactly.
        rho_u = (torch.ones_like(user_out["signals"]["rho"])
                 if self.cfg["ablate_adaptive_target"]
                 else user_out["signals"]["rho"])
        rho_i = (torch.ones_like(item_out["signals"]["rho"])
                 if self.cfg["ablate_adaptive_target"]
                 else item_out["signals"]["rho"])
        L_align_u = self.align_user(
            h_d_user, h_raw_u,
            user_out["signals"]["h_hat_txt"], rho_u,
        )
        L_align_i = self.align_item(
            h_d_item, h_raw_i,
            item_out["signals"]["h_hat_txt"], rho_i,
        )
        L_align = L_align_u + L_align_i

        # ---- UI alignment ----
        if self.cfg["ablate_ui"]:
            L_ui = user_out["z_aspect"].new_zeros(())
        else:
            L_ui = self.ui_align(user_out["z_hat_st"], item_out["z_hat_st"])

        # ---- SSW (spherical sliced Wasserstein) ----
        # Computed on z_aspect flattened over (B, n_aspects). Regularises
        # the pre-quantization aspect distribution toward Uniform(S^{d-1}),
        # preventing aggregated-posterior collapse onto a few clusters.
        if float(self.cfg.get("ssw_weight", 0.0)) > 0:
            from .modules.ssw import ssw
            n = self.cfg["n_aspects"]; d = self.cfg["d_aspect"]
            n_proj = int(self.cfg.get("ssw_n_projections", 50))
            L_ssw_u = ssw(user_out["z_aspect"].reshape(-1, d), n_projections=n_proj)
            L_ssw_i = ssw(item_out["z_aspect"].reshape(-1, d), n_projections=n_proj)
            L_ssw = L_ssw_u + L_ssw_i
        else:
            L_ssw = user_out["z_aspect"].new_zeros(())

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
            labels, lab_attn = self.id_builder.build_targets(
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
            w["gen"] * L_gen
            + w["recon"] * L_recon
            + w["Q"] * L_Q
            + w["align"] * L_align
            + w["ui"] * L_ui
            + w["rate"] * L_rate
            + float(self.cfg.get("ssw_weight", 0.0)) * L_ssw
            + float(self.cfg.get("sep_weight", 0.0)) * L_sep
            + float(self.cfg.get("phi_anchor_w", 0.0)) * L_phi_anchor
            + float(self.cfg.get("tree_reg_w", 0.0)) * self._tree_reg_loss()
        )

        return {
            "L_total":   L_total,
            "L_gen":     L_gen,
            "L_recon":   L_recon,
            "L_Q":       L_Q,
            "L_align":   L_align,
            "L_align_u": L_align_u,
            "L_align_i": L_align_i,
            "L_ui":      L_ui,
            "L_rate":    L_rate,
            "L_ssw":     L_ssw,
            "phi_user":  user_out["signals"]["phi"],
            "phi_item":  item_out["signals"]["phi"],
            "user_codes": user_out["codes"],
            "item_codes": item_out["codes"],
        }
