"""FACE original tokenizer adapter — wraps FACE/encoder/FACE.py to plug into
our InATTo end-to-end T5 pipeline (C-option baseline).

Design (C-option):
    * BPR loss dropped (we don't train CF embeddings here)
    * CF backbone: frozen LightGCN checkpoint (same as InATTo)
    * VQ-RAF (FACE original): single VQ, no residual, no variable depth
    * Descriptor + InfoNCE alignment: FACE original (prompt + collaborative repr)
    * T5 generative head: our STE bridge + identifier builder

The wrapper exposes the same dict interface as
``inatto.tokenizer.InATToTokenizer.forward``, so that
``e2e_model.InATToE2E``-style code (and downstream eval) works unchanged.

FACE's modules use cwd-relative paths (``./LLMs/...``, ``./data/...``).
We temporarily chdir to ``InATTo_impl/`` while constructing FACE so its
embedding model and vocabulary loader find the right files.

InATTo_impl-specific dataset name -> human item word mapping (for the
"The <X> attracts those who can be described as..." prompt). This is
applied as a *post-hoc override* of FACE's internal prompt embedding
buffer, so ``FACE/encoder/FACE.py`` is not modified.
"""

from __future__ import annotations
import os
import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F


# Map our dataset names -> the natural-language item type used in FACE's
# prompt template. FACE original supports {amazon, yelp, steam}; we extend
# to the four datasets the InATTo paper evaluates on.
_DATASET_ITEM_WORD = {
    "beauty":  "cosmetic",
    "toys":    "toy",
    "sports":  "sport item",
    "yelp":    "restaurant",   # matches FACE original
    "amazon":  "book",         # FACE original
    "steam":   "game",         # FACE original
}


def _import_face_original(face_root: Path):
    """Import FACE/encoder/FACE.FACE while keeping FACE/ folder untouched.

    Adds both ``FACE/encoder`` and ``FACE/encoder/embedding_models`` to
    sys.path so that ``from FACE import FACE`` and the nested
    ``importlib.import_module('embedding_models.miniLM')`` inside
    ``FACE.py`` both resolve.
    """
    enc = face_root / "encoder"
    if not enc.exists():
        raise FileNotFoundError(f"FACE/encoder not found at {enc}")
    # Insert at the front so we don't shadow with stale entries.
    for p in (str(enc), str(enc / "embedding_models")):
        if p not in sys.path:
            sys.path.insert(0, p)
    # NB: `FACE/encoder/FACE.py` defines a module named `FACE` (no package);
    # importing it works only because sys.path includes `encoder/`.
    import FACE as face_module          # FACE.py
    return face_module.FACE             # the FACE class


class FACETokenizer(nn.Module):
    """FACE original VQ-RAF wrapped in the InATToTokenizer dict interface.

    Parameters
    ----------
    z_user, z_item : Tensor
        Frozen LightGCN CF embeddings, (n_users, d_cf) and (n_items, d_cf).
    h_txt_user, h_txt_item : Tensor
        MiniLM-encoded profile text (we keep them so signals match
        InATToTokenizer; FACE's own collaborative repr is recomputed below).
    rho_user, rho_item : Tensor
        Per-entity text density. FACE does not use rho, but we pass it
        through ``signals`` for downstream loggers.
    dataset_name : str
        One of {beauty, toys, sports, yelp, amazon, steam}.
    n_aspects : int
        FACE's ``word_num`` (default 8).
    d_aspect : int
        FACE's ``word_dim`` (default 256).
    llm_name : str
        Embedding model name FACE expects (default "miniLM").
    face_root : Path
        Path to the FACE/ folder (defaults to ../FACE relative to this file).
    """

    def __init__(
        self,
        z_user: torch.Tensor,
        z_item: torch.Tensor,
        h_txt_user: torch.Tensor,
        h_txt_item: torch.Tensor,
        rho_user: torch.Tensor,
        rho_item: torch.Tensor,
        dataset_name: str,
        n_aspects: int = 8,
        d_aspect: int = 256,
        llm_name: str = "miniLM",
        face_root: Path | str | None = None,
    ):
        super().__init__()
        self.dataset_name = str(dataset_name)
        self.n_aspects = int(n_aspects)
        self.d_aspect = int(d_aspect)
        self.L_max = 1                              # FACE: single VQ, no residual
        d_cf = int(z_user.shape[1])
        assert z_item.shape[1] == d_cf

        # Resolve and import FACE original.
        if face_root is None:
            face_root = Path(__file__).resolve().parents[2] / "FACE"
        face_root = Path(face_root)
        FaceOriginal = _import_face_original(face_root)

        # FACE's `Quantizer.__init__` reads `./LLMs/` and `./data/vocabulary/`
        # relative to cwd. We chdir to InATTo_impl/ for the duration of init.
        repo_impl = Path(__file__).resolve().parents[1]   # InATTo_impl/
        prev_cwd = os.getcwd()
        try:
            os.chdir(repo_impl)
            # Map our dataset to FACE's supported set for prompt safety,
            # then override prompt below.
            face_dataset_for_init = (
                self.dataset_name if self.dataset_name in {"amazon", "yelp", "steam"}
                else "amazon"
            )
            self.face = FaceOriginal(
                input_dim=d_cf,
                word_num=self.n_aspects,
                word_dim=self.d_aspect,
                dataset_name=face_dataset_for_init,
                llm_name=llm_name,
            )
            # Override the per-dataset item prompt without touching FACE.py.
            self._override_prompt_embedding()
        finally:
            os.chdir(prev_cwd)

        # FACE-internal codebook size for downstream sanity / vocab matching.
        self.V = int(self.face.quantizer.token_id.shape[0])
        self.vocabulary: list[str] = list(self.face.quantizer.vocabulary)

        # Buffers — frozen CF and profile embeddings, keyed by id.
        self.register_buffer("z_user_buf",  z_user.detach().float().clone())
        self.register_buffer("z_item_buf",  z_item.detach().float().clone())
        self.register_buffer("h_txt_user_buf", h_txt_user.detach().float().clone())
        self.register_buffer("h_txt_item_buf", h_txt_item.detach().float().clone())
        self.register_buffer("rho_user_buf",   rho_user.detach().float().clone())
        self.register_buffer("rho_item_buf",   rho_item.detach().float().clone())

    # ------------------------------------------------------------------
    def _override_prompt_embedding(self) -> None:
        """Replace FACE's prompt_embedding buffer with one built from
        the natural item word matching our dataset.

        FACE's `Quantizer.get_collaborative_representations` uses
        ``prompt_embedding`` (shape (3, n_prompt_tokens, d_token)) — the
        three rows are [user_prompt, item_prompt, item_prompt] (per FACE.py:50).
        We rebuild it with our item word.
        """
        item_word = _DATASET_ITEM_WORD.get(self.dataset_name)
        if item_word is None:
            # Already a FACE-supported name — leave FACE's prompt as-is.
            return
        prompt_usr = "The user and his likes can be described as the following words:"
        prompt_itm = (f"The {item_word} attracts those who can be "
                      "described as the following words:")
        emb_model = self.face.quantizer.embedding_model
        new_prompt = emb_model.get_text_token_embeddings(
            [prompt_usr, prompt_itm, prompt_itm]
        ).detach()
        # In-place replace the registered buffer.
        self.face.quantizer.prompt_embedding.data = new_prompt.to(
            self.face.quantizer.prompt_embedding.device
        )

    # ------------------------------------------------------------------
    @torch.no_grad()
    def refresh_z(self, mode: str, z_new: torch.Tensor) -> None:
        """Mirror of InATToTokenizer.refresh_z for API parity (unused for
        frozen LightGCN, kept for trainer compatibility)."""
        buf = self.z_user_buf if mode == "user" else self.z_item_buf
        buf.copy_(z_new.detach().to(buf.dtype).to(buf.device))

    # ------------------------------------------------------------------
    def _codeword_indices(self, z_q_reshape: torch.Tensor,
                          z_e_reshape: torch.Tensor) -> torch.Tensor:
        """Recover the chosen codeword indices from FACE's quantizer.

        FACE's `Quantizer.forward` returns the STE-attached z_q but not
        the indices. We mirror its argmin-without-replacement loop on
        the same mapped codebook so the indices match.
        """
        q = self.face.quantizer
        mapped = q.codebook_mapping(q.codebook_tensor_pca)              # (V, d)
        # Compute pairwise distances on z_e_reshape (not the STE one!).
        dist = (
            (z_e_reshape ** 2).sum(dim=1, keepdim=True)
            + (mapped ** 2).sum(dim=1)
            - 2 * (z_e_reshape @ mapped.t())
        )                                                                # (B*n, V)
        dist = dist.reshape(-1, self.n_aspects, dist.shape[-1])         # (B, n, V)
        B = dist.shape[0]
        batch_idx = torch.arange(B, device=dist.device).unsqueeze(1)
        chosen = None
        for k in range(self.n_aspects):
            k_idx = torch.argmin(dist[:, k, :], dim=1, keepdim=True)     # (B, 1)
            # Mask the chosen token from being picked again in subsequent aspects.
            dist[batch_idx, :, k_idx] = float("inf")
            chosen = k_idx if chosen is None else torch.cat([chosen, k_idx], dim=1)
        return chosen                                                    # (B, n)

    # ------------------------------------------------------------------
    def forward(self, ids: torch.Tensor, mode: str = "item",
                hard_mask: bool = False) -> dict:
        """Run FACE forward and reshape outputs to the InATToTokenizer dict.

        Returns
        -------
        dict with keys matching InATToTokenizer.forward (see its docstring):
            codes        (B, n_aspects, L_max=1)  long
            depth_mask   (B, n_aspects, 1)        all-ones (FACE: fixed depth)
            z_aspect     (B, n, d_aspect)
            z_hat        (B, n, d_aspect)         post-decoder reconstruction
            z_hat_st     (B, n, d_aspect)         STE-attached z_q (for align/T5)
            c_levels     (B, n, 1, d_aspect)      same as z_hat_st reshape
            signals      dict
            losses       dict {L_recon, L_Q, L_rate=0}
        """
        z_buf = self.z_user_buf if mode == "user" else self.z_item_buf
        h_txt_buf = self.h_txt_user_buf if mode == "user" else self.h_txt_item_buf
        rho_buf   = self.rho_user_buf   if mode == "user" else self.rho_item_buf

        z = z_buf[ids]                                                  # (B, d_cf)
        h_txt = h_txt_buf[ids]
        rho   = rho_buf[ids]

        # ---- FACE encoder up to VQ (manually re-run so we can recover indices) ----
        linear_out = self.face.linear_encoder(z)                        # (B, n, d)
        z_e = self.face.transformer_encoder(linear_out)                 # (B, n, d)
        z_e_reshape = z_e.reshape(-1, self.d_aspect)                    # (B*n, d)

        # Run the quantizer (this *also* runs argmin-without-replacement
        # internally and returns z_q_st + vq_loss).
        z_q_reshape, vq_loss = self.face.quantizer(z_e_reshape)         # (B*n, d)
        z_q = z_q_reshape.reshape(z.shape[0], self.n_aspects, self.d_aspect)

        # Recover the codeword indices selected. Using the same z_e (not z_q_st)
        # and a fresh distance computation: deterministic w.r.t. the same inputs.
        codes_idx = self._codeword_indices(z_q_reshape.detach(),
                                            z_e_reshape.detach())       # (B, n)

        # ---- FACE decoder (for L_recon, identical to FACE.forward) ----
        trans_dec = self.face.transformer_decoder(z_q)
        decoded = self.face.linear_encoder.reverse(trans_dec)           # (B, d_cf)
        L_recon = F.mse_loss(decoded, z.detach())

        # ---- Pack into the InATToTokenizer-style dict ----
        B = z.shape[0]
        codes = codes_idx.unsqueeze(-1).long()                          # (B, n, 1)
        mask  = torch.ones(B, self.n_aspects, self.L_max,
                            device=z.device, dtype=z.dtype)
        c_levels = z_q.unsqueeze(2)                                     # (B, n, 1, d)

        return {
            "codes":      codes,
            "depth_mask": mask,
            "z_aspect":   z_e,
            "z_hat":      z_q,                  # pre-decoder reconstruction (same shape as z_aspect)
            "z_hat_st":   z_q,                  # FACE returns the STE-attached one directly
            "c_levels":   c_levels,
            "signals": {
                "rho":      rho,
                "delta":    rho.new_zeros(rho.shape[0]),
                "delta_k":  rho.new_zeros(rho.shape[0], self.n_aspects),
                "phi":      rho.new_ones(rho.shape[0], self.n_aspects),
                "h_hat_txt": h_txt,
                "h_cf":     z,
                "r_ortho":  z.new_zeros(z.shape),
                "z":        z,
                "e":        linear_out,
            },
            "losses": {
                "L_recon": L_recon,
                "L_Q":     vq_loss,
                "L_rate":  z.new_zeros(()),     # FACE: no rate (no variable depth)
            },
        }

    # ------------------------------------------------------------------
    def collaborative_representations(self, z_q: torch.Tensor,
                                       kind: str) -> torch.Tensor:
        """Recreate FACE's ``get_collaborative_representations`` for a
        homogeneous batch (all-user or all-item).

        FACE original assumes the batch is laid out as
        ``[anc_users, pos_items, neg_items]`` and picks rows from a
        (3, n_prompt, d_token) prompt_embedding to match — see
        FACE.py:147. Our pipeline calls the user side and item side
        independently, so we select prompt row 0 (user) or row 1 (item)
        and broadcast it across the batch.

        Parameters
        ----------
        z_q : Tensor (B, n_aspects, d_aspect)
            STE-attached quantized codewords.
        kind : {'user', 'item'}

        Returns
        -------
        Tensor (B, d_llm)  L2-normalised collaborative representation.
        """
        if kind not in ("user", "item"):
            raise ValueError(f"kind must be 'user' or 'item', got {kind!r}")
        q = self.face.quantizer
        B = z_q.shape[0]

        z_q_flat = z_q.reshape(-1, self.d_aspect)
        words = q.reverse_codebook_mapping(z_q_flat)               # (B*n, d_token)
        words = words.reshape(B, self.n_aspects, -1)               # (B, n, d_token)

        # Interleave with the comma token between codewords (FACE.py:143-145).
        L = self.n_aspects * 2 - 1
        words_comma = z_q.new_zeros(B, L, words.shape[-1])
        words_comma[:, ::2, :] = words
        words_comma[:, 1::2, :] = q.prompt_comma.squeeze()

        # Select the per-kind prompt slice. FACE's prompt_embedding has shape
        # (3, n_prompt, d_token) corresponding to [user, item, item].
        idx = 0 if kind == "user" else 1
        prompt = q.prompt_embedding[idx:idx + 1]                   # (1, n_prompt, d_token)
        batch_prompt = prompt.expand(B, -1, -1)

        combined = torch.cat([batch_prompt, words_comma], dim=1)
        combined = q.embedding_model.add_special_tokens_for_embeddings(combined)
        return q.embedding_model.encode_embeddings(token_embeddings=combined)
