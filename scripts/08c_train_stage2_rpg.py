"""Stage 2 RPG-style — parallel multi-token prediction on variable-depth identifiers.

Replaces the T5 autoregressive head with the RPG (KDD'25) recipe:
- GPT-2 backbone (small).
- Item-level pooling: each item's ``n_aspects × L_max`` tokens are mean-pooled
  (with the variable-depth mask) into a single vector, then a sequence of
  these item vectors goes through GPT-2.
- N parallel prediction heads (one ResBlock per position) operating on the
  last-position hidden state — produce all ``n_aspects × L_max`` token logits
  in a single forward pass (no autoregressive decoding).
- Per-position cross-entropy with the variable-depth mask: inactive positions
  contribute no loss.
- Generation: score each item by the mean log-probability of its tokens at
  the active positions; top-K item-ranking, no beam search.

Why: the autoregressive Stage 2 was stuck at R@5 ≈ 0.002 on Toys because beam
search over 24-41-token identifiers pruned exponentially. RPG sidesteps this
exact failure mode (parallel decoding, no exact-match requirement).

Cache used: ``identifier_cache.<seed>.bpr.ssw01.pkl`` (or via --cache_suffix).
The cache stores T5-format token sequences with <EOA>/<EOI> separators; we
parse them back into an ``(n_items, n_aspects, L_max)`` grid plus a mask.

Usage:
    pixi run -- python scripts/08c_train_stage2_rpg.py --dataset toys --cuda 0 \\
        --cache_suffix bpr.ssw01
"""

from __future__ import annotations
import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import GPT2Config, GPT2Model

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.seq_loader import InATToSeqDataset


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ----------------------------------------------------------------------------
# Cache parser: T5 token sequence → (n_aspects, L_max) grid + mask
# ----------------------------------------------------------------------------

def parse_cache_to_grid(cache: dict, n_aspects: int, L_max: int,
                         side: str = "item",
                         ) -> tuple[torch.Tensor, torch.Tensor]:
    """For each entity (item or user — controlled by ``side``), parse its variable-length T5 token sequence into a
    ``(n_aspects, L_max)`` grid of codebook indices (1-indexed for the
    RPG-style vocab; 0 = pad) plus a ``(n_aspects, L_max)`` {0,1} mask.

    The cache uses these special ids:
        <EOA>  = end-of-aspect separator (between aspects)
        <EOI>  = end-of-item terminator
    Codewords map to T5 vocab ids via cache['cfg']['code_to_t5'][i] = T5_id,
    so the inverse map t5_id → codebook index (0..V-1) lets us recover the
    grid.
    """
    if side not in ("item", "user"):
        raise ValueError(f"side must be 'item' or 'user', got {side!r}")
    eoa = int(cache["cfg"]["special_ids"]["<EOA>"])
    eoi = int(cache["cfg"]["special_ids"]["<EOI>"])
    t5_to_code: dict[int, int] = {int(t5): i for i, t5 in enumerate(cache["cfg"]["code_to_t5"])}
    n_entities = max(cache[side].keys()) + 1     # iids/uids start at 0
    grid = torch.zeros(n_entities, n_aspects, L_max, dtype=torch.long)
    mask = torch.zeros(n_entities, n_aspects, L_max, dtype=torch.bool)

    for iid, seq in cache[side].items():
        aspect_lists: list[list[int]] = [[] for _ in range(n_aspects)]
        cur: list[int] = []
        ai = 0
        for tok in seq:
            if tok == eoa:
                if ai < n_aspects:
                    aspect_lists[ai] = cur
                cur = []; ai += 1
            elif tok == eoi:
                break
            else:
                cidx = t5_to_code.get(int(tok), -1)
                if cidx >= 0:
                    cur.append(cidx)
        # If the cache didn't end with EOA before EOI, ``cur`` holds the
        # last aspect — fold it in.
        if cur and ai < n_aspects:
            aspect_lists[ai] = cur

        for k, asp in enumerate(aspect_lists):
            for l in range(min(len(asp), L_max)):
                grid[iid, k, l] = asp[l] + 1   # 1-indexed (0 = pad)
                mask[iid, k, l] = True
    return grid, mask


# ----------------------------------------------------------------------------
# Datasets — recycle InATToSeqDataset; collate emits (user_id, target_id,
# history_ids).  Only ``history_ids`` and ``target_id`` matter for RPG.
# ----------------------------------------------------------------------------

class HistoryItemDataset(torch.utils.data.Dataset):
    """Per-sample dict {'input_ids', 'attention_mask', 'label'}."""

    def __init__(self, base: InATToSeqDataset, history_max_len: int):
        self.base = base
        self.history_max_len = int(history_max_len)

    def __len__(self): return len(self.base)

    def __getitem__(self, idx):
        uid, tgt, hist = self.base[idx]
        hist = [int(h) for h in hist][-self.history_max_len :]
        return {"input_ids": hist, "label": int(tgt), "user_id": int(uid)}


def make_collate(history_max_len: int, pad_id: int = 0):
    def collate(batch):
        B = len(batch)
        T = history_max_len
        input_ids = torch.full((B, T), pad_id, dtype=torch.long)
        attn      = torch.zeros((B, T), dtype=torch.long)
        seq_lens  = torch.zeros(B, dtype=torch.long)
        labels    = torch.zeros(B, dtype=torch.long)
        user_ids  = torch.zeros(B, dtype=torch.long)
        for i, b in enumerate(batch):
            L = min(len(b["input_ids"]), T)
            input_ids[i, T-L:] = torch.tensor(b["input_ids"][-L:], dtype=torch.long)
            attn[i, T-L:] = 1
            seq_lens[i] = L
            labels[i] = b["label"]
            user_ids[i] = b.get("user_id", 0)
        return {"input_ids": input_ids, "attention_mask": attn,
                "seq_lens": seq_lens, "labels": labels, "user_ids": user_ids}
    return collate


# ----------------------------------------------------------------------------
# Model
# ----------------------------------------------------------------------------

class ResBlock(nn.Module):
    """Identity-init residual block (RPG paper). Linear + SiLU + skip."""
    def __init__(self, hidden_size: int):
        super().__init__()
        self.linear = nn.Linear(hidden_size, hidden_size)
        nn.init.zeros_(self.linear.weight)
        self.act = nn.SiLU()
    def forward(self, x):
        return x + self.act(self.linear(x))


class InATToRPG(nn.Module):
    """RPG (KDD'25) backbone adapted to InATTo's variable-depth identifier.

    Differences from the original RPG:
    - ``item_id2tokens`` is variable-depth → ``item_id2mask`` weighs the
      item-level pooling and the per-position loss.
    - Single shared codebook (we use a global GPT-2 vocab for all positions
      rather than per-digit slices, matching InATTo's design philosophy).
    """

    def __init__(self,
                 item_id2tokens: torch.Tensor,    # (n_items, n_aspects, L_max), 0=pad
                 item_id2mask:   torch.Tensor,    # (n_items, n_aspects, L_max), bool
                 n_aspects: int, L_max: int, V: int,
                 n_embd: int = 256, n_layer: int = 2, n_head: int = 4,
                 n_inner: int = 1024, dropout: float = 0.5,
                 max_seq_len: int = 32, temperature: float = 0.05,
                 depth_weighted_pool: bool = False,
                 n_intents: int = 0,
                 intent_fuse: str = "add",       # "add" | "concat" | "off"
                 intent_temperature: float = 0.1,
                 user_id2tokens: torch.Tensor | None = None,  # (n_users, P)
                 user_id2mask:   torch.Tensor | None = None,
                 user_fuse: str = "off",            # "prepend" | "add" | "off"
                 ):
        super().__init__()
        self.n_aspects = int(n_aspects)
        self.L_max = int(L_max)
        self.V = int(V)
        self.n_total_pos = self.n_aspects * self.L_max
        self.temperature = float(temperature)
        self.depth_weighted_pool = bool(depth_weighted_pool)

        # ---- Intent anchors (Level-1 ELCRec-style; SSW prior planned for L2) ----
        # k learnable unit-sphere intent centers. User context (the GPT-2
        # last hidden state, used as the user representation) attends to
        # them via cosine softmax; the resulting intent embedding is fused
        # back into the user vector before the 32 per-position prediction
        # heads. With n_intents == 0 the path is bypassed entirely.
        self.n_intents = int(n_intents)
        self.intent_fuse = str(intent_fuse) if self.n_intents > 0 else "off"
        self.intent_temperature = float(intent_temperature)
        if self.n_intents > 0:
            init = torch.randn(self.n_intents, n_embd)
            init = init / init.norm(dim=-1, keepdim=True).clamp_min(1e-6)
            self.intent_centers = nn.Parameter(init)
            if self.intent_fuse == "concat":
                self.intent_proj = nn.Linear(2 * n_embd, n_embd)
            else:
                self.intent_proj = None
        else:
            self.intent_centers = None
            self.intent_proj    = None

        # ---- User codeword (Stage-1) buffers + fuse mode ----------------
        # If provided, the user's static identifier (a codeword sequence of
        # the same shape as an item) is mean-pooled into a single d-vector
        # and either prepended to the history sequence ("prepend", U1) or
        # additively fused into the user representation after the GPT-2
        # last hidden state ("add", U2).
        self.user_fuse = str(user_fuse) if user_id2tokens is not None else "off"
        if user_id2tokens is not None:
            assert user_id2mask is not None
            n_users = user_id2tokens.shape[0]
            self.register_buffer("user_id2tokens",
                                  user_id2tokens.reshape(n_users, self.n_total_pos))
            self.register_buffer("user_id2mask",
                                  user_id2mask.reshape(n_users, self.n_total_pos).bool())

        n_items = item_id2tokens.shape[0]
        # Flatten the (n_aspects, L_max) grid into a 1-D position axis.
        self.register_buffer("item_id2tokens",
                              item_id2tokens.reshape(n_items, self.n_total_pos))
        self.register_buffer("item_id2mask",
                              item_id2mask.reshape(n_items, self.n_total_pos).bool())

        # GPT-2 vocab:  0 = pad, 1..V = codewords, V+1 = eos
        vocab_size = self.V + 2
        cfg = GPT2Config(
            vocab_size=vocab_size, n_positions=max(64, max_seq_len + 2),
            n_embd=n_embd, n_layer=n_layer, n_head=n_head, n_inner=n_inner,
            activation_function="gelu_new",
            resid_pdrop=0.0, embd_pdrop=dropout, attn_pdrop=dropout,
            layer_norm_epsilon=1e-12,
        )
        self.gpt2 = GPT2Model(cfg)
        self.pred_heads = nn.ModuleList([ResBlock(n_embd) for _ in range(self.n_total_pos)])
        self.loss_fct = nn.CrossEntropyLoss(reduction="none")

    # ------------------------------------------------------------------
    def _item_pool(self, item_ids: torch.Tensor) -> torch.Tensor:
        """item_ids: (B, S) item indices.  Returns (B, S, d) item-level pooled.

        Two modes:

        - uniform mean (default, ``depth_weighted_pool=False``): RPG's
          recipe — average the token embeddings of every active position
          equally.  Variable-depth information is only used to *mask*
          the average.

        - depth-weighted aspect pooling (``depth_weighted_pool=True``,
          our extension): aggregate hierarchically and weight each
          aspect by how deep it is, so an "info-rich" aspect (more
          residual codewords) contributes more to the item vector
          than an "info-poor" aspect (single codeword). This puts the
          variable-depth signal directly into the input representation
          rather than only into the loss mask.
        """
        tokens = self.item_id2tokens[item_ids]           # (B, S, n_total_pos)
        mask   = self.item_id2mask[item_ids].float()     # (B, S, n_total_pos)
        embs   = self.gpt2.wte(tokens)                    # (B, S, n_total_pos, d)

        if not self.depth_weighted_pool:
            denom = mask.sum(dim=-1, keepdim=True).clamp_min(1.0)
            return (embs * mask.unsqueeze(-1)).sum(dim=-2) / denom

        # Depth-weighted aspect pooling.
        B, S = tokens.shape[:2]
        n, L, d = self.n_aspects, self.L_max, embs.size(-1)
        embs_g = embs.view(B, S, n, L, d)
        mask_g = mask.view(B, S, n, L)

        # 1. Per-aspect mean over its *active* residual codewords.
        denom_a = mask_g.sum(dim=-1, keepdim=True).clamp_min(1.0)              # (B, S, n, 1)
        aspect_emb = (embs_g * mask_g.unsqueeze(-1)).sum(dim=-2) / denom_a     # (B, S, n, d)

        # 2. Aspect-level weights from the depth (#active codewords per aspect).
        #    Active aspects: those with depth >= 1; inactive ones get weight 0.
        depth = mask_g.sum(dim=-1)                                              # (B, S, n)
        total = depth.sum(dim=-1, keepdim=True).clamp_min(1.0)                  # (B, S, 1)
        aspect_w = depth / total                                                # (B, S, n)
        return (aspect_emb * aspect_w.unsqueeze(-1)).sum(dim=-2)                # (B, S, d)

    def _user_pool(self, user_ids: torch.Tensor) -> torch.Tensor:
        """Pool the Stage-1 user codeword sequence into a single d-vector.
        Same mean-over-active-positions recipe as :meth:`_item_pool` (uniform mode).
        """
        if self.user_fuse == "off":
            return None
        tokens = self.user_id2tokens[user_ids]            # (B, P)
        mask   = self.user_id2mask[user_ids].float()      # (B, P)
        embs   = self.gpt2.wte(tokens)                    # (B, P, d)
        denom  = mask.sum(-1, keepdim=True).clamp_min(1.0)
        return (embs * mask.unsqueeze(-1)).sum(-2) / denom  # (B, d)

    def _intent_fuse(self, last: torch.Tensor) -> torch.Tensor:
        """ELCRec-style intent anchoring (Level 1).

        Treats the GPT-2 last hidden state ``last`` as a user representation,
        attends it over the learnable unit-sphere intent centers, and fuses
        the resulting intent embedding back. No clustering loss yet (Level
        2 will add the MvMF / SSW prior).
        """
        if self.n_intents == 0:
            return last
        last_n   = F.normalize(last, dim=-1)
        center_n = F.normalize(self.intent_centers, dim=-1)
        attn     = F.softmax(last_n @ center_n.t() / self.intent_temperature, dim=-1)  # (B, k)
        intent_emb = attn @ self.intent_centers                                          # (B, d)
        if self.intent_fuse == "add":
            return last + intent_emb
        if self.intent_fuse == "concat":
            return self.intent_proj(torch.cat([last, intent_emb], dim=-1))
        return last

    def _last_state(self, batch: dict) -> torch.Tensor:
        input_embs = self._item_pool(batch["input_ids"])  # (B, S, d)
        attn_mask  = batch["attention_mask"]

        # ─ User textual-identifier prepend variants ─
        #   "prepend"       → single mean-pooled user vector at the front
        #   "prepend_multi" → all user codeword tokens (variable, masked) at the front
        #                     (multi-token textual user identifier prefix)
        #   "add"           → fused after GPT-2 last hidden state
        user_emb_static = None
        if self.user_fuse != "off":
            if self.user_fuse == "prepend_multi":
                u_tokens = self.user_id2tokens[batch["user_ids"]]                   # (B, U_P)
                u_mask   = self.user_id2mask  [batch["user_ids"]].long()            # (B, U_P)
                u_embs   = self.gpt2.wte(u_tokens)                                  # (B, U_P, d)
                input_embs = torch.cat([u_embs, input_embs], dim=1)                 # (B, U_P+S, d)
                attn_mask  = torch.cat([u_mask, attn_mask], dim=1)                  # (B, U_P+S)
            else:
                user_emb_static = self._user_pool(batch["user_ids"])  # (B, d)
                if self.user_fuse == "prepend":
                    input_embs = torch.cat([user_emb_static.unsqueeze(1), input_embs], dim=1)
                    attn_mask  = torch.cat([torch.ones_like(attn_mask[:, :1]), attn_mask], dim=1)

        out = self.gpt2(inputs_embeds=input_embs, attention_mask=attn_mask)
        seq_lens = attn_mask.sum(dim=-1) - 1                    # adjusted for prepend
        idx = seq_lens.clamp_min(0).view(-1, 1, 1).expand(-1, 1, out.last_hidden_state.size(-1))
        last = out.last_hidden_state.gather(1, idx).squeeze(1)  # (B, d)

        # ─ U2 add: static user vector additively fused (separate from prepend) ─
        if self.user_fuse == "add" and user_emb_static is not None:
            last = last + user_emb_static

        return self._intent_fuse(last)                          # Level-1 intent anchor

    def forward(self, batch: dict, return_loss: bool = True):
        last = self._last_state(batch)                              # (B, d)
        states = torch.stack([h(last) for h in self.pred_heads], dim=1)  # (B, P, d)
        states = F.normalize(states, dim=-1)
        token_emb = F.normalize(self.gpt2.wte.weight, dim=-1)        # (vocab, d)
        logits = torch.matmul(states, token_emb.t()) / self.temperature  # (B, P, vocab)

        if not return_loss:
            return logits                                            # (B, P, vocab)

        target_tokens = self.item_id2tokens[batch["labels"]]         # (B, P)
        target_mask   = self.item_id2mask[batch["labels"]].float()   # (B, P)
        # CE per position (over the whole GPT-2 vocab so logits[:, i, 0]
        # would mean "pad" — fine because target_tokens is in 1..V at
        # active positions and the loss is masked at inactive ones).
        logits_flat  = logits.reshape(-1, logits.size(-1))            # (B*P, vocab)
        targets_flat = target_tokens.reshape(-1)                       # (B*P,)
        loss_per_pos = self.loss_fct(logits_flat, targets_flat)        # (B*P,)
        loss = (loss_per_pos.view_as(target_mask) * target_mask).sum() / target_mask.sum().clamp_min(1.0)
        return loss, logits

    # ------------------------------------------------------------------
    @torch.no_grad()
    def rank_all_items(self, batch: dict, item_chunk: int = 2048) -> torch.Tensor:
        """Return (B, n_items) per-item score = mean over active positions of
        log-prob assigned to the item's token at that position."""
        logits = self.forward(batch, return_loss=False)            # (B, P, vocab)
        log_p  = F.log_softmax(logits, dim=-1)                     # (B, P, vocab)
        B, P, _ = log_p.shape

        n_items = self.item_id2tokens.size(0)
        scores  = torch.empty(B, n_items, device=logits.device)
        for s in range(0, n_items, item_chunk):
            e = min(s + item_chunk, n_items)
            tokens_c = self.item_id2tokens[s:e]                    # (C, P)
            mask_c   = self.item_id2mask[s:e].float()              # (C, P)
            # log_p[b, p, tokens_c[c, p]] -> (B, C, P)
            idx = tokens_c.unsqueeze(0).expand(B, -1, -1)          # (B, C, P)
            log_p_expand = log_p.unsqueeze(1).expand(-1, e - s, -1, -1)  # (B, C, P, vocab)
            gathered = log_p_expand.gather(dim=-1, index=idx.unsqueeze(-1)).squeeze(-1)  # (B, C, P)
            mask_b = mask_c.unsqueeze(0).expand(B, -1, -1)         # (B, C, P)
            scores[:, s:e] = (gathered * mask_b).sum(dim=-1) / mask_b.sum(dim=-1).clamp_min(1.0)
        return scores


# ----------------------------------------------------------------------------
# Eval
# ----------------------------------------------------------------------------

@torch.no_grad()
def eval_recall_ndcg(model: InATToRPG, loader: DataLoader, device: torch.device,
                       Ks: tuple = (5, 10, 20), max_samples: int | None = None
                       ) -> dict:
    model.eval()
    n = 0
    hits = {K: 0.0 for K in Ks}
    ndcgs = {K: 0.0 for K in Ks}
    Kmax = max(Ks)
    for batch in loader:
        batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
        scores = model.rank_all_items(batch)                    # (B, n_items)
        # Mask the target's own self-rank? Not for full ranking — but we
        # do mask history items (TIGER/LETTER convention: avoid suggesting
        # already-seen items).
        for b in range(batch["input_ids"].size(0)):
            seen = batch["input_ids"][b][batch["attention_mask"][b].bool()]
            scores[b, seen] = -float("inf")
            scores[b, 0]    = -float("inf")          # pad item never recommended
        topk = scores.topk(Kmax, dim=-1).indices                # (B, Kmax)
        labels = batch["labels"]                                # (B,)
        for K in Ks:
            top_K = topk[:, :K]
            hit_mask = (top_K == labels.unsqueeze(-1)).any(dim=-1)
            hits[K] += hit_mask.sum().item()
            # NDCG@K
            pos = (top_K == labels.unsqueeze(-1)).float().argmax(dim=-1)
            ndcg = hit_mask.float() / torch.log2(pos.float() + 2.0)
            ndcgs[K] += ndcg.sum().item()
        n += batch["input_ids"].size(0)
        if max_samples is not None and n >= max_samples: break
    return {**{f"R@{K}": hits[K] / max(1, n) for K in Ks},
             **{f"N@{K}": ndcgs[K] / max(1, n) for K in Ks}, "n": n}


# ----------------------------------------------------------------------------
# Main
# ----------------------------------------------------------------------------

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=0)
    p.add_argument("--data_root",   type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--ckpt_dir",    type=Path, default=Path(__file__).resolve().parent.parent / "checkpoints/inatto")
    p.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--cache_suffix", type=str, default="bpr.ssw01",
                    help="identifier_cache.<seed>.<suffix>.pkl")
    p.add_argument("--output_tag", type=str, default=None,
                    help="Extra suffix on ckpt/log filenames. Use when running "
                         "multiple ablations off the SAME cache (so they don't "
                         "overwrite each other).")
    # ---- backbone ----
    p.add_argument("--n_embd", type=int, default=256)
    p.add_argument("--n_layer", type=int, default=2)
    p.add_argument("--n_head", type=int, default=4)
    p.add_argument("--n_inner", type=int, default=1024)
    p.add_argument("--dropout", type=float, default=0.5)
    p.add_argument("--temperature", type=float, default=0.05)
    p.add_argument("--history_max_len", type=int, default=50,
                    help="LETTER/RPG convention is 50.")
    p.add_argument("--depth_weighted_pool", action="store_true",
                    help="Use depth-weighted aspect pooling for input embedding "
                         "(ours; default RPG uses uniform mean).")
    # ---- Intent anchoring (ELCRec-style; Level-1 only — no clustering loss yet) ----
    p.add_argument("--n_intents", type=int, default=0,
                    help="# of learnable unit-sphere intent centers. 0 disables.")
    p.add_argument("--intent_fuse", type=str, default="add", choices=["add", "concat", "off"],
                    help="How to fuse the soft intent embedding into the user vector.")
    p.add_argument("--intent_temperature", type=float, default=0.1,
                    help="Softmax temperature over user · intent cosines.")
    # ---- User codeword reuse (Stage-1 user identifier, U1 prepend / U2 add) ----
    p.add_argument("--user_fuse", type=str, default="off",
                    choices=["prepend", "prepend_multi", "add", "off"],
                    help="Reuse the Stage-1 user codeword sequence: 'prepend' "
                         "puts the user-pooled vector as the first token of "
                         "the history sequence; 'prepend_multi' puts the full "
                         "variable-length user codeword sequence (Align3GR-style) "
                         "as a multi-token prefix; 'add' adds it to the last "
                         "hidden state after GPT-2.")
    # ---- training ----
    p.add_argument("--batch_size",  type=int, default=256)
    p.add_argument("--eval_batch_size", type=int, default=64)
    p.add_argument("--lr",          type=float, default=3e-3,    help="RPG default 3e-3.")
    p.add_argument("--weight_decay", type=float, default=0.0)
    p.add_argument("--warmup_ratio", type=float, default=0.05)
    p.add_argument("--total_epochs", type=int, default=150,
                    help="RPG default 150 epochs (early stop expected).")
    p.add_argument("--patience",    type=int, default=20,
                    help="Stop after N evaluations without R@5 improvement.")
    p.add_argument("--eval_every",  type=int, default=1)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_grad_norm", type=float, default=1.0)
    p.add_argument("--resume", action="store_true")
    args = p.parse_args()

    _set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    tag = f".{args.cache_suffix}" if args.cache_suffix else ""
    if args.output_tag:
        tag = f"{tag}.{args.output_tag}"
    print(f"[args] {vars(args)}")
    print(f"[device] {device}")

    # ---- Load cache + build grid ----
    ddir = args.data_root / args.dataset
    cache_path = ddir / f"identifier_cache.{args.seed}.{args.cache_suffix}.pkl"
    print(f"[cache] {cache_path}")
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    n_aspects = int(cache["cfg"]["n_aspects"])
    L_max     = int(cache["cfg"]["L_max"])
    V         = int(cache["cfg"]["V"])
    grid, mask = parse_cache_to_grid(cache, n_aspects, L_max)
    n_items = grid.shape[0]
    print(f"[grid] n_items={n_items}  n_aspects={n_aspects}  L_max={L_max}  V={V}  "
          f"active positions per item: min={mask.sum(dim=(1,2)).min().item()} "
          f"max={mask.sum(dim=(1,2)).max().item()} "
          f"mean={mask.sum(dim=(1,2)).float().mean().item():.1f}")

    # ---- Datasets ----
    train_base = InATToSeqDataset(args.data_root, args.dataset, "train", history_max_len=args.history_max_len)
    val_base   = InATToSeqDataset(args.data_root, args.dataset, "val",   history_max_len=args.history_max_len)
    train_ds = HistoryItemDataset(train_base, args.history_max_len)
    val_ds   = HistoryItemDataset(val_base,   args.history_max_len)
    coll = make_collate(args.history_max_len)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=coll, pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.eval_batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=coll, pin_memory=True)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"train_batch={args.batch_size}  eval_batch={args.eval_batch_size}")

    # ---- Model ----
    # Optional Stage-1 user codeword reuse
    u_grid, u_mask = None, None
    if args.user_fuse != "off":
        u_grid, u_mask = parse_cache_to_grid(cache, n_aspects, L_max, side="user")
        print(f"[user codeword] mode={args.user_fuse}  grid={tuple(u_grid.shape)}")

    model = InATToRPG(item_id2tokens=grid, item_id2mask=mask,
                       n_aspects=n_aspects, L_max=L_max, V=V,
                       n_embd=args.n_embd, n_layer=args.n_layer, n_head=args.n_head,
                       n_inner=args.n_inner, dropout=args.dropout,
                       max_seq_len=args.history_max_len, temperature=args.temperature,
                       depth_weighted_pool=args.depth_weighted_pool,
                       n_intents=args.n_intents, intent_fuse=args.intent_fuse,
                       intent_temperature=args.intent_temperature,
                       user_id2tokens=u_grid, user_id2mask=u_mask,
                       user_fuse=args.user_fuse).to(device)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] params={n_params:,}  vocab={V+2}  n_total_pos={n_aspects*L_max}")

    # ---- Optimizer + scheduler ----
    optim = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    total_steps = args.total_epochs * len(train_loader)
    warmup_steps = max(1, int(args.warmup_ratio * total_steps))
    def _lr_lambda(step):
        if step < warmup_steps: return step / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    print(f"[sched] cosine warmup={warmup_steps}/{total_steps}")

    # ---- Resume ----
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    args.results_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage2rpg{tag}.latest.pth"
    best_ckpt   = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage2rpg{tag}.best.pth"
    log_path    = args.results_dir / f"inatto-{args.dataset}-{args.seed}.stage2rpg{tag}.log.json"
    history: list[dict] = []
    start_epoch = 0
    best_R5 = -1.0
    patience_counter = 0
    if args.resume and latest_ckpt.exists():
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        model.load_state_dict(ck["model_state"])
        optim.load_state_dict(ck["optim_state"])
        history = ck.get("history", [])
        start_epoch = ck.get("next_epoch", len(history))
        best_R5 = ck.get("best_R5", -1.0)
        patience_counter = ck.get("patience_counter", 0)
        print(f"[resume] from epoch {start_epoch}, best_R5={best_R5:.4f}")

    # ---- Train loop ----
    for epoch in range(start_epoch, args.total_epochs):
        t0 = time.time()
        model.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"ep {epoch}", leave=False):
            batch = {k: v.to(device, non_blocking=True) for k, v in batch.items()}
            loss, _ = model(batch, return_loss=True)
            optim.zero_grad()
            loss.backward()
            if args.max_grad_norm:
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
            optim.step()
            scheduler.step()
            losses.append(loss.item())
        train_loss = float(np.mean(losses))

        log: dict = {"epoch": epoch, "epoch_time": time.time() - t0,
                     "train_loss": train_loss}

        # Eval R@5 / N@5 / R@10 / N@10 / R@20 / N@20 every eval_every epochs.
        do_eval = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.total_epochs)
        if do_eval:
            t1 = time.time()
            m = eval_recall_ndcg(model, val_loader, device, Ks=(5, 10, 20))
            log.update(m); log["eval_time"] = time.time() - t1
            R5 = m["R@5"]
            improved = R5 > best_R5
            if improved:
                best_R5 = R5
                patience_counter = 0
                torch.save({"model_state": model.state_dict(),
                             "epoch": epoch, "best_R5": best_R5,
                             "cache_path": str(cache_path)}, best_ckpt)
            else:
                patience_counter += 1
            log["best_R5"] = best_R5
            log["patience"] = f"{patience_counter}/{args.patience}"
            star = " ★" if improved else ""
            print(f"[ep {epoch:>3}/{args.total_epochs}] L={train_loss:.4f}  "
                  f"R@5={m['R@5']:.4f}  R@10={m['R@10']:.4f}  R@20={m['R@20']:.4f}  "
                  f"N@5={m['N@5']:.4f}  N@10={m['N@10']:.4f}  "
                  f"(best R@5={best_R5:.4f}{star}, patience {patience_counter}/{args.patience}, "
                  f"{log['epoch_time']:.1f}s + {log['eval_time']:.1f}s)")
        else:
            print(f"[ep {epoch:>3}/{args.total_epochs}] L={train_loss:.4f}  ({log['epoch_time']:.1f}s)")

        history.append(log)
        Path(log_path).write_text(json.dumps(history, indent=2))

        torch.save({"model_state": model.state_dict(),
                     "optim_state": optim.state_dict(),
                     "history": history,
                     "next_epoch": epoch + 1,
                     "best_R5": best_R5,
                     "patience_counter": patience_counter,
                     "cache_path": str(cache_path)}, latest_ckpt)

        if do_eval and patience_counter >= args.patience:
            print(f"[early stop] best R@5 = {best_R5:.4f}  at epoch {epoch}")
            break

    print(f"[done] best R@5 = {best_R5:.4f}  → {best_ckpt}")


if __name__ == "__main__":
    main()
