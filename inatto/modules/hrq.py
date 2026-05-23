"""Hierarchical Residual Quantization (HRQ) — IAHQ Phase 2A.

Replaces the flat single-codebook RQ with a tree-structured one. Each
level l uses its own *cluster-center* book obtained by Agglomerative
Hierarchical Clustering (AHC) on the MiniLM word embeddings:

    level 1: 32 coarse clusters   (e.g., "BODY/MEDICAL", "GEOGRAPHY")
    level 2: 128 mid clusters     (e.g., "FOOD")
    level 3: 512 fine clusters    (e.g., "CHRISTIAN RELIGION")
    level 4: 9338 leaves          = original codebook (full granularity)

At a given aspect:

    r₁ = z_aspect
    for l = 1..L:
        c_l = argmin_v  cosine_dist( r_l, W_l(center_l[v]) )
              (optionally restricted to children of c_{l-1} — parent path)
        r_{l+1} = r_l - W_l(center_l[c_l])
    z_hat = Σ_l m_l · W_l(center_l[c_l])

The level-1..3 centers are the AHC cluster means of the frozen MiniLM
embeddings (registered as buffers, never trained). Each level has its
own trainable linear projection W_l so the geometry can still adapt —
this matches the SimVQ pattern of the flat codebook.

The leaf level (l=4) reuses the original codebook (V=9338) so the final
identifier still bottoms out at single English words for interpretability.
"""

from __future__ import annotations
from pathlib import Path
import pickle
from collections import defaultdict
from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


def _level_means(raw: torch.Tensor, cluster_ids: torch.Tensor, K: int
                  ) -> torch.Tensor:
    """Mean of ``raw`` rows per cluster id (frozen prototype per cluster)."""
    out = torch.zeros(K, raw.size(-1), dtype=raw.dtype)
    cnt = torch.zeros(K, dtype=raw.dtype)
    out.index_add_(0, cluster_ids, raw)
    cnt.index_add_(0, cluster_ids, torch.ones_like(cluster_ids, dtype=raw.dtype))
    out = out / cnt.clamp_min(1.0).unsqueeze(-1)
    return out                                                 # (K, d_llm)


def _build_children(cluster_ids_per_level: list[torch.Tensor]
                     ) -> list[dict[int, torch.Tensor]]:
    """children[l-1][c_l] = LongTensor of valid level-(l+1) cluster ids."""
    children = []
    for l in range(len(cluster_ids_per_level) - 1):
        cur = cluster_ids_per_level[l].tolist()
        nxt = cluster_ids_per_level[l + 1].tolist()
        m = defaultdict(set)
        for c, n in zip(cur, nxt):
            m[int(c)].add(int(n))
        children.append({k: torch.tensor(sorted(v), dtype=torch.long)
                          for k, v in m.items()})
    return children


class HierarchicalRQ(nn.Module):
    """Tree-aware RQ.

    Parameters
    ----------
    raw : (V, d_llm)
        Frozen MiniLM word embeddings of the V codewords (same tensor
        the flat Codebook uses).
    tree_pkl : path
        Output of ``scripts/11_build_codebook_ahc_tree.py``.
    d_aspect : int
        Latent dimension Stage 1 uses (256 in InATTo's main config).
    L_max : int
        Number of levels actually quantized (must be ≤ len(tree levels)).
    parent_constraint : bool
        If True, level l candidates are restricted to children(c_{l-1}).
        If False, every level uses its full set of cluster centers.
    commit_beta : float
        Commit-loss weight (matches the flat RQ).
    """

    def __init__(self,
                  raw: torch.Tensor,
                  tree_pkl: Path,
                  d_aspect: int = 256,
                  L_max: int = 4,
                  parent_constraint: bool = False,
                  commit_beta: float = 0.25,
                  use_elcrec_proto: bool = False,
                  use_ctfidf_repr: bool = True,
                  ):
        super().__init__()
        with open(tree_pkl, "rb") as f:
            tree = pickle.load(f)
        assert raw.shape[0] == len(tree["tokens"])
        self.L_max = int(L_max)
        self.commit_beta = float(commit_beta)
        self.parent_constraint = bool(parent_constraint)
        # HRQ does not use VRVQ's full_codebook gimmick — depth budget comes
        # from the tree structure itself. Exposed at 0.0 so the tokenizer's
        # `self.rq.full_codebook_rate` access in the rate-loss branch works.
        self.full_codebook_rate = 0.0
        d_llm = raw.size(-1)

        cluster_ids_per_level = [torch.as_tensor(c, dtype=torch.long)
                                   for c in tree["cluster_ids"]]
        assert len(cluster_ids_per_level) >= self.L_max, (
            f"tree has only {len(cluster_ids_per_level)} levels, need {L_max}"
        )
        cluster_ids_per_level = cluster_ids_per_level[: self.L_max]

        # ---- Per-level frozen cluster centers (mean of MiniLM embeddings) ----
        self.level_K = []
        for l, cid in enumerate(cluster_ids_per_level):
            K = int(cid.max().item()) + 1
            self.level_K.append(K)
            self.register_buffer(f"center_raw_l{l}", _level_means(raw, cid, K))
            self.register_buffer(f"cluster_ids_l{l}", cid)

        # ---- Mode flags (★ must be set BEFORE init blocks below use them) ----
        self.use_elcrec_proto = bool(use_elcrec_proto)
        self.use_ctfidf_repr  = bool(use_ctfidf_repr)

        # ---- Level-aware vocabulary mapping (legacy global-offset codes) ----
        offsets = [0]
        for K in self.level_K:
            offsets.append(offsets[-1] + K)
        self.level_offset = offsets[:-1]
        self.V_global = offsets[-1]

        # ---- Single trainable projection (SimVQ pattern) ----
        # W_c is single (frozen vocab + one mapping), preserving SimVQ.
        # Multi-scale geometry across levels comes from ELCRec prototypes
        # (below) instead of per-level projections.
        self.W_c = nn.Linear(d_llm, d_aspect, bias=True)

        # ---- Representative leaf word per cluster ----
        # tree['repr_word']        — nearest-center (geometric)
        # tree['repr_word_ctfidf'] — ★ C-TF-IDF (semantic, GRAM-adapted)
        repr_key = "repr_word_ctfidf" if (self.use_ctfidf_repr and "repr_word_ctfidf" in tree) else "repr_word"
        if repr_key not in tree:
            raise RuntimeError(
                "AHC tree pickle is missing the requested representative — "
                "rebuild via scripts/11_build_codebook_ahc_tree.py."
            )
        for l, rw in enumerate(tree[repr_key][: self.L_max]):
            self.register_buffer(f"repr_word_l{l}",
                                  torch.as_tensor(rw, dtype=torch.long))

        # ---- ELCRec prototypes init (★ trainable per-level cluster centers) ----
        if self.use_elcrec_proto:
            protos = []
            with torch.no_grad():
                C = self.W_c(raw)                                # (V, d_aspect)
                C = C - C.mean(dim=0, keepdim=True)
                for l in range(self.L_max):
                    rw = getattr(self, f"repr_word_l{l}")        # (K_l,)
                    init = C[rw]                                  # (K_l, d_aspect)
                    protos.append(nn.Parameter(init.clone()))
            self.protos = nn.ParameterList(protos)
        else:
            self.protos = None

        # ---- Parent → children adjacency (optional path constraint) ----
        self._children = _build_children(cluster_ids_per_level)

    # ------------------------------------------------------------------
    def _level_centers(self, l: int) -> torch.Tensor:
        """Returns the (K_l, d_aspect) center bank for level l.

        - If ELCRec prototypes are enabled, the bank is the *trainable
          parameter* ``self.protos[l]`` — this is what makes hierarchical
          geometry adapt across levels (the SimVQ W_c stays single).
        - Otherwise the bank is the W_c-projected cluster mean of MiniLM
          embeddings (frozen-anchor SimVQ).
        """
        if self.use_elcrec_proto:
            return self.protos[l]                              # (K_l, d_aspect), trainable
        raw = getattr(self, f"center_raw_l{l}")
        return self.W_c(raw)                                   # (K_l, d_aspect), SimVQ-only

    # ------------------------------------------------------------------
    def _quantize_one_level(self,
                              r: torch.Tensor,                     # (B*n, d_aspect)
                              centers: torch.Tensor,               # (K, d_aspect)
                              candidate_mask: Optional[torch.Tensor] = None,
                              ) -> tuple[torch.Tensor, torch.Tensor]:
        """Returns (cluster_id, picked_center). If ``candidate_mask`` is
        provided (B, K) bool, only True entries are candidates."""
        r_n = F.normalize(r, dim=-1)                              # (BN, d)
        c_n = F.normalize(centers, dim=-1)                        # (K, d)
        sim = r_n @ c_n.t()                                       # (BN, K)
        if candidate_mask is not None:
            sim = sim.masked_fill(~candidate_mask, float("-inf"))
        idx = sim.argmax(dim=-1)                                  # (BN,)
        picked = centers[idx]                                     # (BN, d_aspect)
        return idx, picked

    # ------------------------------------------------------------------
    def forward(self, z_aspect: torch.Tensor, depth_mask: torch.Tensor
                ) -> dict:
        """
        z_aspect    : (B, n, d_aspect)
        depth_mask  : (B, n, L_max) {0,1} from Ep MLP (variable depth)
        returns dict with codes / c_levels / z_hat / z_hat_st / L_Q
        """
        B, n, d = z_aspect.shape
        L = self.L_max
        BN = B * n
        zf = z_aspect.reshape(BN, d)
        mask = depth_mask.reshape(BN, L)

        codes        = torch.zeros(BN, L, dtype=torch.long, device=zf.device)
        codes_global = torch.zeros(BN, L, dtype=torch.long, device=zf.device)
        c_levels     = torch.zeros(BN, L, d, device=zf.device)
        commit_l     = 0.0
        codebook_l   = 0.0

        # Parent indices at level 0 are "no parent" — all level-1 candidates.
        parent_idx = None
        r = zf
        for l in range(L):
            centers = self._level_centers(l)                       # (K_l, d_aspect)

            # Build candidate mask if parent_constraint
            cand_mask = None
            if self.parent_constraint and parent_idx is not None and l > 0:
                K_l = centers.size(0)
                cand_mask = torch.zeros(BN, K_l, dtype=torch.bool,
                                         device=zf.device)
                ch = self._children[l - 1]
                # vectorise on python side (BN is moderate)
                for b in range(BN):
                    p = int(parent_idx[b].item())
                    if p in ch:
                        cand_mask[b, ch[p].to(zf.device)] = True
                # Avoid empty rows: if no children listed, allow all.
                empty = ~cand_mask.any(dim=-1)
                if empty.any():
                    cand_mask[empty] = True

            idx, picked = self._quantize_one_level(r, centers, cand_mask)
            codes[:, l]        = idx                                  # local cluster id (0..K_l-1)
            codes_global[:, l] = idx + self.level_offset[l]           # disjoint vocab slot
            # Map cluster id → representative leaf word index (∈ [0, V-1])
            # so the cache and Stage-2 vocab stay flat.
            repr_w = getattr(self, f"repr_word_l{l}")
            codes[:, l]        = repr_w[idx]                          # ★ remap to flat V
            c_levels[:, l]     = picked

            # masked commit losses (per VRVQ)
            ml = mask[:, l]
            if ml.sum() > 0:
                # Codebook commit: pull center toward stop-grad(z_aspect)
                codebook_l = codebook_l + (
                    ((picked - zf.detach()) ** 2).sum(-1) * ml).mean()
                # Commit: pull z_aspect toward stop-grad(center)
                commit_l = commit_l + (
                    ((zf - picked.detach()) ** 2).sum(-1) * ml).mean()

            # subtract picked center from residual (STE-style; not detaching)
            r = r - picked
            parent_idx = idx

        # ---- Variable-depth aggregation (masked sum) ----
        z_hat = (c_levels * mask.unsqueeze(-1)).sum(dim=1)         # (BN, d)
        z_hat = z_hat.view(B, n, d)
        c_levels     = c_levels.view(B, n, L, d)
        codes        = codes.view(B, n, L)
        codes_global = codes_global.view(B, n, L)

        # STE attach (match flat RQ behaviour)
        z_hat_st = z_aspect + (z_hat - z_aspect).detach()

        L_Q = commit_l * self.commit_beta + codebook_l

        # ---- ELCRec prototype separation regularizer (★ push apart) ----
        # We only use this when prototypes are enabled. The pull-side
        # (L_proto) is already in L_Q's commit term, so we don't double
        # it here; L_sep just makes prototypes diverge within each level.
        if self.use_elcrec_proto:
            L_sep = 0.0
            for l in range(self.L_max):
                p = self.protos[l]
                # Mean pairwise cosine similarity (lower = better separation)
                pn = F.normalize(p, dim=-1)
                sim = pn @ pn.t()
                K = p.size(0)
                off_diag = sim - torch.eye(K, device=p.device)
                L_sep = L_sep + (off_diag.sum() / max(K * (K - 1), 1))
            L_sep = L_sep / self.L_max
        else:
            L_sep = z_aspect.new_zeros(())

        return {
            "codes":        codes,           # ← representative-mapped (∈ [0, V-1])
            "codes_global": codes_global,    # level-offset id (for analysis)
            "c_levels":     c_levels,
            "z_hat":        z_hat,
            "z_hat_st":     z_hat_st,
            "L_Q":          L_Q,
            "L_sep":        L_sep,
        }
