"""Tree-guide regularizer — LETTER-style diversity reg upgraded with our
CF-supervised hierarchical codebook clusters.

Idea:
  - SSW (baseline) regularises the codebook toward a *uniform* spherical
    distribution. It only prevents collapse — there is no semantics in
    where each codeword lands.
  - Tree-guide replaces / complements SSW with a *cluster-aware*
    pull-push loss: each codeword is pulled toward its cluster's center,
    and cluster centers are pushed apart from each other.
  - The cluster structure comes from a pre-computed AHC tree
    (preferably CF-supervised via hybrid MiniLM + LightGCN embedding).

Loss:
    L_tree = L_pull + L_push
    L_pull = -mean(cos_sim(codeword, its_cluster_mean))      # same-cluster pull
    L_push = +mean(cos_sim(cluster_mean_i, cluster_mean_j))  # i ≠ j (push)

Implementation note (efficiency):
  Computing pairwise similarities on (V=9338, V=9338) is heavy. We only
  materialise (K, K) where K = level-1 cluster count (default 32), so
  the cost is negligible per step.

This is the operational version of "tree-guide" reg discussed in the
session: LETTER's collaborative + diversity registers fused with a
CF-supervised semantic hierarchy, applied as a soft regulariser
*alongside* the flat RQ baseline (which already produces our best R@5).
"""

from __future__ import annotations

import torch
import torch.nn.functional as F


def tree_pull_push_loss(codebook_emb: torch.Tensor,
                          cluster_ids,
                          temperature: float = 0.5,
                          level_weights: list[float] | None = None,
                          ) -> torch.Tensor:
    """Tree-guide regulariser, **LETTER-style contrastive** variant.

    Earlier (V0) we used (L_pull + L_push) with direct cosine objectives:
    every codeword was *uniformly* dragged to its cluster mean, which
    collapsed codebook diversity (Phase 2I: 9338 → 4070 unique codewords
    after Stage-1, baseline keeps 6746). The pull was raw and applied
    equally to *all* V codewords.

    LETTER's diversity loss instead uses a contrastive (softmax) form:
    each codeword's similarity with its own cluster's representative is
    pulled up *relative to* its similarity with other cluster
    representatives. The softmax denominator self-normalises the pull
    strength — codewords already near their own center receive little
    gradient, while those far away receive more. Diversity stays intact
    because the only "winner" each codeword needs is its own cluster,
    not its cluster's mean specifically.

    L_tree = CE(sim(codeword, all_cluster_means) / temperature,
                target = own cluster id)

    codebook_emb  : (V, d_aspect)  — trainable W_c(codebook_raw)
    cluster_ids   : either
                      - single (V,) long  → legacy single-level
                      - list/tuple of (V,) long, one per level → multi
    temperature   : softmax temperature
    level_weights : per-level scalar weights (defaults to [1, 0.5, 0.25]
                    truncated/extended to match #levels). Used only when
                    cluster_ids is a list.

    Multi-level: sums contrastive CE losses across levels, encouraging
    the codebook to organise itself coarse-to-fine simultaneously.
    """
    device = codebook_emb.device
    d = codebook_emb.size(-1)

    # Wrap single-level into a list so we can branch uniformly below.
    if isinstance(cluster_ids, torch.Tensor):
        levels = [cluster_ids]
    else:
        levels = list(cluster_ids)

    if level_weights is None:
        defaults = [1.0, 0.5, 0.25, 0.125]
        level_weights = defaults[: len(levels)]
    assert len(level_weights) == len(levels), \
        f"level_weights len {len(level_weights)} != #levels {len(levels)}"

    total = codebook_emb.new_zeros(())
    weight_sum = sum(level_weights)
    for w_l, cid in zip(level_weights, levels):
        K = int(cid.max().item()) + 1
        # Cluster means via scatter (V → K, very cheap).
        means = torch.zeros(K, d, device=device, dtype=codebook_emb.dtype)
        counts = torch.zeros(K, device=device, dtype=codebook_emb.dtype)
        means.index_add_(0, cid, codebook_emb)
        ones = torch.ones(codebook_emb.size(0), device=device,
                            dtype=codebook_emb.dtype)
        counts.index_add_(0, cid, ones)
        counts = counts.clamp_min(1.0)
        means = means / counts.unsqueeze(-1)

        emb_n   = F.normalize(codebook_emb, dim=-1)
        means_n = F.normalize(means,        dim=-1)
        logits  = (emb_n @ means_n.t()) / float(temperature)
        total = total + w_l * F.cross_entropy(logits, cid)

    return total / max(weight_sum, 1e-6)
