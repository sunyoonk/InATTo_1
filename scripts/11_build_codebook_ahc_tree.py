"""Build a hierarchical tree over the codebook by AHC on MiniLM
embeddings — IAHQ Phase-1 quality check.

The Stage-1 codebook is a curated list of V = 9,338 English words; their
frozen MiniLM-L6-v2 embeddings live in the codebook's `codebook_raw`
buffer. To enable Information-Adaptive Hierarchical Quantization we
need a *semantic* tree over these words so that a residual at a given
tree level corresponds to a meaningful granularity (coarse → fine).

This script:
  1. Loads the 9,338-word vocab + MiniLM frozen embeddings.
  2. Runs Agglomerative Hierarchical Clustering (Ward linkage, cosine
     pre-normalisation) once and cuts the tree at K = 64 / 256 / 1024
     / 9338 to give a 4-level hierarchy.
  3. Saves the per-level cluster IDs.
  4. Prints sample words for the first ~10 clusters at each level so we
     can eyeball whether the tree captures coarse semantic categories
     before committing to a full Stage-1 redesign.

Output:
  data/<dataset>/codebook_ahc_tree.pkl  (or assets/ if dataset-agnostic)
    {
      'level_K':       {1: 64, 2: 256, 3: 1024, 4: V},
      'cluster_ids':   [(V,)] × 4   # cluster id at each level
      'tokens':        list[str]    # codeword strings, length V
    }
"""

from __future__ import annotations
import argparse
import pickle
import sys
from pathlib import Path
from collections import defaultdict, Counter

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.modules.codebook import Codebook  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minilm_path", type=Path,
                    default=Path(__file__).resolve().parent.parent / "LLMs" / "all-MiniLM-L6-v2")
    p.add_argument("--vocab_xlsx", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets"
                          / "word_frequency_list_60000_English.xlsx")
    p.add_argument("--levels", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--linkage", type=str, default="average",
                    choices=["ward", "average", "complete", "single"])
    p.add_argument("--metric", type=str, default="cosine",
                    choices=["cosine", "euclidean"])
    p.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets"
                          / "codebook_ahc_tree.pkl")
    p.add_argument("--sample_clusters", type=int, default=10,
                    help="Print this many cluster samples per level.")
    p.add_argument("--samples_per_cluster", type=int, default=12)
    args = p.parse_args()

    # ---- 1. Load codebook (frozen MiniLM word embeddings) ----
    print(f"[load] codebook from {args.minilm_path}  vocab={args.vocab_xlsx}")
    cb = Codebook(
        llm_path=str(args.minilm_path),
        coca_path=args.vocab_xlsx,
        d_aspect=256,                                  # not used (we only need raw)
    )
    raw = cb.codebook_raw.detach().float().cpu().numpy()   # (V, 384)
    tokens = cb.vocabulary                                   # list[str]
    V, d = raw.shape
    assert V == len(tokens), f"V {V} mismatch with tokens {len(tokens)}"
    print(f"[load] V={V}  d_llm={d}  examples: {tokens[:8]}")

    # ---- 2. L2-normalise (cosine ≡ euclidean on unit sphere) ----
    X = raw / np.linalg.norm(raw, axis=-1, keepdims=True).clip(min=1e-8)

    # ---- 3. AHC (memory-aware: 9338² pairs ≈ 87M floats = 350MB) ----
    print(f"\n[AHC] linkage={args.linkage}  metric={args.metric}  V={V}")
    from sklearn.cluster import AgglomerativeClustering

    cluster_ids_per_level = []
    for K in args.levels:
        print(f"  cutting at K={K} ...")
        # AHC re-fit at each K (sklearn doesn't cache the tree directly,
        # but the cost is amortised — pairwise distances dominate).
        ac = AgglomerativeClustering(
            n_clusters=K, metric=args.metric, linkage=args.linkage,
        )
        cid = ac.fit_predict(X)
        cluster_ids_per_level.append(cid.astype(np.int32))
        sizes = np.bincount(cid, minlength=K)
        print(f"    sizes: min={sizes.min()}  max={sizes.max()}  "
               f"median={int(np.median(sizes))}  mean={sizes.mean():.1f}")

    # Add the leaf level (each codeword is its own cluster)
    levels_with_leaf = list(args.levels) + [V]
    cluster_ids_per_level.append(np.arange(V, dtype=np.int32))

    # ---- Representative word per cluster ----
    # Two parallel schemes computed and stored:
    #   (1) nearest_center   — cluster mean에 가장 가까운 leaf word (geometric)
    #   (2) ★ c_tf_idf       — C-TF-IDF (Class-based TF-IDF, BERTopic-style):
    #                          cluster 안 codewords가 *cluster-internal vs external*
    #                          frequency ratio로 most discriminative leaf 선택.
    #
    # C-TF-IDF score for codeword w in cluster c at level l:
    #     TF(w, c)  = count of word w among codewords assigned to c at level l
    #                 / sum of all codeword counts in c
    #     IDF(w)    = log(L / Σ_{c'} TF(w, c'))     (smoothed)
    #     C-TF-IDF  = TF(w, c) · IDF(w)
    # Top-1 codeword (within cluster c's member set) = representative leaf.
    #
    # This makes the representative semantically meaningful (the word that
    # distinguishes this cluster from its siblings) instead of just
    # geometrically central.
    repr_word_per_level         = []   # nearest-center (legacy)
    repr_word_ctfidf_per_level  = []   # ★ C-TF-IDF
    for li, cid in enumerate(cluster_ids_per_level[:-1]):
        K_l = levels_with_leaf[li]
        cid_np = cid.numpy() if hasattr(cid, 'numpy') else np.asarray(cid)

        # ---- (1) nearest center ----
        centers = np.zeros((K_l, d), dtype=np.float32)
        counts  = np.zeros(K_l, dtype=np.int64)
        for w in range(V):
            c = int(cid_np[w])
            centers[c] += X[w]; counts[c] += 1
        centers = centers / counts.clip(min=1)[:, None]
        centers = centers / np.linalg.norm(centers, axis=-1, keepdims=True).clip(min=1e-8)
        sims = centers @ X.T                                  # (K_l, V)
        nearest_idx = sims.argmax(axis=-1).astype(np.int32)
        repr_word_per_level.append(nearest_idx)

        # ---- (2) C-TF-IDF ----
        # Each cluster c has a member set of leaf indices (the leaves
        # whose AHC cluster id at level l == c). TF(w in c) = 1 if w is a
        # member of c else 0 — i.e., each leaf contributes "one unit" to
        # exactly one cluster at every level. So TF here is simply the
        # incidence matrix (V, K_l); per-cluster normalisation gives the
        # within-cluster word frequency. IDF is global rarity across
        # cluster cells: a leaf belongs to a unique cluster id per level,
        # so naive IDF is uninformative — we instead use *cluster size*
        # as an inverse-popularity proxy (larger clusters are less
        # discriminative, like document length penalty).
        member_count = counts.clip(min=1)                     # (K_l,)
        # For each cluster c, candidate members are the leaves with cid==c
        ctfidf_idx = np.zeros(K_l, dtype=np.int32)
        for c in range(K_l):
            members = np.where(cid_np == c)[0]
            if members.size == 0:
                ctfidf_idx[c] = nearest_idx[c]
                continue
            # Sort members by (a) inverse cluster size (smaller cluster → more discriminative);
            # actually all members share the same cluster size, so use
            # distance to the cluster center as a tiebreaker — nearest
            # word that's also *short / common* tends to be a better label.
            # Plus we *do* downweight leaves that appear close to many
            # other cluster centers (low across-cluster discriminability).
            member_emb = X[members]                            # (m, d)
            center_sim = member_emb @ centers[c]              # (m,) within
            cross_sim  = member_emb @ centers.T                # (m, K_l) across
            # Discriminative: high within, low max-cross-others.
            others_max = np.delete(cross_sim, c, axis=1).max(axis=1)
            score = center_sim - 0.5 * others_max
            ctfidf_idx[c] = int(members[score.argmax()])
        repr_word_ctfidf_per_level.append(ctfidf_idx)

        # Print a few comparisons
        print(f"  level {li+1} (K={K_l}) representatives — first 6 clusters:")
        for cc in range(min(6, K_l)):
            print(f"    cluster {cc:>3}: nearest='{tokens[nearest_idx[cc]]}' "
                   f"vs c-tf-idf='{tokens[ctfidf_idx[cc]]}'  "
                   f"(size={int(counts[cc])})")
    # Leaf level: each word is its own representative (both schemes)
    repr_word_per_level.append(np.arange(V, dtype=np.int32))
    repr_word_ctfidf_per_level.append(np.arange(V, dtype=np.int32))

    # ---- 4. Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "level_K":     {i + 1: K for i, K in enumerate(levels_with_leaf)},
        "cluster_ids": cluster_ids_per_level,
        "repr_word":   repr_word_per_level,    # ★ NEW: per-cluster representative leaf index
        "tokens":      tokens,
        "linkage":     args.linkage,
        "metric":      args.metric,
    }
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"\n[saved] {args.out}")

    # ---- 5. Sample inspection — semantic coherence eyeball ----
    print("\n" + "=" * 80)
    print("=== Cluster samples (eyeball whether semantics look coherent) ===")
    rng = np.random.default_rng(2023)
    for li, (K, cid) in enumerate(zip(levels_with_leaf[:-1], cluster_ids_per_level[:-1])):
        print(f"\n--- Level {li + 1}  (K = {K}) ---")
        groups = defaultdict(list)
        for tok, c in zip(tokens, cid):
            groups[int(c)].append(tok)
        # pick the largest clusters first (more representative)
        order = sorted(groups.keys(), key=lambda k: -len(groups[k]))
        for c in order[: args.sample_clusters]:
            members = groups[c]
            shown = members[: args.samples_per_cluster]
            print(f"  cluster {c:>4} ({len(members):>4} words):  "
                   f"{', '.join(shown)}{' …' if len(members) > len(shown) else ''}")


if __name__ == "__main__":
    main()
