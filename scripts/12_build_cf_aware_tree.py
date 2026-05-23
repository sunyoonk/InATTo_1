"""CF-supervised hierarchical codebook tree.

Stage 0 / once-off. Builds a hierarchy over the V = 9,338 codeword
vocabulary using *both* signals:

    text-side  X_text = MiniLM(word)            ∈ R^{384}      (frozen)
    CF-side    X_cf   = mean LightGCN(z_i)       ∈ R^{256}
                       for items whose RLMRec
                       metadata contains the word
    hybrid     X      = [X_text ; α · X_cf]      ∈ R^{384+256}

Agglomerative Hierarchical Clustering on this hybrid embedding gives a
tree whose internal nodes reflect *both* semantic similarity and
behavioural co-occurrence — the property that lets multi-level
quantization actually carry distinct information at each level
(addresses the "level-1 sufficiency / φ stuck" pathology of the
text-only AHC tree).

Output: assets/codebook_ahc_tree_cf{α}.pkl
"""

from __future__ import annotations
import argparse
import pickle
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.modules.codebook import Codebook
from inatto.backbone.lightgcn import LightGCN
from data_utils.adj_builder import build_torch_adj


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--minilm_path", type=Path,
                    default=Path(__file__).resolve().parent.parent / "LLMs/all-MiniLM-L6-v2")
    p.add_argument("--vocab_xlsx", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets/word_frequency_list_60000_English.xlsx")
    p.add_argument("--dataset", default="toys",
                    help="Dataset that provides the CF supervision (LightGCN + itm_text).")
    p.add_argument("--data_root", type=Path,
                    default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--lightgcn_ckpt", type=Path,
                    default=Path(__file__).resolve().parent.parent / "checkpoints/lightgcn/lightgcn-toys-2023.pth")
    p.add_argument("--levels", nargs="+", type=int, default=[32, 128, 512])
    p.add_argument("--alpha", type=float, default=0.5,
                    help="CF weight in the hybrid embedding ([0,1]). "
                         "0 = text only (legacy AHC), 1 = CF only.")
    p.add_argument("--linkage", default="average")
    p.add_argument("--metric",  default="cosine")
    p.add_argument("--out", type=Path,
                    default=Path(__file__).resolve().parent.parent / "assets/codebook_ahc_tree_cf.pkl")
    p.add_argument("--sample_clusters", type=int, default=6)
    p.add_argument("--samples_per_cluster", type=int, default=8)
    args = p.parse_args()

    print(f"[CF-AHC] dataset={args.dataset}  α={args.alpha}")
    device = torch.device("cuda:2" if torch.cuda.is_available() else "cpu")

    # ---- 1. Codebook (frozen MiniLM word embeddings) ----
    cb = Codebook(llm_path=str(args.minilm_path),
                   coca_path=args.vocab_xlsx, d_aspect=256)
    X_text = cb.codebook_raw.detach().float().cpu().numpy()       # (V, 384)
    tokens = cb.vocabulary
    V, d_txt = X_text.shape
    print(f"  V = {V}  d_text = {d_txt}")

    # ---- 2. CF anchor per codeword ----
    # For each codeword, find items whose RLMRec metadata contains that
    # word; CF anchor = mean of LightGCN-propagated item embeddings.
    ddir = args.data_root / args.dataset
    with open(ddir / "itm_text.pkl", "rb") as f:
        itm_text = pickle.load(f)
    with open(ddir / "trn_mat.pkl", "rb") as f:
        trn = pickle.load(f)
    n_users, n_items = trn.shape

    # Tokenise each item's text into the codebook vocabulary (word match)
    word_to_idx = {w: i for i, w in enumerate(tokens)}
    item_to_codewords: list[set[int]] = [set() for _ in range(n_items)]
    for iid, text in itm_text.items():
        if not isinstance(text, str):
            continue
        for tok in text.lower().split():
            tok = tok.strip(",.;:!?\"'()[]{}")
            if tok in word_to_idx:
                item_to_codewords[iid].add(word_to_idx[tok])
    coverage = sum(1 for s in item_to_codewords if s) / n_items
    print(f"  item ↦ codeword coverage: {coverage*100:.1f}% items have ≥1 codeword in vocab")

    codeword_to_items: list[list[int]] = [[] for _ in range(V)]
    for iid, words in enumerate(item_to_codewords):
        for w in words:
            codeword_to_items[w].append(iid)
    word_hit = sum(1 for items in codeword_to_items if items)
    print(f"  codeword hit rate: {word_hit}/{V} ({word_hit/V*100:.1f}%)")

    # ---- 3. Propagate LightGCN to get z_item ----
    print(f"  loading LightGCN: {args.lightgcn_ckpt}")
    adj = build_torch_adj(trn, n_users, n_items, device)
    lgcn = LightGCN(n_users=n_users, n_items=n_items,
                     embedding_size=256, layer_num=3).to(device)
    ck = torch.load(args.lightgcn_ckpt, map_location=device, weights_only=False)
    # The training script saves user_embeds/item_embeds alongside metadata
    # keys (n_users, n_items, epoch, ...). Filter to the actual params.
    state = {k: ck[k] for k in ("user_embeds", "item_embeds") if k in ck}
    lgcn.load_state_dict(state, strict=False)
    lgcn.eval()
    with torch.no_grad():
        z_user, z_item = lgcn.propagate(adj)
    z_item = z_item.cpu().numpy()                              # (n_items, 256)

    # Each codeword's CF anchor: mean of items containing that word
    X_cf = np.zeros((V, 256), dtype=np.float32)
    for w in range(V):
        items = codeword_to_items[w]
        if items:
            X_cf[w] = z_item[items].mean(axis=0)
    # L2-normalise both sides for fair concat
    X_text_n = X_text / np.linalg.norm(X_text, axis=-1, keepdims=True).clip(min=1e-8)
    X_cf_n   = X_cf   / np.linalg.norm(X_cf,   axis=-1, keepdims=True).clip(min=1e-8)

    # Hybrid embedding
    X_hybrid = np.concatenate([X_text_n, args.alpha * X_cf_n], axis=-1)
    # Re-normalise the concat to put text and CF on the same sphere scale
    X_hybrid = X_hybrid / np.linalg.norm(X_hybrid, axis=-1, keepdims=True).clip(min=1e-8)
    print(f"  hybrid embedding: shape={X_hybrid.shape}  α={args.alpha}")

    # ---- 4. AHC per level ----
    from sklearn.cluster import AgglomerativeClustering
    cluster_ids_per_level = []
    print(f"\n[AHC] linkage={args.linkage}  metric={args.metric}")
    for K_l in args.levels:
        print(f"  K={K_l} ...")
        cid = AgglomerativeClustering(
            n_clusters=K_l, metric=args.metric, linkage=args.linkage
        ).fit_predict(X_hybrid)
        cluster_ids_per_level.append(cid.astype(np.int32))
        sizes = np.bincount(cid, minlength=K_l)
        print(f"    sizes: min={sizes.min()}  max={sizes.max()}  median={int(np.median(sizes))}")

    # Leaf level
    levels_with_leaf = list(args.levels) + [V]
    cluster_ids_per_level.append(np.arange(V, dtype=np.int32))

    # ---- 5. Representative leaves ----
    # Compute both nearest-center and C-TF-IDF reps from the *hybrid*
    # embedding (so reps reflect CF-supervised clusters too).
    repr_nearest = []
    repr_ctfidf  = []
    for li, cid in enumerate(cluster_ids_per_level[:-1]):
        K = levels_with_leaf[li]
        # Cluster center (mean of hybrid embeddings, then re-normalise)
        centers = np.zeros((K, X_hybrid.shape[1]), dtype=np.float32)
        cnt = np.zeros(K, dtype=np.int64)
        for w in range(V):
            c = int(cid[w]); centers[c] += X_hybrid[w]; cnt[c] += 1
        centers = centers / cnt.clip(min=1)[:, None]
        centers = centers / np.linalg.norm(centers, axis=-1, keepdims=True).clip(min=1e-8)
        sims = centers @ X_hybrid.T                                # (K, V)
        nearest_idx = sims.argmax(axis=-1).astype(np.int32)
        repr_nearest.append(nearest_idx)
        # C-TF-IDF: within-cluster minus 0.5 × max across other centers
        ctfidf_idx = np.zeros(K, dtype=np.int32)
        for c in range(K):
            members = np.where(cid == c)[0]
            if members.size == 0:
                ctfidf_idx[c] = nearest_idx[c]; continue
            within = X_hybrid[members] @ centers[c]
            cross  = X_hybrid[members] @ centers.T
            others = np.delete(cross, c, axis=1).max(axis=1)
            score = within - 0.5 * others
            ctfidf_idx[c] = int(members[score.argmax()])
        repr_ctfidf.append(ctfidf_idx)
        print(f"  level {li+1} (K={K}) — first 6 reps: "
               f"nearest={[tokens[nearest_idx[c]] for c in range(min(6, K))]}  "
               f"vs C-TF-IDF={[tokens[ctfidf_idx[c]] for c in range(min(6, K))]}")
    repr_nearest.append(np.arange(V, dtype=np.int32))
    repr_ctfidf.append(np.arange(V, dtype=np.int32))

    # ---- 6. Save ----
    args.out.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "level_K":          {i + 1: K for i, K in enumerate(levels_with_leaf)},
        "cluster_ids":      cluster_ids_per_level,
        "repr_word":        repr_nearest,
        "repr_word_ctfidf": repr_ctfidf,
        "tokens":           tokens,
        "linkage":          args.linkage,
        "metric":           args.metric,
        "alpha":            args.alpha,
        "cf_supervised":    True,
        "cf_dataset":       args.dataset,
    }
    with args.out.open("wb") as f:
        pickle.dump(payload, f)
    print(f"\n[saved] {args.out}")

    # Inspect: how different is the CF-aware tree from text-only?
    print("\n=== Sample clusters (CF-aware, level 1) ===")
    cid_l1 = cluster_ids_per_level[0]
    groups = defaultdict(list)
    for tok, c in zip(tokens, cid_l1):
        groups[int(c)].append(tok)
    order = sorted(groups.keys(), key=lambda k: -len(groups[k]))
    for c in order[: args.sample_clusters]:
        members = groups[c][: args.samples_per_cluster]
        print(f"  cluster {c:>3} ({len(groups[c]):>4} words): {', '.join(members)} ...")


if __name__ == "__main__":
    main()
