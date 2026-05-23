"""Strict S1/S2 — Step 4: MiniLM-encode itm_prf.{s1,s2}.pkl → itm_emb_np.{s1,s2}.pkl.

Re-uses the same encoder as 06_encode_profile_text.py (mean-pool + L2 normalize).
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModel

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))


def mean_pool(model_output, attention_mask):
    last_hidden = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
    return (last_hidden * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(texts, tok, model, device, batch_size=128, max_length=128):
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="MiniLM"):
        batch = texts[s : s + batch_size]
        enc = tok(batch, padding=True, truncation=True,
                  max_length=max_length, return_tensors="pt").to(device)
        h = model(**enc)
        emb = torch.nn.functional.normalize(mean_pool(h, enc["attention_mask"]),
                                            p=2, dim=1)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    repo = Path(__file__).resolve().parent.parent
    ddir = repo / "data" / "toys"
    minilm = repo / "LLMs" / "all-MiniLM-L6-v2"

    device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(minilm))
    model = AutoModel.from_pretrained(str(minilm)).to(device).eval()
    print(f"[device] {device}")

    n_total = len(pickle.load((ddir / "itm_text.pkl").open("rb")))
    print(f"[n_items] {n_total}")

    for tag in ("s1", "s2"):
        prf_path = ddir / f"itm_prf.{tag}.pkl"
        out_path = ddir / f"itm_emb_np.{tag}.pkl"
        with prf_path.open("rb") as f:
            data: dict[int, dict] = pickle.load(f)
        print(f"\n[{tag}] loaded {prf_path}  entries={len(data)}")
        texts = [data.get(i, {}).get("profile", "").strip() for i in range(n_total)]
        n_empty = sum(1 for t in texts if not t)
        print(f"[{tag}] empty: {n_empty}/{n_total}")
        emb = encode_texts(texts, tok, model, device)
        # fallback: mean embedding for empty slots
        if n_empty > 0:
            nz = np.linalg.norm(emb, axis=1) > 1e-6
            if nz.any():
                mean = emb[nz].mean(axis=0, keepdims=True)
                mean = mean / np.linalg.norm(mean, axis=1, keepdims=True).clip(1e-9)
                emb[~nz] = mean
        assert emb.shape == (n_total, 384)
        with out_path.open("wb") as f:
            pickle.dump(emb, f)
        print(f"[{tag}] saved {out_path}  shape={emb.shape}")


if __name__ == "__main__":
    main()
