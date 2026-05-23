"""Phase 0b — Step 2: MiniLM-encode raw item text -> itm_text_embeds.pkl.

This is h^txt in the spec (Eq 1+ of §3.1.1). It's used by SATP, by the
adaptive alignment target, and by Reliability via W_proj.

Output (per dataset, in <out>/data/<dataset>/):
    itm_text_embeds.pkl   np.ndarray [n_items, 384], float32

Usage:
    pixi run -- python scripts/02_encode_raw_item_text.py --datasets beauty toys sports yelp
"""

from __future__ import annotations
import argparse
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
def encode_texts(
    texts: list[str],
    tokenizer,
    model,
    device,
    batch_size: int = 128,
    max_length: int = 256,
) -> np.ndarray:
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="MiniLM encode"):
        batch = texts[s : s + batch_size]
        enc = tokenizer(
            batch, padding=True, truncation=True, max_length=max_length,
            return_tensors="pt",
        ).to(device)
        h = model(**enc)
        emb = mean_pool(h, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--minilm_path", type=str,
                   default=str(Path(__file__).resolve().parent.parent
                                / "LLMs/all-MiniLM-L6-v2"))
    p.add_argument("--datasets", nargs="+",
                   default=["beauty", "toys", "sports", "yelp"])
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--batch_size", type=int, default=128)
    p.add_argument("--max_length", type=int, default=256)
    args = p.parse_args()

    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"device={device}")
    print(f"loading {args.minilm_path}")
    tok = AutoTokenizer.from_pretrained(args.minilm_path)
    model = AutoModel.from_pretrained(args.minilm_path).to(device).eval()

    for ds in args.datasets:
        ddir = args.data_root / ds
        with open(ddir / "itm_text.pkl", "rb") as f:
            item_text = pickle.load(f)
        n = len(item_text)
        # Ensure ordering by id
        texts = [item_text[i] for i in range(n)]
        print(f"\n=== {ds}: encoding {n} item texts ===")
        emb = encode_texts(
            texts, tok, model, device,
            batch_size=args.batch_size, max_length=args.max_length,
        )
        assert emb.shape == (n, 384), emb.shape
        out = ddir / "itm_text_embeds.pkl"
        with open(out, "wb") as f:
            pickle.dump(emb, f)
        print(f"  saved {out}  shape={emb.shape}  dtype={emb.dtype}")


if __name__ == "__main__":
    main()
