"""Strict S1/S2 — Step 1: build itm_text.s{1,2}.pkl and itm_rho.s{1,2}.pkl.

No GPT/API call here. Just:
- itm_text.s1 = title + brand + categories + price + salesrank  (no description)
- itm_text.s2 = title only
- itm_rho.s{1,2} = normalized token entropy of the corresponding raw text,
                   computed with the same MiniLM tokenizer as 01_prepare_data.py.
"""
from __future__ import annotations
import pickle
import sys
from pathlib import Path

import numpy as np
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from data_utils.entropy import batch_normalized_token_entropy


def parse_fields(text: str) -> dict[str, str]:
    out = {}
    for p in text.split("; "):
        if ":" in p:
            k, v = p.split(":", 1)
            out[k.strip()] = v.strip()
    return out


def to_s1(f):
    keep = ["title", "brand", "categories", "price", "salesrank"]
    return "; ".join(f"{k}: {f.get(k, 'na')}" for k in keep)


def to_s2(f):
    return f"title: {f.get('title', 'na')}"


def main():
    repo = Path(__file__).resolve().parent.parent
    ddir = repo / "data" / "toys"
    minilm_path = repo / "LLMs" / "all-MiniLM-L6-v2"

    with (ddir / "itm_text.pkl").open("rb") as f:
        itm_text: dict[int, str] = pickle.load(f)
    n = len(itm_text)
    print(f"[load] n_items={n}")

    tok = AutoTokenizer.from_pretrained(str(minilm_path))
    vocab_size = tok.vocab_size
    print(f"[tokenizer] MiniLM vocab_size={vocab_size}")

    def tokenize_fn(text: str):
        return tok.tokenize(text)

    for tag, builder in [("s1", to_s1), ("s2", to_s2)]:
        new_text = {iid: builder(parse_fields(itm_text[iid])) for iid in range(n)}
        # Sanity
        print(f"\n[{tag}] sample[0]: {new_text[0][:140]!r}")
        # Save text
        with (ddir / f"itm_text.{tag}.pkl").open("wb") as f:
            pickle.dump(new_text, f)
        print(f"[{tag}] saved itm_text.{tag}.pkl  entries={len(new_text)}")
        # Compute rho
        rho = np.asarray(
            batch_normalized_token_entropy(
                (new_text[i] for i in range(n)),
                tokenize_fn=tokenize_fn,
                vocab_size=vocab_size,
                show_progress=True,
            ),
            dtype=np.float32,
        )
        with (ddir / f"itm_rho.{tag}.pkl").open("wb") as f:
            pickle.dump(rho, f)
        print(f"[{tag}] saved itm_rho.{tag}.pkl  shape={rho.shape}  mean={rho.mean():.4f}  std={rho.std():.4f}")

    # Print S0 rho stats for comparison
    s0_rho = pickle.load((ddir / "itm_rho.pkl").open("rb"))
    print(f"\n[ref] S0 rho: mean={s0_rho.mean():.4f}  std={s0_rho.std():.4f}")


if __name__ == "__main__":
    main()
