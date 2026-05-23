"""One-off: generate itm_text_embeds.s1.pkl (no description) and .s2.pkl
(title only) from data/toys/itm_text.pkl, using the same MiniLM as 02.

S1 = title + brand + categories + price + salesrank   (drop description)
S2 = title only                                       (drop everything else)
"""
from __future__ import annotations
import pickle
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
from tqdm import tqdm


def parse_fields(text: str) -> dict[str, str]:
    fields = {}
    for p in text.split("; "):
        if ":" in p:
            k, v = p.split(":", 1)
            fields[k.strip()] = v.strip()
    return fields


def to_s1(fields: dict[str, str]) -> str:
    keep = ["title", "brand", "categories", "price", "salesrank"]
    return "; ".join(f"{k}: {fields.get(k, 'na')}" for k in keep)


def to_s2(fields: dict[str, str]) -> str:
    return f"title: {fields.get('title', 'na')}"


def mean_pool(model_output, attention_mask):
    h = model_output[0]
    mask = attention_mask.unsqueeze(-1).expand(h.size()).float()
    return (h * mask).sum(1) / mask.sum(1).clamp(min=1e-9)


@torch.no_grad()
def encode_texts(model, tok, texts, device, batch_size=256):
    model.eval()
    out = []
    for s in tqdm(range(0, len(texts), batch_size), desc="MiniLM"):
        chunk = texts[s : s + batch_size]
        enc = tok(chunk, padding=True, truncation=True, max_length=512, return_tensors="pt").to(device)
        h = model(**enc)
        emb = mean_pool(h, enc["attention_mask"])
        emb = torch.nn.functional.normalize(emb, p=2, dim=1)
        out.append(emb.cpu().numpy().astype(np.float32))
    return np.concatenate(out, axis=0)


def main():
    repo = Path(__file__).resolve().parent.parent
    ddir = repo / "data" / "toys"
    minilm_path = repo / "LLMs" / "all-MiniLM-L6-v2"

    with (ddir / "itm_text.pkl").open("rb") as f:
        itm_text: dict[int, str] = pickle.load(f)
    n_items = len(itm_text)
    print(f"[load] n_items={n_items}  source={ddir / 'itm_text.pkl'}")

    # build S1, S2 texts (ordered by item id)
    s1_texts, s2_texts = [], []
    for iid in range(n_items):
        fields = parse_fields(itm_text[iid])
        s1_texts.append(to_s1(fields))
        s2_texts.append(to_s2(fields))

    # sanity check
    print(f"[sample S0] {itm_text[0][:120]!r}")
    print(f"[sample S1] {s1_texts[0][:120]!r}")
    print(f"[sample S2] {s2_texts[0][:120]!r}")

    device = torch.device("cuda:5" if torch.cuda.is_available() else "cpu")
    tok = AutoTokenizer.from_pretrained(str(minilm_path))
    model = AutoModel.from_pretrained(str(minilm_path)).to(device)

    emb_s1 = encode_texts(model, tok, s1_texts, device)
    assert emb_s1.shape == (n_items, 384)
    out_s1 = ddir / "itm_text_embeds.s1.pkl"
    with out_s1.open("wb") as f:
        pickle.dump(emb_s1, f)
    print(f"[saved] {out_s1}  shape={emb_s1.shape}")

    emb_s2 = encode_texts(model, tok, s2_texts, device)
    assert emb_s2.shape == (n_items, 384)
    out_s2 = ddir / "itm_text_embeds.s2.pkl"
    with out_s2.open("wb") as f:
        pickle.dump(emb_s2, f)
    print(f"[saved] {out_s2}  shape={emb_s2.shape}")


if __name__ == "__main__":
    main()
