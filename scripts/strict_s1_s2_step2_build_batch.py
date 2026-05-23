"""Strict S1/S2 — Step 2: build batch JSONL for GPT-4o-mini item profiles.

Reads itm_text.s{1,2}.pkl, writes:
    batches/profile/toys_item_s1.jsonl
    batches/profile/toys_item_s2.jsonl

Same prompt template as 03_build_profile_batch.py (item kind only).
User profiles are NOT regenerated (Strict applies to item side only).
"""
from __future__ import annotations
import json
import pickle
from pathlib import Path


SYSTEM_PROMPT_ITEM = "Use one short sentence: 'The item attributes are xxx.'"


def make_request(custom_id, txt, model="gpt-4o-mini"):
    return {
        "custom_id": custom_id,
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": model,
            "messages": [
                {"role": "system", "content": SYSTEM_PROMPT_ITEM},
                {"role": "user", "content": f'An item description: "{txt}". Summarize its attributes.'},
            ],
            "temperature": 0.6,
            "top_p": 0.9,
            "max_tokens": 100,
        },
    }


def main():
    repo = Path(__file__).resolve().parent.parent
    ddir = repo / "data" / "toys"
    out_dir = repo / "batches" / "profile"
    out_dir.mkdir(parents=True, exist_ok=True)

    for tag in ("s1", "s2"):
        with (ddir / f"itm_text.{tag}.pkl").open("rb") as f:
            itm_text = pickle.load(f)
        out_path = out_dir / f"toys_item_{tag}.jsonl"
        n_written = n_skipped = 0
        with out_path.open("w") as f:
            for iid in sorted(itm_text.keys()):
                txt = itm_text[iid].strip()
                if not txt:
                    n_skipped += 1
                    continue
                if len(txt) > 3500:
                    txt = txt[:3500]
                req = make_request(custom_id=f"toys_item_{tag}_{iid}", txt=txt)
                f.write(json.dumps(req, ensure_ascii=False) + "\n")
                n_written += 1
        print(f"[{tag}] wrote {out_path}  n={n_written}  skipped(empty)={n_skipped}  size={out_path.stat().st_size/1e6:.2f}MB")


if __name__ == "__main__":
    main()
