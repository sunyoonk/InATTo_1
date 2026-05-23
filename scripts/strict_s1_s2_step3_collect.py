"""Strict S1/S2 — Step 3: poll OpenAI Batch status and collect when done.

For each of toys_item_s1, toys_item_s2:
  - retrieve batch status
  - if completed → download output JSONL → parse → save itm_prf.{s1,s2}.pkl
  - emit one stdout line per status change (for Monitor)

Usage:
    pixi run -- python scripts/strict_s1_s2_step3_collect.py --watch
"""
from __future__ import annotations
import argparse
import json
import os
import pickle
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

KEYS = ["toys_item_s1", "toys_item_s2"]


def parse_output_jsonl(path: Path) -> dict[str, str]:
    out = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            cid = obj["custom_id"]
            try:
                content = obj["response"]["body"]["choices"][0]["message"]["content"].strip()
            except Exception:
                continue
            out[cid] = content
    return out


def collect(client, key: str, state: dict, batches_dir: Path, data_root: Path) -> str:
    entry = state[key]
    bid = entry["batch_id"]
    info = client.batches.retrieve(bid)
    status = info.status
    counts = info.request_counts.model_dump() if info.request_counts else None
    state[key]["status"] = status
    state[key]["request_counts"] = counts

    if status != "completed":
        return status

    outputs_dir = batches_dir / "outputs"
    outputs_dir.mkdir(exist_ok=True)
    out_jsonl = outputs_dir / f"{key}.jsonl"
    if not out_jsonl.exists():
        ofid = info.output_file_id
        content = client.files.content(ofid).read()
        out_jsonl.write_bytes(content)
        print(f"  [{key}] downloaded -> {out_jsonl}")

    # key = "toys_item_s1" → tag = "s1"
    tag = key.split("_")[-1]
    id_prefix = f"toys_item_{tag}_"
    mapping = parse_output_jsonl(out_jsonl)
    pkl_dict = {}
    for cid, text in mapping.items():
        if not cid.startswith(id_prefix):
            continue
        try:
            idx = int(cid[len(id_prefix):])
        except ValueError:
            continue
        pkl_dict[idx] = {"profile": text}
    save_path = data_root / "toys" / f"itm_prf.{tag}.pkl"
    with save_path.open("wb") as f:
        pickle.dump(pkl_dict, f)
    print(f"  [{key}] saved {save_path}  entries={len(pkl_dict)}")
    return status


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--batches_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "batches/profile")
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--watch", action="store_true",
                   help="Poll repeatedly until all done; emit a line per status change.")
    p.add_argument("--interval", type=int, default=60,
                   help="Poll interval in seconds.")
    args = p.parse_args()

    load_dotenv()
    if not os.getenv("OPENAI_API_KEY"):
        sys.exit("OPENAI_API_KEY not set.")
    from openai import OpenAI
    client = OpenAI()

    state_path = args.batches_dir / "batch_ids.json"
    state = json.loads(state_path.read_text())

    last_status = {k: None for k in KEYS}
    done = {k: False for k in KEYS}

    while True:
        all_done = True
        for k in KEYS:
            if done[k]:
                continue
            s = collect(client, k, state, args.batches_dir, args.data_root)
            if s != last_status[k]:
                print(f"[{k}] {last_status[k]} -> {s}", flush=True)
                last_status[k] = s
            if s == "completed":
                done[k] = True
                print(f"[{k}] COMPLETED_AND_SAVED", flush=True)
            elif s in ("failed", "cancelled", "expired"):
                done[k] = True
                print(f"[{k}] TERMINAL_{s.upper()}", flush=True)
            else:
                all_done = False
        state_path.write_text(json.dumps(state, indent=2))
        if all_done or not args.watch:
            break
        time.sleep(args.interval)

    print("[strict] DONE", flush=True)


if __name__ == "__main__":
    main()
