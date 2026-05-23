"""FACE baseline evaluation — trie-constrained beam search on the test split.

Mirror of ``09_eval_e2e.py`` but loads the FACE checkpoint produced by
``08_train_face.py``. Uses the same identifier extraction, trie, and
beam-search machinery as InATTo for direct comparability.

Usage:
    pixi run -- python scripts/09_eval_face.py --dataset toys --cuda 2
"""

from __future__ import annotations
import argparse
import json
import pickle
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from inatto.face_e2e_model import FACEE2E
from data_utils.seq_loader import InATToSeqDataset, make_collate
from generative.identifier_extractor import extract_item_identifiers
from generative.trie import ItemTrie
from generative.eval import evaluate


def _load(p):
    with open(p, "rb") as f:
        return pickle.load(f)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--ckpt", type=Path, default=None)
    p.add_argument("--beam_width", type=int, default=50)
    p.add_argument("--batch_size", type=int, default=64)
    p.add_argument("--history_max_len", type=int, default=10)
    p.add_argument("--data_root", type=Path,
                   default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--results_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--t5_path",    type=Path,
                   default=Path(__file__).resolve().parent.parent / "LLMs/t5-small")
    p.add_argument("--face_root",  type=Path,
                   default=Path(__file__).resolve().parents[2] / "FACE")
    p.add_argument("--lgn_ckpt_dir", type=Path,
                   default=Path(__file__).resolve().parent.parent / "checkpoints/lightgcn")
    args = p.parse_args()

    if args.ckpt is None:
        args.ckpt = Path(__file__).resolve().parent.parent / \
            f"checkpoints/face/face-{args.dataset}-{args.seed}.pth"
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[device] {device}")
    print(f"[ckpt] {args.ckpt}")

    # ---- Load buffers (same as training driver) ----
    ddir = args.data_root / args.dataset
    h_txt_item = torch.tensor(_load(ddir / "itm_text_embeds.pkl")).float()
    itm_rho = torch.tensor(_load(ddir / "itm_rho.pkl")).float()
    usr_rho = torch.tensor(_load(ddir / "usr_rho.pkl")).float()
    itm_emb_np = _load(ddir / "itm_emb_np.pkl")
    usr_emb_np = _load(ddir / "usr_emb_np.pkl")
    h_raw_item = torch.tensor(np.asarray(itm_emb_np)).float()
    h_raw_user = torch.tensor(np.asarray(usr_emb_np)).float()
    h_txt_user = h_raw_user.clone()
    n_items = h_txt_item.shape[0]; n_users = h_raw_user.shape[0]

    lgn_path = args.lgn_ckpt_dir / f"lightgcn-{args.dataset}-{args.seed}.pth"
    state = torch.load(lgn_path, map_location="cpu", weights_only=False)
    z_user = state["user_embeds"].float()
    z_item = state["item_embeds"].float()

    # Load trained config from ckpt so n_aspects / d_aspect match training.
    print(f"[load] {args.ckpt}")
    ckpt_state = torch.load(args.ckpt, map_location=device, weights_only=False)
    cfg = ckpt_state.get("cfg", {})

    model = FACEE2E(
        user_buffers=(z_user, h_txt_user, usr_rho, h_raw_user),
        item_buffers=(z_item, h_txt_item, itm_rho, h_raw_item),
        t5_path=args.t5_path,
        dataset_name=args.dataset,
        face_root=args.face_root,
        cfg=cfg,
    ).to(device)
    model.load_state_dict(ckpt_state["model_state"])

    # ---- Extract identifiers + build trie ----
    print("[trie] extracting identifiers ...")
    item_ids, lengths = extract_item_identifiers(model, n_items, device,
                                                  batch_size=512)
    print(f"[trie] identifier length: min={lengths.min()}  max={lengths.max()}  "
          f"mean={lengths.mean():.1f}")
    trie = ItemTrie.from_identifiers(item_ids.values(), item_ids.keys())
    print(f"[trie] n_items={trie.n_items}  max_depth={trie.max_depth}  "
          f"collisions={trie.collision_count()}")

    # ---- Eval ----
    test_ds = InATToSeqDataset(args.data_root, args.dataset, "test",
                                history_max_len=args.history_max_len)
    coll = make_collate(args.history_max_len)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False,
                              num_workers=2, collate_fn=coll)
    print(f"[test] N={len(test_ds)}")
    metrics = evaluate(model, test_loader, trie,
                        ks=(5, 10, 20), beam_width=args.beam_width,
                        device=device)
    print(f"[result] {metrics}")

    args.results_dir.mkdir(parents=True, exist_ok=True)
    out = args.results_dir / f"face-{args.dataset}-{args.seed}.eval.json"
    out.write_text(json.dumps({
        **metrics,
        "n_items": trie.n_items,
        "collisions": trie.collision_count(),
        "id_length_mean": float(lengths.mean()),
        "id_length_min": int(lengths.min()),
        "id_length_max": int(lengths.max()),
        "beam_width": args.beam_width,
    }, indent=2))
    print(f"[saved] {out}")


if __name__ == "__main__":
    main()
