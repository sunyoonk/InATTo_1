"""Stage 2 — T5 generative training on frozen identifiers from Stage 1.

Reads the identifier cache produced by 08a_train_stage1.py and trains a
T5-small to predict the next item's identifier given (user identifier,
history item identifiers). The tokenizer / codebook / depth-mask are no
longer trained — only T5 weights and the extended-vocabulary embeddings
that the STE bridge added during Stage 1 build.

This is the standard TIGER/GRAM-style generative-rec recipe; the only
InATTo-specific aspect is the variable-length identifier produced by
Stage 1.

Usage:
    pixi run -- python scripts/08b_train_stage2.py --dataset toys --cuda 2

Saves:
    checkpoints/inatto/inatto-<ds>-<seed>.stage2.pth
    checkpoints/inatto/inatto-<ds>-<seed>.stage2.latest.pth
    results/inatto-<ds>-<seed>.stage2.log.json
"""

from __future__ import annotations
import argparse
import json
import math
import pickle
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from data_utils.seq_loader import InATToSeqDataset
from inatto.modules.ste_bridge import STEBridge, SPECIAL_TOKENS, _codeword_to_t5_token
from inatto.modules.codebook import Codebook
from generative.trie import ItemTrie
from generative.beam_search import make_prefix_allowed_tokens_fn


def _set_seed(seed: int):
    random.seed(seed); np.random.seed(seed)
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------------
# Dataset wrapper that builds T5 input/target token sequences from the cache
# ---------------------------------------------------------------------------

class FrozenIdentifierDataset(Dataset):
    """Wrap InATToSeqDataset; on __getitem__ assemble T5 token sequences.

    Each sample is a dict of pre-tokenized ints:
        input_ids   : [USER_BOS  user_codes  USER_EOI  HIST_BOS  hist_codes  HIST_EOI]
        labels      : target_item_codes (already ends with <EOI>)
    """

    def __init__(self, base: InATToSeqDataset, cache: dict,
                 max_input_len: int = 1024, max_target_len: int = 80):
        self.base = base
        self.user_id_seq: dict[int, list[int]] = cache["user"]
        self.item_id_seq: dict[int, list[int]] = cache["item"]
        self.special = cache["cfg"]["special_ids"]   # name -> T5 token id
        self.max_input_len = int(max_input_len)
        self.max_target_len = int(max_target_len)

    def __len__(self):
        return len(self.base)

    def _build_input(self, uid: int, history: list[int]) -> list[int]:
        u = self.user_id_seq[int(uid)]
        seq = [self.special["<USER_BOS>"]] + u + [self.special["<USER_EOI>"]] + \
              [self.special["<HIST_BOS>"]]
        for hid in history:
            seq = seq + self.item_id_seq[int(hid)]      # item seq already ends with EOI
        seq = seq + [self.special["<HIST_EOI>"]]
        return seq[: self.max_input_len]

    def _build_target(self, target_id: int) -> list[int]:
        return self.item_id_seq[int(target_id)][: self.max_target_len]

    def __getitem__(self, idx: int) -> dict:
        # InATToSeqDataset.__getitem__ returns (uid, target, history_list).
        # The history is already truncated to history_max_len; no `valid` mask
        # is produced at the per-sample level (that lives in the collate fn).
        uid, tgt, hist = self.base[idx]
        uid_int = int(uid)
        tgt_int = int(tgt)
        history_int = [int(h) for h in hist]

        input_ids = self._build_input(uid_int, history_int)
        labels = self._build_target(tgt_int)
        return {"input_ids": input_ids, "labels": labels}


@torch.no_grad()
def quick_eval_R5(t5, val_loader, trie, eoi_id: int, device,
                  max_samples: int = 1000, beam_width: int = 20, K: int = 5):
    """Quick val R@K for early stopping.

    Uses trie-constrained beam search on the first ``max_samples`` val
    examples with a reduced beam width. Returns hit rate (= R@K).

    Computation:
        for each test (user, target):
            generate top-K item ids via trie + beam
            hit if target item_id is in the top-K

    Uses ``trie.get_node(seq).item_id`` to map a generated token sequence
    back to its item id, matching how full eval works in
    generative/eval.py.
    """
    t5.eval()
    decoder_start = t5.config.decoder_start_token_id
    if decoder_start is None:
        decoder_start = t5.config.pad_token_id
    prefix_fn = make_prefix_allowed_tokens_fn(trie, decoder_start, eos_token_id=eoi_id)

    hits = 0
    total = 0
    for batch in val_loader:
        if total >= max_samples: break
        input_ids = batch["input_ids"].to(device)
        attn      = batch["attention_mask"].to(device)
        labels    = batch["labels"].to(device)

        out = t5.generate(
            input_ids=input_ids,
            attention_mask=attn,
            num_beams=beam_width,
            num_return_sequences=min(K, beam_width),
            max_new_tokens=trie.max_depth + 2,
            prefix_allowed_tokens_fn=prefix_fn,
            eos_token_id=eoi_id,
            early_stopping=True,
            return_dict_in_generate=True,
            output_scores=False,
        )
        sequences = out.sequences
        B = input_ids.shape[0]
        R = sequences.shape[0] // B
        sequences = sequences.view(B, R, -1)

        for b in range(B):
            # Ground-truth target item id from labels (labels are the
            # cached target_item identifier sequence; map via trie).
            tgt_tokens = [int(t) for t in labels[b].cpu().tolist() if t != -100]
            node_t = trie.get_node(tgt_tokens)
            target_iid = node_t.item_id if (node_t is not None and node_t.is_leaf) else -1
            if target_iid < 0:
                # Shouldn't happen if cache is correct; skip this sample.
                continue

            # Top-K predicted item ids by trie traversal.
            top_ids: list[int] = []
            seen: set[int] = set()
            for r in range(R):
                pred = sequences[b, r].cpu().tolist()
                if pred and pred[0] == decoder_start:
                    pred = pred[1:]
                if eoi_id in pred:
                    pred = pred[: pred.index(eoi_id) + 1]
                node = trie.get_node(pred)
                if (node is not None and node.is_leaf and
                        node.item_id is not None and node.item_id not in seen):
                    top_ids.append(node.item_id)
                    seen.add(node.item_id)

            if target_iid in top_ids[:K]:
                hits += 1
            total += 1
            if total >= max_samples:
                break

    return hits / total if total > 0 else 0.0


def make_collate(pad_id: int, label_pad: int = -100):
    """Right-pad input_ids with pad_id and labels with -100 (HF CE ignores -100)."""
    def collate(batch: list[dict]) -> dict:
        max_in = max(len(b["input_ids"]) for b in batch)
        max_lb = max(len(b["labels"])    for b in batch)
        B = len(batch)
        input_ids = torch.full((B, max_in), pad_id, dtype=torch.long)
        attn      = torch.zeros((B, max_in), dtype=torch.long)
        labels    = torch.full((B, max_lb), label_pad, dtype=torch.long)
        for i, b in enumerate(batch):
            n = len(b["input_ids"])
            input_ids[i, :n] = torch.tensor(b["input_ids"], dtype=torch.long)
            attn[i, :n] = 1
            m = len(b["labels"])
            labels[i, :m] = torch.tensor(b["labels"], dtype=torch.long)
        return {"input_ids": input_ids, "attention_mask": attn, "labels": labels}
    return collate


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--seed", type=int, default=2023)
    p.add_argument("--cuda", type=int, default=2)
    p.add_argument("--data_root",  type=Path, default=Path(__file__).resolve().parent.parent / "data")
    p.add_argument("--llm_path",   type=Path, default=Path(__file__).resolve().parent.parent / "LLMs/all-MiniLM-L6-v2")
    p.add_argument("--t5_path",    type=Path, default=Path(__file__).resolve().parent.parent / "LLMs/t5-small")
    p.add_argument("--coca_path",  type=Path, default=Path(__file__).resolve().parent.parent / "assets/word_frequency_list_60000_English.xlsx")
    p.add_argument("--ckpt_dir",   type=Path, default=Path(__file__).resolve().parent.parent / "checkpoints/inatto")
    p.add_argument("--results_dir", type=Path, default=Path(__file__).resolve().parent.parent / "results")
    p.add_argument("--batch_size", type=int, default=128,
                   help="T5-only forward+backward; smaller than Stage 1.")
    p.add_argument("--lr_t5", type=float, default=1e-4)
    p.add_argument("--total_epochs", type=int, default=40,
                   help="Max Stage 2 epochs (early stopping usually cuts this short).")
    p.add_argument("--history_max_len", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--max_input_len", type=int, default=1024)
    p.add_argument("--max_target_len", type=int, default=80)
    p.add_argument("--resume", action="store_true",
                   help="Resume from latest .stage2.latest.pth if present.")
    p.add_argument("--cache_suffix", type=str, default="",
                   help="Optional suffix for identifier_cache path; default uses "
                        "identifier_cache.<seed>.pkl, set 'bpr' to load .bpr.pkl. "
                        "Also propagates to ckpt names so BPR/no-BPR runs do not collide.")
    # ---- Early stopping (R@5 based) ----
    p.add_argument("--patience", type=int, default=5,
                   help="Stop if quick val R@5 doesn't improve for this many "
                        "evaluations in a row.")
    p.add_argument("--eval_every", type=int, default=5,
                   help="Run quick val R@5 every N epochs (also val_L_gen plot).")
    p.add_argument("--quick_eval_users", type=int, default=1000,
                   help="Subset size for quick R@5; full eval runs after training.")
    p.add_argument("--quick_eval_beam", type=int, default=20,
                   help="Beam width for quick R@5 (full eval uses 50).")
    p.add_argument("--quick_eval_batch", type=int, default=8,
                   help="Eval batch size for quick R@5 beam search. LETTER's "
                        "test_batch_size=2-32 while keeping train batch 256+. "
                        "Smaller than train batch to avoid OOM during beam.")
    p.add_argument("--warmup_ratio", type=float, default=0.01,
                   help="Fraction of total steps to warm up linearly from 0 "
                        "to lr_t5. LETTER convention.")
    p.add_argument("--lr_scheduler", type=str, default="cosine",
                   choices=["cosine", "constant"],
                   help="LR schedule after warmup. LETTER uses cosine.")
    p.add_argument("--weight_decay", type=float, default=0.01,
                   help="AdamW weight decay. LETTER uses 0.01.")
    args = p.parse_args()

    _set_seed(args.seed)
    device = torch.device(f"cuda:{args.cuda}" if torch.cuda.is_available() else "cpu")
    print(f"[args] {vars(args)}")
    print(f"[device] {device}")

    # ---- Load identifier cache from Stage 1 ----
    ddir = args.data_root / args.dataset
    # cache_suffix lets us pick which Stage-1 variant's identifiers to use
    # ("" -> identifier_cache.<seed>.pkl, "bpr" -> identifier_cache.<seed>.bpr.pkl, ...).
    suffix = f".{args.cache_suffix}" if args.cache_suffix else ""
    tag    = f".{args.cache_suffix}" if args.cache_suffix else ""
    cache_path = ddir / f"identifier_cache.{args.seed}{suffix}.pkl"
    print(f"[stage1] loading identifier cache from {cache_path}")
    with cache_path.open("rb") as f:
        cache = pickle.load(f)
    item_lens = cache["item_lengths"]; user_lens = cache["user_lengths"]
    print(f"  items={len(cache['item'])}  users={len(cache['user'])}")
    print(f"  item len mean={item_lens.mean():.1f}  user len mean={user_lens.mean():.1f}")

    # ---- Rebuild STE bridge so T5 token-id mapping matches the cache exactly ----
    # STEBridge needs the same vocabulary list; reconstruct from the codebook
    # (small CSV cached by FACE, or rebuilt from MiniLM filter — fast).
    codebook = Codebook(llm_path=args.llm_path, d_aspect=256, coca_path=args.coca_path)
    assert codebook.V == cache["cfg"]["V"], (
        f"codebook mismatch: built V={codebook.V}, cache V={cache['cfg']['V']}"
    )
    bridge = STEBridge(t5_path=args.t5_path, vocabulary=codebook.vocabulary).to(device)
    # Sanity-check that the bridge's code_to_t5 matches what Stage 1 saved.
    expected = torch.tensor(cache["cfg"]["code_to_t5"], dtype=torch.long)
    assert torch.equal(bridge.code_to_t5.cpu(), expected), \
        "code_to_t5 mismatch between Stage 1 and Stage 2 — vocabulary order changed."
    print(f"[bridge] V={bridge.V}  d_t5={bridge.d_t5}  T5 vocab={bridge.t5.config.vocab_size}")

    # T5 pad id (T5 uses pad_token_id 0 by default).
    pad_id = bridge.tokenizer.pad_token_id
    if pad_id is None:
        pad_id = bridge.t5.config.pad_token_id

    # ---- Datasets ----
    train_base = InATToSeqDataset(args.data_root, args.dataset, "train",
                                   history_max_len=args.history_max_len)
    val_base   = InATToSeqDataset(args.data_root, args.dataset, "val",
                                   history_max_len=args.history_max_len)
    train_ds = FrozenIdentifierDataset(train_base, cache,
                                        max_input_len=args.max_input_len,
                                        max_target_len=args.max_target_len)
    val_ds   = FrozenIdentifierDataset(val_base,   cache,
                                        max_input_len=args.max_input_len,
                                        max_target_len=args.max_target_len)
    coll = make_collate(pad_id=pad_id)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                               num_workers=args.num_workers, collate_fn=coll,
                               pin_memory=True)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False,
                               num_workers=args.num_workers, collate_fn=coll,
                               pin_memory=True)
    # Separate, small-batch loader for beam-search R@5 (LETTER's test_batch_size
    # convention). Beam search materialises (B × num_beams × seq × vocab) logits
    # so memory scales linearly in batch — keep this loader's batch <= ~16.
    eval_loader = DataLoader(val_ds,   batch_size=args.quick_eval_batch, shuffle=False,
                              num_workers=args.num_workers, collate_fn=coll,
                              pin_memory=True)
    print(f"[data] train={len(train_ds)}  val={len(val_ds)}  "
          f"train_batch={args.batch_size}  eval_batch={args.quick_eval_batch}")

    # ---- Optimizer (T5 only — bridge.tokenizer is just a vocab adapter) ----
    optim = torch.optim.AdamW(bridge.t5.parameters(), lr=args.lr_t5,
                               weight_decay=args.weight_decay)

    # ---- LR scheduler: linear warmup + (cosine | constant) decay (LETTER) ----
    total_steps   = args.total_epochs * len(train_loader)
    warmup_steps  = max(1, int(args.warmup_ratio * total_steps))
    def _lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return step / max(1, warmup_steps)
        if args.lr_scheduler == "constant":
            return 1.0
        # cosine
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optim, _lr_lambda)
    print(f"[sched] {args.lr_scheduler}  warmup_steps={warmup_steps}/{total_steps}  "
          f"(warmup_ratio={args.warmup_ratio})  weight_decay={args.weight_decay}")

    # ---- Resume ----
    args.ckpt_dir.mkdir(parents=True, exist_ok=True)
    latest_ckpt = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage2{tag}.latest.pth"
    start_epoch = 0
    history: list[dict] = []
    if args.resume and latest_ckpt.exists():
        print(f"[resume] {latest_ckpt}")
        ck = torch.load(latest_ckpt, map_location=device, weights_only=False)
        bridge.t5.load_state_dict(ck["t5_state"])
        optim.load_state_dict(ck["optim_state"])
        history = ck.get("history", [])
        start_epoch = ck.get("next_epoch", len(history))
        print(f"[resume] continuing from epoch {start_epoch}")

    # ---- Build the item trie once (frozen identifiers, no rebuild needed) ----
    print("[trie] building from cache (frozen identifiers)...")
    trie = ItemTrie.from_identifiers(cache["item"].values(), cache["item"].keys())
    eoi_id = int(cache["cfg"]["special_ids"]["<EOI>"])
    print(f"  trie: {trie.n_items} items, max_depth={trie.max_depth}, "
          f"collisions={trie.collision_count()}")

    # ---- Train loop (T5 only, L_gen) + R@5 early stopping ----
    args.results_dir.mkdir(parents=True, exist_ok=True)
    log_path = args.results_dir / f"inatto-{args.dataset}-{args.seed}.stage2{tag}.log.json"
    best_ckpt_path = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage2{tag}.best.pth"
    best_R5 = -1.0
    patience_counter = 0

    bridge.t5.to(device)
    for epoch in range(start_epoch, args.total_epochs):
        t0 = time.time()
        bridge.t5.train()
        losses = []
        for batch in tqdm(train_loader, desc=f"ep {epoch}", leave=False):
            input_ids = batch["input_ids"].to(device, non_blocking=True)
            attn      = batch["attention_mask"].to(device, non_blocking=True)
            labels    = batch["labels"].to(device, non_blocking=True)
            out = bridge.t5(input_ids=input_ids, attention_mask=attn, labels=labels)
            optim.zero_grad()
            out.loss.backward()
            optim.step()
            scheduler.step()
            losses.append(out.loss.item())

        # Quick val L_gen (cheap monitor every epoch)
        bridge.t5.eval()
        with torch.no_grad():
            val_losses = []
            for i, batch in enumerate(val_loader):
                if i >= 50: break
                input_ids = batch["input_ids"].to(device); attn = batch["attention_mask"].to(device)
                labels    = batch["labels"].to(device)
                out = bridge.t5(input_ids=input_ids, attention_mask=attn, labels=labels)
                val_losses.append(out.loss.item())
            val_loss = float(np.mean(val_losses)) if val_losses else 0.0

        log = {
            "epoch": epoch,
            "epoch_time": time.time() - t0,
            "train_loss": float(np.mean(losses)),
            "val_loss":   val_loss,
        }

        # ---- Early stopping check (val R@5 every args.eval_every epochs) ----
        do_r5 = ((epoch + 1) % args.eval_every == 0) or (epoch + 1 == args.total_epochs)
        if do_r5:
            t1 = time.time()
            R5 = quick_eval_R5(
                bridge.t5, eval_loader, trie, eoi_id, device,
                max_samples=args.quick_eval_users,
                beam_width=args.quick_eval_beam,
                K=5,
            )
            log["val_R5"] = R5
            log["r5_eval_time"] = time.time() - t1
            improved = R5 > best_R5
            if improved:
                best_R5 = R5
                patience_counter = 0
                torch.save({
                    "t5_state":   bridge.t5.state_dict(),
                    "epoch":      epoch,
                    "val_R5":     R5,
                    "history":    history + [log],
                    "dataset":    args.dataset,
                    "seed":       args.seed,
                    "cache_path": str(cache_path),
                }, best_ckpt_path)
            else:
                patience_counter += 1

        # Print line (train_L_gen always, R@5 only when measured)
        msg = (f"[ep {epoch:>2}/{args.total_epochs}] "
               f"train_L_gen={log['train_loss']:.4f}  val_L_gen={log['val_loss']:.4f}  "
               f"({log['epoch_time']:.1f}s)")
        if do_r5:
            star = " ★" if improved else ""
            msg += (f"  | val_R@5={log['val_R5']:.4f} (best={best_R5:.4f}{star}, "
                    f"patience {patience_counter}/{args.patience}, "
                    f"r5_eval {log['r5_eval_time']:.1f}s)")
        print(msg)

        history.append(log)
        Path(log_path).write_text(json.dumps(history, indent=2))

        # Per-epoch rolling checkpoint (latest, for crash resume)
        torch.save({
            "t5_state":    bridge.t5.state_dict(),
            "optim_state": optim.state_dict(),
            "history":     history,
            "next_epoch":  epoch + 1,
            "best_R5":     best_R5,
            "patience_counter": patience_counter,
        }, latest_ckpt)

        # Early stopping termination
        if do_r5 and patience_counter >= args.patience:
            print(f"[early stop] R@5 did not improve for {args.patience} evals; "
                  f"stopping at epoch {epoch}. best R@5 = {best_R5:.4f}")
            break

    # ---- Final checkpoint ----
    ckpt = args.ckpt_dir / f"inatto-{args.dataset}-{args.seed}.stage2{tag}.pth"
    torch.save({
        "t5_state":   bridge.t5.state_dict(),
        "tokenizer_vocab_size": bridge.t5.config.vocab_size,
        "history":    history,
        "dataset":    args.dataset,
        "seed":       args.seed,
        "stage":      2,
        "cache_path": str(cache_path),
        "best_R5":    best_R5,
    }, ckpt)
    print(f"[done] stage-2 final checkpoint -> {ckpt}")
    if best_R5 > 0:
        print(f"[done] best-R@5 checkpoint -> {best_ckpt_path}  (val R@5 = {best_R5:.4f})")


if __name__ == "__main__":
    main()
