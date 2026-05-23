# InATTo — Implementation

Reference implementation of **InATTo** (Information-Aware Textual
Tokenization for Generative Recommendation).

The two-stage pipeline:

```
Stage 0 (data) ──► Stage 1 (tokenizer, BPR + SSW) ──► identifier_cache ──►
                                                       │
                                                       ▼
                                       Stage 2 (RPG-style generative head)
                                                       │
                                                       ▼
                                              full-ranking R@K / N@K
```

Full method specification with hyperparameters and per-module details lives
in **[`METHOD.md`](METHOD.md)**. This README is the quick-start guide for running the
pipeline.

---

## Repository layout

```
InATTo_impl/
├── README.md                  ← you are here
├── pixi.toml / pixi.lock      pixi environment (matches the paper)
├── requirements.txt           non-pixi fallback (PyTorch + transformers + ...)
├── .env.example               LLM API keys template (only needed for data prep)
│
├── assets/                    word_frequency_list_60000_English.xlsx, ...
├── data/                      per-dataset preprocessed pickles (see below)
├── LLMs/                      MiniLM-L6-v2, t5-small (huggingface snapshots)
├── checkpoints/               LightGCN backbones + tokenizer + RPG checkpoints
│
├── inatto/                    main package
│   ├── e2e_model.py           InATToE2E (Stage 1 wrapper)
│   ├── tokenizer.py           InATToTokenizer (signal extraction + RQ)
│   ├── modules/
│   │   ├── codebook.py        Frozen-C₀ + trainable W_c (SimVQ-style)
│   │   ├── satp.py            Sparse-aware text propagation
│   │   ├── reliability.py     δ (CF-text gap) signal
│   │   ├── encoder.py         Dual-branch aspect encoder + Ep MLP
│   │   ├── depth_mask.py      STE depth mask (variable per aspect)
│   │   ├── rq.py              Residual quantization with cross-aspect exclusion
│   │   ├── descriptor.py      Frozen MiniLM descriptor head (FACE-style)
│   │   ├── alignment.py       Per-side InfoNCE (h_d ↔ h_raw)
│   │   ├── ui_alignment.py    Cross-side user↔item InfoNCE
│   │   ├── ssw.py             S2WTM-style spherical sliced Wasserstein
│   │   ├── ste_bridge.py      Codeword → T5-vocab token bridge
│   │   └── id_builder.py      Identifier assembly (USER_BOS / aspect / EOI)
│   └── backbone/lightgcn.py
│
├── data_utils/                adj builder, sequence loader, batch collate
├── trainer/                   phase scheduler, Stage-1 trainer
├── generative/
│   ├── identifier_extractor.py
│   ├── trie.py                trie for legacy autoregressive eval
│   └── beam_search.py
│
└── scripts/
    01_prepare_data.py         RLMRec-format → InATTo per-dataset
    02_encode_raw_item_text.py
    03_build_profile_batch.py  GPT batch input
    04_submit_profile_batch.py
    05_collect_profile_batch.py
    06_encode_profile_text.py
    07_train_lightgcn.py       BPR pretraining
    08a_train_stage1_bpr.py    ★ Stage 1 (main path)
    08b_train_stage2.py          (legacy T5 autoregressive — kept for ablation)
    08c_train_stage2_rpg.py    ★ Stage 2 (main path, RPG-style parallel)
    08_train_face.py           FACE baseline adapter
    09_eval_face.py
    10_sparsity_group_eval.py  per-sparsity-quantile R@K analysis
```

---

## Reproducing the main result (Toys, the best so far)

Assumes data has already been prepared (`scripts/01_*` through `scripts/06_*`),
the LightGCN backbone is pretrained
(`checkpoints/lightgcn/lightgcn-toys-2023.pth`), and MiniLM-L6-v2 +
T5-small live in `LLMs/`.

### Stage 1 — Tokenizer (BPR + SSW, ~17 min on a single GPU)

```bash
pixi run -- python scripts/08a_train_stage1_bpr.py \
    --dataset toys --cuda 0 \
    --total_epochs 20 --batch_size 512 \
    --ablate_adaptive_target \
    --ssw_weight 0.1 \
    --tag ssw01
```

Outputs:

- `checkpoints/inatto/inatto-toys-2023.stage1bpr.ssw01.pth`
- `data/toys/identifier_cache.2023.bpr.ssw01.pkl`

### Stage 2 — RPG generative head (~25–30 min)

```bash
pixi run -- python scripts/08c_train_stage2_rpg.py \
    --dataset toys --cuda 0 \
    --cache_suffix bpr.ssw01 --output_tag wd05 \
    --weight_decay 0.05 \
    --total_epochs 150 --batch_size 256 --eval_batch_size 64 \
    --lr 3e-3 --history_max_len 50 \
    --eval_every 1 --patience 20
```

Outputs:

- `checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.wd05.best.pth`
  (val-R@5 best; early-stopping with patience 20)

### Test evaluation (full ranking)

A small inline driver works for any best.pth — see `scripts/10_sparsity_group_eval.py`
for the supported entry point or copy the snippet below.

```python
import torch, pickle, importlib.util
from pathlib import Path
from torch.utils.data import DataLoader
from data_utils.seq_loader import InATToSeqDataset

spec = importlib.util.spec_from_file_location('rpg', 'scripts/08c_train_stage2_rpg.py')
mod  = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)

device = torch.device('cuda:0')
with open('data/toys/identifier_cache.2023.bpr.ssw01.pkl','rb') as f:
    cache = pickle.load(f)
grid, mask = mod.parse_cache_to_grid(
    cache, int(cache['cfg']['n_aspects']), int(cache['cfg']['L_max']))

model = mod.InATToRPG(
    grid, mask,
    int(cache['cfg']['n_aspects']),
    int(cache['cfg']['L_max']),
    int(cache['cfg']['V']),
    n_embd=256, n_layer=2, n_head=4, n_inner=1024,
    dropout=0.5, max_seq_len=50, temperature=0.05,
).to(device)
ck = torch.load(
    'checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.wd05.best.pth',
    map_location=device, weights_only=False)
model.load_state_dict(ck['model_state'])

test_base = InATToSeqDataset(Path('data'), 'toys', 'test', history_max_len=50)
test_ds   = mod.HistoryItemDataset(test_base, 50)
loader    = DataLoader(test_ds, batch_size=64, shuffle=False, num_workers=4,
                        collate_fn=mod.make_collate(50))
print(mod.eval_recall_ndcg(model, loader, device, Ks=(5,10,20)))
```

---

## Current best (test, full ranking, leave-one-out, seed 2023)

| Dataset | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 |
|---|---|---|---|---|---|---|
| **Toys** | **0.0550** | **0.0793** | **0.1118** | **0.0382** | **0.0460** | **0.0542** |
| **Beauty** | **0.0507** | **0.0738** | **0.1044** | **0.0352** | **0.0427** | **0.0503** |
| Sports / Yelp | _in progress_ | | | | | |

The Toys row is computed from `wd=0.05` over the `bpr.ssw01` cache; Beauty
is the same recipe trained off of `data/beauty/identifier_cache.2023.bpr.ssw01.pkl`.

---

## Sparsity-group analysis (paper §4 main analysis)

```bash
pixi run -- python scripts/10_sparsity_group_eval.py \
    --dataset toys --cuda 0 \
    --cache_suffix bpr.ssw01 \
    --ckpt checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.wd05.best.pth \
    --n_groups 3 --method_label "InATTo"
```

Splits test items into sparse / medium / rich quantiles by per-item rho
(text-token entropy) and reports R@K / N@K within each group. This is the
operational verification of the "information-aware adaptive quantization
for text sparsity" claim.

---

## Data preparation

See `data/<dataset>/` after running `scripts/01_*` through `scripts/06_*`.
Required files per dataset:

```
data/<ds>/
  item2id.pkl
  user2id.pkl
  trn_mat.pkl / val_mat.pkl / tst_mat.pkl
  user_train_history.pkl
  itm_text.pkl, itm_text_embeds.pkl
  itm_prf.pkl, itm_emb_np.pkl, itm_rho.pkl
  usr_prf.pkl, usr_emb_np.pkl, usr_rho.pkl
  stats.json
```

Profile generation (`itm_prf.pkl` / `usr_prf.pkl`) uses GPT-4o-mini through
the OpenAI batch API; the relevant `.env.example` lists the keys to set.

---

## Notes / conventions

- All hyperparameters in `METHOD.md` §2.13 and §3.7 are the ones used in
  the reported numbers above.
- `weight_decay=0.05` is the Stage-2 sweet spot found by sweeping
  `0 → 0.01 → 0.03 → 0.05` on Toys; the same value transfers to the other
  three datasets (Beauty already confirmed, Sports/Yelp in progress).
- `08b_train_stage2.py` (T5 autoregressive) is kept *only* for the
  `w/o RPG` ablation; the main path is `08c_train_stage2_rpg.py`.
- FACE adapter scripts (`08_train_face.py`, `09_eval_face.py`,
  `inatto/face_*.py`) are baseline-only and untouched from the FACE upstream.
