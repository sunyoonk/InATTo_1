# InATTo — Method Specification

End-to-end detailed specification of the InATTo pipeline as of the
current Toys best (Stage 2 `wd=0.05`).

---

## 0. Overview Pipeline

```
[Raw data]
  user-item sequence  ──┐
  item text + categories ─┤
  user/item profile (GPT) ─┤
  pretrained LightGCN ────┘
                          │
            ┌─────────────┴───────────────┐
            ▼                             ▼
     [Stage 0: Pre-data]            [Pretrained LightGCN]
   MiniLM token embeddings        BPR로 사전 학습된 ckpt
   COCA60000 vocab filter         (Stage 1에서 unfreeze)
            │
            ▼
   ┌───────────────────────────────────┐
   │   STAGE 1: InATTo Tokenizer       │
   │   (BPR + recon + Q + align + ui   │
   │    + rate + SSW)                  │
   └───────────────────────────────────┘
            │
            ▼
   identifier_cache.<seed>.bpr.ssw01.pkl
   (item[iid] / user[uid] = T5 token sequences)
            │
            ▼
   ┌───────────────────────────────────┐
   │   STAGE 2: RPG Generative Rec     │
   │   (parallel MTP + variable mask)  │
   └───────────────────────────────────┘
            │
            ▼
   [Test] full ranking → R@K, N@K
```

---

## 1. Stage 0 — Pre-data Components

### 1.1 Vocabulary (codebook 단어 집합)

```python
# inatto/modules/codebook.py: filter_minilm_vocabulary()
1. MiniLM-L6-v2 tokenizer.get_vocab()    # 30,522 BERT subwords
2. Filter: lowercase, [a-z]+ only, no ##subword
3. Intersect with COCA-60000 word list  (assets/word_frequency_list_60000_English.xlsx)
4. ──→ V = 9,338 codewords
```

### 1.2 Pre-extracted features (`data/<ds>/`)

| file | shape | 출처 |
|---|---|---|
| `itm_text_embeds.pkl` | (n_items, 384) | item title+categories → MiniLM |
| `itm_emb_np.pkl` | (n_items, 384) | item profile (GPT-4o-mini) → MiniLM |
| `usr_emb_np.pkl` | (n_users, 384) | user profile (GPT-4o-mini) → MiniLM |
| `itm_rho.pkl` | (n_items,) | item profile token entropy ∈ [0,1] |
| `usr_rho.pkl` | (n_users,) | user profile token entropy ∈ [0,1] |
| `lightgcn-<ds>-<seed>.pth` | (n_users+n_items, 256) | pretrained LightGCN |

### 1.3 LightGCN backbone (`inatto/backbone/lightgcn.py`)

```python
n_users, n_items = dataset.shape
embedding_size = 256
layer_num      = 3

E^(0) = [user_embeds (n_users,256); item_embeds (n_items,256)]
E^(k+1) = A_norm @ E^(k)
E_final = sum_{k=0..3} E^(k)
score(u, i) = E_final[u] · E_final[n_users + i]

BPR loss = -log σ(score(u, pos) - score(u, neg))
```

Pre-trained for `--epochs 1000`, then loaded as initialization for Stage 1.

---

## 2. Stage 1 — InATTo Tokenizer

### 2.1 Codebook (SimVQ, `inatto/modules/codebook.py`)

```python
# Frozen part
codebook_raw  = MiniLM.input_embeddings[token_ids]    # (V=9338, d_llm=384)
register_buffer('codebook_raw', codebook_raw)

# Trainable part
W_c = nn.Linear(d_llm=384, d_aspect=256, bias=True)

def codebook(self):
    return self.W_c(self.codebook_raw)               # (9338, 256)
```

**SimVQ 정신**: C₀ 고정 (semantic anchor), W_c만 학습 (latent geometry).

### 2.2 SATP — Sparse-Aware Text Propagation (`inatto/modules/satp.py`)

per mode (user or item):

```python
# Pre-computed cached buffers
self.z      = z_lgcn               # (N, 256)  CF embedding (live with BPR)
self.h_txt  = MiniLM embedding     # (N, 384)
self.rho    = entropy              # (N,)
self.nn_idx = top-K=10 neighbors   # (N, 10)   in CF space, cosine

# Forward(ids):
z_b   = z[ids]
h_b   = h_txt[ids]
rho_b = rho[ids]
nbr   = nn_idx[ids]                       # (B, 10)
z_nbr = z[nbr]; h_nbr = h_txt[nbr]

attn   = softmax(z_b · z_nbr / sqrt(256), dim=1)        # (B, 10)
h_bar  = attn · h_nbr                                   # (B, 384)
h_hat  = rho_b · h_b + (1-rho_b) · h_bar                # (B, 384)
return (h_hat, rho_b, z_b)
```

K = 10, neighbors rebuilt every 5 epochs (`--rebuild_neighbors_every`).

### 2.3 Reliability + δ (`inatto/modules/reliability.py`)

```python
W_proj = nn.Linear(384, 256)               # text → CF space projection
h_cf = W_proj(h_hat)                       # (B, 256)
r_ortho = z - h_cf                          # CF residual
δ = ||r_ortho||                             # (B,) — info-content score
δ_k = e_k · r_ortho (e_k = E1 weight row k) # (B, n_aspects=8) — per-aspect projection
```

δ 의미: CF가 텍스트로 *설명되지 않는* 부분의 크기. 우리 information-aware signal.

### 2.4 Dual-Branch Encoder (`inatto/modules/encoder.py`)

```python
n_aspects = 8       # FACE word_num
d_aspect  = 256

# E1: aspect-wise projection
e_k    = z · W_E1[k]                         # (B, n=8, 256)
W_E1   = nn.Parameter(8, 256, 256)           # trainable

# E2: TransformerEncoder (refine aspects)
TransformerEncoder(d_model=256, nhead=1, num_layers=1, dropout=0.0)
z_aspect = E2(e_k)                            # (B, 8, 256)

# Ep: Importance MLP
phi_k = sigmoid(MLP([e_k, rho_b, δ_k_std]))  # (B, 8) ∈ [0,1]
MLP   = [Linear(d+1+1 → 64) - ReLU - Linear(64 → 1)]
```

φ_k가 각 aspect의 "활성 depth"를 결정.

### 2.5 Depth Mask — STE (`inatto/modules/depth_mask.py`)

```python
L_max = 4   # 각 aspect 최대 residual depth

s     = phi_k * L_max                       # (B, 8) ∈ [0, 4]
l     = 1..L_max                            # level indices

# Forward (hard binary)
m_hard[..., l-1] = (s >= l)
m_hard[..., 0]   = 1                         # 최소 1 codeword 보장

# Backward (smooth surrogate, sigmoid)
m_smooth[..., l-1] = sigmoid(alpha_ste * (s - l + 0.5))
alpha_ste = 2.0                              # VRVQ convention

# STE combine
m = m_smooth + (m_hard - m_smooth).detach()  # (B, 8, 4)
```

### 2.6 Residual Quantization (`inatto/modules/rq.py`)

```python
commit_beta        = 0.25                    # FACE convention
full_codebook_rate = 0.25                    # VRVQ: 25% batch는 m=1 강제

C   = codebook.codebook()                    # (V=9338, 256)
C_n = F.normalize(C, dim=-1)

For batch b, aspect k:
    r_k^(1) = z_aspect[b, k]
    used_codes = set()                       # cross-aspect exclusion
    For l in 1..L_max:
        # cosine argmin
        dist = 2 - 2 * (r_k^(l) / ||r_k^(l)||) · C_n          # (V,)
        dist[used_codes] = inf               # exclusion
        c_k^(l) = argmin(dist)
        used_codes.add(c_k^(l))
        r_k^(l+1) = r_k^(l) - C[c_k^(l)]

# Masked sum
z_hat = sum_{l=1..L_max} m_l * C[c_k^(l)]    # (B, 8, 256)

# STE attach (for downstream gradient)
z_hat_st = z_aspect + (z_hat - z_aspect).detach()

# Q loss (commit + codebook commit, masked)
L_Q = sum_{b,k,l} m_l * ( ||C[c_k^(l)] - sg(z_aspect[b,k])||²
                        + 0.25 * ||z_aspect[b,k] - sg(C[c_k^(l)])||² )
```

### 2.7 CF-Space Decoder (`inatto/modules/encoder.py:CFSpaceDecoder`)

```python
TransformerDecoder(d=256, layers=1, heads=1, dropout=0)
reverse_linear = nn.Linear(8 * 256, d_cf=256)

decoded = reverse_linear(transformer_decoder(z_hat))   # (B, 256)
L_recon = MSE(decoded, z.detach())
```

### 2.8 Descriptor — FACE convention (`inatto/modules/descriptor.py`)

```python
W_c_reverse = pseudo_inverse(W_c)            # d_aspect → d_llm
word_embs   = W_c_reverse(z_hat_st)          # (B, 8, 384)

# Build sentence: "[BOS] This item is described by w1, w2, ..., w8 [EOS]"
prompt_tokens = MiniLM.tokenize(prompt_template)
combined      = cat([prompt, word_emb_with_commas])
h_d           = MiniLM.encode_embeddings(combined)   # (B, 384)
```

Prompt template per-mode:
- item: `"This [domain] item can be described as: "`
- user: `"This user prefers items described as: "`

`[domain]` ∈ {`cosmetic`, `toy`, `sport item`, `restaurant`}.

### 2.9 Alignment (`inatto/modules/alignment.py`)

```python
W_align = nn.Linear(d_txt=384, d_llm=384)    # text projection

# Adaptive target — ablate_adaptive_target=True (FACE convention, our best)
h_adp = h_raw

# (Off:  h_adp = rho * h_raw + (1-rho) * W_align(h_hat_txt))

# Per-side InfoNCE
L_align_user = InfoNCE(h_d_user, h_adp_user, tau=0.07)
L_align_item = InfoNCE(h_d_item, h_adp_item, tau=0.07)
L_align      = L_align_user + L_align_item
```

### 2.10 UI Alignment — 우리 NEW (`inatto/modules/ui_alignment.py`)

```python
L_ui = InfoNCE(z_hat_st_user, z_hat_st_item_pos, tau_ui=0.07)
# user의 aspect representation ↔ 그 user의 positive item aspect representation
```

### 2.11 Rate Loss (VRVQ style)

```python
# n_full = batch * 0.25  (last n_full samples 강제 m=1)
# rate loss는 첫 n_imps = B - n_full 에서만 계산
L_rate = phi[:n_imps].mean()                 # push toward 0
```

### 2.12 SSW — S2WTM Spherical (`inatto/modules/ssw.py`)

```python
# z_aspect (B, 8, 256) flatten → (B*8, 256)
x_unit = F.normalize(x, dim=-1)               # → S^{255}

# Stiefel V_{d,2} via QR
Z = randn(M=50, d=256, 2)
U, _ = torch.linalg.qr(Z)                     # (50, 256, 2)

# Project each x to each 2-plane
Xps = U^T · x                                 # (50, N, 2)
Xps = F.normalize(Xps, dim=-1)                # → S^1

# Convert to angle in [0, 1]
angles = (atan2(-y, -x) + π) / (2π)           # (50, N)

# Closed-form W2² vs Uniform([0, 1]) per projection
u_sorted = sort(angles, -1)
cpt1   = mean(u²)
mean_  = mean(u)
ns_n2  = arange(n-1, -n, -2) / n²
cpt2   = sum(ns_n2 * u_sorted)
W2²    = cpt1 - mean_² + cpt2 + 1/12

L_ssw = mean(W2²) over M projections
```

### 2.13 Stage 1 Loss & Schedule

**Final total** (per batch):

```
L_total = λ_BPR · L_BPR
        + λ_recon · L_recon        (1.0)
        + λ_Q     · L_Q            (1.0)
        + λ_align · L_align        (0.5)
        + λ_ui    · L_ui           (0.1)
        + λ_rate  · L_rate         (0.0005)
        + λ_SSW   · L_SSW          (0.1)
```

| Hyperparameter | Value |
|---|---|
| λ_BPR | 1.0 (FACE convention) |
| λ_recon | 1.0 |
| λ_Q (commit) | 1.0 with β = 0.25 |
| λ_align | 0.5 |
| λ_ui | 0.1 |
| λ_rate | 0.0005 (낮음 — phi saturation 의도적 허용; depth는 δ correlation으로 information-aware) |
| λ_SSW | 0.1 |
| optimizer | AdamW (lr=1e-3 tokenizer, 1e-3 LightGCN; wd=2e-4 LGCN, 0 tokenizer) |
| epochs | 20 |
| batch_size | 512 |
| Stage 1 학습 시간 (Toys) | ~17분 |

**Ablation flags** (current best):
- `ablate_adaptive_target = True` (h_adp = h_raw, FACE convention) ★
- `ablate_satp = False` (SATP 사용)
- `ablate_variable_depth = False` (variable depth 사용)
- `use_vrvq_mask = False` (STE sigmoid surrogate 사용)

### 2.14 Stage 1 Output

`identifier_cache.<seed>.bpr.ssw01.pkl`:

```python
{
  'item': {iid: [t5_token_ids]},   # variable length 24-41 (Toys)
  'user': {uid: [t5_token_ids]},
  'item_lengths': np.array,
  'user_lengths': np.array,
  'cfg': {
    'n_aspects': 8, 'L_max': 4,
    'V': 9338, 'd_t5': 512,
    'special_ids': {'<USER_BOS>': 32100, ..., '<EOI>': 32105},
    'code_to_t5': [32106, ..., 41443],   # codebook idx i → T5 vocab id
    'variant': 'bpr_stage1'
  }
}
```

---

## 3. Stage 2 — RPG Generative Recommender

### 3.1 Cache → Grid (`scripts/08c_train_stage2_rpg.py:parse_cache_to_grid`)

```python
# 각 item의 variable T5 sequence → (8, 4) grid + bool mask
# Format: codewords separated by <EOA>, ended by <EOI>
For item:
    Parse 8 aspect blocks separated by <EOA>
    grid[i, k, l] = codeword_idx (1-indexed; 0 = pad)
    mask[i, k, l] = 1 if active, 0 if padded

# Final shape
item_id2tokens (n_items, 8, 4)  → flatten to (n_items, 32)
item_id2mask   (n_items, 8, 4)  → flatten to (n_items, 32)
```

Active position 분포 (Toys): min = 15, max = 32, mean = 28.9.

### 3.2 Model — InATToRPG (`scripts/08c_train_stage2_rpg.py:InATToRPG`)

```python
GPT2Config:
    vocab_size            = V + 2 = 9340   # 0 = pad, 1..V = codeword, V+1 = eos
    n_positions           = 64
    n_embd                = 256
    n_layer               = 2
    n_head                = 4
    n_inner               = 1024
    activation_function   = "gelu_new"
    resid_pdrop           = 0.0
    embd_pdrop            = 0.5
    attn_pdrop            = 0.5
    layer_norm_epsilon    = 1e-12

self.gpt2 = GPT2Model(config)

self.n_total_pos = 32   # n_aspects × L_max
self.pred_heads  = nn.ModuleList([ResBlock(256) for _ in range(32)])

class ResBlock(nn.Module):
    linear = Linear(256, 256), zero-init
    act    = SiLU
    forward(x) = x + act(linear(x))
```

Total params: ~6M (T5-small의 1/10).

### 3.3 Forward — Item-level Pooling

```python
# History items: input_ids (B, S=50)
tokens = item_id2tokens[input_ids]                  # (B, 50, 32)
mask   = item_id2mask[input_ids]                    # (B, 50, 32)

embs   = gpt2.wte(tokens)                           # (B, 50, 32, 256)
denom  = mask.sum(-1, keepdim=True).clamp_min(1)
item_emb = (embs * mask.unsqueeze(-1)).sum(-2) / denom  # (B, 50, 256)

# GPT-2 forward
outputs = gpt2(inputs_embeds=item_emb, attention_mask=attn)
last    = outputs.last_hidden_state[:, last_valid_pos]   # (B, 256)
```

### 3.4 Parallel Multi-token Prediction

```python
# 32 parallel heads
final_states = stack([pred_heads[i](last) for i in range(32)], dim=1)  # (B, 32, 256)

# Per-position logits (cosine, shared codebook)
states_n   = F.normalize(final_states, dim=-1)
codebook_n = F.normalize(gpt2.wte.weight, dim=-1)
logits     = states_n @ codebook_n.T / temperature       # (B, 32, 9340)
temperature = 0.05
```

### 3.5 Loss — Per-position CE with Mask

```python
target_tokens = item_id2tokens[labels]              # (B, 32)
target_mask   = item_id2mask[labels]                # (B, 32)

# CE per position
loss_per_pos = F.cross_entropy(
    logits.reshape(-1, 9340),
    target_tokens.reshape(-1),
    reduction='none'
).view(B, 32)

# Mask: inactive positions의 loss는 카운트 안 함
L_gen = (loss_per_pos * target_mask).sum() / target_mask.sum().clamp_min(1)
```

### 3.6 Inference — Full Ranking

```python
# rank_all_items
log_p = F.log_softmax(logits, dim=-1)               # (B, 32, 9340)

For each item i:
    log_p_at_item_token = gather(log_p, item_id2tokens[i])  # (B, 32)
    score = (log_p_at_item_token * item_id2mask[i]).sum(-1) / item_id2mask[i].sum().clamp_min(1)
# scores (B, n_items)

# Mask history (already-seen)
scores[seen_items_per_user] = -inf
scores[:, 0]                = -inf            # pad item

# Top-K
preds = scores.topk(K).indices
```

### 3.7 Stage 2 Training — Optimizer + Schedule

```python
optim = torch.optim.AdamW(
    model.parameters(),
    lr           = 3e-3,          # RPG default
    weight_decay = 0.05,          # ★ Toys best (from sweep 0 → 0.01 → 0.03 → 0.05)
)

total_steps  = 150 epochs × 428 iter/ep ≈ 64,200
warmup_steps = 0.01 × total_steps = 642
scheduler    = cosine warmup → 0 over remaining

batch_size       = 256
eval_batch_size  = 64
history_max_len  = 50
quick_eval_users = 1000          # subset for in-training R@5
quick_eval_beam  = 20            # kept as legacy arg (full ranking used)

max_epochs       = 150
patience         = 20            # early stop based on val R@5
eval_every       = 1             # measure R@5/N@K each epoch
max_grad_norm    = 1.0
```

### 3.8 Stage 2 Output

```
checkpoints/inatto/inatto-<ds>-<seed>.stage2rpg.bpr.ssw01.<tag>.best.pth
  ├── model_state: GPT-2 + 32 ResBlocks
  ├── epoch: 109   (Toys wd05 best)
  ├── best_R5: 0.0709 (val)
  └── cache_path: stage 1 cache
```

---

## 4. Evaluation Protocol

```
Split:           leave-one-out
                 train  = items[:-2]
                 val    = items[:-1], target = items[-2]
                 test   = items,       target = items[-1]

Ranking:         full item ranking (no negative sampling)
Mask:            history items 제외 (TIGER/LETTER convention)
Metrics:         R@5, R@10, R@20, N@5, N@10, N@20
Selection:       val R@5 best epoch → best.pth 저장
Final eval:      best.pth로 test 평가 (한 번)
```

---

## 5. Current Best — Toys (test)

| Metric | Value | vs baseline (wd=0) | vs LETTER | vs GRAM |
|---|---|---|---|---|
| **R@5** | **0.0550** | +19.6% | +5.8% | -21.2% |
| **R@10** | **0.0793** | +14.4% | +1.7% | -16.9% |
| **R@20** | **0.1118** | +12.0% | +4.5% | — |
| **N@5** | **0.0382** | +20.1% | +15.8% | -24.7% |
| **N@10** | **0.0460** | +17.3% | +2.2% | -22.0% |
| **N@20** | **0.0542** | +15.6% | +4.2% | — |

**Validation** at best epoch (ep109): R@5 = 0.0709.

Beats: GRU4Rec, SASRec, TIGER, LETTER, IDGenRec, RPG paper number.
Below: GRAM (rich-text-strong baseline).

---

## 6. Hyperparameter 빠른 참조

| Stage | Hyperparam | Value |
|---|---|---|
| **Codebook** | V | 9338 |
| | d_llm | 384 (MiniLM) |
| | d_aspect | 256 |
| **Tokenizer** | n_aspects | 8 |
| | L_max | 4 |
| | K_neighbors | 10 |
| | alpha_ste | 2.0 |
| | commit_beta | 0.25 |
| | full_codebook_rate | 0.25 |
| | tau_align / tau_ui | 0.07 / 0.07 |
| **Stage 1 losses** | λ_BPR | 1.0 |
| | λ_recon | 1.0 |
| | λ_Q | 1.0 |
| | λ_align | 0.5 |
| | λ_ui | 0.1 |
| | λ_rate | 0.0005 |
| | λ_SSW | 0.1 |
| **Stage 1 training** | epoch | 20 |
| | batch | 512 |
| | lr (tokenizer / LGCN) | 1e-3 / 1e-3 |
| | reg_weight (LGCN) | 1e-4 |
| **Stage 2 model** | n_embd | 256 |
| | n_layer | 2 |
| | n_head | 4 |
| | n_inner | 1024 |
| | dropout (embd / attn) | 0.5 / 0.5 |
| | n_pred_heads | 32 (= 8 × 4) |
| | temperature | 0.05 |
| **Stage 2 training** | epoch (max) | 150 |
| | patience | 20 |
| | batch (train / eval) | 256 / 64 |
| | history_max_len | 50 |
| | lr | 3e-3 |
| | **weight_decay** | **0.05** ★ |
| | warmup_ratio | 0.01 |
| | scheduler | cosine |
| | max_grad_norm | 1.0 |

---

## 7. File / Script Map

| Concern | File |
|---|---|
| LightGCN backbone | `inatto/backbone/lightgcn.py` |
| Codebook (SimVQ) | `inatto/modules/codebook.py` |
| SATP | `inatto/modules/satp.py` |
| Reliability + δ | `inatto/modules/reliability.py` |
| Dual-branch encoder + Ep | `inatto/modules/encoder.py` |
| Depth mask STE | `inatto/modules/depth_mask.py` |
| Residual quantization | `inatto/modules/rq.py` |
| Descriptor (frozen MiniLM) | `inatto/modules/descriptor.py` |
| Alignment | `inatto/modules/alignment.py` |
| UI alignment | `inatto/modules/ui_alignment.py` |
| SSW (S2WTM) | `inatto/modules/ssw.py` |
| Stage 1 model | `inatto/e2e_model.py`, `inatto/tokenizer.py` |
| **Stage 1 train (BPR)** | `scripts/08a_train_stage1_bpr.py` |
| **Stage 2 train (RPG)** | `scripts/08c_train_stage2_rpg.py` |
| Identifier extractor | `generative/identifier_extractor.py` |
| Sparsity-group eval | `scripts/10_sparsity_group_eval.py` |
