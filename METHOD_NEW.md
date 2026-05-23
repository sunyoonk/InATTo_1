# InATTo — Paper-Ready Method Specification (★ Gumbel STE v2 verified)

Two-stage **Information-Aware Textual Tokenization** for lightweight,
text-robust generative recommendation.  This document is the
**paper-grade** specification of the *current verified configuration*:
`ssw01` baseline plus a **forward-deterministic Gumbel-STE** depth
mask.  Numbers in §7 are the experimentally verified Toys test result.

> **Loss budget** — Stage 1 trains a 7-term internal objective that is
> presented in §3.11 as **four external blocks** following LETTER's
> aggregation convention (one per design concern).  Inner term
> weights appear in Appendix A.

---

## 0. Pipeline overview

```
Stage 0  (offline, frozen, once-per-dataset)
   raw RLMRec interactions   ──►  LightGCN BPR-pretrained backbone
   item metadata              ──►  MiniLM-L6-v2 frozen encoder
   GPT-4o-mini profiles       ──►  MiniLM + token-entropy ρ
   COCA ∩ MiniLM vocab        ──►  V = 9,338 English-word codebook
                          ▼
Stage 1  Dual InATTo Tokenizer       (joint with LightGCN, ~17 min)
   per side  m ∈ {user, item}  (shared parameters):
       SATP  ─►  Reliability  ─►  E1 + E2 + Ep (importance MLP)
                                        │
                ★ Gumbel-STE v2 depth mask  ◄─┐
                                        │     │  forward-deterministic,
                       Residual Quantizer     │  backward Gumbel-noisy
                       (cross-aspect excl.)   │  → train/test forward identical
                                        │     │
                       CF Decoder  +   Descriptor + Alignment
                          ▼
Stage 2  RPG Generative Recommender       (~25–30 min/dataset)
   GPT-2 small  +  32 parallel ResBlock heads
   weight_decay = 0.05,  dropout = 0.3
                          ▼
   Full-ranking inference        →   R@K, NDCG@K
```

Stages 1 and 2 are trained **separately**.  Stage 2 consumes only the
frozen identifier cache produced at the end of Stage 1.

---

## 1. Notation

| Symbol | Shape | Meaning |
|---|---|---|
| z_u, z_i | (n, 256) | LightGCN-propagated CF embedding (trainable via BPR) |
| h_txt_m | (n, 384) | MiniLM(item title+categories+desc) or MiniLM(user profile) |
| h_raw_m | (n, 384) | MiniLM(GPT-4o-mini profile) — alignment target |
| ρ_m | (n,) ∈ [0, 1] | token-level entropy of the profile |
| K | 8 | number of aspects per entity |
| L_max | 4 | maximum residual quantizer depth |
| V | 9,338 | codebook size (English-word vocab) |
| d_cf, d_aspect, d_llm | 256, 256, 384 | latent dimensions |
| φ_k_m | (B, K) ∈ (0, 1) | per-aspect importance for entity m |
| m_k^(l) | (B, K, L_max) ∈ {0, 1} | depth mask (active levels) |
| c_k^(l) | ∈ [0, V) | l-th codeword chosen for aspect k |
| ẑ_m | (B, K, d_aspect) | quantized aspect representation Σ_l m_l · C[c_l] |

---

## 2. Stage 0 — Data Preparation (frozen)

### 2.1 LightGCN backbone

```
embedding_size = 256          layer_num = 3
BPR pretraining (1,000 epochs):
   L_BPR = -log σ( score(u, i⁺) − score(u, i⁻) )
   score(u, i) = E_final[u] · E_final[n_users + i]
checkpoint: checkpoints/lightgcn/lightgcn-{ds}-{seed}.pth
```

### 2.2 Pre-extracted per-entity features

| File | Shape | Description |
|---|---|---|
| `itm_text_embeds.pkl` | (n_items, 384) | MiniLM(title + categories + description) |
| `itm_emb_np.pkl` / `usr_emb_np.pkl` | (n, 384) | MiniLM(GPT-4o-mini profile) |
| `itm_rho.pkl` / `usr_rho.pkl` | (n,) | token-level entropy of the profile |

### 2.3 Codebook construction (frozen; shared user ↔ item)

```python
vocab_minilm = MiniLMTokenizer.get_vocab()           # 30,522 sub-words
vocab_clean  = { w for w in vocab_minilm
                  if w.isalpha() and w.islower()
                  and not w.startswith("##") }
vocab_coca   = top-60,000 frequency-ranked English words
V            = vocab_clean ∩ vocab_coca               # |V| = 9,338
codebook_raw = MiniLM.input_embeddings()[V]           # (9338, 384), frozen
```

A single linear projection `W_c ∈ ℝ^{256 × 384}` (the SimVQ head) is
the **only** trainable codebook parameter during Stage 1.  Codeword
vectors at training time are

```
C = W_c · codebook_rawᵀ      ∈ ℝ^{V × 256}
```

---

## 3. Stage 1 — Dual InATTo Tokenizer

The same module is instantiated once and forward-passed twice per
batch with `mode ∈ {user, item}`.  All trainable parameters (SATP,
Reliability, Encoder, Codebook, Descriptor, Alignment) are **shared**
across modes; only per-mode buffer tensors (`z, h_txt, ρ, nn_idx`)
differ.

### 3.1 SATP — Sparse-Aware Text Propagation

```
per mode:
   z       : (n, 256)  — LightGCN embedding (TRAINABLE via BPR)
   h_txt   : (n, 384)  — MiniLM embedding (frozen)
   ρ       : (n,)      — token entropy (frozen)
   nn_idx  : (n, K_nn=10) — top-K_nn z-cosine neighbours

forward(ids):
   a_{∗ j} = softmax( z_∗ · z_j / √d_cf ,    j ∈ nn_idx[∗] )
   h_bar   = Σ_j  a_{∗ j} · h_txt[ nn_idx[∗, j] ]
   ĥ       = ρ · h_txt  +  (1 − ρ) · h_bar
```

Neighbours are rebuilt every 5 epochs (cheap top-K refresh on the
live LightGCN embedding).

### 3.2 Reliability — global δ and per-aspect δ_k

```
W_proj  : Linear(384 → 256)
h_cf     = W_proj(ĥ)                                    (B, 256)
coef     = ⟨z, h_cf⟩ / ⟨h_cf, h_cf⟩                    (B, 1)
r_⊥      = z  −  coef · h_cf                            (B, 256)     ★ Gram-Schmidt
δ        = ‖r_⊥‖₂                                         (B,)
δ_k      = ‖W_k · r_⊥‖₂        k = 1..K                  (B, K)
```

The `W_k` matrices are the K rows of the multi-projector `W`
(§3.3.1), shared with the encoder — zero parameter overhead.

### 3.3 Dual-branch encoder + importance subnetwork

#### 3.3.1 E1 — orthogonally-initialised projection

```
W ∈ ℝ^{K × d_aspect × d_cf}        # K independent orthogonal heads
b ∈ ℝ^{K × d_aspect}
e   = einsum("naj,bj→bna", W, z) + b                    (B, K, d_aspect)
```

#### 3.3.2 E2 — Disentangled Transformer

```
single-layer Transformer (n_heads = 1, dropout = 0, batch_first):
z_aspect = TransformerEncoder(e)                         (B, K, d_aspect)
```

#### 3.3.3 Ep — Importance Subnetwork  (★ shared MLP, per-aspect)

```
δ̃_k = (δ_k − μ_batch(δ)) / σ_batch(δ)         ★ batch-standardised
u_k  = [ e_k ; ρ ; δ̃_k ]                       ∈ ℝ^{d_aspect + 2}
φ_k  = sigmoid( MLP_φ(u_k) )                    ∈ (0, 1)
MLP_φ : Linear(d_aspect+2 → 64) − ReLU
      − Linear(64 → 64)            − ReLU
      − Linear(64 → 1)              # last layer zero-initialised → φ ≈ 0.5
```

`φ_k` is **per-item, per-aspect**; MLP weights are shared across
all aspects (only the per-aspect inputs change).

### 3.4 ★ Variable-depth mask — **Gumbel-STE v2** (★ NEW)

The single algorithmic novelty on top of the FACE/VRVQ baseline.

**Motivation.**  A plain sigmoid STE saturates φ → 1 in practice (every
item gets the same length); a naïve Gumbel-Softmax (Jang et al. 2017)
injects noise into *both* forward and backward, which produces a
train-inference *distribution shift* — at test time the deterministic
threshold is no longer matched by the codebook learned for noisy
masks.

**Our fix — forward-deterministic, backward Gumbel-noisy:**

```
s            = φ · L_max                                  (B, K, 1)
ℓ            = (1, 2, …, L_max)                            (1, 1, L_max)
g_ℓ          = −log(−log(U(0, 1)))                        ★ Gumbel noise, TRAINING ONLY

★ m_hard^(l) = 𝟙[ s ≥ ℓ ]              ← no noise (★ train == test)
   m̃^(l)    = sigmoid( (s + g_ℓ − ℓ + 0.5) / τ )        smooth surrogate with noise
   m^(l)    = m̃^(l)  +  (m_hard^(l) − m̃^(l)).detach()  STE: forward = hard, backward = noisy soft
   m^(1)    = 1                                            level 1 always on
              τ = 1.0
```

At inference (`model.eval()`), `g_ℓ` collapses to zero and the
forward returns the standard deterministic hard mask.

**Why this works.**  The forward path is *bit-identical* between
training and inference, so the codebook is no longer learned for a
distribution-shifted training condition.  Yet the backward gradient
still flows through a *Gumbel-perturbed* sigmoid, which acts as a
gradient regulariser on the importance MLP — pushing φ to a wider
distribution that survives the noise.

**Empirically (Toys, vs sigmoid-STE baseline):**
- Codebook utilisation 72.2 % → 74.9 %
- Identifier length std 1.65 → 2.21 (★ stronger spread)
- Test R@5 0.0540 → **0.0552** (★ +2.2 %)
- Test NDCG@5 0.0382 → **0.0389** (★ +1.8 %)

Implementation:
`inatto/modules/depth_mask.py :: depth_mask_gumbel_ste_v2`.

### 3.5 Codebook — SimVQ on a frozen MiniLM vocabulary

```
W_c      : Linear(384 → 256, bias)                       ★ trainable
C        = W_c(codebook_raw)                              (V, 256)
```

The frozen MiniLM input embeddings guarantee that every codeword
corresponds to an actual English word; `W_c` only re-projects them
into the CF subspace.  This is the SimVQ single-W pattern.

### 3.6 Residual Quantizer with cross-aspect exclusion

```
r          = z_aspect.clone()                             (B, K, 256)
used       = ∅                                            per-item set of used codes
for ℓ = 1 .. L_max:
    r̃      = r / ‖r‖                                      (B, K, 256)
    sim    = r̃ · Cᵀ                                       (B, K, V)
    sim[ used ] = −∞                                       cross-aspect exclusion
    for k = 1 .. K:
        c_ℓ_k       = argmax_v  sim[:, k, v]               (B,)
        used.add(c_ℓ_k)
        r[:, k]    −= m_ℓ_k · C[c_ℓ_k]                     mask-aware residual update

ẑ          = Σ_ℓ m_ℓ · C[ codes_ℓ ]                       (B, K, d_aspect)
ẑ_ste      = z_aspect + (ẑ − z_aspect).detach()           STE for downstream losses

L_codebook = ⟨ m, ‖C[c] − sg(z_aspect)‖² ⟩                # trains W_c
L_commit   = β · ⟨ m, ‖z_aspect − sg(C[c])‖² ⟩            # trains encoder, β = 0.25
L_recon    = MSE( ẑ_decoded, z.detach() )                  # after §3.7
```

VRVQ-style codebook warmup: the last 25 % of every batch is forced
to use `m = 1` (all levels active) so codebook learning sees full-
depth signal regardless of φ's current trajectory.  Rate-budget
contributions from those samples are excluded.

### 3.7 CF decoder — reconstruction back to z

```
decoder  = Transformer(K tokens, d_aspect, layers = 1, heads = 1, dropout = 0)
          + Linear(K · d_aspect → d_cf)
ẑ_decoded = decoder(ẑ)                                    (B, d_cf)
L_recon   = MSE( ẑ_decoded, z.detach() )
```

### 3.8 Descriptor + Alignment (semantic identifier ↔ raw text)

```
W_c_reverse = Moore-Penrose pseudoinverse of W_c           (256 → 384)
words       = W_c_reverse( ẑ_ste )                        (B, K, 384)
prompt      = "This {entity} can be described as:"          (item)
            | "This user prefers items described as:"      (user)
sentence    = [ prompt embed, comma, words[0], comma, words[1], … ]
h_d         = MiniLM(inputs_embeds=sentence) → mean-pool → L2-norm
                                                            (B, 384)

★ ablate_adaptive_target = True   (FACE convention, verified best):
    h_align = h_raw
    L_align = InfoNCE( h_d, h_raw,  τ = 0.07 )
```

Cross-side coupling:

```
L_ui = InfoNCE( vec(ẑ_u_ste),  vec(ẑ_{i⁺}_ste),  τ = 0.07 )
```

`L_align` and `L_ui` are aggregated into the *Align block* in §3.11.

### 3.9 SSW — Spherical Sliced-Wasserstein regulariser

S2WTM-exact form (Adhya & Sanyal, 2024), evaluated on the
**pre-quantization** aspect representation, flattened across
(user, item):

```
x         = [z_aspect_u ; z_aspect_i].reshape(N, 256), L2-norm  → S²⁵⁵
Z ∼ N(0, I)   shape (M = 50, 256, 2);    U, _ = QR(Z)            → V_{256, 2}
xp        = Uᵀ · x  ∈ ℝ²                                          (M, N, 2)
xp       ← xp / ‖xp‖                                              → S¹
angle     = ( atan2(−xp_y, −xp_x) + π ) / 2π                       ∈ [0, 1]
L_SSW     = (1/M) Σ_m  W₂²( {angle_{m, n}}_n ,  Uniform[0, 1] )
```

`W₂²` is the closed-form Wasserstein-2 distance from a discrete
sample to the uniform on a circle (S2WTM, Eq. 4) — no Monte-Carlo
target sampling.

`L_SSW` is aggregated into the *Quantize block* in §3.11.

### 3.10 Joint LightGCN — BPR loss

```
for each (u, i⁺) in the batch:
    i⁻  ∼ Uniform(items)
    z_u, z_{i⁺}, z_{i⁻}  =  lightgcn.propagate(adj)[u, i⁺, i⁻]
    L_BPR = -mean   log σ( z_u · z_{i⁺}  −  z_u · z_{i⁻} )
```

Stage-1 propagates LightGCN **live** every batch; the tokenizer's
SATP buffer is refreshed each step so that gradients flow back into
the LightGCN base embeddings.

### 3.11 Stage-1 objective — **4 external blocks** (LETTER-style)

```
L_total  =  L_BPR
         +  L_quantize        ←  L_recon + L_codebook + β · L_commit + λ_SSW · L_SSW
         +  λ_a · L_align     ←  L_align_u + L_align_i + λ_ui · L_ui
         +  λ_d · L_depth     ←  λ_rate · φ.mean()      depth-budget regulariser
```

| Block weight | Value |
|---|---|
| λ_a    (align block) | 0.5 |
| λ_d    (depth budget block) | 0.0005 |

The 7 *internal* terms and their inner weights appear in Appendix A.
The external block grouping mirrors LETTER (Wang et al. 2024)'s
recipe: reconstruction + commitment + diversity are folded into
the quantizer block; we extend this to also absorb `L_SSW` (which
is structurally an anti-collapse term on the codebook embedding).

### 3.12 Stage-1 training protocol

| | |
|---|---|
| Optimiser | AdamW |
| lr (tokenizer / LightGCN) | 1e-3 / 1e-3 |
| weight_decay (tokenizer / LightGCN) | 0 / 2e-4 |
| Epochs | 20 |
| Batch size | 512 |
| K_nn (SATP neighbours) | 10 |
| α_ste (sigmoid smoothness, unused with Gumbel-STE v2) | 2.0 |
| **Gumbel τ** | **1.0** |
| n_aspects, L_max | 8, 4 |
| Wall-clock (Toys) | ~17 min, single GPU |

### 3.13 Stage-1 outputs

```
identifier_cache.{seed}.bpr.ssw01.gumbel_v2.pkl
   ├─ item[iid]          list[int]   1-indexed codeword + EOA + EOI
   ├─ user[uid]          list[int]
   ├─ item_lengths       np.array    per-item identifier length
   ├─ cfg                {n_aspects, L_max, V, special_ids, …}
```

Verified Toys cache statistics (★ Gumbel STE v2):

| Statistic | Value |
|---|---|
| n_items | 11,924 |
| identifier length | min 24, max 41, **mean 35.2, std 2.21** |
| unique codewords used | **6,995 / 9,338 (74.9 %)** ★ |

Examples (real cache, Toys):

| item | aspect 0 | aspect 1 | aspect 2 | aspect 3 |
|---|---|---|---|---|
| "lego ninjago skull truck" | skull, boss, stairs | skeleton, blackout, weapon | nightmare, steel | head, wreck |
| "my little pony princess" | princess, pan, glen | mediterranean, furniture | duchess, cat, hero | royal, tablet |
| "melissa & doug dinosaur toy" | dinosaur, fisher | corn, assembly | chair, dna | tree, procedure |

---

## 4. Stage 2 — RPG Generative Recommender

We adopt RPG (KDD '25) verbatim and adapt only the position-mask
handling to support our variable-depth identifiers.

### 4.1 Cache → grid

```
parse_cache_to_grid(cache, K = 8, L_max = 4):
    grid (n_items, K, L_max)  long      1-indexed codeword id (0 = pad)
    mask (n_items, K, L_max)  bool      1 = active
flatten:  position axis  P = K · L_max = 32
```

### 4.2 Model — RPG architecture

```python
GPT2Config(
    vocab_size   = V + 2 = 9,340,          # 0 = pad, 1..V = codeword, V+1 = EOS
    n_positions  = 64,                      # ≥ history_max_len + 2
    n_embd       = 256,
    n_layer      = 2,
    n_head       = 4,
    n_inner      = 1024,
    activation   = "gelu_new",
    resid_pdrop  = 0.0,
    embd_pdrop   = 0.3,                     # ★ tuned
    attn_pdrop   = 0.3,                     # ★ tuned
)

pred_heads = ModuleList([ ResBlock(256) for _ in range(P) ])
class ResBlock:
    linear = Linear(256, 256, zero-init)
    act    = SiLU
    forward(x) = x + act(linear(x))           # identity-init residual

Total trainable parameters ≈ 6.1 M.
```

### 4.3 Forward — item-level pooling

```
tokens    = item_id2tokens[input_ids]                      (B, S = 50, P = 32)
mask      = item_id2mask  [input_ids].float()              (B, S, P)
embs      = gpt2.wte(tokens)                                (B, S, P, 256)
denom     = mask.sum(-1, keepdim = True).clamp_min(1.0)
item_emb  = (embs * mask.unsqueeze(-1)).sum(-2) / denom    (B, S, 256)

out       = gpt2(inputs_embeds = item_emb, attention_mask = attn)
last      = out.last_hidden_state[:, last_valid_pos]        (B, 256)
```

### 4.4 Parallel multi-token prediction

```
final_states = stack([ h(last) for h in pred_heads ], dim = 1)     (B, P, 256)
states_n     = F.normalize(final_states, dim = -1)
codebook_n   = F.normalize(gpt2.wte.weight, dim = -1)              (vocab, 256)
logits       = states_n @ codebook_nᵀ / T                           T = 0.05
```

### 4.5 Loss

```
target_tokens = item_id2tokens[ labels ]                            (B, P)
target_mask   = item_id2mask  [ labels ].float()                    (B, P)
loss_per_pos  = F.cross_entropy(
                   logits.flatten(0, 1), target_tokens.flatten(),
                   reduction = 'none'
               ).view_as(target_mask)
L_gen         = ( loss_per_pos · target_mask ).sum()
                / target_mask.sum().clamp_min(1.0)
```

### 4.6 Inference — full ranking

```
log_p   = F.log_softmax(logits, dim = -1)                            (B, P, vocab)
score_i = mean over P of   log_p[b, p, item_id2tokens[i, p]] · item_id2mask[i, p]
scores[ history_items ] = -∞
top_K   = scores.topk(K)
```

### 4.7 Stage-2 training (★ verified best)

| | value |
|---|---|
| Optimiser | AdamW |
| lr | 3e-3 |
| **weight_decay** | **0.05** ★ |
| **dropout (embd / attn)** | **0.3 / 0.3** ★ |
| Scheduler | cosine + warmup_ratio 0.01 |
| max_grad_norm | 1.0 |
| Batch (train / eval) | 256 / 64 |
| history_max_len | 50 |
| total_epochs (cap) | 150 |
| patience (val-R@5) | 20 |
| eval_every | 1 |
| Wall-clock (Toys) | ~25–30 min |

---

## 5. Evaluation protocol

```
Split          leave-one-out
                 train  = items[ : −2 ]
                 val    = items[ : −1 ],  target = items[ −2 ]
                 test   = items,           target = items[ −1 ]
Ranking        full item ranking (no negative sampling, no graph propagation)
Mask           history items per user, pad item id (0)
Selection      epoch of best val-R@5
Final report   best.pth on test split, single shot
Metrics        R@5, R@10, R@20,  NDCG@5, NDCG@10, NDCG@20
```

---

## 6. Sparsity analysis (paper §4 main contribution)

Items are bucketed by **raw text word count** of the item metadata
(title + categories + description, **not** any model-internal
signal).  Per-group full-ranking eval gives, on Toys:

| Group | n_users | R@5 | NDCG@5 |
|---|---|---|---|
| sparse (Q1) | 4,692 | 0.0578 | **0.0419** ★ |
| medium     | 6,450 | 0.0482 | 0.0339 |
| rich   (Q4) | 8,270 | 0.0586 | 0.0394 |

NDCG@5 **sparse > rich**:  +6 % advantage in the sparse-text regime.

**Item-metadata sparsity (RLMRec):** the table below records the
missing fraction of each RLMRec field per dataset.

| Dataset | title | categories | brand | description | avg wc |
|---|---|---|---|---|---|
| Beauty | 0.1 % | 0.0 % | 17.3 % | 0.0 % | 95 |
| Toys | 0.5 % | 0.0 % | 100 % | 0.0 % | 99 |
| Sports | 0.5 % | 0.0 % | 100 % | 0.0 % | 110 |
| **Yelp** | **100 %** ★ | 1.4 % | **100 %** ★ | **100 %** ★ | 79 |

Yelp is structurally text-sparse (only `categories` available); we
interpret InATTo's Yelp result as a lightweight-model retention
under extreme metadata loss.

---

## 7. Main result (★ paper §4 table)

★ Toys is verified end-to-end with Gumbel-STE v2.  Other datasets
will be updated once their Stage-2 Gumbel-v2 runs complete; the
sigmoid-STE baseline numbers (Beauty / Sports / Yelp) are reported
below as the floor.

| Dataset | R@5 | R@10 | R@20 | N@5 | N@10 | N@20 | tier |
|---|---|---|---|---|---|---|---|
| **Toys** (★ Gumbel v2) | **0.0552** | 0.0788 | 0.1085 | **0.0389** | 0.0465 | 0.0539 | ★ above TIGER (0.047), LETTER (~0.052), IDGenRec (0.054), RPG (0.054) |
| **Beauty** | 0.0528 (sigmoid-STE; v2 running) | 0.0749 | 0.1055 | 0.0371 | 0.0442 | 0.0519 | ★ above FACE (0.049), TIGER (0.039), LETTER (~0.040), RPG (0.039) |
| Sports | 0.0288 (sigmoid-STE; v2 running) | 0.0415 | 0.0592 | 0.0198 | 0.0239 | 0.0283 | TIGER tier |
| Yelp | 0.0258 (sigmoid-STE; v2 running) | 0.0428 | 0.0670 | 0.0162 | 0.0216 | 0.0277 | TIGER tier |

**Parameter budget**:

| Method | Params | Comment |
|---|---|---|
| SASRec | ~1 M | sequential baseline |
| **InATTo (★ ours)** | **6.1 M** | GPT-2 small (Stage 2) + MiniLM frozen |
| TIGER | ~60 M | T5 |
| LETTER | ~60 M | T5 LC-Rec |
| RPG (KDD'25) | ~60 M | GPT-2 + ID-specific recipe |
| FACE | 220 M | T5-base |
| IDGenRec | 770 M | T5-large |
| GRAM | 7 B | NV-Embed-v2 |

InATTo is **the smallest** among published generative recommenders
and **above all 60 M-class competitors on Toys**.  Beauty result
already places it above FACE (220 M).

---

## 8. Hyperparameter quick reference

| Stage | Knob | Value |
|---|---|---|
| Codebook | V, d_llm, d_aspect | 9,338, 384, 256 |
| Tokenizer | K, L_max, K_nn (SATP) | 8, 4, 10 |
| | α_ste (sigmoid), **Gumbel τ** | 2.0, **1.0** |
| | commit β, full_codebook_rate | 0.25, 0.25 |
| | τ_align, τ_ui | 0.07, 0.07 |
| Stage-1 blocks | (BPR, quantize, align, depth-budget) | (1.0, 1.0, 0.5, 0.0005) |
| Stage-1 internal (Appendix A) | (L_SSW, L_ui) | (0.1, 0.1) |
| Stage-1 train | epochs, batch, lr | 20, 512, 1e-3 |
| Stage-2 model | n_embd, n_layer, n_head, n_inner | 256, 2, 4, 1024 |
| | dropout (embd / attn) | 0.3 / 0.3 |
| | n_heads (parallel) | 32 |
| | softmax τ | 0.05 |
| Stage-2 train | epochs (cap), patience | 150, 20 |
| | batch (train / eval) | 256 / 64 |
| | history_max_len | 50 |
| | lr, weight_decay | 3e-3, 0.05 |

---

## 9. Reproducing the main result (Toys)

### Stage 0 — pre-data (one-shot, all datasets)

```bash
pixi run -- python scripts/01_lightgcn_bpr.py \
       --dataset toys --seed 2023 --epochs 1000
pixi run -- python scripts/06_encode_profile.py --dataset toys
pixi run -- python scripts/02_build_codebook.py
```

### Stage 1 — InATTo tokenizer  (★ ssw01 + Gumbel-STE v2)

```bash
pixi run -- python scripts/08a_train_stage1_bpr.py \
       --dataset toys --cuda 0 \
       --total_epochs 20 --batch_size 512 \
       --ablate_adaptive_target \
       --ssw_weight 0.1 \
       --use_gumbel_mask --gumbel_v2 --gumbel_tau 1.0 \      # ★ NEW
       --tag ssw01.gumbel_v2
# → checkpoints/inatto/inatto-toys-2023.stage1bpr.ssw01.gumbel_v2.pth
# → data/toys/identifier_cache.2023.bpr.ssw01.gumbel_v2.pkl
```

### Stage 2 — RPG

```bash
pixi run -- python scripts/08c_train_stage2_rpg.py \
       --dataset toys --cuda 0 \
       --cache_suffix bpr.ssw01.gumbel_v2 \
       --output_tag wd05.do30 \
       --weight_decay 0.05 \
       --dropout 0.3 \
       --total_epochs 150 \
       --batch_size 256 --eval_batch_size 64 \
       --lr 3e-3 --history_max_len 50 \
       --eval_every 1 --patience 20
# → checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.wd05.do30.best.pth
```

Substitute `--dataset {beauty, sports, yelp}` for the other three
datasets; the flags are unchanged.

### Sparsity analysis

```bash
pixi run -- python scripts/10_sparsity_group_eval.py \
       --dataset toys --cuda 0 \
       --cache_suffix bpr.ssw01.gumbel_v2 \
       --ckpt checkpoints/inatto/inatto-toys-2023.stage2rpg.bpr.ssw01.gumbel_v2.wd05.do30.best.pth \
       --n_groups 4 --method_label "InATTo"
```

---

## 10. Appendix A — Loss decomposition

Internal terms, their weights, and the external block (§3.11) they
belong to:

| Internal term | Block | Weight |
|---|---|---|
| L_recon | L_quantize | 1.0 |
| L_codebook | L_quantize | 1.0 |
| L_commit  (β) | L_quantize | 0.25 |
| L_SSW | L_quantize | 0.1 |
| L_align_u + L_align_i | L_align | 0.5 |
| L_ui | L_align | 0.1 |
| L_rate  ( φ.mean() ) | L_depth | 0.0005 |
| L_BPR | L_BPR | 1.0 |

External presentation (paper §3, 4 terms):

```
L = L_BPR  +  L_quantize  +  λ_a · L_align  +  λ_d · L_depth
```

---

## 11. Appendix B — Source files

```
InATTo_impl/
├── inatto/
│   ├── e2e_model.py             InATToE2E (Stage-1 wrapper)
│   ├── tokenizer.py             InATToTokenizer (assembles all modules)
│   ├── modules/
│   │   ├── codebook.py          SimVQ (frozen C₀ + trainable W_c)
│   │   ├── satp.py              Sparse-aware text propagation
│   │   ├── reliability.py       δ, δ_k (Gram-Schmidt projection)
│   │   ├── encoder.py           MultiProjector (E1), DisTransformer (E2),
│   │   │                         ImportanceSubnetwork (Ep), CFSpaceDecoder
│   │   ├── depth_mask.py        depth_mask_ste,
│   │   │                         depth_mask_gumbel_ste_v2 ★
│   │   ├── rq.py                Residual quantizer w/ cross-aspect exclusion
│   │   ├── descriptor.py        FACE-style descriptor head (frozen MiniLM)
│   │   ├── alignment.py         Per-side InfoNCE
│   │   ├── ui_alignment.py      Cross-side InfoNCE
│   │   └── ssw.py               S2WTM-exact spherical sliced Wasserstein
│   └── backbone/lightgcn.py
├── scripts/
│   ├── 08a_train_stage1_bpr.py        ★ Stage 1 entry
│   ├── 08c_train_stage2_rpg.py        ★ Stage 2 entry
│   └── 10_sparsity_group_eval.py      per-sparsity-quantile analysis
└── figures/                            generate_figures.py + 7 PNGs
```

---

## 12. Notes on naming and design choices

- **`inatto/e2e_model.py :: InATToE2E`** is a Stage-1 *wrapper* class;
  the name predates the move to two-stage training (we keep it for
  checkpoint compatibility — no end-to-end interaction loss ties
  Stage 1 and Stage 2).
- **Depth mask** defaults to the **Gumbel-STE v2** form
  (`depth_mask_gumbel_ste_v2`).  The standard sigmoid STE
  (`depth_mask_ste`) remains available as the *ablation* in §13.
- **`--ablate_adaptive_target`** is *enabled* in the verified best.
  This is the FACE convention (`h_adp = h_raw`); the alternative
  ρ-weighted target under-performed in our sweep and is reported as
  an ablation row.
- **Stage 2** is RPG (KDD '25) verbatim except for the variable-
  depth-aware position mask used in (i) item-level pooling, (ii) the
  per-position loss, and (iii) the inference-time score aggregation.

---

## 13. Ablation summary (paper §5)

| Variant | Toys R@5 | Note |
|---|---|---|
| **InATTo (★ ours)** | **0.0552** | ssw01 + Gumbel-STE v2 + wd05 / do30 |
| w/o Gumbel-STE v2 (sigmoid STE only) | 0.0540 | baseline ssw01 |
| w/o `ablate_adaptive_target`  (ρ-weighted) | < baseline | FACE alignment ablation |
| w/o BPR (LightGCN frozen) | < baseline | joint-CF importance |
| w/o SSW | diversity ↓ | anti-collapse necessity |
| w/o variable depth  (m ≡ 1 forced) | TBD | fixed-depth ablation |
