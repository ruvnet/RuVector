# TriAttention: Trigonometric KV Cache Sparsity for ruvLLM

## Abstract

TriAttention (arXiv:2604.04921, April 2026) is a training-free KV cache
compression method that achieves 10.7x memory reduction with preserved
reasoning accuracy. Unlike quantization approaches (TurboQuant, KIVI), it
operates on the **token dimension** — pruning unimportant keys via
trigonometric scoring in pre-RoPE space.

This document maps TriAttention to ruvLLM's inference stack, where it
complements the existing TurboQuant implementation (doc 08) for a stacked
compression pipeline targeting 30-50x total KV cache reduction.

## 1. Paper Citation

**Title:** TriAttention: Efficient Long Reasoning with Trigonometric KV Compression

**Authors:** Weian Mao*, Xi Lin*, Wei Huang*, Yuxin Xie, Tianfu Fu, Bohan Zhuang, Song Han, Yukang Chen

**Affiliations:** MIT, NVIDIA, Monash University (* = equal contribution)

**Venue:** arXiv preprint, arXiv:2604.04921, April 6, 2026

**Categories:** cs.CL, cs.CV

**Repo:** https://github.com/WeianMao/triattention

## 2. Core Algorithm

### 2.1 Foundation: RoPE as Complex Rotation

RoPE divides a d-dimensional vector into d/2 two-dimensional subspaces
indexed by frequency band f ∈ {0, ..., d/2 - 1}. Each band rotates at:

```
ω_f = θ^(-2f/d),  θ = 10000
```

In complex form:

```
q̃_f(p) = q_f · exp(i · ω_f · p)
k̃_f(p) = k_f · exp(i · ω_f · p)
```

### 2.2 RoPE Attention Logit Decomposition

The dot product between query at position p_q and key at p_k decomposes as:

```
logit(q, k) = Σ_f ||q_f|| · ||k_f|| · cos(ω_f · Δ + φ_f)     [Eq. 2]
```

where:
- Δ = p_q - p_k (Q-K distance)
- φ_f = arg(q_f) - arg(k_f) (phase difference in band f)

### 2.3 Q/K Concentration Phenomenon

**Key observation:** Pre-RoPE Q and K vectors are highly concentrated around
fixed non-zero centers across most attention heads. This is stable across
token positions and input contexts.

**Quantified via Mean Resultant Length (MRL):**

```
R_f = ||E[q_f]|| / E[||q_f||]
```

- R_f = 1: perfect concentration
- R_f = 0: uniform dispersion
- ~90% of heads: R > 0.95 across Math, Coding, Chat domains
- MRL values: 0.977-0.980 (nearly identical across domains)

### 2.4 Trigonometric Series Approximation

When Q/K are concentrated (q_f ≈ q̄_f), the logit becomes a function of
distance alone:

```
logit(Δ) ≈ Σ_f ||q̄_f|| · ||k̄_f|| · cos(ω_f · Δ + φ̄_f)
         = Σ_f [a_f · cos(ω_f · Δ) + b_f · sin(ω_f · Δ)]     [Eq. 3]
```

This is a trigonometric series in Δ with RoPE geometric frequencies.

### 2.5 The TriAttention Scoring Function

**Trigonometric Series Score:**

```
S_trig(k, Δ) = Σ_f ||E[q_f]|| · ||k_f|| · cos(ω_f · Δ + φ_f)     [Eq. 6]
```

**Norm-Based Score (MRL-adaptive):**

```
S_norm(k) = Σ_f (1 - R_f) · E[||q_f||] · ||k_f||     [Eq. 8]
```

When R_f is high, S_trig dominates. When R_f is low, S_norm provides fallback.

**Combined:**

```
S(k, Δ) = S_trig(k, Δ) + S_norm(k)     [Eq. 10]
```

**Multi-Offset Averaging (geometric spacing):**

```
S̃(k) = (1/|D|) · Σ_{δ∈D} S(k, Δ + δ)     [Eq. 11]
```

where D = {1, 2, 4, ..., 2^16} (17 offsets, geometric spacing).

**GQA Aggregation:** Z-score normalize per head, then max across G query heads:

```
S_final(k) = max_{g∈{0,...,G-1}} (S̃^(g)(k) - μ_g) / σ_g     [Eq. 13]
```

### 2.6 Compression Procedure

1. **Offline calibration** (one-time per model): Compute Q/K center vectors
   E[q_f], E[k_f], magnitudes E[||q_f||], and MRL values R_f from calibration
   data. Robust to data quality (50K-960K tokens sufficient).

2. **Inference** (every β=128 tokens when cache > budget B):
   a. Invert RoPE on cached keys to recover pre-RoPE representations
   b. Compute S_trig + S_norm for each cached key using calibration Q centers
   c. Average over geometric future offsets
   d. For GQA: z-score normalize, aggregate via max
   e. Retain top-B keys, evict the rest

3. **Protected window:** Recent 128 tokens always preserved.

## 3. Performance Results

### 3.1 Reasoning Benchmarks (KV Budget = 2048, Qwen3-8B)

| Method | AIME24 | AIME25 | MATH-500 |
|--------|--------|--------|----------|
| Full Attention | 57.1 | 40.8 | 69.6 |
| SnapKV | 34.6 | 20.0 | 49.2 |
| H2O | 19.2* | — | — |
| R-KV | 25.4 | 17.5 | 46.4 |
| **TriAttention** | **42.1** | **32.9** | **56.0** |

*H2O at 10% budget on DS-R1-Qwen-7B

### 3.2 Key Results

| Metric | Value | Configuration |
|--------|-------|---------------|
| KV memory reduction | 10.7× | At matched accuracy on AIME25 |
| Throughput improvement | 2.5× | AIME25, single A100 80GB |
| Throughput (MATH-500) | 6.3× | Budget 1024 vs Full Attention |
| FlashAttention compat | Yes | FA-2 and FA-3 |
| Training required | None | Calibration only |
| RULER retrieval | 66.1 | vs SnapKV 55.6 (+10.5) |

### 3.3 Exceeds Full Attention

At 30% KV budget (AIME24, DS-R1-Qwen-7B): TriAttention 46.7% > Full
Attention 43.8%, suggesting KV cache contains redundancy.

At 4096 budget (AIME25, Qwen3-8B): TriAttention 43.3% > Full Attention 40.8%.

### 3.4 Ablation: What Matters Most

| Component Removed | AIME24 Delta |
|-------------------|-------------|
| S_trig (norm only) | **-20.4pp** |
| S_norm (trig only) | -5.4pp |
| MRL weighting | -3.7pp |
| Linear spacing (vs geometric) | **-17.1pp** |

Trigonometric series and geometric offset spacing are critical.

## 4. Pre-RoPE vs Post-RoPE: Why It Matters

| Dimension | Post-RoPE (H2O, SnapKV) | Pre-RoPE (TriAttention) |
|-----------|-------------------------|------------------------|
| Signal source | Transient attention scores | Stable model-intrinsic centers |
| Observation window | ~25 queries useful | All positions via centers |
| Memory complexity | O(n²) — full attention materialization | Sub-quadratic |
| FlashAttention | Incompatible (H2O) | Fully compatible |
| Domain sensitivity | High | Low (MRL 0.977-0.980 across all) |
| Failure mode | Premature eviction | Calibration-dependent |

## 5. Limitations

### 5.1 Architectural Requirements

- **Requires RoPE**: Works with GQA and MLA, but not absolute positional
  encoding or ALiBi
- **Q/K concentration assumption**: ~10% of heads have R < 0.95, relying
  more on norm-based fallback
- **No custom CUDA kernel**: Current PyTorch implementation; fused kernels
  would improve latency

### 5.2 Observed Failure Modes

- **Deep recursive tasks (depth ≥ 18)**: Begins to lag Full Attention on
  Recursive State Query benchmark
- **Tight budgets**: Gap at 2048 tokens (32.9% vs 40.8% on AIME25); closes
  at 4096 (43.3% vs 40.8%)
- **vLLM prefix caching incompatible**: Must disable `--enable-prefix-caching`

### 5.3 Operational Constraints

- Scoring requires RoPE inversion of cached keys (per-round overhead)
- Compression triggers at fixed 128-token intervals (cache may temporarily
  exceed budget)
- One-time calibration step per model

## 6. Comparison Summary

### 6.1 vs Other KV Cache Sparsity Methods

| Method | AIME25 (Qwen3) | Throughput | FA Compatible | Training-Free |
|--------|----------------|-----------|---------------|---------------|
| Full Attention | 40.8 | 222.8 tok/s | Yes | — |
| TriAttention | **32.9** | **1405 tok/s** | Yes | Yes |
| R-KV | 17.5 | 760 tok/s | Partial | Yes |
| SnapKV | 20.0 | — | Partial | Yes |
| H2O | — | — | No | Yes |
| StreamingLLM | — | — | Yes | Yes |

### 6.2 vs KV Cache Quantization (Orthogonal Axis)

| Method | Type | Compression | What It Reduces |
|--------|------|------------|-----------------|
| TriAttention | Sparsity | 10.7× | Token count (rows) |
| TurboQuant | Quantization | 6× | Bit width (columns) |
| KVTC | Transform coding | 20-40× | PCA dims + bits |
| MLA | Architectural | 71× | Head dimensions |
| SALS | Latent projection | 6.4× | Dimensions + tokens |

## 7. Mapping to ruvLLM Architecture

### 7.1 Integration Point

TriAttention operates BEFORE TurboQuant in the KV cache pipeline:

```
Token Generation
  └── TriAttention: Score & evict unimportant keys (10× reduction in token count)
       └── TurboQuant: Compress surviving keys to 3.5 bits (5× reduction in bit size)
            └── KV Cache: Store ~50× compressed data
```

### 7.2 Architecture with Existing ruvLLM KV Cache

**Current** (from kv_cache.rs):
```
TurboQuantKvCache:
  Hot tier (FP16):        Recent 256 tokens
  Cold tier (TurboQuant): Older tokens (~3.5 bits)
```

**Proposed** (TriAttention + TurboQuant stacked):
```
TriQuantKvCache:
  Hot tier (FP16):          Recent 128 tokens (protected window)
  Warm tier (TriAttention): Sparsified keys (~10% retained, FP16)
  Cold tier (TurboQuant):   Quantized survivors (~3.5 bits)
```

### 7.3 Implementation Plan

**Phase 1: Calibration Infrastructure**
- `triattention_calibrate.rs`: Compute Q/K centers, MRL, phase offsets from
  calibration data
- Store per-model calibration in RVF format
- One-time cost per model

**Phase 2: Scoring Engine**
- `triattention_score.rs`: Trigonometric series scoring + norm fallback
- RoPE inversion for cached keys
- GQA normalize-then-aggregate
- SIMD-optimized trigonometric evaluation (NEON/AVX2)

**Phase 3: Cache Integration**
- `triattention_cache.rs`: Window-based pruning every 128 tokens
- Integration with existing `TurboQuantKvCache` as upstream stage
- Protected sliding window management

**Phase 4: Stacked Pipeline**
- Fused TriAttention→TurboQuant pipeline
- Benchmark stacked compression ratios
- Quality validation on AIME/MATH/RULER benchmarks

### 7.4 Interaction with RuVector Coherence Layer

TriAttention's pre-RoPE centers can be viewed as a coherence signal:
- High MRL (R > 0.95): Head has strong positional preference → predictable
  attention pattern → safe to prune aggressively
- Low MRL (R < 0.95): Head has diffuse attention → content-dependent →
  rely on norm-based scoring

This aligns with RuVector's mincut coherence gating: heads with high
coherence (high R) can be compressed more aggressively, similar to how
mincut identifies structurally important edges.

## 8. Risks & Mitigations

### 8.1 RoPE Dependency

TriAttention requires RoPE. ruvLLM supports multiple position encodings.

**Mitigation**: Feature-gate TriAttention behind `#[cfg(feature = "rope")]`.
Fall back to TurboQuant-only for non-RoPE models.

### 8.2 Calibration Data Requirements

One-time calibration per model (50K-960K tokens).

**Mitigation**: Ship pre-computed calibration files for supported models
(Llama3, Qwen3, DeepSeek-R1). Provide calibration CLI tool for custom models.

### 8.3 Interaction with TurboQuant

TurboQuant's PolarQuant assumes Euclidean geometry on full key vectors.
After TriAttention pruning, the remaining keys are a sparse subset.

**Mitigation**: TurboQuant operates per-vector, not cross-vector, so
sparsification doesn't affect its compression quality. Validated by
MiniKV's 2-bit + sparsity co-design achieving 86% compression.

### 8.4 Quality at Extreme Compression

Stacking 10× sparsity + 5× quantization = 50× total may degrade quality
on reasoning-heavy tasks.

**Mitigation**: Use existing ruvLLM delta checks and witness gating to
detect quality degradation. Dynamically adjust sparsity budget if attention
error exceeds coherence threshold.

## 9. References

1. TriAttention (arXiv 2026): arxiv.org/abs/2604.04921
2. TriAttention GitHub: github.com/WeianMao/triattention
3. TriAttention Project Page: weianmao.github.io/tri-attention-project-page/
4. TurboQuant (ICLR 2026): arxiv.org/abs/2504.19874
5. MiniKV (ACL 2025): arxiv.org/abs/2411.18077
6. TailorKV (ACL 2025): arxiv.org/abs/2505.19586
7. SparK (AAAI 2026): arxiv.org/abs/2508.15212
8. KVTC (ICLR 2026): arxiv.org/abs/2511.01815
9. SALS (NeurIPS 2025): arxiv.org/abs/2510.24273
10. DeepSeek-V3 MLA: arxiv.org/abs/2412.19437
11. NVIDIA kvpress: github.com/NVIDIA/kvpress
12. KV Cache Compression Survey: arxiv.org/abs/2508.06297

## 10. File Inventory

| File | Description |
|------|-------------|
| `crates/ruvllm/src/kv_cache.rs` | Existing TurboQuantKvCache (extend) |
| `crates/ruvllm/src/triattention/` | New: calibration, scoring, cache integration |
| `crates/ruvllm/src/quantize/turbo_quant.rs` | Existing TurboQuant (unchanged) |
| `docs/research/quantization-edge/08-turboquant-kv-cache-compression.md` | TurboQuant docs |
| `docs/research/quantization-edge/09-triattention-kv-sparsity.md` | This document |
| `docs/research/quantization-edge/10-stacked-kv-compression.md` | Stacked architecture |
