# ADR-147: Stacked KV Cache Compression: TriAttention + TurboQuant Pipeline

## Status
Proposed

## Date
2026-04-08

## Context

ruvLLM targets edge devices (Raspberry Pi 5, Apple Silicon, WASM browsers) where memory is the primary bottleneck for long-context inference. At 128K context with FP16, a Qwen3-8B model requires ~8 GB for the KV cache alone, exceeding available RAM on most edge platforms.

TurboQuant (ICLR 2026) is already implemented in ruvLLM (phases 1-3, `crates/ruvllm/src/quantize/turbo_quant.rs`) and provides ~6x KV cache compression via data-oblivious PolarQuant + QJL residual correction at 3.5 bits per value. This is insufficient for 128K context on edge devices -- a Pi 5 with 8 GB RAM cannot fit both model weights and a 1.3 GB KV cache.

TriAttention (arXiv:2604.04921, April 2026, MIT/NVIDIA/Monash) is a new training-free method that prunes unimportant KV tokens using trigonometric scoring in pre-RoPE space. It achieves 10.7x KV memory reduction while preserving reasoning accuracy, and in some configurations exceeds full-attention quality (suggesting KV cache redundancy). The key insight: pre-RoPE Q/K vectors are highly concentrated around fixed centers (~90% of heads have MRL > 0.95), enabling a trigonometric series approximation of attention logits as a function of token distance alone.

These two methods operate on **orthogonal compression axes**:
- TriAttention reduces **token count** (rows of the KV matrix)
- TurboQuant reduces **bit width** (precision of each element)

No published work has combined these two approaches despite confirmed orthogonality. Existing stacked systems demonstrate the multiplicative principle but leave this specific combination unexplored:

| System | Compression | Axes | Venue |
|--------|-------------|------|-------|
| MiniKV | 14x (86% reduction) | Token sparsity + 2-bit quantization | ACL 2025 |
| TailorKV | 34.2x | Layer-discriminative sparsity + 1-bit quantization | ACL 2025 |
| KVTC | 20-40x | PCA + adaptive bits + entropy coding | ICLR 2026 |
| KVSculpt | Variable | Distillation + quantization | arXiv 2026 |
| **TriAttention + TurboQuant** | **30-50x** | **Trigonometric sparsity + data-oblivious quantization** | **Proposed (this ADR)** |

A three-axis taxonomy of KV cache compression exists:

| Axis | What It Reduces | Analogy | ruvLLM Method |
|------|----------------|---------|---------------|
| Token sparsity | Sequence length (rows) | Delete spreadsheet rows | TriAttention (new) |
| Bit quantization | Bits per element (cell size) | Shrink each cell's format | TurboQuant (existing) |
| Dimension compression | Feature dims (columns) | Delete spreadsheet columns | Future (SparK, KVTC PCA) |

## Decision

Implement a stacked KV cache compression pipeline combining TriAttention (token-level sparsity via trigonometric RoPE analysis, 10.7x) with TurboQuant (bit-level quantization via PolarQuant + QJL, 6x) for a multiplicative **~50x KV cache reduction** in ruvLLM. TriAttention operates upstream of TurboQuant: tokens are first pruned by importance, then survivors are quantized to 3.5 bits.

### Pipeline Architecture

```
Token Generation
  |
  v
Stage 1: TriAttention (Token Sparsity)

  1. Invert RoPE on cached keys to recover pre-RoPE representations
  2. Compute S_trig: trigonometric series score using calibrated Q/K centers
     S_trig(k, D) = Sum_f ||E[q_f]|| * ||k_f|| * cos(w_f * D + phi_f)
  3. Compute S_norm: MRL-weighted norm-based fallback for low-concentration heads
     S_norm(k) = Sum_f (1 - R_f) * E[||q_f||] * ||k_f||
  4. Average over geometric future offsets: D = {1, 2, 4, ..., 2^16}
  5. GQA: z-score normalize per head, max aggregate across group
  6. Retain top-B keys, evict rest
  |
  | Result: ~10x fewer tokens in cache
  v
Stage 2: TurboQuant (Bit Quantization)

  1. Hadamard rotation (make dimensions approximately independent)
  2. PolarQuant scalar quantization (3.5 bits, no codebooks)
  3. QJL residual correction (1-bit signs, unbiased inner product estimator)
  |
  | Result: ~5x smaller per surviving token
  v
KV Cache: ~50x compressed total

  Hot tier  (FP16):         Recent 128 tokens (protected window, never evicted)
  Warm tier (TriAttention): Sparsified keys, FP16 (pre-quantization staging)
  Cold tier (TurboQuant):   Quantized survivors (~3.5 bits per value)
```

### Concrete Memory Impact

Model: Qwen3-8B, 128K context, batch=1

| Configuration | KV Cache Size | Reduction | Pi 5 Feasible |
|---------------|--------------|-----------|---------------|
| FP16 (baseline) | ~8 GB | 1x | No |
| TurboQuant only (3.5-bit) | ~1.75 GB | 4.6x | Marginal |
| TriAttention only (10% budget) | ~0.8 GB | 10x | Yes (tight) |
| **Stacked (TriAttention + TurboQuant)** | **~175 MB** | **~46x** | **Yes (comfortable)** |

At 175 MB for 128K context, ruvLLM can serve long-context inference on a Raspberry Pi 5 with room for quantized model weights (Q4: ~4.5 GB) and application overhead.

### Integration with Existing kv_cache.rs

**Current** (`TurboQuantKvCache`):
```
Hot tier (FP16):        Recent 256 tokens
Cold tier (TurboQuant): Older tokens (~3.5 bits)
```

**Proposed** (`TriQuantKvCache`):
```
Hot tier (FP16):          Recent 128 tokens (protected window)
Warm tier (TriAttention): Sparsified keys (~10% retained, FP16, staging)
Cold tier (TurboQuant):   Quantized survivors (~3.5 bits)
```

The warm tier acts as a staging area. Every 128 tokens (TriAttention's compression interval), the scoring engine evaluates all cached keys and evicts tokens below the budget threshold. Survivors that age out of the warm tier migrate to the cold tier for TurboQuant compression.

## Alternatives Considered

### 1. TurboQuant Only (Current State)

6x compression. Already implemented (phases 1-3). Insufficient for 128K context on edge devices -- 1.75 GB KV cache still exceeds practical budgets when combined with model weights.

**Rejected because:** Does not meet edge memory targets. Leaves 10x+ improvement available on the orthogonal token sparsity axis.

### 2. KVTC (NVIDIA, ICLR 2026)

20-40x compression via PCA decorrelation + dynamic bit allocation + entropy coding. Three-stage transform coder inspired by JPEG.

**Rejected because:** Complex pipeline (PCA requires SVD per batch), no Rust implementation exists, entropy coding (DEFLATE/LZMA2) adds latency incompatible with real-time inference on edge devices. Also, no open-source fused kernel.

### 3. MLA (DeepSeek-V3)

71x reduction by compressing KV cache to 576-dim latent vectors. The most aggressive published compression.

**Rejected because:** Architectural change, not inference-time compression. Requires model training with the MLA architecture. Cannot be applied to existing RoPE-based models (Llama, Qwen, Mistral).

### 4. SALS (NeurIPS 2025)

6.4x via latent projection + sparse token selection with RoPE-free Q-K interactions.

**Rejected because:** Moderate improvement (6.4x) comparable to TurboQuant alone. Combining two 6x methods is less effective than combining a 10x sparsity method with a 6x quantization method.

### 5. TailorKV (ACL 2025)

34.2x via layer-discriminative sparsity + 1-bit quantization. Closest competitor to the proposed pipeline.

**Rejected as primary approach because:** Uses extremely aggressive 1-bit quantization (quality concerns for reasoning tasks), requires CPU offloading for deep layers, and the layer-discriminative routing adds complexity. However, TailorKV's insight -- different layers prefer different compression strategies -- is valuable and should inform future per-layer adaptive compression in ruvLLM.

### 6. Pure Token Eviction (H2O, SnapKV, StreamingLLM)

Post-RoPE attention-score-based eviction.

**Rejected because:** Post-RoPE methods are domain-sensitive, incompatible with FlashAttention (H2O), and inferior to TriAttention's pre-RoPE trigonometric scoring on reasoning benchmarks (AIME24: TriAttention 42.1 vs SnapKV 34.6 vs H2O 19.2).

## Consequences

### Positive

- **128K context on edge devices**: 50x KV compression reduces 8 GB to ~175 MB, enabling long-context inference on Pi 5 and Apple Silicon with comfortable memory margins.
- **First-mover advantage**: No published work combines TriAttention with TurboQuant despite confirmed orthogonality. ruvLLM can establish the reference implementation.
- **Training-free**: Neither component requires fine-tuning. TriAttention needs lightweight offline calibration (50K-960K tokens, one-time per model), and TurboQuant is fully data-oblivious.
- **Composable architecture**: The stacked pipeline is modular -- TriAttention and TurboQuant can each be enabled/disabled independently, and a future third axis (SparK channel pruning) can stack on top for 70x+.
- **Quality at compression**: TriAttention exceeds full-attention accuracy at 30% budget on some benchmarks (AIME24: 46.7% vs 43.8%), suggesting KV cache contains exploitable redundancy.
- **FlashAttention compatible**: TriAttention's pre-RoPE scoring is fully compatible with FlashAttention-2 and FlashAttention-3, unlike H2O and partial SnapKV.
- **Existing infrastructure**: TurboQuant is already production-quality in ruvLLM (13 tests, three-tier cache). TriAttention integrates upstream with minimal disruption.
- **Coherence alignment**: TriAttention's MRL concentration metric aligns with RuVector's mincut coherence gating -- high-coherence heads (MRL > 0.95) can be compressed more aggressively.

### Negative

- **Per-model calibration required**: TriAttention requires one-time offline calibration per model to compute Q/K center vectors, MRL values, and phase offsets. Must ship pre-computed calibration files for supported models (Llama3, Qwen3, DeepSeek-R1) and provide a calibration CLI tool for custom models.
- **RoPE-only**: TriAttention requires Rotary Position Encoding. Models using absolute positional encoding or ALiBi cannot use the token sparsity stage. Must feature-gate behind `#[cfg(feature = "rope")]` with fallback to TurboQuant-only.
- **Fused kernel complexity**: Achieving peak performance requires fused Metal/NEON kernels for the combined RoPE-inversion + trigonometric-scoring + quantization pipeline. MiniKV demonstrates this is feasible (fused Triton kernels) but engineering cost is high.
- **Warm tier memory overhead**: The FP16 warm tier (TriAttention-selected, pre-quantization) temporarily holds sparsified keys at full precision. This is transient (keys migrate to cold tier) but adds ~10% overhead during the staging window.
- **Compression interval granularity**: TriAttention triggers every 128 tokens. Cache may temporarily exceed budget between compression rounds. Acceptable for edge inference but requires careful buffer management.

### Risks

| Risk | Severity | Likelihood | Mitigation |
|------|----------|-----------|-----------|
| Quality degradation at 50x on reasoning tasks | High | Medium | Use ruvLLM delta checks and witness gating to detect error. Dynamically reduce sparsity budget if coherence threshold exceeded. Validate on AIME, MATH-500, RULER before release. |
| Low-MRL heads (~10%) degrade under aggressive sparsity | Medium | Medium | S_norm fallback scoring preserves content-dependent tokens for diffuse heads. Per-head adaptive budgets. |
| RoPE inversion latency on edge devices | Medium | Low | SIMD-optimized trigonometric evaluation (NEON for Apple Silicon/Pi 5, WASM SIMD for browsers). Amortized over 128-token intervals. |
| Calibration data quality affects TriAttention accuracy | Low | Low | Paper shows robustness to 50K-960K tokens from any domain (MRL 0.977-0.980 across Math, Coding, Chat). |
| TurboQuant error interaction with sparse subset | Low | Low | TurboQuant operates per-vector (not cross-vector), so sparsification does not affect its compression quality. Validated by MiniKV's 2-bit + sparsity co-design achieving 86% compression. |

## Implementation Plan (GOAP Milestones)

### World State Model

```
current_state = {
  turboquant_implemented: true,        // Phases 1-3 done
  triattention_implemented: false,
  stacked_pipeline: false,
  kv_cache_compression: 6x,            // TurboQuant only
  edge_128k_feasible: false,           // 1.75 GB too large
  fused_kernels: false,
  wasm_support: false,
  calibration_infra: false,
  quality_validated: false
}

goal_state = {
  turboquant_implemented: true,
  triattention_implemented: true,
  stacked_pipeline: true,
  kv_cache_compression: 50x,
  edge_128k_feasible: true,            // ~175 MB
  fused_kernels: true,
  wasm_support: true,
  calibration_infra: true,
  quality_validated: true
}
```

### Phase 1: TriAttention Calibration Infrastructure

**Files:** `crates/ruvllm/src/triattention/calibrate.rs`, `crates/ruvllm/src/triattention/mod.rs`

**Preconditions:** None (independent of TurboQuant)

**Deliverables:**
- `TriAttentionCalibration` struct: stores per-head Q/K center vectors E[q_f], E[k_f], magnitudes E[||q_f||], MRL values R_f, phase offsets
- Streaming calibration algorithm: accumulates statistics over calibration corpus without storing full dataset
- RVF serialization: save/load calibration files in ruvLLM's native format
- CLI calibration tool: `ruvllm calibrate-triattention --model <path> --corpus <path> --output <calibration.rvf>`

**Success Criteria:**
- Calibration completes on 50K tokens in <60s on Apple M-series
- MRL values match paper figures (>0.95 for ~90% of heads)
- Calibration files are <1 MB per model
- 8 unit tests passing

**Estimated Effort:** 2 weeks

### Phase 2: Trigonometric Scoring Engine with SIMD

**Files:** `crates/ruvllm/src/triattention/score.rs`

**Preconditions:** Phase 1 (calibration data available)

**Deliverables:**
- `TriAttentionScorer` struct: computes S_trig + S_norm for batches of cached keys
- RoPE inversion: recover pre-RoPE key representations from cached post-RoPE keys
- Geometric offset averaging: 17 offsets {1, 2, 4, ..., 2^16}
- GQA aggregation: z-score normalization + max across query group
- SIMD acceleration: `#[cfg(target_arch = "aarch64")]` NEON intrinsics for cos/sin evaluation, `#[cfg(target_arch = "x86_64")]` AVX2 fallback

**Success Criteria:**
- Scoring 8K cached keys completes in <1ms on Apple M-series
- Score ranking matches Python reference implementation (Spearman rho > 0.99)
- NEON path achieves >3x speedup vs scalar
- 10 unit tests + 2 integration tests passing

**Estimated Effort:** 3 weeks

### Phase 3: KV Cache Tier Integration

**Files:** `crates/ruvllm/src/triattention/cache.rs`, modifications to `crates/ruvllm/src/kv_cache.rs`

**Preconditions:** Phase 2 (scoring engine), existing `TurboQuantKvCache`

**Deliverables:**
- `TriAttentionCacheTier`: manages the warm tier with window-based pruning every 128 tokens
- Protected window: recent 128 tokens always preserved, never scored
- Budget management: configurable token budget B per layer
- Integration with `TurboQuantKvCache`: TriAttention warm tier feeds into TurboQuant cold tier
- `TriQuantKvCache`: unified three-tier cache (hot FP16 + warm TriAttention + cold TurboQuant)

**Success Criteria:**
- Cache correctly evicts lowest-scored tokens at 128-token intervals
- Protected window tokens never evicted
- Warm-to-cold tier migration works without data loss
- Memory usage matches theoretical predictions (+/- 10%)
- 12 unit tests + 3 integration tests passing

**Estimated Effort:** 3 weeks

### Phase 4: Stacked Pipeline with Quality Validation

**Files:** `crates/ruvllm/src/triattention/pipeline.rs`, `crates/ruvllm/tests/triattention_quality.rs`

**Preconditions:** Phase 3 (integrated cache), TurboQuant phases 1-3 (existing)

**Deliverables:**
- `StackedKvPipeline`: orchestrates TriAttention -> TurboQuant with coherence checkpoint between stages
- Delta check: detect excessive quality degradation between stages, dynamically adjust sparsity budget
- Per-head adaptive compression: heads with MRL > 0.95 get aggressive sparsity, low-MRL heads retain more tokens
- Quality benchmark suite: AIME24, AIME25, MATH-500, RULER retrieval, Needle-in-Haystack
- Compression ratio validation: confirm 30-50x on representative workloads

**Success Criteria:**
- Stacked pipeline achieves >30x compression on Qwen3-8B at 128K context
- Quality on MATH-500 within 5 percentage points of TurboQuant-only baseline
- Coherence checkpoint catches >95% of quality regressions
- End-to-end latency <2x vs TurboQuant-only
- 8 quality benchmarks + 6 unit tests passing

**Estimated Effort:** 4 weeks

### Phase 5: Fused Metal/NEON Kernels for Apple Silicon

**Files:** `crates/ruvllm/src/triattention/metal/`, `crates/ruvllm/src/triattention/neon.rs`

**Preconditions:** Phase 4 (validated pipeline)

**Deliverables:**
- Fused Metal compute shader: RoPE inversion + trigonometric scoring + top-K selection in single GPU dispatch
- Fused NEON kernel: combined scoring + eviction for CPU-only inference (Pi 5, non-GPU Apple Silicon)
- Kernel selection heuristic: Metal when GPU available, NEON fallback

**Success Criteria:**
- Metal kernel achieves >5x speedup vs scalar scoring on M-series GPU
- NEON kernel achieves >3x speedup vs scalar on ARM64
- Kernel output bit-exact with scalar reference implementation
- 4 kernel correctness tests + 2 performance benchmarks passing

**Estimated Effort:** 3 weeks

### Phase 6: WASM Compilation for Browser Inference

**Files:** `crates/ruvllm/src/triattention/wasm.rs`, wasm-pack configuration

**Preconditions:** Phase 4 (validated pipeline)

**Deliverables:**
- WASM-compatible TriAttention scoring (WASM SIMD for trigonometric evaluation)
- JavaScript bindings via wasm-bindgen for `TriQuantKvCache`
- Web Worker integration: scoring runs off main thread
- Memory-mapped calibration file loading via fetch API

**Success Criteria:**
- TriAttention scoring compiles to WASM without `std` dependencies that block compilation
- WASM SIMD path achieves >2x speedup vs scalar WASM
- Browser inference of 8K context completes scoring in <10ms
- 3 WASM integration tests passing in headless browser

**Estimated Effort:** 2 weeks

### Dependency Graph

```
Phase 1 (Calibration) ──> Phase 2 (Scoring) ──> Phase 3 (Cache Integration)
                                                        |
                                                        v
                                                 Phase 4 (Stacked Pipeline + Validation)
                                                   /              \
                                                  v                v
                                    Phase 5 (Metal/NEON)    Phase 6 (WASM)
```

Phases 5 and 6 are parallelizable after Phase 4 completes.

**Total Estimated Effort:** 17 weeks (Phases 5 and 6 parallel: 14 weeks critical path)

## Technical Details

### TriAttention Scoring Algorithm (Rust Pseudocode)

```rust
/// Compute TriAttention importance score for a cached key.
fn score_key(
    key: &[f32],           // Post-RoPE cached key, dim d
    calibration: &TriAttentionCalibration,
    delta: usize,          // Q-K positional distance
) -> f32 {
    let pre_rope_key = invert_rope(key, position);

    let mut s_trig = 0.0;
    let mut s_norm = 0.0;

    for f in 0..d/2 {
        let omega_f = calibration.rope_freqs[f];
        let q_center_mag = calibration.q_center_magnitudes[f];
        let k_mag = complex_magnitude(&pre_rope_key[2*f..2*f+2]);
        let phi_f = complex_phase(&calibration.q_centers[f])
                  - complex_phase(&pre_rope_key[2*f..2*f+2]);
        let mrl = calibration.mrl_values[f];

        // Trigonometric series score (Eq. 6)
        s_trig += q_center_mag * k_mag * (omega_f * delta as f32 + phi_f).cos();

        // Norm-based fallback score (Eq. 8)
        s_norm += (1.0 - mrl) * calibration.q_mean_magnitudes[f] * k_mag;
    }

    s_trig + s_norm
}

/// Multi-offset averaging with geometric spacing (Eq. 11)
fn score_key_averaged(key: &[f32], calibration: &TriAttentionCalibration, delta: usize) -> f32 {
    let offsets = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512,
                   1024, 2048, 4096, 8192, 16384, 32768, 65536];
    offsets.iter()
        .map(|&offset| score_key(key, calibration, delta + offset))
        .sum::<f32>() / offsets.len() as f32
}
```

### Three-Tier Cache State Machine

```
                 push()
New Token ──────────────> Hot Tier (FP16, 128 tokens)
                              |
                              | (hot tier full, oldest token migrates)
                              v
                         Warm Tier (FP16, TriAttention-scored)
                              |
                              | (every 128 tokens: score, evict below budget)
                              |──── evict ───> Discarded
                              |
                              | (survivors migrate on next compression round)
                              v
                         Cold Tier (TurboQuant 3.5-bit)
                              |
                              | (asymmetric attention: exact query * compressed key)
                              v
                         Attention Output
```

### Orthogonality Argument

TurboQuant operates **per-vector**: each key vector is independently rotated (Hadamard), quantized (PolarQuant), and residual-corrected (QJL). It does not use cross-vector statistics. Therefore, removing tokens from the KV cache via TriAttention does not affect TurboQuant's compression quality on the surviving tokens.

This is confirmed by MiniKV (ACL 2025), which achieves 86% compression by co-designing 2-bit quantization with token sparsity, and by the KV Cache Compression Survey (arxiv.org/abs/2508.06297) which states: "approaches that reduce per-pair footprint are orthogonal to those that reduce sequence length."

## Future Extensions

1. **Third axis -- SparK channel pruning**: Stack SparK's channel pruning (30%+ additional savings) for potential 70x total compression. Implementation as Phase 7 after the two-axis pipeline is validated.

2. **Coherence-gated compression**: Use RuVector's mincut coherence metric to dynamically adjust per-head sparsity budgets. High-coherence heads (high MRL) get aggressive pruning; low-coherence heads retain more tokens.

3. **Per-layer adaptive strategy**: Inspired by TailorKV, different transformer layers may prefer different compression mixes. Shallow layers (diffuse attention) may benefit from quantization-heavy compression, while deep layers (concentrated attention) may benefit from sparsity-heavy compression.

4. **Streaming calibration updates**: Continuously update TriAttention calibration statistics during inference to adapt to distribution shifts in long conversations.

5. **MLA latent compression**: For DeepSeek-V3 and future MLA models, apply TurboQuant on MLA's 576-dim latent vectors for additional compression on top of architectural savings.

## References

1. TriAttention (arXiv 2026): Mao et al., "TriAttention: Efficient Long Reasoning with Trigonometric KV Compression", arXiv:2604.04921
2. TurboQuant (ICLR 2026): arxiv.org/abs/2504.19874
3. PolarQuant (AISTATS 2026): arxiv.org/abs/2502.02617
4. QJL: arxiv.org/abs/2406.03482
5. MiniKV (ACL 2025): arxiv.org/abs/2411.18077
6. TailorKV (ACL 2025): arxiv.org/abs/2505.19586
7. KVTC (ICLR 2026): arxiv.org/abs/2511.01815
8. SparK (AAAI 2026): arxiv.org/abs/2508.15212
9. KVSculpt (2026): arxiv.org/abs/2603.27819
10. SALS (NeurIPS 2025): arxiv.org/abs/2510.24273
11. DeepSeek-V3 MLA: arxiv.org/abs/2412.19437
12. KV Cache Compression Survey: arxiv.org/abs/2508.06297
13. NVIDIA kvpress: github.com/NVIDIA/kvpress

## Related Documents

- [docs/research/quantization-edge/08-turboquant-kv-cache-compression.md](../research/quantization-edge/08-turboquant-kv-cache-compression.md) -- TurboQuant implementation details
- [docs/research/quantization-edge/09-triattention-kv-sparsity.md](../research/quantization-edge/09-triattention-kv-sparsity.md) -- TriAttention algorithm analysis
- [docs/research/quantization-edge/10-stacked-kv-compression.md](../research/quantization-edge/10-stacked-kv-compression.md) -- Multi-axis compression survey
- ADR-090: Ultra-Low-Bit Quantization Design
- `crates/ruvllm/src/quantize/turbo_quant.rs` -- Existing TurboQuant implementation
- `crates/ruvllm/src/kv_cache.rs` -- Existing KV cache infrastructure

## File Inventory (Planned)

| File | Description | Phase |
|------|-------------|-------|
| `crates/ruvllm/src/triattention/mod.rs` | Module root, public API | P1 |
| `crates/ruvllm/src/triattention/calibrate.rs` | Calibration infrastructure, RVF serialization | P1 |
| `crates/ruvllm/src/triattention/score.rs` | Trigonometric scoring engine, SIMD paths | P2 |
| `crates/ruvllm/src/triattention/cache.rs` | Warm tier, window-based pruning | P3 |
| `crates/ruvllm/src/triattention/pipeline.rs` | Stacked TriAttention->TurboQuant orchestration | P4 |
| `crates/ruvllm/src/triattention/metal/` | Metal compute shaders | P5 |
| `crates/ruvllm/src/triattention/neon.rs` | NEON intrinsics for ARM64 | P5 |
| `crates/ruvllm/src/triattention/wasm.rs` | WASM SIMD bindings | P6 |
| `crates/ruvllm/src/kv_cache.rs` | Modified: TriQuantKvCache three-tier cache | P3 |
| `crates/ruvllm/tests/triattention_quality.rs` | Quality benchmark suite | P4 |
