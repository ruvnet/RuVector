# Stacked KV Cache Compression: TriAttention × TurboQuant × SparK

## Abstract

This document analyzes the emerging field of **multi-axis KV cache compression**
where orthogonal techniques are stacked for multiplicative memory reduction.
We identify three independent compression axes — token sparsity, bit quantization,
and dimension compression — and map their composability to ruvLLM's existing
infrastructure.

The key opportunity: no published work has combined TriAttention (10.7× sparsity)
with TurboQuant (6× quantization). This represents a natural first-mover target
for ruvLLM, enabling 30-50× KV cache reduction on edge devices.

## 1. Three Orthogonal Compression Axes

### 1.1 Taxonomy

| Axis | What It Reduces | Analogy | Methods |
|------|----------------|---------|---------|
| **Token sparsity** | Sequence length (rows) | Delete spreadsheet rows | TriAttention, H2O, SnapKV, StreamingLLM |
| **Bit quantization** | Bits per element (cell size) | Shrink each cell's format | TurboQuant, KIVI, GEAR, KVQuant, NVFP4 |
| **Dimension compression** | Feature dims (columns) | Delete spreadsheet columns | MLA, SparK, KVTC PCA, SALS, Lexico |

### 1.2 Multiplicative Principle

The composite compression ratio is:

```
c_total = c_sparsity × c_quantization × c_dimension
```

Confirmed by the KV Cache Compression Survey (arxiv.org/abs/2508.06297).
However, Q-Hitter warns that "simplistic amalgamation can yield sub-optimal
performance" — interaction effects matter.

### 1.3 Theoretical Ceiling

| Configuration | Sparsity | Quantization | Dimension | Total |
|---------------|----------|-------------|-----------|-------|
| Conservative | 3× | 4× (4-bit) | 1× | **12×** |
| Moderate | 5× | 5× (3.5-bit) | 1× | **25×** |
| Aggressive | 10× | 5× (3.5-bit) | 1× | **50×** |
| Maximum | 10× | 5× (3.5-bit) | 1.4× (SparK) | **70×** |

Quality-preserved practical ceiling: **30-50×** based on published results.

## 2. Existing Stacked Systems (State of the Art)

### 2.1 MiniKV (ACL 2025 Findings)

**Axes:** Token sparsity + 2-bit quantization (co-designed)

- Pyramid token selection + heavy-hitter retention during prefill
- 2-bit asymmetric quantization (subchannel keys, per-token values)
- Two-pass Triton kernel: fused dequant + attention
- **Result:** 86% KV compression, 44K tokens on single A100, 48% throughput gain
- **Key insight:** Naive bolt-on fails — must co-design eviction to preserve
  quantization group boundaries

Paper: arxiv.org/abs/2411.18077

### 2.2 TailorKV (ACL 2025 Findings)

**Axes:** Layer-discriminative sparsity + 1-bit quantization

- Computes "dense preference score" per transformer layer
- Shallow layers (diffuse attention): compress to 1-bit quantization
- Deep layers (concentrated attention): offload to CPU, Top-K retrieval
- **Result:** 34.2× compression on Llama-3.1-8B at 128K context, single RTX 3090
- **Key insight:** Different layers prefer different compression axes

Paper: arxiv.org/abs/2505.19586 | GitHub: github.com/ydyhello/TailorKV

### 2.3 SparK (AAAI 2026, AMD)

**Axes:** Channel pruning (composable third axis)

- Prunes KV cache channels with dynamic recovery during attention
- Explicitly "orthogonal to existing compression techniques"
- Designed to stack on top of quantization or token-eviction
- **Result:** Additional 30%+ memory savings on top of other methods

Paper: arxiv.org/abs/2508.15212

### 2.4 KVTC (ICLR 2026, NVIDIA)

**Axes:** PCA dimensionality reduction + adaptive bit allocation + entropy coding

- Three-stage transform coder inspired by JPEG:
  1. PCA decorrelation (removes RoPE before PCA, reapplies after)
  2. Dynamic programming for optimal bit allocation
  3. DEFLATE/LZMA2 entropy coding
- **Result:** 20× compression, up to 40× for specific use cases
- Reduces TTFT by up to 8× for long contexts
- Tested: Llama 3, Mistral NeMo, R1-Qwen 2.5

Paper: arxiv.org/abs/2511.01815 | GitHub: github.com/OnlyTerp/kvtc

### 2.5 KVSculpt (March 2026)

**Axes:** Distillation (reduce tokens) + quantization (reduce bits)

- Optimizes a smaller set of unconstrained KV pairs via L-BFGS (keys)
  and least-squares (values)
- Confirms: "approaches that reduce per-pair footprint are orthogonal to
  those that reduce sequence length"
- Quantization can be applied on top of distilled cache

Paper: arxiv.org/abs/2603.27819

## 3. Serving Framework Support

| Framework | Quantization | Sparsity | Composable | Status |
|-----------|-------------|----------|------------|--------|
| **vLLM** | FP8, FP4 (prod) | RFC stage | No | Active development |
| **TensorRT-LLM** | FP8, INT8, NVFP4 | Block eviction | No | NVIDIA focus |
| **llama.cpp** | Q4_0, Q8_0, TurboQuant | None built-in | No | Community |
| **SGLang** | FP8, FP4 | DoubleSparsity | Closest | Both available |
| **NVIDIA kvpress** | via HF QuantizedCache | 30+ methods | **Yes** | Research lib |
| **aither-kvcache** | TurboQuant engine | TriAttention engine | **Separate** | v2.0 |

**Gap:** No framework stacks TriAttention + TurboQuant in a unified pipeline.

## 4. The TriAttention + TurboQuant Pipeline for ruvLLM

### 4.1 Why This Combination

1. **Orthogonality confirmed:** TriAttention reduces token count (rows);
   TurboQuant reduces bit width (columns). No mathematical interference.

2. **Both training-free:** Neither requires fine-tuning or calibration datasets
   (TriAttention needs lightweight offline calibration, not training).

3. **Both implemented/researched:** TurboQuant is already in ruvLLM (phases 1-3);
   TriAttention has a clean Python reference implementation.

4. **No published combination:** First-mover opportunity.

5. **Edge-device fit:** ruvLLM targets Pi 5, Seed appliance, Cognitum tiles —
   exactly where 50× KV reduction is transformative.

### 4.2 Pipeline Architecture

```
┌─────────────────────────────────────────────────────┐
│                  Token Generation                    │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │ Stage 1: TriAttention (Token Sparsity)          │ │
│  │                                                  │ │
│  │  1. Invert RoPE on cached keys                  │ │
│  │  2. Compute S_trig (trigonometric distance)      │ │
│  │  3. Compute S_norm (MRL-weighted norms)          │ │
│  │  4. Average over geometric future offsets        │ │
│  │  5. GQA: z-normalize, max aggregate             │ │
│  │  6. Retain top-B keys, evict rest               │ │
│  │                                                  │ │
│  │  Result: 10× fewer tokens in cache              │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│  ┌─────────────────────▼───────────────────────────┐ │
│  │ Stage 2: TurboQuant (Bit Quantization)          │ │
│  │                                                  │ │
│  │  1. Hadamard rotation (independence)             │ │
│  │  2. PolarQuant scalar quantization (3.5 bits)    │ │
│  │  3. QJL residual correction (1 bit)              │ │
│  │                                                  │ │
│  │  Result: ~5× smaller per surviving token         │ │
│  └─────────────────────┬───────────────────────────┘ │
│                        │                              │
│  ┌─────────────────────▼───────────────────────────┐ │
│  │ KV Cache: ~50× compressed                       │ │
│  │                                                  │ │
│  │  Hot:  FP16, recent 128 tokens (protected)       │ │
│  │  Warm: TriAttention-selected, FP16 (pre-quant)   │ │
│  │  Cold: TriAttention-selected + TurboQuant (3.5b)  │ │
│  └─────────────────────────────────────────────────┘ │
└─────────────────────────────────────────────────────┘
```

### 4.3 Memory Impact (Concrete Example)

**Model:** Qwen3-8B, 128K context, batch=1

| Configuration | KV Cache Size | Reduction |
|---------------|--------------|-----------|
| FP16 (baseline) | ~8 GB | 1× |
| TurboQuant only (3.5-bit) | ~1.75 GB | 4.6× |
| TriAttention only (10% budget) | ~0.8 GB | 10× |
| **Stacked (TriAttention + TurboQuant)** | **~175 MB** | **~46×** |

This fits 128K context on a Raspberry Pi 5 (8GB RAM) with room for model
weights (quantized).

### 4.4 Interaction Risks

| Risk | Severity | Mitigation |
|------|----------|-----------|
| Evicted key was important | Medium | TriAttention exceeds Full Attention at 30% budget — redundancy removal helps |
| TurboQuant error on sparse subset | Low | TurboQuant operates per-vector, independent of neighbor count |
| Kernel fusion complexity | High | MiniKV demonstrates fused Triton kernels are feasible |
| Calibration data coverage | Low | TriAttention calibration robust to 50K-960K tokens, any domain |
| Debug difficulty | Medium | Add coherence checkpoint between stages (delta check) |

## 5. Additional Methods for Future Integration

### 5.1 SALS (NeurIPS 2025)

Projects KV cache into compact latent space via low-rank projection, then
performs sparse token selection with RoPE-free Q-K interactions. 6.4× compression,
5.7× attention speedup. Combines dimensionality reduction + token sparsity in
unified pipeline.

Paper: arxiv.org/abs/2510.24273

### 5.2 MLA (DeepSeek-V3)

Architectural approach: compresses KV cache to 576-dim latent vector (from
40,960-dim). 98.6% reduction / 71× memory savings per layer. Orthogonal to
inference-time compression — could apply TurboQuant on top of MLA latents.

Paper: arxiv.org/abs/2412.19437

### 5.3 Lexico (ICML 2025)

Sparse coding over universal dictionaries (~4K atoms) with orthogonal matching
pursuit. 90-95% original performance at 15-25% memory. Outperforms both
quantization and token eviction in extreme low-memory regimes.

Paper: arxiv.org/abs/2412.08890

### 5.4 SQuat

Constructs subspace spanned by query tensors, enforces quantization error
orthogonal to this subspace. Minimizes impact on attention outputs. No
fine-tuning or calibration required.

Paper: arxiv.org/abs/2503.24358

## 6. Research Gaps (Opportunities for ruvLLM)

1. **Three-axis fused kernel**: No system stacks all three axes (sparsity +
   quantization + dimension) with fused CUDA/Metal kernels.

2. **TriAttention + TurboQuant**: Natural combination, confirmed orthogonal,
   not attempted in any published work.

3. **Coherence-gated compression**: Using RuVector's mincut coherence as a
   quality signal to dynamically adjust sparsity/quantization aggressiveness
   per layer or per head. Heads with high MRL → more aggressive sparsity.
   Low-coherence regions → preserve more tokens.

4. **Edge-first stacked compression**: All published stacked systems target
   datacenter GPUs. ruvLLM's edge focus (Pi 5, Apple Silicon, WASM) is unique.

5. **50× regime quality validation**: Only KVTC (20-40×) and TailorKV (34.2×)
   have published results beyond 20×. The 50× regime with broad benchmark
   fidelity is open territory.

## 7. Implementation Roadmap

| Phase | Description | Depends On |
|-------|-------------|-----------|
| P1 | TriAttention calibration infrastructure | — |
| P2 | TriAttention scoring engine (Rust) | P1 |
| P3 | TriAttention KV cache tier | P2, existing kv_cache.rs |
| P4 | Stacked TriAttention→TurboQuant pipeline | P3, existing TurboQuant |
| P5 | SIMD-optimized trigonometric kernels (NEON/AVX2) | P4 |
| P6 | SparK channel pruning (third axis) | P4 |
| P7 | Fused Metal/CUDA kernels | P5, P6 |
| P8 | Quality validation suite (AIME, MATH, RULER) | P4 |

## 8. References

1. KV Cache Compression Survey: arxiv.org/abs/2508.06297
2. MiniKV (ACL 2025): arxiv.org/abs/2411.18077
3. TailorKV (ACL 2025): arxiv.org/abs/2505.19586
4. SparK (AAAI 2026): arxiv.org/abs/2508.15212
5. KVTC (ICLR 2026): arxiv.org/abs/2511.01815
6. KVSculpt (2026): arxiv.org/abs/2603.27819
7. SALS (NeurIPS 2025): arxiv.org/abs/2510.24273
8. DeepSeek-V3 MLA: arxiv.org/abs/2412.19437
9. Lexico (ICML 2025): arxiv.org/abs/2412.08890
10. TurboQuant (ICLR 2026): arxiv.org/abs/2504.19874
11. TriAttention (2026): arxiv.org/abs/2604.04921
12. NVIDIA kvpress: github.com/NVIDIA/kvpress
13. aither-kvcache: pypi.org/project/aither-kvcache/2.0.0/
14. SQuat: arxiv.org/abs/2503.24358
15. ThinKV: arxiv.org/abs/2510.01290
16. CacheGen (SIGCOMM 2024): arxiv.org/abs/2310.07240
17. Q-Hitter: arxiv.org/abs/2508.06297
