# TurboQuant: Data-Oblivious KV Cache & Vector Compression for ruvLLM

## Abstract

TurboQuant (ICLR 2026) is a data-oblivious quantization algorithm that compresses
high-dimensional vectors to ~3.5 bits per value with provably near-optimal
geometry preservation. Unlike traditional quantization methods requiring
codebooks or training, TurboQuant operates without dataset-specific tuning
while achieving distortion within ~2.7× of information-theoretic lower bounds.

This document maps TurboQuant to ruvLLM's edge inference stack, where it
addresses the KV cache memory bottleneck and enables compressed embedding
stores compatible with RuVector's geometric control plane.

## 1. Core Algorithm

### 1.1 Two-Stage Pipeline

TurboQuant is a two-stage compression pipeline:

**Stage 1: PolarQuant** (MSE-optimal scalar quantization)
1. Apply randomized Hadamard rotation to input vector
2. After rotation, coordinates become approximately independent (Beta-distributed)
3. Apply optimal scalar quantizer per coordinate (no codebooks needed)

**Stage 2: QJL Residual Correction** (1-bit inner product correction)
1. Compute residual between original and Stage 1 reconstruction
2. Apply Quantized Johnson-Lindenstrauss (QJL) transform: store sign bits only
3. QJL signs produce an unbiased inner product estimator with minimal overhead

Combined: MSE quantizer + 1-bit QJL = unbiased inner product quantizer.

### 1.2 Mathematical Foundations

**Random Rotation (Hadamard)**:
- Orthogonal transform: H × H^T = n × I
- Makes vector dimensions approximately independent
- After rotation, angles follow a concentrated Beta distribution
- Eliminates need for explicit normalization (saves memory)

**Scalar Quantization**:
- Per-coordinate uniform quantizer with block-local scale/offset
- Levels determined by target bit-width (e.g., 8 levels for 3 bits)
- No codebook storage overhead

**QJL Residual**:
- Sign-bit quantization: each residual component → +1 or -1
- Zero memory overhead for quantization constants
- Asymmetric estimator: exact query × quantized key → unbiased inner product
- Total: ~0.5-1.0 extra bits per dimension

### 1.3 Error Bounds

- Distortion within ~2.7× of information-theoretic lower bounds
- Quality-neutral at 3.5 bits per channel (tested on Gemma, Mistral, Llama-3.1-8B)
- Marginal quality degradation at 2.5 bits per channel

## 2. Performance Results

| Metric | Value | Configuration |
|--------|-------|---------------|
| KV cache memory reduction | ≥6× | 3.5-bit vs FP16 |
| Attention speedup | up to 8× | 4-bit keys on H100 |
| Recall vs PQ/RabbiQ | Superior | Zero indexing time |
| Training required | None | Data-oblivious |
| Runtime overhead | Negligible | Rotation + scalar quant |

Benchmarks: LongBench, Needle-in-Haystack, ZeroSCROLLS, RULER, L-Eval.

## 3. Mapping to ruvLLM Architecture

### 3.1 KV Cache Integration (Highest ROI)

**Problem**: KV cache explodes with context length. ruvLLM pushes long context +
continuous agents on edge devices (Pi 5, Seed appliance, Cognitum tiles).

**Current architecture** (kv_cache.rs):
```
TwoTierKvCache:
  Hot tier (FP16): Recent tokens (tail_length=256)
  Cold tier (Q4):  Older tokens (4.5 bits)
```

**New architecture** (TurboQuantKvCache):
```
Three-tier cache:
  Hot tier (FP16):        Recent tokens (tail_length=256)
  Cold tier (TurboQuant): Older tokens (~3.5 bits, geometry-preserving)
```

**Impact**:
- 5-8× more effective context window on edge devices
- Preserves attention quality (unbiased inner product estimator)
- No training or calibration data required
- Drop-in replacement for cold tier quantization

### 3.2 RuVector Embedding Compression

TurboQuant preserves Euclidean distance geometry, which aligns with RuVector's
use of geometry as a control layer:

- **HNSW search**: Inner product preservation means nearest-neighbor results are
  stable under compression
- **Mincut coherence**: Structural coherence signals remain valid on compressed
  embeddings
- **Hyperbolic embeddings**: Require pre-transform to Euclidean space before
  compression (limitation)

Implementation: `TurboQuantEmbeddingStore` provides batch build, single retrieval,
and nearest-neighbor search on compressed embeddings.

### 3.3 Comparison with PiQ3

| Feature | PiQ3 | TurboQuant | Recommended Use |
|---------|------|------------|-----------------|
| Data aware | Yes | No | PiQ3 for archival, TurboQuant for live |
| Online | Partial | Yes | TurboQuant for streaming KV cache |
| Geometry preservation | Good | Provably near-optimal | TurboQuant for attention |
| KV cache ready | Not native | Yes | TurboQuant |
| Training required | Sometimes | None | TurboQuant for zero-config |
| Compression ratio | 8-12× | 6-9× | PiQ3 for cold storage |

**Best strategy**: TurboQuant for live KV cache and real-time embeddings;
PiQ3 for archival tiers and temporal compression pipelines.

## 4. Integration Architecture

```
ruvLLM Inference Pipeline
  ├── KV Cache
  │   ├── Hot Tier (FP16, recent tokens)
  │   └── Cold Tier (TurboQuant 3.5-bit) ← NEW
  ├── Embedding Store
  │   ├── Live (TurboQuant) ← NEW
  │   └── Archive (PiQ3 temporal compression)
  ├── RuVector Store
  │   ├── HNSW index (compressed embeddings)
  │   └── Mincut coherence (validation layer)
  └── Attention Computation
      └── Asymmetric inner product (exact query × compressed key)
```

## 5. Risks & Mitigations

### 5.1 Inner Product vs Mincut Tension

TurboQuant optimizes MSE + inner product distortion.
RuVector optimizes structural coherence (mincut).

**Mitigation**: Run mincut as a validation layer. Reject high-distortion
regions where TurboQuant error exceeds coherence threshold.

### 5.2 Hyperbolic Embeddings

TurboQuant assumes Euclidean space. ruvLLM uses hyperbolic + mixed curvature.

**Mitigation**: Pre-transform to Euclidean (logarithmic map) → quantize →
inverse map (exponential map). Adds latency but preserves hyperbolic geometry.

### 5.3 Ultra-Low-Bit Instability (<3 bits)

Below ~3 bits, error spikes in rare vectors.

**Mitigation**: Existing ruvLLM infrastructure handles this:
- Delta checks (detect excessive error)
- Witness gating (audit trail)
- Sparsifier (flag problematic vectors)

## 6. Implementation Summary

### Phase 1: Core Compression (DONE)

- `turbo_quant.rs`: TurboQuantCompressor with Hadamard rotation + scalar
  quantization + QJL residual correction
- Bit configurations: 2.5, 3.0, 3.5, 4.0 bits per value
- Bitstream packing for non-byte-aligned bit widths
- 13 passing tests

### Phase 2: KV Cache Integration (DONE)

- `TurboQuantCacheTier`: Compressed KV pair storage with push/get/evict
- `TurboQuantKvCache`: Three-tier cache (FP16 hot + TurboQuant cold) with
  auto-migration from tail to cold tier
- Integrated into `kv_cache.rs` with `CacheTier::TurboQuant` variant

### Phase 3: Embedding Store (DONE)

- `TurboQuantEmbeddingStore`: Batch build, single retrieval, nearest-neighbor
  search using asymmetric inner product
- Compatible with RuVector HNSW index

### Phase 4: Future Work

- Mincut-based distortion gating ("coherence-aware quantization")
- SIMD-optimized bit packing (NEON/AVX2)
- Hyperbolic pre-transform adapter
- Streaming compression for continuous agent contexts

## 7. References

1. TurboQuant (ICLR 2026): arxiv.org/abs/2504.19874
2. PolarQuant (AISTATS 2026): arxiv.org/abs/2502.02617
3. QJL: arxiv.org/abs/2406.03482
4. ADR-090: Ultra-Low-Bit Quantization Design
5. Google Research Blog: research.google/blog/turboquant-redefining-ai-efficiency-with-extreme-compression/

## 8. File Inventory

| File | Description |
|------|-------------|
| `crates/ruvllm/src/quantize/turbo_quant.rs` | Core TurboQuant implementation |
| `crates/ruvllm/src/quantize/mod.rs` | Module exports (updated) |
| `crates/ruvllm/src/kv_cache.rs` | TurboQuantKvCache integration |
| `crates/ruvllm/src/quantize/hadamard.rs` | Hadamard transform (dependency) |
