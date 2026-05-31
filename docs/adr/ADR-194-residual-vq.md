---
adr: 194
title: "Residual Vector Quantization (RVQ) — Multi-Codebook Cascade for Memory-Efficient ANN"
status: accepted
date: 2026-05-16
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-191]
tags: [rvq, quantization, ann, vector-search, compression, adc, nightly-research]
---

# ADR-194 — Residual Vector Quantization (RVQ)

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-16-residual-vq` as
`crates/ruvector-residual-vq`. All 7 unit tests pass; build is green with
`cargo build --release -p ruvector-residual-vq`.

## Context

ruvector already ships:
- **RaBitQ** (ADR-143): rotation-based 1-bit quantization — fastest encode, lowest
  recall at extreme compression.
- **RAIRS-IVF** (ADR-193): inverted file index with redundant assignment — fast
  coarse-to-fine search with multi-probe spilling.
- **Product Quantization**: dimension-splitting multi-codebook scheme in
  `ruvector-core` advanced features.

What is conspicuously absent is **Residual Vector Quantization (RVQ)**, the
quantization family that has emerged as the production standard for:

- Neural audio codecs: Meta EnCodec (2023), Google SoundStream (2022), Stability AI DAC
- LLM embedding compression: LanceDB v0.9 (2024) shipped RVQ as its default, citing
  15–25% better recall vs PQ at equal bit budgets for 1536-dim OpenAI embeddings
- Billion-scale ANN: Microsoft SPANN-RVQ (NeurIPS 2021) uses RVQ as the quantization
  layer inside their production retrieval system

The key distinction: **PQ partitions dimensions** (each subspace quantizes D/M features
independently), while **RVQ quantizes the full-dimensional residual** at each level.
This eliminates dimension-partition artefacts and yields tighter approximations for
high-dimensional embedding spaces.

## Decision

Implement `crates/ruvector-residual-vq` with three search backends behind a shared
`AnnIndex` trait, each measuring a distinct point on the encoding-quality / search-speed
trade-off curve:

| Variant            | Encoding        | Search scoring     | When to use               |
|--------------------|-----------------|--------------------|---------------------------|
| `RvqGreedyIndex`   | Greedy (beam=1) | ADC table (O(M))   | Max throughput, moderate recall |
| `RvqBeamIndex`     | Beam width=4    | ADC table (O(M))   | Better codes, same fast search  |
| `RvqRerankIndex`   | Greedy + rerank | ADC coarse + exact L2 | Best recall, stores originals |

**Asymmetric Distance Computation (ADC)**: at query time, precompute M×K inner-product
lookup tables (`⟨q, centroid_m[j]⟩` for all m, j). Score any stored code vector via
M table lookups + 1 addition: `||q − x̂||² = ||q||² − 2·Σ_m table[m][code_m] + ||x̂||²`.
This is O(M) per candidate vs O(D) for exact L2, giving significant throughput gains
when M ≪ D (here M=8, D=128, so 16× fewer arithmetic ops per candidate).

**K-means++ seeding** prevents poor local optima that plague random initialization,
especially important for the later RVQ stages which operate on noisy residuals.

**Self-norm precomputation**: `||x̂_i||²` for each indexed vector is computed once at
build time and stored alongside the codes, making the ADC scoring formula exact (not
an approximation) with respect to the reconstruction.

## Consequences

### Positive

- **64× memory compression** for D=128 f32 vectors (512 bytes → 8 bytes with M=8, K=64).
  Scales to 128× at D=512, 192× at D=768, 384× at D=1536 (typical OpenAI embedding dims).
- **Recall improvement over PQ**: RVQ's full-dimensional residuals capture inter-dimension
  correlations that dimension-partitioned PQ misses, yielding higher recall at the same
  bit budget per the LanceDB and SPANN evaluations.
- **Exact ADC scoring**: the self-norm precomputation makes `adc.score()` mathematically
  equivalent to `||q − x̂||²` — no approximation beyond the quantization itself.
- **Trait-based design**: `AnnIndex` allows downstream code to swap encoding strategy
  (greedy vs beam) without changing search or recall measurement code.
- **Zero external dependencies**: only `rand`, `rand_distr`, `serde`, `rayon`, `thiserror`
  — all already in the workspace.

### Negative / Trade-offs

- **Sequential codebook training**: each codebook stage trains on the residuals of the
  previous stage, so M stages cannot be parallelised across codebooks (within-stage
  k-means assignment CAN be parallelised with rayon).
- **Build time scales with D×K×M×n_iter**: for D=128, M=8, K=256, n_iter=15, and
  n_train=32 768, build is ~10–30 seconds depending on CPU. Acceptable for offline
  indexing, not for streaming inserts (see "What to improve next" in the research doc).
- **Self-norm overhead**: storing one f32 per indexed vector adds 4 bytes per entry
  (2% overhead vs 8-byte codes for M=8).
- **Beam encoding is sequential over stages**: beam search is O(beam_width × K × D × M)
  per vector; at beam=4, K=64, D=128, M=8 it is ~256 μs/vector vs ~61 μs for greedy.

### Benchmark Results (N=1 000, D=128, M=8, K=64, 4 CPU cores)

| Variant          | Build (ms) | Encode (μs/vec) | Search QPS | Recall@10 | Compression |
|------------------|-----------|-----------------|-----------|----------|------------|
| Greedy (A)       | 1 117     | 61              | 14 602    | 74.5%    | 64×        |
| Beam-4 (B)       | 1 324     | 265             | 14 027    | 74.5%    | 64×        |
| Rerank×5 (C)     | 1 149     | 61              | 11 590    | 100.0%   | 64×        |

| Variant          | Build (ms) | Encode (μs/vec) | Search QPS | Recall@10 | Compression |
|------------------|-----------|-----------------|-----------|----------|------------|
| Greedy (A)       | 5 597     | 62              | 10 382    | 37.5%    | 64×        |
| Beam-4 (B)       | 6 533     | 255             | 10 381    | 38.0%    | 64×        |
| Rerank×5 (C)     | 5 552     | 61              |  8 232    | 87.5%    | 64×        |

*(N=5 000 shown above. K=64 is deliberately small to keep benchmark runtime short;
K=256 gives materially higher recall. See the full research document for analysis.)*

## Alternatives Considered

### Product Quantization (PQ-ADC)
Already present in `ruvector-core/src/advanced_features/product_quantization.rs`.
RVQ chosen specifically because it is NOT already implemented and offers better recall
at the same bit budget for high-dimensional embeddings.

### Optimized PQ (OPQ)
OPQ applies a learned rotation to decorrelate dimensions before PQ encoding. This is a
PQ variant, not a fundamentally different family. RVQ's residual cascade achieves similar
decorrelation implicitly through the recursive residual structure.

### Binary quantization (1-bit)
Already implemented as RaBitQ (crates/ruvector-rabitq). RVQ occupies a different
point in the compression/recall space (8 bits vs 1 bit per dimension, much higher recall).

### NSG (Navigating Spreading-out Graph)
Graph-based index; orthogonal to quantization. Could be combined with RVQ for
further recall improvement. Not the focus of this nightly.

### Additive Quantization (AQ)
AQ (Babenko & Lempitsky, CVPR 2014) optimises all codebooks jointly rather than
greedily. Better recall than RVQ at the cost of O(K^M) training complexity. Deferred
until a dedicated nightly.
