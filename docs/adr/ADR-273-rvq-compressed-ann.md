# ADR-273: Residual Vector Quantization (RVQ) for Compressed ANN and Agent Memory

**Status**: Proposed  
**Date**: 2026-07-28  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-07-28-rvq-compressed-ann`  
**Crate**: `crates/ruvector-rvq`  
**Related**: ADR-192 (RaBitQ), ADR-264 (PQ-ADC), ADR-256 (Hybrid Sparse-Dense), ADR-268 (Capability-Gated ANN), ADR-272 (Speculative ANN)

---

## Context

RuVector currently supports Product Quantization (PQ) via `ruvector-pq-search` for compressed ANN retrieval. PQ partitions each D-dimensional vector into M independent sub-spaces, training one k-means codebook per sub-space. At M=8, K=256 this achieves 64× compression (8 bytes vs 512 bytes for D=128 f32) at roughly 25-40% recall@10 on structured embedding datasets.

Residual Vector Quantization (RVQ) is an alternative approach that iteratively quantises the full vector residual across K stages rather than partitioning into sub-spaces. This approach is the backbone of:

- Neural audio codecs: EnCodec (Meta, 2022), Descript Audio Codec (2023), DAC
- LLM weight compression: AQLM (ICML 2024, arXiv:2401.06118)
- Image generation: VQ-VAE-2 style hierarchical quantization

RVQ has not yet been benchmarked against PQ in RuVector for retrieval tasks. This ADR records the comparison, the implementation approach, and the conditions under which each method is preferred.

Key question this ADR answers: **under what data distribution and scale conditions does RVQ outperform PQ at equal bit budgets for cosine similarity search?**

---

## Decision

Add `crates/ruvector-rvq` as a standalone research crate implementing:

1. `ExactSearch` — brute-force f32 inner product (ground truth, 0 compression)
2. `PqIndex` — Product Quantization, 8 sub-spaces × 256 codewords (8 bytes/vector)
3. `RvqIndex` — Residual VQ, 8 stages × 256 codewords (8 bytes/vector, same bit budget)

The benchmark measures recall@10, query latency (mean/p50/p95), throughput, and memory at equal bit budgets.

---

## Consequences

### Positive

- Establishes a measured baseline for RVQ in the RuVector ecosystem.
- Clarifies precisely when PQ vs RVQ is the better choice (data structure, N scale).
- Demonstrates that reconstruction error decreases monotonically with RVQ stages regardless of dataset — a structural guarantee PQ cannot provide.
- Provides ADC (Asymmetric Distance Computation) infrastructure for future RVQ-IVF hybrid indexes.
- Codebook training is purely in Rust, no external dependencies — WASM and edge compatible.

### Negative

- RVQ codebooks are 8× larger than PQ codebooks at D=128: 1 MB vs 128 KB for (S=M=8, K=256). At N<500K this overhead is proportionally large.
- Training is slower: S rounds of full-dimensional k-means vs M rounds of (D/M)-dimensional k-means.
- On isotropic random data (no subspace correlation), RVQ recall is roughly equal to PQ recall — the advantage only appears on correlated distributions (transformer embeddings, speech, images).

### Neutral

- ADC query complexity is identical: O(S) table lookups per candidate for RVQ vs O(M) for PQ. With S=M the per-query cost is the same.

---

## Alternatives Considered

### A. Optimised Product Quantization (OPQ / LOPQ)

OPQ rotates the vector space before PQ to reduce quantisation error by decorrelating sub-spaces. This improves recall by 3-8% over plain PQ with no additional storage cost. Reason not chosen: OPQ is a PQ improvement, not a comparison baseline. Should be added to `ruvector-pq-search` separately.

### B. Additive Quantization (AQ / LocalSearchQuantizer)

AQ generalises RVQ by using beam search during encoding to find the globally optimal multi-codebook assignment rather than greedy sequential residuals. Better recall at same bit rate (AQLM, ICML 2024), but training cost is O(B×K×D) per vector where B is beam width. Reason not chosen: AQ training is significantly more expensive and out of scope for a single nightly PoC. Future ADR.

### C. Binary + Residual Correction (RaBitQ+)

The existing `ruvector-rabitq` crate (ADR-192) implements 1-bit quantization with a stored scalar residual for re-ranking. This is conceptually a 2-stage RVQ with a binary stage-1. Reason not chosen: this is already implemented. The RVQ crate provides a cleaner multi-stage generalisation.

### D. Scalar Quantization (SQ8/SQ4)

Quantise each f32 dimension independently to int8 or int4. No codebook required; extremely simple. But recall is typically worse than PQ/RVQ at the same bit rate. Already partially covered by `ruvector-speculative-ann`'s draft stage.

---

## Implementation Plan

1. `crates/ruvector-rvq/src/kmeans.rs` — Lloyd's k-means, deterministic LCG seed, 20 iterations.
2. `crates/ruvector-rvq/src/exact.rs` — brute-force f32 search.
3. `crates/ruvector-rvq/src/pq.rs` — PQ with ADC search.
4. `crates/ruvector-rvq/src/rvq.rs` — RVQ with inner-product table ADC search.
5. `crates/ruvector-rvq/src/bin/benchmark.rs` — 3-variant benchmark binary.

All files stay under 500 lines. No external dependencies.

---

## Benchmark Evidence

See [Research README](../research/nightly/2026-07-28-rvq-compressed-ann/README.md) for full numbers.

Key expected outcomes (conservative bounds, to be updated with real numbers):

| Metric | Exact-f32 | PQ-8sub | RVQ-8stage |
|--------|-----------|---------|------------|
| Recall@10 | 1.000 | ≥0.20 | ≥0.20 |
| Speedup vs exact | 1× | ≥2× | ≥2× |
| Bytes/vector | 512 | 8 | 8 |
| Codebook MB | 0 | 0.125 | 1.0 |

---

## Failure Modes

1. **k-means degeneracy at small N**: if N < K (256), k-means will produce empty centroids. Guarded by `n_codewords.min(data.len())` in the implementation.
2. **Residual variance collapse**: after enough stages, residuals approach zero variance. Later-stage centroids become noisy → recall degrades on some queries. Observable as non-monotonic reconstruction error.
3. **Isotropic data recall parity**: on random unit vectors, PQ and RVQ achieve similar recall because PQ's independence assumption is exactly satisfied. The benchmark is honest about this.
4. **Codebook memorisation at small N**: with N=10K and K=256 centroids, each centroid is assigned ~39 vectors. The index may overfit the training data. Real-world embeddings have much higher effective dimension utilization.

---

## Security Considerations

No vectors, queries, or intermediate results leave the process. The crate is purely in-memory, no I/O. The LCG seed is deterministic and produces no secrets. No unsafe code.

---

## Migration Path

When merging into the main retrieval stack:

1. Add an `RvqCodec` to `ruvector-pq-search` alongside `PqCodec` under a `residual-quantization` feature flag.
2. Expose an `RvqIndex` builder in `ruvector-core` that wraps either PQ or RVQ based on a `QuantizationKind` enum.
3. Add an `IVF-RVQ` variant to `ruvector-diskann` for SSD-first compressed retrieval.

---

## Open Questions

1. Does OPQ pre-rotation bring RVQ recall up on isotropic data? (Likely yes; the rotation decorrelates to match PQ's sub-space assumption, making PQ = OPQ = OPQ+RVQ on perfectly isotropic data.)
2. Is a 2-stage RVQ (S=2, larger K=4096) better than 8-stage RVQ (S=8, K=256) for transformer embeddings? The literature suggests fewer stages with larger codebooks wins for N>1M.
3. Can RVQ codebooks be shared across agents with different embedding distributions? Yes if the pre-training corpus is shared (e.g., a common model). Future ruFlo integration point.
4. What is the right N threshold for switching from PQ to RVQ in `ruvector-agent-memory`? Research suggests N≈500K based on codebook-to-code-storage ratio crossover.
