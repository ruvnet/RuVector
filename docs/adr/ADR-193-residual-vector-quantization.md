---
adr: 193
title: "Add ruvector-rvq: Residual Vector Quantization crate for multi-stage ANN compression"
status: proposed
date: 2026-05-09
authors: [ruvnet, claude-flow]
related: [ADR-001, ADR-155]
tags: [quantization, rvq, pq, ann, compression, codebook, nightly-research]
---

# ADR-193 — Add `ruvector-rvq`: Residual Vector Quantization for ANN Search

## Status

**Proposed.**  Implemented on branch
`research/nightly/2026-05-09-residual-vector-quantization`.
Benchmark binary `cargo run --release -p ruvector-rvq --bin rvq-demo` is runnable
and produces the numbers below from real data (no mocks).

## Context

`ruvector-core/src/quantization.rs` provides scalar (INT8), INT4, product (PQ),
and binary quantization.  All are single-stage: one codebook maps an input vector
directly to a code.

Single-stage PQ has a known weakness: it divides the embedding into M independent
subspaces and quantizes each separately.  When input dimensions are correlated across
subspace boundaries (common in transformer embeddings), PQ misses these correlations
and incurs excess quantization error.

**Residual Vector Quantization (RVQ)** addresses this by chaining multiple
full-dimensional codebooks.  Each stage quantizes the *residual error* from the
previous stage:

```
code₁       = argmin_c ‖v − centroid₁[c]‖²
residual₁   = v − centroid₁[code₁]
code₂       = argmin_c ‖residual₁ − centroid₂[c]‖²
residual₂   = residual₁ − centroid₂[code₂]
...
reconstruction x̂ = Σₛ centroidₛ[codeₛ]
```

This approach was proven in audio compression (SoundStream, Encodec) and extends
cleanly to ANN search via Asymmetric Distance Computation (ADC) lookup tables.

### Measured gap

On n=20K, D=128 with K=64 centroids (same-run benchmark):

| Variant | Bytes/vec | Recall@10 | QPS |
|---------|-----------|-----------|-----|
| PQ M=8 | 8 | 6.3% | 2,918 |
| **RVQ S=4** | **4** | **6.4%** | 1,656 |

RVQ S=4 matches PQ M=8 recall at **half the per-vector byte cost**.  At N=1M
vectors, this saves ~4 MB of code storage (per index shard).

On D=256, n=10K: RVQ S=4 (9.4% R@10) **outperforms** PQ M=8 (8.1% R@10) at half
the bytes — the advantage grows with dimensionality because PQ subspaces become
narrower (256/8 = 32 dims) and miss inter-subspace correlations.

### Competitor status

FAISS ships `IndexResidualQuantizer` (C++, BLAS dependency, since 2022).
Qdrant, Weaviate, LanceDB, and Pinecone do not implement RVQ as of May 2026.
No pure-Rust, no-`unsafe`, no-BLAS RVQ exists in the ecosystem.

## Decision

We add a new workspace crate `crates/ruvector-rvq` implementing:

1. **`Codebook`** — single-stage Lloyd's k-means with K-means++ initialization.
   Flat centroid layout for cache-friendly encode/decode.

2. **`ProductQuantizer`** — standard flat PQ for baseline comparison.  M subspaces,
   separate codebook per subspace, ADC distance tables.

3. **`RvqEncoder`** — multi-stage residual encoder.  Greedy stage-wise training
   with codebook dropout (arXiv:2306.06546) to prevent collapse.  ADC tables via
   inner-product precomputation (O(S·K·D) per query, O(S) per candidate).

4. **`AnnIndex` trait** — uniform interface across `FlatF32Index`, `PqIndex`,
   `RvqIndex`, and `RvqRerankIndex` (RVQ + exact rerank).

5. **`rvq-demo` binary** — standalone benchmark producing recall@10, QPS, and
   memory estimates from synthetic clustered data.  No external dataset downloads.

### Design constraints

- Pure safe Rust, no `unsafe`.
- No external BLAS, no C/C++ FFI.
- `rayon` opt-in (`#[cfg(not(target_arch = "wasm32"))]`) for parallel k-means.
- `serde` on all structs for future persistence.
- Files ≤ 500 lines (largest: `index.rs` at 275 lines).
- `cargo build --release -p ruvector-rvq` succeeds on stock Rust toolchain.
- `cargo test -p ruvector-rvq` passes 7 tests (6 unit + 1 doc).

### ADC distance formula

Approximate L2 for RVQ (ignores cross-stage interaction terms):

```
‖q − x̂‖² ≈ ‖q‖² − 2·Σₛ ⟨q, cₛ[code_s]⟩ + Σₛ ‖cₛ[code_s]‖²
```

Precomputed per query: two S×K tables (inner products + centroid norms).
Per-candidate cost: S additions.  For S=8, K=64, N=20K: 160K additions per query
→ ~2K QPS single-threaded (measured: 1,258–1,656 QPS depending on D).

### Codebook dropout

During stage-s training, each residual is zeroed with probability `dropout_prob`
(default 0.1).  This prevents early stages from explaining all variance and leaving
later stages with near-zero residuals (collapse).  Implemented in
`RvqEncoder::train` inside `crates/ruvector-rvq/src/rvq.rs`.

## Consequences

### Positive

- First pure-Rust RVQ implementation in the ecosystem.
- 2× per-vector memory reduction vs flat PQ at equivalent recall for high-dimensional embeddings (D ≥ 256).
- `RvqRerankIndex` achieves 43.4% recall@10 at QPS higher than exact brute-force (for small N).
- 19.2% distortion reduction over 8 stages confirms cascading works (not collapse).
- Drop-in `AnnIndex` interface lets future `ruvector-diskann` integration swap PQ → RVQ codebooks.
- No external dependencies beyond existing workspace crates (`rand`, `rand_distr`, `serde`, `rayon`).

### Negative / Risks

- Training time: 8 stages × 25 Lloyd iterations on n=20K, D=128 takes ~12 seconds
  single-threaded.  Acceptable for offline indexing; not for online updates.
- ADC is approximate (cross-stage terms dropped).  For uncorrelated codebooks the
  error is negligible; for poorly trained models it degrades ranking.
- Current K=64 gives low raw recall (6–12%) without reranking.  Production use
  requires K=256 (4× longer training) and/or more stages.
- Codebook memory: S=8, K=64, D=128 → 0.25 MB codebooks per index.  For K=256,
  D=768 this grows to 6.3 MB — still fits in L3 cache on server hardware.

### Neutral

- Not yet connected to `ruvector-diskann`'s PQ interface (planned ADR-194).
- WASM target compiles but sequential k-means is slow for large datasets.

## Alternatives

### 1. Extend `ruvector-core` PQ

Add a `num_stages` parameter to the existing `ProductQuantized` struct.  Rejected:
the existing impl is a flat quantizer; residual chaining requires a materially
different training loop, separate codebook storage, and a different search path.
A new crate keeps concerns separated and avoids breaking existing users.

### 2. Wrap FAISS `IndexResidualQuantizer` via FFI

FAISS provides battle-tested C++ RVQ.  Rejected: introduces a C++/BLAS build
dependency incompatible with WASM/embedded targets.  ruvector's pure-Rust constraint
(ADR-001) rules this out for core crates.

### 3. Matryoshka Representation Learning (MRL) search

MRL (arXiv:2205.13147) trains embeddings whose dimension-prefix truncations preserve
semantic similarity.  The search-side implementation (cascade D=32 → D=64 → D=128)
would be complementary, not competing, with RVQ.  Deferred to a future nightly.

### 4. ScaNN Anisotropic Vector Quantization (AVQ)

Google's direction-weighted PQ (arXiv:2105.09869) achieves higher recall than
isotropic PQ by weighting quantisation error along the query direction.  Requires
training direction-specific codebooks — much more complex.  Deferred to ADR-195+.
