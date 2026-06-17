# ADR-194: Hybrid Sparse-Dense Search with Reciprocal Rank Fusion

**Status**: Proposed  
**Date**: 2026-05-30  
**Deciders**: ruvnet  
**Supersedes**: —  
**Superseded by**: —

---

## Context

RuVector currently provides dense approximate nearest-neighbour (ANN) search
(RaBitQ, ACORN-HNSW, RAIRS-IVF) and BM25 sparse retrieval as separate,
independently-operated modules.  Production RAG pipelines require both: dense
retrieval finds semantically similar passages; sparse retrieval finds exact-match
product codes, entity names, and rare technical terms.

Elasticsearch, Vespa, and Qdrant Hybrid all ship hybrid retrieval as a first-
class feature.  RuVector lacks a unified hybrid interface.

This ADR proposes `ruvector-hybrid`: a proof-of-concept crate that wires BM25
and cosine-dense retrieval together through Reciprocal Rank Fusion (RRF), and
documents the recall ceiling and future directions.

---

## Decision

Implement `ruvector-hybrid` as a workspace crate exposing three variants of the
`HybridSearch` trait:

| Variant | Description |
|---------|-------------|
| `DenseFlat` | Exact cosine flat scan (oracle, O(N·D)) |
| `SparseBm25` | BM25 inverted index, Robertson-Zaragoza IDF |
| `HybridRrf` | RRF fusion of the above two |

### Fusion formula

```
RRF(d) = 1 / (60 + rank_dense(d))  +  1 / (60 + rank_sparse(d))
```

k = 60 is the empirically established constant from Cormack et al. (2009).

### BM25 parameters

k1 = 1.2, b = 0.75.  IDF is clamped to `max(0.0)` to prevent inversion on
high-frequency terms.

### Trait design

```rust
pub trait HybridSearch {
    fn insert(&mut self, id: u32, vector: &[f32], tokens: &[u32]);
    fn search(&self, query_vec: &[f32], query_tokens: &[u32], k: usize)
        -> Vec<SearchResult>;
    fn len(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}
```

Token IDs are caller-managed (e.g. from a vocabulary map); the index never
sees raw text.  This keeps the crate language-agnostic and avoids tokeniser
dependencies.

---

## Consequences

### Positive

* Unified trait surface: callers switch between Dense, Sparse, and Hybrid with
  zero API change.
* RRF is parameter-free at query time (k=60 is hardcoded); no per-deployment
  tuning required.
* Measured recall improvement: on the structured topic-model dataset, HybridRRF
  achieves **65.8% recall@10** vs. 34.1% for SparseBm25 alone (1.9× gain).
* DenseFlat retains 100% recall as the exact oracle; the gap between 65.8% and
  100% is analytically understood (see research README).

### Neutral

* HybridRRF query latency ≈ DenseFlat latency (dominated by the O(N·D) dense
  scan).  SparseBm25 alone is ~10× faster when only keyword retrieval is needed.
* Memory footprint = dense (4.92 MB) + sparse (0.15 MB) = 5.07 MB at 5 000
  docs, 128 dims — additive overhead.

### Negative

* **Recall ceiling at ~66%** on topic-model data due to BM25 score ties within
  the relevant set.  Documents sharing identical token sets receive the same BM25
  score; their sparse rank is non-deterministic, causing RRF to sometimes promote
  non-top-10 docs above true top-10 docs.
* DenseFlat is O(N·D) — not suitable for N > 100 000 without an ANN upgrade
  (tracked as future work).
* `unsafe` is forbidden (`#![forbid(unsafe_code)]`); SIMD optimisations require
  a safe-wrapper crate.

---

## Alternatives Considered

### Linear score combination

Weight dense and sparse scores directly: `α · cosine(d) + (1−α) · bm25(d)`.
Rejected at this stage because BM25 and cosine operate on different scales;
min-max or z-score normalisation adds latency and a tunable hyperparameter (α).
RRF avoids both.  Score-weighted fusion is the primary future-work item.

### Sparse-only expansion (BM25F / SPLADE)

Encode dense embeddings as high-dimensional sparse vectors (SPLADE) for indexing
in a standard inverted index.  Rejected: changes the embedding pipeline, adds a
heavy ML dependency, and is more of a replacement for dense search than a fusion
strategy.

### Cross-encoder re-ranking

Retrieve a candidate set with BM25 then re-rank with a cross-encoder.  Rejected
for this ADR: requires an inference model, adds latency proportional to candidate
set size, and is orthogonal to the fusion layer.

---

## Implementation

Crate: `crates/ruvector-hybrid/`  
Binary: `cargo run --release -p ruvector-hybrid --bin hybrid-benchmark`  
Tests: `cargo test -p ruvector-hybrid` (9 tests, all passing)  
Research: `docs/research/nightly/2026-05-30-hybrid-sparse-dense-search/README.md`

---

## Measured Numbers (reproducible)

All numbers from `cargo run --release -p ruvector-hybrid --bin hybrid-benchmark`
on Linux x86_64.

**Random dataset (10 000 docs, dim=128)**

| Variant | QPS | Recall@10 |
|---------|----:|----------:|
| DenseFlat | 625 | 100.0% |
| SparseBm25 | 6 567 | 0.2% |
| HybridRrf | 514 | 45.8% |

**Structured dataset (5 000 docs, dim=128, 32 topics)**

| Variant | QPS | Recall@10 |
|---------|----:|----------:|
| DenseFlat | 1 297 | 100.0% |
| SparseBm25 | 17 499 | 34.1% |
| HybridRrf | 1 185 | 65.8% |

---

## Future Work

1. Score-weighted fusion to break BM25 ties and push recall toward 100%.
2. Replace `DenseFlat` with ACORN-HNSW or RAIRS-IVF for sub-linear dense query.
3. SPLADE / COIL sparse encoder integration for richer lexical signal.
4. Quantised BM25 posting lists (8-bit tf/score) to reduce sparse memory.

---

## References

- Cormack, Clarke, Buettcher. *Reciprocal Rank Fusion outperforms Condorcet and
  individual rank learning methods*. SIGIR 2009.
- Robertson, Zaragoza. *The probabilistic relevance framework: BM25 and beyond*.
  Foundations and Trends in IR, 2009.
