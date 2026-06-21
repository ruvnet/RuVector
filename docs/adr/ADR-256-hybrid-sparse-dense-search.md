---
adr: 256
title: "ruvector-hybrid — Reciprocal Rank Fusion and Relative Score Fusion for hybrid sparse-dense search"
status: proposed
date: 2026-06-17
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-194, ADR-210, ADR-253, ADR-254]
tags: [hybrid-search, bm25, ann, rrf, rsf, fusion, retrieval, agent-memory, rag, mcp]
---

# ADR-256 — Hybrid Sparse-Dense Search: RRF and RSF alongside ScoreFusion

## Status

Proposed.  Proof of concept in `crates/ruvector-hybrid` (branch
`research/nightly/2026-06-17-hybrid-sparse-dense`).  Not yet merged into
`ruvector-core`.

---

## Context

### The gap in ruvector-core

`ruvector-core::advanced_features::hybrid_search` (added in ADR-210 context)
provides a `HybridSearch` struct combining BM25 (`k1=1.5`, `b=0.75`) and vector
similarity via a **weighted linear score fusion**:

```
combined = 0.7 × cosine_norm + 0.3 × bm25_norm
```

where normalisation is min-max across **all candidates** fetched from both backends.

This approach works when BM25 and cosine score distributions have compatible shapes.
It breaks when they do not — which is the common case in production:

- BM25 scores are peaky: a doc containing a rare exact-match term can score 10× the
  median.
- Cosine scores within a topic cluster are smooth: same-topic docs differ by <0.05
  in cosine similarity.
- Global min-max normalisation maps these to incompatible [0,1] ranges: a "good" BM25
  doc gets 0.95 normalised score; a "great" cosine doc gets 0.98; a "terrible" cosine
  doc gets 0.02 rather than 0.

Additionally, `BM25::score()` in the existing code **re-tokenises stored doc texts at
query time** — O(|d|) per candidate per query.  This is a latency regression for
large corpora.  The `ruvector-hybrid` implementation pre-computes per-doc TF at index
time (stored in postings), achieving O(|q| × avg\_posting\_len) at query time.

### Industry context (2026)

All major vector databases have added hybrid search in 2025–2026:

| System | Fusion strategy | Notes |
|--------|-----------------|-------|
| Qdrant v1.10+ | RRF (k=60) only | Server-side IDF since v1.15.2 |
| Weaviate v1.24+ | RSF (default) + RRF | α parameter controls blend |
| Milvus 2.5 | Custom RRF variant | BM25 stored as sparse vector |
| Vespa | WAND + ANN + neural | Three-phase ranking |
| LanceDB | BM25 (DuckDB FTS) + ANN | Client-side RRF |

RuVector's current score-fusion approach matches none of these; it is closest to
Weaviate v1.23 (pre-RSF), now obsolete.

### Benchmark evidence (this ADR)

Measured on synthetic corpus: 10,000 documents, 128-D vectors, 20 topics, 500 queries,
ground truth = 0.5×cosine\_norm + 0.5×BM25\_norm (brute force), k=10.

| Variant | Recall@10 | QPS | Memory |
|---------|-----------|-----|--------|
| Dense flat (exact) | 7.5% | 371 | 5,000 KB |
| BM25 (sparse) | 77.3% | 57,174 | 637 KB |
| ScoreFusion α=0.7 | 68.8% | 357 | 5,637 KB |
| **RRF k=60** | 50.5% | 360 | 5,637 KB |
| **RSF α=0.5** | 76.6% | 360 | 5,637 KB |

**Interpretation**: On a keyword-biased combined ground truth (topic-isolated
vocabulary), BM25 alone maximises recall.  RSF with α=0.5 recovers near-BM25
performance while maintaining semantic coverage.  RRF is more conservative
(score-agnostic rank fusion), appropriate when the relevance split between lexical
and semantic signals is unknown.  ScoreFusion with α=0.7 over-weights the dense
signal and performs worst among hybrids.

The existing ruvector-core weight of α=0.7 appears sub-optimal for keyword-heavy
workloads.  Adding an RRF path that requires no weight calibration is the safer
production default.

Hardware: Intel Xeon 2.80 GHz, Linux 6.18.5 x86\_64, rustc 1.94.1 --release.
Full benchmark at `docs/research/nightly/2026-06-17-hybrid-sparse-dense/README.md`.

---

## Decision

Add two new fusion strategies to RuVector's hybrid search infrastructure:

1. **RRF (Reciprocal Rank Fusion, k=60)**: rank-only, score-agnostic.  No weight
   calibration required.  Default fusion for agentic RAG workloads where the
   relevance split is unknown.

2. **RSF (Relative Score Fusion, α=0.5 default)**: per-list min-max normalisation
   + weighted blend.  Configurable α for workloads with known relevance balance.

The `ScoreFusion` path (existing `normalize_and_combine`) is retained as a
compatibility layer.

The `ruvector-hybrid` crate establishes the **trait surface** that this work should
expose in production:

```rust
pub trait SparseSearch {
    fn search(&self, tokens: &[&str], k: usize) -> Vec<SearchResult>;
}

pub trait DenseSearch {
    fn search(&self, vector: &[f32], k: usize) -> Vec<SearchResult>;
}

pub trait HybridSearch {
    fn search(&self, tokens: &[&str], vector: &[f32], k: usize) -> Vec<SearchResult>;
}
```

These traits are the stable API surface that should survive into production.

### What belongs behind a feature flag

- WAND BM25 pruning (not yet implemented; experimental when added).
- Learned sparse vector support (SPLADE / BGE-M3 sparse output).
- ColBERT late-interaction reranking as a third stage.

---

## Consequences

### Positive

- RRF eliminates the score-distribution incompatibility problem.  No α tuning needed.
- RSF with configurable α replaces the hard-coded 0.7/0.3 split.
- Pre-computed TF in postings reduces per-query latency vs. re-tokenising doc texts.
- Trait-based design allows swapping `FlatDenseIndex` for HNSW without changing fusion code.
- Crate compiles to WASM (no unsafe code, no external services).

### Negative

- Hybrid indices store both BM25 postings and dense vectors: 5,637 KB vs. 637 KB (BM25
  alone) or 5,000 KB (dense alone).  This is a deliberate trade-off for combined recall.
- Dense flat-scan latency (2,691 μs / query) does not scale.  Requires HNSW backend
  for production use at N > 100K.
- RRF recall (50.5%) is lower than BM25 alone (77.3%) on keyword-dominated tasks.
  Users who know their workload is keyword-heavy should lower α or use BM25 only.

---

## Alternatives Considered

### A: Add RRF to existing HybridSearch in ruvector-core directly

Rejected at this stage: the existing `HybridSearch` has the re-tokenisation-at-query-time
bug and the global normalisation design flaw.  Adding RRF to a flawed base would produce
a hybrid of old and new idioms.  Better to prove the design in a clean crate first, then
refactor ruvector-core to adopt the trait surface.

### B: Use only RRF (drop ScoreFusion and RSF)

Rejected: RSF with tunable α outperforms RRF on keyword-dominated workloads (76.6% vs.
50.5% recall).  Both strategies serve different use cases.  The trait-based design lets
callers choose.

### C: Integrate SPLADE from the start

Deferred: no production-ready Rust SPLADE implementation exists as of June 2026.  BGE-M3
sparse inference requires ONNX runtime or custom kernel.  BM25 is the practical baseline
for today.  SPLADE can be added as a `LearnedSparseIndex` variant later without changing
the `SparseSearch` trait.

---

## Implementation Plan

### Phase 1 (Now — this PR)

- [x] `crates/ruvector-hybrid`: standalone crate with `Bm25Index`, `FlatDenseIndex`,
  `ScoreFusionIndex`, `RrfHybridIndex`, `RsfHybridIndex`.
- [x] 19 unit tests passing.
- [x] Benchmark binary with real numbers.
- [x] ADR (this document).

### Phase 2 (Next — ruvector-core integration)

- [ ] Add `FusionStrategy` enum to `ruvector-core::advanced_features::hybrid_search`.
- [ ] Add `HybridSearch::search_rrf()` and `HybridSearch::search_rsf()` methods.
- [ ] Fix BM25 re-tokenisation bug (pre-compute TF at index time).
- [ ] Add incremental IDF update for streaming inserts.

### Phase 3 (Later — production hardening)

- [ ] Replace `FlatDenseIndex` with HNSW from `ruvector-core`.
- [ ] Add WAND pruning to `Bm25Index`.
- [ ] Add `LearnedSparseIndex` (SPLADE weights).
- [ ] Expose hybrid search as MCP tool in `ruvector-server`.

---

## Failure Modes

1. **BM25 vocabulary mismatch**: query OOV tokens return zero sparse results.  `RrfHybridIndex`
   degrades gracefully to pure dense.  `ScoreFusionIndex` collapses α to effectively 1.0.
   Mitigate: warn when sparse result set is empty.

2. **Score distribution mismatch in ScoreFusion**: motivating case for RRF.  Document this
   in `HybridConfig` documentation so users know when to switch.

3. **Dense latency at scale**: `FlatDenseIndex` is O(N·D) per query.  Must be replaced with
   HNSW for N > 100K before any production deployment.

4. **IDF staleness**: current batch-build IDF is incorrect after incremental inserts.  Track
   doc count and per-term DF incrementally; rebuild IDF every K inserts.

---

## Security Considerations

- **Fusion weight attestation**: in high-stakes agentic RAG, the α parameter should be
  proof-carried via `ruvector-verified` to prevent adversarial weight manipulation.
- **Term stuffing**: adversaries can inject documents with many rare query terms to dominate
  BM25 rankings.  Apply max-IDF capping and length normalisation.
- **Query logging**: BM25 queries log exact tokens; dense queries log embedding vectors.
  Both may leak user intent.  Apply differential privacy or query truncation in MCP tools.

---

## Migration Path

The `HybridSearch` struct in `ruvector-core` is additive: existing code using
`normalize_and_combine` continues to work.  New code calls `search_rrf()` or
`search_rsf()`.  No breaking change.

---

## Open Questions

1. What α should be the default in `RsfHybridIndex`?  The benchmark suggests α=0.5
   (equal weighting) works well on keyword-dominated tasks.  Does it hold for
   semantic-dominated tasks?  Requires evaluation on a semantic-focused ground truth.

2. Should RRF k=60 be configurable?  The original Cormack paper found k=60 optimal
   across many tasks.  Production systems (Qdrant) use k=60 fixed.  For now, expose
   as a constant; make configurable in Phase 3 if ablations warrant it.

3. Should `Bm25Index::build` accept a `Tokenizer` trait to allow plug-in tokenisation
   (whitespace, BPE, Unicode)?  Deferred to Phase 2.
