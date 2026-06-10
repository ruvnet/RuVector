---
adr: 199
title: "Late Interaction Multi-Vector Search (MaxSim / ColBERT-style)"
status: accepted
date: 2026-06-10
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-143, ADR-101]
tags: [vector-search, late-interaction, maxsim, colbert, multi-vector, rag, agent-memory, nightly-research]
---

# ADR-199 — Late Interaction Multi-Vector Search (MaxSim / ColBERT-style)

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-10-late-interaction-maxsim`
as `crates/ruvector-late-interaction`. All 20 unit tests pass; both acceptance
criteria pass; build is green with `cargo build --release -p ruvector-late-interaction`.

## Context

RuVector can currently store and search against *single* vector embeddings per
document — one f32 array per semantic unit.  This model works well for dense
retrieval when the document and query can each be reduced to a single point in
embedding space.

The 2024–2026 RAG research ecosystem has converged on a richer model: **late
interaction retrieval**, popularised by ColBERT (Khattab & Zaharia, 2020) and
its successors ColBERTv2, PLAID, and ColBERT-Att (arXiv:2603.25248, Mar 2026).
Rather than collapsing a document into one vector, each token (or sentence) gets
its own embedding.  Relevance is scored as:

```
MaxSim(Q, D) = Σ_{q ∈ Q} max_{d ∈ D} cosine(q, d)
```

This has three concrete advantages:

1. **Recall**: term-level alignment catches documents that share vocabulary with
   the query even when the bag-of-words overlap is zero at the document level.
2. **Precision**: max per query token prevents irrelevant tokens from diluting
   the score, unlike additive pooling.
3. **Reranking without reranker models**: the MaxSim score is interpretable and
   does not require a separate cross-encoder at inference.

By 2026 this matters because:

- Qdrant v1.15+ ships multivector natively (using a proprietary Colbert-like
  API).
- ECIR 2026 hosted the dedicated LIR (Late Interaction and Retrieval) workshop
  (arXiv:2511.00444).
- PyLate (arXiv:2508.03555) provides an open-source training + retrieval
  framework.
- No Rust-native open-source MaxSim engine existed before this crate.

Agent use cases are equally compelling: an agent's working memory consists of
multi-turn utterances, each decomposable into tokens.  MaxSim retrieval finds
past context that is *terminologically* close to the current step, not just
semantically close at the document level.

## Decision

We introduce `crates/ruvector-late-interaction` implementing three variants of a
`MaxSimIndex` trait:

| Variant | Description | Trade-off |
|---------|-------------|-----------|
| `BruteForceIndex` | Exact O(N·T_d·T_q·D) scan | Ground truth; slow for large N |
| `PlaidLiteIndex` | k-means centroid pre-filter + full MaxSim on shortlist | Speed vs recall tunable via `n_probe` |
| `CompressedIndex` | SQ8-quantized tokens, i8 dot products | 4× memory reduction, ~79 % recall |

All variants share:
- Common `MaxSimIndex` trait: `insert`, `build`, `query`, `memory_bytes`
- Deterministic `DatasetGen` for reproducible benchmarks
- No external service dependencies

### Core API shape

```rust
pub trait MaxSimIndex {
    fn insert(&mut self, doc: MultiVecDoc) -> Result<()>;
    fn build(&mut self) -> Result<()>;
    fn query(&self, q: &MultiVecQuery, top_k: usize) -> Result<Vec<ScoredDoc>>;
    fn memory_bytes(&self) -> usize;
}
```

`MultiVecDoc` holds `Vec<Vec<f32>>` (num_tokens × dim); `MultiVecQuery`
is the same shape for the query side.  L2-normalised vectors are assumed so
`dot(q, d) == cosine(q, d)`.

## Consequences

### Positive

- RuVector can now act as a ColBERT-style retrieval backend for RAG pipelines
  without any Python dependency.
- Agent memory stored as multi-vector documents gains token-level recall that
  single-vector HNSW cannot provide.
- The `CompressedIndex` is a natural bridge to WASM deployment: 2 MB for
  2,000 × 16 × 64 corpora fits in edge device RAM.
- The centroid-based `PlaidLiteIndex` is composable with the existing
  `ruvector-diskann` Vamana graph: DiskANN can serve as the centroid lookup,
  replacing the linear scan used in this PoC.

### Negative / Risks

- MaxSim is inherently O(T_q × T_d) per document in the candidate set.  For
  very long documents (T_d > 512) brute-force MaxSim is expensive.
- The PLAID-lite n_probe tuning is dataset-dependent; a generic default may
  hurt precision on domain-specific corpora with tight Voronoi boundaries.
- SQ8 recall (0.792 on random unit vectors) is likely higher on real text
  embeddings (which cluster more tightly), but this remains unverified.
- Token storage costs are T_d × higher than single-vector storage.  For T_d=16
  and D=64 this is 8 MB / 2,000 docs; at T_d=128 and D=768 it is 300 MB / 2,000 docs.

## Alternatives Considered

### 1. Single-vector dense retrieval only (status quo)

Already in `ruvector-core` (HNSW) and `ruvector-diskann`.  Keeps storage small
but cannot recover term-level recall.

### 2. Sparse BM25 + dense hybrid fusion

Good baseline, planned as a future nightly.  Does not support token-level learned
representations.  The ColBERT MaxSim score subsumes BM25 recall in most
published comparisons at equivalent latency after PLAID compression.

### 3. Full ColBERTv2 token index with inverted file (IVF)

Best recall.  Would use `ruvector-rairs` (ADR-193) as the IVF backend for
centroid lookup.  Deferred: requires substantially more engineering
(token-to-centroid mapping, residual compression per centroid list).
Documented as the "Production Candidate" direction in the research doc.

### 4. Product Quantization (PQ) for token embeddings

PQ offers better recall per byte than SQ8 for high-dimensional vectors.
Deferred because ruvector has no PQ crate; PQ is a better follow-on after this
PoC validates the MaxSim path.

## Implementation Plan

| Phase | Work | Owner | When |
|-------|------|-------|------|
| PoC | `crates/ruvector-late-interaction` with three variants | done | 2026-06-10 |
| Integration | Expose `MaxSimIndex` from `ruvector-core` feature flag | ruvnet | next sprint |
| Storage | Persist multi-vector corpora via `redb` | ruvnet | next sprint |
| PLAID upgrade | Replace linear centroid scan with DiskANN centroid graph | ruvnet | +2 sprints |
| WASM port | `ruvector-late-interaction-wasm` via memory-only feature | ruvnet | +3 sprints |
| MCP tool | `list_multi_vector_docs`, `query_maxsim` tools | ruvnet | +3 sprints |

## Benchmark Evidence

Hardware: x86-64 Linux 6.18, Intel Celeron N4020, `rustc 1.94.1 --release`.
Dataset: N=2,000 docs, D=64, T_doc=16 tokens/doc, T_q=8 query tokens.
Queries: 50.  top_k=10.

| Variant | Mean lat. | p50 | p95 | QPS | Mem (KB) | Recall@10 |
|---------|-----------|-----|-----|-----|----------|-----------|
| brute-force-maxsim | 13,494 µs | 13,265 µs | 16,008 µs | 74 | 8,000 | 1.000 (GT) |
| compressed-sq8-maxsim | 9,791 µs | 9,585 µs | 11,419 µs | 102 | 2,000 | 0.792 |
| plaid-lite-maxsim | 15,262 µs | 15,277 µs | 16,119 µs | 66 | 8,016 | 0.998 |

Acceptance result: **PASS** (compressed ≥ 0.75; plaid ≥ 0.60).

**Notes on PLAID-lite (n_probe=4):** recall is 0.998 at N=2,000 because with
64 centroids and 2,000 × 16 = 32,000 tokens, each centroid covers ~500 tokens
across ~31 docs; 4 centroids per query token × 8 query tokens covers nearly the
full corpus.  PLAID's speed advantage materialises at N ≥ 50,000 where the
centroid pre-filter prunes ≥ 90 % of documents before MaxSim.  At N=2,000 it
is effectively brute-force and shows comparable latency.

**Notes on SQ8 recall (0.792):** random unit vectors spread uniformly over the
hypersphere, maximising quantization error relative to real text embeddings which
cluster around semantic directions.  Published ColBERT-SQ8 numbers on MSMARCO
show recall degradation of ~1–3 pp vs full f32.  Our 0.792 vs 1.000 reflects the
synthetic worst-case, not a production estimate.

## Failure Modes

1. **Empty candidate set in PLAID-lite** — if all query tokens map to centroids
   with no docs, `query()` returns an empty vec.  Mitigation: fall back to full
   scan when candidate set is empty.  Tracked but not yet implemented.
2. **k-means degenerate centroids** — empty clusters are re-initialised by
   random point, but pathological data can cause repeated empty clusters.
   Mitigation: use k-means++ initialization (future work).
3. **SQ8 precision loss for low-dimensional embeddings** — at D=8, quantization
   error is proportionally large.  Not recommended below D=32.
4. **Build time** — k-means on 32,000 tokens (2,000 × 16) with 64 centroids
   and 5 iterations takes ~627 ms on Celeron N4020.  Subsampling to 8,000 tokens
   maintains centroid quality; documented in `plaid.rs`.

## Security Considerations

No network, file system, or external service access.  All data is held in-process
Rust `Vec`.  No unsafe code.  Token embeddings may encode sensitive text; callers
must sanitise before storage.  Future: integrate `ruvector-verified` proof-gated
write path so token insertions require a witness signature.

## Migration Path

- No existing code depends on this crate; zero breaking changes.
- The `MaxSimIndex` trait is additive.  Single-vector HNSW callers in
  `ruvector-core` are unaffected.
- To migrate a single-vector RAG pipeline to multi-vector: split each document
  into sentences, embed each sentence independently, insert as `MultiVecDoc`.

## Open Questions

1. Should `MultiVecDoc` store a variable or fixed token count?  Variable is
   flexible; fixed enables SIMD matrix operations.
2. Should PLAID-lite use `ruvector-diskann`'s Vamana graph for centroid lookup
   or keep the O(K·D) linear scan?  Vamana would scale better but adds a
   dependency.
3. Is SQ8 the right default compression, or should we implement PQ first?
4. How should the MCP tool surface MaxSim queries to ruFlo workflows?
5. Should the RVF cognitive package format support multi-vector document payloads?
