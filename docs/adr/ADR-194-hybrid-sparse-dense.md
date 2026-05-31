---
adr: 194
title: "Hybrid Sparse-Dense Search — BM25 Inverted Index + Dense ANN with RRF and Linear Fusion"
status: accepted
date: 2026-05-20
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-135]
tags: [hybrid-search, sparse-dense, bm25, rrf, linear-fusion, vector-search, ann, nightly-research]
---

# ADR-194 — Hybrid Sparse-Dense Search: BM25 + Dense ANN with RRF and Linear Fusion

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-20-hybrid-sparse-dense` as
`crates/ruvector-hybrid`. Build is green with `cargo build --release -p ruvector-hybrid`.
All 16 unit tests pass. All 5 acceptance tests pass with real benchmark numbers.

---

## Context

RuVector's retrieval surface in May 2026 consists entirely of dense approximate nearest-neighbor
search (HNSW in `ruvector-core`, flat scan in benchmarks, DiskANN/Vamana in `ruvector-diskann`,
IVF in `ruvector-rairs`). Every major competitor — Qdrant, Milvus 2.6, Weaviate, Elasticsearch,
Vespa, LanceDB, pgvecto.rs — now ships hybrid search as a default or near-default feature,
combining a dense ANN leg with a sparse BM25 or SPLADE-style inverted index leg.

The practical consequence is that RuVector agent memory retrieval fails silently for queries that
mix semantic intent with exact symbolic references (entity names, identifiers, code tokens,
dates). A query like "find ADR-194 and related coherence work" requires both a dense leg
(semantic proximity to "coherence") and a sparse leg (exact match on "ADR-194").

This ADR introduces the foundational hybrid search infrastructure that corrects this gap:

1. `SparseVec` — a sorted `(term_id, weight)` pair representation compatible with BM25 and
   SPLADE impact-score formats.
2. `SparseIndex` — BM25-style inverted index with inner-product (IMPACT) scoring.
3. `DenseFlatIndex` — exact flat dense search as a correct baseline for benchmarking.
4. `HybridIndex` — composite index implementing the `HybridSearch` trait.
5. Three fusion strategies: RRF, linear interpolation, and max-of-signals.
6. A real benchmark binary with measured latency, QPS, memory, and recall.

---

## Decision

Introduce `crates/ruvector-hybrid` as a new standalone research-tier crate in the workspace.

### Why this belongs in RuVector

RuVector is a cognitive substrate for agents, not just a vector database. Agent memory requires
both symbolic (sparse, exact) and semantic (dense, approximate) retrieval. Without the sparse
leg, agents cannot reliably retrieve memories anchored to exact identifiers. This is a functional
gap, not a performance optimization.

### Why this is not just an experiment

Hybrid search is the 2026 industry baseline. Shipping without it puts RuVector below parity with
every competitor. The `HybridSearch` trait provides the stable API surface that production
integration (HNSW swap-in, ruFlo automation, MCP tool binding) will depend on.

### API shape that should survive into production

```rust
pub trait HybridSearch {
    fn insert(&mut self, doc: HybridDoc);
    fn search_dense(&self, q: &HybridQuery, k: usize) -> Vec<Scored>;
    fn search_sparse(&self, q: &HybridQuery, k: usize) -> Vec<Scored>;
    fn search_rrf(&self, q: &HybridQuery, k: usize, candidate_k: usize) -> Vec<Scored>;
    fn search_linear(&self, q: &HybridQuery, k: usize, candidate_k: usize, alpha: f32) -> Vec<Scored>;
}
```

`SparseVec`, `DenseVec`, `HybridDoc`, `HybridQuery`, `Scored` — these types are stable. They
should be re-exported from `ruvector-core` or a new `ruvector-types` crate when the hybrid
search tier graduates to production.

### What should remain behind a feature flag

- HNSW dense leg (requires `ruvector-core` dependency, adds ~30s build time). Keep behind
  `features = ["hnsw"]` in `ruvector-hybrid`.
- WASM exports (need `wasm-bindgen`). Keep behind `features = ["wasm"]`.
- MCP tool bindings. Keep behind `features = ["mcp"]`.

### What would make us reject this direction

- If agent memory queries turn out to be 100% semantic (no keyword-exact intent), the sparse leg
  adds cost without recall benefit. This can be measured on real agent query logs.
- If `candidate_k=50` proves insufficient to achieve production recall targets (>80% vs oracle)
  on realistic corpora, the architecture needs BMP (Block-Max Pruning) in the sparse leg and
  HNSW in the dense leg before any production recommendation.

---

## Consequences

### Positive

- Closes the hybrid search gap vs all major competitors.
- `HybridSearch` trait enables zero-code-change dense backend swap (flat → HNSW → DiskANN).
- WASM-compatible by construction: no unsafe, no OS syscalls, no heavyweight dependencies.
- BM25 weights are compatible with SPLADE output — upgrading to learned sparse requires only a
  weight-generation function swap, not an index format change.
- Sparse inverted index memory is lower than dense flat at equal doc count when term count is
  small: 774 KB vs 2,500 KB at N=5K, D=128, 20 terms/doc.

### Negative

- `candidate_k` at 50 gives ~30% recall vs exact oracle at N=5K corpus with balanced queries.
  Production will require 100–500 candidates per channel (BMP reduces the latency cost of this).
- The dense leg is O(N·D) flat scan. Production requires HNSW swap-in before N > 100K.
- No streaming inserts: `SparseIndex` is append-only in this PoC.
- No stop-word filtering: callers must validate input before `bm25_weights()`.

---

## Alternatives Considered

### 1. Extend `ruvector-filter` to include term matching

**Rejected.** `ruvector-filter` does boolean predicate filtering on metadata, not term-weighted
ranking. Adding a BM25 scoring path there would conflate two distinct concerns (filtering vs
retrieval) and break the ADR-143 separation between storage, filter, and search layers.

### 2. Use `tantivy` as the sparse leg

**Rejected.** `tantivy` is a full Lucene-equivalent search engine that adds significant
complexity and build time. For RuVector's use case (in-memory inverted index, no full-text
analysis pipeline required), a 150-line `SparseIndex` struct is simpler, faster to build, and
avoids a heavyweight dependency.

### 3. Implement only RRF, no linear fusion

**Rejected.** RRF is parameter-free but cannot be tuned toward the stronger signal. Linear
fusion with α calibration consistently outperforms RRF when labeled data is available.
Implementing both keeps options open without significant code cost (the linear path is 20 lines
of code).

### 4. Require dense embedding model at index time

**Rejected.** This ADR treats dense vectors as caller-provided (pre-computed). Bundling an
embedding model (ONNX, Candle) couples retrieval infrastructure to model infrastructure. The
`HybridDoc` type accepts any `DenseVec` — the source of embeddings is out of scope.

---

## Implementation Plan

### Phase 1 — PoC (this ADR, complete)

- [x] `crates/ruvector-hybrid` with all core types
- [x] `SparseIndex` (BM25 inverted posting lists)
- [x] `DenseFlatIndex` (exact inner product)
- [x] `HybridIndex` (composite)
- [x] `fusion::{rrf, linear, max_signal}`
- [x] `HybridSearch` trait with RRF and linear fusion
- [x] 16 unit tests, all passing
- [x] Real benchmark binary with latency / QPS / recall / memory
- [x] All 5 acceptance tests passing

### Phase 2 — Production Hardening

- [ ] Block-Max Pruning (BMP) in `SparseIndex` — reduces sparse leg latency 10x–25x
- [ ] HNSW swap-in via `HybridSearch` trait implementation on `ruvector-core::HnswIndex`
- [ ] Query term thresholding (zero terms < thresh_ratio × max_weight)
- [ ] α calibration from labeled query pairs
- [ ] Stop-word filter integration at the `bm25_weights()` call site
- [ ] Streaming delta log via `ruvector-delta-*`

### Phase 3 — Ecosystem Integration

- [ ] MCP `memory_search` tool backed by `HybridIndex`
- [ ] ruFlo nightly α recalibration workflow
- [ ] ruvector-verified proof-gated insert wrapper
- [ ] WASM feature gate + `wasm-bindgen` exports
- [ ] RVF manifest serialisation of `HybridIndex` state

---

## Benchmark Evidence

From `cargo run --release -p ruvector-hybrid --bin benchmark` (seed=2026, deterministic):

```
  Variant      | Mean µs | p50 µs | p95 µs |  QPS  | Recall@10 | Memory
  DenseOnly    |   791.4 |  793.2 |  851.8 | 1,264 |    12.9%  | 2,500KB
  SparseOnly   |    30.7 |   30.0 |   45.3 |32,548 |    27.2%  |   774KB
  HybridRRF    |   824.5 |  830.3 |  879.5 | 1,213 |    30.1%  | 3,274KB
  HybridLinear |   826.0 |  830.8 |  880.4 | 1,211 |    29.8%  | 3,274KB

  5/5 acceptance tests: PASS
```

Context: oracle = exact hybrid fusion over ALL 5,000 docs; candidate_k=50 for hybrid variants.
Platform: x86_64, Linux 6.18.5, rustc 1.94.1, `cargo run --release`.

---

## Failure Modes

| Failure | Detection | Response |
|---------|-----------|----------|
| Dense candidate_k exhaustion | Recall vs oracle < target threshold | Increase candidate_k or enable BMP |
| Vocabulary mismatch | Sparse recall plateaus | Switch from BM25 to SPLADE weights |
| α miscalibration | Hybrid degrades vs single channel | Re-run calibration on recent queries |
| Memory pressure at scale | OOM at N > 1M | Enable int8 dense quantization; prune sparse below weight threshold |
| Score injection | Inflated sparse scores in adversarial docs | Validate term weights at ingestion boundary |

---

## Security Considerations

- Term IDs in posting lists come from caller-provided sparse vectors. Validate that term IDs
  are within the expected vocabulary range at ingestion time.
- Query term weights must be non-negative (BM25 and SPLADE both produce non-negative weights).
  Reject negative weights at the `HybridQuery` boundary.
- The `bm25_weights()` function clips negative IDF values with `.max(0.0)` — this is correct
  and tested.
- No network calls, no file I/O, no external services. The crate is safe-only Rust.

---

## Migration Path

1. **Immediate**: Use `crates/ruvector-hybrid` as a standalone hybrid search engine by
   constructing a `HybridIndex` and calling `search_rrf` or `search_linear`.
2. **Short-term** (Phase 2): Add `ruvector-core` HNSW as the dense leg behind a feature flag.
   The `HybridSearch` trait interface is already defined — no API change needed.
3. **Long-term** (Phase 3): Promote `HybridIndex` to `ruvector-core` or a new
   `ruvector-retrieval` meta-crate that bundles both dense and sparse search.

---

## Open Questions

1. **BMP block size**: SIGIR 2024 recommends b=64 to b=256. What is optimal for ruvector's
   typical corpus sizes (10K–10M documents)?
2. **SPLADE model hosting**: Where should learned sparse weight generation live? A separate
   `ruvector-splade` crate wrapping Candle inference? Or caller-provided pre-computed weights?
3. **α auto-tuning**: Should α be learned per-namespace (e.g., per agent session) or globally?
   What is the minimum number of labeled pairs needed for reliable calibration?
4. **Sparse index sharding**: How does `SparseIndex` shard across multiple `ruvector-raft`
   nodes? Per-shard inverted index with merge at query time, or term-partitioned sharding?
5. **WASM memory limit**: At what corpus size does the WASM 32-bit address space (4 GB limit,
   practical limit ~2 GB) require switching to streaming retrieval?
