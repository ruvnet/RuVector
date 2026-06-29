# ADR-252: Multi-Vector MaxSim Late Interaction Search

**Status:** Accepted — PoC merged, production graduation pending  
**Date:** 2026-06-15  
**Crate:** `crates/ruvector-maxsim`  
**Research:** `docs/research/nightly/2026-06-15-multi-vector-maxsim/README.md`

---

## Context

RuVector's existing index variants (HNSW in `ruvector-core`, IVF in
`ruvector-rairs`, filtered HNSW in `ruvector-acorn`) all assume a
**single embedding vector per document**. This is a fundamental limitation
for documents that cover multiple topics, since averaging token embeddings
into one vector destroys facet-level information.

**ColBERT** (Khattab & Zaharia, 2020; arXiv 2004.12832) introduced
*late interaction*: store K token vectors per document and score queries as

```
score(Q, D) = Σ_{q ∈ Q}  max_{d ∈ D}  cosine(q, d)
```

The sum-of-max aggregation (MaxSim) lets a document be discovered by ANY
of its topic facets independently. ColBERT and its descendants (ColBERTv2
2022, PLAID 2022, ColPali 2024) have become the SOTA for passage retrieval
tasks where single-vector approaches lose information.

No Rust-native multi-vector index existed in the ruvector workspace.

---

## Decision

Add `crates/ruvector-maxsim` implementing the `MultiVecIndex` trait with
three variants:

| Variant | Algorithm | Recall | Latency | Use case |
|---------|-----------|--------|---------|----------|
| `FlatMaxSim` | Exhaustive scan | 100% (oracle) | O(N·Td·Tq·D) | Ground truth, small corpora |
| `BucketMaxSim` | Centroid pre-filter + exact MaxSim | 35–80% | O(M·Td·Tq·D) | Speed-first retrieval |
| `HnswMaxSim` | NSW token graph + grouped MaxSim | 40–70% | Sub-linear | Balanced retrieval |

All three share the `MultiVecIndex` trait:

```rust
pub trait MultiVecIndex {
    fn add(&mut self, doc: MultiVecDoc) -> Result<(), MaxSimError>;
    fn search(&self, query: &MultiVecQuery, k: usize) -> Result<Vec<SearchResult>, MaxSimError>;
    fn len(&self) -> usize;
    fn dims(&self) -> usize;
}
```

---

## Consequences

### Positive

- Enables faceted agent memory: a memory about "Rust + safety + async" can
  be found by queries about any one of those facets independently.
- Provides a ground truth oracle (`FlatMaxSim`) for evaluating other indexes.
- `MultiVecIndex` trait is composable: future variants (quantized token
  vectors, HNSW-per-topic, product quantization over tokens) can plug in
  without API changes.
- No external service dependencies; fully self-contained pure Rust.

### Negative

- Multi-vector storage is inherently more memory-hungry: 6 tokens × 64 dims
  × 4 bytes × 5K docs = 7.3 MB vs 1.2 MB for single-vector.
- `HnswMaxSim` uses a flat NSW graph (single layer), not a full HNSW with
  layer hierarchy — higher construction cost and lower recall than a tuned
  HNSW at the same EF.
- `BucketMaxSim` recall collapses with centroid averaging for documents
  spanning very different topic directions.

---

## Alternatives Considered

1. **Single-vector with mean pooling**: Simple but loses facet information.
   This is what all current ruvector indexes do.

2. **Per-facet separate indexes**: Maintain one HNSW per topic cluster and
   union results. Avoids MaxSim arithmetic but requires topic labeling at
   insert time, which is not available in the agent memory use case.

3. **PLAID-style inverted index over token IDs**: High throughput (Santhanam
   et al. 2022) but requires a fixed vocabulary of token IDs, incompatible
   with continuous embedding spaces.

4. **PageANN page-aligned DiskANN extension**: Scored higher in the research
   agent's analysis (4.50 vs 3.80) but requires SSD page-fault measurement
   which is unreliable in a cloud VM, making benchmark validation impossible
   tonight.

---

## Implementation Plan

### Now (this crate)

- [x] `MultiVecIndex` trait
- [x] `FlatMaxSim` (exact oracle)
- [x] `BucketMaxSim` (centroid pre-filter)
- [x] `HnswMaxSim` (flat NSW token graph)
- [x] 19 unit + integration tests
- [x] Benchmark binary with acceptance tests
- [x] Workspace member

### Next

- [ ] `HnswMaxSim` upgrade to proper layered HNSW (using `hnsw_rs`)
- [ ] Product quantization over token vectors to reduce memory 4–8×
- [ ] SIMD-accelerated MaxSim kernel (AVX2/NEON via `simsimd`)
- [ ] `rayon`-parallel scoring for `FlatMaxSim` (parallel map over docs)
- [ ] Integration with `ruvector-core` AgenticDB as the memory backend
- [ ] Streaming inserts (currently index is insert-only)

### Later (2028–2036)

- Integration with RVF format: multi-vector documents as a first-class
  field type in cognitive packages
- ruFlo-driven self-optimizing oversampling: auto-tune `BucketMaxSim`
  oversampling based on query distribution history
- WASM-safe token vector scoring for Cognitum edge appliance

---

## Benchmark Evidence

Hardware: x86_64 Linux 6.18.5, Intel Celeron N4020, `rustc 1.87.0 --release`  
Dataset: synthetic Gaussian clusters, 32 topics, noise σ=0.3, N=5000 docs,
6 tokens/doc, 3 tokens/query, D=64, k=10

| Variant | QPS | Recall@10 | Memory |
|---------|-----|-----------|--------|
| FlatMaxSim (oracle) | 179 | 1.000 | 7.3 MB |
| BucketFast (os=50) | 1855 | 0.348 | 8.5 MB |
| BucketQuality (os=500) | 873 | 0.797 | 8.5 MB |
| HnswMaxSim | 774 | 0.437 | 11.0 MB |

**Key result**: BucketFast delivers 10.4× speedup over FlatMaxSim at 34.8%
recall. BucketQuality delivers 4.9× speedup at 79.7% recall. Multi-token
document advantage confirmed: doc covering two topics scores 1.0 vs 0.0 for
a single-topic query; single-topic doc scores −0.02.

---

## Failure Modes

1. **Centroid averaging collapse**: When a document spans orthogonal topics,
   its centroid lands between them, making centroid pre-filtering unreliable.
   Mitigation: increase oversampling or use `HnswMaxSim`.

2. **NSW connectivity breaks at scale**: Flat NSW (no hierarchy) degrades
   to linear scan at >100K tokens. Mitigation: upgrade to full HNSW.

3. **Token count imbalance**: Documents with 1 token vs 50 tokens have very
   different MaxSim score scales. Normalise by query token count for fairness.

4. **Memory explosion**: 100K docs × 32 tokens × 384 dims × 4B = 4.9 GB.
   Mitigation: product quantization, 1-bit token codes, or lazy loading.

---

## Security Considerations

- No network I/O, no unsafe code (`#![forbid(unsafe_code)]`).
- Adversarial documents with many redundant tokens can inflate their MaxSim
  score without semantic merit. Mitigation: normalise by document token count
  before summing.
- Cosine similarity handles zero-magnitude vectors gracefully (returns 0.0).

---

## Migration Path

Existing single-vector `ruvector-core` users can migrate by:
1. Splitting documents into chunks and generating one embedding per chunk.
2. Wrapping in `MultiVecDoc` and indexing via `FlatMaxSim` or `HnswMaxSim`.
3. No change to query embedding required — single-query-token queries work.

---

## Update — 2026-06-29: `GraphMaxSim` variant

A fourth index variant, **`GraphMaxSim`** (`crates/ruvector-maxsim/src/graph.rs`),
was added. Unlike `HnswMaxSim` — which builds a navigable graph over *individual
token vectors* — `GraphMaxSim` builds a greedy kNN graph over *per-document
centroids*, runs a multi-seed beam search to select a candidate document set,
and reranks it with the exact MaxSim kernel. One node per document (vs one per
token) keeps the graph small for corpora with many tokens per document, trading
a little recall for a smaller, faster-to-traverse structure.

**Correctness finding — beam-search seeding.** Multi-seed beam search must seed
its entry points from the **first `K` consecutive** centroids, *not* a
strided/step-based sample (`step_by(N/K)`). When the step is a multiple of the
underlying cluster count, every seed lands in the same cluster and recall
collapses to a few percent. Consecutive seeding of the first `K ≥ N_CLUSTERS`
documents guarantees cluster coverage for the common interleaved ordering
(`doc i → cluster i % N_CLUSTERS`). This is a general hazard for any
cluster-graph ANN seeding strategy, not just MaxSim. (Salvaged from nightly
research PR #622, whose duplicate `ruvector-maxsim` crate was not merged; the
public writeup is preserved as a gist linked from the 2026-06-29 nightly notes.)

---

## Open Questions

1. Should MaxSim be normalised by `|Q|` (number of query tokens) to make
   scores comparable across queries of different lengths?

2. Does `HnswMaxSim` benefit from separate per-topic NSW layers (one per
   detected topic cluster) at the cost of insert-time clustering?

3. Can `BucketMaxSim` be adapted to use the existing `ruvector-mincut`
   coherence scoring to detect centroid averaging collapse and fall back to
   exhaustive scan automatically?

4. What is the right representation for multi-vector documents in the
   RVF (Ruvector Format) specification?
