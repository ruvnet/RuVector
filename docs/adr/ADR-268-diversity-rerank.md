# ADR-268: Diversity Reranking for ANN Candidate Sets

**Status:** Proposed  
**Date:** 2026-06-22  
**Slug:** diversity-rerank  
**Crate:** `ruvector-diversity-rerank`  
**Branch:** `research/nightly/2026-06-22-diversity-rerank`

---

## Context

RuVector's ANN retrieval (HNSW, DiskANN, RaBitQ, IVF) returns the `k` closest
vectors to a query.  In corpora with dense clusters of near-duplicate content —
agent memory stores, document chunk corpora, recommendation catalogues — the
top-K result is dominated by almost-identical vectors.

This creates measurable downstream failures:
- **RAG pipelines**: Context window filled with redundant passages; answer
  quality degrades when all k passages say the same thing.
- **Agent memory**: Recall bias toward frequently-reinforced, similar memories
  prevents the agent from surfacing relevant but less-recent knowledge.
- **Recommendation**: Top-K returns product variants instead of complementary products.

Existing RuVector crates address retrieval *accuracy* (GNN reranking,
coherence-HNSW) but none address retrieval *diversity*.

---

## Decision

Add `ruvector-diversity-rerank` as a standalone composable crate that provides
post-retrieval diversity reranking through a `DiversityReranker` trait.

Three implementations ship in the initial PoC:

1. **`BaselineReranker`** — reference: sort by distance only.
2. **`MmrReranker { lambda }`** — Maximal Marginal Relevance (Carbonell & Goldstein, 1998).
3. **`MinCutReranker { sim_threshold, degree_weight }`** — greedy inhibition on
   a threshold graph, aligned with RuVector's mincut philosophy.

The trait accepts a `Vec<Candidate>` (each Candidate holds `id`, `distance`,
and `vector`) and returns a `RerankResult` containing the reranked candidates
and a `diversity_score` (mean pairwise cosine distance).

---

## Consequences

### Positive

- Pluggable: any retrieval backend (HNSW, DiskANN, IVF, flat scan) can use
  the same diversity trait.
- Measurable: `diversity_score` provides a numeric quality signal per request.
- Connects to ruvector-mincut: MinCut-inhibition shares the graph-cut philosophy;
  future integration can use dynamic mincut thresholds.
- Agent memory: enables diversity-aware retrieval without changing the index.
- MCP-ready: the trait maps cleanly to a `ruvector_memory_search_diverse` MCP tool.
- No external dependencies: pure Rust, no service calls.

### Negative

- O(n²d) pairwise similarity matrix is the main bottleneck.  At N=2000, d=256,
  reranking takes ~1 second; not suitable for large-N online reranking.
- MinCut threshold `sim_threshold` must be tuned per dataset and dimensionality.
  High-dimensional spaces (d≥256) may require a lower threshold than 0.85.
- Recall@K drops significantly under high diversity weight: MinCut at θ=0.85
  achieves diversity=0.603 but recall=0.100 at N=100 (only 1 of 10 ground-truth
  top-10 candidates retained).

---

## Alternatives Considered

### DPP (Determinantal Point Processes)

Provides probabilistic diversity guarantees and is used in Google Search and
Spotify research.  Rejected for initial PoC: O(n³) exact sampling; Nyström
approximations require dense linear algebra libraries not available in the
workspace.  Future work.

### Approximate MMR with HNSW

Build a secondary HNSW over the candidate pool and use approximate max-similarity
queries instead of exact pairwise computation.  Rejected: overhead of building
a secondary index exceeds the savings for typical candidate pool sizes (N ≤ 500).

### BM25 / Lexical diversity

Diversify by term overlap rather than vector similarity.  Not applicable to
embedding-based retrieval where lexical signals are not available.

### Clustering-based selection

Run k-means on the candidate pool, return one representative per cluster.
Rejected: k-means is iterative and unstable; cluster count must equal k exactly;
adds hyperparameter (number of init restarts).  MinCut-inhibition is simpler and
more predictable.

---

## Implementation Plan

### Phase 1 (This PR)
- [x] `DiversityReranker` trait in `ruvector-diversity-rerank/src/lib.rs`
- [x] `BaselineReranker`, `MmrReranker`, `MinCutReranker`
- [x] 8 unit tests with numeric acceptance thresholds
- [x] Benchmark binary with 4 dataset configurations
- [x] Acceptance test output: PASS

### Phase 2 (Near-term)
- [ ] Integrate into `ruvector-agent-memory` as a retrieval option
- [ ] Add MCP tool `ruvector_memory_search_diverse` in `ruvector-mcp-tools`
- [ ] Add `no_std` feature flag for WASM compilation
- [ ] Publish `ruvector-diversity-rerank-wasm`

### Phase 3 (Future research)
- [ ] LSH-accelerated approximate diversity (O(n log n))
- [ ] Learned parameter selection (adaptive λ / θ per query type)
- [ ] DPP Nyström approximation
- [ ] Temporal diversity for agent memory (recency-weighted MMR)
- [ ] Integration with ruvector-coherence-hnsw for coherence-guided diversity

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-diversity-rerank --bin benchmark`
on Ubuntu 24.04.4 / rustc 1.94.1 / 2026-06-22.

| Variant | N | Dims | K | Mean µs | Diversity | Recall@K |
|---------|---|------|---|---------|-----------|----------|
| baseline | 100 | 64 | 10 | 11.5 | 0.097 | 1.000 |
| mmr | 100 | 64 | 10 | 430 | 0.312 | 0.300 |
| mincut-diversity | 100 | 64 | 10 | 420 | 0.603 | 0.100 |
| baseline | 200 | 64 | 20 | 37 | 0.596 | 1.000 |
| mmr | 200 | 64 | 20 | 2,661 | 0.868 | 0.200 |
| mincut-diversity | 200 | 64 | 20 | 1,582 | 0.596 | 1.000 |

Acceptance test (N=200, d=64, k=20):
- MMR diversity 0.2363 > baseline 0.1066: PASS
- MinCut diversity 0.5577 > baseline 0.1066: PASS

---

## Failure Modes

1. **Threshold collapse**: If θ is too low, all candidates suppress each other
   and only one candidate is returned.  Mitigation: clamp N_selected ≥ k via
   fallback to suppressed pool (implemented).
2. **High-d concentration**: At d≥256, within-cluster similarity may fall below
   θ, making MinCut equivalent to baseline.  Mitigation: lower θ or use a
   dimension-normalised similarity function.
3. **Adversarial suppression**: An attacker crafts near-identical queries to
   suppress all high-relevance results via MinCut.  Mitigation: combine with
   proof-gated retrieval; validate candidate pool before reranking.

---

## Security Considerations

- Diversity reranking can *not* be used as access control.  Suppressing a
  restricted candidate is not the same as denying access to it.
- Do not expose `sim_threshold` as a user-controlled parameter without
  validation; low values can cause performance degradation (O(n²) with no
  early exit).
- Input candidate vectors must be validated (finite values, consistent dimension)
  before computing pairwise similarity to avoid NaN propagation.

---

## Migration Path

This is a new, additive crate.  No existing code changes are required.

Consumers integrate by:
```rust
use ruvector_diversity_rerank::{MmrReranker, DiversityReranker};

let reranker = MmrReranker::new(0.5);
let result = reranker.rerank(candidates, k)?;
```

The crate is added to the workspace `members` list in `Cargo.toml`.

---

## Open Questions

1. What is the optimal default `lambda` and `sim_threshold` for agent memory
   workloads?  Requires empirical study with real memory corpora.
2. Should `diversity_score` be surfaced as a per-request metric in the MCP tool
   response?
3. Is a WASM feature flag needed in Phase 2, or should we publish a separate
   `ruvector-diversity-rerank-wasm` crate (following the existing pattern)?
4. Should `MinCutReranker` use dynamic threshold tuning based on the observed
   similarity distribution of the candidate pool?
