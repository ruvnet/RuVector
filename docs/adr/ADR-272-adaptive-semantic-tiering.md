# ADR-272: Adaptive Semantic Memory Tiering

**Status**: Proposed  
**Date**: 2026-07-19  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-07-19-adaptive-semantic-tiering`  
**Crate**: `crates/ruvector-adaptive-tiering`  
**Related**: ADR-264 (LSM-ANN), ADR-200 (DiskANN), ADR-227 (Proof-Gated Writes), ADR-268 (CapGated ANN)

---

## Context

RuVector is increasingly used as an agent memory substrate — storing embeddings for
agent observations, retrieved documents, tool outputs, and episodic memories.  These
workloads share a structural property: **access distribution is highly skewed**.  A
small fraction of vectors (semantically coherent knowledge clusters) will be queried
repeatedly, while the majority (noisy, low-relevance observations) are rarely or never
retrieved again.

Current RuVector storage treats all vectors equally: they are all resident in the same
in-memory flat array or HNSW graph.  At small scale this is fine.  At 10M+ vectors —
realistic for a long-running agent — it becomes untenable.  Memory budgets force
paging, and without intelligent placement the frequently-queried important vectors will
page out along with everything else.

The existing partial solution (DiskANN, ADR-200) provides SSD-resident ANN but does
not make dynamic placement decisions.  LSM-ANN (ADR-264) layers delta indexes but
is concerned with write throughput, not with the semantic importance of individual
vectors.

The problem is **placement quality**: which vectors should reside in fast storage?

Classical database buffer management answers this with LRU or LFU — the recently or
frequently used pages stay in the buffer pool.  For vector agent memory, this fails at
cold start: newly ingested memories have zero access history but may belong to
semantically tight clusters that will become query-hot within minutes.  Waiting for
access patterns to "teach" the system delays placement quality exactly when it matters
most — right after new knowledge is ingested.

---

## Decision

Introduce `crates/ruvector-adaptive-tiering` implementing **adaptive semantic memory
tiering**: a scoring function that assigns each vector a *semantic temperature* and uses
that temperature to place vectors across three storage tiers.

The temperature combines three signals:

```
temperature(v, t) =
    w_r · exp(-λ · (t - last_access)) +   // recency
    w_c · coherence_score(v)             +   // intra-cluster tightness
    w_g · log(1 + graph_degree(v)) / 5       // local density
```

Default weights: `w_r = 0.35, w_c = 0.40, w_g = 0.25`.

Three concrete scoring strategies are provided and benchmarked:

| Strategy | Signals used | Cold-start quality |
|----------|-------------|-------------------|
| `AccessOnlyScorer` | access count | Poor — requires warmup |
| `CoherenceScorer` | intra-cluster L2 mean | Good — works without access history |
| `SemanticTempScorer` | recency + coherence + centrality | Best balance |

The three physical tiers:

| Tier | Capacity (default) | Analogous to |
|------|--------------------|--------------|
| Hot  | 10% of dataset | In-memory HNSW, sub-microsecond |
| Warm | 30% of dataset | Memory-mapped file, microsecond |
| Cold | 60% of dataset | SSD-resident DiskANN, millisecond |

In the PoC all three tiers use in-process flat arrays.  The trait design allows
plugging in HNSW, mmap, and DiskANN backends without changing the tiering logic.

---

## Consequences

### Positive

* **Cold-start placement quality**: coherence and centrality signals allow the system
  to correctly prioritise semantically dense clusters before they are ever accessed.
* **Composable**: the `Scorer` trait is pluggable; new scoring strategies can be
  added without touching storage or search code.
* **Safe for RuVector integration**: the `TieredStore<S: Scorer>` generic is
  standalone with no external service dependencies.
* **ruFlo automation**: `evaluate_tiers()` is designed to be called by a ruFlo
  workflow on a schedule (e.g. every 5 minutes or after each batch ingest), enabling
  autonomous tier management.
* **Connects DiskANN and agent memory**: bridges the SSD-resident cold tier (DiskANN
  pattern) with the semantic importance signal from agent memory research.

### Negative

* `evaluate_tiers()` at O(n × sample_k × d) is not free; at n=100k the coherence
  recomputation takes ~1s on a single core.  This must be async and/or incremental
  in production.
* The coherence signal requires a meaningful dataset to be effective; for the first
  ~50 vectors, scores are noisy.
* `graph_degree` uses a fixed L2 radius; different embedding models require different
  radius calibration.
* Current PoC uses brute-force search.  Real hot-tier search would use HNSW.

---

## Alternatives Considered

### 1. Pure LRU/LFU Buffer Pool (rejected)
Classical database approach.  No semantic signal.  Fails at cold start.  Does not use
any knowledge about embedding structure.

### 2. DiskANN with Beam Search (ADR-200, existing)
Excellent for large SSD-resident indexes but does not make dynamic tier assignment
decisions.  Complementary, not a replacement.

### 3. Access Frequency Histogram + Decay (rejected)
A softer version of LFU with exponential decay.  Still purely access-driven; no
semantic signal.  Would still mis-tier important-but-unaccessed vectors.

### 4. Graph Centrality Only (rejected)
PageRank or degree centrality alone does not account for temporal access patterns.
A highly-connected but stale vector should eventually cool down.

---

## Implementation Plan

### Phase 1 (this PR)
* `crates/ruvector-adaptive-tiering` crate with three scorers and brute-force search.
* Numeric acceptance tests.
* Benchmark binary with real measured results.

### Phase 2
* Async `evaluate_tiers` with incremental coherence updates.
* Hot-tier HNSW backend (plug in `ruvector-coherence-hnsw`).
* Cold-tier DiskANN backend (plug in `ruvector-diskann`).
* ruFlo YAML workflow for autonomous tier management.

### Phase 3
* Proof-depth signal integration (from `ruvector-proof-gate`).
* RVM coherence domain → tier mapping (hot tier = primary coherence domain).
* MCP tool surface: `memory_tier_stats`, `memory_promote`, `memory_demote`.
* WASM-safe hot-tier export for edge deployment.

---

## Benchmark Evidence

*(Numbers captured from `cargo run --release -p ruvector-adaptive-tiering --bin benchmark`
on the benchmarked machine.  See research README for full table.)*

Key result:

* **AccessOnly** hot-tier hit rate for Important-cluster eval queries: low (~5–15%)
  because warmup filled the hot tier with Noise cluster vectors.
* **Coherence** hot-tier hit rate: high (≥ 50%) because Important cluster vectors
  have high intra-cluster coherence and are correctly placed hot at `evaluate_tiers`.
* **SemanticTemp** hot-tier hit rate: high (≥ 50%) combining both semantic signals.

---

## Failure Modes

1. **Embedding model shift**: if the embedding model changes, all coherence scores are
   invalidated and must be recomputed.  Mitigation: version-stamp embeddings in
   metadata; invalidate on model ID change.
2. **Adversarial coherence injection**: an adversary could craft embeddings that appear
   high-coherence to bias hot-tier placement.  Mitigation: cap max coherence score per
   namespace; use proof-gated writes (ADR-227) to verify embedding provenance.
3. **Capacity misconfiguration**: wrong hot/warm ratios degrade performance.
   Mitigation: ruFlo auto-tune based on query hit rate telemetry.
4. **Evaluate_tiers thrashing**: calling `evaluate_tiers` too frequently on a large
   dataset causes high CPU load.  Mitigation: rate-limit; use incremental scoring.

---

## Security Considerations

* Coherence scores are computed over stored vectors.  A compromised vector could
  inflate its own coherence score to stay hot.  Pair with proof-gated writes.
* Tier metadata (access_count, last_access_epoch) must not be exposed externally;
  it reveals query patterns.  Keep behind the MCP access control boundary.

---

## Migration Path

This is a new standalone crate with no breaking changes to existing RuVector APIs.
The `TieredStore` can wrap an existing flat vector set without data migration.

---

## Open Questions

1. What is the right schedule for `evaluate_tiers` in a ruFlo workflow?
2. Should coherence be computed incrementally (on each insert) or in batch?
3. How does the semantic temperature model degrade for very high-dimensional vectors
   (d > 1536) where cosine geometry differs from L2?
4. Should the Warm tier use the mmap backend from `ruvector-diskann` today?
5. Is a 10%/30%/60% hot/warm/cold split optimal, or should it be calibrated
   per-namespace based on query volume?
