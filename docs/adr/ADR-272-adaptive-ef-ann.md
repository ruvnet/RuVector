# ADR-272: Adaptive Beam-Width ANN with Query Difficulty Estimation

**Status:** Proposed  
**Date:** 2026-07-13  
**Authors:** RuVector Nightly Research Agent  
**Supersedes:** None  
**Related:** ADR-268 (CapabilityGatedANN), ADR-264 (LSM-ANN), ADR-264 (PQ-ADC)

---

## Context

Standard HNSW and proximity-graph-based ANN search use a fixed beam width (`ef`) for every
query.  The `ef` parameter controls the size of the candidate set explored during graph
traversal: larger ef → higher recall, more distance computations, higher latency.

This one-size-fits-all approach is wasteful.  Queries vary widely in difficulty:

- **Easy queries** have a clearly separated nearest neighbour (distance ratio d₁ ≪ d_k).
  A small ef (e.g. 16) is sufficient to achieve the same recall as ef=64.
- **Hard queries** have many equidistant candidates (d₁ ≈ d_k, high Steiner-hardness).
  A small ef misses true nearest neighbours just outside the beam.

The 2026 state of the art confirms this gap.  DARTH (PACMMOD 2025, arxiv:2505.19001) and
Ada-ef (SIGMOD 2026, arxiv:2512.06636) both show that per-query adaptation can reduce
mean distance computations 2–4× while maintaining recall targets.  However, both require
offline learning (gradient boosting or distribution tables).  No production vector database
(Qdrant, Weaviate, Milvus, LanceDB) implements per-query adaptive ef.

RuVector is positioned as a Rust-native cognition substrate for agents.  Agent memory
workloads are especially heterogeneous: simple factual lookups (low difficulty) coexist
with cross-modal and episodic queries (high difficulty).  A fixed ef either wastes
throughput on easy queries or drops recall on hard ones.

---

## Decision

Add `ruvector-adaptive-ef`, a new crate implementing three search strategies over a
shared proximity graph index:

1. **FixedEf** (baseline) — fixed ef=32 for every query.
2. **TwoStage** — probe with ef=16, escalate to ef=64 only if distance-ratio difficulty
   score exceeds a threshold (default 0.70).
3. **AdaptiveEf** — continuously predict ef ∈ [16, 64] from the distance-ratio score,
   avoiding binary escalation for medium-difficulty queries.

The difficulty estimator is the **distance-ratio score**:

```
difficulty = d₁ / d_k
```

where d₁ is the distance to the nearest candidate from the probe pass and d_k is the
distance to the k-th candidate.  A ratio near 0 → easy query.  A ratio near 1 → hard.

This estimator requires **zero extra distance computations** (it reuses the probe pass
candidates) and **zero offline learning** (no models, no tables, no preprocessing).

The crate exposes a unified `AnnSearch` trait so any search strategy is a drop-in
replacement.  All three strategies operate on the same `ProximityGraph` index.

---

## Consequences

### Positive

- Reduces mean distance computations on easy queries without recall loss.
- Provides a retrieval confidence signal (`1 − difficulty`) usable by ruFlo agents for
  downstream trust scoring.
- Zero learning overhead: the difficulty score requires no training data, no extra passes,
  and no configuration beyond the ef bounds.
- The `AnnSearch` trait allows future strategies (learned ef predictors, Steiner-hardness
  based routing) to plug in without changing index code.
- Composable with CapabilityGatedANN (ADR-268): difficulty routing sits above the access
  control layer.

### Negative

- TwoStage and AdaptiveEf always make a probe pass (ef_min distance computations) before
  deciding whether to escalate.  On a corpus where all queries are hard (high-dimensional
  Gaussian clusters), both strategies use more total ops than FixedEf(32).
- The distance-ratio score is a heuristic, not a theoretically grounded measure like
  Steiner-hardness.  On adversarial corpora it may misclassify difficulty.
- The proximity graph (bottom HNSW layer) is the shared index.  If the calling code uses
  a full HNSW with multiple layers, integrating adaptive ef requires exposing the layer-0
  graph or the per-layer candidate sets.

---

## Alternatives Considered

### A. Fixed High ef (ef=64)

Simple.  Achieves high recall.  Wastes computation on easy queries.  Rejected because
agent memory workloads have a bimodal distribution: many easy queries (recall lookups)
and occasional hard queries (reasoning steps).

### B. Learned Difficulty Predictor (DARTH-style)

Train a gradient boosting model on query features (entry distance, LID estimate) to
predict the minimum ef for a target recall.  Achieves 4× improvement over fixed-ef per
Ada-ef (SIGMOD 2026).  Rejected for this ADR because it requires offline training and
increases deployment complexity.  Suitable as a future upgrade under a separate ADR.

### C. Early Exit Based on Candidate Stability

Stop beam search when adding new candidates does not change the top-k results.  More
principled than distance ratio.  Harder to implement without modifying the beam search
inner loop.  Deferred: the `AnnSearch` trait makes this a future strategy implementor.

### D. Steiner-Hardness Based Routing

Route queries to different ef values based on Steiner-hardness estimated from graph
neighbourhood density.  More accurate but requires O(M) extra distance computations per
query for the graph-density probe.  Deferred to a future ADR.

---

## Implementation Plan

1. Create `crates/ruvector-adaptive-ef` with:
   - `src/lib.rs` — `ProximityGraph`, `l2_sq`, `brute_knn`, `generate_vectors`
   - `src/difficulty.rs` — `distance_ratio_score`, `predict_ef`, `retrieval_confidence`
   - `src/search.rs` — `AnnSearch` trait, `FixedEfSearch`, `TwoStageSearch`, `AdaptiveEfSearch`
   - `src/bin/benchmark.rs` — standalone benchmark binary
2. Add crate to workspace `Cargo.toml` members.
3. Run `cargo test -p ruvector-adaptive-ef` — all tests green.
4. Run `cargo run --release -p ruvector-adaptive-ef --bin benchmark` — capture results.
5. Write `docs/research/nightly/2026-07-13-adaptive-ef-ann/README.md`.
6. Commit and push `research/nightly/2026-07-13-adaptive-ef-ann`.

---

## Benchmark Evidence

Real benchmark results from `cargo run --release -p ruvector-adaptive-ef --bin benchmark`
on N=5,000 × D=128, k=10, 200 queries.  See research document for full table.

Key measured result: AdaptiveEf achieves recall within 5% of FixedEf(32) while using
fewer distance computations on the fraction of easy queries that do not escalate.

---

## Failure Modes

1. **All queries hard** (high-dimensional, dense corpus): both TwoStage and AdaptiveEf
   always escalate, using more total ops than FixedEf.  Mitigation: the benchmark
   measures this and the caller can fall back to FixedEf if escalation rate > 90%.

2. **Threshold miscalibration**: a threshold set too low causes unnecessary escalation;
   too high causes recall loss.  Mitigation: expose threshold as a tunable parameter;
   ruFlo can tune it from recall telemetry over time.

3. **Distance-ratio NaN**: occurs if all k candidates are at distance 0 (exact duplicate
   corpus).  Handled: `distance_ratio_score` returns 1.0 (treats as hard) when d_k < 1e-8.

4. **Graph disconnection**: if the proximity graph has isolated components (rare with M≥8
   but possible), beam search from node-0 may miss some vectors.  The proximity graph
   build algorithm mitigates this by enforcing bidirectional edges.

---

## Security Considerations

- No user input is processed at the Rust API boundary.  Vectors are caller-provided and
  must be validated (dimensionality, NaN/Inf checks) before insertion.
- The difficulty score is not a security mechanism and must not be used for access control.
  Access control remains in CapabilityGatedANN (ADR-268).
- No I/O, no network, no external dependencies.

---

## Migration Path

`ruvector-adaptive-ef` is a standalone research crate.  To integrate into
`ruvector-coherence-hnsw` or `ruvector-core`:

1. Extract `ProximityGraph` into a shared crate or expose it behind a feature flag in
   `ruvector-coherence-hnsw`.
2. Implement `AnnSearch` for the existing `CoherenceHnsw` struct.
3. Replace fixed-ef calls at the search boundary with `AdaptiveEfSearch::search`.
4. Surface `retrieval_confidence` in the `SearchResult` struct for downstream agents.

---

## Open Questions

1. Should the difficulty threshold be per-query-type (learned from ruFlo telemetry) or
   global per collection?
2. Is the distance-ratio score sufficient, or should we integrate Steiner-hardness
   estimation as the primary difficulty signal in a follow-up ADR?
3. Should `retrieval_confidence` be exposed in the MCP `ruvector_search` tool response
   so agents can conditionally verify low-confidence results?
4. What is the right ef range for high-dimensional data (D=768, D=1536)?  Initial
   experiments suggest [32, 256] is more appropriate than [16, 64] for embedding models.
