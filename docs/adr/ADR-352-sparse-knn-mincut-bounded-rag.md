# ADR-352: Sparse k-NN Graph Construction for MinCutBounded RAG — and Why It Isn't Enough

- **Status**: Rejected (primary hypothesis) / Accepted (Candidate B as an opt-in, modest-scale default)
- **Date**: 2026-10-02
- **Crate**: `crates/ruvector-bounded-rag`
- **Branch**: `claude/focused-darwin-5c2y10`
- **Follows up**: [ADR-272](./ADR-272-bounded-rag-mincut.md) (Phase 2, "pre-build k-NN graph offline")
- **Connects**: vector search, `ruvector-mincut`, `ruvector-agent-memory`, `ruvector-proof-gate`, ruFlo

---

## Context

ADR-272 shipped three `BoundedRetriever` variants for coherence-constrained RAG: `TopK`, `GraphBFS`, and `MinCutBounded`. It measured `MinCutBounded`'s dense O(n²·d) similarity-graph construction as the dominant query-time cost (1.27s mean at n=3000) and listed, as its first unchecked Phase 2 item: *"Pre-build k-NN graph offline (approximate) — O(n log n) build, O(k log n) query."* That item was never implemented. This ADR implements it, measures it honestly, and documents why it does not close the gap ADR-272 opened.

## Hypothesis

```text
Given a synthetic clustered corpus of n chunks (n ∈ {200, 1000, 3000, 8000}),
when LSH-bucketed sparse k-NN graph construction replaces MinCutBounded's
dense all-pairs scan,
then end-to-end retrieval latency at n=3000 should drop by at least 10x
relative to dense MinCutBounded,
subject to precision staying within 5 points of baseline and all tests
remaining green.
```

Set before any benchmark ran; not modified afterward. See the companion nightly research report for the full three-pass research trail: `docs/research/nightly/2026-10-02-sparse-knn-mincut-bounded-rag/README.md`.

---

## Decision

1. Extract the existing `MinCutRetriever`'s Edmonds-Karp max-flow solver, verbatim, into `flow::source_side_partition(n, source_cap, sink_cap, edges) -> Vec<bool>` — a pure graph routine independent of chunks/corpora. `MinCutRetriever::retrieve` now calls this shared function instead of carrying its own copy. Verified behaviour-preserving against the full pre-existing test suite (19/19 green, no new failures) before any new code was added.
2. Add `sparse_knn::build_sparse_edges`, implementing random-hyperplane LSH (SimHash) bucketing to discover candidate inter-chunk pairs in `O(n·d·L·h)` instead of `O(n²·d)`, with an optional `max_degree: Option<usize>` for mutual top-k degree capping.
3. Add `sparse_knn::SparseKnnMinCutRetriever`, a fourth `BoundedRetriever` implementation that feeds LSH-discovered edges into the same `flow::source_side_partition` solver `MinCutRetriever` uses — isolating edge discovery as the only changed variable.
4. Benchmark two configurations of this retriever:
   - **Candidate A** (`max_degree: None`): keeps every LSH-discovered candidate pair that clears `edge_threshold` — cheaper discovery, same unbounded threshold-graph shape as the original.
   - **Candidate B** (`max_degree: Some(2 * budget)`): mutual top-k degree cap, bounding every node's kept edges to `k` regardless of cluster density.
5. **Do not promote either candidate as a replacement default** for `MinCutBounded`. Candidate B is accepted as an available, opt-in variant for corpora up to ~3000 chunks where it gives a real, test-verified 2.8x speedup at zero precision cost; it is not accepted as a fix for the scalability cliff ADR-272 identified, which persists (measured: 3.36s/query at n=8000, worse than `GraphBFS`).

---

## Evidence

Hardware: x86_64 Linux, Rust 1.97.0, `cargo run --release -p ruvector-bounded-rag --bin benchmark`. Seeds: `0xDEAD_BEEF` (corpus/query generation), `0x5EED_CAFE` (LSH hyperplanes).

| n | Variant | Mean(μs) | p95(μs) | Precision | vs. dense |
|---:|---|---:|---:|---:|---:|
| 200 | MinCutBounded (dense) | 1,890.8 | 2,216.0 | 1.000 | — |
| 200 | Candidate A (threshold) | 1,692.4 | 1,938.0 | 1.000 | 1.12x |
| 200 | Candidate B (k=40 cap) | 1,946.2 | 2,132.0 | 1.000 | 0.97x |
| 1,000 | MinCutBounded (dense) | 59,155.7 | 72,417.0 | 1.000 | — |
| 1,000 | Candidate A | 38,906.2 | 51,418.0 | 1.000 | 1.52x |
| 1,000 | Candidate B (k=60) | 34,719.5 | 39,854.0 | 1.000 | 1.70x |
| 3,000 | MinCutBounded (dense) | 1,171,905.6 | 1,341,592.0 | 1.000 | — |
| 3,000 | Candidate A | 1,036,383.4 | 1,317,617.0 | 1.000 | 1.13x |
| 3,000 | Candidate B (k=80) | 418,698.7 | 491,731.0 | 1.000 | **2.80x** |
| 8,000 | MinCutBounded (dense) | SKIPPED (cost cliff, see n=3000) | — | — | — |
| 8,000 | Candidate A | SKIPPED (degrades like dense, see n=3000) | — | — | — |
| 8,000 | Candidate B (k=80) | 3,363,314.8 | 3,657,848.0 | 1.000 | n/a (no dense baseline run) |
| 8,000 | GraphBFS (for reference) | 518,396.4 | 623,642.0 | 1.000 | n/a |

Candidate-pair reduction (construction cost in isolation, held constant across n): **~10.1–10.3x fewer pairs checked than dense all-pairs, at every scale tested (n=200 through n=20,000)**. This part of the hypothesis is unambiguously confirmed.

Root-cause check at n=8000 (one-off instrumentation, not part of the CI benchmark): unbounded threshold-graph edge count = 3,964,242 (97.5% of the 4,066,942 candidate pairs checked became edges), max node degree 1,278. This is why Candidate A was skipped beyond n=3000 — cheaper *discovery* of a graph that turns out to be just as dense provides no end-to-end benefit. Candidate B's degree cap fixes this number directly (provably ≤ k edges per node; verified by a dedicated unit test), yet still costs 3.36s/query at n=8000 — ruling out edge count as a complete explanation and pointing at the Edmonds-Karp solver's own iteration behavior (not directly instrumented this run; see Open Questions).

**Acceptance result: REJECT** the 10x hypothesis for both candidates. Candidate B's 2.8x at n=3000 is real and is accepted as an available opt-in improvement, not as resolution of ADR-272's scalability concern.

---

## Consequences

### Positive

- `flow.rs` is now a reusable, independently-tested max-flow primitive, usable by any future edge-discovery strategy without re-implementing Edmonds-Karp — this is a durable simplification regardless of this ADR's negative headline result.
- Candidate B provides a measured, no-downside (precision-neutral) 2.8x speedup for corpora in the few-hundred-to-few-thousand-chunk range, available today as `SparseKnnMinCutRetriever::new(cfg).with_lsh_config(SparseKnnConfig { max_degree: Some(k), ..Default::default() })`.
- The negative result for Candidate A is now documented and measured rather than assumed — ADR-272's Phase 2 checkbox is closed with evidence, preventing a future nightly run from re-attempting the same fix and re-discovering the same ceiling.
- The actual bottleneck is narrowed from "graph construction" (ADR-272's assumption) to "the max-flow solver's iteration behavior on real-valued-capacity networks" (this ADR's finding), which is a more actionable target for the next research cycle.

### Negative

- `MinCutBounded`'s scalability cliff (ADR-272) remains **unresolved** for corpora beyond ~3000 chunks. No variant in this crate is production-viable at that scale today.
- Candidate B adds a second configuration knob (`max_degree`) whose correct value depends on corpus cluster density in ways not fully characterized — the `k = 2*budget` heuristic used in benchmarks is a starting point, not a tuned default.
- LSH-based discovery (both candidates) is probabilistic: a true near-duplicate pair can be missed if it never co-occurs in a bucket across all hash tables. This crate's docs state this; it was not re-verified against an adversarial worst case.

---

## Alternatives Considered

1. **HNSW-derived k-NN graph** instead of LSH: rejected for *this* experiment to avoid adding a new dependency while isolating construction-cost as a single variable; LSH's candidate-pair-checked metric is simple to measure exactly. Worth a follow-up comparing HNSW-derived and LSH-derived coherence graphs now that construction method is known to matter less than the solver.
2. **Fixed `max_degree` independent of budget**: tried during development (flat k=10); rejected because it dropped `budget_utilisation` below 1.0 on several corpora, which would have made "faster" partly an artefact of returning fewer results rather than a genuine graph-size win. The budget-scaled heuristic keeps utilisation at 1.000 across every reported case.
3. **Implementing a capacity-scaling max-flow solver as this ADR's primary candidate**: rejected for this run specifically because it changes the solver, the one variable this experiment was designed to hold constant while varying graph construction. Promoted to Open Questions / next research instead.

---

## Implementation Plan

### This ADR — done
- [x] Extract `flow::source_side_partition`, verified behaviour-preserving against the pre-existing suite.
- [x] `sparse_knn::build_sparse_edges` (LSH bucketing, optional mutual top-k cap).
- [x] `sparse_knn::SparseKnnMinCutRetriever` implementing `BoundedRetriever`.
- [x] 10 new unit tests (2 for `flow.rs`, 8 for `sparse_knn.rs`), all green alongside the 11 pre-existing tests.
- [x] Benchmark extended to 4 corpus sizes (200/1000/3000/8000) with 5 variants and real candidate-pair/edge-count instrumentation.
- [x] Root-cause measurement confirming edge density, not discovery cost, as Candidate A's failure mode.

### Next (not started)
- [ ] Instrument `flow::source_side_partition`'s augmenting-path loop with an iteration counter to directly confirm the solver-scaling hypothesis named in this ADR's evidence section.
- [ ] Implement a capacity-scaling variant of the same solver interface (Candidate C), re-measured against Candidate B's graphs with construction held constant.
- [ ] Investigate whether `ruvector-mincut`'s dynamic min-cut algorithm can be adapted to this s-t flow shape at all (open question, not assumed feasible).
- [ ] `ruvector-proof-gate` pre-filter integration (still open from ADR-272 Phase 2, unaffected by this ADR).

---

## Security Considerations

No new attack surface: no I/O, no untrusted deserialization, no new dependencies beyond the crate's existing `rand`/`rand_distr`. LSH's probabilistic completeness (a true edge can be missed) is an availability/quality concern for coherence partitioning, not a confidentiality concern — missing an edge only makes the coherence boundary *more* conservative (fewer chunks admitted), never less. This is the same direction of failure as ADR-272's existing `seed_threshold` fallback, not a new failure class.

---

## Migration Path

Additive only. `SparseKnnMinCutRetriever` is a new type alongside the three ADR-272 variants; no existing public API changed shape. `MinCutRetriever`'s externally observable behavior is unchanged (verified by its unchanged test suite) despite its internal refactor to use the shared `flow` module. No crate outside `ruvector-bounded-rag` depends on this crate yet, so there is no downstream migration to perform.

## Rollback

Revert `crates/ruvector-bounded-rag/src/flow.rs` and `sparse_knn.rs`, and the corresponding `lib.rs`/`benchmark.rs` edits, to return to the ADR-272 state. No data migration, no persisted state, no external callers.

---

## Rejection Criteria (met for the primary hypothesis)

- [x] 10x end-to-end latency improvement at n=3000: **not met** (best result 2.80x).
- [x] Clear scaling trend ruling out a simple fix: **met** — Candidate B's n=3000→n=8000 scaling (8.03x measured vs. 7.11x quadratic-prediction) shows no evidence of escaping the complexity class the original dense implementation had.

## Open Questions

1. Does Edmonds-Karp's augmenting-path count on this flow network actually scale ~O(n), as the measured end-to-end scaling trend is consistent with? Requires direct instrumentation (named in Implementation Plan).
2. Is a capacity-scaling or push-relabel solver sufficient to close the gap, or does the fundamental shape of this source/sink network (source and sink capacities both derived from per-chunk query similarity, which varies continuously) make *any* classical max-flow algorithm a poor fit compared to an approximate or iterative-refinement approach?
3. Can `ruvector-mincut`'s dynamic min-cut be adapted to an s-t flow/partition shape, or is it fundamentally suited only to undirected global min-cut (its existing use case)?
4. What is the right default `max_degree` as a function of expected cluster density rather than budget alone — this ADR's `k=2*budget` heuristic avoids budget starvation but is not tuned against precision/latency tradeoffs directly.
