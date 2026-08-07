# ADR-298: Hierarchical Cluster-Summary Retrieval for Agent Memory RAG

- **Status**: Proposed
- **Date**: 2026-08-07
- **Author**: nightly research agent
- **Crate**: `ruvector-cluster-rag`
- **Related**: ADR-272 (speculative-ann), ADR-254 (turbovec), ADR-269 (agent-memory-compaction)
- **Branch**: `research/nightly/2026-08-07-hierarchical-cluster-rag`

---

## Context

Agent memory corpora grow continuously. A corpus of 10K vectors (a single agent session) can be searched by brute force at ~670 QPS. At 1M vectors the same approach yields ~6 QPS — too slow for interactive use. RuVector needs a simple, zero-dependency cluster index that:

1. Reduces per-query scan cost without requiring a full HNSW graph.
2. Integrates with the coherence primitives already in `ruvector-coherence`.
3. Compiles to WASM for edge deployments.
4. Supports incremental inserts without graph maintenance.

This ADR records the design decision for a two-level cluster-summary index (ClusterTree) and a coherence-weighted query routing variant (CoherenceTree), both benchmarked against brute-force baseline.

---

## Decision

### What is being decided

Introduce `ruvector-cluster-rag` as a standalone zero-dependency crate implementing:
- K-means based cluster tree (`ClusterTree`) with per-cluster cohesion scores.
- `ClusterSearch`: IVF-style retrieval routing queries to top-nprobe clusters by centroid L2 distance.
- `CoherenceTree`: modified routing that weights centroid similarity by cluster internal cohesion.
- `FlatBrute`: brute-force reference for recall measurement.
- `AnnVariant` trait shared with other nightly crates.

### What belongs in this crate

- Cluster construction and cohesion computation.
- Inverted list management.
- Query routing logic (both L2 and coherence-weighted).
- Benchmark binary with acceptance gate.

### What remains behind a feature flag or future work

- Online insert with deferred centroid update (no flag yet; planned as `online-insert` feature).
- SIMD distance acceleration (behind `simd` feature in future).
- Three-level tree for n > 1M.
- MCP tool surface (separate integration crate).
- RVF serialisation (to `ruvector-cluster-rag-rvf`).

---

## Consequences

### Positive

- Zero external dependencies; compiles to WASM without changes.
- 1.44–1.52× measured speedup over brute-force at 50% nprobe coverage.
- 2% memory overhead over raw leaf storage.
- Clean separation: `ClusterTree` is an immutable index; routing policy is pluggable.
- Coherence scoring connects to the existing `ruvector-coherence` primitive set.
- Build time (k-means, 4s for 10K vectors) amortises over many queries.

### Negative

- On uniform random data, CoherenceTree provides no recall advantage over ClusterSearch (same routing decisions when all clusters have near-equal cohesion).
- At 50% nprobe coverage, recall is 0.78 — lower than HNSW's typical ~0.95 at similar latency on real embeddings.
- k-means build time is O(n·k·d·iters); rebuild required when corpus shifts significantly.
- No persistence format yet; index must be rebuilt on restart.

### Neutral

- The `AnnVariant` trait mirrors the pattern from `ruvector-speculative-ann` (ADR-272); these should be unified into `ruvector-core::ann` in a future pass.

---

## Alternatives Considered

### 1. HNSW (ruvector-coherence-hnsw)

Already implemented (ADR-241). Achieves ~0.95 recall at comparable latency but requires O(M·log(n)) memory for the graph and non-trivial graph maintenance under inserts/deletes. For the growing-memory use case, cluster indexes are simpler to maintain. Decision: HNSW remains the primary production index; ClusterTree is the simpler, insert-friendly complement.

### 2. LSM-ANN (ruvector-lsm-ann)

LSM-style indexing (ADR-256) buffers inserts and merges periodically. LSM-ANN handles streaming inserts well but requires more complex merge logic. ClusterTree insert (assign new vector to nearest centroid) is O(k·d) — simpler than LSM merge. Decision: the two approaches are complementary; ClusterTree is the read-optimised half of a future LSM+Cluster hybrid.

### 3. SPANN (ruvector-spann)

SPANN (ADR-261) handles billion-scale by combining in-memory posting list heads with SSD tails. Heavier infrastructure, requires SSD. ClusterTree targets the 10K–1M range where everything fits in RAM. Decision: different scale targets; not in conflict.

### 4. RAPTOR with LLM summarisation

Full RAPTOR builds cluster summaries using an LLM — the text summary becomes the centroid. Richer but requires Python or a model inference dependency. Out of scope for a zero-dependency Rust crate. Decision: centroid-as-mean is the practical default; LLM-enhanced summaries can be embedded as externally computed vectors inserted into the same tree structure.

---

## Implementation Plan

1. `ruvector-cluster-rag` crate: **done** (this ADR).
2. Validate on real embedding corpus: next step (ann-benchmarks SIFT1M or MS-MARCO embeddings).
3. Online insert feature: buffer new vectors, absorb into nearest centroid after `N` inserts or `ttl` seconds.
4. Adaptive nprobe controller: borrow ruFlo feedback loop from `ruvector-speculative-ann`.
5. SIMD L2/cosine: add AVX2 path behind `simd` feature flag, measure improvement.
6. RVF serialisation: pack centroid + inverted lists into `.rvf` manifest.
7. MCP tool: `memory_search(query, nprobe, k)` wrapper.

---

## Benchmark Evidence

Run: `cargo run --release -p ruvector-cluster-rag --bin benchmark`
Date: 2026-08-07, x86_64 Linux, release build.
Dataset: n=10,000, dim=128, k=10, 500 queries, k_clusters=40, nprobe=20.

| Variant | Mean µs | p95 µs | QPS | Recall@10 |
|---------|---------|--------|-----|-----------|
| FlatBrute (ground truth) | 1490.9 | 1567.7 | 671 | 1.000 |
| ClusterSearch (20% nprobe) | 1034.9 | 1270.1 | 966 | 0.779 |
| CoherenceTree (20% nprobe) | 981.4 | 1070.6 | 1019 | 0.776 |

Memory overhead: 2.0% above raw leaf storage.
Acceptance gate: PASS (both cluster variants ≥ 0.70 recall@10).

All numbers are from a real `cargo run --release` invocation. No aspirational values.

---

## Failure Modes

| Failure | Trigger | Detection | Mitigation |
|---------|---------|-----------|-----------|
| Low recall at nprobe/k boundary | nprobe too small for corpus structure | Per-query recall monitoring | Increase nprobe or switch to HNSW |
| Stale centroids | Bulk insert without re-cluster | Cohesion decay rate > threshold | ruFlo-triggered periodic re-cluster |
| Empty clusters | k too large, sparse regions | Cluster size monitoring | k ≤ sqrt(n) heuristic; merge empty clusters |
| Memory OOM at scale | n=10M+ with large dim | Pre-flight memory estimate | Three-level tree splits the problem |
| CoherenceTree offers no advantage | Uniform corpus | Recall parity with ClusterSearch | Expected; use ClusterSearch instead |

---

## Security Considerations

- No network I/O; pure in-process computation.
- Centroid vectors embed statistical averages over members — equivalent sensitivity to member vectors themselves. Apply same access controls.
- For proof-gated deployments: add witness signature requirement at `ClusterTree::insert` following the `ruvector-proof-gate` pattern (ADR-239).
- No `unsafe` code in this crate.

---

## Migration Path

From brute-force `FlatBrute`:
1. Construct `ClusterTree::new(corpus, kmeans(&corpus, k, 20))`.
2. Replace `flat.search(q, k)` calls with `cluster_search.search(q, k)`.
3. Set `nprobe` to achieve target recall from pre-flight measurement.
4. Optionally enable `CoherenceTree` variant when corpus is structured.

From HNSW:
- No migration needed; ClusterTree is a complementary index, not a replacement.
- Use ClusterTree for insert-heavy workloads; HNSW for highest recall.

---

## Open Questions

1. Does CoherenceTree achieve measurable recall advantage on real structured embedding corpora (e.g., MS-MARCO, AgentBench)?
2. What is the optimal adaptive nprobe policy for a target recall of 0.90?
3. Should the `AnnVariant` trait be lifted to `ruvector-core` to unify the nightly crate interface?
4. Is k-means the right clustering algorithm, or would Gaussian Mixture Models (GMM) better capture natural cluster shapes in agent memory?
5. Can cohesion decay serve as a practical memory eviction signal when combined with `ruvector-temporal-coherence`?
