---
adr: 194
title: "Agent Memory Compaction via Graph-Cut Clustering"
status: accepted
date: 2026-05-26
authors: [ruvnet, claude-flow]
related: [ADR-014, ADR-016, ADR-193, ADR-143]
tags: [agent-memory, compaction, graph-cut, union-find, cosine-similarity, recall, nightly-research]
---

# ADR-194 — Agent Memory Compaction via Graph-Cut Clustering

## Status

**Accepted.**  Implemented on branch `research/nightly/2026-05-26-agent-memory-graph-compaction`
as `crates/ruvector-memcompact`.  All 12 unit tests pass; build is green with
`cargo build --release -p ruvector-memcompact`.

---

## Context

AI agent memory stores grow without bound during long-running sessions.  Every
production agent memory system surveyed (MemGPT, Letta, Mem0, A-MEM) treats
compaction as either manual, agent-directed, or absent entirely.  The nearest
prior art — LanceDB's fragment merge — is a structural I/O optimisation with no
semantic awareness: it reorganises file pages but does not consider whether two
entries carry the same information.

Three failure modes result:

1. **Recall collapse** — age-based eviction discards historical semantic clusters.
   An agent that encountered a concept one week ago loses it entirely if that
   period has been evicted.

2. **Relevance drift** — importance-based eviction correlates with recent
   reinforcement, not semantic coverage.  Niche but critical memories evaporate.

3. **Index bloat** — without compaction, HNSW and IVF indexes accumulate
   near-duplicate vectors, increasing query latency and memory footprint
   quadratically with agent lifespan.

RuVector already contains `ruvector-mincut` (dynamic min-cut), `ruvector-graph`
(distributed graph storage), and `ruvector-core` (HNSW).  Connecting these with
a cosine-similarity graph over stored memory vectors enables semantic-aware
compaction as a first-class RuVector primitive.

---

## Decision

Introduce `crates/ruvector-memcompact` implementing a `CompactionStrategy` trait
and three concrete variants:

| Variant              | Mechanism                                              |
|----------------------|--------------------------------------------------------|
| `AgeEviction`        | Sort by timestamp desc, keep top-`budget`              |
| `ImportanceEviction` | Sort by importance score desc, keep top-`budget`       |
| `GraphCutCompaction` | Pairwise cosine graph → Union-Find clusters → centroids|

### GraphCutCompaction algorithm

```
1. For all pairs (i, j): compute cosine_sim(v_i, v_j)
2. Union-Find: merge i ↔ j when sim ≥ threshold θ  [O(n² · d) + O(n α(n))]
3. For each component C: synthesise centroid vector, max(importance), max(ts)
4. If |representatives| > budget: trim by descending importance
5. Return compacted MemoryStore of representative entries
```

The key insight is that agent memories within a semantic cluster are largely
redundant: the centroid captures what the cluster "knows" while shrinking the
index proportionally to cluster density.  Recall is preserved because every
query pointing at a cluster's region finds its centroid in the first few
results.

### API shape (stable)

```rust
pub trait CompactionStrategy {
    fn name(&self) -> &'static str;
    fn compact(&self, store: &MemoryStore, budget: usize) -> MemoryStore;
}
```

`MemoryStore` and `MemoryEntry` are minimal types that can be adapted to
`ruvector-core`'s vector storage without changes to the trait.

---

## Consequences

### Positive

- GraphCutCompaction achieves **100% cluster-level recall@10** at 5% budget
  (20 entries from 500) vs 5% for AgeEviction and 75% for ImportanceEviction.
- 96% memory reduction while preserving all semantic cluster representatives.
- O(n² · d) compaction time is acceptable for agent memory stores (n ≤ 10 K).
- Zero external dependencies; no Python, no services, no network.
- The `CompactionStrategy` trait is composable: pipelines can chain strategies
  (e.g., age-gate new entries, then graph-compact the survivors).

### Negative / risks

- O(n²) pairwise scan becomes expensive beyond ~10 K entries; a k-NN graph
  approximation (as used by HNSW construction) would reduce this to O(n log n)
  but adds implementation complexity.
- Threshold θ is a hyperparameter; wrong values split true clusters or merge
  distinct ones.  Auto-calibration from the empirical similarity distribution
  is left for a follow-up.
- Centroid vectors are not actual stored memories — they are synthetic
  aggregates.  Downstream explainability tools must account for this.

---

## Alternatives considered

### A. Streaming eviction with decay
Time-decay importance scoring, where each entry's effective score falls
exponentially.  Simple but fundamentally age-based; loses rare-but-important
memories at the same rate as frequent-but-stale ones.

### B. LRU + importance hybrid
Combine recency and importance with a weighted score.  Better than pure age
or importance alone, but still scalar: no semantic structure in the eviction
decision.

### C. Hierarchical agglomerative clustering (HAC)
Full linkage-based dendrogram construction.  More principled than Union-Find
thresholding but O(n² log n) memory and quadratic-to-cubic time; overkill for
agent memory stores at current scale.

### D. Apply MinCut to the HNSW graph directly
Rather than a fresh pairwise pass, reuse the existing HNSW proximity graph for
clustering.  This would be O(E) where E is the HNSW edge set (sparse), making
it vastly faster.  Requires integration with `ruvector-core`'s HNSW internals;
left as the primary production path in the roadmap.

---

## Implementation plan

| Phase | Description                                | Milestone |
|-------|--------------------------------------------|-----------|
| PoC   | `crates/ruvector-memcompact` standalone    | Done ✓    |
| 1     | Wire `MemoryStore` to `ruvector-core` VecDB | Q3 2026  |
| 2     | HNSW-graph-reuse compaction path           | Q4 2026   |
| 3     | ruFlo hook for scheduled compaction        | Q1 2027   |
| 4     | MCP tool: `memory_compact`                 | Q1 2027   |
| 5     | k-NN graph approximation for n > 10 K     | Q2 2027   |

---

## Benchmark evidence

Hardware: Intel(R) Xeon(R) @ 2.80 GHz, Linux x86_64.
Dataset:  N=500, D=64, K=20 clusters, σ=0.15, budget=25.
Command:  `cargo run --release -p ruvector-memcompact`

| Strategy             | Entries | Compact ms | Query μs | Recall@10 | Memory KB | Reduction |
|----------------------|---------|------------|----------|-----------|-----------|-----------|
| AgeEviction          | 25      | 0.07       | 2.27     | 5.0%      | 7.8       | 95.0%     |
| ImportanceEviction   | 25      | 0.04       | 2.26     | 75.0%     | 7.8       | 95.0%     |
| **GraphCutCompaction** | **20** | **9.50** | **1.75** | **100.0%** | **6.2** | **96.0%** |

Acceptance: GraphCutCompaction recall@10 ≥ 75% → **100.0% PASS**.
            GraphCut beats AgeEviction by ≥ 10 pp → **+95 pp PASS**.

---

## Failure modes

1. **Threshold miscalibration** — if θ is too high, entries cluster poorly
   and the algorithm degrades to ImportanceEviction; if too low, unrelated
   memories merge and recall drops.  Mitigation: emit the intra-cluster
   similarity distribution in the benchmark output for operator inspection.

2. **Adversarial memory injection** — an agent receiving adversarially crafted
   embeddings could cause two legitimate memories to appear similar and merge,
   destroying one.  Mitigation: proof-gated writes (ADR-ruvector-verified)
   before compaction.

3. **Centroid drift** — repeated compaction cycles may shift cluster centroids
   away from real memories toward synthetic averages, degrading retrieval
   quality over time.  Mitigation: cap compaction depth; retain raw entry
   metadata alongside the centroid vector.

---

## Security considerations

- Compaction modifies the memory store in place; a faulty compaction strategy
  could irreversibly destroy agent memory.  The `compact` function returns a
  new `MemoryStore`, leaving the original intact — callers decide when to swap.
- No network calls, no external state.  Side-channel risk is minimal.
- Cluster label leakage: if cluster membership reveals agent operational
  patterns, the centroid-only output reduces leaked information versus storing
  all raw entries.

---

## Migration path

`crates/ruvector-memcompact` is standalone today.  Production integration
requires:

1. Implement `From<ruvector_core::VecStore> for MemoryStore` (trivial mapping).
2. Add `Compactor` to `ruvector-server` as an optional background task.
3. Expose `CompactionStrategy` as a ruFlo step type.

No changes to `ruvector-core` public API are required for Phase 1.

---

## Open questions

1. What is the correct auto-calibration procedure for threshold θ?
2. Should centroid entries carry a `source_count` field for provenance?
3. At what corpus size should we switch from pairwise to k-NN graph?
4. Can `ruvector-mincut` replace Union-Find for higher-quality cuts at
   moderate n?
5. Should compaction be triggered by size, staleness, or query latency SLO?
