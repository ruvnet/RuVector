---
adr: 196
title: "MinCut-Guided Agent Working Memory Compaction"
status: accepted
date: 2026-06-02
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-143, ADR-159]
tags: [agent-memory, graph-cut, compaction, vector-search, mincut, mcp, ruvector, nightly-research]
---

# ADR-196 — MinCut-Guided Agent Working Memory Compaction

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-06-02-mincut-memory-compaction` as
`crates/ruvector-mincut-memory`.  18 unit tests pass; build is green with
`cargo build --release -p ruvector-mincut-memory`; all three strategies pass
the numeric acceptance test (recall_after ≥ 0.60 × recall_before at 50%
compaction).

## Context

Long-running AI agents accumulate working memory as vectors.  Without
principled compaction:

1. Storage grows unboundedly.
2. Retrieval latency increases (more vectors to scan).
3. Recall degrades (relevant items compete with stale ones).
4. Agent attention is diluted across outdated context.

No current vector database in the ruvnet ecosystem or in competitors provides
a *graph-coherence-aware* compaction primitive.  All known implementations
(Qdrant TTL, Milvus scalar metadata, FAISS rebuild) are graph-blind.

RuVector is already graph-native via `ruvector-mincut`, `ruvector-graph`, and
`ruvector-coherence`.  This ADR adds the missing agent memory lifecycle
primitive: *which entries should be evicted when the store is full?*

## Decision

We introduce `crates/ruvector-mincut-memory` implementing three variants of
the agent memory compaction problem, each satisfying a common `Compactor`
trait:

```rust
pub trait Compactor {
    fn compact(&self, store: &mut MemoryStore, target_size: usize) -> CompactionResult;
}
```

### AgeEvict (baseline)

Evict the oldest `N - target_size` entries by logical timestamp.  O(N log N).
No graph reasoning.  Useful as a deterministic baseline and as a fallback when
no graph edges exist.

### CoherenceEvict

Score each entry by mean cosine similarity to its graph neighbours.  Evict
lowest-scored entries.  O(N²·D) for graph rebuild + O(N) for scoring.
Preserves semantically dense clusters.

### MinCutEvict (primary recommendation)

Score each entry by *weighted degree* — the sum of all incident edge weights
in the similarity graph.  Evict entries with lowest weighted degree.
O(N²·D) + O(N).

**Why weighted degree approximates minimum cut:**  In max-adjacency orderings
(Stoer-Wagner, Karger-Stein), the vertex with the smallest cumulative
adjacency weight in the ordering defines one side of the minimum cut.
Weighted degree is a polynomial-time proxy: vertices with low total edge
weight are statistically most likely to lie on minimum cuts.  The
approximation is deterministic, auditable, and runs in O(N) after graph
construction.

## Consequences

### Positive

- Agents can compact working memory in < 100 ms for N ≤ 1,000 entries on
  embedded hardware (measured: 53 ms at N=1,000, D=64 on Celeron N4020).
- MinCutEvict retains 2.67× more graph edges than AgeEvict at 50% compaction
  (measured: 2,026 vs 759 at N=1,000).
- All three strategies maintain perfect recall@10 on clustered Gaussian data
  at 50% compaction (measured: 1.000 for all strategies at N=1,000).
- Zero external dependencies beyond `rand` and `rand_distr`.
- WASM-portable with minor adaptation (replace `Instant` with timer argument).
- Trait-based: strategies are swappable without API changes.

### Negative

- Graph rebuild is O(N²·D): too slow for N > 5,000 without sparse adjacency.
- The dense adjacency matrix uses N² × 4 bytes: 4 MB at N=1,000, 400 MB at
  N=10,000.  Needs CSR adjacency for larger stores.
- Weighted-degree is a heuristic; it is not guaranteed to find the true
  minimum cut.

### Neutral

- The API is sync-only; async wrappers are straightforward but not included.

## Alternatives Considered

### 1. Use `ruvector-mincut` exact algorithm

The existing `ruvector-mincut` crate provides exact dynamic minimum cut with
O(n^{o(1)}) amortised update time.  However, it operates on abstract edge
streams and is not designed for batch compaction on a dense adjacency matrix.
Integration is planned (ADR-196 §Implementation Plan step 3) but was deferred
to keep this crate self-contained and independently buildable.

### 2. Forgetting curves (Ebbinghaus decay)

Assign each entry a forgetting score based on time since last access.  Evict
entries with highest forgetting score.  This is well-studied (MemoryBank,
Zhong et al. 2023) but ignores graph coherence — it can evict an entry that
is semantically central simply because it has not been recently queried.

### 3. LLM-summarisation

Compress memory by calling an LLM to summarise and replace.  Effective but
requires network access, is non-deterministic, and is far too slow for
real-time compaction.  Incompatible with edge-first deployment.

### 4. Random eviction

Evict uniformly at random.  Extremely fast, but provides no semantic
guarantee.  Adding a `RandomEvict` strategy as a falsification baseline is
planned but not yet implemented.

### 5. Hierarchical clustering (K-means)

Run K-means on the current entries, identify the smallest cluster, evict it.
More principled than weighted degree but requires K-means convergence (O(N·K·D
per iteration) and non-deterministic cluster assignment.  Considered for future
work.

## Implementation Plan

1. **Now:** Merge `crates/ruvector-mincut-memory` with AgeEvict, CoherenceEvict,
   MinCutEvict as-is.  API is stable.

2. **Next:** Add `RandomEvict` as falsification baseline; add access-count
   weighting to CoherenceEvict and MinCutEvict; add sparse CSR adjacency for
   N > 5,000.

3. **Next:** Integrate `ruvector-mincut` exact algorithm as `ExactMinCutEvict`
   for N ≤ 100 where exact guarantees matter.

4. **Next:** Add WASM build target following `ruvector-rabitq-wasm` pattern.

5. **Later:** Add MCP tool surface in `mcp-gate`: `memory_compact` tool
   accepting `(strategy, target_size)` and returning `CompactionResult` JSON.

6. **Later:** ruFlo integration — workflow action that triggers compaction
   when `store.len() > capacity_threshold`.

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-mincut-memory`.
Hardware: x86-64 Linux 6.18, Intel Celeron N4020.
Rust: `rustc 1.94.1 (e408947bf 2026-03-25)`.

**N=500, D=32, 6 clusters, K=10, 50% compaction:**

| Strategy | Recall_b | Recall_a | Mean µs | Edges_b | Edges_a | Accept |
|---|---|---|---|---|---|---|
| AgeEvict | 1.000 | 1.000 | 6 340 | 7 652 | 1 955 | PASS |
| CoherenceEvict | 1.000 | 0.980 | 6 807 | 7 652 | 3 114 | PASS |
| MinCutEvict | 1.000 | 1.000 | 6 562 | 7 652 | 3 629 | PASS |

**N=1000, D=64, 8 clusters, K=10, 50% compaction:**

| Strategy | Recall_b | Recall_a | Mean µs | Edges_b | Edges_a | Accept |
|---|---|---|---|---|---|---|
| AgeEvict | 1.000 | 1.000 | 51 859 | 2 997 | 759 | PASS |
| CoherenceEvict | 1.000 | 1.000 | 53 392 | 2 997 | 1 420 | PASS |
| MinCutEvict | 1.000 | 1.000 | 53 056 | 2 997 | 2 026 | PASS |

Acceptance floor: `recall_after / recall_before >= 0.60`.

## Failure Modes

| Mode | Trigger | Mitigation |
|---|---|---|
| All vectors in one cluster | Uniform distribution; no graph structure | Fall back to AgeEvict |
| Threshold too high | No edges form; all degrees = 0 | Auto-tune to ~5% density |
| Graph rebuild too slow | N > 5,000 on embedded hardware | Switch to sparse CSR adjacency |
| All relevant items evicted | Aggressive compaction target | Increase target_size; acceptance test catches |
| NaN similarity | Near-zero vector | Guard: if norm < 1e-9, return 0.0 (implemented) |

## Security Considerations

- No network I/O; no credential handling.
- No file system access in the library; the benchmark binary writes only to stdout.
- Deterministic for a given seed — compaction decisions are auditable.
- Future: MCP tool surface must validate `target_size` (minimum floor, no
  evict-all) and authenticate the caller in multi-tenant deployments.
- Future: `ruvector-verified` witness log integration enables regulatory
  auditability of compaction decisions.

## Migration Path

`ruvector-mincut-memory` is a new, additive crate.  No existing crate is
modified.  Adoption path:

1. Add `ruvector-mincut-memory` as a dependency in agent memory code.
2. Replace manual `store.delete(oldest_ids)` with
   `MinCutEvict.compact(&mut store, target)`.
3. Capture `CompactionResult` for logging.
4. (Optional) Wire to ruFlo for automated scheduling.
5. (Optional) Add MCP tool wrapper for agent-driven compaction.

## Open Questions

1. Does `RandomEvict` match MinCutEvict recall at 50% compaction on clustered
   data?  (Answer would validate or falsify the graph-cut approach.)
2. What compaction ratio triggers measurable recall degradation for MinCutEvict?
   (Empirical threshold needed for production configuration guidance.)
3. Should the similarity threshold be a constructor parameter or a runtime
   parameter?  Current design: constructor parameter (`MemoryStore::new(dims, threshold)`).
4. Should `Entry.access_count` be weighted in MinCutEvict scoring?  Early
   hypothesis: yes, with tunable coefficient.
5. What is the correct benchmark for the MCP latency budget?  Agent tool calls
   should complete in < 500 ms; current 53 ms is comfortably inside this budget
   at N=1,000.
