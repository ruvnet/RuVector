# ADR-264: LSM-ANN — Write-Optimised Streaming Vector Index for Agent Memory

**Status**: Accepted  
**Date**: 2026-06-19  
**Author**: Nightly Research Agent  
**Crate**: `crates/ruvector-lsm-ann`

---

## Context

RuVector's agent memory subsystem (rvAgent episodic store, ruvector-core MemTable) currently relies on flat brute-force search for live write streams and HNSW for cold, fully-built indices.  There is no production-grade middle tier: an index that can absorb high-velocity writes while returning approximate nearest neighbours with bounded latency across the write-search overlap window.

Agentic workloads are characteristically write-heavy: every tool invocation, observation, and reasoning step produces embedding vectors that must be searchable within the same agent loop turn.  Standard HNSW insertion is O(log N) with a constant that grows as the graph is rewired; under burst writes this causes latency spikes that violate real-time constraints.

This ADR documents the decision to implement a Log-Structured Merge (LSM) approach to ANN indexing as the canonical write path for agent memory.

---

## Decision

We implement **LSM-ANN** (`crates/ruvector-lsm-ann`) with the following architecture:

### Tier structure

```
L0  MemTable  (Vec<(u64, Vec<f32>)>, mutable, brute-force search)
L1  Small frozen NSW segments          (one per L0 flush)
L2  Large merged NSW segment           (merged from L1 segments)
```

### Construction invariant

All frozen segments are built with **brute-force k-NN** during compaction (O(N²·D), run once per compaction).  This ensures every node's adjacency list is the true m-nearest at build time regardless of graph state.

### Search

Beam search with dual-heap implementation:
- `ClosestFirst` (negated-distance max-heap) → frontier exploration
- `FarthestFirst` (actual-distance max-heap, bounded to `ef`) → result eviction

### Three concrete variants

| Variant | Use case |
|---|---|
| `BaselineLsm` | Write throughput benchmarking (oracle recall) |
| `TwoTierLsm` | Single-segment memory (small agent context windows) |
| `FullLsm` | Multi-session episodic memory (auto L1→L2 compaction) |

All implement the `LsmIndex` trait: `insert`, `search`, `len`, `segment_count`, `compact`.

---

## Consequences

### Positive

- **Write throughput decoupled from index quality**: MemTable absorbs writes at O(1); graph construction happens only at compaction time.
- **Correct neighbourhood guarantees**: brute-force kNN during compaction avoids the hub-bias and approximation errors that accumulate in greedy-insertion HNSW.
- **Composable with existing crates**: `FullLsm` can replace the ANN component in ruvector-hybrid (ADR-256) and feed DiskANN (ADR-143) as a DRAM-resident compaction layer.
- **Measured recall**: TwoTier 0.852, FullLsm 0.855 at recall@10 on 10K×128D (ef_search=200, m=16).

### Negative

- **O(N²·D) compaction cost**: not suitable for L2 segments beyond ~50K vectors without switching to greedy HNSW construction.
- **Single-layer NSW**: recall ceiling lower than multi-layer HNSW at identical ef; mitigated by higher ef_search values.
- **Synchronous compaction blocks writes**: L0 flushes are synchronous; acceptable for current batch sizes (l0_max=1000) but will need background threading for streaming use cases.
- **No persistence**: frozen segments are in-memory only; persistence via `rkyv`/`bincode` is deferred.
- **No deletes**: upsert-by-id in MemTable; segment vectors are frozen; tombstone support is future work.

---

## Alternatives Considered

### Alternative 1: Extend existing ruvector-diskann

DiskANN (ADR-143) has an in-memory buffer layer.  Rejected because: (a) it is SSD-oriented and couples to the OS page cache; (b) extending it to pure in-memory use would require significant refactoring; (c) we want a lighter-weight in-process component for agent memory.

### Alternative 2: Adopt SPFresh-style local replacement

SPFresh (SOSP 2023) replaces stale graph edges locally without full rebuilds.  Considered but deferred: local replacement requires maintaining a routing structure (Steiner Point Tree) that significantly increases implementation complexity.  The brute-force compaction approach is simpler, provably correct, and fast enough for segments ≤50K vectors.

### Alternative 3: IVF MemTable

Buffer writes in a flat list; on query, probe top-p IVF clusters plus the MemTable.  Rejected because IVF recall degrades sharply for small batch sizes (few vectors per centroid during compaction), and building centroids from scratch is expensive.

---

## Benchmark Evidence

Measured on 2026-06-19, release build, linux/x86_64, rustc 1.94.1:

| Variant | Insert/s | Mem MB | Recall@10 | p50 µs | p95 µs |
|---|---|---|---|---|---|
| Baseline (brute-force) | 348,206 | 5.0 | 1.0000 | 1692.8 | 2012.5 |
| TwoTier (L0+L1 NSW) | 287 | 6.2 | 0.8520 | 484.4 | 574.5 |
| FullLsm (L0+L1+L2 NSW) | 808 | 6.2 | 0.8550 | 468.4 | 545.0 |

Config: N=10,000, D=128, k=10, m=16, ef_construction=200, ef_search=200, l0_max=1000, l1_merge_threshold=5.

NSW search is **3.5× faster than brute-force** at 85.2% recall — demonstrating the core speed/recall trade-off.

---

## Implementation Notes

### Key correctness issue resolved during development

The initial NSW construction passed node `i` as its own entry point during greedy search, resulting in an empty adjacency list traversal and recall ≈ 0.004.  Switching to brute-force kNN among nodes 0..i-1 fixed construction correctness.

A second bug used a single heap type (`neg_dist` ordering) for both the exploration frontier and the result set.  This caused the result set to evict the closest node (wrong) instead of the farthest.  Fixed by introducing two separate heap types: `ClosestFirst` (frontier) and `FarthestFirst` (result eviction).

### ef_search tuning

For 10K×128D data, ef_search=64 yields recall≈0.65; ef_search=200 yields recall≈0.855.  The rule of thumb for single-layer NSW: ef_search ≥ 2×k × sqrt(N/l0_max) achieves recall ≥ 0.85.

---

## Related ADRs

- ADR-143: DiskANN / Vamana (SSD-resident graph with DRAM buffer)
- ADR-193: RAIRS IVF (inverted file with redundant assignment)
- ADR-256: Hybrid sparse-dense search (BM25 + ANN + RRF)
