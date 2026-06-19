---
adr: 196
title: "LSM-Segmented Vector Index — Epoch-Based Three-Tier HNSW for Streaming Inserts"
status: proposed
date: 2026-06-05
authors: [ruvnet, claude-flow-nightly]
related: [ADR-193-rairs-ivf, ADR-195-ruvector-embedder-unification-plan]
tags: [ruvector, hnsw, lsm, streaming, vector-index, agent-memory, edge, wasm]
---

# ADR-196 — LSM-Segmented Vector Index

## Status

**Proposed.** Proof of concept implemented in `crates/ruvector-lsm-index`.
Benchmark results are real. Production integration requires follow-on work (see §9).

## Context

RuVector currently provides HNSW and DiskANN indexes via `ruvector-core` and
`ruvector-diskann`. Both require either (a) full batch construction before querying
or (b) online single-vector inserts into an existing HNSW graph. Neither handles
the streaming agent-memory workload well:

- **Batch construction**: forces a full O(n log n) rebuild whenever new agent memories
  arrive. Unacceptable for a ruFlo loop that writes every few seconds.
- **Online HNSW insert**: incremental inserts degrade graph quality over time because
  back-edges are limited and tombstoned deletes accumulate. Microsoft Research
  (IP-DiskANN, arXiv:2502.13826) documents recall degradation after 10–20% deletes.

The state of the art (June 2026) shows three convergent design directions for streaming
ANN:
1. **LSM + HNSW graph storage** — LSM-VEC (arXiv:2505.17152, VLDB 2026 candidate)
   maintains the HNSW neighbor graph across LSM levels to avoid global rebuilds.
2. **Balanced graph streaming** — UBISS (arXiv:2602.00563) continuously rebalances a
   proximity graph without batch-rebuild phases.
3. **In-place graph surgery** — IP-DiskANN (arXiv:2502.13826) reconnects deleted nodes'
   neighbors without rebuilding the full graph.

**RuVector's differentiated position:** none of these targets embedded, edge, or WASM
deployments. The Cognitum Seed appliance, `rvAgent` WASM modules, and ruFlo workflows
running on-device all need a streaming vector index that:
- Works without background threads (synchronous compaction)
- Fits in `no_std` environments
- Uses `<10 MB` total memory for typical agent context workloads
- Integrates with the RVF temperature-tiering spec

This ADR proposes `ruvector-lsm-index`: an epoch-driven, three-tier vector index where
vectors flow hot → warm → cold with synchronous in-process compaction.

## Decision

Introduce `crates/ruvector-lsm-index` as a standalone composable crate implementing
a three-tier LSM-style vector index:

```
hot  (FlatSegment)  — newest writes, O(1) insert, O(n_hot) linear scan
warm (NswSegment)   — recent epochs, NSW graph, O(log n_warm) search
cold (NswSegment)   — stable bulk, NSW graph, O(log n_cold) search
```

**Write path**: `insert → hot`. When `hot.len() ≥ hot_capacity`, flush hot → warm
(rebuild warm NSW). When `warm.len() ≥ warm_capacity`, flush warm → cold (rebuild
cold NSW). Compaction is synchronous — no background thread, no OS timer.

**Read path**: fan-out search to hot + warm + cold, merge results, deduplicate, return top-k.

**Compaction bounds**: rebuild cost is O(segment_size × ef_build × log segment_size),
bounded by tier capacity settings (not by total dataset size).

## Consequences

**Positive**
- O(1) amortised insert latency (hot is a flat append, flushes are batched)
- Search recall is additive — LSM-NSW achieved 62.7% recall vs 57.5% for single NSW
  in the PoC benchmark (multi-tier coverage finds additional candidates)
- Synchronous compaction enables `no_std` / WASM compatibility
- Segment-level compaction bounds rebuild cost regardless of total dataset size
- Natural integration with RVF hot/warm/cold temperature tiers

**Negative**
- Higher build cost due to multiple NSW rebuilds: 14.9s vs 2.3s for N=10K (6.5x)
- Single-layer NSW (no HNSW hierarchy) limits recall at high dimensions (128d: ~60%)
- Write amplification: each vector may participate in 2–3 NSW rebuilds over its lifetime
- Synchronous compaction can cause p99 latency spikes during flush events

**Neutral**
- Memory footprint is comparable to single HNSW: 6,783 KB vs 6,749 KB for N=10K, 128d

## Alternatives Considered

### A. In-Place HNSW Graph Surgery (IP-DiskANN style)
- Maintain a single HNSW graph with online inserts and delete reconnection.
- **Rejected for this ADR**: complex concurrent implementation; recall degrades after
  10–20% deletes; requires background consolidation thread (not `no_std` compatible).

### B. IVF Partition-Based Streaming (Ada-IVF / SPFresh style)
- Use IVF partitions as the streaming unit; adaptive centroid rebalancing.
- **Rejected**: IVF recall at low nprobe is inferior to NSW/HNSW; k-means training
  required (compute-intensive, unsuitable for edge); SPFresh targets billion-scale
  servers, not embedded/WASM.

### C. UBISS Balanced Graph Streaming
- Continuous in-place graph balance maintenance without explicit epochs.
- **Rejected**: complex background balancing process; no synchronous compaction path;
  not yet proven outside the research prototype.

### D. Full HNSW with Hierarchical Layers
- Implement proper multi-layer HNSW instead of single-layer NSW.
- **Not rejected, deferred**: would improve recall from ~60% to ~95%+ at same ef.
  Planned as a follow-on upgrade to the warm/cold segments in a future ADR.

## Implementation Plan

### Phase 0 (this ADR) — PoC
1. `crates/ruvector-lsm-index` with FlatSegment, NswSegment, LsmVectorIndex.
2. 10 unit tests passing.
3. Benchmark binary with 3 variants on N=10K, dim=128.
4. Workspace member added.

### Phase 1 — Production hardening
1. Replace NswSegment with full HNSW (hierarchical layers) from `ruvector-core`.
2. Add per-segment quantization codebook (int8 warm, binary cold).
3. Implement tombstone-aware delete propagation through flush.
4. Add `Arc<RwLock<>>` concurrent read path for multi-threaded ruFlo loops.
5. Export `#[no_std]` compatible flat + warm tiers for WASM.

### Phase 2 — RuVector integration
1. Plug `LsmVectorIndex` as an alternative backend in `ruvector-core::VectorIndex`.
2. Wire into `ruvector-delta-index` as the segment manager.
3. Add RVF serialisation for cold segments (pack sealed cold tier into an RVF blob).
4. MCP tool surface: `memory_insert`, `memory_search`, `memory_stats` as ruFlo tools.

## Benchmark Evidence

Measured on 2026-06-05. Hardware: x86_64 Linux (cloud VM). Release build.
Dataset: 10,000 vectors × 128 dims. Queries: 1,000. k=10.

| Variant     | Build(ms) | mean(ms) | p50(ms) | p95(ms) | Throughput(q/s) | Mem(KB) | Recall@10 |
|-------------|-----------|----------|---------|---------|-----------------|---------|-----------|
| Flat (base) | 2.6       | 1.829    | 1.813   | 1.962   | 547             | 5,078   | 1.000     |
| NSW         | 2,338     | 1.052    | 1.044   | 1.145   | 950             | 6,749   | 0.575     |
| LSM-NSW     | 14,902    | 1.323    | 1.312   | 1.432   | 756             | 6,783   | 0.627     |

Hot insert throughput: mean=0.56ms, p50=0.0001ms (pure hot path), p95=0.0015ms.

NSW ef_build=40, ef_search=160 (4×). LSM-NSW ef_build=40, ef_search=120 (3×).
Single-layer NSW (no HNSW hierarchy). See §Open Questions for recall improvement path.

**Key result**: LSM-NSW achieves *higher* recall than single NSW (0.627 vs 0.575)
because fan-out across three tiers covers more candidates than a single-tier search.
Trade-off: 1.26x higher query latency than single NSW.

## Failure Modes

1. **p99 flush spikes**: synchronous compaction during hot→warm flush blocks inserts.
   Detection: record flush_duration per compaction event in LsmStats. Mitigation: cap
   warm_capacity to limit flush cost; future Phase 1 can move to async compaction.

2. **Recall collapse after many flushes**: NSW graph quality degrades with incremental
   rebuild of warm segment. Each hot→warm flush absorbs new vectors into the warm NSW
   by calling build_from(warm + hot). This is batch-build, not incremental, so quality
   should be stable. Monitored by the test `lsm_recall_at_least_60_pct`.

3. **Memory spike during compaction**: during cold flush, both the old cold segment and
   the new merged cold segment coexist momentarily (2× cold memory). Max spike is
   bounded by 2 × cold_capacity × (8 + dims × 4 + M × 8) bytes.

4. **WASM incompatibility**: `Vec<Vec<f32>>` causes many small allocations; WASM
   allocators (wee_alloc, dlmalloc) may fragment. Mitigation: use flat `Vec<f32>` with
   stride indexing for the WASM target (Phase 1).

## Security Considerations

- No network I/O, no file I/O, no external service dependencies.
- Input validation: vector dimensions must match `LsmConfig::dims` (currently not
  enforced; will panic on dimension mismatch in `l2sq`). Phase 1 must add explicit
  dimension check with `Result` return.
- No secret material stored in the index.

## Migration Path

No existing RuVector users are affected. `ruvector-lsm-index` is a new standalone crate.
When Phase 2 integration lands, the existing `VectorIndex` trait is unchanged — LSM
is an optional backend selected via feature flag.

## Open Questions

1. **Hierarchical layers**: The single-layer NSW limits recall. What is the minimal
   HNSW hierarchy (2 layers) that fits in `<50 lines` additional code and raises recall
   to 85%+ on 128d data? This is the most important quality improvement.

2. **ef_search vs ef_build trade-off**: The benchmark uses ef_search=4×ef_build.
   Is this the right ratio? Should ef_search be a per-query parameter exposed via the
   MCP tool surface?

3. **Segment merging strategy**: Current strategy is "absorb hot into warm" (rebuild
   warm with all warm+hot vectors). LSM-VEC uses level-based merge (like RocksDB
   levelled compaction). Should warm have multiple sub-segments at the same tier?

4. **Delete propagation**: Tombstones in hot must propagate to warm/cold during flush.
   Current implementation has no delete support. Phase 1 critical path item.

5. **Concurrent read/write**: Current implementation is not thread-safe (no locking).
   RuFlo loops may want concurrent query + insert. Phase 1: add `parking_lot::RwLock`.
