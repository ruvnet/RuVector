# LSM-ANN: Write-Optimised Streaming Vector Index for Agent Memory

**Nightly research session** — 2026-06-19  
**ADR** — [ADR-264](../../adr/ADR-264-lsm-ann.md)  
**Crate** — `crates/ruvector-lsm-ann`

---

## Abstract

Modern agentic systems produce continuous high-velocity write streams (tool call embeddings, observation vectors, episodic memory traces) that standard HNSW indices cannot absorb without either stalling the write path or degrading index quality.  We adapt the classical Log-Structured Merge (LSM) tree design to approximate nearest-neighbour search, yielding LSM-ANN: a tiered index that buffers writes in an in-memory MemTable, compacts them into small frozen NSW graph segments (L1), and merges those periodically into a large NSW segment (L2).

This report documents the SOTA landscape, our architecture, three implemented variants, and independently measured benchmark results on 10K × 128D data.

---

## Motivation

| Pain point | Root cause | LSM-ANN fix |
|---|---|---|
| HNSW write stalls | Locks held during graph rewiring | Write to lock-free MemTable first |
| 10× WAL amplification (pgvector) | In-place node updates | Immutable frozen segments; MemTable WAL only |
| Recall collapse under burst writes | Graph fragmentation | Brute-force kNN during compaction rebuilds exact neighbourhood |
| No incremental compaction knob | Monolithic index rebuild | Three separate tiers with independent compaction triggers |

---

## SOTA Survey

### Key papers (2021–2025)

| Paper | Venue | Insight |
|---|---|---|
| FreshDiskANN (arXiv 2105.09613) | 2021 | Single-writer SSD graph + in-memory buffer for freshness without full rebuild |
| SPFresh (SOSP 2023) | SOSP 2023 | Local-replacement compaction — incremental graph update via SPT routing |
| CleANN (arXiv 2507.19802) | 2025 | Clean insertions in DiskANN without backpropagation via candidate caching |
| IP-DiskANN (arXiv 2502.13826) | 2025 | In-place updates to DiskANN eliminating full rebuild cycles |
| Quake (OSDI 2025) | OSDI 2025 | Partition-based ANN with incremental IVF maintenance under Pareto constraint |
| Ada-IVF (arXiv 2411.00970) | 2024 | Adaptive IVF list sizing under changing distributions |
| DGAI (arXiv 2510.25401) | 2025 | Dynamic graph ANN for streaming inserts with lazy compaction |
| Wolverine (VLDB 2025) | VLDB 2025 | Version-based graph snapshots for MVCC-style ANN |

### Competitor write architectures

| System | Write design | Weakness |
|---|---|---|
| Qdrant | WAL + growing/sealed segments | Segment sealing latency spikes |
| LanceDB | Append-only fragments + reindex | Slow query across many fragments |
| TurboPuffer | LSM on S3 + SPFresh compaction | S3 latency unsuitable for sub-ms recall |
| Chroma | wal3 + async compaction | Non-deterministic compaction timing |
| pgvector | In-place HNSW, 10× WAL amplification | Write throughput bounded by index rewiring |
| Milvus | Sealed/growing segment model | JVM GC pauses on segment promotion |

### Gap our work fills

All production systems above conflate the write buffer and the graph structure in the same data structure, creating coupling between write throughput and index quality.  LSM-ANN decouples them cleanly: the MemTable absorbs all writes at O(1) cost; compaction rebuilds graph structure from scratch using O(N²·D) brute-force kNN that guarantees correct neighbourhood regardless of prior graph state.

---

## Architecture

```
Writes ──► L0 MemTable (mutable, brute-force search)
               │ flush when |L0| ≥ l0_max
               ▼
           L1 Small frozen NSW segments  (one per flush)
               │ merge when |L1 segs| ≥ l1_merge_threshold
               ▼
           L2 Large merged NSW segment
```

### NSW graph construction (FrozenSegment::build)

During construction we use **brute-force kNN** among the already-inserted prefix of nodes (nodes 0..i-1) to find each new node's m nearest neighbours.  This is O(N²·D) but run exactly once per compaction event.  The guarantee: every node's adjacency list is the true m-nearest at insertion time, not an approximation from a partially-built graph.

Bidirectional edges are added with pruning: if a neighbour's adjacency list exceeds m after adding the reverse edge, we re-sort by distance and truncate.

### Beam search (greedy_search_internal)

Two heaps are used simultaneously:

- **`frontier: BinaryHeap<ClosestFirst>`** — min-heap by distance (negated to use Rust's max-heap).  Pops the closest unexplored node next.
- **`best: BinaryHeap<FarthestFirst>`** — max-heap by actual distance.  Bounded to `ef` entries; evicts the farthest when full.

Termination: when the closest node on the frontier is farther than the worst node in `best`, and `best` is full, no further exploration can improve results.

### Three variants

| Variant | L0 | L1 | L2 | Auto-compact |
|---|---|---|---|---|
| `BaselineLsm` | ✓ (brute-force only) | — | — | — |
| `TwoTierLsm` | ✓ | ✓ (one segment) | — | Yes (on insert when L0 full) |
| `FullLsm` | ✓ | ✓ (up to threshold) | ✓ | Yes (L0→L1 on flush; L1→L2 on threshold) |

---

## Benchmark Results

Measured on 2026-06-19, release build (`opt-level=3, lto=fat`).  
**Environment**: linux / x86_64 / rustc 1.94.1  
**Dataset**: N=10,000, D=128, Gaussian N(0,1), seed=42  
**Queries**: 100, k=10  
**Config**: m=16, ef_construction=200, ef_search=200, l0_max=1000, l1_merge_threshold=5

```
───────────────────────────────────────────────────────────────────
 Variant │ Insert/s │ Mem (MB) │ Recall@10  │ p50 µs │ p95 µs │ Pass
───────────────────────────────────────────────────────────────────
 Baseline │  348,206/s │   5.0  │  1.0000   │ 1692.8 │ 2012.5 │ PASS
 TwoTier  │      287/s │   6.2  │  0.8520   │  484.4 │  574.5 │ PASS
 FullLsm  │      808/s │   6.2  │  0.8550   │  468.4 │  545.0 │ PASS
───────────────────────────────────────────────────────────────────
```

### Key observations

**Insert throughput**: Baseline achieves 348K/s (pure MemTable, no graph work).  TwoTier shows 287/s because every insert that triggers a compaction pays O(N²·D) brute-force kNN synchronously.  FullLsm appears faster (808/s) because its smaller L0 window (1K vectors vs the growing TwoTier merged segment) keeps compaction cost lower per event.

**Recall**: Both NSW variants reach ≥0.85 with ef_search=200 on this dataset.  The ~15% miss rate is the expected cost of approximate graph traversal in a single-layer NSW vs full HNSW.  L2 merge (FullLsm) gives a slight recall advantage (+0.003) over TwoTier because a larger, denser graph has better connectivity.

**Search latency**: NSW search (484–574µs p50/p95 for TwoTier) is 3.5× faster than brute-force (1692µs p50) at 85% recall — demonstrating the core trade-off.

**Memory**: Both NSW variants add ~1.2 MB overhead vs the raw vector bytes (graph adjacency lists with m=16, bidirectional).

### Acceptance criteria

| Variant | Threshold | Actual | Result |
|---|---|---|---|
| Baseline recall@10 | ≥ 0.999 | 1.0000 | **PASS** |
| TwoTier recall@10 | ≥ 0.850 | 0.8520 | **PASS** |
| FullLsm recall@10 | ≥ 0.850 | 0.8550 | **PASS** |

---

## Unit Tests

8 tests, all passing:

| Test | Covers |
|---|---|
| `test_baseline_len` | 200-vec insert count |
| `test_twotier_len` | len across L0+L1 after compact |
| `test_fulllsm_len` | len across L0+L1+L2 after compact |
| `test_baseline_perfect_recall` | brute-force oracle = 1.0 |
| `test_twotier_recall_threshold` | NSW recall ≥ 0.70 on 500×32 |
| `test_fulllsm_recall_threshold` | NSW recall ≥ 0.70 on 500×32 |
| `test_merge_dedup` | de-dup by id, keep min dist, top-k |
| `test_recall_perfect` | recall_at_k = 1.0 for identical sets |

---

## Known Limitations and Future Work

1. **O(N²·D) compaction** — acceptable for segments ≤50K vectors; for larger L2 segments, a greedy HNSW-style construction (O(N·log(N)·D)) should replace the brute-force pass.

2. **Single-layer NSW** — recall ceiling is lower than multi-layer HNSW at identical ef and m.  Upgrading L2 to HNSW (with a skip hierarchy) is the natural next step.

3. **Synchronous compaction** — L0 flushes block the write path.  Background compaction threads (as in LevelDB) would decouple write throughput from compaction cost entirely.

4. **No deletions** — the current MemTable `insert` is upsert-by-id; segment vectors are frozen.  Tombstone propagation and segment rebuilds are needed for a production delete path.

5. **No persistence** — all tiers are in-memory.  Serialising frozen segments via `rkyv` or `bincode` (already workspace dependencies) would complete the storage story.

6. **SIMD distance kernel** — `sq_dist` is scalar; `simsimd` (workspace dependency) can provide 4–8× speedup on AVX2/AVX-512 platforms.

---

## Connections to RuVector Ecosystem

- **ruvector-diskann** (ADR-143): DiskANN operates on SSD-resident graph with a DRAM buffer — closely related L0 concept; LSM-ANN provides the in-memory compaction layer that could feed DiskANN's beam graph builder.
- **ruvector-hybrid** (ADR-256): BM25+ANN hybrid retrieval needs an updateable ANN component; LSM-ANN's TwoTierLsm fits as the ANN side.
- **ruvector-core** `MemTable` pattern: the rvAgent framework's episodic memory would benefit from LSM-ANN for dense vector recall alongside sparse keyword search.
- **ruvector-rairs** (ADR-193): RAIRS IVF computes inverted file indices; LSM-ANN's tiered approach could replace the IVF rebuild step with incremental L1→L2 merges.
