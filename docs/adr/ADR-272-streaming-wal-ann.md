# ADR-272: Streaming Write-Ahead Log for ANN with Coherence-Gated Merge (SWAL-ANN)

- **Status**: Proposed (PoC benchmarked — nightly research 2026-07-14)
- **Date**: 2026-07-14
- **Crate**: `crates/ruvector-wal-ann`
- **Research doc**: `docs/research/nightly/2026-07-14-streaming-wal-ann/README.md`
- **Relates to**: ADR-224 (proof-gated writes), ADR-254 (coherence-gated HNSW search), ADR-264 (LSM-ANN), ADR-268 (capability-gated ANN)

---

## Context

All major vector databases (Milvus, LanceDB, Qdrant, Weaviate, Pinecone) handle streaming vector ingestion with a **size-only WAL flush policy**: buffer inserts until the buffer reaches N entries, then build or update the ANN index. No production system uses a graph quality signal to determine *when* to promote buffered vectors to the searchable index.

This creates two failure modes under adversarial or distribution-shifting workloads:

1. **Size-based over-flushing**: if the data distribution is stable and new vectors are close to existing graph nodes, frequent merges waste CPU without improving recall.
2. **Size-based under-flushing**: if a burst of isolated vectors arrives (new semantic cluster), the size gate may delay the merge, leaving queries for that cluster to fall back to slow WAL linear scan indefinitely.

RuVector's coherence scoring infrastructure (ADR-254) already measures how well-connected the graph is relative to an input set. SWAL-ANN applies this measurement to merge-time decisions.

---

## Decision

Introduce `ruvector-wal-ann`, a streaming vector index crate implementing a three-tier architecture:

1. **WAL tier** — `VectorWal`: bounded in-memory buffer, always searchable via brute-force linear scan. No inserted vector is ever invisible to queries.

2. **Coherence gate** — `MergeGate` trait with three implementations:
   - `EagerGate`: merge after every N inserts (small N). Approximates insert-per-query consistency.
   - `LazyGate`: merge after every N inserts (large N). Maximises throughput via infrequent large batches.
   - `CoherenceGate`: merge when `coherence < threshold OR wal_size >= max`. Quality-driven early firing.

3. **Main graph tier** — `NavGraph`: incrementally-insertable navigable small-world graph using beam-search neighbor selection during insert. Absorbs WAL in O(|WAL|·ef·M·D) batch merges.

The coherence score is defined as:

```
isolation(v, G) = min{ L2(v, g) | g in G }
coherence       = 1 / (1 + mean(isolation) for sampled WAL vectors)
```

Evaluated every 8 inserts using 16 WAL samples × 64 graph samples → O(1024·D) amortised.

---

## Consequences

### Positive

- All inserted vectors are immediately searchable (WAL linear scan tier).
- Merge cost is amortised across batches rather than paid per-insert.
- `CoherenceGate` provides quality-adaptive merging: fires early for isolated data, defers for redundant data.
- The `MergeGate` trait enables drop-in replacement of merge strategy (open/closed principle).
- Architecture is `no_std`-compatible (no OS threads, no async, no file I/O in core types).

### Negative / Trade-offs

- WAL linear scan cost scales linearly with WAL size. Must cap via `max_wal_size`.
- Incremental NSW recall (0.716 at 3K × 64-dim) is below batch HNSW (typically >0.90) because early nodes have fewer diverse neighbours. Recall improves with multi-layer HNSW backend.
- Coherence threshold requires calibration per data distribution. Miscalibration degrades to EagerGate or LazyGate behaviour.
- Current PoC is single-threaded; concurrent multi-writer access requires a Mutex or channel.

---

## Alternatives Considered

### In-place per-insert graph update (IP-DiskANN style)
Eliminates the WAL entirely; every insert immediately updates the graph with reverse-edge tracking. Avoids batch merge latency spikes but has higher per-insert cost (O(ef·M·D) per vector regardless of flush timing) and more complex deletion handling. Suitable for insert-heavy workloads without bursty patterns.

### LSM-tiered index (LSM-VEC / prior ADR-264)
Multi-level LSM structure where levels are ANN indexes. Compaction is triggered by level size ratio. Does not use a quality signal and compaction scheduling differs from WAL semantics. Better for disk-resident storage; SWAL-ANN is better for in-memory agent memory.

### Batch offline rebuild
Simplest: accumulate all inserts then rebuild the full index. Zero recall degradation. Unacceptable latency gap between insert and availability. Excluded.

### Size-only WAL (production baseline)
Current approach in Milvus/LanceDB/Qdrant. No quality gate. Reference implementation for comparison; CoherenceGate should behave at least as well for uniform distributions.

---

## Implementation Plan

### Phase 1 (PoC — done, 2026-07-14)
- [x] `VectorWal` with linear-scan search.
- [x] `MergeGate` trait + EagerGate, LazyGate, CoherenceGate.
- [x] `NavGraph` incremental NSW with 8-probe entry and long-jump edges.
- [x] `WalAnnIndex<G>` combining all three tiers.
- [x] Benchmark binary with three variants.
- [x] 15 unit tests, all passing.
- [x] Acceptance criteria: recall@10 ≥ 0.70, mean latency < 5ms. **All PASS.**

### Phase 2 (Production hardening)
- [ ] Replace single-layer NSW with multi-layer HNSW (`crates/ruvector-coherence-hnsw` backend) to reach recall ≥ 0.90.
- [ ] Durable WAL: mmap'd or append-only file, integrated with `crates/ruvector-snapshot`.
- [ ] Multi-writer access: MPSC channel for insert serialisation.
- [ ] Auto-calibrated coherence threshold: online distance distribution estimation.
- [ ] Deletion handling: soft-delete from WAL + tombstone in main graph.

### Phase 3 (Ecosystem integration)
- [ ] MCP tool surface (`ruvector-wal-ann-mcp`).
- [ ] ruFlo integration: coherence-monitor-driven flush scheduling.
- [ ] WASM target (`ruvector-wal-ann-wasm`).
- [ ] Benchmark against NeurIPS 2023 big-ANN streaming track (arXiv:2409.17424).

---

## Benchmark Evidence

**Command**: `cargo run --release -p ruvector-wal-ann --bin benchmark`  
**Dataset**: 3,000 × 64-dim f32 Normal(0,1)  
**Queries**: 100 × k=10  

| Variant | Merges | Insert(ms) | Vecs/sec | Recall@10 | Mean(µs) | p95(µs) | QPS |
|---------|--------|-----------|----------|-----------|---------|---------|-----|
| EagerMerge | 94 | 319.6 | 9,388 | 0.716 | 110.1 | 150.8 | 9,084 |
| LazyMerge | 6 | 316.7 | 9,472 | 0.716 | 113.5 | 150.6 | 8,811 |
| CoherenceGatedMerge | 12 | 354.7 | 8,458 | 0.716 | 105.5 | 132.8 | 9,477 |

All variants PASS acceptance criteria. Memory: 1,156 KB per index at 3K × 64-dim.

---

## Failure Modes

1. **WAL scan latency at scale**: a WAL of 4,096 × 768-dim (LLM embedding) vectors = 12 MB to scan per query. Fix: hard-cap max_wal_size, or use a WAL-level approximate index (e.g., small IVF).
2. **Coherence threshold miscalibration**: see Consequences. Fix: auto-calibration in Phase 2.
3. **Incremental graph quality**: early-node connectivity degrades over long insert sequences. Fix: multi-layer HNSW backend (Phase 2).
4. **Concurrent write races**: single-threaded PoC. Fix: MPSC channel (Phase 2).
5. **No crash recovery**: in-memory WAL only. Fix: durable WAL (Phase 2).

---

## Security Considerations

- WAL entries are unencrypted in the PoC. Production: symmetric encryption of WAL entries.
- Coherence threshold is public knowledge; adversaries can craft insert sequences that suppress or trigger merges. Fix: add calibrated noise to the threshold (differentially private gate).
- WAL flush timing is observable via latency spikes. Fix: background async flush with fixed-latency insert acknowledgement.
- ID monotonicity leaks insert ordering. Production: random IDs or encrypted monotonic IDs.

---

## Migration Path

- `MergeGate` is a trait; all three implementations are additive. No existing crates are affected.
- `NavGraph` is an independent implementation; does not modify `ruvector-coherence-hnsw` or other graph crates.
- Phase 2 will replace `NavGraph` with a proper HNSW backend behind the same `AnnIndex` trait (to be added).
- Feature flags: `wal-durable`, `wal-mcp`, `wal-wasm` to be gated behind Cargo features.

---

## Open Questions

1. Should the coherence gate threshold be a fixed float or a function of the current graph size (adaptive scaling)?
2. What is the right level of WAL scan granularity for WASM targets where brute-force 512×768-dim is too slow?
3. Can the coherence score be computed incrementally (O(D) per new WAL entry by maintaining a running min-distance estimate)?
4. Should WAL merge be triggered by ruFlo as a scheduled workflow action, rather than inline in `insert()`?
5. How does SWAL-ANN interact with ADR-224's proof-gated writes: should the proof gate apply to WAL entries or to the merge event itself?
