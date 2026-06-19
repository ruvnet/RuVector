---
adr: 258
title: "ruvector-hnsw-repair — Pluggable Deletion Strategies for HNSW Graphs"
status: accepted
date: 2026-06-18
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-200, ADR-254]
supersedes: []
tags: [hnsw, ann, deletion, graph-repair, agent-memory, recall, vector-database, rust]
---

# ADR-258 — ruvector-hnsw-repair: Pluggable HNSW Deletion Strategies

## Status

**Accepted (PoC implemented).** `crates/ruvector-hnsw-repair` ships three measurable deletion strategies. Production integration into `ruvector-core` and reverse-adjacency index optimisation are staged as follow-on work.

---

## Context

### The Problem

HNSW is the dominant in-memory ANN structure in RuVector. The current `ruvector-core` HNSW wraps `hnsw_rs` and handles deletion by tombstoning the node and skipping it at search time. The graph structure is never modified.

This is adequate for append-mostly workloads, but agent memory stores are not append-mostly. An agent session that runs for hours continuously evicts old context, code snippets, or tool outputs. With 20% deletion, tombstone-only costs 1.9 pp recall@10 (measured). With higher deletion fractions or non-uniform deletion patterns, recall degradation can be much worse.

No published HNSW library provides a configurable deletion strategy that lets the caller choose between performance and recall recovery.

### The Gap

- `hnsw_rs` (used by ruvector-core): tombstone only.
- `hnswlib`: tombstone only.
- `usearch`: tombstone + experimental eviction (algorithm not published).
- `pgvector`: tombstone only.

None expose a `DeletionStrategy` trait that lets callers choose based on their workload.

---

## Decision

Introduce `crates/ruvector-hnsw-repair` as a research crate with:

1. A self-contained HNSW implementation (no external HNSW library) to enable full graph access.
2. A `DeletionStrategy` trait.
3. Three concrete strategies: `TombstoneOnly`, `BatchRepair`, `EagerRepair`.
4. Real benchmark numbers (5,000 × 64-dim, 100 queries, 20% deletion).

The `DeletionStrategy` trait and its three implementations are the API that should survive into production. The self-contained HNSW in this crate is a research vehicle; production would use `ruvector-core`'s existing HNSW with the trait injected.

---

## Consequences

**Positive:**
- Agent memory workloads can choose their deletion strategy at construction time.
- BatchRepair provides a ruFlo automation hook: flush the repair queue when tombstone fraction exceeds a threshold.
- EagerRepair provides GDPR-grade removal: no inbound edges remain after deletion.
- The `DeletionStrategy` trait is simple and composable with `ruvector-proof-gate`.

**Negative:**
- EagerRepair is O(N × degree) per deletion — unusable above ~50K vectors without the reverse-adjacency index optimisation.
- Adding a reverse-adjacency index doubles edge memory footprint.
- The PoC HNSW does not support persistence, quantization, or SIMD distance — it is a research vehicle only.

**Neutral:**
- TombstoneOnly (the status quo) is a valid, first-class strategy. This ADR does not deprecate it.

---

## Alternatives Considered

### A: Full Rebuild on Eviction

**Rejected.** Full rebuild is O(N log N) and causes a query blackout window. Unacceptable for continuously-running agent memory stores.

### B: LSM-Style Compaction (delete from segment, merge segments)

**Rejected for this ADR.** LSM compaction is the right long-term approach for disk-backed indexes (see `ruvector-diskann`). For in-memory HNSW, it adds unnecessary complexity and latency spikes.

### C: Mark-and-Rebuild with Background Thread

**Deferred.** Running a background rebuild thread while serving queries requires concurrent data structures and careful synchronisation. Worthwhile for production but out of scope for this research PoC.

### D: Adopt usearch's Deletion Algorithm

**Rejected.** usearch's deletion algorithm is not published. We cannot adopt an unpublished algorithm without verification.

---

## Implementation Plan

**M1 (this ADR):** `crates/ruvector-hnsw-repair` with self-contained HNSW, `DeletionStrategy` trait, three strategies, benchmark binary. **Done.**

**M2:** Add reverse-adjacency index to reduce EagerRepair from O(N) to O(degree) per deletion. Target: deletion under 1 µs for 1M vector graph.

**M3:** Integrate `DeletionStrategy` into `ruvector-core::HnswIndex`. `HnswIndex::new()` accepts an optional `Box<dyn DeletionStrategy>`.

**M4:** ruFlo workflow trigger: `tombstone_fraction` metric exposed via MCP; `flush_repair_queue` MCP tool.

**M5:** Entry-point election: detect entry-point deletion and elect replacement automatically.

---

## Benchmark Evidence

Measured on x86_64 Linux, release build, 5,000 × 64-dim uniform random vectors, 100 queries, 20% deletion (1,000 nodes).

| Variant | Delete (ms) | Search mean µs | p50 µs | p95 µs | Recall@10 | Δ baseline |
|---------|------------|----------------|--------|--------|-----------|-----------|
| Baseline | — | — | — | — | 0.9140 | — |
| TombstoneOnly | 0.00 | 213.5 | 208.2 | 241.5 | 0.8950 | −0.0190 |
| BatchRepair(50) | 81.69 | 231.7 | 229.1 | 259.3 | 0.9040 | −0.0100 |
| EagerRepair | 83.02 | 230.1 | 228.2 | 252.3 | 0.9040 | −0.0100 |

Acceptance: best recall ≥ 75% of baseline. **PASS** (best = 0.9040 ≥ 0.6855).

---

## Failure Modes

1. **Entry-point deletion** (unhandled in M1): if the global entry point is deleted, `find_live_entry()` falls back to O(N) linear scan. Must be fixed before production use.

2. **Clustered deletion**: deleting a tight spatial cluster creates a graph hole larger than any single-node repair can fix. BatchRepair should run a re-insertion of affected nodes' neighbours in this case.

3. **O(N) repair on large graphs**: EagerRepair's O(N) scan makes it impractical above ~50K vectors. The reverse-adjacency index (M2) is the required mitigation.

4. **Concurrent access**: the PoC is single-threaded. Production HNSW requires `Arc<RwLock<>>` or lockfree structures around the layers array during repair.

---

## Security Considerations

- **GDPR erasure**: TombstoneOnly does NOT guarantee vector data is unreachable — the backing store still holds the vector. EagerRepair removes all inbound edge references; combining with `graph.vectors[id].fill(0.0)` provides physical erasure.
- **Proof-gated deletion**: deletion should integrate with `ruvector-proof-gate` so that repair only executes after the deletion is committed to the witness log.
- **Bulk-delete DoS**: an agent with write access could trigger O(N × degree) repair by deleting many nodes. BatchRepair with configurable batch_size is the rate-limiting mechanism.
- **No unsafe code**: the crate uses no `unsafe` blocks.

---

## Migration Path

1. Existing code using `ruvector-core::HnswIndex` is unaffected (M1 is a new crate).
2. When M3 lands, `HnswIndex::new()` gains an optional strategy parameter; the default is `TombstoneOnly` to preserve existing behaviour.
3. Agent memory workloads can opt into `EagerRepair` or `BatchRepair` by passing the strategy at construction.
4. The `ruvector-hnsw-repair` crate remains a standalone research and testing vehicle.

---

## Open Questions

1. What is the minimum `ef_search` multiplier needed to maintain recall@k ≥ 95% of baseline under d% tombstone fraction, for the three strategies? (Empirical study needed.)
2. Can the BatchRepair flush be made non-blocking (run in a background task) without correctness issues?
3. Should EagerRepair remove all inbound edges or only at levels where the deleted node was a *critical* bridge (no alternative path)?
4. What is the right batch_size heuristic for BatchRepair? (Currently user-provided; could be auto-tuned from observed deletion rate.)
