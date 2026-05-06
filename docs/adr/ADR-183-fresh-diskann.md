# ADR-183: FreshDiskANN — Streaming Online Index Maintenance

**Status:** Accepted  
**Date:** 2026-05-06  
**Branch:** research/nightly/2026-05-06-fresh-diskann  
**Crate:** `crates/ruvector-fresh-diskann`

---

## Context

The existing `ruvector-diskann` crate implements the static Vamana / DiskANN algorithm:
all vectors must be loaded before `build()` is called, and any new vector after that
point requires a full graph rebuild.  Production deployments that continuously receive
new embeddings (RAG pipelines, recommendation systems, real-time semantic search) cannot
tolerate full rebuilds — even on 100k-vector indices the rebuild cost can exceed tens
of seconds (observed: 28.8 s for 10k × 128-dim on a 4-core Xeon @ 2.80 GHz).

FreshDiskANN (Jayaram Subramanya et al., arXiv:2105.09613, VLDB 2022) solves this by
introducing two orthogonal mechanisms:

1. **In-memory buffer** — new vectors land in a buffer and are immediately searchable
   via brute-force scan while the graph is undisturbed.
2. **Lazy consolidation** — when the buffer reaches threshold *T*, each buffered vector
   is beam-inserted into the Vamana graph using the same α-robust pruning used at build
   time, with backlink repair to maintain the out-degree bound *R*.  No full rebuild.

---

## Decision

Add a new standalone crate `ruvector-fresh-diskann` implementing:

* **`FreshDiskAnn`** — the streaming-capable index struct with configurable
  `ConsolidationPolicy` (`Manual`, `Eager`, `Lazy(T)`).
* **`preload()` / `build()`** — bulk-load path identical to the static DiskANN approach.
* **`insert()`** — streaming path that respects the policy.
* **`consolidate()`** — explicit consolidation for `Manual` policy or end-of-batch flush.
* **`delete()`** — soft-delete via tombstone set; filtered at search time.
* **`search()`** — hybrid: graph beam search on consolidated nodes ∪ brute-force scan
  of buffer, results merged by distance.

The crate is intentionally self-contained (no dependency on `ruvector-diskann`) so the
Vamana graph internals can evolve independently and the crate builds cleanly on all
targets without any optional features.

---

## Consequences

### Positive

* Streaming inserts are now O(R · log n) per vector (one beam search + backlink repair)
  instead of O(n · R · log n) for a full rebuild.
* Search quality is preserved: recall@10 ≈ 0.751 after streaming 1k vectors into a 9k
  base graph, matching the static 10k baseline (0.744) within noise.
* Throughput: ~3 100–3 200 QPS at k=10 on a 10k × 128-dim corpus regardless of
  consolidation policy — the hybrid search adds negligible overhead.
* Tombstone-based deletes avoid graph surgery; a periodic vacuum can compact the graph
  in a maintenance window.
* The trait-based `ConsolidationPolicy` enum is open for extension (e.g., background-
  thread auto-consolidation, SSD-spill for very large buffers).

### Negative / Risks

* During the consolidation window (buffer non-empty) search falls back to O(|buffer|·dim)
  brute-force for buffer vectors — acceptable for small T, but may introduce latency
  spikes if the buffer grows very large without consolidation.
* The current single-threaded beam-insert path consolidates T=100 vectors in ~275 ms
  on 4 cores (2.75 s total for 1k inserts); parallel consolidation (rayon) is a clear
  next step (see Roadmap).
* Medoid is not updated after streaming inserts; for heavily skewed distributions this
  could degrade recall.  Periodic medoid recomputation is recommended after large
  streaming batches.
* No SSD spill: the entire graph must fit in RAM.  For billion-scale deployments the
  `ruvector-diskann` mmap path remains necessary.

---

## Alternatives Considered

| Alternative | Reason Rejected |
|---|---|
| Full rebuild on every insert | O(n) rebuild cost; 28.8 s observed for 10k; unacceptable latency |
| HNSW in-place patching (Qdrant approach) | Requires locking the graph per insert; no α-robust quality guarantee |
| LSM-style segment merging (LanceDB, Weaviate) | Higher implementation complexity; segment-merge recall is harder to bound |
| FAISS IVF re-training | Requires periodic k-means; not graph-based; different recall profile |
| SSD posting-list SPANN (arXiv:2111.08566) | Overlaps with existing DiskANN mmap; lower marginal value |

---

## References

* Jayaram Subramanya et al., "DiskANN: Fast Accurate Billion-point Nearest Neighbor
  Search on a Single Node", NeurIPS 2019. arXiv:1908.10396
* Aditi Singh et al., "FreshDiskANN: A Fast and Accurate Graph-Based ANN Index for
  Streaming Similarity Search", VLDB 2022. arXiv:2105.09613
* Research document: `docs/research/nightly/2026-05-06-fresh-diskann/README.md`
