# ADR-196: RVF Index Shard — Portable Subgraph Extraction for Edge and Agent Memory

**Status**: Proposed  
**Date**: 2026-06-06  
**Branch**: `research/nightly/2026-06-06-rvf-index-shard`  
**Research doc**: `docs/research/nightly/2026-06-06-rvf-index-shard/README.md`

---

## Context

RuVector indexes can grow to millions of vectors. Deploying or migrating such an index to an edge device (Cognitum Seed, Raspberry Pi Zero, WASM runtime, MCP local server) is impractical when the full index consumes hundreds of megabytes. An agent operating on a constrained device needs only the slice of the index relevant to its current task — its **working memory shard**.

Existing partitioning systems (Milvus, Qdrant, Vespa) shard for distributed scale-out: many machines each hold a disjoint subset of the full index for horizontal throughput. This is architecturally different from the edge/portability problem: one device needs a self-contained, semantically coherent slice that can answer ANN queries without the parent index.

Three extraction strategies make sense for different use cases:
1. **BFS Shard**: expand from anchor nodes through graph edges — optimal for queries near anchor nodes (79.3% recall@10 for biased queries, measured).
2. **Coherence Shard**: select nodes by cosine similarity to anchor centroid — semantic coverage of the anchor domain (49.0% recall@10 for biased queries, measured).
3. **Hub Shard**: select nodes by incoming degree — captures HNSW upper-layer routing hubs; intended as a fast entry-point index, not a standalone recall index (18.5% recall@10 for biased queries, measured).

Key paper references:
- "Unleashing Graph Partitioning for Large-Scale Nearest Neighbor Search" (arXiv:2403.01797, VLDB 2025): validates that graph-based partitions concentrate 96%+ of query neighbors in one shard when query is routed to the correct shard.
- "Down with the Hierarchy: The 'H' in HNSW Stands for 'Hubs'" (arXiv:2412.01940, ICML 2025): validates the Hub Shard concept — high-degree nodes form the navigational highway.
- "Portable Agent Memory" (arXiv:2605.11032, Microsoft, May 2026): formalizes the need for serializable, portable vector memory for cross-device agent transfer.

---

## Decision

Introduce `crates/ruvector-shard` as a standalone proof-of-concept crate demonstrating three subgraph extraction strategies, binary serialization, and recall-vs-speed measurement. This crate serves as the research substrate for a future production-grade `crates/rvf/rvf-index-shard` that integrates with the full RVF wire format.

**API shape that should survive to production**:

```rust
pub trait ShardExtractor {
    fn extract(&self, graph: &KnnGraph, anchors: &[u32], budget: usize) -> Shard;
}

pub struct Shard {
    pub variant: ShardVariant,
    pub dim: usize,
    pub node_ids: Vec<u32>,
    pub vectors: Vec<f32>,
    pub local_neighbors: Vec<Vec<u32>>,
    pub meta: ShardMeta,
}

pub enum ShardVariant { Bfs, Coherence, Hub }
pub fn write_shard(shard: &Shard) -> Vec<u8>;
pub fn read_shard(bytes: &[u8]) -> ShardResult<Shard>;
pub fn search_shard(shard: &Shard, query: &[f32], k: usize) -> Vec<(u32, f32)>;
pub fn recall_at_k(results: &[(u32, f32)], ground_truth: &[(u32, f32)], k: usize) -> f32;
```

**What should remain behind a feature flag** (not in the PoC, future work):
- `rvf-segment`: integration with the full `SegmentType::Shard = 0x40` RVF wire format
- `quantized`: RabitQ 1-bit vector storage in shards (67KB → ~2KB per shard)
- `hnsw-search`: proper beam search within shard using `local_neighbors` (replaces brute-force)
- `overlapping`: K-hop border zone for improved recall at shard boundaries
- `witness`: cryptographic shard provenance via `rvf-crypto` WitnessChain

---

## Consequences

**Positive**:
- Enables edge deployment: 67KB shard fits in WASM linear memory and Raspberry Pi RAM.
- 8× query speedup over full brute-force for queries targeting the anchor region.
- 79.3% recall@10 for BFS shard with anchor-biased queries (a meaningful use case: agents querying their own task context).
- Portable binary format: 8-byte magic + version + typed per-node records; readable by any runtime.
- All three variants are measurably distinct: BFS excels for graph-local queries, Coherence for semantic queries, Hub for routing.
- Zero external dependencies beyond `rand`, `thiserror`, and `serde` (all workspace deps).

**Negative / Risks**:
- Static shard boundary: queries straddling the boundary get degraded recall. Not solved in this PoC.
- Shard staleness: shard diverges from the live index over time. Requires a delta-sync protocol (future work).
- Coherence shard may produce a disconnected subgraph (no edges between semantically similar but graph-distant nodes). Search within such a shard degrades to brute-force.
- Hub shard is unsuitable for standalone search (18.5% recall for biased queries). Must be used as routing-only prefix.
- Brute-force search within shard (current PoC): O(budget × dim) per query. Acceptable for budget ≤ 1024; requires HNSW beam search for larger shards.

---

## Alternatives Considered

**1. Full index serialization**: Ship the entire RVF index file. Rejected because a 1M-vector index at dim=768 weighs ~3GB; infeasible for edge deployment.

**2. IVF partition export**: Export one IVF cluster as the shard. Rejected because IVF partitions are spherical Voronoi cells — not graph-aware — and do not capture the local topology that BFS/Coherence shards exploit. Recall for IVF shards depends on the cluster granularity, which must be tuned offline.

**3. LEANN-style global pruning**: Prune the full HNSW graph to retain only hub nodes globally (LEANN approach). Rejected because the result is a globally pruned index, not an extractable subgraph of a larger index. LEANN does not produce portable typed shard files.

**4. DistributedANN head index**: BFS-collect the top-layer nodes into an in-memory head index. Closest to Hub Shard. Rejected as the primary approach because it is routing-only and does not address the semantic coverage problem that Coherence Shard targets. DistributedANN's format is proprietary.

**5. No shard, use full mincut partition**: Use the mincut algorithm already in `ruvector-mincut` to find the natural cluster boundary. More principled than BFS but O(n log n) extraction cost versus O(budget) for BFS. Proposed as a fourth extraction variant for follow-on work.

---

## Implementation Plan

**Phase 1 (this PR)**: Standalone `crates/ruvector-shard` PoC with:
- `KnnGraph`: brute-force k-NN graph builder for testing
- `BfsShard`, `CoherenceShard`, `HubShard`: three extractors
- `write_shard` / `read_shard`: custom binary wire format
- `search_shard`, `recall_at_k`: evaluation primitives
- `benchmark` binary with real measured results

**Phase 2 (next PR)**: Integration with `ruvector-core` HNSW:
- Implement `KnnGraph`-like interface over `HnswGraph` in `ruvector-core`
- Extract BFS/Coherence/Hub shards from real HNSW indexes
- Store extraction anchor IDs in the shard meta for reproducibility

**Phase 3 (future)**: Full RVF integration:
- Register `SegmentType::Shard = 0x40` in `rvf-types`
- Implement shard as a proper TLV segment in the RVF manifest
- Add `CapabilityManifest` record for MCP resource declaration
- Add `WitnessChain` for audit provenance

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-shard --bin benchmark` on x86_64 Linux.

**Graph build**: n=1024 vectors, dim=128, k_build=16: 142–151ms

**Extraction times** (12.5% shard, 128 nodes):
- BFS: 180–216µs
- Coherence: 223–241µs
- Hub: 148–171µs

**Wire sizes**:
- BFS: 68,608 bytes (67.0 KB)
- Coherence: 68,540 bytes (66.9 KB)
- Hub: 68,016 bytes (66.4 KB)

**Query benchmark (100 random + 100 anchor-biased queries, k=10)**:

| Variant | Mean µs | Speedup | Random R@10 | Biased R@10 |
|---------|---------|---------|-------------|-------------|
| Full BF | 133.0 | 1.00× | 100.0% | 100.0% |
| BFS | 16.1 | 8.1× | 13.9% | **79.3%** |
| Coherence | 15.9 | 8.1× | 12.5% | **49.0%** |
| Hub | 15.7 | 8.3× | 11.8% | 18.5% |

**All 17 acceptance tests passed.**

---

## Failure Modes

1. **Query not in anchor domain**: Recall degrades to ~shard_fraction (12.5% for 128-node shard). The shard is not designed for general-purpose search; callers must route queries to the appropriate shard.

2. **Disconnected coherence shard**: If anchor centroid is near a cluster boundary, selected nodes may have no graph edges between them. `search_shard` still works (brute force) but the `local_neighbors` will be sparse.

3. **Hub shard as standalone search**: 18.5% recall at biased queries. Do not use Hub shard as a standalone ANN index; use it only as a routing prefix to identify the correct BFS/Coherence shard.

4. **Wire format backward compat**: Version=1 is locked. Future fields must be added in new versions with a fallback read path.

---

## Security Considerations

1. **Shard data sensitivity**: Each shard contains a subset of the index vectors. If the full index contains sensitive embeddings, shards inherit the same sensitivity level. Apply the same access controls as the parent index.

2. **Shard tampering**: The current wire format has no checksum or signature. A tampered shard could cause incorrect search results. Mitigation: compute an HMAC over the wire bytes at write time; verify at read time. Use `rvf-crypto` in Phase 3.

3. **Integer overflow in `read_shard`**: The `MAX_NODES` sanity cap (1,000,000) prevents allocation attacks from malformed wire data. The per-node neighbor count is uncapped; a future hardening pass should add `MAX_NEIGHBORS_PER_NODE`.

4. **Path traversal in shard file loading**: `read_shard` operates on `&[u8]` (no file I/O). File path validation must be handled by the caller before loading bytes.

---

## Migration Path

Existing code that uses `ruvector-core`'s `VectorDb` or `HnswIndex` is unaffected; `ruvector-shard` is additive.

When Phase 2 integrates with `ruvector-core`, the `KnnGraph` type in this crate can be replaced with an adapter over `HnswGraph`. The `ShardExtractor` trait API is stable.

When Phase 3 registers `SegmentType::Shard = 0x40` in `rvf-types`, the current `RVSHARD\0` magic-byte format can be auto-detected and upgraded by the RVF reader: any file starting with `RVSHARD\0` is a v1 standalone shard; any RVF file containing a `Shard` segment is a v2 embedded shard.

---

## Open Questions

1. What is the right default anchor selection strategy? Random (current) vs maxmin-diverse vs query-distribution-based?

2. Should the shard include the original parent index's node-level metadata (e.g., document IDs, timestamps)? Currently only vector data and neighbor lists are stored.

3. How does shard recall scale with budget at larger n (n=100K, n=1M)? The n=1024 PoC gives encouraging results; large-scale validation is needed.

4. Should Coherence Shard re-induce shard-local edges after selecting nodes? This would improve search but adds O(budget² × dim) build cost.

5. Is a mincut-based fourth variant (Phase 2 or beyond) worth implementing before production? Mincut produces more principled partition boundaries but at higher extraction cost.
