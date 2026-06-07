---
adr: 194
title: "GCVS — Graph-Coherence Vector Search with Coherence-Gated BFS Expansion"
status: accepted
date: 2026-05-22
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-192, ADR-186]
tags: [graph, ann, vector-search, graph-rag, coherence, bfs, agent-memory, nightly-research]
---

# ADR-194 — GCVS: Graph-Coherence Vector Search

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-22-graph-coherence-search`
as `crates/ruvector-gcvs`. All 6 unit tests pass; build is green with
`cargo build --release -p ruvector-gcvs`.

## Context

RuVector has a complete ANN stack (HNSW in `ruvector-core`, DiskANN in `ruvector-diskann`,
IVF in `ruvector-rairs`, filtered ANN in `ruvector-acorn`) and a graph substrate
(`ruvector-graph`, `ruvector-mincut`, `ruvector-gnn`). However, there is no crate that
**uses the graph during ANN retrieval** — graph and vector search are disjoint pipelines.

This gap matters for two critical 2026 use cases:

1. **GraphRAG**: retrieval augmented generation where relevant context is reached via
   multi-hop graph traversal, not just embedding proximity. Microsoft's GraphRAG, spreading-
   activation RAG (arXiv 2512.15922), and HMGI (arXiv 2510.10123) all demonstrate that
   combining graph structure with vector similarity significantly improves recall on
   multi-hop queries.

2. **Agent memory**: an agent's memory graph connects concepts that the embedding model
   separates. When recalling context, traversal through the association graph recovers
   memories that pure nearest-neighbour search misses.

No major open-source vector database (Qdrant, Weaviate, LanceDB, Milvus, FAISS, pgvector)
performs in-retrieval coherence-gated graph traversal. This is a novel capability for
the RuVector ecosystem.

## Decision

We introduce `crates/ruvector-gcvs` implementing three variants via a common `GcvsIndex`
trait:

### Variant 1 — `FlatSearch` (baseline)

Brute-force O(N·D) cosine similarity scan. Returns exact top-K by embedding similarity.
Recall = 0% on cross-cluster graph-only ground truth by construction (cannot reach
orthogonal semantic clusters). Serves as the recall baseline.

### Variant 2 — `GraphAugSearch` (alternative A)

Three phases:
1. Vector scan for `seed_k` nearest seeds.
2. BFS through the semantic graph up to `bfs_depth` hops from each seed.
3. Cosine re-rank of all candidates; return top-K.

Recovers cross-cluster graph neighbours unreachable by vector similarity alone.

### Variant 3 — `GraphCohSearch` (alternative B)

Same as GraphAugSearch but with a coherence gate in BFS: edge (u→v) is only traversed
if `cosine(query, v) ≥ coherence_threshold`. Prunes semantically irrelevant branches,
reducing candidate set size while maintaining recall on relevant targets.

### API shape

```rust
pub trait GcvsIndex {
    fn insert(&mut self, id: usize, vector: Vec<f32>) -> Result<()>;
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<Hit>>;
    fn len(&self) -> usize;
    fn name(&self) -> &'static str;
}

pub struct Hit { pub id: usize, pub score: f32 }
```

`add_edge(from: usize, to: usize)` is an associated method on the graph-aware variants.

## Consequences

### Positive

- First in-retrieval graph-coherence traversal primitive in the RuVector ecosystem.
- Connects `ruvector-graph`, `ruvector-coherence`, and ANN stack in a single retrieval path.
- Demonstrated +32 pp recall improvement on cross-cluster graph targets vs FlatSearch.
- Trait-based API allows swapping in HNSW seeds (vs brute-force) without changing call sites.
- `GcvsIndex` is extensible to weighted graphs, multi-hop decay, and GNN-driven gating.

### Negative / Trade-offs

- Brute-force seed phase: O(N·D) per query. Production use requires HNSW seed phase.
- Graph memory overhead: +12.5% for `HashMap<usize, Vec<usize>>` at N=5K. Larger with CSR.
- `coherence_threshold` is a free parameter; wrong values reduce recall or block traversal.
- BFS can explode on dense graphs: must add `max_candidates` cap in production.
- Graph is not yet persistent (in-memory only); requires `serde + bincode` for persistence.

## Alternatives Considered

### A. Implement a unified HNSW+BM25 sparse graph (researcher's winner, score 4.75)

The SOTA winner from the goal-planner sub-agent. Builds a single HNSW proximity graph
hosting both dense vector edges and BM25 sparse term edges. Rejected for this nightly
run because:
1. BM25 + vector hybrid already exists partially in `ruvector-core/advanced_features/hybrid_search.rs`.
2. The unified graph approach requires modifying core HNSW internals — higher risk for one
   nightly run.
3. GCVS is genuinely novel (no overlap with existing code) and connectable to more ecosystem
   components.
Recommended as a future nightly topic.

### B. Semantic drift detector for agent memory

Would track angular velocity of memory embeddings over time. Novel but purely
monitoring-oriented; GCVS provides a retrieval primitive with clearer ROI.

### C. Proof-gated vector writes with witness chains

`ruvector-verified` already provides the proof infrastructure. GCVS is complementary:
proof-gate the graph edge writes, then use GCVS for retrieval.

### D. Streaming HNSW with lazy deletes

`ruvector-delta-index` partially covers this. More invasive than GCVS and requires deeper
HNSW internals modification.

## Implementation Plan

**Phase 1 (today)**:
- [x] `crates/ruvector-gcvs` with three variants implementing `GcvsIndex`
- [x] Deterministic benchmark binary with real measured numbers
- [x] 6 unit tests including acceptance recall threshold test
- [x] ADR-194 and research README

**Phase 2 (next nightly)**:
- [ ] Swap brute-force seeds for `hnsw_rs` call to `ruvector-core`
- [ ] CSR graph layout in `graph.rs` for O(1) neighbour access
- [ ] Add `max_candidates` cap to prevent BFS explosion

**Phase 3 (production hardening)**:
- [ ] Graph serialisation via `bincode`
- [ ] Edge weights (`f32`) for weighted coherence gating
- [ ] Expose `GcvsServer` on `ruvector-server` HTTP API
- [ ] Add to `mcp-brain-server` as `graph_coherence_search` MCP tool
- [ ] RVF packaging: bundle graph + vector index as `.rvf`
- [ ] WASM feature flag for Cognitum Seed target

## Benchmark Evidence

Hardware: x86-64, Linux 6.18.5, Intel Celeron N4020, rustc 1.94.1, release build.
Dataset: N=5,000, DIM=128, 3 orthogonal clusters, 20,000 directed cross-cluster edges.
Ground truth: direct cross-cluster graph neighbours (not same-cluster vectors).

| Variant | Recall@10 | Mean µs | p50 µs | p95 µs | QPS |
|---------|-----------|---------|--------|--------|-----|
| FlatSearch (baseline) | 0.0% | 1,306 | 1,298 | 1,340 | 765 |
| GraphAugSearch | **32.0%** | 1,284 | 1,281 | 1,321 | 779 |
| GraphCohSearch | **32.0%** | 1,276 | 1,274 | 1,317 | 783 |

Acceptance test: graph variants recall improvement ≥ 5 pp over FlatSearch. **PASS.**

The 32% recall improvement is honest and bounded by the scenario: with `seed_k=3` and
`bfs_depth=1`, the BFS reaches the query's direct graph neighbours (avg 4.0 per query).
The query itself is one of the 3 seeds; its graph neighbours appear in the top-10 after
re-ranking. Averaged over 200 queries (including those with fewer than 4 graph edges),
recall = 32%.

**Competitor comparison**: No open-source vector database was directly benchmarked. The
claim "no competitor ships in-retrieval coherence-gated graph traversal" is based on public
documentation review, not head-to-head benchmarks.

## Failure Modes

| Failure | Trigger | Mitigation |
|---------|---------|------------|
| 0% recall | seed_k too small; query not its own seed | Guarantee query is always a seed (special case) |
| BFS explosion | Dense graph + large bfs_depth | Add `max_candidates` hard cap |
| Gate blocks targets | Threshold too strict | Start at -1.0; tune upward with ruFlo |
| Stale edges | Vectors updated without edge repair | Wire into `ruvector-delta-index` repair loop |
| Graph poisoning | Adversary inserts malicious edges | Proof-gate edge writes via `ruvector-verified` |

## Security Considerations

1. **Graph poisoning attack**: a write path that accepts graph edges without authentication
   allows an adversary to redirect retrieval to injected documents. Mitigation: require
   proof attestation from `ruvector-verified` on every `add_edge` call.
2. **Information leakage via graph structure**: the adjacency list reveals which documents
   are associated. In multi-tenant deployments, use mincut partitioning to enforce tenant
   isolation on the graph.
3. **Coherence threshold bypass**: a crafted query could be constructed to have high cosine
   similarity with adversarial documents if embeddings are controllable. Mitigation:
   proof-gate the vector writes, not just the edge writes.

## Migration Path

`ruvector-gcvs` is an additive crate. No existing crate is modified. Migration to production:

1. Add `ruvector-gcvs` dependency to `ruvector-server`.
2. Add `GET /graph_search` endpoint routing to `GraphCohSearch`.
3. Expose as `graph_coherence_search` MCP tool in `mcp-brain-server`.
4. Bundle in RVF packages as an optional cognitive kernel.

## Open Questions

1. What is the theoretically optimal `coherence_threshold` for a given graph? (Candidate:
   the Fiedler value of the local subgraph — computable via `ruvector-coherence/spectral`.)
2. Should GCVS merge with `ruvector-graph` or remain a separate retrieval-layer crate?
3. Does multi-hop BFS (depth=2+) require a different coherence decay model?
4. Should `GcvsIndex::search` accept an optional graph reference, making the graph
   a query-time parameter rather than index-time configuration?
5. Can `ruvector-gnn` provide a learned coherence score as a drop-in replacement for
   the cosine gate?
