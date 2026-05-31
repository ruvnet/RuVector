---
adr: 194
title: "Tiered Agent Memory — Coherence-Driven Hot/Warm/Cold Tier Promotion"
status: accepted
date: 2026-05-19
authors: [ruvnet, claude-flow]
related: [ADR-143, ADR-193, ADR-191, ADR-178]
tags: [agent-memory, tiered-memory, coherence, vector-search, quantization, nightly-research]
---

# ADR-194 — Tiered Agent Memory: Coherence-Driven Hot/Warm/Cold Tier Promotion

## Status

**Accepted.** Implemented on branch `research/nightly/2026-05-19-tiered-agent-memory` as
`crates/ruvector-tiered-memory`. All 11 unit tests pass; build is green with
`cargo build --release -p ruvector-tiered-memory`. Acceptance test passes (recall@10 ≥ 75%).

## Context

RuVector's flat vector store model keeps all vectors in RAM at full precision. For short-lived
search workloads this is optimal. For long-running AI agents that accumulate vector memory over
hours or days, it is unsustainable:

| Memory size | Embedding dims | RAM usage |
|-------------|----------------|-----------|
| 10,000 vectors | 768 | 30.7 MB |
| 100,000 vectors | 1,536 | 614 MB |
| 1,000,000 vectors | 1,536 | 6.1 GB |

A 1M-vector agent memory at LLM embedding size (1,536 dims) requires 6 GB of RAM for vectors
alone. No embedded Cognitum Seed, no edge device, and no cost-efficient cloud deployment can
sustain this.

The solution — tiered memory — is well-established in database engineering (buffer pools,
page tables, NUMA hierarchies). It is not yet applied to vector databases with agent-specific
semantics.

MEMTIER (arXiv:2605.03675, May 2026) formalizes tiered agent memory with three axes:
temporal decay, semantic relevance, and explicit importance. This ADR implements two of these
axes (relevance and recency) as a production-grade starting point.

### Why coherence-gated promotion

RuVector already has `prime-radiant`, a coherence scoring engine that computes cosine
similarity between vectors and a running centroid. Applying this to tier promotion:

- Vectors whose cosine similarity to the current query centroid exceeds a threshold are
  *coherent* with the agent's current task. They belong in the hot tier.
- Vectors with intermediate coherence belong in warm (compressed, decoded at search time).
- Vectors with low coherence belong in cold (archived, accessed rarely).

This is semantically superior to LRU: LRU promotes whatever was accessed most recently,
even if that was an off-topic search. Coherence promotion maintains a semantic model of
what the agent cares about.

## Decision

We introduce `crates/ruvector-tiered-memory` with a `TieredMemoryStore` trait and three
implementations:

```rust
pub trait TieredMemoryStore {
    fn insert(&mut self, id: u64, vector: Vec<f32>);
    fn search(&mut self, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn tier_stats(&self) -> TierStats;
    fn name(&self) -> &str;
}
```

1. **`FlatMemory`**: baseline, no tiering, recall = 100%.
2. **`LruTieredMemory`**: access-frequency tiering. Hot/warm/cold capped by vector count.
   Warm tier is INT8 quantized (4× memory compression). 24.5% memory reduction; 80.5% recall.
3. **`CoherenceTieredMemory`**: coherence-score tiering. Tier assignment updates via
   running centroid. Periodic rebalancing. 3.7% memory reduction; 100% recall.

The `CoherenceTieredMemory` variant is the primary recommendation for production use because
it achieves full recall while reducing memory. The LRU variant is appropriate for use cases
where an explicit 80% recall is acceptable and 24% memory savings are valuable.

## Consequences

### Positive

1. **Memory-scalable agent runtime**: Agents with 100K+ memories can operate with hot tier
   limited to a configurable count (e.g., top-10% of memories).
2. **Ecosystem composability**: `TieredMemoryStore` is a trait; any future implementation
   (HNSW hot tier, persistent cold tier, distributed tier) plugs in without API changes.
3. **Coherence reuse**: The centroid-based scoring directly reuses `prime-radiant`'s
   coherence model, creating a meaningful connection between two existing crates.
4. **Tier-annotated results**: `SearchResult.tier` tells downstream consumers (MCP tools,
   ruFlo workflows) which tier answered each query — enabling smarter caching.
5. **Zero external dependencies**: The crate depends only on `rand` (for tests). Safe Rust throughout.

### Negative

1. **Recall tradeoff**: The LRU variant has 80.5% recall due to quantization errors when
   vectors traverse warm→cold. The coherence variant avoids this by keeping warm small.
2. **Rebalancing cost**: Periodic rebalancing is O(N×D). For N > 100K this must be async.
3. **Threshold calibration**: `hot_threshold` and `warm_threshold` must be tuned per
   embedding dimension. In 128-dim space, thresholds of 0.15/0.05 work; in 1536-dim space,
   thresholds of 0.04/0.01 are appropriate (cosine sims concentrate near 0 as D grows).
4. **No persistence**: Cold tier is in-RAM. A production cold tier needs `sled` or `redb`.
5. **No distributed consensus**: Centroid updates from multiple agents need Raft coordination.

## Alternatives Considered

### 1. Time-to-live (TTL) eviction

Assign each memory a TTL based on insertion time. Move to cold when TTL expires.  
**Rejected**: TTL is blind to semantic relevance. A 3-day-old memory about the current task
is more valuable than a 1-second-old memory from an off-topic tool call.

### 2. LLM-scored importance

Before eviction, query an LLM to score each memory's importance.  
**Rejected**: O(N) LLM calls for rebalancing would be prohibitively expensive. Not suitable
for a vector database embedded in Rust without an LLM.

### 3. IVF-based tiering (cluster-level)

Assign entire IVF clusters to tiers. A cluster is hot if it was recently probed.  
**Rejected**: IVF requires a training phase; `ruvector-rairs` (ADR-193) covers that approach.
Per-cluster tiering is coarser than per-vector; coherence tiering is more flexible.

### 4. DiskANN-style SSD cold tier

Use `ruvector-diskann`'s graph-on-SSD model for the cold tier.  
**Deferred**: This is the natural production path. The cold tier in this PoC is in-RAM;
`ruvector-diskann` integration is the obvious next step for a production cold tier.

## Implementation Plan

### Phase 1 (this PoC — complete)
- [x] `TieredMemoryStore` trait with `insert`, `search`, `tier_stats`
- [x] `FlatMemory` baseline
- [x] `LruTieredMemory` with hot/warm/cold, INT8 warm quantization
- [x] `CoherenceTieredMemory` with running centroid and periodic rebalancing
- [x] 11 unit tests, all passing
- [x] Benchmark binary with recall, latency, throughput, memory metrics
- [x] Acceptance test (recall@10 ≥ 75%)

### Phase 2 (production hardening)
- [ ] Async rebalancing via `rayon` or `tokio::task::spawn_blocking`
- [ ] Persistent cold tier using `sled` (append-only log + index)
- [ ] Auto-calibrated thresholds (sample first 1K inserts, set at 80th/60th percentiles)
- [ ] Per-namespace isolation (HashMap<Namespace, CoherenceTieredMemory>)
- [ ] Exact cold tier (store original fp32 alongside quantized; use fp32 at eviction)

### Phase 3 (ecosystem integration)
- [ ] HNSW hot tier: replace flat scan with `ruvector-core` HNSW
- [ ] Distributed centroid: use `ruvector-raft` for multi-agent centroid consensus
- [ ] MCP tool surface: expose via `mcp-gate` (insert, search, tier_stats, rebalance)
- [ ] ruFlo integration: schedule nightly rebalancing as a ruFlo workflow step
- [ ] Proof-gated eviction: require `ruvector-verified` witness on warm→cold transition
- [ ] RVF snapshot format: serialize tiered memory state as a portable RVF package

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-tiered-memory`.  
Hardware: x86-64, Intel Celeron N4020, Linux 6.18.5, rustc 1.87.0.  
Dataset: N=5,000, D=128, Q=500, K=10.

| Variant | mean µs | p50 µs | p95 µs | QPS | memory KB | recall@10 |
|---------|---------|--------|--------|-----|-----------|-----------|
| FlatMemory (baseline) | 884.9 | 880.9 | 934.9 | 1,119 | 2,500 | 100.0% |
| LruTieredMemory (alt-A) | 1,067.5 | 1,049.3 | 1,189.2 | 926 | 1,888 | 80.5% |
| CoherenceTieredMemory (alt-B) | 956.6 | 930.9 | 1,104.0 | 1,044 | 2,408 | 100.0% |

Acceptance threshold: recall@10 ≥ 75%. All variants: **PASS**.

**Note on LRU recall**: The 80.5% recall for LruTieredMemory is not a bug. It reflects a
genuine tradeoff: the warm tier (1,666/5,000 vectors) stores INT8 quantized vectors, which
introduces squared-distance errors ≤ 1.88 for 128-dim vectors with range 20. When multiple
true nearest neighbors in the same cluster differ by less than this error in squared distance,
rank swaps occur. This is the honest behavior of an approximate tiered store. See the research
document for the full mathematical analysis.

## Failure Modes

1. **All-cold startup**: Until the first query, all inserts go to cold (centroid uninitialized).
   Search on a fresh store with no prior queries returns correct results but from cold only.
   Mitigation: warm up with representative queries before serving production traffic.

2. **Centroid drift attack**: An adversary flooding the system with queries in a specific
   direction shifts the centroid and demotes legitimate memories to cold. Mitigation: rate-limit
   centroid updates; validate query vectors at system boundaries.

3. **Rebalance timeout**: Synchronous rebalancing on N=1M vectors takes seconds. Mitigation:
   Phase 2 async rebalancing.

4. **Warm→cold precision loss**: Vectors that pass through warm accumulate quantization error.
   After 5+ encode-decode cycles, the error can exceed the quantization bound. Mitigation:
   track encode count per vector; evict multi-cycle vectors directly to fp32 cold.

## Security Considerations

1. **Tier information disclosure**: `SearchResult.tier` reveals which tier answered each query.
   In a multi-tenant system, strip tier from externally visible results.
2. **Namespace isolation**: Phase 2 must enforce per-namespace isolation so one agent cannot
   influence another's tier state.
3. **Proof-gated eviction**: Phase 3 integration with `ruvector-verified` provides cryptographic
   audit trail of tier transitions.

## Migration Path

This crate introduces a new trait and three new structs. There is no migration required from
existing `ruvector-core` users. Adoption is opt-in.

For users of `mcp-gate` (Phase 3): the MCP `memory_*` tools will be new tools, not replacements.
Existing HNSW search tools remain unchanged.

## Open Questions

1. **What is the right acceptance recall threshold for a production tiered store?** 75% is
   a reasonable default for an approximate store; production deployments may require 90%+ in
   which case only `CoherenceTieredMemory` qualifies.

2. **Should the warm tier use per-vector or global quantization?** Global quantization (across
   all warm vectors) would give more consistent distance estimates but requires scanning all warm
   vectors to compute min/max before the first quantization.

3. **When should the centroid be reset?** For task-switching agents (a new conversation starts),
   the centroid from the previous task is misleading. A `reset_centroid()` method or task-scoped
   centroid namespacing is needed.

4. **Should tier annotations be persisted?** If tier assignments are persisted (e.g., in RVF
   snapshot format), agents can resume with hot-tier memories pre-loaded, avoiding cold-start
   recall loss.
