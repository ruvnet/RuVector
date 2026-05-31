---
adr: 196
title: "Graph-Cut Memory Compaction — k-NN clustering with farthest-point sampling for agent memory"
status: accepted
date: 2026-05-31
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-194, ADR-143]
tags: [agent-memory, compaction, graph-cut, fps, vector-search, ruFlo, mcp, nightly-research]
---

# ADR-196 — Graph-Cut Memory Compaction

## Status

**Accepted.** Implemented on branch
`research/nightly/2026-05-31-graph-cut-memory-compaction` as
`crates/ruvector-mem-compact`.  All unit and integration tests pass.
Build is green with `cargo build --release -p ruvector-mem-compact`.

---

## Context

RuVector's agent substrate (AgenticDB, ruFlo, MCP memory tools) accumulates
vector embeddings continuously.  Without compaction, three problems occur:

1. **Memory growth** — unbounded embedding stores exhaust RAM, especially on
   edge devices (Cognitum Seed, Pi Zero 2W).
2. **Recency bias** — the obvious fix (drop oldest N%) silently destroys
   coverage of early-inserted concept clusters.  In a 400-entry store of 20
   concepts × 20 copies each, AgeTtl at 50% compaction drops the first 10
   concepts entirely (cluster-coverage recall = 50%).
3. **Duplicate accumulation** — long-running agents repeatedly embed the same
   concept with slight paraphrase variation, inflating the index without adding
   retrieval value.

No production vector database (Qdrant, Milvus, Weaviate, LanceDB, Pinecone)
exposes a principled content-aware compaction API as of 2026.  The closest
published work — GaussDB-Vector (VLDB 2025), which compacts at the segment
level — addresses storage layout, not semantic redundancy.

The agent memory literature (MemGPT/Letta, A-MEM, Mem0, GraphRAG) acknowledges
compaction as an open problem but does not provide geometric solutions.

---

## Decision

Introduce **`ruvector-mem-compact`** as a standalone crate implementing the
`MemoryCompactor` trait with three strategies, benchmarked against each other:

### Trait

```rust
pub trait MemoryCompactor {
    fn compact(&self, store: &MemoryStore, target_ratio: f32) -> CompactionResult;
    fn name(&self) -> &'static str;
}
```

### Primary strategy: `GraphCutCompactor`

Three-phase algorithm:

**Phase 1 — Cluster discovery** (graph-cut via union-find):
Build a k-NN similarity graph at `cluster_threshold`.  Connect any two vectors
whose cosine similarity exceeds the threshold.  Apply union-find to find
connected components.  This correctly handles transitivity: A≈B≈C merges into
one cluster even if A and C are only moderately similar.

**Phase 2 — Proportional farthest-point sampling**:
Each cluster of size `m` keeps `ceil(m × target_ratio)` vectors, chosen by
farthest-point sampling (FPS): seed with the cluster centroid, then greedily
add the vector maximising min-distance to all already-chosen vectors.  FPS
provides a 2-approximation to the k-center problem (González 1985).

**Phase 3 — Global trim / pad**:
Trim the globally-most-redundant vectors if over-budget; pad with the
globally-most-diverse non-representative vectors if under-budget.

### Baseline strategies (for comparison)

- **`AgeTtlCompactor`**: drop the N oldest entries.  O(N log N).
- **`ThresholdCompactor`**: pointwise cosine-threshold deduplication.  O(N²)
  worst case but terminates early.  Misses transitive redundancy.

---

## Consequences

### Positive

- GraphCutCompact achieves **100% cluster-coverage recall** at 50% compaction
  on high-redundancy episodic data (measured: 40 concepts × 20 near-duplicate
  copies, D=64), compared to 50% for AgeTtl.
- GraphCutCompact achieves **+6.6 pp ID-exact recall@10** over AgeTtl on
  moderate Gaussian data (N=1000, D=64, 10 clusters, std=0.2).
- Trait-based design: new strategies add in one file without changing existing
  code.
- No external service dependencies: compiles offline, WASM-compatible.

### Negative / Limitations

- **O(N²) compaction time** (brute-force k-NN): 847ms at N=1000, D=64 in
  release build.  Not suitable for N > ~5K without HNSW-accelerated Phase 1.
- `cluster_threshold` requires calibration per dataset.  A wrong threshold
  merges distinct concepts (too low) or fails to find clusters (too high).
- FPS is O(m² × n_reps) per cluster — fast for small clusters but potentially
  slow if one cluster dominates.

---

## Alternatives Considered

| Alternative | Why not chosen |
|-------------|----------------|
| k-means clustering | Requires specifying K in advance; fails when K varies; non-deterministic without fixed seeding |
| Pure ThresholdCompactor | Misses transitive redundancy; parameter harder to tune for diverse cluster sizes |
| Random 50% sample | Equivalent to AgeTtl; drops entire old clusters; no semantic awareness |
| Summarisation (like MemGPT) | Modifies embedding values; breaks exact retrieval; requires LLM call |
| Importance-weighted retention | Requires domain-specific importance scores; not general |

---

## Implementation Plan

1. **PoC (done):** `crates/ruvector-mem-compact` — standalone crate with all three
   strategies, real benchmarks, integration tests.

2. **HNSW acceleration (next):** Replace Phase 1 brute-force k-NN with
   `ruvector-core` HNSW index query.  Target: O(N log N) compaction.

3. **ruFlo integration:** Add a ruFlo workflow trigger that fires compaction
   when `namespace.memory_usage > threshold`.

4. **MCP tool:** Add `memory_compact` tool to `mcp-brain-server` using
   `GraphCutCompactor` as the default strategy.

5. **Auto-threshold calibration:** Compute the 70th-percentile k-NN similarity
   at index build time; use as default `cluster_threshold`.

6. **Feature flag:** Gate `GraphCutCompactor` behind `graph-cut` feature flag
   to keep the default build lightweight.

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-mem-compact` (2026-05-31):

**Suite A — Moderate Gaussian (ID-exact recall@10):**

| Strategy | N-kept | ID-recall | Cmpct ms |
|----------|--------|-----------|----------|
| AgeTtl | 500/1000 | 52.2% | 0.01 |
| ThresholdMerge | 500/1000 | 47.8% | 6.29 |
| **GraphCutCompact** | **500/1000** | **58.8%** | **847.5** |

**Suite B — High-redundancy episodic (cluster-coverage recall@5):**

| Strategy | Cluster-cov | Cmpct ms |
|----------|-------------|----------|
| AgeTtl | 50.0% | 0.01 |
| ThresholdMerge | 100.0% | 0.23 |
| **GraphCutCompact** | **100.0%** | **13.04** |

Acceptance gate: **PASS** (cluster-cov recall@5 ≥ 90% → 100% achieved).

---

## Failure Modes

| Mode | Trigger | Mitigation |
|------|---------|------------|
| Concept merger | cluster_threshold too low | Raise threshold; test with known-distinct queries |
| Cluster fragmentation | cluster_threshold too high | Lower threshold; check component count |
| O(N²) timeout | Large store (N > 10K) | Activate HNSW-accelerated Phase 1 (next iteration) |
| FPS outlier selection | Noisy cluster edges | Pre-filter outliers (> 3σ from cluster centroid) before FPS |
| Silent memory loss | Compaction removes provably unique entry | Witness log (ruvector-mincut/witness) before compaction |

---

## Security Considerations

1. **PII**: compacted stores still contain the kept embeddings; PII obligations apply.
2. **Adversarial cluster poisoning**: an attacker inserting near-duplicate vectors
   of a target memory can cause it to be classified as redundant and evicted.
   Mitigation: check access_count before eviction (high-access entries resist
   removal).
3. **Audit trail**: record compaction events in the witness log for compliance.

---

## Migration Path

`ruvector-mem-compact` is a new standalone crate; there is no breaking change to
existing crates.  The `MemoryCompactor` trait is additive.  Future integration
into `AgenticDB` will add an optional `compactor: Option<Box<dyn MemoryCompactor>>`
field without changing existing APIs.

---

## Open Questions

1. Should `cluster_threshold` be auto-calibrated or require explicit configuration?
2. Should FPS be replaced by a learned diversity sampler (LFPS, ICLR 2025) in
   the production path?
3. What is the right `target_ratio` for edge devices with tight RAM?
   (Cognitum Seed: ~512 MB → need ≥ 80% compaction for large sessions.)
4. Should compaction be triggered by entry count, RAM usage, or both?
5. Should the witness log be a ring buffer or an append-only log?

---

## References

- González (1985). k-center 2-approximation via farthest-point sampling.
- Azizi et al., SIGMOD 2025. Graph-Based Vector Search survey.
- Zhang et al., arXiv:2602.08097 (2026). "Prune, Don't Rebuild."
- Xu et al., arXiv:2502.12110 (2025). A-MEM agent memory.
- Chhikara et al., arXiv:2504.19413 (2025). Mem0 production agent memory.
- Sun et al., VLDB 2025. GaussDB-Vector production segment compaction.
- Yang et al., arXiv:2602.05665 (2026). Graph-based Agent Memory Taxonomy.
- Du, arXiv:2603.07670 (2026). Memory for Autonomous LLM Agents survey.
