# ADR-272: Namespace-Partitioned Multi-Agent HNSW Memory

**Status**: Proposed  
**Date**: 2026-07-10  
**Author**: Nightly Research Agent  
**Branch**: `research/nightly/2026-07-10-ns-partitioned-ann`  
**Crate**: `crates/ruvector-ns-partition`  
**Related**: ADR-240 (Coherence-HNSW), ADR-256 (Hybrid Search), ADR-268 (Capability-Gated ANN), ADR-227 (Proof-Gated Writes)

---

## Context

RuVector is being built as a Rust-native cognition substrate for autonomous
agents.  Multi-agent deployments — orchestrators, coding agents, RAG pipelines,
ruFlo workflow loops — each need an **isolated vector memory space** while also
requiring controlled cross-agent knowledge retrieval.

Current state of the ecosystem:

- There is no first-class namespace concept in `ruvector-core`.
- Global HNSW search with post-hoc namespace filtering (the most common approach
  in Pinecone, pgvector, Chroma) degrades recall when the filter selectivity is
  low relative to ef_search capacity.
- `ruvector-capgated` (ADR-268) provides per-vector access control but not
  per-namespace index isolation or performance locality.

This PoC quantifies the tradeoffs between three namespace strategies on a
6 000-vector, 8-namespace workload and proposes the `NamespacedIndex` trait as
a production API shape for `ruvector-core`.

---

## Decision

**Adopt the Partitioned strategy as the recommended production path** for
namespace-aware agent memory in RuVector.

One HNSW per namespace provides:
- 22× faster single-namespace search (202 µs vs 4 390 µs for GlobalFlat).
- 97.5% cross-namespace recall (vs 42.7% for GlobalFlat at same ef).
- Faster construction (7.9 s for 8 × 750 vectors vs 14.8 s for 6 000 vectors).
- Zero extra memory overhead vs a single global index (4 779 KB vs 4 988 KB).

The `NamespacedIndex` trait (see Implementation Plan) should be added to
`ruvector-core` as a stable interface, with `Partitioned` as the default
implementation.

`HierarchicalNS` (routing index + per-namespace HNSWs) is a research candidate
for large-namespace deployments (>32 namespaces) where sequential cross-NS sweep
becomes the bottleneck.  It requires route_k tuning before production use.

`GlobalFlat` (single HNSW + post-filter) is deprecated for namespace-aware
workloads.  It should NOT be used when namespace selectivity is < 50%.

---

## Consequences

### Positive

- Single-agent queries are focused and fast (O(N/K) instead of O(N)).
- Cross-agent recall is high (97.5% measured).
- Construction cost scales sublinearly with total vectors.
- Namespace HNSW can be exported as `.rvf` bundle for edge deployment.
- Clear security boundary: per-namespace index + per-namespace capability mask.
- Integration path with `ruvector-mincut` for namespace-level graph compaction.

### Negative / Risk

- Cross-namespace search requires O(K_namespaces) sequential sweeps, scaling
  linearly with namespace count.  Above ~32 namespaces, the HierarchicalNS
  variant becomes necessary.
- Each namespace maintains its own HNSW state, so namespace deletion requires
  proper cleanup (tombstoning, graph rebuild if needed).
- Very small namespaces (<50 vectors) have poorly connected HNSW graphs; ef must
  be increased proportionally.

---

## Alternatives Considered

| Alternative | Why Rejected |
|-------------|-------------|
| GlobalFlat (single HNSW + post-filter) | 42.7% cross-NS recall at ef=64 is unacceptable for agent memory |
| Metadata-filtered search (ACORN style, ADR-256) | Requires tight integration with existing filter index; doesn't reduce construction cost |
| Full collection isolation (Weaviate style) | No cross-namespace search without application-level orchestration |
| IVF-based namespace routing | IVF recall degrades for small namespaces; no advanage over HNSW at N=750 |
| Shared flat index + bitmap per namespace | O(N) scan for every query; does not scale |

---

## Implementation Plan

### Phase 1: Trait Stabilization (this PR)

Define `NamespacedIndex` trait in `ruvector-ns-partition`:
```rust
pub trait NamespacedIndex {
    fn insert(&mut self, ns: &str, id: u64, vector: Vec<f32>);
    fn search_single(&self, ns: &str, query: &[f32], k: usize, ef: usize) -> Vec<NsResult>;
    fn search_cross(&self, query: &[f32], k: usize, ef: usize) -> Vec<NsResult>;
    fn memory_bytes(&self) -> usize;
}
```

Ship three concrete implementations: `GlobalFlat`, `Partitioned`,
`HierarchicalNS`.

### Phase 2: Core Integration (follow-on PR)

Move `NamespacedIndex` into `ruvector-core`.  Replace `MiniHnsw` (minimal PoC
implementation) with the production `HnswGraph` from `ruvector-core`.  Add:
- Namespace eviction + RVF snapshot/restore.
- Prometheus metrics per namespace.
- Async parallel cross-NS search via Tokio tasks.

### Phase 3: MCP Surface (follow-on PR)

Expose via `mcp-brain` as:
```
memory_ns_insert, memory_ns_search_single, memory_ns_search_cross,
memory_ns_list, memory_ns_export
```

### Phase 4: Capability + Proof Integration (follow-on PR)

Integrate with `ruvector-capgated` (ADR-268): each namespace registers a
`CapMask`; cross-namespace search gates per-namespace access.  Wire with
`ruvector-proof-gate` (ADR-227): cross-namespace writes require witness log entry.

---

## Benchmark Evidence

Measured on 6 000 vectors (8 × 750), 128 dims, 200 queries, M=16,
ef_construction=200, ef_search=64, k=10.

### Single-Namespace Search

| Variant        | Mean(µs) | p50(µs) | p95(µs) |   QPS | Recall@10 |
|----------------|----------|---------|---------|-------|-----------|
| GlobalFlat     |   4390.2 |    4364 |    4545 |   228 |     97.4% |
| **Partitioned**|  **201.8**| **189** | **303** |**4955**|   **96.3%** |
| HierarchicalNS |    184.4 |     170 |     304 |  5422 |     96.2% |

### Cross-Namespace Search

| Variant        | Mean(µs) | p50(µs) | p95(µs) |   QPS | Recall@10 | Memory |
|----------------|----------|---------|---------|-------|-----------|--------|
| GlobalFlat     |    300.6 |     298 |     350 |  3327 |     42.7% | 4988KB |
| **Partitioned**| **1446.1**|**1424**|**1633** | **692**|  **97.5%**|**4779KB**|
| HierarchicalNS |    691.1 |     688 |     746 |  1447 |     52.6% | 4797KB |

**Key finding**: GlobalFlat cross-NS recall of 42.7% is consistent with
SIGMOD'24 (ACORN paper) measurements of post-filter degradation at ~12.5%
namespace selectivity.  Partitioned avoids this entirely.

---

## Failure Modes

| Failure Mode | Trigger | Response |
|--------------|---------|----------|
| Very small namespace (< 20 vectors) | Poor graph quality, low recall | Fallback to brute-force for N < 50 |
| Namespace count explosion (> 100) | Cross-NS sweep becomes O(100 × ns_latency) | Enable HierarchicalNS with learned route_k |
| Centroid router staleness | Mass inserts without centroid rebuild | Rebuild centroid every 100 inserts or on explicit flush |
| Memory exhaustion | Too many namespaces, each with large HNSW | Namespace eviction policy (LRU) + RVF snapshot |
| Cross-NS data leakage | Bug in capability check | Defense-in-depth: capgated mask verification before any cross-NS call |

---

## Security Considerations

Namespace partitioning is a **performance boundary**, not a security boundary.
Security is provided by the combination of:
1. **Per-namespace `CapMask`** (ruvector-capgated, ADR-268): querier must hold
   the capability mask required by the target namespace.
2. **Proof-gated cross-NS inserts** (ruvector-proof-gate, ADR-227): writing to
   another agent's namespace requires a witness log entry.
3. **Audit logging**: all cross-NS queries should log {querier_id, target_ns,
   timestamp, k, recall_estimate} for compliance.

Do not use namespace boundaries alone as a data isolation guarantee in
multi-tenant deployments.  Always layer capgated access control on top.

---

## Migration Path

1. Applications using a global HNSW with namespace metadata can migrate by
   iterating all vectors and re-inserting via `Partitioned::insert(ns, id, vec)`.
2. Existing `.rvf` bundles can be mounted as individual namespaces.
3. The `GlobalFlat` variant remains available behind a feature flag for
   applications that explicitly want single-index cross-NS performance.
4. No breaking change to existing `ruvector-core` APIs; the `NamespacedIndex`
   trait is additive.

---

## Open Questions

1. **What is the right cross-NS parallelism strategy?**  Rayon data parallelism
   vs Tokio async vs `std::thread` — each has tradeoffs for WASM compat.
2. **How should HierarchicalNS adapt route_k per query?**  Query entropy vs
   namespace centroid similarity spread — need experiments.
3. **Should namespace boundaries align with RVM coherence domains?**  The
   ruvector-coherence-hnsw ADR-240 defines coherence over single indexes; does
   coherence gating survive namespace boundaries?
4. **Maximum useful namespace count?**  Beyond ~100 namespaces with sequential
   sweep, the HierarchicalNS router needs to be a learned model, not centroid
   distance.
5. **Namespace merge semantics?**  When two agents merge their working context,
   should their HNSWs be merged (expensive but accurate) or just added to the
   same cross-NS pool (cheap but may miss inter-agent neighbours)?
