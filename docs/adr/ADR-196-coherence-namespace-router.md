---
adr: 196
title: "Coherence-Gated Multi-Tenant Vector Namespace Router"
status: accepted
date: 2026-06-07
authors: [ruvnet, claude-flow]
related: [ADR-010, ADR-042, ADR-116, ADR-193]
tags: [namespace, multi-tenant, coherence, routing, witness-log, rvf, mcp, agent-memory, nightly-research]
---

# ADR-196 — Coherence-Gated Multi-Tenant Vector Namespace Router

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-07-coherence-namespace-router` as
`crates/ruvector-namespace-router`. All 16 unit tests pass; all three variants achieve recall@10 = 1.000
(acceptance threshold: ≥ 0.99). Build is green with `cargo build --release -p ruvector-namespace-router`.

## Context

RuVector is deployed as a shared vector substrate in multi-agent scenarios — multiple ruFlo pipeline stages,
MCP tool surfaces, and RVF-domain agents all querying a single vector store instance. As of June 2026,
RuVector has no mechanism for:

1. **Retrieval isolation**: preventing Agent A's query from returning vectors belonging to Agent B's namespace.
2. **Semantic routing**: routing queries to the *most semantically relevant* namespace rather than requiring
   the caller to enumerate candidate namespaces.
3. **Access auditing**: recording which agent retrieved which cross-namespace vectors and under what conditions.

Production vector databases solve this with coarse mechanisms (separate collections per tenant in Qdrant,
partition keys in Milvus), but none expose the *semantic coherence* between namespaces as a first-class
routing signal.

The closest RuVector work is `ruvector-mincut` (graph partitioning) and `ruvector-coherence` (coherence
scoring). This ADR introduces `ruvector-namespace-router` as the routing layer that connects these components
to retrieval-time isolation policy.

## Decision

We introduce `crates/ruvector-namespace-router` implementing a `NamespaceIndex` trait with three variants:

### Variant 1: FlatIsolated (baseline)

Each namespace is an independent `Vec<Entry>`. Search performs a full linear scan within the requested
namespace only. Zero cross-namespace visibility. No centroid index. O(N_ns) per query.

**When to use:** Maximum isolation, small namespace sizes, WASM/edge environments, regulatory compliance
requiring strict per-tenant isolation.

### Variant 2: CentroidRouted (alternative A)

Each namespace maintains an incrementally updated centroid (Welford algorithm). Before scanning, namespaces
are ranked by centroid-to-query distance; only the `probe` closest namespaces are scanned. Enables
opt-in cross-namespace retrieval while still performing exact scan (recall = 1.0) within probed namespaces.

**When to use:** Agent memory federation where related ruFlo stages share context; semantic cross-namespace
discovery; `probe=1` gives FlatIsolated behaviour with centroid overhead.

### Variant 3: CoherenceGated (alternative B)

Extends CentroidRouted with a coherence threshold τ. A foreign namespace contributes results only if
`coherence(source_ns, foreign_ns) ≥ τ`, where coherence is defined as:

```
coherence(a, b) = exp(−L2(centroid_a, centroid_b) / (spread_a + spread_b))
```

Every cross-boundary result appended to an embedded `WitnessLog` for audit. Zero cross-boundary events
when namespaces are well-separated (as in the benchmark).

**When to use:** Enterprise RAG with compliance requirements; proof-gated retrieval pipelines;
ruFlo workflows needing selective memory federation; MCP memory tools with auditable access.

### Common `NamespaceIndex` trait

```rust
pub trait NamespaceIndex {
    fn insert(&mut self, ns: NamespaceId, id: VectorId, vector: Vec<f32>) -> Result<(), String>;
    fn search(&self, ns: NamespaceId, query: &[f32], k: usize) -> Vec<SearchResult>;
    fn namespace_count(&self) -> usize;
    fn total_vectors(&self) -> usize;
    fn memory_bytes(&self) -> usize;
}
```

The `NamespaceId` type alias (`u32`) is intentionally narrow to support dense arrays and WASM-safe
representation.

## Consequences

### Positive

- **Isolation by default**: FlatIsolated prevents cross-namespace leakage with no configuration.
- **Gradual opt-in**: Callers migrate from FlatIsolated → CentroidRouted → CoherenceGated as their
  governance requirements mature.
- **Witness log**: Every cross-boundary access event is recorded, enabling post-hoc audit without
  modifying the query path.
- **No external dependencies**: `ruvector-namespace-router` depends only on `rand` (test data); the
  core logic is pure Rust.
- **WASM compatible**: FlatIsolated compiles to WASM32 without modification.

### Negative / limitations

- **Linear scan only**: Current variants perform O(N_ns) scan per namespace. Production use requires
  a HNSW-backed namespace variant (future work).
- **In-process WitnessLog**: Log is lost on process exit unless serialized. Not suitable for
  distributed audit without persistence integration.
- **Centroid approximation**: Welford-updated centroids are accurate for stationary distributions;
  for concept-drifting namespaces, centroids lag behind the true distribution.
- **No dynamic τ**: CoherenceGated's threshold is set at construction; ruFlo workflows that need to
  adjust τ per stage must reconstruct the index.

## Alternatives Considered

### A — Namespace via metadata filter (ACORN-style)

Store all vectors in a single HNSW index with a `namespace_id` metadata field. Apply predicate filter
during search. Implementation: extend `ruvector-acorn` with namespace predicates.

**Rejected:** ACORN-style filtered search still traverses the full HNSW graph for graph navigation,
with predicate checks only at result collection. This gives no isolation guarantee during graph walk —
a vector from a foreign namespace may be a graph neighbor that guides the walk to a result. True
isolation requires separate per-namespace data structures.

### B — Database-level isolation (separate crate instances)

Each namespace is a completely independent RuVector instance. Isolation is guaranteed by the process
boundary.

**Rejected:** Eliminates cross-namespace federation entirely. Operational overhead scales with
namespace count. Not suitable for 100+ agent namespaces.

### C — mincut-partition-derived namespaces

Use `ruvector-mincut` to partition the global vector graph, then assign namespace IDs to partition
membership. Each namespace corresponds to a natural cluster in the graph topology.

**Not rejected — future work.** The `ruvector-namespace-router` trait is compatible with this approach.
Partition IDs become namespace IDs; the coherence formula naturally reflects graph-structural separation.

## Implementation Plan

| Milestone | Action | Status |
|-----------|--------|--------|
| M1 | `NamespaceIndex` trait + FlatIsolated | ✅ Complete |
| M2 | CentroidRouted with Welford update | ✅ Complete |
| M3 | CoherenceGated with WitnessLog | ✅ Complete |
| M4 | HNSW namespace backend (per-namespace HNSW) | Future |
| M5 | Persistent WitnessLog (RVF binary format) | Future |
| M6 | Dynamic τ via `set_policy(ns, tau)` | Future |
| M7 | ruvector-mincut partition → namespace assignment | Future |
| M8 | MCP resource URI → NamespaceId mapping | Future |

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-namespace-router`, Intel Celeron N4020, rustc 1.94.1,
N=4,000 vectors (8 namespaces × 500), D=128, K=10, 1,600 queries.

| Variant | Insert (vecs/s) | Mean (µs) | p50 (µs) | p95 (µs) | Recall@10 | Memory (KB) | Accept |
|---------|----------------|-----------|----------|----------|-----------|-------------|--------|
| FlatIsolated | 3,037,157 | 78.79 | 76.17 | 96.19 | 1.000 | 2,031.2 | PASS |
| CentroidRouted | 5,222,566 | 81.05 | 78.08 | 98.40 | 1.000 | 2,035.2 | PASS |
| CoherenceGated | 3,045,712 | 84.99 | 81.45 | 105.08 | 1.000 | 2,035.2 | PASS |

**Key observations:**
- All variants achieve perfect recall (exact linear scan within probed namespaces).
- CoherenceGated adds ~7.8% latency overhead over FlatIsolated (coherence scoring + witness infrastructure).
- CentroidRouted is 72% faster for inserts (Welford update is sequential and cache-friendly).
- Well-separated namespaces (τ=0.30, inter-namespace coherence < 0.30): zero cross-boundary events.

## Failure Modes

| Mode | Impact | Detection | Mitigation |
|------|--------|-----------|------------|
| Centroid staleness | Wrong namespace ranked; correct results missed | Monitor coherence score time series | Periodic full centroid recompute |
| τ too low | Cross-namespace leakage | Spike in WitnessLog event rate | Raise τ or switch to FlatIsolated |
| τ too high | No federation; agents can't share knowledge | Zero cross-boundary events when expected | Lower τ; add explicit allow-list |
| WitnessLog overflow | Unbounded memory growth in high-traffic sessions | Monitor log size | Flush to RVF on size threshold |
| Linear scan bottleneck | High latency for large namespaces (N_ns > 100K) | Latency regression in monitoring | Switch to HNSW namespace backend (M4) |

## Security Considerations

**What CoherenceGated prevents:**
- Accidental cross-namespace retrieval when namespaces are semantically distinct (τ enforcement)
- Unlogged cross-boundary access (WitnessLog records all events)

**What CoherenceGated does not prevent:**
- Deliberate τ=0 policy by a privileged caller
- Centroid poisoning (adversarial inserts to manipulate coherence scores)
- WitnessLog tampering by the process owner

**Recommended hardening for production:**
1. Set τ through a signed `NamespacePolicy` object (ruvector-verified integration)
2. Persist WitnessLog entries with HMAC signatures
3. Limit insert permissions per namespace via MCP capability tokens

## Migration Path

| From | To | Steps |
|------|----|-------|
| No isolation (single flat index) | FlatIsolated | Tag all vectors with a namespace ID; split into per-namespace `Vec<Entry>` |
| FlatIsolated | CentroidRouted | Rebuild with CentroidRouted; centroids computed incrementally on insert |
| CentroidRouted | CoherenceGated | Replace CentroidRouted with CoherenceGated; choose initial τ (start with 0.5) |
| CoherenceGated | HNSW namespace backend | Replace inner Vec with ruvector-core HNSW; trait interface unchanged |

## Open Questions

1. **What is the right default τ?** 0.30 was chosen empirically. A principled method (e.g., train τ
   on historical cross-namespace retrieval benefit) is needed.
2. **How does coherence score interact with quantization?** When namespaces use RaBitQ-quantized vectors,
   centroids are computed from quantized values. Does this degrade coherence accuracy enough to matter?
3. **Should WitnessLog entries be ZK-provable?** If `ruvector-verified` adds ZK proof support, witness
   entries could prove "this access was permitted by the policy at time T" without revealing the vectors.
4. **Namespace ID namespace**: Should NamespaceId be a u32 (current) or a string (MCP URI)? String IDs
   are more ergonomic for MCP integration but require a registry lookup.
