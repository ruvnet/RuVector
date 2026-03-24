# ADR-124: Dynamic MinCut with Partition Cache

## Status
Accepted — Tier 1 shipped, Tiers 2-3 in progress

## Context
The pi.ruv.io brain server's `/v1/partition` endpoint runs exact Stoer-Wagner MinCut on the knowledge graph. With 2,090+ nodes and 971K+ edges, this computation exceeds Cloud Run's 300s timeout and the MCP SSE transport's 60s tool call timeout.

PR #287 (ADR-117) introduced a canonical source-anchored MinCut with deterministic hashing and WASM FFI bindings, providing the foundation for all three tiers.

## Decision

### Three-Tier MinCut Architecture

#### Tier 1: Exact Engine (Shipped)
- **Stoer-Wagner** global MinCut — O(V*E), exact solution
- **Source-anchored canonical cut** — deterministic via lexicographic tuple `(λ, first_separable_vertex, |S|, π(S))`
- **SHA-256 cut hash** for witness chain compatibility (FIPS 180-4)
- **Dinic's max-flow** for minimum s-t cut computation
- **Fixed-point arithmetic** (32.32 `FixedWeight`) for determinism
- **WASM FFI** bindings: `canonical_init()`, `canonical_compute()`, `canonical_get_hash()`

#### Tier 2: Tree Packing Fast Path (In Progress)
- **Gomory-Hu tree construction** — builds a flow-equivalent tree using Dinic's max-flow
- **Fast global MinCut** from tree — minimum edge in Gomory-Hu tree gives MinCut in O(V) after construction
- **Complexity**: O(V^2 log V) vs O(V*E) for Stoer-Wagner
- **API**: `SourceAnchoredConfig::fast()` method for tree-packing mode
- Target: make `/v1/partition` feasible for graphs up to ~500K edges

#### Tier 3: Dynamic/Incremental MinCut (In Progress)
- **DynamicMinCut struct** — wraps `SourceAnchoredMinCut` with incremental updates
- **Edge insertion**: if new edge doesn't cross current cut, cut value unchanged; otherwise recompute only affected s-t cut
- **Edge deletion**: if removed edge not in cut set, unchanged; otherwise recompute
- **Batch updates**: `apply_batch(additions, removals)` for bulk operations
- **Epoch tracking**: increment on each mutation, track changed edges since last full recomputation
- **Staleness detection**: after N incremental updates, trigger full recomputation to correct drift
- **Complexity**: O(V * sqrt(E)) amortized per update
- Target: real-time partition updates as new memories are added to the brain

### Partition Caching
- `cached_partition: Arc<RwLock<Option<PartitionResult>>>` in `AppState`
- **MCP `brain_partition`**: Returns cached compact partition (sub-millisecond)
- **REST `/v1/partition`**: Returns cache by default; `?force=true` recomputes
- **Large-graph guard**: Skip exact MinCut during training for graphs >100K edges

### Brain Server Integration
- Training cycles populate cache for small graphs (<100K edges)
- Scheduled `rebuild_graph` job handles large graphs asynchronously
- Dynamic MinCut (Tier 3) will enable real-time cache updates on memory insertion
- `?canonical=true` parameter uses source-anchored cut with witness-compatible hash

## Implementation

### Crate Structure (`ruvector-mincut`)
```
src/canonical/
├── mod.rs                    # Re-exports
├── source_anchored/
│   ├── mod.rs                # Tier 1: exact engine + canonical cut
│   └── tests.rs              # 503 lines of tests
├── tree_packing.rs           # Tier 2: Gomory-Hu tree fast path
└── dynamic.rs                # Tier 3: incremental MinCut
```

### Feature Flags
All behind `canonical` feature in `Cargo.toml`.

### WASM Bindings
```
src/wasm/canonical.rs
├── canonical_init/compute/get_result/get_hash    # Tier 1
├── tree_packing_init/compute                      # Tier 2
└── dynamic_init/add_edge/remove_edge/batch/query  # Tier 3
```

## Consequences
- MCP `brain_partition` returns instantly from cache instead of timing out
- Enhanced training cycle no longer blocks on MinCut for large graphs
- Tier 2 makes fresh partition computation feasible for medium graphs
- Tier 3 enables real-time partition updates as the brain grows
- All cuts are deterministic and witness-chain compatible via SHA-256

## Benchmark Results

### Tier 1: Canonical MinCut (ADR-117)

| Graph Type | Nodes | Time |
|-----------|-------|------|
| Cycle | 6 | 2.18 us |
| Cycle | 50 | 3.09 us |
| Complete | 10 | 2.61 us |
| Hash stability | 100 | 1.39 us |

### Before vs After (Brain Server)

| Metric | Before | After |
|--------|--------|-------|
| `brain_partition` via MCP | Timeout (>60s) | Sub-millisecond (cached) |
| `/v1/partition` REST | Timeout (>300s) | Sub-millisecond (cached) |
| Training cycle (971K edges) | 504 timeout | Completes (skips MinCut) |

### Tier 2-3 benchmarks pending implementation.
