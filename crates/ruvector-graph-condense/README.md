# RuVector Graph Condense

[![Crates.io](https://img.shields.io/crates/v/ruvector-graph-condense.svg)](https://crates.io/crates/ruvector-graph-condense)
[![Documentation](https://docs.rs/ruvector-graph-condense/badge.svg)](https://docs.rs/ruvector-graph-condense)
[![License](https://img.shields.io/crates/l/ruvector-graph-condense.svg)](LICENSE)
[![GitHub](https://img.shields.io/badge/GitHub-ruvnet%2Fruvector-blue?logo=github)](https://github.com/ruvnet/ruvector)
[![ruv.io](https://img.shields.io/badge/ruv.io-AI%20Infrastructure-orange)](https://ruv.io)

**Training-free, structure-preserving, provenance-retaining graph condensation.**

*Collapse a large feature graph into a small synthetic graph of super-nodes while preserving its cut structure — plus a differentiable min-cut loss.*

---

## Why This Matters

Graph condensation shrinks a graph + per-node embeddings (+ optional labels) into a much smaller graph that downstream tasks can still reason over. The published SOTA — GCond, SFGC, GEOM, SGDD — *synthesises* a fake graph via expensive, supervised bi-level gradient/distribution/trajectory matching, and **discards the node→original mapping**.

`ruvector-graph-condense` takes the complementary, training-free route the 2024–2026 condensation surveys flag as under-explored:

- **Min-cut community structure as the condensation prior** (not k-means).
- **Cuts preserved by construction** — boundary edges become weighted super-edges; `metrics::cut_inflation` quantifies fidelity.
- **Provenance retained** — every `CondensedNode` keeps its `members`, so each super-node is auditable and explainable.
- **A differentiable min-cut *loss*** (`diffcut`, MinCutPool-style relaxed normalized cut + orthogonality) — analytic gradients, gradient-checked across K=2,3,4 to <1e-5.

Built on the dynamic min-cut engine [`ruvector-mincut`](https://crates.io/crates/ruvector-mincut).

## Quick Start

```rust
use ruvector_graph_condense::{CondenseConfig, GraphCondenser, NodeFeatures};
use ruvector_mincut::DynamicGraph;

// Build a graph (insert_edge returns a Result; &self — it is concurrent).
let graph = DynamicGraph::new();
let _ = graph.insert_edge(0, 1, 1.0);
let _ = graph.insert_edge(1, 2, 1.0);
let _ = graph.insert_edge(2, 3, 0.1); // weak boundary edge

// Per-vertex embeddings (+ optional labels): NodeFeatures::new(dim, num_classes).
let mut features = NodeFeatures::new(2, 1);
for v in 0..4u64 {
    features.set(v, vec![v as f32, 0.0], 0).unwrap();
}

let condenser = GraphCondenser::new(CondenseConfig::default()); // WeakBoundary, 0.5
let condensed = condenser.condense(&graph, &features).unwrap();

for node in &condensed.nodes {
    // Each super-node keeps the original vertices it came from (provenance).
    println!("super-node {:?} <- members {:?}", node.representative, node.members);
}
```

## Region Methods (`CondenseMethod`)

| Method | Notes |
|--------|-------|
| `WeakBoundary` (default) | Linear-time union-find on weak edges. ~4 ms @ 2048 nodes. |
| `MinCutCommunity` / `Partition` | Delegate to the min-cut engine. Structure-aware on graphs with sharp bottlenecks; documented best-effort otherwise. |
| `ConnectedComponents` | Cheap baseline — one region per component. |
| `DiffMinCut` | Differentiable, *trained* assignment (opt-in). |

## Honest Limitations

- The recursive **global min-cut engine methods degenerate to singleton-peeling** on graphs without sharp bottlenecks and are super-linear (~24 s @ 96 nodes) — which is why the linear-time `WeakBoundary` is the default.
- **`DiffMinCut` is K-sensitive** (known MinCutPool finickiness): it recovers small/dense graphs but underperforms `WeakBoundary` at large K. Momentum + unit-scale init help, but there is no convergence guarantee.
- This is structure-preserving **coarsening-condensation** (keeps provenance) — **not** accuracy-matched GCond-style condensation; no GNN-retrain accuracy numbers are claimed.

See **ADR-196** (structure-preserving condensation) and **ADR-197** (differentiable min-cut loss) for the full design and findings.

## License

MIT © [ruv.io](https://ruv.io)
