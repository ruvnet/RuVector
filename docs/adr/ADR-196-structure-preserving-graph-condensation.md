---
adr: 196
title: "Structure-Preserving Graph Condensation (ruvector-graph-condense)"
status: accepted
date: 2026-06-07
authors: [ruvnet, claude]
related: [ADR-197]
tags: [graph, condensation, coarsening, min-cut, gnn, ruview, worldgraph, hnsw]
---

# ADR-196 — Structure-Preserving Graph Condensation

## Status

**Accepted (implemented).** Crate `crates/ruvector-graph-condense` landed on
branch `claude/graph-condensation-ruvector-lVAKm`. ADR-197 covers the
differentiable min-cut loss added on top.

## Context

We want to shrink large feature graphs (a graph plus a per-node embedding and an
optional class label) into a much smaller graph that a downstream consumer —
GNN training, edge inference, or RuView's `WorldGraph → OccWorld` retraining
pipeline — can use in place of the original. Two bodies of evidence shaped the
decision:

### 1. The SOTA literature (graph condensation, 2022–2026)

The published field — GCond (gradient matching), DosCond (one-step), GCDM
(distribution matching), SFGC (structure-free trajectory matching), SGDD
(graphon/Laplacian-Energy-Distribution), GEOM (curriculum trajectories), GC-SNTK
(kernel ridge regression), GDEM (eigenbasis), DisCo (disentangled, scales to
111M nodes) — defines **condensation** as *synthesising a small fake graph* by
optimising a bi-level learning objective so a GNN trained on the synthetic graph
matches one trained on the original. That paradigm is:

- **Expensive** — bi-level optimisation, often second-order, hard to scale past
  ~1M nodes; "lossless" results (GEOM) need 1–5% ratios and banks of expert
  trajectories.
- **Supervised** — requires labels `Y'`.
- **Provenance-destroying** — a condensed node is synthetic; the mapping back to
  real nodes is intentionally discarded. This breaks audit, explainability, and
  link-back.

The surveys (arXiv:2401.11720, IJCAI'24 arXiv:2402.03358) and benchmarks (GC4NC,
GC-Bench) explicitly flag as **under-explored or unpublished**: community
detection (not k-means) as a structural prior, min-cut/modularity objectives in
the condensation loss, condensation of temporal/streaming graphs, and
condensation co-designed for edge deployment. The closest training-free analogs
are CGC (clustering, 2025) and GCTD (tensor decomposition, 2025).

### 2. The RuVector / RuView substrate

`ruvector-mincut` already ships the relevant primitives with default features:
`DynamicGraph` (streaming insert/delete/update), `CommunityDetector` and
`GraphPartitioner` (recursive global min cut), `ClusterHierarchy`, and an exact
`MinCutBuilder`. RuView (ruvnet/RuView) consumes RuVector's mincut/HNSW/GNN/RVF
primitives and records `WorldGraph` JSON snapshots that feed an OccWorld
world-model retrainer — but has **no graph condensation anywhere**, giving this
work a concrete downstream consumer.

## Decision

Add a new crate, `ruvector-graph-condense`, implementing **training-free,
structure-preserving, provenance-retaining** graph condensation built on
`ruvector-mincut`. Concretely this is closer to **coarsening with synthetic
representatives** than to GCond-style condensation, and we say so plainly:

- Partition the graph into structural **regions**.
- Collapse each region to a `CondensedNode { centroid, weight,
  class_distribution, coherence, representative (medoid), members }`. `members`
  is retained — the original↔condensed mapping survives.
- Rebuild **super-edges** from the *original* graph's boundary edges, so the
  condensed topology reproduces the source cut structure by construction rather
  than by training.

### Region-detection methods (`CondenseMethod`)

| Method | Mechanism | Reduction | Cost | When to use |
|---|---|---|---|---|
| **WeakBoundary** (default) | remove edges `< rel·mean_weight`, then union-find connected components | reliable when weights have contrast | linear (single pass) | general default; weighted graphs |
| MinCutCommunity | `ruvector_mincut::CommunityDetector` (recursive global min cut) | graph-dependent | **super-linear** | dense clusters + sharp bottlenecks only |
| Partition | `ruvector_mincut::GraphPartitioner` bisection | best-effort | super-linear | fixed region budget on clustered graphs |
| ConnectedComponents | components only | structural | linear | baseline / pre-separated graphs |
| DiffMinCut | trained soft assignment (see ADR-197) | `K`-bounded | iterative GD | learned cut-preserving regions |

The **default is `WeakBoundary`** because of an empirical finding during
implementation: recursive *global* min cut (`CommunityDetector`/`GraphPartitioner`)
**degenerates to singleton-peeling** — it shaves off the single lowest-degree
boundary vertex each step — on graphs without sharp bottlenecks, giving ~N
regions and zero reduction. This is the classic reason the community-detection
literature uses modularity/conductance, not raw min cut. We keep the engine
methods available (they *are* the literal min-cut-engine integration and work on
clearly-bottlenecked graphs) but document the degeneracy and do not default to
them.

### Quality metrics (retrain-free)

`metrics::evaluate` returns node/edge reduction ratios, `intra_weight_ratio`
(fraction of edge weight kept inside regions), mean `coherence`, and weighted
`label_purity`. `metrics::cut_inflation` (opt-in, solves an exact min cut on both
graphs) reports `mincut(condensed)/mincut(source)`: `1.0` means the source's
global min cut survives coarsening exactly.

### Streaming

`StreamingCondenser` buffers edges/features into a `DynamicGraph` and
re-condenses lazily (on dirty read or every `rebuild_interval` mutations). This
is **lazy re-condensation, not sublinear incremental region surgery** — an
honest amortisation for growing graphs (e.g. a WorldGraph as it accumulates),
with true incremental updates left as future work.

## Consequences

**Positive**
- Fast: `WeakBoundary` condenses ~2048 nodes in ~4 ms (benchmarked); linear scaling.
- Deterministic, label-optional, dependency-light (only `ruvector-mincut` + serde/rand/thiserror).
- Interpretable: every super-node carries its `members` and a `coherence` score.
- Cuts preserved by construction; `cut_inflation` quantifies fidelity.
- Reuses the existing min-cut engine rather than reimplementing graph algorithms.

**Negative / limitations**
- This is *not* accuracy-matched GCond-style condensation; it trades peak
  downstream GNN accuracy for speed, determinism, and provenance. We do not
  claim accuracy retention numbers — no GNN-retrain evaluation is in scope.
- Engine methods (MinCutCommunity/Partition) are super-linear (~24 s at 96 nodes,
  measured) and prone to peeling; usable only on small/well-structured graphs.
- `WeakBoundary` needs weight contrast; on near-uniform weights it degrades to
  ConnectedComponents (documented).
- Every graph vertex must have a feature vector, or condensation errors
  (`MissingFeature`).

## Alternatives considered

1. **Implement GCond/SFGC-style learned condensation.** Rejected for v1:
   requires an autodiff stack and GNN training loop, is expensive, supervised,
   and destroys provenance. (ADR-197 adds the differentiable *min-cut* angle,
   which is the novel, lighter-weight slice of this.)
2. **Put condensation inside `ruvector-mincut` or `ruvector-graph`.** Rejected:
   condensation is a distinct bounded context with its own data model; the
   workspace convention is one crate per capability.
3. **Default to an engine method (MinCutCommunity/Partition).** Rejected after
   benchmarks showed singleton-peeling and super-linear cost.

## References

- Surveys: arXiv:2401.11720 (Graph Condensation: A Survey), arXiv:2402.03358
  (Graph Reduction, IJCAI'24); benchmarks GC4NC (arXiv:2406.16715), GC-Bench
  (arXiv:2407.00615).
- Methods: GCond (ICLR'22), SFGC (NeurIPS'23), SGDD (NeurIPS'23), GEOM (ICML'24),
  GDEM (ICML'24), DisCo (2024), CGC (2025), GCTD (WSDM'26).
- Substrate: `ruvector-mincut` (`DynamicGraph`, `CommunityDetector`,
  `GraphPartitioner`, `MinCutBuilder`); RuView (github.com/ruvnet/RuView).
- Example: `crates/ruvector-graph-condense/examples/worldgraph.rs` — a RuView
  `WorldGraph → condense → OccWorld` demo (600 observations → 12 event
  summaries at 50× reduction, 100% activity purity, cut preserved).
- **Accuracy validation** (`gnn_eval` module + `examples/accuracy_eval.rs` +
  `tests/accuracy.rs`): a gradient-checked 2-layer GCN runs the field's standard
  protocol (train on condensed, test on original held-out nodes). On a controlled
  unweighted node-classification task, `DiffMinCut` condensing 360 → 18 nodes
  (20×) reaches **100% accuracy retention**. Honest scope: controlled synthetic
  data, not Cora/Citeseer; `WeakBoundary` needs weight contrast (it collapses on
  uniform-weight graphs, which is why the accuracy path uses `DiffMinCut`).
- **WASM deployment**: `crates/ruvector-graph-condense-wasm` exposes the
  condenser to JS/browser/edge (`wasm32-unknown-unknown`, 667 KB release before
  wasm-opt). The `parallel` (Rayon) feature is default-on for native and off for
  wasm (no threads).
- ADR-197 (differentiable min-cut loss).
