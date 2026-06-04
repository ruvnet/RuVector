---
adr: 196
title: "SepRAG — CCH-Inspired Separator-Tree Retrieval for Hybrid Vector + Graph Memory"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-002, ADR-193, ADR-194, ADR-195, ADR-197, ADR-198, ADR-199]
tags: [ruvector, retrieval, cch, contraction-hierarchies, graph-rag, mincut, jtree, hnsw, diskann, knn]
---

# ADR-196 — SepRAG: CCH-Inspired Separator-Tree Retrieval

## Status

**Proposed → empirically NO-GO for embedding retrieval (2026-06-04).** Keystone design
ADR. Depends on the navigation-graph and ordering decisions in [ADR-197], the metric
layer in [ADR-198], and is validated by the benchmark harness in [ADR-199].

Prototyped in `crates/ruvector-seprag` (M0 + M1). The separator-tree **query** algorithm
is correct (exact recall) and prunes ~100% of search space — but CCH **full contraction**
blows up on embedding/citation backbones (high treewidth; see [ADR-199] Empirical
Outcome). The design's stated edge — *constrained / relational* retrieval over a sparse
structured backbone, not pure embedding kNN — remains the only plausible niche and is
unvalidated against HNSW. Treat the contraction-based core as not-fit-for-embedding-kNN.

## Context

Customizable Contraction Hierarchies (CCH) achieve ~35,000x speedups over Dijkstra
on continental road networks by reducing a million-vertex search to ~1,200–1,450
vertices. The speedup comes from three ideas:

1. **Nested-dissection ordering** — recursively split the graph by *balanced
   separators*; rank separator vertices highest.
2. **Contraction + shortcuts** — eliminating a vertex makes its higher-ranked
   neighbors a clique (fill-in). The shortcut *set* depends only on the order and
   topology, **not on edge weights**.
3. **Elimination-tree query** — the search space from any vertex is exactly its
   ancestors in the elimination tree (a contiguous array walk), so point-to-point
   and k-NN queries touch a tiny, bounded set of vertices.

Two grounding papers: Buchhold & Wagner (2021), *Nearest-Neighbor Queries in
Customizable Contraction Hierarchies* (the separator-tree k-NN algorithm); and
Bläsius et al. (arXiv:2502.10519), the modern CCH implementation survey.

**Why this matters for RuVector.** A repository audit found that RuVector already
ships ~70% of the required machinery — it has simply never been composed as a
retrieval hierarchy:

| CCH primitive | Existing crate / module |
|---|---|
| Balanced separators / nested dissection | `ruvector-mincut::{expander, cluster, sparsify, algorithm}` |
| Elimination / junction tree, leveled hierarchy | `ruvector-mincut::jtree` (`hierarchy`, `level`, `coordinator`) |
| Dynamic tree operations (incremental updates) | `ruvector-mincut::linkcut` |
| Path/cut duality, customization-style sweeps | `ruvector-solver::{bmssp, forward_push, backward_push, simd}` |
| Effective-resistance edge importance | `ruvector-sparsifier` |
| Hybrid vector+graph retrieval surface | `ruvector-graph::hybrid::{semantic_search, rag_integration, graph_neural, vector_index}` |
| Entry-point search | `ruvector-diskann` (Vamana), `ruvector-hyperbolic-hnsw` |
| Quantized distance evaluation | `ruvector-rabitq` |
| Learned metric + rerank | `ruvector-gnn`, `ruvector-attn-mincut` |

**The problem being solved.** Flat ANN (HNSW/DiskANN) is excellent at pure-semantic
top-k cosine, but degrades on the queries RuVector actually wants to be great at:
*constrained, relational, multi-hop, and re-weightable* retrieval (Graph-RAG).
Post-filtering a flat index is wasteful, results are semantically scattered (no
topic coherence), and any change to the relevance metric forces an index rebuild.

## Decision

**Adopt a CCH-inspired retrieval layer ("SepRAG") that complements — does not
replace — HNSW/DiskANN.** Division of labor:

- **HNSW/DiskANN answers "where am I?"** — find entry leaves in O(log n). Do not
  reinvent this; flat ANN is the right tool for landing near the query.
- **SepRAG answers "what is structurally near here, under these constraints, by
  this learned metric?"** — separator-tree branch-and-bound that prunes entire
  semantic regions in O(1) per cell, supports subtree filter predicates, and uses a
  re-customizable cost.

### Architecture (three phases, mirroring CCH)

```
Phase 1  ORDER      (metric-independent)  → vertex order + shortcut SET + elim tree
Phase 2  CUSTOMIZE  (metric-dependent, ~s) → shortcut WEIGHTS for current metric   [ADR-198]
Phase 3  QUERY      (metric-fixed, ~µs–ms) → separator-tree k-NN with pruning
```

Phase 1 + the navigation graph it runs on are specified in [ADR-197]. Phase 2 (the
self-learning metric loop) is [ADR-198].

### Phase-3 query algorithm (the workhorse)

```
fn knn(s, k, w, sep_tree, poi_buckets) -> TopK:
    d_anc = upward(s, w, elim_parent)          # d(s -> x) for ancestors/separators
    topk  = BoundedHeap(k)                      # exposes delta_k = current k-th best
    PQ    = MinHeap()                           # ordered by admissible lower bound
    PQ.push(lb=0, node=sep_tree.root)
    while PQ not empty:
        (lb, node) = PQ.pop()
        if lb > topk.delta_k(): break           # global prune; nothing better remains
        for p in poi_buckets[node]:             # POIs attached at this separator node
            for x in node.separator_vertices:
                topk.offer(p, d_anc[x] + downdist[x][p])
        for child in node.children:
            lb_child = min_{x in child.boundary} d_anc[x]   # separator sits ABOVE cell
            if lb_child <= topk.delta_k() and child.may_satisfy(filter):
                PQ.push(lb_child, child)        # else: whole subtree pruned in O(1)
    return topk
```

Admissibility of the bound rests on the separator-above-cell property: any path from
`s` into a cell must pass through that cell's separator, so `d(s -> cell) >=
d(s -> its separator)`. Region-level pruning is where the search-space reduction comes
from.

### Three techniques layered on the same topology

1. **SepRAG k-NN** (above) — hybrid graph-distance top-k with region pruning.
2. **Hierarchical hybrid filtering** — query constraints (tenant, recency, relation
   type, entity reachability) become *subtree predicates*. Semantic lower-bound
   pruning and constraint pruning run in the same branch-and-bound, so structured +
   semantic + filtered retrieval is a single traversal. This is SepRAG's decisive
   advantage over flat-index post-filtering.
3. **Multi-metric quiver** — one topology, several cheap customizations
   (`semantic`, `recency`, `trust`, `task`, on-the-fly blends). Per-query lens
   selection at near-zero marginal cost. Detailed in [ADR-198].

### Composition pipeline

```
query ─► HNSW/DiskANN top-m (entry leaves)
      ─► SepRAG separator-tree branch & bound  (metric from ADR-198, filters as predicates)
      ─► ruvector-attn-mincut rerank           (cut-as-attention gating)
      ─► top-k memories + elimination-tree path (free provenance / explanation)
```

## Consequences

**Positive.**
- Reuses existing, tested crates; this is composition, not green-field.
- Region pruning + subtree filters target the exact queries flat ANN handles poorly.
- The elimination-tree path is a free provenance trail ("why was this retrieved").
- Metric updates do not rebuild topology (the self-learning payoff — [ADR-198]).

**Negative / risk.**
- **Expander risk (decisive).** Dense kNN graphs have good expansion → large
  separators → shortcut blowup → CCH collapses. Mitigation is the navigation-graph
  design in [ADR-197]; the risk is *measured* (not argued) in [ADR-199] via the
  shortcut-blowup ratio `|G+| / |G_nav|`.
- Preprocessing (Phase 1 ordering) is superlinear; viable only because it is
  metric-independent and amortized over all future customizations.
- Graph-distance k-NN is not Euclidean k-NN — recall must be defined against a
  hybrid-distance oracle, not cosine top-k (see [ADR-199]).

**Neutral.**
- SepRAG is additive behind a feature gate; HNSW/DiskANN paths are untouched.

## Alternatives considered

- **Replace HNSW with CCH for plain top-k.** Rejected — embedding graphs are too
  expander-like; HNSW wins on pure cosine. CCH's edge is constrained/relational.
- **Plain CH (weights baked into order).** Rejected — every relevance update would
  re-run the expensive ordering. Metric-independence ([ADR-198]) is the whole point.
- **HNSW post-filtering for constraints.** Rejected as the *primary* path — wasteful
  and incoherent; kept only as a baseline in [ADR-199].

## Open questions

Carried into [ADR-199], to be answered empirically by the benchmark corpus rather
than guessed: dominant query shape, real graph sparsity, GNN metric-update cadence,
whether `jtree` already runs on data graphs, and exact vs approximate recall target.
