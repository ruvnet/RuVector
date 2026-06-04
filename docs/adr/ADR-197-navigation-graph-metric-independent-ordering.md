---
adr: 197
title: "Navigation-Graph Construction & Metric-Independent Nested-Dissection Ordering"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-198, ADR-199]
tags: [ruvector, cch, nested-dissection, separators, mincut, jtree, diskann, hyperbolic, treewidth]
---

# ADR-197 — Navigation-Graph Construction & Metric-Independent Ordering

## Status

**Proposed — make-or-break question answered NO for embedding backbones (2026-06-04).**
Implements Phase 1 of [ADR-196]. This was correctly identified as the make-or-break
decision: the navigation graph's separator size determines viability. Measured outcome
([ADR-199] Empirical Outcome): the road-network control has small separators (elim height
≈ 3.5·√n), but **both embedding backbones — citation small-world and Euclidean α-pruned
feature kNN — have near-linear treewidth (elim height ≈ 0.5·n)**. The "expander risk"
flagged below is real and confirmed. Degree-bounding the backbone made it worse, not
better. The hyperbolic-backbone mitigation remains untested (needs genuine hyperbolic
embeddings).

## Context

CCH's speedup depends on **small balanced separators** (low treewidth). Road
networks have them naturally; arbitrary embedding kNN graphs generally do not — a
dense, well-connected kNN graph is *expander-like*, which is precisely the worst case
(good expansion ⇒ large separators ⇒ massive fill-in ⇒ no pruning).

So the central engineering question is **what graph to build the hierarchy on**. The
dense kNN graph is the wrong answer. The decision below selects a sparse, structured
"navigation graph" `G_nav` whose separator structure is favorable, then orders it
metric-independently using `ruvector-mincut`.

## Decision

### 1 — Navigation graph `G_nav`: sparse, structured, road-like

Build `G_nav` as the union of:

- **Knowledge-graph relation edges** (`ruvector-graph::Edge`, `RelationType`) —
  sparse, hierarchical, low-treewidth, the most "road-like" component.
- **A degree-bounded, diversified kNN graph** — RNG / α-pruned, exactly what
  `ruvector-diskann`'s Vamana already produces. Pruning is what *creates* separators;
  it removes the expander-inducing density.
- **(Optional) HNSW upper-layer skeleton** as a coarse long-range backbone.

Explicitly **do not** run nested dissection on the full dense kNN graph.

**Hyperbolic option (high-synergy).** `ruvector-hyperbolic-hnsw` embeds in hyperbolic
space, whose tree-like geometry yields naturally small separators. Building `G_nav`
in hyperbolic space is a first-class alternative backbone and is benchmarked head-to-head
in [ADR-199].

### 2 — Metric-independent ordering via nested dissection

Reuse `ruvector-mincut`:

```
fn nested_dissection_order(G_nav) -> (order, sep_tree):
    fn recurse(cell, parent_sep_node):
        if |cell| <= LEAF: assign ranks, attach leaf; return
        S        = balanced_separator(G_nav, cell)   # ruvector-mincut expander/cluster/sparsify
        (A, B)   = components(cell \ S)
        node     = sep_tree.add_separator(parent_sep_node, S)
        recurse(A, node); recurse(B, node)
        assign ranks to S AFTER A,B                  # separators ranked highest
    recurse(all_vertices, ROOT)
```

`ruvector-mincut::jtree` already produces a leveled hierarchical decomposition with
BMSSP integration and O(n^ε) updates — adapt it to emit a CCH vertex order and the
separator decomposition tree, rather than re-implementing nested dissection.

### 3 — Symbolic contraction → chordal graph + elimination tree (metric-free)

```
fn build_chordal(G_nav, order) -> (G_plus, elim_parent):
    for v in order (low -> high rank):
        Hi = { higher-ranked neighbors of v in G_plus }
        elim_parent[v] = argmin_rank(Hi)             # lowest higher neighbor
        make Hi a clique in G_plus                   # fill-in / shortcuts
```

The shortcut set is fixed here and reused across every metric customization
([ADR-198]).

### 4 — Cache-friendly layout

- **Relabel vertices to rank order** so the upward CSR rows are ascending and
  SIMD-comparable (`ruvector-solver::simd` can vectorize triangle relaxations).
- **Store the elimination tree in DFS post-order** so a vertex's ancestors occupy a
  near-contiguous band — the ancestor walk is a cache-friendly stride, not pointer
  chasing. Mirrors `jtree::level` bucketing.

```rust
pub struct CchTopology {           // metric-INDEPENDENT, built once
    n: u32,
    up_offsets: Vec<u32>,          // CSR, len n+1 (upward chordal graph)
    up_targets: Vec<u32>,          // ranks ascending within a row → SIMD-friendly
    elim_parent: Vec<u32>,         // rank-indexed; parent[root] = SENTINEL
    dfs_order:   Vec<u32>,         // elim tree in post-order (contiguous ancestors)
    sep_tree:    SepTree,          // separator decomposition (cells = vertex ranges)
}
```

### 5 — Incremental updates

New memories arrive continuously; rebuilding topology per insert is infeasible.
Use `ruvector-mincut::{jtree (O(n^ε) updates), linkcut}` for incremental contraction
on insert; re-order in periodic batches. Weight-only changes never touch topology —
they are pure re-customization ([ADR-198]).

## Consequences

**Positive.** Reuses `ruvector-mincut`/`jtree`/`linkcut` wholesale. The expander risk
is contained by construction (sparse backbone) and measured in [ADR-199].

**Negative.**
- Ordering is the heavy, superlinear step; only justified because it is amortized.
- Quality of separators (hence everything downstream) depends on the balanced-cut
  heuristics in `ruvector-mincut` — must be validated on real data, not assumed.
- Bounding contraction degree to cap fill-in trades exactness for blowup control and
  must be recall-tested.

**Neutral.** The hyperbolic vs Euclidean backbone choice is deferred to measurement.

## Decision drivers / metrics gating success

The single gating metric is the **shortcut-blowup ratio** `|G+| / |G_nav|` and the
**separator-size distribution**, both reported in [ADR-199]. If blowup is small and
separators are sublinear, proceed; if not, fall back to the hyperbolic backbone, then
to GNN-learned ordering (ADR-196 extension E1), before abandoning the approach.

## Alternatives considered

- **Full dense kNN graph as backbone.** Rejected — expander-like, the worst case.
- **METIS/standard ND library.** Viable, but `ruvector-mincut` already implements
  dynamic, sparsifier-backed balanced cuts with incremental updates — reusing it keeps
  the dynamic-insert story coherent.
