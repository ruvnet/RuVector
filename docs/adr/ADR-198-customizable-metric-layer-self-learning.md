---
adr: 198
title: "Customizable Metric Layer for Self-Learning Retrieval (CCH Customization ↔ GNN Loop)"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-197, ADR-199]
tags: [ruvector, cch, customization, gnn, self-learning, ewc, multi-metric, solver, bmssp, sparsifier]
---

# ADR-198 — Customizable Metric Layer for Self-Learning Retrieval

## Status

**Proposed.** Implements Phase 2 of [ADR-196] on the topology from [ADR-197].
Validated by [ADR-199].

## Context

RuVector is a *self-learning* memory system: its GNN continuously re-estimates
relevance from feedback (`ruvector-gnn`, with `ewc` for catastrophic-forgetting
control). In a flat-index world, changing the relevance metric means **rebuilding the
index** — prohibitively expensive to do on every learning step.

CCH's defining feature is the separation of **metric-independent topology** (order +
shortcut set, [ADR-197]) from **metric-dependent weights** (customization). A new
metric is absorbed by re-running customization only — seconds on continental graphs,
topology untouched.

**The key mapping:** RuVector's self-learning loop *is a stream of new metrics*. The
GNN plays the role that live traffic plays in road routing; CCH customization is the
engine that re-absorbs it without rebuilding. This is the single strongest argument
for the whole SepRAG program.

## Decision

### 1 — Edge cost is supplied by the GNN; the metric is non-negative and additive

```
w(u, v) = f_θ(h_u, h_v, edge_feats)        # GNN edge head, >= 0
```

Cost definitions used (all additive along paths, valid for triangle relaxation):

- **Manifold-semantic:** `1 - cos(u,v)` (or angular `sqrt(2 - 2cos)`, a true metric)
  defined only on `G_nav` edges → path cost follows the data manifold's geodesics, not
  flat cosine across empty embedding space.
- **Relational:** `-log strength(e)` on KG edges → multiplicative confidence becomes
  additive path cost (max-product ↔ shortest path; trust decays along a chain).
- **Learned:** the GNN edge head above — the metric that customization re-absorbs on
  every `θ` update.

Cosine *similarity* itself is never used as a path cost (not additive, not a metric).

### 2 — Customization as a sparse triangular sweep (reuse the solver)

```
fn customize(G_plus, order, w_init) -> w:        # weights over all G_plus edges
    w = w_init                                    # orig edges = metric; shortcuts = +inf
    for level in elim_tree_levels_bottom_up:      # PARALLEL within a level
        for v in level:
            for (u, x) in lower_triangles(v):      # u,x = higher-ranked nbrs of v
                w[u,x] = min(w[u,x], w[u,v] + w[v,x])
    return w                                       # re-run ONLY this on metric change
```

This is structurally a DP sweep over a chordal graph by elimination-tree level —
the same shape as `ruvector-solver`'s `bmssp` multigrid V-cycle and `forward_push` /
`backward_push`. **Decision: implement customization as a specialization of
`ruvector-solver`**, not a fresh kernel; vectorize triangle relaxations with
`ruvector-solver::simd`. Effective-resistance importance from `ruvector-sparsifier`
can prioritize which shortcuts to refresh first under a time budget.

### 3 — Multi-metric quiver: one topology, many cheap customizations

```rust
pub struct CchMetric {                 // one per metric; many coexist cheaply
    up_weight: Vec<f32>,               // parallel to CchTopology.up_targets
    metric_id: MetricId,               // semantic | recency | trust | task | blend
}
```

Maintain a bank: `w_semantic`, `w_recency`, `w_trust`, `w_task`, and on-the-fly
blends `Σ λ_i w_i`. A query selects or blends a **lens** at near-zero marginal cost —
the retrieval analogue of CCH's car/truck/bike profiles, and infeasible with
per-metric HNSW rebuilds. Natural fit for multi-tenant and task-conditioned retrieval.

### 4 — Update cadence and forgetting

- **Weight change (frequent):** re-customize only. No topology touch.
- **Topology change (batched):** incremental insert via `jtree`/`linkcut` ([ADR-197]);
  periodic re-order.
- **Catastrophic forgetting:** `ruvector-gnn::ewc` constrains `θ` drift so the metric
  evolves without collapsing prior structure; customization then reflects the
  EWC-regularized metric.

### 5 — Rerank with cut-as-attention

Survivors are reranked by `ruvector-attn-mincut`, reusing the *same separator cuts*
as attention masks (only attend across separators "open" for this query). Keeps
retrieval and attention on one shared structure.

## Consequences

**Positive.**
- Relevance updates cost a customization pass, not an index rebuild — this is the
  quantifiable self-learning benefit ([ADR-199] measures customization time vs HNSW
  rebuild and "adaptation lag" in queries).
- Multi-lens retrieval is essentially free.
- Reuses `ruvector-solver`, `ruvector-sparsifier`, `ruvector-gnn`, `attn-mincut`.

**Negative.**
- Requires a non-negative additive metric; rich learned scores must be mapped into
  that form (mitigated by the cost definitions above).
- Customization parallelism is bounded by elimination-tree level structure (deep
  narrow trees parallelize poorly) — another reason separator quality ([ADR-197])
  matters.
- A metric that violates the triangle structure (e.g. negative learned costs) breaks
  relaxation; the GNN edge head must be constrained to `>= 0`.

**Neutral.** The quiver's memory cost is `O(#metrics × |G+|)` floats — cheap, but
capped by a configured lens budget.

## Alternatives considered

- **Bake the metric into the order (plain CH).** Rejected — defeats the entire
  self-learning premise.
- **Rebuild HNSW on metric change.** Rejected — the cost this ADR exists to avoid;
  retained only as the [ADR-199] baseline to quantify the win.
- **Recompute all shortcut weights from scratch each query.** Rejected — customization
  amortizes this across queries between metric updates.
