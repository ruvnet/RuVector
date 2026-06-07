---
adr: 197
title: "Differentiable Min-Cut Condensation Loss (diffcut)"
status: accepted
date: 2026-06-07
authors: [ruvnet, claude]
related: [ADR-196]
tags: [graph, condensation, min-cut, normalized-cut, mincutpool, differentiable, gnn]
---

# ADR-197 — Differentiable Min-Cut Condensation Loss

## Status

**Accepted (implemented).** Module `crates/ruvector-graph-condense/src/diffcut.rs`
plus `CondenseMethod::DiffMinCut`. Builds on ADR-196.

## Context

ADR-196 condenses graphs by *detecting* regions (weak-boundary components,
recursive min cut, etc.) and collapsing them. The graph-condensation surveys
(arXiv:2401.11720, arXiv:2402.03358) and our own SOTA review identified a
specific, **genuinely unpublished gap**: while spectral structural terms appear
in condensation losses — SGDD's Laplacian Energy Distribution (optimal transport
on the spectrum), GDEM's eigenbasis/eigenvalue matching — there is **no
published graph-condensation method whose loss is an explicit, differentiable
min-cut / normalized-cut / modularity term**. Min-cut objectives are mature in
GNN *pooling* (MinCutPool, Bianchi et al. 2020) and in *coarsening*, but using a
relaxed-min-cut objective as the condensation mechanism itself is open.

We want region structure that is **trained to preserve the cut**, not just
heuristically detected — without taking on the cost/complexity of a full
GCond-style bi-level GNN-gradient-matching pipeline, and without adding a heavy
autodiff dependency to a Rust crate that currently depends only on
`ruvector-mincut` + serde/rand/thiserror.

## Decision

Implement a self-contained **differentiable relaxed-min-cut condenser** with
**analytic gradients** (no autodiff framework), after MinCutPool.

### Objective

For a soft cluster assignment `S ∈ R^{N×K}` (row-softmax of learned logits `Θ`),
weighted adjacency `A`, and degree matrix `D = diag(A·1)`:

```
L_cut   = - Tr(Sᵀ A S) / Tr(Sᵀ D S)          ∈ [-1, 0]   (relaxed normalized cut)
L_ortho = ‖ SᵀS / ‖SᵀS‖_F  −  I_K / √K ‖_F   ∈ [0, 2]    (anti-collapse / balance)
L       = L_cut + λ · L_ortho
```

`L_cut` rewards heavy edges inside clusters; `L_ortho` prevents the degenerate
"all nodes in one cluster" solution (which by itself drives `L_cut → -1`).

### Gradients (analytic, all maths in `f64`)

- `∂L_cut/∂S = -(2/Tr(SᵀDS)) · (A S + L_cut · D S)`
- `∂L_ortho/∂S = 2 · S · G_P`, where with `P = SᵀS`, `N_P = ‖P‖_F`,
  `Q = P/N_P − I/√K`, `Gf = Q/L_ortho`:
  `G_P = Gf/N_P − (⟨Gf, P⟩_F / N_P³) · P`
- Backprop through row-softmax: `∂L/∂Θ_il = S_il · (gS_il − Σ_k gS_ik S_ik)`

`A S` is computed sparsely from the edge list (`O(nnz · K)` per step); the rest
is `O(N·K + K²)`. Optimisation is plain gradient descent on `Θ`.

### Correctness

The analytic `∂L/∂Θ` is verified against **central finite differences** in
`gradient_matches_finite_differences` across **K = 2, 3, 4** (max abs error
`< 1e-5`). This is the decisive test; it would catch any sign or chain-rule
error and proves the K-general formulas, not just K=2.

### API and integration

- `DiffCutConfig { num_clusters K, ortho_weight λ, learning_rate, momentum,
  iterations, seed }`; `DiffCutCondenser::train(&DynamicGraph) -> DiffCutResult`.
  Optimisation is heavy-ball momentum GD (`momentum 0` = plain GD) from
  unit-scale random logits (strong symmetry-breaking matters for K > 2).
- `DiffCutResult::soft_assignment()` (the `N×K` matrix) and `hard_regions()`
  (argmax grouping → `Vec<Vec<VertexId>>`).
- `min_cut_loss(graph, soft, k, λ)` — public, evaluates the loss for any
  assignment (a quality metric for learned or hand-built assignments).
- Wired in as `CondenseMethod::DiffMinCut(DiffCutConfig)`: train the soft
  assignment, harden to regions via argmax, then flow through ADR-196's existing
  provenance-preserving super-node/super-edge construction. It is the only region
  method whose structure is *trained* to preserve the cut.

Vertices are sorted ascending for a deterministic row order; logit init is
seeded — same seed ⇒ identical result (tested).

## Consequences

**Positive**
- Fills the specific open gap: a differentiable min-cut term as the condensation
  mechanism, integrated end-to-end and provenance-preserving.
- No new heavy dependency (no candle/burn/tch); pure Rust `f64` maths.
- Gradient-checked, deterministic, label-free (uses topology only; features are
  applied later for centroids).
- Recovers planted structure (e.g. the barbell → exactly two clusters, tested);
  drives the cut term toward −1 on clean partitions.

**Negative / limitations**
- `K` (cluster count) is a fixed hyperparameter; empty clusters are dropped but
  `K` must be chosen.
- Gradient descent is `O(iterations · nnz · K)` and slower than `WeakBoundary`;
  it is opt-in, not the default. Benchmarked under `condense_diffcut`.
- **Convergence is K-sensitive.** Heavy-ball momentum + unit-scale init help,
  but there is no convergence guarantee (non-convex). Empirically it recovers
  small/moderate-K dense graphs (the barbell exactly; ~86% activity purity on a
  3-activity scene in `examples/worldgraph.rs`) but underperforms on large K —
  on a 12-event WorldGraph it does far worse than the structure-aware
  `WeakBoundary` default (which recovers it perfectly). This is the known
  finickiness of MinCutPool-style optimisation and is precisely why
  `WeakBoundary`, not `DiffMinCut`, is the default (ADR-196).
- Topology-only objective: it optimises the structural cut, not feature/label
  matching, so it is not a substitute for supervised GCond-style accuracy
  matching.

## Alternatives considered

1. **Add an autodiff backend (candle/tch/burn) and a learned GNN condenser.**
   Rejected: heavy dependency and build cost for a structural objective whose
   gradients are short closed forms.
2. **Spectral objective (SGDD LED / GDEM eigenbasis) instead of min cut.**
   Rejected for this ADR: those are already published; the min-cut term is the
   unaddressed gap. (A spectral term remains possible future work.)
3. **Only expose the loss as a metric (no training).** Rejected: the request and
   the novelty are the *trainable* loss; we expose both the metric
   (`min_cut_loss`) and the optimiser (`DiffCutCondenser`).

## References

- Bianchi, Grattarola, Alippi — "Spectral Clustering with GNNs for Graph
  Pooling" (MinCutPool), ICML 2020.
- SGDD (arXiv:2310.09192), GDEM (arXiv:2310.09202) — spectral condensation terms.
- Surveys: arXiv:2401.11720, arXiv:2402.03358 (open-problem framing).
- ADR-196 (structure-preserving graph condensation; method taxonomy & substrate).
