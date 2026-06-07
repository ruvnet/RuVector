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
is `O(N·K + K²)`. The loss + analytic gradients live in `cutloss.rs`; the
optimiser and orchestration in `diffcut.rs`.

### Optimisation (the part that makes large K work)

Plain gradient descent stalls at large `K` (a known property of MinCutPool-style
objectives). Three standard ingredients fix it, all defaults:

1. **Adam** (`Optimizer::Adam`, default) — adaptive per-parameter moments; far
   more robust than SGD on the ill-conditioned, non-convex cut objective.
   `Optimizer::Sgd { momentum }` remains available.
2. **Warm-start init** (`InitStrategy::WarmStart`, default) — seed the logits
   from the cheap `WeakBoundary` structural prior (largest regions → own
   clusters, overflow round-robin, +bias into the logits) and *refine* with the
   differentiable objective, instead of descending from random noise. This is
   the coreset/K-Center idea GCond/SFGC use, and it is what makes `K ≫ 2`
   converge. `InitStrategy::Random` remains available.
3. **Restarts** (`restarts`) — keep the lowest-loss run.

Result: on a 12-event WorldGraph (`examples/worldgraph.rs`) DiffMinCut reaches
**100% activity purity, cut preserved (inflation 1.000)** — matching
`WeakBoundary` — where plain-GD/random scored ~30%. Training cost fell from
~24 s (plain GD, 96 nodes) to milliseconds (Adam, `condense_diffcut` bench:
~0.96 ms @ 64, ~6.4 ms @ 192 nodes). Tests `warm_start_recovers_many_clusters`
(K=8, purity > 0.85) and `warm_start_beats_random_at_large_k` lock this in.

### Scale levers (for large / million-node graphs)

Three further levers, off by default, target very large graphs:

4. **Early-stopping** (`tolerance`, default `1e-6`) — warm-start lands near the
   optimum, so most iterations are wasted; stop when the loss plateaus. Test
   `early_stopping_cuts_iterations`.
5. **Parallelism** (`parallel`, Rayon) — the per-iteration `A·S` (CSR,
   row-parallel) and the `O(N·K²)` `SᵀS` + ortho-gradient loops run in parallel.
   **Deterministic / bit-identical to sequential** (both use the same chunked
   partial-sum ordering), proven by `parallel_matches_sequential_exactly`.
6. **Edge-minibatching** (`minibatch_edges`) — estimate the gradient from a
   sampled subset of edges per step (`O(batch·K)` instead of `O(|E|·K)`); the
   final reported loss is still computed full-batch (exact). Test
   `minibatch_recovers_structure`.

Bench (`condense_diffcut_levers`, 1024 nodes, 4 cores, 100 iters): sequential
~95 ms, parallel ~83 ms (~1.15×), minibatch(2048) ~77 ms (~1.2×). Gains are
modest at this size (Rayon dispatch overhead vs. small per-iteration work) and
grow with `N`; the value is enabling graphs that do not fit a single-threaded
full-batch budget, not speeding up small ones.

### Correctness

The analytic `∂L/∂Θ` is verified against **central finite differences** in
`gradient_matches_finite_differences` across **K = 2, 3, 4** (max abs error
`< 1e-5`) — the decisive test, proving the K-general formulas, not just K=2.

### API and integration

- `DiffCutConfig { num_clusters K, ortho_weight λ, learning_rate, iterations,
  optimizer, init, restarts, tolerance, parallel, minibatch_edges, seed }`;
  `DiffCutCondenser::train(&DynamicGraph) -> DiffCutResult`. Default = Adam +
  warm-start + early-stop, large-K-ready. `DiffCutResult::iterations_run()`
  reports how many iterations actually ran.
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
- Recovers planted structure at small *and* large K (barbell exactly; K=8/K=12
  recovered via Adam + warm-start), and drives the cut term toward −1.
- Fast: milliseconds per train (was tens of seconds under plain GD).

**Negative / limitations**
- `K` (cluster count) is a fixed hyperparameter; empty clusters are dropped but
  `K` must be chosen.
- Still slower than `WeakBoundary` (`O(restarts · iterations · nnz · K)`) and
  non-convex with no formal convergence guarantee, so it is opt-in, not the
  default. Large-K reliability leans on the warm-start prior; `InitStrategy::
  Random` at large K remains hard (documented, and what `warm_start_beats_random`
  measures). `WeakBoundary` stays the default (ADR-196) for speed/simplicity.
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
