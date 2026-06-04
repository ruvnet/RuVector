---
adr: 200
title: "Customizable Re-Weighting: Fixed-Topology ANN Under Metric Drift"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-198, ADR-199]
tags: [ruvector, retrieval, ann, vamana, hnsw, self-learning, metric-drift, customization]
---

# ADR-200 — Customizable Re-Weighting: Fixed-Topology ANN Under Metric Drift

## Status

**Proposed — experimentally validated across diagonal, rotational, and non-linear drift;
bounded by scale (2026-06-04).** This salvages the one idea from the SepRAG exploration
([ADR-196]) that survived every test — the *customizable metric* of [ADR-198] — and
re-tests it **standalone, decoupled from CCH**, since CCH full-contraction was found NO-GO
on embedding graphs ([ADR-199]). The fixed topology matches full rebuild on **both recall
and per-query cost** at **zero** rebuild cost; the only open caveats are scale, region-local
drift, and an incremental-rebuild baseline.

## Context

RuVector is a self-learning memory: a GNN continuously re-estimates relevance, so the
effective distance/relevance metric **drifts** over time. A flat ANN index
(HNSW / `ruvector-diskann` Vamana) is built *for* a metric; when the metric drifts, its
proximity graph becomes suboptimal and the textbook remedy is a costly **rebuild**
(superlinear; minutes-to-hours at corpus scale).

ADR-198 proposed that topology and metric can be decoupled — re-weight cheaply, rebuild
rarely. CCH was one (failed) vehicle for that. The question this ADR answers: **does a
fixed ANN topology, with only distances recomputed under the new metric, retain recall
as well as a full rebuild — and for how much drift?**

## Decision / Finding

**Reuse the navigation topology under metric drift; recompute only distances. Rebuild is
deferred, not per-update.** Validated head-to-head (pre-registered gate) against a full
rebuild, on real ogbn-arxiv embeddings, with a stale-index negative control.

Harness: `crates/ruvector-seprag/examples/reweight_vs_rebuild.rs` (self-contained
Vamana-lite: RobustPrune + greedy beam search). Drift modelled as a vector-space
transform `A`, metric `M = AᵀA`; sweep `A(t) = (1−t)I + t·A_target`.

Strategies (recall@10 vs brute-force truth **under the drifted metric**):
- **A re-weight** — graph built once in the original space, searched under the drifted
  metric. Rebuild cost: **0**.
- **B rebuild** — graph rebuilt under the drifted metric. Rebuild cost: full.
- **C stale** — original graph searched under the *original* metric (ignores drift). Floor.

### Evidence (n=2000, dim=128, Vamana R=24 L=64 α=1.2, k=10)

DIAGONAL drift (per-axis rescale):

| t | set churn | A re-weight | B rebuild | C stale | A−B |
|---|---|---|---|---|---|
| 0.25 | 8% | 90.1% | 90.2% | 86.0% | −0.1% |
| 0.50 | 15% | 90.1% | 90.0% | 80.4% | +0.1% |
| 1.00 | 27% | 90.0% | 90.0% | 70.0% | +0.0% |

ROTATIONAL drift (anisotropic scale on rotated axes — adversarial, general Mahalanobis):

| t | set churn | A re-weight | B rebuild | C stale | A−B |
|---|---|---|---|---|---|
| 0.10 | 10% | 90.1% | 90.1% | 84.3% | +0.0% |
| 0.25 | 25% | 90.1% | 90.0% | 70.3% | +0.1% |
| 0.50 | 36% | 90.0% | 90.1% | 61.0% | −0.1% |
| 1.00 | 23% | 90.1% | 90.0% | 73.0% | +0.1% |

NON-LINEAR drift (residual tanh warp `v + s·tanh(Wv)` — adversarial non-linear):

| t | set churn | A re-weight | B rebuild | C stale | A−B |
|---|---|---|---|---|---|
| 0.10 | 24% | 90.1% | 90.1% | 72.1% | +0.0% |
| 0.25 | 35% | 90.0% | 90.1% | 61.6% | −0.1% |
| 0.50 | 29% | 90.0% | 90.0% | 67.2% | +0.0% |
| 1.00 | 18% | 90.1% | 89.9% | 77.7% | +0.2% |

**Gate (pre-registered): WIN** — A within 0.2% of B across *all three* drift modes, up to
36% relevant-set churn. The C control degrades up to 29 points, proving the graph matters
(the benchmark is not insensitive) — so A's parity is genuine adaptation, not insensitivity.

**Query cost is also equal.** Mean distance-evals/query: A ≈ B within ~1% in every row
(e.g. 590 vs 583 at peak churn). So reuse does **not** trade build savings for slower
queries — it matches B on recall *and* per-query work.

**Mechanism:** a RobustPrune graph is a *navigation scaffold* of diversified directions;
greedy search uses the *new* distances to choose direction, while the *old* edges remain
sufficient to navigate. For navigable graphs, top-k recall is governed by navigability +
beam width, not edge metric-optimality — and navigability survives smooth remetrization,
linear *or* non-linear. (Edge optimality would matter more for path length / efficiency,
which is why we also checked per-query evals and found them equal.)

## Consequences

**Positive.**
- A self-learning system can **defer/avoid index rebuilds under linear metric drift** at
  no recall cost — the customizable-metric capability HNSW lacks. Cost asymmetry grows
  with corpus size (rebuild is superlinear; re-weight is free), so the value increases at
  scale.
- This is a *cost* win at equal *quality* (not higher recall) — stated precisely to avoid
  overclaiming.

**Boundaries / not yet proven (the honest caveats).**
- **Scale.** n=2000; recall-at-scale (n≥10⁵) and the rebuild-cost curve unconfirmed. This
  is now the *primary* open question — and the cost asymmetry only grows with n.
- **Global drift.** Same transform for all points; **region-local** metric change (different
  relevance in different regions) is harder and untested.
- **Baseline.** Compared vs *full* rebuild; an *incremental*-update baseline is not yet in.
- **Synthetic drift.** Drift is parametric (diag/rot/tanh), not a real learned-GNN metric
  trajectory — realistic, but the live GNN loop is the eventual proof.

*(Resolved: the "linear drift only" caveat — non-linear tanh-warp drift now tested and
passes, so navigability robustness is not limited to linear remetrization.)*

## Next steps

1. **Scale to n≥10⁵** on a real ANN index (`ruvector-diskann`) + measure the rebuild-cost
   curve — the decisive remaining test (cost asymmetry grows with n).
2. Region-local drift.
3. Incremental-rebuild baseline for a fair cost comparison.
4. Wire re-weight-on-drift into the `ruvector-diskann`/GNN loop behind a flag and validate
   on a real learned-metric trajectory.

## Alternatives considered

- **Rebuild on every metric update** — the incumbent; the cost this ADR removes (kept as
  the baseline B).
- **CCH customization** ([ADR-198] via [ADR-196]) — rejected: contraction blows up on
  embedding graphs ([ADR-199]). The *idea* (cheap re-weight) is retained; the *vehicle*
  (CCH) is dropped in favour of plain fixed-topology ANN.
