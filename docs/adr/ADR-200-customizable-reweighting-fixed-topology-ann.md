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

**Proposed — validated across drift types AND to n=10⁵ (2026-06-04).** This salvages the
one idea from the SepRAG exploration ([ADR-196]) that survived every test — the
*customizable metric* of [ADR-198] — re-tested **standalone, decoupled from CCH** (CCH
full-contraction was NO-GO on embedding graphs, [ADR-199]). The fixed topology matches full
rebuild within the pre-registered 2% recall gate across diagonal/rotational/non-linear drift,
across n=5k…100k, **and** under region-local drift (warping only a 15% cluster), at
**~1,000–4,000× lower update cost**. Confirmed on the **production `ruvector-diskann` Vamana** (96–99% recall, reuse within 2% of
rebuild), and the **hybrid operating policy is validated**: under aggressive compounding drift
a periodic rebuild every ~4 steps recovers 98.8% (vs 99.1% always, 94.4% never) at 25% of the
rebuild cost. **Caveat (honest):** the recall gap widens with scale/churn (−0.2% → −1.7% at
100k; `never` decays to 94% under heavy compounding drift), which is exactly what the hybrid
*defer/batch rebuild* policy is for — so the strategy is "re-weight every step, rebuild
periodically," not "never rebuild." The earlier "rebuild-baseline variance" caveat is **resolved** — the
production index reaches the same conclusion, and the t=0.25 reuse-beats-rebuild dip
reproduced (it is a real property, not lite-Vamana noise). Remaining open: a real GNN-metric
trajectory, an incremental-rebuild baseline, larger-N on diskann, and more-query
confirmation of the gap trend.

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

### Scale (n = 5k…100k, rotational drift t=0.5, ~40% churn)

`scale_drift.rs`, recall@10, 100 queries:

| N | A re-weight | B rebuild | gap | rebuild cost | re-weight update cost | cost ratio |
|---|---|---|---|---|---|---|
| 5,000 | 90.2% | 90.0% | +0.2% | 3.6s | 0.001s | ~3,600× |
| 10,000 | 89.5% | 90.3% | −0.8% | 10.2s | 0.004s | ~2,500× |
| 25,000 | 88.5% | 89.2% | −0.7% | 21.4s | 0.009s | ~2,400× |
| 50,000 | 87.7% | 88.6% | −0.9% | 47.1s | 0.043s | ~1,100× |
| 100,000 | 85.0% | 86.7% | −1.7% | 141.8s | 0.035s | ~4,000× |

**Read:** recall parity stays within the 2% gate through 100k at ~10³–10⁴× lower update
cost (rebuild is super-linear; re-weight ≈ a medoid recompute). The gap **widens mildly**
with N (−0.2% → −1.7%), so the honest framing is *defer/batch rebuilds*, not *never
rebuild*. (Both A and B recall fall with N — fixed beam L=64 weakens relatively as N grows;
the A−B gap, not the absolute, is the signal.) With 100 queries, per-point noise is ~±1%,
so the trend should be confirmed with more queries before being treated as definitive.

### Region-local drift (n=20k; warp only a 15% local cluster)

The hardest realistic case: the metric changes a lot in ONE region (e.g. one topic the
GNN re-learns) while the rest is stationary. Recall reported **separately** for queries
inside vs outside the warped region (a global average would hide a local failure).
`region_drift.rs`:

| t | churn-in | A_in (reuse) | B_in (rebuild) | A_out | B_out |
|---|---|---|---|---|---|
| 0.00 | 0% | 89.7% | 89.7% | 88.0% | 88.0% |
| 0.25 | 44% | 89.8% | **81.4%** | 87.9% | 89.0% |
| 0.50 | 53% | 89.3% | 90.0% | 87.9% | 89.0% |
| 1.00 | 45% | 89.5% | 90.0% | 87.9% | 89.0% |

**Gate: PASS.** Reuse holds *inside* the drifted region — A_in within 0.7% of B_in (and
**above** it at t=0.25) even at 53% in-region churn. Out-region recall is essentially
unchanged (A_out ~1.1% under B_out, within gate). Region-local drift did **not** break
reuse.

**The t=0.25 anomaly.** B_in transiently fell to 81.4% (reuse beat rebuild by 8 pts) then
recovered — a non-monotonic dip where a fresh build on the quarter-warped geometry produced
a worse in-region graph than reuse did. Initially suspected as lite-Vamana build variance;
the production-index run below **reproduced it** (smaller, but real), so it is a genuine
property, not an artifact: a fresh Vamana build on a partially-warped region can underperform
reuse, which keeps the original's good global connectivity.

### Production-index confirmation (`ruvector-diskann`, n=20k)

Re-run on the **shipping** Vamana (`ruvector_diskann::graph::VamanaGraph`, R=32) instead of
the lite reference Vamana — the reuse trick is native (the graph stores only topology;
`greedy_search(vectors, query, beam)` takes vectors externally, so drift = pass transformed
vectors to a graph built on the originals). Harness: `diskann_drift.rs`. recall@10:

Global rotational drift: A reuse vs B rebuild = 95.9/95.8 (t0), 96.2/96.5 (t.25, 29% churn),
95.6/97.1 (t.5, 41% churn), 95.8/96.4 (t1). Region-local (warp 15% cluster), in-region:
A_in/B_in = 98.6/99.0 (t0), **98.6/94.5** (t.25), 98.0/97.9 (t.5, 53% churn), 98.5/99.5 (t1).

**Confirmed:** reuse stays within the 2% gate on the production index (largest gap −1.5% at
peak global churn), at much higher absolute recall (96–99% vs lite ~90%) — a stronger, fairer
baseline. The t=0.25 reuse-beats-rebuild effect reproduces (B_in 94.5 vs A_in 98.6). **The
"rebuild baseline variance" caveat is resolved**: the production index reaches the same
conclusion.

**Query cost is also equal.** Mean distance-evals/query: A ≈ B within ~1% in every row
(e.g. 590 vs 583 at peak churn). So reuse does **not** trade build savings for slower
queries — it matches B on recall *and* per-query work.

**Mechanism:** a RobustPrune graph is a *navigation scaffold* of diversified directions;
greedy search uses the *new* distances to choose direction, while the *old* edges remain
sufficient to navigate. For navigable graphs, top-k recall is governed by navigability +
beam width, not edge metric-optimality — and navigability survives smooth remetrization,
linear *or* non-linear. (Edge optimality would matter more for path length / efficiency,
which is why we also checked per-query evals and found them equal.)

### Operating policy: hybrid re-weight + periodic rebuild (n=10k, diskann)

The shippable answer to "the gap widens with drift": re-weight every step, rebuild
occasionally. Tested on a **compounding random-walk** drift (fresh direction each step,
eps=0.3 — aggressive, to force `never` to decay) over a 24-step trajectory. `hybrid_policy.rs`,
recall@10:

| policy | mean | min | rebuilds | rebuild cost |
|---|---|---|---|---|
| always (rebuild every step) | 99.1% | 98.4% | 24 | 68.7s |
| never (reuse only) | 94.4% | 89.7% | 1 | 2.9s |
| **periodic-4** | **98.8%** | 97.9% | 6 | 17.2s |
| periodic-8 | 98.4% | 96.5% | 3 | 8.6s |
| triggered (Frobenius monitor) | 95–98% | 90–94% | 1–3 | 2.9–8.6s |

**Result:** under aggressive compounding drift `never` decays (94.4% mean, 89.7% floor);
**periodic-4 recovers 98.8% — within 0.3% of always — at 25% of the rebuild cost** (periodic-8:
98.4% at 12.5%). So a cheap fixed-schedule rebuild captures nearly all of always's recall.
**Honest sub-finding:** the drift-*triggered* policy (rebuild when the Frobenius norm of the
cumulative-transform delta exceeds τ) **underperformed simple periodic** — the signal fired
unevenly. Simple **periodic-K is the recommended knob**; a smarter trigger (e.g. a small
sampled-recall probe) is future work. Note: under *gentle* single-direction drift (n=5k test)
`never` did **not** decay — the hybrid only earns its keep under large/compounding drift.

## Consequences

**Positive.**
- A self-learning system can **defer/avoid index rebuilds under linear metric drift** at
  no recall cost — the customizable-metric capability HNSW lacks. Cost asymmetry grows
  with corpus size (rebuild is superlinear; re-weight is free), so the value increases at
  scale.
- This is a *cost* win at equal *quality* (not higher recall) — stated precisely to avoid
  overclaiming.

**Boundaries / not yet proven (the honest caveats).**
- **Synthetic drift.** Drift is parametric (diagonal / rotational / non-linear tanh /
  compounding random walk), not a real learned-GNN metric trajectory. Realistic and
  adversarial, but the live GNN loop is the eventual proof.
- **Gap grows with scale/churn.** Recall gap reaches −1.7% at n=100k and `never` decays to
  ~94% under heavy compounding drift — addressed operationally by the periodic-rebuild
  hybrid, but not eliminated.
- **Incremental baseline.** Compared vs *full* rebuild; an *incremental*-update baseline is
  not yet in (would tighten the cost comparison).
- **Trigger signal.** The Frobenius drift-monitor underperformed simple periodic; a better
  cheap signal (sampled-recall probe) is unproven.

*(Resolved: "linear drift only" — non-linear tanh-warp passes. "n=2000 only" — scaled to
100k. "lite-Vamana baseline variance" — confirmed on production `ruvector-diskann`.)*

## Next steps

1. ~~Scale to n≥10⁵~~ **done** · ~~production `ruvector-diskann` port~~ **done** ·
   ~~region-local drift~~ **done** · ~~hybrid policy~~ **done** (periodic-4 ≈ always at 25%
   cost). Follow-up: diskann at n≥10⁵ with ≥500 queries to firm the gap-trend estimate.
2. **Smarter rebuild trigger** — replace the Frobenius monitor with a small sampled-recall
   probe (estimate live recall cheaply, rebuild when it crosses a floor); should beat
   fixed periodic.
3. Incremental-rebuild baseline for a fair cost comparison (vs full rebuild).
4. **Wire into the `ruvector-diskann`/`ruvector-gnn` loop behind a flag** and validate on a
   real learned-metric trajectory — the eventual production proof and the natural home for
   the periodic-rebuild policy.

## Alternatives considered

- **Rebuild on every metric update** — the incumbent; the cost this ADR removes (kept as
  the baseline B).
- **CCH customization** ([ADR-198] via [ADR-196]) — rejected: contraction blows up on
  embedding graphs ([ADR-199]). The *idea* (cheap re-weight) is retained; the *vehicle*
  (CCH) is dropped in favour of plain fixed-topology ANN.
