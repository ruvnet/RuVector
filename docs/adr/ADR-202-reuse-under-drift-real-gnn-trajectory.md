---
adr: 202
title: "Fixed-Topology Reuse + Periodic Rebuild on a Real Learned-GNN Trajectory"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-198, ADR-199, ADR-200]
tags: [ruvector, retrieval, ann, vamana, diskann, gnn, self-learning, metric-drift, productionization]
---

# ADR-202 — Fixed-Topology Reuse + Periodic Rebuild on a Real Learned-GNN Trajectory

## Status

**Proposed — WIN on a real learned trajectory (2026-06-04).** This closes ADR-200's named
open frontier (next-step #4): productionize the BET 1 reuse-under-drift result by wiring
"re-weight every step + periodic rebuild" into the production `ruvector-diskann` loop behind a
feature flag, and validate it on a **genuine learned-GNN embedding trajectory** — contrastive
link-prediction over the ogbn-arxiv citation graph — instead of the synthetic `A(t)` transforms
of ADR-200.

The result **transfers, at both n=20k and n=50k**: on a real trajectory, pure topology reuse
(`ReweightOnly`) holds recall@10 **within 2% of a full rebuild up to a 40% top-10 churn ceiling
(identical at both scales)** — at or beyond ADR-200's synthetic ~36% holding regime — and the
**periodic-rebuild hybrid recovers the high-churn tail completely** (`Periodic{k:4}`: gap
**−0.01%** at n=20k and **+0.1% (above rebuild)** at n=50k, at **20–24%** of the cumulative
rebuild cost, equal per-query work). The stale control collapses (92% → 33%), proving the benchmark is
drift-sensitive. **Honest boundary:** pure reuse, run past its holding ceiling on a deliberately
overdriven trajectory, decays (−4.73% averaged to 67% churn, 1.05× per-query distance-evals) —
which is precisely what the periodic policy is for, and the shippable periodic policy carries
neither penalty.

The gate was **pre-registered and frozen before any contender run**
(`docs/plans/bet1-productionize/PRE-REGISTRATION.md`).

## Context

RuVector is a self-learning memory: a GNN continuously re-estimates node embeddings, so the
effective L2 metric over those embeddings drifts. ADR-200 showed — under *synthetic* drift, on
the production `ruvector-diskann` Vamana — that the navigation topology can be **reused** (build
once on `E₀`, recompute only distances under `E_t`) within a 2% recall gate up to ~36% churn,
at ~10³–10⁴× lower update cost, with a periodic rebuild recovering the residual gap under heavy
drift. ADR-200's explicitly-named caveat was that the drift was parametric, not a real learned
trajectory, and its next-step #4 was to wire the policy into the live loop and prove it there.

Two facts established the substrate (both verified, not assumed):

1. **The reuse hook is native.** `VamanaGraph` (`crates/ruvector-diskann/src/graph.rs`) stores
   only topology (`neighbors` + `medoid`); `greedy_search(vectors, query, beam)` takes the
   vectors externally. So "adapt to drift" = pass the drifted snapshot to a graph built on the
   original — zero structural change.
2. **`GraphMAE::train_step` does not learn.** It takes `&self` and only returns a loss — no
   backprop, no weight update — so it cannot produce drift. The repo's genuine learnable path is
   direct embedding optimization via `Optimizer` (Adam/SGD) + a real objective. The trajectory is
   built from those primitives, documented up front so its provenance is auditable.

## Decision / Finding

**Ship `ReweightOnly` + `Periodic{k}` as a feature-gated rebuild policy on the production
index; reuse the topology every step and rebuild on a fixed cadence.** Validated head-to-head
(pre-registered gate) against a full rebuild on a real learned trajectory, with a stale-index
negative control.

### Production wiring — `ruvector-diskann::reuse` (feature `reuse-under-drift`, default off)

`RebuildPolicy { AlwaysRebuild, ReweightOnly, Periodic { k } }` + `DriftingIndex`, which owns a
`VamanaGraph` + build params and exposes `on_metric_update(&mut self, vectors)` (bumps a step
counter; rebuilds iff the policy dictates) and `search(vectors, q, beam)`. The index owns only
the *rebuild decision*; the consumer (the GNN) owns the drifting embeddings and passes snapshots
in. The default build is byte-identical (the module is `#[cfg]`-gated out). 5 unit tests cover
cadence + search.

### Trajectory — contrastive link-prediction on ogbn-arxiv (real, public)

Node embeddings are the trainable parameters, initialised from the raw 128-d features (`E₀`,
L2-normalised). Each epoch optimises **InfoNCE** (`ruvector_gnn::training::info_nce_loss`) over
citation edges (positives) + sampled non-edges (negatives) with `ruvector_gnn`'s `Optimizer`
(Adam); embeddings are renormalised onto the unit sphere after each step (so cosine = dot and the
diskann L2 ranking agrees with the contrastive metric), and snapshotted to form `E₀ … E_T`. A
genuinely learned trajectory driven by real arxiv structure. Harness:
`crates/ruvector-gnn/examples/diskann_real_trajectory.rs`. Build params: production Vamana
R=32, L=64, α=1.2; recall@10; 200 queries.

### Evidence (n = 20,000; gradual trajectory, 30 epochs, cumulative churn → 67%)

Strategies (recall@10 vs brute-force truth recomputed under `E_t`):

| cum. churn | B always | **A reuse** | P k=2 | P k=4 | P k=8 | C stale |
|---|---|---|---|---|---|---|
| 7%  | 98.7% | 98.1% | 98.6% | 98.4% | 98.2% | 91.9% |
| 20% | 98.5% | 98.2% | 98.7% | 98.5% | 97.9% | 78.7% |
| 29% | 98.4% | 97.7% | 98.6% | 98.3% | 98.6% | 70.4% |
| 37% | 98.5% | 97.1% | 98.9% | 98.3% | 98.8% | 62.7% |
| **40%** | 98.2% | **96.8%** | 98.6% | 98.8% | 98.8% | 59.7% |
| 42% | 98.9% | 95.9% | 98.8% | 98.8% | 98.6% | 57.5% |
| 54% | 99.2% | 92.4% | 98.9% | 98.6% | 99.0% | 45.8% |
| 67% | 98.8% | 87.4% | 99.2% | 99.0% | 98.8% | 33.2% |

| policy | mean recall | cumulative rebuild cost | evals/query |
|---|---|---|---|
| B always (rebuild every step) | 98.7% | 246.3s (30 builds) | 982 |
| **A reuse** (never rebuild) | 94.0% | **0s** | 1034 |
| **P k=2** | 98.8% | 124.2s | 982 |
| **P k=4** | **98.7%** | **58.7s (24% of B)** | 983 |
| P k=8 | 98.6% | 25.2s (10% of B) | 988 |

**Gate (pre-registered): WIN.**
- **Precondition (teeth) PASS** — trajectory churn 67% (≥ 15% floor); the `C` stale control
  collapses 92% → 33%, so the benchmark is genuinely drift-sensitive (not insensitive).
- **Reuse transfers in-regime** — `A` holds within 2% of `B` up to a **40% churn holding
  ceiling**, at/beyond ADR-200's synthetic ~36%. Through 40% churn the gap is ≤1.6% and at low
  churn `A` is occasionally *above* `B` (a fresh build on partially-drifted geometry can
  underperform reuse — the t=0.25 effect ADR-200 first saw and reproduced).
- **Periodic recovers the tail** — `Periodic{k:4}` within **0.01%** of `B` at **24%** of its
  cumulative rebuild cost, with **equal** per-query work (1.00× evals). `k=8` within ~0.1% at
  10% cost. ADR-200's hybrid finding (periodic-4 ≈ always at 25% cost) reproduced on real drift.

### Scale confirmation (n = 50,000; 20 epochs, cumulative churn → 50%)

The result holds at 2.5× scale — the **holding ceiling is identical (40% churn)**, and at low
churn reuse is again *above* full rebuild:

| cum. churn | B always | **A reuse** | P k=2 | P k=4 | P k=8 | C stale |
|---|---|---|---|---|---|---|
| 12% | 97.0% | **97.5%** | 96.9% | 97.3% | 97.2% | 85.8% |
| 28% | 96.7% | 97.1% | 96.9% | 96.9% | 97.1% | 70.5% |
| 36% | 97.1% | 96.1% | 96.9% | 97.2% | 96.2% | 62.0% |
| **40%** | 96.8% | **95.4%** | 97.1% | 97.1% | 95.5% | 58.2% |
| 50% | 97.5% | 93.1% | 97.3% | 97.3% | 96.7% | 48.9% |

| policy | mean recall | cumulative rebuild cost | evals/query |
|---|---|---|---|
| B always | 97.0% | 271.2s (10 builds) | 1129 |
| A reuse | 95.8% | 0s | 1138 |
| P k=2 | 97.0% | 132.0s (49% of B) | 1127 |
| **P k=4** | **97.1%** (above B) | **53.7s (20% of B)** | 1126 |
| P k=8 | 96.7% | 26.8s (10% of B) | 1130 |

Same verdict: **WIN.** Holding ceiling 40% churn (matches 20k, ≥ ADR-200's 36%); stale control
collapses 86% → 49% (teeth); `Periodic{k:4}` matches/exceeds full rebuild (97.1% vs 97.0%) at
**20% of the cost**, equal per-query work. The whole-trajectory reuse gap is only −1.18% here
(this trajectory tops out at 50% churn vs 20k's 67%) — even pure reuse nearly clears 2% across
the entire run at this drift level.

## Consequences

**Positive.**
- The reuse-under-drift result **transfers from synthetic to real learned drift** — the ADR-200
  WIN is not an artifact of parametric `A(t)` transforms. A self-learning system can defer index
  rebuilds under genuine GNN embedding drift.
- **The shippable policy is `Periodic{k}`, not pure reuse.** It tracks full-rebuild recall within
  ~0.01–0.1% at 10–24% of the cost *and* equal per-query work — capturing nearly all of the cost
  asymmetry with none of pure reuse's high-churn decay or eval penalty. `k` is a single, legible
  knob (rebuild cadence).
- The policy lives behind a default-off feature flag, so it ships with zero impact on the
  existing index.

**Boundaries / honest caveats.**
- **Pure `ReweightOnly` decays past its holding ceiling.** On the deliberately overdriven
  trajectory (to 67% churn) it falls to −4.73% mean and pays 1.05× per-query distance-evals. This
  is the predicted failure mode, addressed operationally by `Periodic{k}` — *use the hybrid, not
  never-rebuild.*
- **The trajectory is one objective (contrastive link-prediction) on one corpus (arxiv).** Other
  learned objectives (node classification, GraphMAE with real backprop) may drift differently;
  the holding ceiling is objective-dependent.
- **The "metric update" is snapshot-granular**, not per-gradient-step; a production loop would
  call `on_metric_update` on its own embedding-flush cadence.
- **Membership is fixed** (drift changes vector *values*, not the point set); streaming
  insert/delete under reuse is unaddressed.
- **A smarter rebuild trigger** (sampled-recall probe, ADR-200 next-step #2) — **now tested and
  WON; see the addendum below.** `Periodic{k}` remains the zero-dependency default; the trigger
  is the better knob when a probe set is available.

*(Resolved from ADR-200: "synthetic drift only" — a real learned-GNN trajectory now confirms the
transfer, with the holding ceiling at 40% churn ≥ the synthetic 36%.)*

## Addendum (2026-06-04): Sampled-recall trigger — WIN

ADR-200 next-step #2 asked whether a smarter rebuild trigger beats fixed `Periodic{k}`; ADR-200's
own Frobenius-norm monitor had *lost* to periodic. Re-tested under **variable-rate** drift (the
only regime where a trigger can earn its keep — periodic is near-optimal under steady drift), with
the gate **pre-registered and frozen** (`docs/plans/bet1-productionize/PRE-REGISTRATION-trigger.md`).

**Stage:** a bursty trajectory — 3-epoch high-lr bursts (per-step churn ~45%) separated by
5-epoch low-lr calm (~2%), 89% end churn, n=20k. **Contenders:** `Recall{floor}` (the bet) vs
`Periodic{k}` (the ADR-202 winner) vs `Frobenius{τ}` (ADR-200's failed monitor), compared on the
(rebuilds, recall) Pareto frontier.

| policy | recall@10 | rebuilds | rebuild cost | probe evals |
|---|---|---|---|---|
| Always | 97.4% | 24 | 333s | — |
| Periodic k=2 | 96.8% | 12 | 168s | — |
| Periodic k=3 | 96.5% | 8 | 113s | — |
| Frobenius τ=0.15 | 97.3% | 9 | 118s | — |
| **Recall floor=0.95** | **97.2%** | **7** | **95s** | 14.4M (~1s) |
| Recall floor=0.93 | 96.6% | 6 | 85s | 14.4M |

**Verdict: WIN.** `Recall{floor=0.95}` reaches 97.2% recall at **7 rebuilds** — beating
`Periodic{k=2}` (96.8% @ 12) on *both* axes (higher recall, **42% fewer rebuilds**) and beating
the best `Frobenius{τ}` (97.3% @ 9) on rebuilds at equal recall. **Probe-cost trap passed:** the
probe's 14.4M distance-evals (~1s total) are <2% of the ~73s of rebuild time saved.

**Mechanism (visible, not asserted):** the per-step churn line `45 44 45 | 2 2 2 | 45 44 …` shows
the trigger rebuilds right after each burst and skips calm stretches, while periodic wastes
rebuilds during calm and under-protects during bursts. Frobenius measures *how much the metric
moved*; the recall probe measures *whether the move broke navigability* — and ADR-202 showed those
decouple, which is why the probe is the better signal.

**Productionized:** `ruvector_diskann::reuse::RecallTrigger` (a `DriftingIndex` in `ReweightOnly`
mode driven by a probe + `force_rebuild`). Its knob `floor` **is the recall SLA** (`0.95` = "keep
recall ≥ 95%"), unlike `k`/`τ` which are indirect proxies. Honest caveat: the probe needs an exact
small-set kNN each update (counted, negligible) and a representative probe set; with no probe
available, `Periodic{k}` remains the zero-dependency fallback. Harness:
`crates/ruvector-gnn/examples/triggered_rebuild.rs`.

## Addendum (2026-06-04): Objective-dependence — generality CONFIRMED, with a degeneracy caveat

This ADR's headline was established on **one** learned objective (contrastive link-prediction);
the named caveat was that the 40% holding ceiling might be objective-dependent. Re-tested with a
**second, different objective** — supervised **node classification** (real ogbn-arxiv 40-class
labels, cross-entropy on a linear head, embeddings as the trainable params) — via the same
harness, contenders, and 2% gate (`objective=nodeclass`; gate pre-registered in
`PRE-REGISTRATION-objective.md`). n=20k, recall@10.

**CONFIRM (the pre-registered question):** in the well-behaved early regime, reuse holds within
2% of full rebuild up to a **54% churn holding ceiling** — *higher* than link-prediction's 40%:

| cum. churn | B always | A reuse | gap |
|---|---|---|---|
| 13% | 98.4% | 98.5% | +0.1 (A above) |
| 37% | 98.3% | 97.7% | −0.6 |
| 47% | 98.4% | 97.4% | −1.0 |
| **54%** | 97.9% | 96.8% | **−1.1** |
| 59% | 98.4% | 94.8% | −3.6 (crosses) |

So the reuse-vs-rebuild parity **generalizes across two distinct learned objectives** (40% and
54% ceilings); the objective-dependence caveat is resolved in the direction of "it generalizes,
and node-class drift is, early, *more* reuse-friendly." `Periodic{k:4}` again recovers at ~22% of
rebuild cost with ~equal per-query work.

**Honest caveat (a real finding, not buried):** past ~60% churn the node-class trajectory
**collapses the embeddings into ~40 class blobs**, and there recall@10 becomes **ill-posed** — with
~500 nodes/class on the unit sphere, a query's top-10 are near-tied intra-blob points whose order
reshuffles under tiny perturbations (churn *saturates* at 67%, never reaching 100%, because
cross-class order is stable but intra-class order is noise). In that degenerate tail the
**full-rebuild baseline itself destabilizes** (B swings 55–96%, its evals/query drop to 721 — a
fresh Vamana build needs distance spread that collapsed geometry denies), so the trajectory-wide
summary shows reuse (92.1%) numerically *above* rebuild (87.8%). **That is a benchmark-degeneracy
artifact (ADR-200's t=0.25 reuse-beats-rebuild dip, amplified), not a genuine "reuse > rebuild"
claim** — recall@10 is not a meaningful target once the metric collapses. The *operational*
conclusion is unaffected: reuse + periodic is never worse than rebuild here. Reporting the artifact
rather than the flattering headline is the point.

## Next steps

1. Wire `on_metric_update` / `RecallTrigger` into the actual `ruvector-gnn` embedding-flush path
   (the policies are validated via the harness; the live serving hook is the remaining glue).
2. ~~Smarter rebuild trigger — sampled-recall probe vs fixed periodic~~ **DONE (addendum: WIN).**
3. ~~Confirm the holding ceiling under a second learned objective (node-classification)~~ **DONE
   (addendum: CONFIRMED, ceiling 54% ≥ link-pred 40%; surfaced a class-collapse degeneracy caveat).**
4. Incremental-rebuild baseline for a fair cost comparison (ADR-200 #3 still open).
5. **(New, from the degeneracy finding)** recall@10 is ill-posed under extreme class collapse — a
   collapse-aware quality metric (or capped-churn operating regime) for self-learning indices whose
   objective tightens clusters over time.

## Alternatives considered

- **Rebuild on every metric update** (`AlwaysRebuild`) — the incumbent; the cost this removes
  (kept as baseline B). Highest recall, full cost every step.
- **Never rebuild** (`ReweightOnly` alone) — rejected as the *default*: transfers in-regime but
  decays past ~40% churn on real drift. Retained as a policy for low-drift / cost-critical
  deployments, with the ceiling documented.
- **CCH customization** (ADR-198 via ADR-196) — rejected earlier (ADR-199: contraction blows up
  on embedding graphs). Fixed-topology ANN reuse is the surviving vehicle.
