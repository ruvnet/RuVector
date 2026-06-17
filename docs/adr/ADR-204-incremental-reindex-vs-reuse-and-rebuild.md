---
adr: 204
title: "Incremental Reindex vs Topology-Reuse vs Full Rebuild Under Metric Drift"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-196, ADR-198, ADR-199, ADR-200, ADR-202]
tags: [ruvector, retrieval, ann, vamana, diskann, gnn, self-learning, metric-drift, incremental]
---

# ADR-204 — Incremental Reindex vs Topology-Reuse vs Full Rebuild Under Metric Drift

## Status

**Proposed — WIN (scale-qualified, regime-concentrated) on a real learned-GNN trajectory
(2026-06-04).** This is the adversarial check ADR-200/202 never ran: those compared exactly two
index-maintenance strategies under metric drift — reuse *everything* (`ReweightOnly`, zero cost,
decays) vs rebuild *everything* (`AlwaysRebuild`, full cost) — interleaved by `Periodic{k}`.
There is a **structural missing middle**: repair only the part of the graph that went stale.
This ADR builds that third policy (`IncrementalIndex`) faithfully and measures it head-to-head
on the identical ADR-202 trajectory.

**Result, reproduced at n=20k AND n=50k AND on a gradual trajectory:** targeted incremental
repair of the displaced subset **matches full-rebuild recall@10 (within ~0.2 pts) at ~42% of
the rebuild cost, and beats the strongest periodic policy (`Periodic{k=2}`)** — earning a
Pareto point on the maintenance frontier that neither pure reuse nor full rebuild occupies.
The gate was **pre-registered and frozen before any contender run**
(`docs/plans/bet1-productionize/PRE-REGISTRATION-incremental.md`, commit `b388c427`).

**Honest bounding (three narrowings, all measured):**
1. **Scale-sensitive.** At n=20k (heavy collapse) incremental *swept* the frontier — every
   `Periodic{k}` and full rebuild was dominated. At n=50k and on the gradual trajectory it
   does **not** sweep: incremental wins the **high-recall tier** (`f=50%` dominates `k=2` + full
   rebuild) but the **cheaper periodic tiers (`k=4`, `k=8`) reclaim Pareto-optimality**. So
   incremental **extends** the frontier at the high-recall end; it does not replace periodic.
2. **Regime-concentrated.** The advantage lives in the high-churn decay tail (the regime ADR-202
   explicitly handed to periodic rebuild). At moderate churn (≤35%) all policies cluster within
   ~1 pt — incremental adds nothing because reuse has not yet decayed.
3. **Degeneracy caveat.** At >90% churn (n=20k) incremental reads *above* full rebuild — the
   known fresh-build-on-collapsed-geometry effect (ADR-200 t=0.25 / ADR-202 collapse). At n=50k
   incremental ≈ rebuild *exactly* (no contamination), so the conservative claim is **"matches
   rebuild," not "beats" it.**

## Context

RuVector is a self-learning memory: a GNN re-estimates node embeddings, so the L2 metric over
them drifts. ADR-200 (synthetic drift) and ADR-202 (real learned-GNN trajectory) established
that the production `ruvector-diskann` Vamana topology can be **reused** under drift —
recompute distances, not the graph — within a 2% recall gate up to a ~40% churn holding ceiling,
with `Periodic{k}` rebuilds recovering the high-churn tail. ADR-200's named open frontier
(next-step #3) was an **incremental-update baseline** for a fair cost comparison; ADR-202's
caveats list reads *"streaming insert/delete under reuse is unaddressed."* This ADR closes that.

**The cheap pre-check (done first, per protocol): `ruvector-diskann` has no faithful incremental
update.** `DiskAnnIndex::insert` (`index.rs:98`) appends to the flat slab and sets
`built=false` → the next search requires a full `build()` (`index.rs:126`, a from-scratch
rebuild). `DiskAnnIndex::delete` (`index.rs:207`) is a pure tombstone (zeros the vector, drops
the id; the graph node is left as a zombie — its own doc-comment: *"marks as deleted, doesn't
rebuild graph"*). So the incremental baseline had to be **built**, faithfully — not assumed.

## Decision / Finding

**Add `IncrementalIndex` as the third maintenance policy: under metric drift, repair only the
displaced subset of the Vamana graph.** Validated head-to-head (pre-registered gate) against
pure reuse (`A`), full rebuild (`B`), and the `Periodic{k}` incumbents, on the same real
learned trajectory, with the stale-index negative control.

### The faithful incremental operation (what it is, and is not)

Under metric drift **membership is fixed** — a point never leaves the set, its coordinates only
move — so the faithful operation is **not** FreshDiskANN delete+reinsert (whose
delete-consolidation and reverse-edge index are inapplicable when nothing is removed). It is, for
each displaced node `u`:

> recompute `u`'s out-edges via `greedy_search(E_t, E_t[u]) → robust_prune` at the new position,
> set `neighbors[u]`, and add back-edges into its new out-neighbours (degree-bounded re-prune) —
> exactly the per-node step `VamanaGraph::build` runs, applied to one node.

`reindex_frac` `f` selects the top-`f` of nodes by **displacement since their last reindex** to
repair each update — the cost/recall knob, analogous to `Periodic{k}`'s `k`. Residual stale
*in*-edges from non-displaced neighbours `u` moved away from are left to **decay** — the exact
tolerance ADR-200/202 proved Vamana has (a neighbour that is itself reindexed re-prunes and drops
the stale edge). **Scope (stated, not buried):** in-memory graph repair only — no on-disk
streaming, no PQ delta, no concurrency, no crash-consistency. The only always-compiled change is
exposing `VamanaGraph::robust_prune` at `pub(crate)` (visibility, no logic change); all new logic
is feature-gated (`reuse-under-drift`). `ruvector_diskann::reuse::IncrementalIndex`, 3 unit tests.

### Evidence — the (recall@10, cost) frontier (200 queries, R=32 L=64 α=1.2, recall vs brute-force under `E_t`)

`<- Pareto` marks frontier-optimal points (no other policy has ≥ recall at ≤ cost).

**n = 20,000, overdriven trajectory (60 epochs, cumulative churn → 93%):**

| policy | recall@10 | cost (s) | Pareto |
|---|---|---|---|
| A reuse | 67.0% | 0.0 | ✓ |
| inc 5% | 82.8% | 7.5 | ✓ |
| inc 10% | 91.3% | 16.7 | ✓ |
| P k=8 | 90.3% | 22.1 | dominated by inc-10% |
| inc 20% | 95.7% | 34.1 | ✓ |
| P k=4 | 95.0% | 53.6 | dominated by inc-20% |
| **inc 50%** | **98.1%** | **87.5** | ✓ |
| P k=2 | 95.9% | 105.2 | dominated by inc-50% |
| B always | 96.3% | 208.4 | dominated by inc-50% |

Incremental **sweeps**: every periodic and full rebuild is dominated. (Reproduced across two
runs within ±0.3 pts.) Caveat: at this churn `inc-50% (98.1%) > B (96.3%)` is the
fresh-build-on-collapsed-geometry degeneracy, not a "beats rebuild" claim.

**n = 50,000, overdriven trajectory (50 epochs, cumulative churn → 94%):**

| policy | recall@10 | cost (s) | Pareto |
|---|---|---|---|
| A reuse | 62.8% | 0.0 | ✓ |
| inc 5% | 74.7% | 24.9 | ✓ |
| inc 10% | 84.6% | 49.5 | ✓ |
| P k=8 | 86.0% | 73.5 | ✓ |
| inc 20% | 92.2% | 102.1 | ✓ |
| P k=4 | 93.8% | 146.6 | ✓ |
| **inc 50%** | **96.5%** | **254.9** | ✓ |
| P k=2 | 96.1% | 292.3 | dominated by inc-50% |
| B always | 96.3% | 611.3 | dominated by inc-50% |

Incremental does **not** sweep: it wins the high-recall tier (`inc-50%` dominates `P k=2` + full
rebuild) but `P k=4`/`P k=8` stay Pareto-optimal. Here `inc-50% (96.5%) ≈ B (96.3%)` **exactly**
— a clean "matches rebuild at 42% cost," no degeneracy.

**n = 20,000, gradual trajectory (30 epochs lr=0.005, churn spans 18% → 77%):** the
anti-overdrive check. Base BET-1 verdict reproduced ADR-202's WIN (reuse holds in-regime).

| policy | recall@10 | cost (s) | Pareto |
|---|---|---|---|
| A reuse | 88.8% | 0.0 | ✓ |
| inc 5% | 91.2% | 4.7 | ✓ |
| P k=8 | 96.5% | 8.3 | ✓ |
| inc 10% | 94.6% | 9.9 | dominated by P k=8 |
| inc 20% | 98.1% | 20.8 | ✓ |
| P k=4 | 98.4% | 25.1 | ✓ |
| **inc 50%** | **99.0%** | **53.7** | ✓ |
| P k=2 | 98.8% | 58.8 | dominated by inc-50% |
| B always | 98.9% | 127.8 | dominated by inc-50% |

Per-step regime structure (the honest core): at **18–35% churn** all policies cluster
(~97–99%) — incremental adds nothing; at **43–77% churn** reuse decays (96% → 79%) while
`inc-20/50%` track full rebuild (~98–99%). The advantage emerges *progressively* with churn —
not an overdrive artifact. `inc-50%` again dominates `P k=2` + full rebuild; `P k=8` is strongly
Pareto-optimal at the cheap tier.

### The robust claim (reproduced in all three runs)

> **`inc-50%` matches full-rebuild recall@10 within ~0.2 pts at ~42% of the rebuild cost, and
> Pareto-dominates the strongest periodic policy (`Periodic{k=2}`).** At the high-recall
> operating point a production system actually targets, spread-out targeted repair beats both
> lumped periodic rebuilds and full rebuild.

**Mechanism (visible, not asserted).** `Periodic{k}` spends each rebuild on *all* `n` nodes
(most of which did not move) and lets recall sawtooth-decay between rebuilds; incremental spends
the same compute *only* on displaced nodes, every step, so recall never decays. Under continuous
drift, evenly-spread targeted repair beats lumped blind rebuilds at equal cost — the missing
middle paying off, in exactly the decay-tail regime ADR-202 assigned to periodic.

## Consequences

**Positive.**
- A **third, dominant-at-the-high-recall-tier maintenance policy** for self-learning indices:
  `IncrementalIndex{f≈0.5}` gives full-rebuild recall at ~42% of the cost and beats the best
  periodic schedule — at both n=20k and n=50k and on a gradual trajectory.
- `f` is a single legible knob (fraction of nodes repaired per update); the incremental frontier
  is **finely tunable** where `Periodic{k}` offers only the coarse points `k∈{2,4,8}`.
- Feature-gated (`reuse-under-drift`, default off) — zero impact on the shipping build.

**Boundaries / honest caveats.**
- **Does not sweep at scale.** At n=50k and moderate churn, `Periodic{k=4,8}` reclaim
  Pareto-optimality at cheaper tiers. Incremental **extends** the frontier at the high-recall
  end; it is a complement to periodic, not a replacement. The frontier *sweep* was specific to
  the most-collapsed case (n=20k, 93% churn).
- **Advantage grows with churn.** At ≤35% churn all policies cluster — incremental earns its
  keep only once reuse has begun to decay (≳40% churn).
- **Degeneracy at extreme churn.** The `inc > B` reading at >90% churn (n=20k) is the
  fresh-build-on-collapsed-geometry effect, not a genuine "beats rebuild." At n=50k `inc ≈ B`.
- **Per-query cost at tiny budgets.** At `f=5%` the incremental graph cost 1.12× B's per-query
  evals at n=50k (failed the ≤1.10× honesty bar); the clean win regime is `f ∈ [0.2, 0.5]`.
- **Recall margins vs periodic** (+0.2 to +2.2 pts) are near per-run build-noise; the **cost**
  advantage and the **frontier shape** are the robust signals (the recall edge is at-worst a tie).
- **Membership fixed.** Drift changes vector values, not the point set; true streaming
  insert/delete (with delete-consolidation) remains out of scope — a heavier FreshDiskANN-class
  baseline.

*(Resolved from ADR-200 next-step #3 / ADR-202 caveat: the incremental baseline now exists and
is measured; reuse + periodic is **not** strictly sufficient — incremental dominates the
high-recall tier.)*

## Next steps

1. **Adaptive `f`** — a displacement-threshold (reindex what actually moved past τ) instead of a
   fixed top-fraction would make incremental cheap when drift is calm and heavy when it bursts;
   pairs naturally with the ADR-202 sampled-recall trigger.
2. **Incremental + trigger** — drive `IncrementalIndex` from the `RecallTrigger` probe (repair
   when measured recall dips) rather than every step.
3. **Larger n / more queries** — confirm the scale-attenuation trend (sweep → high-tier-only)
   past n=10⁵ with ≥500 queries.
4. **True streaming membership** — delete-consolidation + insert for an *open* corpus, the
   heavier baseline this ADR deliberately scoped out.

## Alternatives considered

- **Pure reuse / full rebuild / `Periodic{k}`** — the ADR-200/202 incumbents; kept as the
  baselines `A`/`B`/`P`. Incremental dominates them only at the high-recall tier.
- **FreshDiskANN delete+reinsert with consolidation** — rejected as out of scope: membership is
  fixed under drift, so no point is deleted; consolidation solves a problem this regime does not
  have, at much higher complexity.
