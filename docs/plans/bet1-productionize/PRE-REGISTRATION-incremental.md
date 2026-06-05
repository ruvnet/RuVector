# BET 1 adversarial check — Incremental reindex vs topology-reuse vs full rebuild under metric drift

**Status:** Pre-registered (gate frozen before any contender run) · **Date:** 2026-06-04 ·
**Research line:** SepRAG (ruvnet/RuVector issue #534) · **Self-contained:** depends only on
crates already on `main` (`ruvector-diskann`, `ruvector-gnn`) — **independent of PR #535
(`ruvector-seprag`).** ·
**Branch:** `feat/seprag-bet1-incremental-baseline` (off `feat/seprag-bet1-reuse-under-drift`,
PR #537) ·
**Builds on (by reference):** ADR-200 (BET 1 WIN under synthetic drift), ADR-202 (BET 1 WIN
on a real learned-GNN trajectory — reuse + periodic rebuild) ·
**Outcome ADR:** ADR-204 (written from the result — WIN, PARTIAL, or NO-GO).

> This document is the **pre-registration**, committed before the validation harness runs the
> incremental contender. A loss is an acceptable, reportable outcome (cf. ADR-199, ADR-201). A
> result that *narrows* BET 1 (e.g. "incremental never beats periodic-rebuild") is equally
> reportable. Editing the gate after seeing results voids the bet. Plumbing (the
> `IncrementalIndex` module + harness wiring) may be built before freeze; the contender run may
> not.

## Prove-not-hype protocol (mandatory — all five)

1. **One claim, one number.** 2. **Beat the strongest in-repo incumbent, tuned** — here the
   incumbent is **not** naive pure-reuse; it is the *shippable BET 1 policy* (`ReweightOnly`
   AND `Periodic{k}`, the ADR-202 winners) AND the full-rebuild gold standard. Incremental
   must earn a place none of them already occupy. 3. **Public data + ground truth** (ogbn-arxiv,
   the identical trajectory ADR-202 used). 4. **Pre-register WIN *and* KILL.** 5. **Adversarial
   check** — incremental must beat **`Periodic{k}`** (the BET 1 incumbent), not only the
   naive pure-reuse strawman; reported regardless of the headline gate.

## What this bet proves that ADR-200/202 did not

ADR-200 and ADR-202 compared exactly two update strategies under metric drift:

- **`AlwaysRebuild` (B)** — rebuild the whole Vamana graph every step. Full cost, top recall.
- **`ReweightOnly` (A)** — reuse the `E₀` topology, recompute only distances. Zero cost,
  decays past ~40% churn.
- (`Periodic{k}` interleaves the two on a fixed cadence.)

There is a **structural missing middle**: repair *only the part of the graph that went stale*.
Under metric drift, membership is fixed and only coordinates move, so the natural incremental
operation is to **re-index the displaced nodes** — recompute their out-edges (greedy-search →
robust-prune at the new position) and refresh their back-edges — leaving the rest of the graph
untouched. At churn `C`, this touches ≈`C`·n nodes for ≈`C`× a rebuild's per-node work, which
*could* dominate both A (better recall — it actually fixes stale edges) and B (much cheaper — it
skips the unchanged majority) in the mid/high-churn band where ADR-202 showed pure reuse decays.

**The cheap pre-check (done before this bet):** `ruvector-diskann` has **no faithful incremental
update today.** `DiskAnnIndex::insert` (`index.rs:98`) appends to the flat slab and sets
`built=false` → the next search needs a full `build()` (`index.rs:126` — rebuild from scratch).
`DiskAnnIndex::delete` (`index.rs:207`) is a pure tombstone (zeros the vector, drops the id; the
graph node is left as a zombie — *"marks as deleted, doesn't rebuild graph"*). So the incremental
baseline must be **built**, faithfully, not assumed to exist.

## The incremental baseline — exactly what it is, and is not (so it is not a strawman)

**Operation (faithful, named precisely):** under metric drift no point is ever removed — a point
only moves. So the incremental op is **not** FreshDiskANN delete+reinsert (which needs a
reverse-edge index and delete-consolidation, *inapplicable* when nothing leaves). It is:

> For each displaced node `u`: recompute `u`'s out-edges via `greedy_search(E_t, E_t[u]) →
> robust_prune`, set `neighbors[u]`, and add back-edges `u → c` into each new out-neighbour `c`
> (degree-bounded re-prune, identical to `VamanaGraph::build`'s back-edge step, `graph.rs:117`).

**Targeting knob (`reindex_frac` `f`):** each update reindexes the top-`f` fraction of nodes by
**displacement since their last reindex** (`‖E_t[u] − reference[u]‖`, `reference` updated per
reindex). `f` is the cost/recall knob, analogous to `Periodic{k}`. Swept `f ∈ {0.05, 0.1, 0.2,
0.5}`. (`f=1.0` reindexes everything every step → a sanity upper bound that should approach B.)

**Honest scope of the baseline (stated up front, not buried):**
- In-memory graph repair only — **not** a full FreshDiskANN: no on-disk streaming, no PQ delta,
  no concurrency, no crash-consistency. The comparison is *graph-quality + update-cost*, not a
  systems benchmark.
- **No delete-consolidation** — correct here because membership is fixed (nothing is deleted).
  Residual stale *in*-edges from non-displaced neighbours that `u` moved away from are left to
  **decay** — the exact tolerance the BET 1 reuse result proved Vamana has. If a displaced
  neighbour is itself reindexed (likely under global drift) it re-prunes and drops the stale edge.
- Built behind the existing `reuse-under-drift` feature flag; the default shipping build is
  byte-identical (the module is `#[cfg]`-gated out). The only always-compiled change is exposing
  `VamanaGraph::robust_prune` as `pub(crate)` (visibility only — no logic change to `build`).

## Thesis (one claim, one number)

> On the ADR-202 real learned-GNN ogbn-arxiv trajectory, there exists a `reindex_frac` knob and
> a churn band in which **incremental reindex beats pure `ReweightOnly` by >2 points recall@10**
> while costing **≤0.5× the cumulative full-rebuild cost** and staying **within 2% recall@10 of
> `AlwaysRebuild`** — i.e. incremental carves a (recall, cost) Pareto point that neither pure
> reuse nor full rebuild occupies.

Primary metric = **recall@10** vs brute-force ground truth recomputed under `E_t` (as ADR-202).
Cost metric = **cumulative update wall-clock** (incremental reindex time vs B's rebuild time),
reported as a fraction of B. Honesty guard = **per-query distance-evals** (a recall win that
makes queries slower is not clean).

## WIN / KILL gate (frozen)

Let `f*` be the best incremental knob. Over the trajectory:

- **WIN** — **all** of:
  1. **Beats pure reuse:** ∃ a contiguous churn band where incremental(`f*`) mean recall@10
     exceeds `ReweightOnly` (A) by **> 2.0 points**.
  2. **Cheaper than rebuild:** incremental(`f*`) cumulative update cost **≤ 0.5×** B's cumulative
     rebuild cost.
  3. **Matches rebuild quality:** within that band incremental(`f*`) stays **within 2.0 points**
     recall@10 of `AlwaysRebuild` (B).
  4. **Eval honesty:** incremental(`f*`) per-query evals **≤ 1.10×** B's (no hidden query-cost
     penalty).
- **PARTIAL** — incremental beats pure reuse by >2 pts and is ≤0.5× B cost, **but** is itself
  dominated by some `Periodic{k}` on the (recall, cost) frontier (i.e. a periodic policy gives
  ≥ incremental's recall at ≤ its cost). Reported as: "the missing middle exists but the BET 1
  periodic incumbent already covers it."
- **KILL / NO-GO** — incremental never beats pure reuse by >2 pts within the cost bar, **or** its
  only recall edge comes at >0.5× B cost (i.e. you may as well rebuild). Reported as a narrowing:
  "reuse + periodic rebuild is sufficient; incremental repair earns no Pareto place."

**Adversarial check (reported regardless of verdict):** the full (recall, cost) frontier of
{B, A, Periodic{k=2,4,8}, Incremental{f}} — does incremental dominate the **`Periodic{k}`**
incumbent, or only the naive pure-reuse strawman? A WIN that does not also beat `Periodic{k}` is
downgraded to PARTIAL in the prose, even if the frozen numeric gate above passes.

**Precondition (teeth, inherited from ADR-202):** the trajectory must induce ≥ 15% top-10 churn
`E₀→E_T`, and the stale control must collapse — else the run is **VOID** (a too-gentle trajectory
where every policy ties proves nothing). The Adam-driven generator + ≥15% churn assertion from
the ADR-202 trigger addendum are reused unchanged.

## A-priori risk register (named before the run, to keep the verdict honest)

1. **Cost-squeeze (most likely outcome).** Incremental's recall edge over reuse only matters
   *above* ~40% churn (where reuse decays); but re-indexing >40% of nodes costs ≈ a rebuild, so
   the cost edge erodes exactly where the recall edge appears. Plausible result: **NO-GO /
   narrowing** — the two advantages never co-exist.
2. **Periodic already covers it.** Even if incremental beats *pure reuse*, `Periodic{k}` (ADR-202)
   may match it at lower cost → **PARTIAL**, not WIN. This is why the adversarial check is
   mandatory.
3. **Stale-in-edge decay underperforms.** Without delete-consolidation, residual stale in-edges
   might drag incremental below rebuild quality (fail WIN clause 3). If so, report it — and note
   that adding consolidation is a heavier (FreshDiskANN-class) baseline, deliberately out of scope.

## Data & harness

Identical to ADR-202: ogbn-arxiv slice (n ∈ {20k, 50k}), 128-d features, contrastive
link-prediction (InfoNCE, Adam) trajectory `E₀…E_T`; production Vamana R=32, L=64, α=1.2;
recall@10; 200 queries; per-snapshot brute-force ground truth under `E_t`.
Harness: `crates/ruvector-gnn/examples/diskann_real_trajectory.rs` — **extended** with the
incremental contender measured on the *same* trajectory/queries/truth (not a parallel copy).
Module under test: `ruvector_diskann::reuse::IncrementalIndex` (feature `reuse-under-drift`).

Run: `cargo run --release -p ruvector-gnn --example diskann_real_trajectory --features
ruvector-diskann/reuse-under-drift -- [N] [EPOCHS] [LR] [SNAP_EVERY] [objective]`
