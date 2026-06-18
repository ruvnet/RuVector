# BET 1 productionize — Fixed-topology reuse + periodic rebuild on a REAL learned-GNN trajectory

**Status:** Pre-registered (gate frozen before any contender run) · **Date:** 2026-06-04 ·
**Research line:** SepRAG (ruvnet/RuVector issue #534) · **Self-contained:** depends only on
crates already on `main` (`ruvector-diskann`, `ruvector-gnn`) — **independent of PR #535
(`ruvector-seprag`).** ·
**Builds on (by reference):** ADR-200 (BET 1 WIN under *synthetic* drift), ADR-199 (CCH
NO-GO → why fixed-topology, not separators) ·
**Outcome ADR:** ADR-202 (written from the result — WIN *or* NO-GO).

> This document is the **pre-registration**, committed before the validation harness runs on a
> real trajectory. A loss is an acceptable, reportable outcome (cf. ADR-199). Editing the gate
> after seeing results voids the bet. Plumbing (M0–M1) may be built before freeze; contender
> runs (M3+) may not.

> **OUTCOME: WIN** (2026-06-04) — see [ADR-202](../../adr/ADR-202-reuse-under-drift-real-gnn-trajectory.md).
> Reuse holds within 2% recall@10 of full rebuild up to a **40% churn ceiling** (identical at
> n=20k and n=50k, ≥ ADR-200's synthetic ~36%); `Periodic{k:4}` recovers the high-churn tail to
> within 0.01% at 20–24% of rebuild cost. The "early-trajectory" WIN clause was operationalized
> post-hoc as the *holding ceiling* (max contiguous churn where reuse stays within 2%) — the
> regime-resolved statistic this gate named, not the trajectory-wide mean.

## Prove-not-hype protocol (mandatory — all five)

1. **One claim, one number.** 2. **Beat the strongest in-repo incumbent, tuned** (here the
   incumbent *is* the production remedy: full `VamanaGraph` rebuild on the shipping index).
3. **Public data + ground truth** (ogbn-arxiv, in hand). 4. **Pre-register WIN *and* KILL.**
5. **Adversarial check** (here: the *minimum-drift precondition* — the test must not pass
   vacuously on a trajectory that barely moves).

## What this bet proves that ADR-200 did not

ADR-200 established the WIN under *synthetic* drift (`v_t = A(t)·v_0`: diagonal, rotational,
non-linear tanh, compounding random-walk) on the production `ruvector-diskann` Vamana. Its
explicitly-named open frontier (next-step #4): **a real learned-GNN metric trajectory.** This
bet closes exactly that gap and wires the validated policy into the production loop behind a
flag.

**The metric here is L2 over node embeddings** (`ruvector_diskann::distance::l2_squared`). The
GNN re-estimates embeddings over training, so the metric trajectory *is* the embedding
trajectory `E₀ → E₁ → … → E_T`. The reuse hook is native: `VamanaGraph` stores only topology
(`neighbors` + `medoid`); `greedy_search(vectors, query, beam)` (`graph.rs:208`) takes vectors
externally — so "adapt to drift" = build on `E₀`, search with `E_t`, **zero rebuild**.

## Thesis (one claim, one number)

> On a **real learned-GNN embedding trajectory** on ogbn-arxiv, **`ReweightOnly`** (fixed `E₀`
> topology, distances recomputed under `E_t`) holds **recall@10 within 2%** of **`AlwaysRebuild`**
> (full `VamanaGraph` rebuild every step), and where it decays under accumulated drift,
> **`Periodic{k}`** recovers to **within 1%** of `AlwaysRebuild` at **≤ 50% of its cumulative
> rebuild cost**.

Primary metric = **recall@10** vs brute-force ground truth recomputed under `E_t` (as ADR-200).
Secondary, reported as honesty guards: **cumulative rebuild cost (s)** and **per-query
distance-evals** (a recall win that costs more per query is not a clean win).

## Why this scope is the honest one (central insight)

The risk **inverts** relative to a contender benchmark. There the danger is the benchmark being
too easy on the contender; here the danger is the **test being too easy on reuse** — if the
real GNN embeddings drift only slightly, `ReweightOnly` passes *vacuously* and proves nothing.
So the gate carries a **minimum-drift precondition** and a **stale control**, the mirror of
ADR-200's stale-index control ("the C control degrades up to 29 points, proving the graph
matters").

**A second honesty point:** `GraphMAE::train_step` (`graphmae.rs:405`) takes `&self` and only
returns a loss — it has **no backprop and never updates weights**, so it cannot produce drift.
The trajectory is therefore assembled from the repo's *real* learnable primitives
(`Optimizer::step`, `info_nce_loss`, SGD on node embeddings), not from GraphMAE, and not from a
synthetic transform. This is stated up front so the trajectory's provenance is auditable.

## Data & trajectory (real, public — ogbn-arxiv)

n ≈ 169,343 nodes, 128-d features, ~1.17M citation edges (`target/m1-data/arxiv/raw/`:
`node-feat.csv.gz`, `edge.csv`, `node-label.csv.gz`, `node_year.csv.gz` — all in hand).
Validation runs at a tractable slice (n ∈ {20k, 50k}; full-n is a stretch goal).

**Trajectory generation (contrastive link-prediction — chosen path):** node embeddings are the
trainable parameters, initialised from the raw 128-d features (`E₀`). Each epoch optimises
**InfoNCE** (`ruvector_gnn::training::info_nce_loss`) over the citation graph — positives =
sampled edges, negatives = sampled non-edges — with the existing `Optimizer` (Adam/SGD, the
harness computes the InfoNCE gradient w.r.t. embeddings). Embeddings are snapshotted each epoch
to form `E₀ … E_T`. This is a *genuinely learned* trajectory driven by real arxiv structure —
not a parametric `A(t)`.

## Contenders (all scored vs brute-force truth recomputed under `E_t`)

| ID | Strategy | Role |
|---|---|---|
| **A** | `ReweightOnly` — graph built once on `E₀`, searched under `E_t` | **the bet**; rebuild cost 0 |
| **B** | `AlwaysRebuild` — `VamanaGraph` rebuilt under `E_t` every step | incumbent / production remedy |
| **P** | `Periodic{k}` — reuse every step, full rebuild every `k` steps | the shippable hybrid (ADR-200's recommended knob) |
| **C** | `Stale` — built on `E₀`, searched on `E₀`, graded vs `E_t` truth (ignores drift) | floor / teeth control |

`k` sweep: {2, 4, 8}. Build params: production Vamana R=32, L=64, α=1.2 (as `diskann_drift.rs`).

## Pre-registered gate

- **Minimum-drift precondition (teeth — adversarial check):** the trajectory must induce
  **≥ 15% top-10 relevant-set churn** from `E₀` to `E_T` (else the trajectory is too gentle →
  escalate the objective: more epochs / higher LR; a pass on a near-static trajectory is
  **void**). Independently, the **`Stale` control (C)** must degrade **materially** below
  `AlwaysRebuild` (proving the benchmark is drift-sensitive, not insensitive).
- **WIN** — `ReweightOnly (A)` within **2% recall@10** of `AlwaysRebuild (B)` over the early
  trajectory **and**, where A decays under accumulated drift, **some `Periodic{k} (P)`**
  recovers to **within 1%** of B at **≤ 50% of B's cumulative rebuild cost**.
- **Per-query-cost honesty guard** — A's mean distance-evals/query must stay **within ~5%** of
  B's (reuse must not buy build savings with slower queries; ADR-200 found parity within ~1%).
- **Wall-clock honesty guard** — rebuild cost reported in wall-clock seconds; the cost win is
  the *cumulative rebuild* asymmetry (B rebuilds T times, A zero, P `T/k` times).
- **KILL (reportable NO-GO, written like ADR-199)** — `ReweightOnly` **collapses** (>2% below
  B) **early** in the trajectory **and no** `Periodic{k}` recovers within the 1%/≤50%-cost bar:
  i.e. **BET 1 does not transfer from synthetic to real GNN drift.** A clean, publishable
  negative result.
- **Reported regardless:** the recall-vs-step curves for A/B/P/C, the churn-vs-step curve, and
  the cost/recall Pareto point of the best `Periodic{k}`.

**Named live risk (not a formality):** a real link-prediction trajectory may drift the
embeddings *non-uniformly* (some clusters re-learn hard, others barely) — closer to ADR-200's
region-local case than its global case. If `ReweightOnly` holds globally but a re-learned
cluster's in-region recall collapses, that is a **partial result** (report in/out-region
separately, as `region_drift.rs` did), not a silent global-average pass.

## Where it lives (self-contained off `main`)

- **Production wiring — `crates/ruvector-diskann/src/reuse.rs`**, behind cargo feature
  **`reuse-under-drift`** (`default = []`, so the shipping build is byte-identical):
  `RebuildPolicy { AlwaysRebuild, ReweightOnly, Periodic { k } }` + `DriftingIndex` that owns a
  `VamanaGraph` + build params, with `on_metric_update(&mut self, vectors: &FlatVectors)` (bumps
  a step counter; rebuilds iff `Periodic && step % k == 0`) and `search(vectors, q, k)`. The GNN
  side is a pure *consumer* — it writes a new snapshot, then calls `on_metric_update`. Clean
  dependency direction: diskann knows nothing about the GNN.
- **Validation harness — `crates/ruvector-gnn/examples/diskann_real_trajectory.rs`** (dev-deps
  on `ruvector-diskann`): generates the contrastive trajectory, drives all four contenders,
  emits the WIN/KILL table.

No dependency on `ruvector-seprag` (PR #535) — this PR stands alone.

## Milestones

- **M0 — substrate + flag.** Add `reuse-under-drift` feature; scaffold `reuse.rs`
  (`RebuildPolicy`, `DriftingIndex`) + unit tests (policy step-counting, rebuild cadence).
  *Gate: `cargo test -p ruvector-diskann --features reuse-under-drift` green; default build
  unchanged.*
- **M1 — trajectory generator.** arxiv loader (feat + edges); InfoNCE link-prediction loop
  (embeddings as params, `Optimizer::step`, snapshots). *Gate: loss decreases monotonically;
  trajectory induces ≥ 15% top-10 churn (the precondition) — else escalate before freeze.*
- **M2 — contender plumbing.** `AlwaysRebuild` / `ReweightOnly` / `Periodic{k}` / `Stale` over
  the trajectory; recall@10, distance-eval, and rebuild-cost counters; in/out-region split.
  *Gate: `Stale` control degrades materially (teeth).*
- **M3 — full run + gate eval. [FROZEN — post-registration]** Sweep `k ∈ {2,4,8}` over the
  trajectory at n ∈ {20k, 50k}; emit WIN/KILL table; apply both honesty guards.
- **M4 — ADR-202.** Write the outcome (WIN or NO-GO) with ADR-199/200 honesty; update issue
  #534 and `FUTURE-DIRECTIONS.md` (close open item #2).

## Out of scope (named, not silently assumed)

- The smarter sampled-recall rebuild trigger (ADR-200 next-step #2) — `Periodic{k}` is the knob
  under test; the trigger remains future work.
- Incremental-rebuild baseline (vs *full* rebuild) — ADR-200 open item, not this bet.
- Disk-resident / billion-scale; the live multi-tenant serving path. In-memory arxiv at
  n ≤ 50k is the stage.
- Filtered / multi-predicate retrieval (that is BET 2 / ADR-201).
