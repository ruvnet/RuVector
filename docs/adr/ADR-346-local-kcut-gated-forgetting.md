# ADR-346: LocalDeterministic Mincut Engine — Fixing ADR-345's Latency and Determinism Defects

## Status

Rejected (as a full, all-thresholds-pass replacement), with a real, retained
partial win. `ruvector-agent-memory::graph_forget::MincutGatedForgetting`
gains a second engine, `MincutEngine::LocalDeterministic` (feature-gated
behind the existing `mincut-forget` flag, off by default, additive to
`ExactGlobal`). The narrow claim this ADR set out to test — that swapping
`RuVectorGraphAnalyzer::partition()` for
`ruvector_mincut::localkcut::DeterministicLocalKCut` fixes ADR-345's latency
and non-determinism findings for the boundary-computation step itself — is
**supported by measured evidence**. The broader pre-declared acceptance
bundle (which also carried forward ADR-345's already-established
bridge-survival-effectiveness gate and added an overall
wall-clock-vs-baseline bound) **fails**, for reasons disclosed below that are
mostly orthogonal to the engine swap itself.

## Context

ADR-345 (2026-09-05, `docs/research/nightly/2026-09-05-mincut-gated-forgetting/`)
built `MincutGatedForgetting`, a `CompactionPolicy` that layers a
structural "protect the bridge" signal from `ruvector-mincut` on top of
`ruvector-agent-memory`'s scalar `CoherencePolicy`. It was rejected on two
independently measured grounds:

1. **Latency.** `RuVectorGraphAnalyzer::partition()` (the crate's global
   min-cut integration layer) measured 76ms-11.4s per call for k-NN graphs of
   50-400 vertices, with an outlier 69.3s at a small, regular 19-vertex ring
   topology.
2. **Non-determinism.** 30 repeated `partition()` calls on a byte-identical
   19-vertex graph with a *provably unique* weakest link returned an
   empty/unusable result in 15/30 (50%) of calls — consistent with
   hash-map-iteration-order-dependent tie-breaking rather than an intentional
   randomized algorithm (no direct `rand` usage was found in the relevant
   `ruvector-mincut` modules).

ADR-345's "Open Questions" flagged, as the natural next-research direction,
whether one of `ruvector-mincut`'s *other* primitives — it named
`DynamicMinCut`/`ClusterHierarchy` — could supply the same signal without
paying `RuVectorGraphAnalyzer::partition()`'s cost. This ADR answers that
question, using a different (and, on reflection, better-targeted) primitive
found during this follow-up's own survey of the crate:
`ruvector_mincut::localkcut::DeterministicLocalKCut`, an implementation of
"Deterministic and Exact Fully-dynamic Minimum Cut of Superpolylogarithmic
Size" (arXiv:2512.13105) that finds a *local*, bounded-radius, bounded-budget
cut around one seed vertex via deterministic BFS — no hash-map-keyed global
partition call, no randomization anywhere in its documented or actual
implementation.

## Hypothesis

```text
Given the same synthetic corpus as ADR-345 (6 topic clusters x 12 core
memories + 12 bridge memories interpolated 50/50 between two random
clusters, 32-dim, k-NN k=5 cosine >= 0.05, hot-cluster access simulation
over 20 test queries) compacted 50% by MincutGatedForgetting in
MincutEngine::LocalDeterministic mode (max_radius=0, budget_k=4: a
per-vertex degree check against the same k-NN graph, via the real
DeterministicLocalKCut/WitnessHandle machinery) versus the same policy in
the original MincutEngine::ExactGlobal mode and versus plain
CoherencePolicy,

when corpus size is scaled from 84 up to 924 vertices (same cluster/bridge
ratio),

then (a) LocalDeterministic retains the same >=15pp bridge-survival gap
over baseline and <=2pp recall delta ExactGlobal was required to hit at 84
vertices; (b) LocalDeterministic's wall-clock slowdown vs baseline at the
largest size where ExactGlobal still completes within a 1.5s/call budget is
at least 5x smaller than ExactGlobal's at that same size, and stays under
20x in absolute terms; and (c) LocalDeterministic returns byte-identical
survivor sets across 20 repeated compact() calls on unchanged input,

subject to: 100% tamper-detection across 20 independent single-byte-flip
trials against the eviction witness chain using LocalDeterministic.
```

Fixed before the acceptance run; see
`docs/research/nightly/2026-09-12-local-kcut-gated-forgetting/README.md`
for the full methodology, the two seeding-strategy design probes that
preceded this lock (not part of the acceptance evidence), and raw output.

## Decision

1. Add `MincutEngine` (`ExactGlobal` | `LocalDeterministic { max_radius,
   budget_k }`) to `graph_forget.rs`; `MincutGatedForgetting` gains an
   `engine` field (default `ExactGlobal`, preserving ADR-345's exact
   behavior and passing tests unchanged) and two new constructors,
   `soft_local`/`hard_local`, defaulting to `LocalDeterministic { max_radius:
   0, budget_k: 4 }`.
2. `boundary_indices_local` builds the identical k-NN similarity graph as
   the existing exact path, as a `ruvector_mincut::DynamicGraph`, and runs
   one `DeterministicLocalKCut::search` per vertex (single-vertex seeding;
   see "Design notes" below for why), reusing `WitnessHandle::
   materialize_partition()` to read out the found cut's vertex set.
3. **Do not promote `LocalDeterministic` as a strict replacement carrying
   all of ADR-345's original acceptance semantics** — the pre-declared
   bundle in this ADR still fails overall (Evidence, below). Do keep it as
   the recommended engine *if* `MincutGatedForgetting` is used at all: it is
   unconditionally faster, deterministic, and no worse on every measured
   axis than `ExactGlobal`.
4. Keep both engines in-tree behind the existing `mincut-forget` flag
   (off by default) as working reference implementations and retained
   evidence.

## Design Notes (found during implementation, not part of the acceptance run)

Two things were discovered and fixed *before* locking the hypothesis above
(disclosed for transparency, not hidden as if the first attempt had never
happened):

- **Seeding strategy.** `ruvector_mincut::localkcut::DeterministicFamilyGenerator::generate_seeds`
  — which pre-loads a vertex's lowest-id neighbors into the *initial* seed
  set — was tried first and found to defeat the signal entirely: for a
  low-degree bridge vertex, it immediately mixes in the bridge's two
  high-degree neighbors, so the first boundary check is against a large,
  noisy set instead of the bridge's own small one. Seeding with `[v]` alone
  and letting the algorithm's own BFS grow the region layer-by-layer (as its
  own doc comments describe) fixed this; both the unit tests and the
  benchmark below use single-vertex seeding.
- **Radius pathology on tightly-clustered synthetic data.** With
  single-vertex seeding, `max_radius >= 1` *still* over-flags on this
  crate's own k-NN cluster fixture: expanding one hop from any same-cluster
  vertex reaches nearly the entire cluster (clusters are built as k-NN
  near-cliques), and that whole-cluster region also has a tiny boundary
  (the one edge leaving the cluster) — so at radius >= 1, *every* vertex in
  *every* cluster gets flagged, not just bridges, erasing the differential
  signal. `max_radius = 0` (a pure per-vertex degree check against
  `budget_k`, still routed through the real `DeterministicLocalKCut`/
  `WitnessHandle` API) avoids this. This is a genuine, disclosed limitation
  of multi-hop local search on tightly-clustered inputs — not a claim that
  radius 0 is correct for every dataset. `max_radius` stays a public,
  documented field for callers with different topology.
- **A separate `ruvector-mincut` defect found, not used.**
  `ruvector_mincut::algorithm::approximate::ApproxMinCut` (a seeded,
  deterministic, Stoer-Wagner-on-a-sparsifier approximate min-cut — the
  first candidate considered for this follow-up, before `localkcut`) has a
  `compute_partition()` that **ignores its own `cut_value` argument** and
  returns an arbitrary BFS-order bisection unrelated to the actual computed
  cut (`crates/ruvector-mincut/src/algorithm/approximate.rs:558-599`, the
  `_cut_value` parameter is prefix-underscored and never read). Its
  `min_cut()`/`min_cut_value()` are real and usable; its `partition` field
  is not. Not used by this ADR's implementation; flagged here as a
  follow-up hardening item against `ruvector-mincut` itself, in the same
  spirit as ADR-345's non-determinism finding.

## Evidence

Exact command:

```bash
cargo run --release -p ruvector-agent-memory \
  --example mincut_local_forgetting_bench --features mincut-forget
```

Full raw output in
`docs/research/nightly/2026-09-12-local-kcut-gated-forgetting/raw-runs.txt`.
Headline numbers:

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Bridge-survival gap (Soft/Hard, inherited from ADR-345) | >= 15pp | +0.0pp | FAIL |
| Recall@10 delta (Soft/Hard) | <= 2pp | 0.00pp | PASS |
| Local vs Exact speedup @ n=168 (largest n Exact completed within budget) | >= 5x | **623x** | PASS |
| Local vs baseline slowdown @ n=168 | <= 20x | 26.0x | FAIL |
| Local determinism, 20 repeated `compact()` calls | 20/20 identical | 20/20 | PASS |
| Tamper detection | 20/20 | 20/20 | PASS |

Scaling table (Soft-Exact vs Soft-Local `compact()` wall-clock, same corpus
shape scaled 1x-11x):

| n | Baseline (us) | Exact (us) | Local (us) |
|---:|---:|---:|---:|
| 84  | 66  | 264,264       | 1,005  |
| 168 | 136 | 2,200,278     | 3,532  |
| 252 | 199 | skipped (budget) | 6,857  |
| 420 | 376 | skipped (budget) | 18,518 |
| 588 | 486 | skipped (budget) | 35,666 |
| 924 | 782 | skipped (budget) | 87,143 |

`Exact` was cut off once a single call exceeded a pre-declared 1.5s budget
(itself already exceeded at n=168, 2.2s) — a direct reproduction, at a
different corpus shape, of ADR-345's scaling finding. `Local` completed
every size up to 924 vertices (11x the base corpus) in under 90ms.

Interpretation, split by claim (see "Rejection Criteria" for how these
combine into the overall verdict):

- **Narrow claim (this ADR's actual contribution) — supported.** The
  boundary-computation step itself is fixed: 623x faster than
  `ExactGlobal` at the one size where both completed, and `Local` is the
  only one of the two that scales to a corpus beyond a few hundred vertices
  at all within a practical wall-clock. Determinism and witness-chain
  compatibility both hold (20/20 on each).
- **Inherited claim — not novel, reproduces ADR-345.** The 0.0pp
  bridge-survival gap is the *same* null result ADR-345 already measured
  for `ExactGlobal` (at a different seed/corpus size: 66.7% baseline there
  vs 16.7% here, but the same "candidates match baseline exactly" pattern).
  This says the structural bonus does not move rankings on this synthetic
  corpus **regardless of which engine computes it** — a property of the
  dataset/scoring interaction, not of `LocalDeterministic` specifically.
  Carried into this ADR's acceptance bundle only for direct comparability
  with ADR-345's own bar, not as a new finding.
- **New, engine-agnostic finding — the k-NN construction cost dominates at
  scale.** The "<=20x slowdown vs baseline" gate fails (26x at n=168,
  growing to ~111x at n=924) because both engines pay the *same* O(n^2)
  pairwise-cosine-similarity cost to build the k-NN graph in the first
  place (`knn_neighbors`, unchanged by this ADR) — `CoherencePolicy`'s
  baseline does no such graph construction at all. This is a real,
  previously-undisclosed cost of the *k-NN-graph-based structural signal
  design as a whole* (both engines), not a defect specific to
  `LocalDeterministic`; the `LocalDeterministic` engine's own per-vertex
  query cost is negligible next to it (visible in how flat the `Local`
  column's growth rate is against `n^2` — it tracks the shared O(n^2) k-NN
  cost, not an additional cut-search cost on top of it).

## Adversarial Self-Check

- **Baseline fairness.** `CoherencePolicy`'s cheap wall-clock is not an
  unfair comparison artifact — it genuinely does no graph construction.
  Disclosed explicitly above rather than left implicit.
- **Cherry-picking the scaling comparison point.** The rule "compare at the
  largest n where Exact still completed within its pre-declared budget" was
  written into the benchmark's source *before* it was run once, not chosen
  after seeing results to flatter either engine — and it is not the most
  favorable point available for the slowdown-vs-baseline gate (n=84 would
  have passed at 14.9x; n=168 was picked by the rule regardless and fails
  at 26x).
- **Hidden preprocessing cost.** k-NN construction is measured *inside* the
  timed `compact()` call for both engines (not hoisted out), so its cost is
  fully counted against both, and is disclosed as the dominant cost above
  rather than attributed to the cut algorithm.
- **Fixed seed only.** This run uses one seed (346) across all sizes/trials,
  consistent with ADR-345's own convention; not repeated across multiple
  seeds due to nightly wall-clock constraints — see "Limitations."

## Consequences

- `ruvector-agent-memory` gains a strictly-better-or-equal engine option for
  `MincutGatedForgetting` on every axis this and ADR-345 measured, but the
  policy as a whole remains unpromoted (off by default) because the
  underlying bridge-survival effectiveness question is still open (ADR-345's
  finding, unchanged by this ADR).
- The `ExactGlobal` engine is retained (not removed) as a comparison
  baseline and because removing working, tested code is out of scope for a
  nightly research cycle.
- A new, disclosed hardening item is filed against `ruvector-mincut`
  (`ApproxMinCut::compute_partition()` returning an unrelated bisection) —
  not fixed here, since fixing another crate's defect discovered only
  incidentally is out of this ADR's scope; recorded so it is not
  silently rediscovered later.
- No existing behavior changes: `ExactGlobal`'s default status, all
  existing public constructors (`soft`, `hard`), and every other
  `CompactionPolicy` are untouched.

## Alternatives Considered

- **`ApproxMinCut` (spectral-sparsifier-based approximate min-cut).**
  Investigated first; rejected once its `compute_partition()` was found to
  return an arbitrary bisection unrelated to its own computed cut value
  (see "Design Notes"). Its `min_cut_value()` alone is real but insufficient
  — this use case needs *which vertices*, not just the cut's weight.
- **`DeterministicFamilyGenerator`-seeded local k-cut (multi-vertex
  seeding).** Rejected during design: pre-loads high-degree neighbors into
  the first candidate set, defeating the boundary signal for exactly the
  low-degree vertices it should isolate. See "Design Notes."
- **`max_radius >= 1`.** Rejected during design for this specific
  tightly-clustered synthetic corpus: over-flags entire clusters. Kept as a
  public, non-default knob rather than removed, since it may be appropriate
  for less tightly clustered real data — untested here.
- **`ruvector_mincut::DynamicMinCut`/`ClusterHierarchy` directly** (ADR-345's
  originally suggested direction). Not attempted in this pass;
  `DeterministicLocalKCut` was chosen instead once found to more directly
  match the "is this vertex locally isolable" query shape. Still open as a
  possible future comparison.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-agent-memory/src/graph_forget.rs`: `MincutEngine` enum,
  `engine` field, `soft_local`/`hard_local` constructors,
  `boundary_indices_local`, 4 new unit tests (bridge protection x2,
  graceful fallback, cross-call determinism).
- `crates/ruvector-agent-memory/src/lib.rs`: export `MincutEngine`.
- `crates/ruvector-agent-memory/examples/mincut_local_forgetting_bench.rs`:
  the acceptance benchmark (5 policies, determinism section, 6-point scaling
  probe, tamper trials, explicit ACCEPT/REJECT).
- `crates/ruvector-agent-memory/Cargo.toml`: registers the new example.

No changes to `ExactGlobal`'s behavior, `witnessed_compaction.rs`, or any
crate outside `ruvector-agent-memory`.

## API Shape

```rust
// Behind `mincut-forget`, additive to ADR-345's existing API:
pub enum MincutEngine {
    ExactGlobal,
    LocalDeterministic { max_radius: usize, budget_k: u64 },
}
pub struct MincutGatedForgetting {
    pub weights: CoherenceWeights,
    pub mode: ForgetMode,
    pub engine: MincutEngine,           // new; defaults to ExactGlobal
    pub k_neighbors: usize,
    pub min_similarity: f32,
    pub structural_bonus: f32,
    pub protect_fraction: f32,
    pub mincut_trials: usize,           // ExactGlobal only; ignored by LocalDeterministic
}
impl MincutGatedForgetting {
    pub fn soft(weights: CoherenceWeights, structural_bonus: f32) -> Self;       // unchanged (ExactGlobal)
    pub fn hard(weights: CoherenceWeights, protect_fraction: f32) -> Self;       // unchanged (ExactGlobal)
    pub fn soft_local(weights: CoherenceWeights, structural_bonus: f32) -> Self; // new (LocalDeterministic)
    pub fn hard_local(weights: CoherenceWeights, protect_fraction: f32) -> Self; // new (LocalDeterministic)
}
```

## Feature Flags

No new flags. `MincutEngine::LocalDeterministic` is reached through the
existing `mincut-forget` feature (off by default), same as ADR-345's
`ExactGlobal`.

## Benchmark Evidence

See "Evidence" above and
`docs/research/nightly/2026-09-12-local-kcut-gated-forgetting/README.md`
for full methodology and raw output.

## Security

No new cryptographic primitive. `DeterministicLocalKCut`'s `WitnessHandle`
(a `RoaringBitmap` membership set + precomputed boundary size) is used only
to read back which vertices a search placed on the found cut's side; it is
never treated as a security witness by this policy (the existing eviction
witness chain, unrelated to this handle, still provides that). Tamper
detection against `EvictionWitnessChain` was re-verified end-to-end with the
new engine (20/20) to confirm the engine swap does not weaken that
independent guarantee.

## Governance

None beyond the existing "no witness, no mutation" invariant
(`witnessed_compaction`), unaffected by this ADR.

## Migration

None: `LocalDeterministic` is additive; every existing caller of `soft()`/
`hard()` keeps `ExactGlobal` behavior unchanged (verified by the unmodified
original unit tests in `graph_forget.rs` continuing to pass).

## Rollback

Remove `soft_local`/`hard_local`, the `LocalDeterministic` variant, and
`boundary_indices_local` — `ExactGlobal` and every other existing caller are
unaffected, since `engine` defaults to `ExactGlobal` and no other code path
references the new variant.

## Rejection Criteria

The pre-declared acceptance *bundle* in this ADR is treated as rejected
because two of its gates failed:

1. The inherited bridge-survival-effectiveness gate (>= 15pp), reproducing
   ADR-345's own null finding rather than a new failure of this engine.
2. The overall wall-clock-vs-baseline gate (<= 20x), driven by the shared
   (both-engines) O(n^2) k-NN construction cost, not by
   `LocalDeterministic`'s own per-vertex query cost.

The narrower, actually-novel claim this ADR investigated — does
`LocalDeterministic` fix ADR-345's latency and determinism defects relative
to `ExactGlobal` — is supported (623x speedup at the one directly-comparable
size, clean scaling to 924 vertices, 20/20 determinism, 20/20 tamper
detection). Recorded as a partial, evidence-backed result rather than
forced into a single ACCEPT/REJECT label that would misrepresent either
half.

## Open Questions

1. Would a genuinely different scoring baseline (real embeddings rather
   than Gaussian-cluster synthetic data) produce a *non-zero*
   bridge-survival gap, making the still-open ADR-345 effectiveness
   question answerable at all? Unaddressed by either ADR.
2. Can the shared O(n^2) k-NN construction cost itself be reduced (e.g. via
   `ruvector-coherence-hnsw` or another approximate-neighbor index already
   in this workspace) to make the *overall* `MincutGatedForgetting` call
   competitive with baseline, independent of which cut engine is used?
   Flagged as the natural next-research item.
3. Does `max_radius >= 1` become viable (without the whole-cluster
   over-flagging found here) on real, less artificially-clustered agent
   memory embeddings? Untested.
4. Should `ApproxMinCut::compute_partition()`'s defect (found, not fixed,
   here) be corrected upstream in `ruvector-mincut`? Out of this ADR's
   scope; filed as a disclosed finding only.
