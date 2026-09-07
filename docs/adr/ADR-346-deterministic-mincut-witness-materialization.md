# ADR-346: Deterministic Minimum-Cut Witness Materialization

## Status

Accepted. Implemented and merged into `ruvector-mincut`'s default (always-on)
code path -- no feature flag, no opt-in required. Directly answers Open
Question #2 from the 2026-09-05 nightly (ADR-345, "Mincut-Gated
Forgetting"): *"What specifically causes the measured non-determinism inside
`ruvector-mincut` -- instance construction order, witness materialization,
or something else in the `MinCutWrapper::process_instances` path?"*

## Context

ADR-345 rejected `MincutGatedForgetting` (a `ruvector-agent-memory`
compaction policy built on `ruvector-mincut::RuVectorGraphAnalyzer`) for two
independent reasons: unacceptable latency scaling, and ~50% of repeated
`partition()` calls on a byte-identical 19-vertex graph returning an
empty/unusable result. That ADR flagged the non-determinism's root cause as
an open question, hypothesizing "hash-map iteration-order-dependent
tie-breaking" without confirming it.

This run root-causes that finding. Reproducing ADR-345's exact probe graph
first surfaced a bug in the probe itself (see "A trap in the original
probe" below); once corrected, the *true* non-determinism/degeneracy rate on
this repository's current `main` measured **100%**, not ~50% -- worse than
previously reported, and, more importantly, traceable to a concrete,
deterministic defect rather than pure randomness:

`WitnessHandle::materialize_partition()`
(`crates/ruvector-mincut/src/instance/witness.rs`) computes the complement
side `V \ U` as `(0..=membership.max()).filter(not in U)`. It derives the
vertex *universe* from the found cut set `U`'s own highest member, not from
the graph's actual vertex set. Because a minimum cut's small side `U`
essentially never happens to contain the graph's globally-highest vertex ID,
every vertex numbered above `U.max()` is silently dropped from **both**
returned sides -- not just omitted from `V \ U`, but absent from the output
entirely. `RuVectorGraphAnalyzer::partition()`
(`crates/ruvector-mincut/src/integration/mod.rs`) called this method
directly, so every consumer of `.partition()` inherited the defect.

A second, smaller contributor sits in `BoundedInstance`
(`crates/ruvector-mincut/src/instance/bounded.rs`): both
`brute_force_min_cut()`'s exhaustive subset search and
`search_for_cuts()`'s LocalKCut seed-vertex ordering derive their vertex
iteration order from `HashSet<VertexId>` (Rust's default `RandomState`
hasher, whose seed is randomized per `HashSet` instance). On a graph with
genuine automorphisms -- multiple vertex subsets tying for the same minimum
boundary value, which is common for near-duplicate embedding clusters --
"first tied subset encountered wins" means the *specific* winning subset
varies run to run even though the minimum cut *value* does not.

### A trap in the original probe

`mincut_determinism_probe.rs` (2026-09-05) built its 19-vertex graph with a
single interpolated "gateway" vector on one axis only. A correct two-cluster-
plus-bridge topology needs a gateway on *each* axis, or one cluster has no
path to the bridge at all and the graph is genuinely disconnected -- which
is indistinguishable, under that probe's `a.is_empty() || b.is_empty()`
check, from a witness-materialization bug. This run's probe
(`crates/ruvector-mincut/examples/determinism_probe.rs`) fixes that (two
gateways) and additionally checks `a.len() + b.len() == n`, which is what
actually exposed the `materialize_partition` defect as a *near-universal*,
not merely occasional, failure. This is recorded here as a methodology
lesson for whoever next builds a synthetic graph probe against this crate.

## Hypothesis

```text
Given the current `main` implementation of `RuVectorGraphAnalyzer::partition()`,
and a fixed, byte-identical connected graph (three topologies: n=19 vertices
via BoundedInstance's brute-force path, n=21 and n=85 via its LocalKCut-oracle
path),

when a fresh `RuVectorGraphAnalyzer` is constructed and `.partition()` called
for each of 200 independent trials per topology (matching ADR-345's
methodology of a new analyzer per trial, so no result cache masks the
defect),

then applying (A) a corrected witness-to-partition materialization that uses
the graph's real vertex set as its universe, and (B) deterministic
(sorted) vertex ordering in BoundedInstance's tie-breaking paths, should
reduce the empty-or-incomplete-partition rate from its measured baseline to
0% and the count of distinct partitions observed across the 200 trials (on
a fixed graph, ideally 1) to exactly 1,

subject to: the existing `ruvector-mincut` and `ruvector-agent-memory`
(`mincut-forget` feature) test suites remaining green, and no min-cut
*value* changing as a result (only which witness is returned, and whether it
is returned completely, should change).
```

Full raw probe output for baseline / fix-A-only / fix-A+B is in the nightly
README (linked below).

## Decision

1. Add `WitnessHandle::materialize_partition_within(&self, universe: &[VertexId])`
   (`crates/ruvector-mincut/src/instance/witness.rs`): computes `(U, V \ U)`
   against a caller-supplied vertex universe instead of guessing one from
   `membership.max()`. No feature flag; pure addition, no dependency change.
2. Fix `RuVectorGraphAnalyzer::partition()`
   (`crates/ruvector-mincut/src/integration/mod.rs`) to call
   `materialize_partition_within(&self.graph.vertices())` instead of the
   universe-guessing `materialize_partition()`. This is the actual bug fix;
   every direct and indirect consumer of `.partition()`
   (`CommunityDetector`, `GraphPartitioner`, and any downstream crate)
   inherits the correction automatically.
3. **Keep `materialize_partition()` unchanged and public**, with its
   documentation strengthened to explicitly warn about the
   vertex-dropping behavior, since it is used by existing doctests and may
   have out-of-tree callers who supply their own universe reasoning
   downstream. Removing or silently changing its behavior would be a
   breaking change beyond this ADR's scope; the new method is the
   recommended replacement for any caller that has (or can obtain) the
   graph's real vertex set.
4. Fix the two `HashSet`-iteration-order dependencies in
   `BoundedInstance` (`crates/ruvector-mincut/src/instance/bounded.rs`):
   sort `vertex_vec` in `brute_force_min_cut()` and `seed_vertices` in
   `search_for_cuts()` before using them for tie-breaking / search order.
5. Add regression tests: `witness_tests::materialize_partition_within_*`
   (witness.rs), `integration::tests::test_partition_deterministic_and_complete`
   (a 30-trial fresh-analyzer determinism check against a graph shaped like
   the bug's trigger condition -- an unbalanced cut whose small side omits
   higher-numbered vertices).
6. Ship `crates/ruvector-mincut/examples/determinism_probe.rs` as the
   reproducible probe backing this ADR's evidence table.

This ADR does **not** address `RuVectorGraphAnalyzer::partition()`'s latency
scaling (ADR-345's other, independent rejection reason) or `find_bridges()`'s
O(E) full-recompute-per-edge cost. Both remain open; see "Open Questions".

## Evidence

Methodology: `TRIALS=200` per topology, release build, `crates/ruvector-mincut/examples/determinism_probe.rs`,
one fresh `RuVectorGraphAnalyzer::from_knn(..)` per trial (no cache reuse
across trials), a fixed byte-identical k-NN graph per topology (k=8,
min_sim=0.05, two-cluster-plus-bridge topology with a gateway vector on each
axis). "Empty or degenerate" = returned partition has an empty side, or
`|side_a| + |side_b| != n` (vertices silently dropped). "Distinct
partitions" = number of unique canonicalized `(U, V\U)` pairs observed
across the 200 trials for a fixed graph -- 1 is the deterministic ideal.
Hardware: 4-core x86_64 Linux container, `rustc 1.94.1`, `--release`.

| Topology | Metric | Baseline (main) | Fix A only | Fix A + Fix B |
|---|---|---|---|---|
| n=19 (brute-force path) | empty/degenerate | 200/200 (100%) | 0/200 (0%) | 0/200 (0%) |
| n=19 | distinct partitions | 0 (all degenerate) | 2 | **1** |
| n=19 | avg latency/call | 1,236.9 ms | 1,196.9 ms | 1,167.6 ms |
| n=21 (LocalKCut-oracle path) | empty/degenerate | 200/200 (100%) | 0/200 (0%) | 0/200 (0%) |
| n=21 | distinct partitions | 0 | 2 | **1** |
| n=21 | avg latency/call | 0.2 ms | 0.2 ms | 0.2 ms |
| n=85 (LocalKCut-oracle path, >50 => cluster hierarchy) | empty/degenerate | 198/200 (99%) | 0/200 (0%) | 0/200 (0%) |
| n=85 | distinct partitions | 2 | 2 | **1** |
| n=85 | avg latency/call | 5.8 ms | 5.4 ms | 5.4 ms |

Interpretation:

- **Fix A alone (materialize_partition universe correction) eliminates 100%
  of empty/degenerate results** across all three topologies -- confirming
  this was the dominant defect, not primarily a randomization artifact.
  Residual non-determinism (2 distinct partitions across 200 trials) remains
  after Fix A alone: the min-cut *value* was always correct, but *which*
  tied-optimal witness got returned still varied.
  fix B (sorted tie-breaking) closes that gap: **exactly 1 distinct
  partition observed across all 600 trials (3 topologies x 200) once both
  fixes are applied.**
- Latency is essentially unchanged by either fix (as expected -- sorting a
  vector of <100 elements is not the bottleneck). The n=19 case's ~1.2s/call
  outlier matches ADR-345's "outlier 69s at n=19 on a small regular
  topology" finding qualitatively (a pathologically slow case at small n on
  the brute-force/LocalKCut boundary) and is **not fixed by this ADR** --
  see "Open Questions".

Correctness / regression evidence:

- `cargo test --release -p ruvector-mincut --lib`: 515 passed, 0 failed, 5
  ignored (pre-existing, unrelated -- one is a documented known bug in
  `witness::tests::test_delete_tree_edge`, tracked separately per its own
  in-code comment referencing a prior PR; the other four are `#[ignore]`d
  doctests in `wrapper/mod.rs` unrelated to this change).
- `cargo test --release -p ruvector-mincut --doc`: 27 passed, 0 failed
  (includes the new `materialize_partition_within` doctest).
- `cargo test --release -p ruvector-agent-memory --features mincut-forget`
  (the downstream consumer from ADR-345): 31 lib tests + 32 integration
  tests (1 bench_tests + 10 arbitration + 8 atomic_observation_fusion + 13
  tarl_ledger), all passed, including
  `graph_forget::tests::soft_mode_protects_the_structural_bridge` and
  `hard_mode_reserves_budget_for_boundary_vertices`.
- `cargo clippy --release -p ruvector-mincut --lib`: no new warnings.
- `cargo fmt -p ruvector-mincut -- --check`: clean after applying `cargo fmt`.

## Consequences

- Every existing and future consumer of `RuVectorGraphAnalyzer::partition()`
  (`CommunityDetector::detect`, `GraphPartitioner::partition`, and any
  out-of-tree caller) now gets a complete, deterministic partition for a
  fixed graph, with no code changes required on their part.
- This directly removes one of ADR-345's two independent rejection causes
  for `MincutGatedForgetting` (non-determinism). The other -- latency
  scaling -- is untouched, so `MincutGatedForgetting` remains correctly
  un-promoted; this ADR does not reopen that decision. It does mean a future
  attempt at that hypothesis (or at ADR-345's Open Question #1, using
  `DynamicMinCut`/`ClusterHierarchy` directly) starts from a codebase whose
  witness plumbing is no longer independently broken, isolating any future
  finding to the actual latency/algorithm-selection question.
- `materialize_partition()` is left in place (not removed or behaviorally
  changed) to avoid an unscoped breaking change; its documentation now
  makes the defect explicit rather than merely hinting at it ("For sparse
  graphs, V \ U may contain vertex IDs that don't exist" understated the
  actual failure mode, which is *omission*, not merely *extraneous IDs*).
- No new dependency, no new feature flag, no API removal.

## Alternatives Considered

- **Change `materialize_partition()`'s behavior in place** (compute the
  universe from the graph rather than from `U`). Rejected: the method has no
  access to a graph reference (only to the implicit witness), so it cannot
  fix itself without a signature change, which would be a breaking API
  change for any external caller relying on its current (buggy but stable)
  signature. Adding `materialize_partition_within` alongside it is additive
  and lets `RuVectorGraphAnalyzer` (which does have graph access) opt in
  immediately.
- **Replace `HashSet<VertexId>` with `BTreeSet<VertexId>` throughout
  `BoundedInstance`.** Considered as a more "structural" fix than sorting at
  point of use. Rejected for this pass: `BTreeSet` changes the type of
  `self.vertices` (a struct field touched by `insert`/`delete`/`is_connected`
  elsewhere in the same file), a wider-blast-radius change than this ADR's
  scope warrants for a fix that sort-at-use already resolves completely per
  the evidence above. Left as a possible future hardening, not required by
  the measured evidence.
- **Do nothing, on the grounds that `MincutGatedForgetting` is already
  rejected and unused in production.** Rejected: `RuVectorGraphAnalyzer` is
  the ecosystem's general-purpose graph-partitioning integration point
  (also used by `CommunityDetector`, `GraphPartitioner`, and by the
  `ruvector-namespace-merge` nightly's mincut integration), not solely by
  the rejected agent-memory policy. Silent data loss in `.partition()`'s
  return value is a live correctness bug regardless of which nightly
  experiment first exposed it.

## Implementation Plan

Already implemented in this PR:

- `crates/ruvector-mincut/src/instance/witness.rs`:
  `WitnessHandle::materialize_partition_within`, plus strengthened docs on
  `materialize_partition`, plus `witness_tests` module (2 new tests).
- `crates/ruvector-mincut/src/integration/mod.rs`: `partition()` fixed to
  use `materialize_partition_within(&self.graph.vertices())`; new
  `test_partition_deterministic_and_complete` regression test.
- `crates/ruvector-mincut/src/instance/bounded.rs`: sorted `vertex_vec` in
  `brute_force_min_cut()`, sorted `seed_vertices` in `search_for_cuts()`.
- `crates/ruvector-mincut/examples/determinism_probe.rs`: the reproducible
  probe backing this ADR's evidence table.

No further implementation is planned under this ADR; latency scaling and
`find_bridges()`'s recomputation cost are out of scope (see "Open
Questions").

## API Shape

```rust
impl WitnessHandle {
    // New; the correct replacement for materialize_partition() when the
    // caller has (or can obtain) the graph's real vertex set.
    pub fn materialize_partition_within(
        &self,
        universe: &[VertexId],
    ) -> (HashSet<VertexId>, HashSet<VertexId>);

    // Unchanged signature; documentation now explicit about the
    // vertex-dropping behavior.
    pub fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>);
}
```

`RuVectorGraphAnalyzer::partition()`'s signature is unchanged
(`Option<(Vec<VertexId>, Vec<VertexId>)>`); only its internal implementation
changed.

## Feature Flags

None. This is a default-on bug fix in `ruvector-mincut`'s existing public
API, gated behind no feature flag (the crate has no relevant feature to gate
it behind -- `RuVectorGraphAnalyzer` is unconditionally compiled).

## Benchmark Evidence

See "Evidence" above for the full before/after table and raw methodology.
Raw command: `TRIALS=200 cargo run --release -p ruvector-mincut --example determinism_probe`,
run three times (unfixed / fix-A-only / fix-A-plus-B) against the identical
probe source via `git stash` to isolate the fix under test.

## Security

No new cryptographic or witness-signing primitive. `materialize_partition_within`
is a pure function over already-public data (`WitnessHandle::contains`,
already `pub`); it does not change what information a witness exposes, only
how completely a caller reconstructs the two sides. Fixing silent vertex
loss is itself a hardening: any downstream witness-consuming code that
audits "every vertex accounted for" (e.g. a future eviction-witness
extension to `ruvector-agent-memory::witnessed_compaction`) can now rely on
`.partition()`'s output actually summing to the graph's vertex count.

## Governance

None beyond ordinary code review. No change to any existing invariant,
ledger, or witness-chain schema.

## Failure Modes

- **Latency at small n (the n=19 case, ~1.2s/call) is unresolved by this
  ADR** and remains a real, measured problem: `BoundedInstance`'s brute-force
  path is exhaustive-subset (`O(2^n)`) and the wrapper's `process_instances`
  loop can invoke it redundantly across dozens of `[lambda_min, lambda_max]`
  ranges that all collapse to `[1, 1]` for small `i` (see ADR-345 and Open
  Question #1 below). This ADR only confirms the fix does not worsen it.
- `materialize_partition()` (the old method) is still present and still has
  its documented defect for any caller that doesn't migrate to
  `materialize_partition_within`. This is a known, accepted, and now
  clearly documented limitation, not a regression introduced here.
- The determinism fix is verified against three synthetic topologies (two
  cluster sizes, two BoundedInstance code paths) and the existing test
  suite; it has not been verified against adversarially constructed graphs
  designed to defeat sorted tie-breaking (e.g. via crafted vertex-ID
  collisions), which is out of scope for a correctness fix of this shape.

## Migration

None required. `RuVectorGraphAnalyzer::partition()` callers get the fix
automatically with no code change. Callers of `WitnessHandle::materialize_partition()`
directly are unaffected (same behavior as before, now better documented);
they may opt into `materialize_partition_within` at their own pace.

## Rollback

Revert the four changed files
(`witness.rs`, `integration/mod.rs`, `bounded.rs`, `examples/determinism_probe.rs`).
No other code depends on `materialize_partition_within` yet (this PR is its
only caller), so rollback is a clean, isolated revert with no cascading
effect on other crates.

## Rejection Criteria

Not applicable -- this ADR's hypothesis was accepted. For completeness, it
would have been rejected if either had held:

1. Fix A + Fix B failed to bring the empty/degenerate rate to 0% on any of
   the three tested topologies. (Measured: 0% on all three.)
2. Fix A + Fix B failed to bring distinct-partitions-per-200-trials to
   exactly 1 on any topology, i.e. genuine non-determinism remained.
   (Measured: exactly 1 on all three.)
3. Either fix caused a regression in the existing `ruvector-mincut` or
   `ruvector-agent-memory` (`mincut-forget`) test suites. (Measured: none.)

## Open Questions

1. **Latency scaling** (ADR-345's other rejection cause, and this ADR's
   n=19 outlier) remains unaddressed. ADR-345's Open Question #1 -- whether
   `ruvector_mincut::DynamicMinCut`/`ClusterHierarchy`, used directly
   instead of going through `MinCutWrapper`'s O(log n)-bounded-instance
   sweep, avoids this cost -- is still open and is the natural next nightly
   topic in this lineage.
2. `find_bridges()` still recomputes a full `MinCutWrapper::query()` per
   edge (O(E) full min-cut recomputations); unaffected by this ADR, flagged
   in ADR-345 as a known-worse alternative to `.partition()`, not fixed
   here.
3. Whether replacing `HashSet<VertexId>` with an order-preserving or
   sorted-by-construction collection throughout `BoundedInstance` (rather
   than sorting at each point of use) is worth the wider diff, given the
   evidence above shows sort-at-use is already fully sufficient.
