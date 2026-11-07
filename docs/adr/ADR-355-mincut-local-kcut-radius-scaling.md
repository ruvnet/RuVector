# ADR-355: Bounded `DeterministicLocalKCut` Search Radius for `RuVectorGraphAnalyzer`

## Status

Accepted (additive only). New opt-in constructors
(`BoundedInstance::with_max_radius`, `RuVectorGraphAnalyzer::with_radius` /
`::from_knn_with_radius`), zero change to existing default behavior
(`BoundedInstance::new`, `RuVectorGraphAnalyzer::new` / `::from_knn`).

## Context

The 2026-09-05 nightly (ADR-345, `mincut-gated-forgetting`) rejected a
`ruvector-agent-memory` compaction policy built on
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` on latency grounds:
76ms-11.4s per call for ring k-NN graphs of 50-400 vertices, "far too
slow... 1,800-2,700x slower than the scalar baseline even at a
deliberately tiny 84-memory corpus." The 2026-09-11 follow-up (ADR-346)
fixed an unrelated non-determinism bug in the same code path and left,
as unresolved Next Research, a guess that the latency problem was the
top-level `DynamicConnectivity` fast path and that swapping it for the
in-tree, unused `connectivity::polylog::PolylogConnectivity` (implementing
arXiv:2510.08297) would fix it.

`RuVectorGraphAnalyzer::partition()` is shared production code. ADR-346's
own dependency audit lists 18 crates depending on `ruvector-mincut`
directly, including WASM/Node bindings, `ruvector-agent-memory`,
`ruvector-graph-condense`, `prime-radiant`, `cognitum-gate-kernel`, and
`mcp-brain-server`. A latency fix here benefits all of them; a wrong guess
about the fix benefits none of them and burns a future run re-discovering
the same dead end.

## Hypothesis

```text
Given fixed-degree ring k-NN graphs (k=8) at n in {50, 100, 200, 400},
reproducing the 2026-09-05 scaling probe exactly,

when the number of BoundedInstance lambda-ranges touched per query
(MinCutWrapper::num_instances(), already public) is measured alongside
partition() latency at each size,

then if instance count stays flat while latency keeps growing, the
top-level connectivity check (run once per query, independent of instance
count) cannot be the dominant cost, falsifying the prior run's specific
guess; and if a new RuVectorGraphAnalyzer::with_radius(graph, r) lowers
the DeterministicLocalKCut search radius (hardcoded to 20 before this
change) to 8, partition() latency should measurably drop at larger n,

subject to: the reported cut value matching the radius-20 production
default exactly on every tested graph (path, cycle, two cliques joined by
a single bridge edge, and all four ring sizes).
```

## Decision

1. Add `BoundedInstance::DEFAULT_MAX_RADIUS: usize = 20` (pulls the
   previously-inline magic number into a named constant) and
   `BoundedInstance::with_max_radius(lambda_min, lambda_max, max_radius)`.
   `BoundedInstance::new` now delegates to it with `DEFAULT_MAX_RADIUS`,
   producing byte-identical state to before this change.
2. Add `RuVectorGraphAnalyzer::with_radius(graph, max_radius)` and
   `::from_knn_with_radius(neighbors, max_radius)`, built on the
   pre-existing `MinCutWrapper::with_factory` (not new wrapper-level
   machinery), plus `::num_instances_touched()` (a passthrough of the
   already-public `MinCutWrapper::num_instances()`). `new()` /
   `from_knn()` are untouched.
3. **Do not change the production default radius.** The measured evidence
   supports radius=8 as a faster, correctness-preserving *option* on the
   graphs tested, not as a universally safe replacement default — see
   Evidence and Rejection Criteria below.
4. **Reject the `PolylogConnectivity` swap** proposed by the 2026-09-11
   nightly, for this specific latency problem: the instrumented probe
   shows the top-level connectivity check is not where `partition()`'s
   time goes on this workload, so swapping its backend would not have
   addressed the measured bottleneck. `PolylogConnectivity` remains
   in-tree and unused; this ADR does not claim it is useless in general,
   only that it was not the fix for *this* problem.

## Evidence

`cargo run --release -p ruvector-mincut --example mincut_radius_probe`
(ring k-NN, k=8, n in {50,100,200,400}, 3 reps/cell):

```text
n      variant           partition_ms   instances
50     baseline_r20            86.252          16
400    baseline_r20         12385.457          16
400    candidate_a_r8        3922.387          16
400    candidate_b_r4        2266.771          16
```

Instance count: flat at 16 across every n and every radius tested.
`partition()` latency at radius 20: ~86ms at n=50 to ~12.4s at n=400 (a
~112x). That combination is the direct falsification of the connectivity-
backend guess: a cost that scales with n while a count that would gate it
through that code path doesn't, cannot be sitting in the once-per-query
connectivity check. Lowering the radius to 8 gives 1.97x (n=200) and 3.16x
(n=400) speedups with zero change in reported cut value, checked both on
the same ring graphs (`cut_side_len` matched baseline at every size) and
on `cargo test --release -p ruvector-mincut --test radius_variant_tests`
(6/6 passed: path graph, cycle graph, bridged two-cliques, API-shape
checks).

## Consequences

Positive: any of the 18 dependent crates can opt into a measurably faster
`partition()`/`min_cut()` without any API break or migration, and a
specific dead-end ("swap the connectivity backend") is now closed with
evidence instead of left open as a guess. Negative: the new radius
parameter is a correctness/latency tradeoff a caller must understand
before using it; this ADR does not provide a universal safe default, only
a measured starting point (8) for the specific graph shapes tested.

## Alternatives Considered

1. **Swap `conn_ds` to `PolylogConnectivity`** — rejected; not supported
   by the measured bottleneck (see Evidence).
2. **Rewrite `DeterministicLocalKCut`'s internal algorithm** — out of
   scope; this is a parameter exposed on an existing, paper-grounded
   implementation, not a rewrite of it, keeping risk bounded.
3. **Change the production default to radius=8** — rejected; the test
   coverage (synthetic ring/path/cycle/bridge graphs) is not broad enough
   to justify changing behavior every existing caller inherits silently.
   An opt-in constructor keeps that decision with each caller.

## Implementation Plan

Already implemented in this change: `crates/ruvector-mincut/src/instance/
bounded.rs`, `crates/ruvector-mincut/src/integration/mod.rs`,
`crates/ruvector-mincut/examples/mincut_radius_probe.rs`,
`crates/ruvector-mincut/tests/radius_variant_tests.rs`. No feature flag
needed — purely additive public API, no behavior change to existing
entry points.

## API Shape

```rust
impl BoundedInstance {
    pub const DEFAULT_MAX_RADIUS: usize; // = 20
    pub fn with_max_radius(lambda_min: u64, lambda_max: u64, max_radius: usize) -> Self;
}

impl RuVectorGraphAnalyzer {
    pub fn with_radius(graph: Arc<DynamicGraph>, max_radius: usize) -> Self;
    pub fn from_knn_with_radius(neighbors: &[(usize, Vec<(usize, f64)>)], max_radius: usize) -> Self;
    pub fn num_instances_touched(&self) -> usize;
}
```

## Feature Flags

None required; no new dependency, no new unsafe code, no new
wasm/non-wasm divergence.

## Benchmark Evidence

See Evidence above and the full raw table in
`docs/research/nightly/2026_10_30_mincut-local-kcut-radius-scaling/README.md`.

## Security

No new attack surface: parameter generalization of an existing public
constructor, no new I/O or dependency. The only caller-visible risk (a
radius too small silently missing a valid cut) is opt-in and documented,
not hidden behind a changed default.

## Governance

Consistent with this harness's Darwin bounded-authority rule even without
CLI tooling to enforce it mechanically: one promotion (the new opt-in API)
out of two evaluated candidates (radius 8, radius 4), parent (radius 20)
retained as the unchanged default.

## Failure Modes

A radius too small for a given graph's actual shortest path to its true
minimum cut would make `BoundedInstance::query()` return `AboveRange` when
it should not, propagating to a wrong (too-large or `None`) `partition()`
result with no error signal distinguishing "wrong" from "disconnected."
Not observed in this run's test graphs, including one (bridged cliques)
specifically designed to stress it, but not exhaustively searched either —
see Next Research in the companion nightly doc.

## Migration

None required; fully additive.

## Rollback

Revert the two source-file diffs (`bounded.rs`, `integration/mod.rs`) and
delete the new example/test files; no data migration, no persisted state
touched by this change.

## Rejection Criteria

This ADR's decision would be reversed if: (a) a realistic (non-synthetic)
k-NN graph is found where radius=8 diverges from radius=20's reported cut
value, or (b) `DeterministicLocalKCut`'s own paper-grounded guarantees are
shown to require radius >= some value this change lets callers violate
silently. Neither has been checked against (a) and (b) is out of scope
for this run — both are listed under Next Research in the companion
nightly document.

## Open Questions

1. Does the radius/correctness tradeoff hold on real embedding-derived
   k-NN graphs (non-circulant)?
2. Is `find_bridges()` (same file, different and likely larger latency
   problem — re-solves per edge) amenable to the same fix?
3. Should radius be adaptive (escalate only on `AboveRange`) rather than a
   fixed caller-supplied constant?
