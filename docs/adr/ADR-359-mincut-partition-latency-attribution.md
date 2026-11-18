# ADR-359: Attributing `BoundedInstance::partition()` Latency — Reject the `PolylogConnectivity` Backend-Swap Fix

- **Status**: Rejected (the proposed fix); instrumentation accepted and merged
- **Date**: 2026-11-11
- **Deciders**: Nightly research run (autonomous), recorded for `ruv` (owner) review
- **Related**: ADR-345 (`mincut-gated-forgetting`, the consumer this performance question blocks); `docs/research/nightly/2026-09-11-mincut-partition-determinism` (source of the proposal this ADR evaluates)
- **Tags**: ruvector-mincut, performance, dynamic-connectivity, benchmarking, falsification
- **Numbering**: ADR-358 was the highest ADR on `main` at session start; 359 was free.

## Context

ADR-345 measured `RuVectorGraphAnalyzer::partition()` (the integration-layer
entry point into `ruvector-mincut`'s dynamic min-cut engine) scaling from
~77ms at n=50 vertices to ~11.4s at n=400, on a synthetic ring k-NN graph.
That performance problem is one of the two reasons ADR-345 rejected
`ruvector-agent-memory::graph_forget::MincutGatedForgetting` for production
use. The root cause was left unattributed.

The 2026-09-11 nightly (`docs/research/nightly/2026-09-11-mincut-partition-determinism`)
fixed a separate, unrelated non-determinism bug in the same code
(`BoundedInstance::search_for_cuts`'s seed ordering) and, in its own "Next
Research" list, proposed a specific fix for the still-open latency
question: "root-cause the `partition()` latency scaling problem from
ADR-345 ... likely by evaluating
`connectivity::polylog::PolylogConnectivity` (already in-tree, not yet
wired into `BoundedInstance`) as a replacement backend."

That proposal had not been tested. This ADR records the result of testing
it.

## Hypothesis

```text
Given the same ring k-NN corpus (k=8, n in {19,50,100,200,400}) ADR-345's
own scaling probe used,

when BoundedInstance's connectivity-check backend is replaced with
connectivity::DynamicConnectivity (already in-tree, used elsewhere) or
connectivity::polylog::PolylogConnectivity (the backend named by the
proposal),

then partition() wall-clock time at n=400 should measurably improve
relative to baseline, by a margin plausibly consistent with explaining
ADR-345's 77ms->11.4s scaling,

subject to: the swap not changing is_connected()'s boolean result on any
tested graph, and call-site instrumentation (not assumption) confirming
where partition()'s time is actually spent.
```

## Decision

**Reject the proposed fix.** Do not wire `PolylogConnectivity` (or
`DynamicConnectivity`) into `BoundedInstance` as a connectivity-check
backend. **Accept and merge the instrumentation** built to test the
proposal, as a permanent, default-off diagnostic capability:

1. `crates/ruvector-mincut/src/profiling.rs` — atomic counters for
   connectivity-check calls/cost, oracle-search calls, and per-instance
   attempts, gated behind a new `profile-counters` Cargo feature (default
   off, zero cost when disabled).
2. Three call sites instrumented (no behavior change):
   `BoundedInstance::is_connected` (`instance/bounded.rs`),
   `BoundedInstance::search_for_cuts`'s oracle invocation
   (`instance/bounded.rs`), and `MinCutWrapper::process_instances`'s
   per-instance loop (`wrapper/mod.rs`).
3. Two new benchmark examples:
   `examples/profile_partition_scaling.rs` (reruns ADR-345's exact corpus
   with attribution) and `examples/connectivity_backend_bench.rs`
   (standalone three-way: baseline BFS vs. `DynamicConnectivity` vs.
   `PolylogConnectivity`, build-from-scratch-and-query-once, matching the
   only usage pattern available without further engineering).

## Evidence

`profile_partition_scaling` (release build, same corpus as ADR-345):

| n | partition_ms | oracle_calls | conn_calls | conn time contribution |
|---|---|---|---|---|
| 50 | 69.195 | 1,501 | 16 | 0.259% |
| 100 | 364.540 | 3,001 | 16 | 0.144% |
| 200 | 1,360.473 | 6,001 | 16 | 0.050% |
| 400 | 4,454.035 | 12,001 | 16 | 0.028% |

`connectivity_backend_bench` (mean of 20 reps, microseconds, release build):

| n | baseline (BFS) | DynamicConnectivity | PolylogConnectivity |
|---|---|---|---|
| 50 | 24.38 | 244.27 (10.0x) | 479.11 (19.7x) |
| 400 | 222.07 | 11,274.93 (50.8x) | 2,424.27 (10.9x) |
| 800 | 371.70 | 43,753.29 (117.7x) | 5,369.59 (14.5x) |

Both tables are read verbatim from this run's raw command output
(`docs/research/nightly/2026-11-11-mincut-partition-latency-attribution/README.md`
reproduces the full stdout; no numbers here were transcribed from a
different run).

Two independent findings, either sufficient on its own to reject the
hypothesis:

1. The connectivity check is called at most 16 times per `partition()`
   call and costs under 0.26% of wall time at every size — there is no
   room for a backend swap at this call site to matter.
2. Both candidate backends are 10x-118x *slower* than the current naive
   BFS when used the only way `BoundedInstance` could use them (built
   from scratch, queried once) — implementing the proposed swap would
   make `partition()` worse, not better.

A newly-identified, correctly-scoped lead for the real bottleneck: oracle
call count (`DeterministicLocalKCut::search`) scales linearly with n
(`~30n` calls) but `partition_ms` scales roughly quadratically over the
same range — each oracle call is independently getting more expensive as
n grows, not just more numerous. This ADR does not attempt to fix that;
see Open Questions.

## Consequences

- `ruvector-agent-memory::graph_forget::MincutGatedForgetting` remains
  rejected per ADR-345; this ADR narrows, but does not resolve, its
  performance blocker. A future attempt should target oracle call
  volume/cost, not the connectivity-check backend.
- `ruvector-mincut` gains permanent, low-risk instrumentation
  (`profiling` module, `profile-counters` feature) usable by any future
  performance investigation of `BoundedInstance`/`MinCutWrapper` without
  re-deriving ad hoc timing code.
- `connectivity::DynamicConnectivity` and `connectivity::polylog::PolylogConnectivity`
  are confirmed correct (agree with baseline BFS on every tested graph)
  but confirmed unsuitable for a build-once-query-once usage pattern — a
  reusable finding for any other consumer in this crate considering either
  structure.
- No existing behavior changes. `BoundedInstance`, `MinCutWrapper`,
  `RuVectorGraphAnalyzer`'s public APIs and outputs are untouched;
  `profile-counters` is additive and off by default.

## Alternatives Considered

- **Implement the incremental version anyway** (maintain
  `DynamicConnectivity`/`PolylogConnectivity` as persistent
  `BoundedInstance` fields, synced through `insert`/`delete`). Rejected:
  finding (1) means even a free connectivity check would not help; the
  non-trivial correctness-sensitive wiring this would require is not
  justified by the evidence.
- **Fix the oracle call-volume bottleneck in this same run.** Rejected as
  out of scope: this run's mandate was to test one specific, previously
  named hypothesis, not to begin a new optimization effort against
  correctness-critical oracle internals. Recorded as Next Research
  instead.
- **Skip measuring and just implement the proposed swap**, since it was
  "likely" per the prior nightly's phrasing. Rejected on principle: the
  entire point of this run is to replace "likely" with a measured answer,
  per the nightly process's own evidence-over-plausibility rule.

## Implementation Plan

Already implemented and merged in this ADR's companion commits:
`profiling.rs`, the three instrumentation call sites, the
`profile-counters` feature flag, and the two benchmark examples. No
further implementation is planned under this ADR; see Next Research in
the companion nightly document for the follow-on work this ADR scopes out.

## API Shape

```rust
// crates/ruvector-mincut/src/profiling.rs
pub fn record_connectivity_check(nodes_visited: u64);
pub fn record_oracle_search_call();
pub fn record_instance_tried();

#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct Snapshot {
    pub connectivity_check_calls: u64,
    pub connectivity_check_nodes_visited: u64,
    pub oracle_search_calls: u64,
    pub instances_tried: u64,
}
pub fn snapshot() -> Snapshot;
pub fn reset();
```

All functions are no-ops (compiling to nothing) unless the
`profile-counters` feature is enabled.

## Feature Flags

- `profile-counters` (new, default off) — `ruvector-mincut/Cargo.toml`.
  Enables the atomic counters; `cargo test`/`cargo build` behavior and
  output are otherwise identical with or without it (verified: 519 tests
  pass, 3 ignored, both ways).

## Benchmark Evidence

See Evidence above and the companion nightly document's "Benchmark
Results (raw)" section for full command lines and unedited stdout.

## Security

No security-relevant surface changed. The three instrumented call sites
are read-only observers; counters are process-global non-sensitive
integers (call counts) with no new timing side-channel beyond what a
caller could already construct by timing the existing public
`partition()` call themselves.

## Governance

No public API removed or changed. The `profiling` module is new and
additive; existing consumers (`ruvector-agent-memory`, any external user
of `RuVectorGraphAnalyzer`/`MinCutWrapper`) are unaffected whether or not
they enable `profile-counters`.

## Failure Modes

- If a future caller enables `profile-counters` in a multi-threaded
  context and calls `reset()` concurrently with in-flight `record_*`
  calls on another thread, counts can be dropped or double-counted
  (documented as unsupported in `profiling.rs`'s doc comment). Not a
  concern for the single-threaded benchmark harnesses this run ships,
  and not exercised in any multi-threaded production path today (no
  existing caller invokes `BoundedInstance`/`MinCutWrapper` from more
  than one thread concurrently per instance).
- If this instrumentation is later extended to cover
  `DeterministicLocalKCut::search()`'s internals (Next Research item 1 in
  the companion document) without care, it could add measurable overhead
  inside the hottest loop in the whole crate; any such extension should
  re-run this ADR's same before/after test-count check
  (519 passed / 3 ignored either way) plus a wall-clock regression check
  with the feature off.

## Migration

None required. Purely additive; no existing call site's signature or
behavior changed.

## Rollback

Revert the three call-site edits and the `profiling` module/feature; no
other code depends on them. Low risk given their read-only nature.

## Rejection Criteria

This ADR's own hypothesis was rejected per its stated falsification
criteria (see Hypothesis / Evidence above): connectivity-check
instrumentation showed <0.3% time contribution at every size, and the
standalone backend bench showed both candidates are slower than baseline
in the only usage pattern available. Both criteria were specified before
the benchmarks were run (see the companion nightly document's Hypothesis
section) and neither was altered after seeing results.

## Open Questions

1. Why does each `oracle.search()` call get more expensive as n grows,
   given `DeterministicLocalKCut`'s search is nominally radius-bounded
   (radius=20)? Not instrumented at the sub-call level in this run.
2. Would reordering or parallelizing `search_for_cuts`'s
   `for budget { for seed { oracle.search(...) } }` loop (e.g., with
   `rayon`, already a dependency) reduce wall time without changing
   which cut is found (determinism must be preserved per the
   2026-09-11 nightly's fix)?
3. Does fixing (1) or (2) change ADR-345's rejection of
   `MincutGatedForgetting`? Only testable once a fix exists; out of
   scope for this ADR.
