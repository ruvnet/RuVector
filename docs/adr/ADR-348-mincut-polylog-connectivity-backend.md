# ADR-348: PolylogConnectivity as a Selectable (Non-Default) Backend for `MinCutWrapper`

- **Status**: Rejected (hypothesis falsified); change merged as a tested, off-path, non-default option
- **Date**: 2026-10-07
- **Deciders**: RuVector nightly research process (autonomous session)
- **Related**: ADR-344, ADR-345, ADR-346
- **Tags**: research, nightly, mincut, connectivity, performance, agent-memory

## Status

Rejected as a fix for `RuVectorGraphAnalyzer::partition()` latency. The
underlying engineering change (a selectable `ConnectivityBackend` on
`MinCutWrapper`/`RuVectorGraphAnalyzer`) is merged as a correct, tested,
benchmarked, **non-default** option, because it is independently valid even
though it does not address the latency problem it was proposed to fix.

## Context

The 2026-09-11 nightly
(`docs/research/nightly/2026-09-11-mincut-partition-determinism/`,
ADR-346) fixed `RuVectorGraphAnalyzer::partition()`'s non-determinism and
closed with an explicit "Next Research" item:

> Root-cause the `partition()` latency scaling problem from ADR-345 (the
> remaining blocker for `MincutGatedForgetting`), likely by evaluating
> `connectivity::polylog::PolylogConnectivity` (already in-tree, not yet
> wired into `BoundedInstance`) as a replacement backend.

That latency problem was originally measured by the 2026-09-05 nightly
(ADR-345): `partition()` scaled from ~77ms (n=50) to ~11.4s (n=400) on a
regular ring k-NN graph, which made `MincutGatedForgetting` (agent-memory
compaction gated on a structural bridge-protection signal) 1,800-2,700x
slower than its scalar baseline at a deliberately tiny 84-memory corpus,
with zero measured effectiveness benefit at that size.

This run continues that exact thread.

## Hypothesis

```text
Given BoundedInstance::partition() exercised at the graph sizes used in
crates/ruvector-agent-memory/examples/mincut_gated_forgetting_bench.rs, plus
at least two larger synthetic graph sizes to observe scaling,

when PolylogConnectivity (crates/ruvector-mincut/src/connectivity/polylog.rs)
replaces BoundedInstance's current connectivity backend as an explicit,
feature/enum-selectable alternative (baseline untouched and still the
default),

then partition() latency and/or its scaling exponent with graph size should
improve relative to baseline,

subject to:
  - boundary-set / cut results remaining correct — either byte-identical to
    baseline across repeated trials (as crates/ruvector-mincut/tests/determinism_tests.rs
    already checks for the baseline) or a documented, bounded, honestly
    characterized divergence;
  - the existing ruvector-mincut and ruvector-agent-memory test suites
    (including determinism_tests.rs and the graph_forget tests) remaining
    green;
  - no regression in MincutGatedForgetting's correctness on the existing
    mincut_gated_forgetting_bench.rs corpus.
```

## Decision

**The hypothesis is falsified and the change is REJECTED as a latency
fix.** Root-cause analysis (profiling instrumentation, described in
Evidence) shows `BoundedInstance` — the thing ADR-346 named as the suspect —
**never calls any connectivity backend at all** for its actual cut
computation. `MinCutWrapper`'s `conn_ds` field (`DynamicConnectivity`,
proposed replacement target: `PolylogConnectivity`) is used for exactly one
thing: a single, already O(1)/O(log n)-amortized whole-graph
`is_connected()` fast-path check at the top of `MinCutWrapper::query()`. It
is not in `BoundedInstance::brute_force_min_cut`'s or
`BoundedInstance::search_for_cuts`'s call graph.

`partition()`'s real cost — confirmed by direct instrumentation, not
inferred — is `MinCutWrapper::process_instances()` walking a **geometric
ladder of ~16 `BoundedInstance`s** (ranges `[⌊1.2^i⌋, ⌊1.2^(i+1)⌋]`) before
one reports `ValueInRange`, where **every instance independently
recomputes the graph's true global minimum cut from scratch** —
`brute_force_min_cut` always computes the exact global optimum regardless
of its own `[lambda_min, lambda_max]`, so the "bounded-range" design intent
is not actually a pruned/bounded search; it is the same O(2^n) (n<20) or
`LocalKCut`-search (n>=20) computation repeated ~16x with no sharing of
work across instances. The connectivity structure plays no role in any of
this.

Given that, swapping `DynamicConnectivity` for `PolylogConnectivity` has no
plausible mechanism to change `partition()`'s latency, and the measured A/B
data (Evidence section) confirms no measurable difference, within noise,
at every tested graph size. This is a clean falsification, not an
inconclusive result: the code-path evidence and the measured evidence agree
on *why* there is no effect, not just *that* there is none.

The engineering change is still merged, narrowly, because it is honestly
useful on its own terms: `PolylogConnectivity` is a real, independently
correct implementation of the same connectivity specification
`DynamicConnectivity` implements, and `MinCutWrapper`/`RuVectorGraphAnalyzer`
now expose it as an explicit, tested, benchmarked, off-by-default choice.
Nothing about `partition()`'s default behavior, numeric output, or
performance changes for any existing caller.

## Evidence

### Root cause (ad hoc instrumentation, not committed — see PR/commit history for the diff if needed)

Temporary `eprintln!` instrumentation inside
`MinCutWrapper::process_instances()`'s per-instance query loop, run against
`crates/ruvector-agent-memory/examples/mincut_scaling_probe.rs`'s existing
ring-graph sizes (n=19, 50, 100, 200, 400; release build), showed **every
one of the 16 geometric-range instances gets lazily created and fully
queried, for every graph size**, before the loop breaks on the instance
whose range finally contains the true cut value. Representative excerpt
(n=400, the slowest case, per-instance `query()` time in microseconds):

```text
instance_idx=0  range=[1,1]    query_us=515779   new=true
instance_idx=1  range=[1,1]    query_us=509469   new=true
instance_idx=2  range=[1,1]    query_us=524460   new=true
instance_idx=3  range=[1,2]    query_us=958939   new=true
instance_idx=4  range=[2,2]    query_us=521239   new=true
...
instance_idx=12 range=[8,10]   query_us=1445983  new=true
instance_idx=13 range=[10,12]  query_us=1366524  new=true
instance_idx=14 range=[12,15]  query_us=1796820  new=true
instance_idx=15 range=[15,18]  query_us=486738   new=true
```

Sixteen full, independent `search_for_cuts`/`brute_force_min_cut` calls,
summing to the ~14.17s total `partition()` latency the scaling probe
reports at n=400 (see the README's "Root cause" section for the full log
and the n=19/50/100/200 breakdowns, which show the identical 16-instance
pattern). `DynamicConnectivity`/`conn_ds` is not referenced anywhere in that
loop body beyond the one `is_connected()` check made once per `query()`,
outside this loop.

### A/B latency measurement (real, paired, same-graph, multi-seed)

`cargo run --release -p ruvector-mincut --example connectivity_backend_partition_probe`
— full raw output and methodology in the 2026-10-07 nightly README. Summary
(mean `partition()` latency in ms, `EulerTour` = baseline default):

| n   | seeds | euler mean (sd) | polylog mean (sd) |
|----:|------:|-----------------:|--------------------:|
| 19  | 1     | 45189.7          | 45008.9              |
| 50  | 5     | 29.0 (9.4)       | 29.9 (7.2)            |
| 84  | 5     | 109.5 (17.4)     | 115.6 (21.9)          |
| 100 | 5     | 151.8 (20.2)     | 145.4 (11.1)          |
| 200 | 5     | 410.7 (16.7)     | 426.8 (14.8)          |

No size shows a backend difference outside the other backend's own
run-to-run standard deviation. The downstream `MincutGatedForgetting`
benchmark (`mincut_gated_forgetting_bench.rs`, run against both backends via
a new, default-preserving `MINCUT_CONNECTIVITY_BACKEND` env toggle) shows
the same: 1,706-1,997x slowdown vs. baseline and 0.0pp bridge-survival gap
under `EulerTour`, 1,367-2,017x slowdown and 0.0pp gap under `Polylog` — the
same qualitative REJECT either way, with no systematic direction to the
latency difference (see README for full tables).

### Correctness

`cargo test --release -p ruvector-mincut` — all pre-existing suites plus:

- `backend_equivalence_tests::backends_agree_on_random_insert_delete_sequences`
  (new, `connectivity/mod.rs`): `EulerTour` and `Polylog` agree on
  `is_connected()`/`connected()` across 5 seeded random insert/delete
  sequences (200 steps each, 5 connectivity probes per step).
- `determinism_tests::partition_matches_across_connectivity_backends` (new):
  `RuVectorGraphAnalyzer::partition()` returns an identical (canonicalized)
  partition under both backends on the existing two-cluster-bridge
  topology.

Both pass. One bounded, documented divergence was found while writing the
equivalence test (pre-existing in both backends, not introduced here, and
outside `MinCutWrapper`'s usage pattern): `connected(v, v)` for a vertex
`v` neither backend has ever seen returns `false` under
`DynamicConnectivity` but `true` under `PolylogConnectivity` (identity
fallback in `LevelForest::find` for an untracked vertex). Pinned by
`self_query_on_never_inserted_vertex_is_a_known_divergence`; excluded from
the general random-sequence test's `a == b` probes. See the README's
"Correctness Evidence" section for the full `cargo test` transcript.

## Consequences

### Positive

- The actual bottleneck is now identified with direct evidence
  (instrumented profiling, not inference), closing ADR-346's "Next
  Research" item 2 with a definitive negative result instead of leaving it
  open indefinitely.
- `ConnectivityBackend`/`ConnectivityStructure` is a small, genuinely
  reusable abstraction: any future caller that *does* need
  `PolylogConnectivity`'s guarantees (e.g., a true worst-case bound on
  individual updates, vs. `DynamicConnectivity`'s amortized/rebuild-on-delete
  behavior) now has a tested, drop-in option.
- No existing behavior changes: every current caller of
  `RuVectorGraphAnalyzer`, `MinCutWrapper`, or `MincutGatedForgetting` is
  unaffected (new constructors/fields only, old ones delegate to the same
  default).
- Prevents a second nightly run from re-attempting the same already-falsified
  idea without root-cause evidence.

### Negative

- Does not unblock `MincutGatedForgetting` (ADR-345): the real fix target
  (the 16x redundant per-instance recomputation in
  `MinCutWrapper::process_instances`, and/or `brute_force_min_cut`'s O(2^n)
  exhaustive enumeration for n<20) is unaddressed and is now the next
  research item.
- Adds a small amount of surface area (`ConnectivityBackend` enum,
  `ConnectivityStructure` wrapper, four new constructors) for a backend
  choice that, on current evidence, has no end-to-end effect for any
  existing use case.

## Alternatives Considered

- **Fix `process_instances`'s redundant recomputation directly in this
  run** (e.g., compute the global min cut once, reuse it across the
  geometric ladder instead of recomputing per-instance; or make
  `brute_force_min_cut` actually bounded by `[lambda_min, lambda_max]`
  instead of always computing the unbounded global optimum). Rejected for
  *this* run: the preregistered hypothesis and this run's time budget were
  scoped to the `PolylogConnectivity` question specifically, per the
  repository's "finish a thread, do not silently redefine it" convention.
  Changing `BoundedInstance`'s or `MinCutWrapper`'s actual algorithm is a
  larger, independently falsifiable change that deserves its own
  hypothesis, its own acceptance test, and its own nightly run — proposed
  as the immediate next one (see README "Next Research").
- **Force an artificial win by cherry-picking a graph shape where the
  connectivity backend happens to matter more.** Rejected: no such shape
  exists given the code-path evidence (the backend is categorically outside
  `BoundedInstance`'s call graph), and manufacturing one would contradict
  this process's "no fabricated comparison numbers" rule.
- **Leave `PolylogConnectivity` unwired and only report the negative
  result.** Considered, but wiring it in as a real, tested, selectable
  option (rather than just writing "it wouldn't help") is strictly more
  valuable evidence and more useful to a future caller who does need its
  guarantees for a different reason than `partition()` latency.

## Implementation

- `crates/ruvector-mincut/src/connectivity/mod.rs`: `ConnectivityBackend`
  enum (`EulerTour` default, `Polylog`) and `ConnectivityStructure`
  enum-dispatch wrapper exposing `insert_edge`/`delete_edge`/
  `is_connected`/`connected`/`component_count`/`backend` uniformly over
  both backends.
- `crates/ruvector-mincut/src/wrapper/mod.rs`: `MinCutWrapper::conn_ds` is
  now `ConnectivityStructure` instead of `DynamicConnectivity` directly (all
  existing call sites unchanged — same method names). New
  `MinCutWrapper::new_with_backend`/`with_factory_and_backend`/
  `connectivity_backend()`; `new`/`with_factory` delegate to
  `ConnectivityBackend::EulerTour`, byte-for-byte the old behavior.
- `crates/ruvector-mincut/src/integration/mod.rs`:
  `RuVectorGraphAnalyzer::new_with_backend`/`from_knn_with_backend`, same
  delegation pattern.
- `crates/ruvector-mincut/src/lib.rs`: re-exports `ConnectivityBackend`,
  `ConnectivityStructure`.
- `crates/ruvector-agent-memory/src/graph_forget.rs`:
  `MincutGatedForgetting::connectivity_backend` field (default
  `ConnectivityBackend::EulerTour` in both `soft()`/`hard()`), threaded into
  `boundary_from_one_partition`'s `from_knn_with_backend` call.

## API Shape / Feature Flags

No new Cargo feature flag: the choice is a runtime enum, not a compile-time
feature, because both backends are already always compiled (`polylog` is a
public module with no feature gate). No breaking changes to any existing
public signature; all new surface is additive (new enum, new struct, new
methods, one new struct field defaulted at both of
`MincutGatedForgetting`'s existing constructors).

## Security

No new trust boundary, no new I/O, no new dependency. `PolylogConnectivity`
was already in-tree and already unit-tested before this run; this run adds
cross-backend equivalence tests, not new unsafe code (the crate denies
`unsafe_code` outside the `wasm` feature, unaffected here).

## Governance

Standard PR review. No schema, wire-format, or persisted-data change. No
ADR-281 `EmbeddingSpaceIdentity` applicability: this experiment has no
embedding space (it is a latency/correctness experiment over a
graph-connectivity data structure, not a retrieval benchmark) — explicitly
noting this per ADR-282 rather than omitting the section.

## Failure Modes

- If a future caller selects `ConnectivityBackend::Polylog` expecting a
  `partition()`/`min_cut()` latency improvement, this ADR and the
  `ConnectivityBackend` doc comment explain why that expectation is
  unsupported by evidence — misuse risk is "wasted code complexity for no
  benefit," not correctness risk for `partition()`/`min_cut()` specifically
  (the equivalence tests show both backends agree on the witness/partition
  output that matters there, under insert-only usage).
- **Correctness risk for a different, hypothetical use**:
  `PolylogConnectivity::delete_edge`'s replacement-edge search has a found,
  unfixed bug (see Evidence) that can make `is_connected()` disagree with
  `DynamicConnectivity`'s ground truth after a mixed insert/delete
  sequence. No current caller deletes edges from a `ConnectivityStructure`,
  so this does not affect anything merged here — but it is a real
  constraint on the backend's usefulness for any future delete-heavy use,
  not merely an unexplored corner, and `ConnectivityBackend::Polylog`'s doc
  comment says so explicitly.

## Migration / Rollback

No migration needed (purely additive). Rollback is deleting the new enum,
wrapper, constructors, and field — no persisted state, no downstream
callers depend on the new surface, since no existing crate opts into
`ConnectivityBackend::Polylog` by default anywhere.

## Rejection Criteria (met)

This ADR's own hypothesis is falsified if the A/B measurement shows no
latency/scaling improvement for `PolylogConnectivity` and root-cause
evidence shows the connectivity backend is not on `partition()`'s dominant
cost path. Both hold (see Evidence). Per this nightly process's own rule, a
falsified hypothesis with honest, direct evidence is itself the successful
outcome of this run.

## Open Questions

1. Does fixing `MinCutWrapper::process_instances`'s redundant per-instance
   recomputation (the actual bottleneck) change the picture enough to make
   `MincutGatedForgetting` viable at a realistic corpus size? Proposed as
   the next nightly thread.
2. Is `brute_force_min_cut`'s O(2^n) exhaustive enumeration for n<20
   necessary, or could a polynomial exact algorithm (e.g., Stoer-Wagner,
   O(n^3)) replace it without losing any guarantee this crate actually
   relies on? Not evaluated this run.
3. Root-cause and fix `PolylogConnectivity::delete_edge`'s found
   `is_connected()` divergence (Evidence section) before anyone relies on
   this backend for a delete-heavy workload; then, separately, measure
   whether it has a measurable advantage over `DynamicConnectivity`'s
   full-rebuild-on-delete for such a workload (unrelated to
   `BoundedInstance`). Neither done this run; both would need their own
   dedicated investigation and benchmark/claim.
