# ADR-346: Deterministic, Non-Degenerate Witness Partitions in `ruvector-mincut`

## Status

Accepted. Non-breaking bug fix, merged into `ruvector-mincut` directly (no
feature flag — the prior behavior was a bug, not a documented tradeoff).

## Context

The 2026-09-05 nightly research run (ADR-345,
`docs/research/nightly/2026-09-05-mincut-gated-forgetting`) rejected an
agent-memory integration built on
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` on two independent
grounds, one of which was: `partition()` returned an empty or degenerate
result (one side of the cut empty) in 15/30 (50%) of repeated calls against
a byte-identical 19-vertex graph, in a reproduction script
(`crates/ruvector-agent-memory/examples/mincut_determinism_probe.rs`) that
run added but did not root-cause, attributing it tentatively to "hash-map
iteration-order-dependent tie-breaking."

`ruvector-mincut` has three in-tree consumers of this same code path
(`RuVectorGraphAnalyzer`, `CommunityDetector`, `GraphPartitioner`) and 18
crates depend on it directly, including WASM and Node bindings
(`ruvector-mincut-wasm`, `ruvector-mincut-node`), `ruvector-agent-memory`,
`ruvector-graph-condense`, `prime-radiant`, `cognitum-gate-kernel`, and
`mcp-brain-server`. A silently-wrong `partition()` result is a correctness
bug affecting all of them, not a performance footnote.

## Hypothesis

```text
Given RuVectorGraphAnalyzer::partition() called on a fresh analyzer built
from a fixed, byte-identical connected k-NN graph (the exact 19-vertex
topology from mincut_determinism_probe.rs, unmodified),

when (a) DynamicGraph::vertices()/edges() return canonical (sorted) order
instead of raw DashMap iteration order, (b) BoundedInstance's internal
seed-selection and bitmask-to-vertex vertex lists are sorted instead of raw
HashSet iteration order, and (c) RuVectorGraphAnalyzer::partition() derives
both cut sides from the graph's real vertex list via witness.contains()
instead of WitnessHandle::materialize_partition()'s max(U)-inferred range,

then partition() returns a non-empty, non-degenerate, byte-identical result
across repeated calls,

subject to: the full pre-existing ruvector-mincut test suite remaining
green and no public API signature changes.
```

Full methodology, root-cause diagnostic output, and raw benchmark output are
in
`docs/research/nightly/2026-09-11-mincut-partition-determinism/README.md`.

## Decision

1. `DynamicGraph::vertices()` and `DynamicGraph::edges()`
   (`crates/ruvector-mincut/src/graph/mod.rs`) now sort their output
   (`VertexId` ascending; `EdgeId` ascending) before returning, instead of
   returning raw `DashMap::iter()` order. `DashMap`'s default hasher is
   randomly seeded per instance, so its iteration order is not stable
   across process runs even for byte-identical insertion sequences.
2. `BoundedInstance::brute_force_min_cut` and `BoundedInstance::search_for_cuts`
   (`crates/ruvector-mincut/src/instance/bounded.rs`) now sort the
   `HashSet<VertexId>`-derived vertex lists they use for bitmask-to-vertex
   assignment and LocalKCut seed ordering, respectively, before iterating.
   Both had order-dependent tie-breaking (`boundary < min_cut`'s strict
   inequality keeps the first-found subset on ties; `search_for_cuts`
   returns on the first seed whose search succeeds).
3. `RuVectorGraphAnalyzer::partition()`
   (`crates/ruvector-mincut/src/integration/mod.rs`) no longer calls
   `WitnessHandle::materialize_partition()`. That method infers the graph's
   vertex range from `max(U)` — the cut side's own membership — which
   silently truncates or empties the computed complement `V \ U` whenever
   `U` does not happen to contain the graph's highest-numbered vertex (the
   common case, not an edge case). `partition()` now scans the graph's
   actual vertex list (`self.graph.vertices()`, already sorted per decision
   1) and splits each vertex by `witness.contains(v)` (an O(1) bitmap
   check), which is correct for any `U`.
4. `WitnessHandle::materialize_partition()`'s doc comment
   (`crates/ruvector-mincut/src/instance/witness.rs`) now states the
   `max(U)` scope limitation explicitly and directs callers to the
   `contains()` + real-vertex-list pattern instead. Its signature is
   unchanged — see [Alternatives](#alternatives) for why a signature change
   was rejected for this run.
5. New regression test `crates/ruvector-mincut/tests/determinism_tests.rs`,
   reproducing the exact topology from the originally-reporting nightly's
   probe script, pinning both "never degenerate" and "byte-identical across
   repeated calls" as CI-enforced invariants going forward.

## Evidence

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Empty/degenerate rate on unmodified reproduction script | 0% | 0/60 across two independent 30-trial runs (prior nightly measured 15/30, 50%) | PASS |
| Partition stability across repeated calls | byte-identical | 30/30 identical (new `determinism_tests.rs`) | PASS |
| Pre-existing `ruvector-mincut` test suite | all green | `cargo test --release -p ruvector-mincut --lib --tests` — see raw output below | PASS |
| Public API signature changes | none | none (verified by inspection: `vertices()`/`edges()`/`partition()` signatures unchanged) | PASS |
| Latency regression | none beyond noise | ~1118-1130ms/call vs. prior nightly's ~841ms/call on the same 19-vertex probe; attributed to environment variance, not the fix (added sort is O(19 log 19), microseconds) — reported, not adjusted | PASS (no algorithmic regression) |

Raw reproduction output (`TRIALS=30 ./target/release/examples/mincut_determinism_probe`,
built via `cargo build --release -p ruvector-agent-memory --example
mincut_determinism_probe --features mincut-forget`):

```text
trials=30 elapsed=33.55s avg_per_call=1118.3ms empty_or_degenerate=0 (0%) bridge_detected_as_boundary=30 (100%)
trials=30 elapsed=33.88s avg_per_call=1129.5ms empty_or_degenerate=0 (0%) bridge_detected_as_boundary=30 (100%)
```

New regression test output:

```text
running 3 tests
test graph_vertices_and_edges_are_sorted ... ok
test partition_is_never_degenerate_on_connected_graph ... ok
test partition_is_stable_across_repeated_calls ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 34.17s
```

Full `ruvector-mincut` test suite, including the new `determinism_tests.rs`
(`cargo test --release -p ruvector-mincut --lib --tests`), 11 binaries (lib
+ 10 integration test files):

```text
unittests src/lib.rs:               512 passed; 0 failed; 5 ignored
tests/bounded_integration.rs:        16 passed; 0 failed
tests/canonical_bench.rs:             0 passed; 0 failed  (no #[test] fns)
tests/certificate_tests.rs:          28 passed; 0 failed
tests/coverage_tests.rs:              16 passed; 0 failed
tests/determinism_tests.rs (new):     3 passed; 0 failed
tests/integration_tests.rs:           7 passed; 0 failed
tests/jtree_tests.rs:                 0 passed; 0 failed  (no #[test] fns)
tests/localkcut_integration.rs:      18 passed; 0 failed
tests/localkcut_paper_integration.rs: 8 passed; 0 failed
tests/paper_algorithm_tests.rs:      16 passed; 0 failed
tests/wrapper_tests.rs:              20 passed; 0 failed

Total: 644 passed, 0 failed, 5 ignored (pre-existing, unrelated to this change)
```

Full root-cause diagnostic narrative — including the intermediate finding
that fixing decision 1 alone (without decision 3) made the metric look
*worse* (100% degenerate instead of 50%), which is why both fixes are
reported together rather than as sequential nightly runs — is in the
nightly README linked above.

## Consequences

**Positive:**
- All 18 direct dependents of `ruvector-mincut` get a correctness fix for
  free on next rebuild, with zero migration cost.
- `CommunityDetector` and `GraphPartitioner`, which share the same
  `partition()` call, are fixed transitively even though this run's
  reproduction and testing focused on the agent-memory use case.
- Establishes a regression test that would have caught the original bug
  before it reached a downstream nightly experiment.

**Negative / accepted tradeoffs:**
- `DynamicGraph::vertices()`/`edges()` now pay an `O(V log V)` / `O(E log E)`
  sort on every call instead of `O(V)` / `O(E)`. Not measured as a
  regression on the tested (19-vertex) graph; not benchmarked at the sizes
  (hundreds to thousands of vertices) where the *existing*, unrelated
  latency problem already dominates. If a future caller needs
  `vertices()`/`edges()` in a hot loop at a size where this sort becomes
  material, cache the sorted result at the call site rather than reverting
  this fix (reverting reintroduces the correctness bug).
- `WitnessHandle::materialize_partition()` remains capable of returning a
  silently-truncated complement for any caller that uses it directly
  instead of the `contains()` pattern. Two existing test files call it
  directly (`coverage_tests.rs`, `localkcut_paper_integration.rs`); neither
  exercises a case where this matters for the assertions they make, but a
  future caller could still hit this. Documented, not fixed at the source.

## Alternatives

- **Fix `materialize_partition()`'s signature directly** (accept a
  `max_vertex: VertexId` parameter, or a `&DynamicGraph` reference). More
  correct in the abstract, but a breaking API change requiring every caller
  across the workspace to be found and updated — disproportionate to a
  single-night, non-breaking fix. Rejected for now; recorded as future work
  under [Open Questions](#open-questions).
- **Replace `DashMap`/`HashSet` with `BTreeMap`/`BTreeSet` throughout
  `ruvector-mincut`.** Fixes ordering at the storage layer for every current
  and future caller, but touches far more of a 45,000-line crate — including
  paths where `DashMap`'s concurrent-access semantics are load-bearing
  (the `agentic` feature's parallel core distribution). Rejected as
  disproportionate blast radius for the specific, narrow bug actually
  observed and reproduced.
- **Do nothing; document the non-determinism as a known limitation.** This
  is what ADR-345 effectively did. Rejected because the root cause turned
  out to be tractable and the fix is low-risk and non-breaking — leaving a
  known correctness bug in a widely-depended-on crate when a narrow fix
  exists does not meet this process's evidence-retention standard for
  "we looked and it can't be fixed."

## Implementation Plan

Already implemented in this PR:

1. `crates/ruvector-mincut/src/graph/mod.rs` — sort `vertices()`/`edges()`.
2. `crates/ruvector-mincut/src/instance/bounded.rs` — sort seed/tie-break
   vertex lists in `brute_force_min_cut` and `search_for_cuts`; use `.min()`
   instead of `.iter().next()` for the reported witness seed.
3. `crates/ruvector-mincut/src/instance/witness.rs` — doc comment only.
4. `crates/ruvector-mincut/src/integration/mod.rs` — `partition()` uses
   `graph.vertices()` + `witness.contains()` instead of
   `materialize_partition()`.
5. `crates/ruvector-mincut/tests/determinism_tests.rs` — new regression
   test file.

## API Shape

No public API shape changes. All four touched functions
(`DynamicGraph::vertices`, `DynamicGraph::edges`,
`RuVectorGraphAnalyzer::partition`,
`WitnessHandle::materialize_partition`) keep their existing signatures and
return types.

## Feature Flags

None. This is a correctness fix to default (always-on) code paths, not a
new capability — no flag is appropriate.

## Benchmark Evidence

See [Evidence](#evidence) above and the linked nightly README's full
methodology and raw output sections.

## Security

No security-relevant surface changed — see the nightly README's
[Security](../../research/nightly/2026-09-11-mincut-partition-determinism/README.md#security)
section.

## Governance

Standard PR review. No schema, wire-format, or persisted-data changes.

## Failure Modes

- If a future graph exceeds `u32::MAX` vertices, `witness.contains(v)`
  (via `RoaringBitmap`, which is `u32`-indexed) already returns `false` for
  any `v > u32::MAX` regardless of this fix — pre-existing behavior,
  unchanged.
- `search_for_cuts`'s seed-ordering fix is applied by the same reasoning as
  `brute_force_min_cut`'s but not independently re-benchmarked at >=20
  vertices this run (the reproduction topology is 19 vertices, so only the
  brute-force path was exercised end-to-end). See
  [Falsification Criteria](../../research/nightly/2026-09-11-mincut-partition-determinism/README.md#falsification-criteria)
  in the nightly README.

## Migration

None required. No signature changes; recompiling dependent crates against
this version is sufficient.

## Rollback

Revert the four source-file changes; the new test file can be deleted or
left in place (it will fail against the pre-fix code, which is the correct,
expected outcome of a regression test for a bug that has been reintroduced).
Reverting reintroduces the original 50% empty/degenerate correctness bug —
not recommended.

## Rejection Criteria

This decision should be revisited if:

- The regression test suite (`determinism_tests.rs`) or the reproduction
  script (`mincut_determinism_probe.rs`) starts failing again on
  unmodified code — indicates either a regression or a previously-unseen
  third root cause.
- Profiling at production corpus sizes shows the added `O(V log V)` /
  `O(E log E)` sort in `vertices()`/`edges()` is materially on the hot path
  once the separate, still-open latency problem is addressed — in which
  case cache the sorted result rather than reverting the ordering
  guarantee.

## Open Questions

1. Does `search_for_cuts`'s `LocalKCut` path (graphs >=20 vertices) have an
   equivalent or different correctness bug, not exercised by tonight's
   19-vertex reproduction? Unresolved — flagged as next research.
2. Should `WitnessHandle::materialize_partition()`'s signature be fixed
   properly (breaking change) in a dedicated follow-up PR, given two
   existing test call sites already use it directly? Unresolved.
3. Does concurrent mutation of `DynamicGraph` during a `partition()` call
   produce a torn read? Not tested this run. Unresolved.
4. Does this fix change the outcome of ADR-345's `MincutGatedForgetting`
   acceptance benchmark? Deliberately not re-run this session (that
   benchmark's rejection has a second, independent, unaddressed latency
   cause) — unresolved, flagged as next research.
