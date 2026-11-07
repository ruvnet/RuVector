# ADR-360: Exact Stoer-Wagner Partition Backend for `RuVectorGraphAnalyzer`

## Status

Accepted (additive). New, opt-in methods on `RuVectorGraphAnalyzer`; no
existing public API signature changed, no existing default behavior changed.

## Context

ADR-345 (`docs/research/nightly/2026-09-05-mincut-gated-forgetting`) rejected
an agent-memory compaction policy (`MincutGatedForgetting`) built on
`RuVectorGraphAnalyzer::partition()` for two independent reasons:

1. **Latency.** `partition()` scaled worse-than-quadratically — ~77ms at 50
   vertices, ~11.4s at 400 — and had a pathological 69.3s outlier on a tiny
   (19-vertex) graph below its own small-graph cutoff. The 84-memory
   acceptance benchmark measured 1,800-2,700x slower than the scalar
   `CoherencePolicy` baseline it was compared against, against a 100x
   acceptance gate.
2. **No measured effectiveness.** At the corpus size performance forced
   (n=84), the structural signal made exactly 0.0 percentage points of
   difference to bridge-memory survival versus the baseline.

ADR-346 (`docs/research/nightly/2026-09-11-mincut-partition-determinism`)
separately root-caused and fixed a correctness bug (non-deterministic,
occasionally-empty partitions) in the same `partition()` call, explicitly
leaving the latency problem as untouched future work and naming
`ruvector_mincut::DynamicMinCut`/`ClusterHierarchy::boundary_size` (bypassing
`RuVectorGraphAnalyzer`) as the next thing to try.

Investigating that lead found the root architectural mismatch:
`RuVectorGraphAnalyzer::partition()`/`min_cut()` always route through
`MinCutWrapper`, which dispatches to `BoundedInstance` —
`crates/ruvector-mincut/src/instance/bounded.rs`, part of this crate's
from-scratch implementation of a December-2024 *subpolynomial* dynamic
min-cut paper (arXiv:2512.13105). `BoundedInstance` is built to **amortize**
maintaining a cut across many incremental graph updates on large graphs; for
graphs under 20 vertices it falls back to an `O(2^n)` brute force, and above
that to a multi-seed, multi-budget `LocalKCut`-oracle search
(`search_for_cuts`) that is a heuristic — it returns *a* cut within a
configured value range, not provably the global minimum.

Meanwhile, this same crate already has an exact, deterministic, polynomial
global minimum-cut solver: `crate::algorithm::exact::minimum_cut`, a sparse
Stoer-Wagner implementation (`O(n)`-memory maximum-adjacency heap per phase,
with a linear bridge/degree pre-check), used internally by
`DynamicMinCut::recompute_min_cut()` every time an edge update cannot cheaply
preserve the existing cut. `RuVectorGraphAnalyzer` never calls it. One-shot,
static-graph callers of `RuVectorGraphAnalyzer` — exactly
`MincutGatedForgetting`'s use case, and two of the crate's own three in-tree
consumers (`CommunityDetector`, `GraphPartitioner`) — pay the amortized
dynamic-maintenance algorithm's per-query overhead without ever getting its
intended benefit (maintaining a cut across *many* updates).

## Hypothesis

```text
Given RuVectorGraphAnalyzer wrapping a static graph, when queried through
new methods (min_cut_exact, partition_exact) that build a DynamicMinCut
from the same graph and reuse its existing exact::minimum_cut solver,
instead of min_cut()/partition()'s MinCutWrapper/BoundedInstance path,

then (a) on the exact ring k-NN topology and sizes (n=19,50,100,200,400)
from ADR-345's mincut_scaling_probe.rs, partition latency drops by at least
10x at every size, and the n=19 pathological 69s outlier disappears, (b)
on ADR-345's unmodified 84-memory mincut_gated_forgetting_bench corpus,
hypothesis, and acceptance thresholds, measured compaction slowdown versus
the scalar baseline drops enough to flip at least one of the two
previously-failing latency gates (Soft, Hard) to PASS, and (c) the exact
path finds the textbook-correct global minimum cut (verified against
brute-force enumeration on small synthetic graphs) and is deterministic
across repeated calls,

subject to: no public API signature changes to existing
RuVectorGraphAnalyzer methods, the full pre-existing ruvector-mincut and
ruvector-agent-memory test suites remaining green, and no change expected
to ADR-345's bridge-survival-effectiveness finding (that failure mode was
independently attributed to the global cut not coinciding with
human-labeled bridges on continuous noisy data, not to backend latency or
correctness — this run treats that as a prediction to confirm, not a
target to fix).
```

Full methodology and raw output:
`docs/research/nightly/2026-10-13-mincut-exact-partition-backend/README.md`.

## Decision

1. **New methods, additive only:**
   `RuVectorGraphAnalyzer::min_cut_exact(&self) -> f64` and
   `RuVectorGraphAnalyzer::partition_exact(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)>`
   (`crates/ruvector-mincut/src/integration/mod.rs`). Both clone the
   analyzer's current `DynamicGraph`, build a `DynamicMinCut` from it
   (`MinCutConfig::default()`), and delegate to its existing
   `min_cut_value()`/`partition()`. No change to `min_cut()`/`partition()`
   themselves, `CommunityDetector`, or `GraphPartitioner` — they keep using
   `MinCutWrapper`/`BoundedInstance` exactly as before.
2. **New regression tests**
   (`crates/ruvector-mincut/tests/exact_partition_tests.rs`): correctness
   against independent `O(2^n)` brute-force enumeration on five random
   weighted graphs (n=4..12), an unambiguous two-9-clique-plus-bridge
   construction, determinism across 20 repeated calls, and the `n<2` edge
   case.
3. **New comparison probe**
   (`crates/ruvector-agent-memory/examples/mincut_scaling_probe_exact.rs`):
   times `partition_exact` on the exact same ring-topology sizes as the
   ADR-345 probe it is a companion to, plus an honest extension (n up to
   6,400) to find where the exact path's own polynomial cost becomes
   expensive — it is not claimed to win at every scale.
4. **New `MincutGatedForgetting` backend switch**
   (`crates/ruvector-agent-memory/src/graph_forget.rs`): a `MincutBackend`
   enum (`BoundedInstance` default, `Exact`) and an `.exact()` builder
   method. `mincut_trials` is ignored (treated as 1) under `Exact`, since a
   deterministic global-optimum solver gains nothing from repeated calls.
5. **New companion acceptance benchmark**
   (`crates/ruvector-agent-memory/examples/mincut_gated_forgetting_bench_exact.rs`):
   a byte-for-byte copy of ADR-345's `mincut_gated_forgetting_bench.rs`
   except for the two `.exact()` calls, re-running the identical dataset,
   seed, hypothesis, and acceptance thresholds — per the "don't move the
   goalposts" rule, the original file is untouched and still reproduces its
   original (REJECT) result unmodified.

## Evidence

### Scaling probe (`mincut_scaling_probe.rs` vs. `_exact`, same ring k-NN graphs, k=8)

| n | `partition()` (ADR-345, BoundedInstance) | `partition_exact()` (this run) | Speedup |
|---:|---:|---:|---:|
| 19  | 69,269.9 ms (69.3s outlier) | 0.472 ms   | ~146,800x |
| 50  | 76.8 ms                     | 0.789 ms   | ~97x |
| 100 | 481.3 ms                    | 1.846 ms   | ~261x |
| 200 | 2,712.9 ms                  | 4.801 ms   | ~565x |
| 400 | 11,415.0 ms                 | 16.174 ms  | ~706x |

Extended range (`partition_exact` only — not measured for the old backend,
whose cost at these sizes is extrapolated to be far worse, not measured,
so no comparison is claimed here): n=800 → 49.5ms, n=1,600 → 216.5ms,
n=3,200 → 931.1ms, n=6,400 → 3,933.7ms. This is the honest limit of the new
backend: it is polynomial (empirically close to the `O(n^2 log n)` the
sparse Stoer-Wagner implementation's own doc comment predicts), not
subpolynomial, so it is expected to lose to a true incremental algorithm at
large enough `n` with many updates. It is not claimed to replace
`BoundedInstance` for that regime.

### Acceptance benchmark (ADR-345's unmodified 84-memory corpus, seed=341)

| Policy | Bridge Surv. | Recall@10 | Compaction (µs) | Slowdown vs. baseline |
|---|---:|---:|---:|---:|
| CoherenceWeighted (baseline) | 66.7% | 100.0% | 36-38 | 1x |
| MincutGatedForgetting-Soft (BoundedInstance, this session's re-run) | 66.7% | 100.0% | 116,147 | 3,226.3x |
| MincutGatedForgetting-Hard (BoundedInstance, this session's re-run) | 66.7% | 100.0% | 118,103 | 3,280.6x |
| MincutGatedForgetting-Soft-Exact (this run) | 66.7% | 100.0% | 1,993 | **52.4x** |
| MincutGatedForgetting-Hard-Exact (this run) | 66.7% | 100.0% | 5,038 | **132.6x** |

(The BoundedInstance row's slowdown is measured higher here than ADR-345's
originally reported 1,800-2,700x — same order of magnitude, attributed to
this session's container hardware, consistent with ADR-346's own note that
absolute timings vary by environment while the qualitative picture does
not.)

| Gate | Threshold | BoundedInstance | Exact (this run) |
|---|---|---|---|
| Soft bridge-survival gap | ≥15pp | +0.0pp — FAIL | +0.0pp — FAIL |
| Hard bridge-survival gap | ≥15pp | +0.0pp — FAIL | +0.0pp — FAIL |
| Soft \|recall delta\| | ≤2pp | 0.00pp — PASS | 0.00pp — PASS |
| Hard \|recall delta\| | ≤2pp | 0.00pp — PASS | 0.00pp — PASS |
| Soft compaction slowdown | ≤100x | 3,226.3x — FAIL | **52.4x — PASS** |
| Hard compaction slowdown | ≤100x | 3,280.6x — FAIL | 132.6x — FAIL (24.7x closer) |
| Tamper detection | 20/20 | 20/20 — PASS | 20/20 — PASS |

**Overall acceptance for both backends: REJECT** (bridge-survival gap is
mandatory and unmet by either). This is the hypothesis's prediction (c)
confirmed: the exact backend resolves the *latency* blocker (fully for
Soft, substantially for Hard) without moving the *effectiveness* number at
all, which is independent evidence that 0.0pp gap is a property of what a
global minimum cut selects on continuous Gaussian-cluster data, not an
artifact of the previous backend's speed or non-exactness.

### Correctness and determinism (`exact_partition_tests.rs`)

```text
running 4 tests
test partition_exact_returns_none_below_two_vertices ... ok
test partition_exact_is_stable_across_repeated_calls ... ok
test partition_exact_finds_textbook_minimum_cut_on_random_graphs ... ok
test partition_exact_isolates_the_unique_bridge ... ok

test result: ok. 4 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out
```

`partition_exact_finds_textbook_minimum_cut_on_random_graphs` checks the
induced cut weight against an independent `O(2^n)` brute-force enumeration
on 5 random weighted graphs (n=4,6,8,10,12); all matched to within `1e-9`.

### Test suites

`cargo test --release -p ruvector-mincut` (lib + all existing integration
test files, including `determinism_tests.rs` and the new
`exact_partition_tests.rs`) and
`cargo test --release -p ruvector-agent-memory --features mincut-forget`
(49 lib tests including the new `exact_backend_protects_the_bridge_in_a_single_trial`,
plus all existing integration test files) both pass with zero failures.
`cargo clippy --release -p ruvector-mincut -p ruvector-agent-memory --features mincut-forget --all-targets`
is clean on the touched files.

## Consequences

**Positive:**
- `MincutGatedForgetting` (ADR-345, still feature-gated and off by default)
  now has a backend option that clears its own latency acceptance gate for
  the `Soft` variant and comes much closer for `Hard`, isolating the
  remaining rejection reason to effectiveness, not performance.
- `RuVectorGraphAnalyzer::min_cut_exact`/`partition_exact` are generally
  useful to any caller that wants the provably-correct global cut on a
  static graph — not limited to agent-memory — including `CommunityDetector`
  and `GraphPartitioner`, though neither was switched to use them in this
  run (see [Open Questions](#open-questions)).
- Zero migration cost: purely additive, no existing caller's behavior
  changes.

**Negative / accepted tradeoffs:**
- `partition_exact`/`min_cut_exact` are polynomial, not subpolynomial: at
  large `n` with many graph updates, `BoundedInstance` (or `DynamicMinCut`
  used incrementally, not via these one-shot methods) is still the right
  choice. This ADR does not change that tradeoff, only exposes the
  already-existing exact solver through `RuVectorGraphAnalyzer` for callers
  who need a one-shot static query.
- Each call clones the analyzer's `DynamicGraph` (`O(V+E)`) to build a fresh
  `DynamicMinCut`; not reused/cached across repeated calls on an unchanged
  graph the way `min_cut()`/`partition()`'s `cached_min_cut`/`cached_partition`
  fields are. Not measured as material at the sizes this run tested (clone
  cost is dwarfed by the solve itself), but a hot-loop caller should be
  aware.
- `MincutGatedForgetting-Hard-Exact` still fails its latency gate (132.6x
  vs. 100x) — this ADR does not claim full resolution, only measured,
  partial, honestly-reported progress.

## Alternatives

- **Wire `connectivity::polylog::PolylogConnectivity` into `BoundedInstance`**
  (the literal suggestion left by ADR-345/346's "Next Research"). Inspected
  and rejected: `PolylogConnectivity` answers "are these two vertices
  connected?" in polylog time — it has no notion of cut *value* or
  *partition*, so it cannot replace `partition()`'s actual job. It remains
  unused by this crate's `BoundedInstance`/`MinCutWrapper` path, as it was
  before this run; a connectivity oracle could only ever help the
  `MinCutResult::Disconnected` fast-path check, not the expensive search.
- **Switch `RuVectorGraphAnalyzer::min_cut()`/`partition()`'s default
  implementation to the exact backend outright**, instead of adding new
  methods. Rejected: `MinCutWrapper`/`BoundedInstance` is the crate's
  flagship subpolynomial *incremental* algorithm, and `min_cut()`/
  `partition()` are also called by `add_edge`/`remove_edge`'s cache
  invalidation pattern, which is designed around amortized incremental
  maintenance. Replacing the default would regress exactly the workload
  (many updates on a large graph) this crate exists to handle well, to fix
  a problem that only exists for the opposite workload (one-shot static
  queries). Additive new methods let each caller pick the right tool.
- **Use `DynamicMinCut` directly in `ruvector-agent-memory`, bypassing
  `RuVectorGraphAnalyzer` entirely** (the literal ADR-345/346 "Next
  Research" wording). Rejected in favor of adding `partition_exact` to
  `RuVectorGraphAnalyzer` itself: it benefits all three in-tree consumers
  (`CommunityDetector`, `GraphPartitioner`, agent-memory) and any future
  one, not just this one call site, for the same implementation cost.

## Implementation Plan

Already implemented in this PR:

1. `crates/ruvector-mincut/src/integration/mod.rs` — `min_cut_exact`,
   `partition_exact`, `build_dynamic_min_cut` helper.
2. `crates/ruvector-mincut/tests/exact_partition_tests.rs` — new test file.
3. `crates/ruvector-agent-memory/src/graph_forget.rs` — `MincutBackend`
   enum, `.exact()` builder, backend dispatch in `boundary_indices`/
   `boundary_from_one_partition`, backend-qualified `name()`, new unit test.
4. `crates/ruvector-agent-memory/examples/mincut_scaling_probe_exact.rs`,
   `mincut_gated_forgetting_bench_exact.rs` — new companion probes/benchmark.
5. `crates/ruvector-agent-memory/Cargo.toml` — register the two new
   examples under the existing `mincut-forget` feature gate.

## API Shape

```rust
impl RuVectorGraphAnalyzer {
    pub fn min_cut_exact(&self) -> f64;
    pub fn partition_exact(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)>;
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default)]
pub enum MincutBackend {
    #[default]
    BoundedInstance,
    Exact,
}

impl MincutGatedForgetting {
    pub fn exact(self) -> Self; // builder: switches `backend` to `Exact`
}
```

No existing signature changed.

## Feature Flags

None new. The new agent-memory example binaries and the `graph_forget`
changes are both already behind the existing `mincut-forget` feature; the
new `ruvector-mincut` methods and test file are unconditional (the crate has
no relevant feature gate around `RuVectorGraphAnalyzer`).

## Benchmark Evidence

See [Evidence](#evidence) above; raw command transcripts are in the linked
nightly README.

## Security

No new security-relevant surface. `min_cut_exact`/`partition_exact` are
read-only queries over a graph the caller already owns (`&self`, not
`&mut self` — stricter than `min_cut()`/`partition()`, which take `&mut
self` for their cache). No new dependency, no new unsafe code, no change to
any witness/signing path.

## Governance

No schema, wire-format, or persisted-data changes. Standard PR review.

## Failure Modes

- At sizes well beyond this run's tested range (the extended probe suggests
  single-digit seconds by n≈6,400 on a ring k-NN graph), `partition_exact`
  becomes an impractical one-shot call; callers with graphs in that range
  should still use `min_cut()`/`partition()` or `DynamicMinCut` directly
  with incremental updates, not this ADR's new methods.
- `min_cut_exact`/`partition_exact` return `f64` (unquantized), while
  `min_cut()`/`partition()`'s underlying `MinCutResult::Value` reports a
  `u64` cut value. The two are not directly comparable number-for-number on
  a weighted graph; this was true before this ADR (the quantization already
  existed in the `BoundedInstance` path) and is now more visible because
  both are reachable from the same type. Not fixed here — a possible
  follow-up is documenting or reconciling the two value representations.

## Migration

None required. Existing callers of `RuVectorGraphAnalyzer` are unaffected;
adopting the new methods or `MincutGatedForgetting::exact()` is opt-in.

## Rollback

Revert the files listed under [Implementation Plan](#implementation-plan).
Nothing else depends on the new methods as of this PR (no in-tree caller
was switched to them by default).

## Rejection Criteria

This decision should be revisited if:

- A future, larger-scale benchmark shows `partition_exact`'s `O(n^2 log n)`-class
  cost becomes a practical bottleneck below the sizes this run assumed were
  safe (hundreds to low thousands of vertices) — in which case the
  recommended fix is caching/reuse at the call site (see
  [Consequences](#consequences)), not reverting this ADR.
- `CommunityDetector`/`GraphPartitioner` are switched to the exact backend
  in a future run and found to regress their own latency or correctness
  tests at production graph sizes.

## Open Questions

1. Should `CommunityDetector::recursive_partition` and
   `GraphPartitioner::recursive_bisect` (both currently call
   `RuVectorGraphAnalyzer::min_cut()`/`partition()`) be switched to the
   exact backend? They operate on recursively-shrinking subgraphs, so the
   same latency argument likely applies, but this run did not benchmark
   either path — flagged as next research, not assumed.
2. Can `MincutGatedForgetting-Hard-Exact`'s remaining 132.6x slowdown
   (vs. the 100x gate) be closed by caching the `DynamicMinCut` across the
   policy's `mincut_trials`-loop construction (currently rebuilt from
   scratch inside `boundary_from_one_partition` every call, even though
   `Exact` only ever makes one call) or by reusing one `DynamicGraph` clone
   across `Soft`/`Hard` runs in the benchmark harness itself? Not
   attempted — the loop is already down to one iteration under `Exact`, so
   the remaining cost is the `O(V+E)` clone plus one solve; not profiled
   further this run.
3. Does the 0.0pp bridge-survival gap genuinely confirm "global min cut
   doesn't track human-labeled bridges on noisy data" (ADR-345's
   hypothesis), or could a different similarity-graph construction
   (different `k`, a learned metric, a sparsifier) produce a cut that does?
   Unresolved; this run deliberately did not touch the graph-construction
   side of `MincutGatedForgetting` to keep the comparison to ADR-345
   apples-to-apples.
4. Is there value in reconciling `min_cut_exact`'s `f64` return with
   `min_cut()`'s `u64`, e.g. a shared `MinCutValue` type? Unresolved,
   flagged as a documentation/follow-up item, not attempted here.
