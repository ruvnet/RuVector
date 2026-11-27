# ADR-361: Opt-In Deduplication of `MinCutWrapper`'s Cold-Start Instance Scan

## Status

Accepted (opt-in, default-off). Non-breaking additive API: a new public
enum (`ScanOptLevel`), a new builder method on `MinCutWrapper` and
`RuVectorGraphAnalyzer`, and a new public field on
`ruvector-agent-memory`'s `MincutGatedForgetting`, each defaulting to the
exact prior behavior. `MincutGatedForgetting` itself remains **not**
promoted; see ADR-345, whose `Rejected (for production use as designed)`
status this ADR's own evidence re-confirms rather than overturns.

## Context

`ruvector_mincut::MinCutWrapper::query()` cold-starts (no cached
`last_min_cut` hint) by scanning its 100 pre-computed `BoundedInstance`
range buckets from index 0 upward until one reports the true min-cut value
in range. ADR-345 (2026-09-05) measured this path costing ~77ms (n=50) up
to ~11.4s (n=400), with a 69s outlier on a 19-vertex graph, and partly
rejected `ruvector-agent-memory`'s experimental `MincutGatedForgetting`
compaction policy on that basis. ADR-346 (2026-09-11) fixed a separate
correctness bug in the same path (non-deterministic, sometimes-degenerate
partition results, root-caused to unstable `HashMap`/`HashSet` iteration
order) but explicitly deferred the latency question to a future run. An
untracked 2026-10-03 commit (`eef4d1b96`) fixed one latency root cause
(`DeterministicLocalKCut::deterministic_bfs` recomputing its cut boundary
from scratch at every BFS depth, instead of incrementally) but was never
benchmarked against ADR-345's own acceptance thresholds afterward.

This nightly run (2026-11-13) re-ran ADR-345/346's own benchmarks against
current code and found: determinism holds (ADR-346's fix stands); the
2026-10-03 fix gives a real but partial win (n=400 latency down ~2.6x);
and the n=19 brute-force outlier is **unchanged** (73.9s today), because
that fix never touched `BoundedInstance::brute_force_min_cut`. Reading
`MinCutWrapper::process_instances` identified a second, independent,
still-live source of redundant work: `compute_bounds(i) = (floor(1.2^i),
floor(1.2^(i+1)))` produces literal duplicate `(lambda_min, lambda_max)`
pairs for several small `i` (e.g. `i=0,1,2` all map to `(1,1)`), and the
cold-start scan re-instantiates, re-populates (including a full
`graph.edges()` collect-and-sort per instance), and re-queries each one —
despite ADR-346's own determinism fix guaranteeing an identical-range query
against identical graph state returns an identical answer every time.

## Hypothesis

```text
Given RuVectorGraphAnalyzer::partition() / MinCutWrapper::query() called
cold-start on (a) ADR-345's own ring k-NN scaling-probe topology at
n in {19, 50, 100, 200, 400} and (b) ADR-345's own exact 84-memory
MincutGatedForgetting acceptance corpus,

when the cold-start scan skips re-instantiating any never-yet-queried
BoundedInstance whose (lambda_min, lambda_max) range duplicates the
immediately-preceding (already-AboveRange) instance's range, and hoists the
per-instance graph.edges() collect-and-sort to once per query() call,

then query()/partition() latency decreases measurably at every tested n,
and the returned cut value / boundary set is byte-identical to the
unmodified path's at every tested n and topology,

subject to: the pre-existing ruvector-mincut (522 tests) and
ruvector-agent-memory graph_forget (3 tests) suites remaining green, and
ADR-345's corpus and acceptance thresholds left completely unmodified.
```

Full methodology and raw benchmark output:
`docs/research/nightly/2026-11-13_bounded_instance_scan_dedup/README.md`.

## Decision

1. Add `ruvector_mincut::ScanOptLevel` (`Baseline` (default) /
   `DedupRanges` / `DedupRangesAndHoistEdges`) and
   `MinCutWrapper::with_scan_opt_level(self, level) -> Self`. Modify
   `MinCutWrapper::process_instances` to, when `scan_opt != Baseline`, skip
   instantiating an instance whose range exactly matches the
   most-recently-`AboveRange` instance scanned in the same call, provided
   that instance has never been separately instantiated (so instances
   touched by a prior incremental-update call are never skipped — only
   genuinely-fresh cold-start duplicates are). `DedupRangesAndHoistEdges`
   additionally collects `self.graph.edges()` once per `process_instances()`
   call instead of once per newly-instantiated instance.
2. Add `RuVectorGraphAnalyzer::with_scan_opt_level`, delegating to the
   wrapper.
3. Add `MincutGatedForgetting::scan_opt_level: ScanOptLevel` (default
   `Baseline`), threaded through to its one `RuVectorGraphAnalyzer::from_knn`
   call site, so the fix can be benchmarked against the exact ADR-345
   acceptance corpus without touching that corpus or its thresholds.
4. **Leave every default unchanged.** No existing caller's behavior
   changes unless it explicitly opts in.
5. **Do not reconsider `MincutGatedForgetting`'s promotion status.** The
   re-run ADR-345 acceptance benchmark, even with
   `DedupRangesAndHoistEdges` applied, still fails the compaction-slowdown
   gate by over an order of magnitude (1151–1292x vs. the required
   &le;100x). ADR-345's `Rejected` decision stands.

## Evidence

| Benchmark | Baseline | `DedupRangesAndHoistEdges` | Change |
|---|---|---|---|
| Ring k-NN `partition()`, n=19 | 74010.6ms | 57868.7ms | −22% |
| Ring k-NN `partition()`, n=400 | 4454.6ms | 3963.0ms | −11% |
| `BoundedInstance`s materialized (all tested n) | 16 | 13 | −3 (the duplicate-range instances) |
| ADR-345 corpus, Soft compaction slowdown | 1864.0x | 1291.8x | −31% |
| ADR-345 corpus, Hard compaction slowdown | 2127.9x | 1151.0x | −46% |
| Cut value / boundary set, all cases above | — | — | **identical** (asserted in-benchmark, never failed) |
| `ruvector-mincut` test suite | 522 passed | 522 passed | no regressions |
| `ruvector-agent-memory` `graph_forget` tests | 3 passed | 3 passed | no regressions |
| New `scan_opt_equivalence` test (4 topologies) | — | — | passes |

ADR-345's own acceptance gates, re-run unmodified against
`DedupRangesAndHoistEdges`:

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Bridge-survival gap (Soft/Hard) | &ge;15pp | +0.0pp / +0.0pp | FAIL (unchanged from ADR-345) |
| Recall@10 delta (Soft/Hard) | &le;2pp | 0.00pp / 0.00pp | PASS (unchanged) |
| Compaction slowdown (Soft/Hard) | &le;100x | 1291.8x / 1151.0x | FAIL (improved from 1864x/2128x, still fails) |
| Tamper detection | 20/20 | 20/20 | PASS (unchanged) |

Full raw output, the ring-graph sweep across all 5 sizes, and the
pre-this-run re-measurement of ADR-345/346's own probes are in the nightly
README (linked above).

## Consequences

- `ruvector-mincut` gains a measured, correctness-proven, opt-in latency
  improvement available to any caller of `MinCutWrapper` or
  `RuVectorGraphAnalyzer` willing to opt in; zero effect on anyone who does
  not.
- `ruvector-agent-memory`'s `MincutGatedForgetting` gains the same opt-in
  knob but keeps `Baseline` as its default — this ADR does not change its
  unpromoted status.
- The 2026-09-11 nightly's Next Research item 2 (which prompted this run)
  is closed: `PolylogConnectivity` was evaluated and found inapplicable
  (see "Alternatives Considered"); the actual remaining bottleneck was a
  different, newly-identified mechanism.
- A regression pass across `ruvector-mincut`'s other 18+ dependents before
  this becomes anyone's *default* remains open (ADR-346 flagged the same
  gap for its own fix and it is still not fully closed); two in-workspace
  dependents (`ruvector-graph-condense`, `ruvector-mincut-gated-transformer`)
  were smoke-checked clean this run via `cargo check`, not the full set.

## Alternatives Considered

- **`connectivity::polylog::PolylogConnectivity` as a `BoundedInstance`
  backend** (the literal suggestion this run was prompted by). Rejected:
  it is an incremental *connectivity* oracle (O(log n) "are u, v
  connected"), with no notion of cut value or boundary set — not a drop-in
  replacement for the min-cut search `BoundedInstance` performs. See the
  nightly README's "Rejected alternatives" for the fuller technical
  argument, including why its one plausible legitimate use (replacing the
  O(V+E) `is_connected()` BFS fallback) was not pursued — that BFS is not
  the measured bottleneck at any tested size.
- **Seeding cold-start `start_idx` above 0 via a cheap bound** (e.g.
  minimum vertex degree). Rejected: minimum degree is a valid upper bound
  on the global min cut, not a lower bound, so using it risks skipping past
  the instance range that actually contains the true value, and
  `process_instances`'s scan never searches backward — a silent correctness
  regression, not merely a performance one. No safe lower bound was
  identified this run.
- **Changing `compute_bounds`'s range math directly** to avoid producing
  duplicate ranges. Rejected as higher-risk than the loop-level skip
  implemented: `get_search_start`'s binary-search hint depends on a stable,
  index-addressable `lambda_max` sequence; changing the sequence's values
  needs a wider re-proof across every caller for a benefit the loop-level
  skip already captures without touching that sequence.

## Implementation Plan

Already implemented and merged in this change:
`crates/ruvector-mincut/src/wrapper/mod.rs` (`ScanOptLevel`,
`MinCutWrapper::with_scan_opt_level`, `process_instances` changes,
`scan_opt_equivalence` test), `crates/ruvector-mincut/src/lib.rs`
(re-export), `crates/ruvector-mincut/src/integration/mod.rs`
(`RuVectorGraphAnalyzer::with_scan_opt_level`),
`crates/ruvector-agent-memory/src/graph_forget.rs`
(`MincutGatedForgetting::scan_opt_level`), plus two new benchmark examples
(`crates/ruvector-mincut/examples/wrapper_scan_opt_bench.rs`,
`crates/ruvector-agent-memory/examples/mincut_gated_forgetting_bench_scanopt.rs`).

## API Shape

```rust
pub enum ScanOptLevel { Baseline, DedupRanges, DedupRangesAndHoistEdges } // Default = Baseline

impl MinCutWrapper {
    pub fn with_scan_opt_level(self, level: ScanOptLevel) -> Self;
}
impl RuVectorGraphAnalyzer {
    pub fn with_scan_opt_level(self, level: ScanOptLevel) -> Self;
}
pub struct MincutGatedForgetting {
    // ...existing fields...
    pub scan_opt_level: ScanOptLevel, // default: Baseline
}
```

## Feature Flags

None added. The change lives behind the existing `mincut-forget` feature
only insofar as `graph_forget.rs` already was; `ScanOptLevel` itself is
runtime-opt-in (a plain enum value), not compile-time feature-gated — the
default value makes that unnecessary.

## Benchmark Evidence

See "Evidence" above and the nightly README for full raw output, exact
commands, and the acceptance re-run.

## Security

No new attack surface: additive public API only, no new I/O,
serialization, or authentication surface. See the nightly README's
"Security / governance" section.

## Governance

No autonomous mutation, no MCP surface change, no change to any default
production code path. Promotion of this ADR's own change is to "opt-in
available," not "default-on."

## Failure Modes

See the nightly README's "Failure modes / limitations" section: the fix
does not address `search_for_cuts`'s internal seed×budget fan-out (the
dominant n&ge;20 cost) or `brute_force_min_cut`'s O(2^n) cost (the dominant
n<20 cost); it only removes *instance-level* redundancy across the outer
scan.

## Migration

None required. No default behavior changed.

## Rollback

Revert the four changed files; the new examples and `ScanOptLevel` are
purely additive, so rollback is a clean revert with no migration concerns
for any caller (no caller can depend on the new enum's existence changing
behavior, since none of the 18+ existing dependents reference it).

## Rejection Criteria

Would reject promoting `ScanOptLevel` even to opt-in status if:
`scan_opt_equivalence` (or the new benchmarks' in-line equivalence
assertions) ever failed; the existing 522+3 tests regressed; or a smoke
test of a dependent crate (WASM/Node bindings in particular) failed to
build. None of these occurred this run.

## Open Questions

Carried forward as "Next research" in the nightly README: WASM/Node
binding smoke tests; applying the same dedup principle inside a single
`search_for_cuts` call's own seed×budget loop; finding a provably-safe
lower bound for `start_idx`; and whether
`MincutGatedForgetting::soft`/`::hard`'s stale `mincut_trials: 3` default
should be lowered now that ADR-346 fixed the determinism issue it was
originally compensating for.
