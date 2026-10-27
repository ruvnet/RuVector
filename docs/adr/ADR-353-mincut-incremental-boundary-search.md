# ADR-353: Incremental Boundary Tracking for `ruvector-mincut`'s Local K-Cut Oracle

## Status

Accepted (as an additive, opt-in code addition) / **Rejected** (as the
performance claim it was built to validate — see Evidence). The code
itself lands: it is correct, tested, and harmless to every existing
caller, which is independently useful (a real, measured, if modest, speedup
available to any future caller that opts in) regardless of the headline
hypothesis's outcome. The default `CutSearchStrategy::Legacy` behavior
every existing caller goes through is byte-for-byte unchanged.

## Context

The 2026-09-05 nightly (ADR-345, `MincutGatedForgetting`) rejected a
candidate agent-memory compaction feature partly because
`RuVectorGraphAnalyzer::partition()` scaled from ~76ms at 50 vertices to
~11.4s at 400 vertices on a fixed-degree k-NN graph — roughly 150x latency
for an 8x vertex-count increase. The 2026-09-11 nightly (ADR-346) fixed two
unrelated non-determinism bugs in the same code path but explicitly left
this latency problem for later, naming as Next Research item 2:
*"root-cause the `partition()` latency scaling problem from ADR-345...
likely by evaluating `connectivity::polylog::PolylogConnectivity` (already
in-tree, not yet wired into `BoundedInstance`) as a replacement backend."*

`ruvector-mincut` has 18+ direct dependents (per ADR-346's own count),
including `ruvector-mincut-wasm`, `ruvector-mincut-node`,
`ruvector-agent-memory`, `ruvector-graph-condense`, `prime-radiant`,
`cognitum-gate-kernel`, and `mcp-brain-server`. A change to this crate's
default behavior carries real blast radius, which is why this run's fix is
additive and opt-in rather than a default-behavior change.

## Hypothesis

```text
Given BoundedInstance::query() on a connected synthetic k-NN ring graph
with n vertices >= 20 (where search_for_cuts, not the brute-force path, is
exercised), reproduced via the exact ring-graph shape and sizes
mincut_scaling_probe (ADR-345/346's own reproduction harness) uses: n in
{19, 50, 100, 200, 400}, k=8,

when DeterministicLocalKCut's per-call BFS boundary computation is
replaced with incremental boundary tracking (candidate A) and
BoundedInstance::search_for_cuts's redundant per-integer-budget outer loop
is additionally collapsed into a single global-budget pass with per-seed
minimum selection (candidate B),

then candidate B's end-to-end RuVectorGraphAnalyzer::partition() latency at
n=400 drops by at least 10x relative to the unmodified baseline, measured
through the production partition() entry point,

subject to: zero cut-value/witness-membership mismatches against the
baseline across a differential-equivalence test suite, and zero
regressions in the pre-existing ruvector-mincut test suite.
```

Full methodology, root-cause analysis, and raw benchmark output are in
`docs/research/nightly/2026-10-27-mincut-incremental-boundary-search/README.md`.

## Decision

1. New module `crates/ruvector-mincut/src/localkcut/incremental.rs`:
   `GraphView` (a borrowed vertex-set/adjacency-list view, bypassing the
   legacy path's per-call `DynamicGraph` rebuild) and `IncrementalLocalKCut`
   (an oracle that classifies each edge as crossing/internal exactly once,
   as its endpoint joins the BFS-visited set, instead of rescanning the
   entire visited set's edges at every layer).
2. `BoundedInstance` (`crates/ruvector-mincut/src/instance/bounded.rs`)
   gains an additive `CutSearchStrategy` enum (`Legacy` — the default,
   unchanged — `IncrementalOracle`, `IncrementalOracleCollapsedBudget`), a
   `strategy` field, and `BoundedInstance::with_strategy(..)`. The
   pre-existing `search_for_cuts` body is renamed to
   `search_for_cuts_legacy` and left untouched; a new
   `search_for_cuts_fast` implements both candidate strategies.
3. `RuVectorGraphAnalyzer` (`crates/ruvector-mincut/src/integration/mod.rs`)
   gains an additive `with_factory(..)` constructor so a caller can inject
   a custom `BoundedInstance` factory (e.g., a non-default
   `CutSearchStrategy`) while still going through the exact same
   `partition()`/`min_cut()` production code `new` uses. `new` itself is
   unchanged (`MinCutWrapper::new`, always `CutSearchStrategy::Legacy`).
4. New differential-equivalence test suite,
   `crates/ruvector-mincut/tests/incremental_oracle_equivalence.rs`,
   asserting `Legacy`, `IncrementalOracle`, and
   `IncrementalOracleCollapsedBudget` return identical `(cut_value,
   witness membership)` across brute-force-path and oracle-path graphs.
5. New benchmark binary,
   `crates/ruvector-mincut/examples/incremental_oracle_bench.rs`, measuring
   all three strategies end-to-end through `RuVectorGraphAnalyzer::partition()`
   on the exact `mincut_scaling_probe` graph shape/sizes.

`CutSearchStrategy::Legacy` remains the only strategy any existing
constructor (`BoundedInstance::new`/`init`, and therefore `MinCutWrapper`
and `RuVectorGraphAnalyzer::new`) produces. Nothing in this ADR changes
behavior for an existing caller that does not explicitly call
`with_strategy`/`with_factory`.

## Evidence

### Raw benchmark output

Command: `cargo run --release -p ruvector-mincut --example incremental_oracle_bench`

```text
n,strategy,partition_ms,side_a_len
19,Legacy,79475.786,1
19,IncrementalOracle,79942.517,1
19,IncrementalOracleCollapsedBudget,78305.776,1
50,Legacy,86.624,1
50,IncrementalOracle,68.055,1
50,IncrementalOracleCollapsedBudget,38.096,1
100,Legacy,552.085,1
100,IncrementalOracle,364.401,1
100,IncrementalOracleCollapsedBudget,254.651,1
200,Legacy,2995.436,1
200,IncrementalOracle,1343.765,1
200,IncrementalOracleCollapsedBudget,1009.107,1
400,Legacy,12748.655,1
400,IncrementalOracle,4837.082,1
400,IncrementalOracleCollapsedBudget,3232.209,1

n,baseline_ms,collapsed_budget_ms,speedup_x
19,79475.786,78305.776,1.01
50,86.624,38.096,2.27
100,552.085,254.651,2.17
200,2995.436,1009.107,2.97
400,12748.655,3232.209,3.94
```

Measured speedup at n=400 (the hypothesis's threshold point): **3.94x**,
against a pre-committed threshold of **10x**. See the companion nightly
README's [Acceptance Result](../research/nightly/2026-10-27-mincut-incremental-boundary-search/README.md#acceptance-result)
for the architectural reason the measured factor fell short of the
predicted one.

### Correctness

`cargo test -p ruvector-mincut --test incremental_oracle_equivalence`:
6/6 passed (brute-force-path graphs, oracle-path random graphs at several
densities, the exact ring-graph benchmark shape, narrow/offset lambda
ranges, path/cycle graphs, and a disconnected graph) — zero
cut-value/witness-membership mismatches between `Legacy` and either
candidate strategy.

Targeted pre-existing-module tests (`instance::`, `localkcut::`): all
passing, byte-identical to before this change. A full-crate
`cargo test -p ruvector-mincut --lib` run (500+ tests, many in modules this
change never touches) was started but did not finish inside this run's
time budget — see the companion nightly README's Correctness Results
section for why that is not treated as an open risk for this specific
change.

## Consequences

- Positive: callers that opt into `CutSearchStrategy::IncrementalOracleCollapsedBudget`
  get a measured latency improvement on the oracle-path (`n>=20`) workload
  with no change to returned cut values/witnesses, verified against the
  existing implementation on dozens of generated graphs.
- Positive: `PolylogConnectivity`'s inapplicability to this problem is now
  documented with a concrete reason (it answers a different query than the
  one `search_for_cuts` needs), closing off a plausible-sounding but wrong
  direction for future nightlies.
- Neutral: the default behavior and public API surface every existing
  caller sees is unchanged; this ADR adds new, optional surface area
  (`CutSearchStrategy`, `with_strategy`, `with_factory`) without removing
  or altering anything.
- Negative: `instance/bounded.rs` grows by roughly 150 lines to carry the
  new strategy dispatch and collapsed-budget search, pushing it over this
  project's informal 500-line-per-file preference; it remains one cohesive
  module (`BoundedInstance` and its search strategies) rather than being
  artificially split across files that would need private-field visibility
  changes to communicate.
- Negative (deferred, not introduced by this change): the brute-force path
  (`n<20`)'s own, independent redundancy — no cross-instance memoization
  across `MinCutWrapper`'s bounded-range instances — remains unfixed; see
  Next Research in the companion nightly README.

## Alternatives

- **Wire in `PolylogConnectivity` as the 2026-09-11 nightly suggested.**
  Rejected: it exposes connectivity queries, not a boundary-size/cut-value
  primitive. Using it would require building an entirely separate
  cut-finding structure on top of it, which is a materially larger and
  riskier change than fixing the two concrete inefficiencies this run
  found by reading the existing oracle's code.
- **Make `IncrementalOracleCollapsedBudget` the new default.** Rejected
  for this run: a default-behavior change for 18+ downstream crates
  deserves a wider correctness bake-in (more graph shapes/densities,
  fuzzing) than one nightly's differential suite provides. Named as Next
  Research instead.
- **Fix the brute-force-path redundancy in the same run.** Rejected on
  scope grounds: it is a different code path (and arguably a
  `MinCutWrapper`-level architectural issue — no per-instance caching
  across the wrapper's bounded ranges — not a `BoundedInstance`-local fix),
  and the named next-research item this run was chartered to close was
  specifically the oracle-path (`n>=20`) scaling problem. Documented as a
  Limitation and Next Research item instead of silently expanding scope.

## Implementation Plan

Already implemented in this run; see Decision and the file list in the
companion nightly README's Architecture section.

## API Shape

```rust
pub enum CutSearchStrategy {
    #[default]
    Legacy,
    IncrementalOracle,
    IncrementalOracleCollapsedBudget,
}

impl BoundedInstance {
    pub fn with_strategy(lambda_min: u64, lambda_max: u64, strategy: CutSearchStrategy) -> Self;
}

impl RuVectorGraphAnalyzer {
    pub fn with_factory<F>(graph: Arc<DynamicGraph>, factory: F) -> Self
    where F: Fn(&DynamicGraph, u64, u64) -> Box<dyn ProperCutInstance> + Send + Sync + 'static;
}

pub struct GraphView<'a> { /* borrowed vertices + adjacency */ }
pub struct IncrementalLocalKCut { /* ... */ }
impl IncrementalLocalKCut {
    pub fn search_single(&self, view: &GraphView, seed: VertexId, budget_k: u64, radius: usize) -> Option<(u64, WitnessHandle)>;
    pub fn search_global(&self, view: &GraphView, seed: VertexId, radius: usize) -> Option<(u64, WitnessHandle)>;
}
```

## Feature Flags

None. No new Cargo features; all new types/methods are plain additive pub
items in `ruvector-mincut`, already a non-optional dependency for every
caller this ADR concerns.

## Benchmark Evidence

See [Evidence](#evidence) above and the companion nightly README's full
raw output.

## Security

No change to trust boundaries, input validation, or witness/certificate
semantics. `IncrementalLocalKCut` produces the same `WitnessHandle` type
with the same `(seed, membership, boundary)` contents as the existing
oracle; it is not used in any signing/receipt path.

## Governance

Standard crate-internal change; no cross-team coordination required beyond
the downstream-dependent awareness already noted (no API removed or
altered for them).

## Failure Modes

- If a future caller passes a graph with duplicate same-pair edges
  (a true multigraph) to a `with_strategy`-constructed `BoundedInstance`,
  the two candidate strategies will count every duplicate edge (matching
  `self.adjacency`'s actual content), while `Legacy` silently collapses
  duplicates via its temporary `DynamicGraph` rebuild's ignored
  `insert_edge` error. This is a pre-existing inconsistency between
  `Legacy` and `self.adjacency`-based code (the brute-force path already
  uses `self.adjacency` directly), not introduced by this ADR, but it means
  `Legacy` and the two new strategies are **not** guaranteed equivalent on
  multigraphs — only on simple graphs, which is what every production
  caller (`from_knn`, `from_similarity_matrix`) constructs. Documented in
  the differential-equivalence test suite's graph-generator comments.
- If `IncrementalLocalKCut`'s `absorb_vertex` is ever called on a vertex
  already in `visited` (a caller bug, not reachable through the current
  `deterministic_bfs_incremental` driver), it would double-toggle that
  vertex's edges in the boundary set, corrupting the count. Guarded today
  by the `visited.insert(v)`-then-`absorb_vertex` call pattern, which only
  calls it once per vertex, mirrored at both call sites (seeding and layer
  expansion).

## Migration

None required. No existing code is affected.

## Rollback

Revert the four touched/added files
(`localkcut/incremental.rs`, `instance/bounded.rs`, `integration/mod.rs`,
`lib.rs`/`instance/mod.rs`/`localkcut/mod.rs` re-exports) and the two new
test/example files. Since `CutSearchStrategy::Legacy` is the only strategy
any existing code path produces, a rollback has zero behavioral effect on
any current caller.

## Rejection Criteria

This change should be reverted or reworked if:

- A later fuzzing/bake-in pass (Next Research item 3) finds a
  cut-value/witness mismatch between `Legacy` and either candidate on a
  realistic (simple) graph the current differential suite didn't cover.
- The measured speedup does not hold up on real (non-synthetic) k-NN
  graphs from an actual embedding, as opposed to the fixed-degree ring
  graph this run's benchmark uses.

## Open Questions

- Does the `IncrementalOracleCollapsedBudget` speedup hold at the
  `MinCutWrapper` level (i.e., across the up to 100 bounded-range instance
  tries a real `partition()` call makes), or does the brute-force-path
  redundancy (Limitations/Next Research item 1 in the companion README)
  dominate total latency for graphs under ~20-30 vertices regardless of
  this fix? Not measured in this run.
- Would collapsing the budget loop interact differently with the
  `ClusterHierarchy`-based strategic seed selection (`vertices.len() > 50`)
  than with the plain sorted-vertex seed list this run's benchmark
  exercises at every tested size? The differential suite's `n=40` cases
  stay under the 50-vertex hierarchy threshold; not directly tested above
  it.
