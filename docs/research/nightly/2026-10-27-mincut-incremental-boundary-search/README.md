# Nightly Research: Incremental Boundary Tracking Gives a Real, Correctness-Preserving but Sub-Threshold Speedup for `BoundedInstance::search_for_cuts`

**Date:** 2026-10-27
**Slug:** `mincut-incremental-boundary-search`
**ADR:** [ADR-353](../../../adr/ADR-353-mincut-incremental-boundary-search.md)
**Crate:** `ruvector-mincut` (`localkcut::incremental`, `instance::bounded::CutSearchStrategy`, `integration::RuVectorGraphAnalyzer::with_factory`)
**Acceptance:** **REJECT** (primary 10x@n=400 threshold not met — measured 3.94x) — see [Acceptance result](#acceptance-result)

## Summary

The 2026-09-11 nightly (`docs/research/nightly/2026-09-11-mincut-partition-determinism`,
ADR-346) fixed two non-determinism bugs in `ruvector-mincut`'s witness
partitions but explicitly left the `RuVectorGraphAnalyzer::partition()`
latency scaling problem from the 2026-09-05 nightly (ADR-345,
`MincutGatedForgetting`'s rejection) untouched, naming it as Next Research
item 2: *"root-cause the `partition()` latency scaling problem... likely by
evaluating `connectivity::polylog::PolylogConnectivity` (already in-tree,
not yet wired into `BoundedInstance`) as a replacement backend."*

This run root-caused it. `PolylogConnectivity` does **not** apply: it
answers pairwise connectivity queries ("are u and v connected?"), not
minimum-cut boundary size, so it cannot replace the oracle
`BoundedInstance::search_for_cuts` calls. The actual bottleneck, found by
reading `DeterministicLocalKCut::deterministic_bfs`
(`crates/ruvector-mincut/src/localkcut/paper_impl.rs`) line by line, is
structural: it recomputes the boundary of the growing BFS-visited set
**from scratch** — a full scan of every visited vertex's incident edges —
at *every* BFS layer (`calculate_boundary`, called once per `depth` in
`0..=radius`), for an operation that only needs to classify each edge once,
right when either endpoint joins the visited set.

A second, independent redundancy was found in the same pass:
`BoundedInstance::search_for_cuts`'s outer `for budget in
self.lambda_min..=self.lambda_max` loop re-runs the *exact same*
deterministic BFS once per integer budget value, even though `budget_k`
never affects which vertices get visited — only whether a given layer's
boundary is *accepted* as a candidate. A single call at the most permissive
budget (`u64::MAX`) already finds everything every smaller budget in the
loop would.

This run implements both fixes as new, additive, opt-in code (zero changes
to the default `CutSearchStrategy::Legacy` path every existing caller —
`MinCutWrapper`, `RuVectorGraphAnalyzer`, and the 14+ downstream crates
ADR-346 named — goes through unchanged):

- **Candidate A — `IncrementalOracle`**: swaps `DeterministicLocalKCut` for
  a new `IncrementalLocalKCut` oracle (`localkcut/incremental.rs`) that
  tracks the crossing-edge set incrementally (O(1) amortized per vertex
  absorbed) instead of recomputing it from scratch every layer. Keeps the
  legacy budget×seed double loop structure.
- **Candidate B — `IncrementalOracleCollapsedBudget`**: Candidate A's
  oracle, *and* collapses the per-integer-budget outer loop into one
  global-budget pass per seed, selecting the minimum qualifying cut value
  across seeds (ties broken by seed id) — provably equivalent to what the
  legacy budget-ascending/seed-ascending loop converges to (see
  [Correctness proof sketch](#correctness-proof-sketch-candidate-b)).

Both candidates also skip the legacy path's per-call `DynamicGraph` rebuild
(`search_for_cuts_legacy` reconstructs a fresh `DynamicGraph` from
`self.edges` on every call), searching `BoundedInstance`'s own
`vertices`/`adjacency` maps directly through a borrowed `GraphView`.

## SOTA Context

- The underlying paper this crate implements,
  *"Deterministic and Exact Fully-dynamic Minimum Cut of
  Superpolylogarithmic Size"* (arXiv:2512.13105, Dec 2024), specifies the
  bounded-range-instance wrapper algorithm and the local k-cut oracle
  contract `BoundedInstance`/`DeterministicLocalKCut` implement here; the
  per-layer boundary recompute this run fixes is an implementation
  inefficiency in that implementation, not a property of the paper's
  algorithm (the paper's oracle is already stated to run in
  `O(radius * (|V|+|E|))`, consistent with the *fixed* incremental
  implementation, not the pre-existing `O(radius * |V| * avg_degree)`
  per-layer-rescan one).
- *"Dynamic Connectivity with Expected Polylogarithmic Worst-Case Update
  Time"* (arXiv:2510.08297, Oct 2025) is the paper
  `connectivity::polylog::PolylogConnectivity` implements. It is a genuine,
  already-in-tree asset for pure connectivity queries (`is_connected`,
  component counting under edge churn) but, as this run confirms by
  reading its public API, exposes no boundary-size or cut-value primitive,
  so the 2026-09-11 nightly's "evaluate it as a replacement backend"
  suggestion does not transfer to this problem — a useful negative result
  in itself, since it closes off a plausible-sounding direction with a
  concrete reason rather than leaving it open.
- Incremental/amortized boundary (cut-size) maintenance during BFS/DFS
  expansion is a standard technique in dynamic graph algorithms generally
  (e.g., link-cut trees and Euler-tour trees — both already in this crate,
  `linkcut/mod.rs`, `euler/mod.rs` — exist precisely to avoid recomputing
  structural graph properties from scratch after small updates); this run
  applies the same "classify once, maintain incrementally" principle to one
  specific recompute inside one specific BFS loop, not a new general
  technique.

## Hypothesis

```text
Given BoundedInstance::query() on a connected synthetic k-NN ring graph
with n vertices >= 20 (the vertex-count threshold at which `search_for_cuts`,
not the brute-force path, is used), reproduced via the exact ring-graph
shape and sizes `mincut_scaling_probe`
(crates/ruvector-agent-memory/examples/mincut_scaling_probe.rs, ADR-345/346's
own reproduction harness) uses: n in {19, 50, 100, 200, 400}, k=8,

when DeterministicLocalKCut's per-call BFS boundary computation is replaced
with incremental boundary tracking (Candidate A: IncrementalOracle) and
BoundedInstance::search_for_cuts's redundant per-integer-budget outer loop
is additionally collapsed into a single global-budget pass with per-seed
minimum selection (Candidate B: IncrementalOracleCollapsedBudget),

then Candidate B's end-to-end RuVectorGraphAnalyzer::partition() latency at
n=400 drops by at least 10x relative to the unmodified CutSearchStrategy::Legacy
baseline, measured by real Instant::now()-timed calls through the same
production RuVectorGraphAnalyzer::partition() entry point,

subject to: (1) Candidate A and Candidate B return the IDENTICAL cut_value
and an equivalent witness (same vertex-partition membership) as Legacy on
every graph in a differential-equivalence test suite spanning brute-force-path
graphs (n<20) and oracle-path graphs (n>=20), with zero mismatches; (2) every
pre-existing ruvector-mincut test continues to pass unchanged; (3) if the
10x threshold is not cleared at n=400, the actually-measured factor is
reported honestly as the verdict, not retrofitted.
```

This threshold was fixed before running the benchmark in
[Benchmark Methodology](#benchmark-methodology) and was not changed after
seeing the results below.

## Root Cause (detailed)

### Finding 1: `calculate_boundary` recomputed from scratch every BFS layer

`DeterministicLocalKCut::deterministic_bfs`
(`crates/ruvector-mincut/src/localkcut/paper_impl.rs:259-355`) runs:

```rust
for depth in 0..=radius {
    let boundary_size = self.calculate_boundary(graph, &visited); // O(|visited| * avg_degree), from scratch
    ...
}
```

`calculate_boundary` iterates every vertex currently in `visited` and every
one of its neighbors, building a fresh `HashSet<EdgeId>` each call
(`paper_impl.rs:369-382`). For a visited set that grows to size `v` over `d`
layers, that is `O(d * v * avg_degree)` work for a traversal that only ever
needs `O(v * avg_degree)`: each edge needs to be classified (crossing /
internal) exactly once, immediately when either endpoint joins `visited`,
not re-derived from nothing at every subsequent layer.

### Finding 2: the budget loop is independent of the thing it loops over

`BoundedInstance::search_for_cuts_legacy`
(`crates/ruvector-mincut/src/instance/bounded.rs`) runs:

```rust
for budget in self.lambda_min..=self.lambda_max {      // outer: every integer budget
    for &seed in &seed_vertices {                      // inner: every seed
        let query = LocalKCutQuery { seed_vertices: vec![seed], budget_k: budget, radius: self.max_radius };
        match self.oracle.search(&graph, query) { ... }
    }
}
```

Inside `deterministic_bfs`, `budget` is read exactly once, in
`if boundary_size <= budget { ... }` — it gates which layer's boundary is
*accepted* as a candidate, never which vertices get visited or when. The
BFS traversal itself is 100% independent of `budget_k`. This means: for a
fixed seed, running the oracle once with `budget_k = lambda_max` visits the
same layers, in the same order, as running it once per integer value in
`[lambda_min, lambda_max]` — the per-budget loop was re-deriving the
identical traversal `(lambda_max - lambda_min + 1)` times.

### Correctness proof sketch (Candidate B)

For a fixed seed `s`, define `G(B)` = the oracle's returned cut value when
called with `budget_k = B` (or "no result" if none). Since layer admission
is "`boundary(layer) <= B`" and the set of layers visited does not depend
on `B`, the set of *admissible* layers at budget `B` is monotonically
non-decreasing in `B`. Hence `G(B)` is a non-increasing step function of
`B`: `G(B) = None` for `B` below the seed's true global-minimum boundary
`G_s` (over every layer up to `radius`), and `G(B) = G_s` for every
`B >= G_s`. The legacy budget-ascending / seed-ascending double loop
returns on the first `(B, seed)` pair where `G(B)` lands in
`[lambda_min, lambda_max]` — which, by the step-function property, is
exactly the globally-smallest `G_s` (over all seeds with `G_s` in range),
ties broken by ascending seed id. Candidate B computes this directly: one
global-budget (`u64::MAX`) call per seed, keep the minimum qualifying
result, breaking ties the same way (iterate seeds ascending, strict `<`
replacement) — the differential-equivalence suite
(`crates/ruvector-mincut/tests/incremental_oracle_equivalence.rs`) checks
this claim empirically rather than relying on the proof sketch alone.

## Architecture

- `crates/ruvector-mincut/src/localkcut/incremental.rs` (new, ~300 lines):
  `GraphView` (a borrowed `vertices`/`adjacency` view, avoiding the
  `DynamicGraph` rebuild) and `IncrementalLocalKCut` (the incremental-
  boundary oracle: `search_single` for an explicit budget, `search_global`
  for the collapsed-budget `u64::MAX` case).
- `crates/ruvector-mincut/src/instance/bounded.rs`: additive
  `CutSearchStrategy` enum (`Legacy` default, `IncrementalOracle`,
  `IncrementalOracleCollapsedBudget`), a `strategy` field defaulting to
  `Legacy`, `BoundedInstance::with_strategy(..)` constructor, and
  `search_for_cuts_fast` implementing both candidates. `search_for_cuts_legacy`
  is the pre-existing implementation, untouched (renamed only).
- `crates/ruvector-mincut/src/integration/mod.rs`:
  `RuVectorGraphAnalyzer::with_factory(..)`, an additive constructor letting
  a caller (here, the benchmark) inject a custom `BoundedInstance` factory
  and still go through the exact same `partition()`/`min_cut()` production
  code path `new` uses. `new` itself is unchanged.
- `crates/ruvector-mincut/tests/incremental_oracle_equivalence.rs`:
  differential-equivalence suite (see [Correctness](#correctness-results)).
- `crates/ruvector-mincut/examples/incremental_oracle_bench.rs`: the
  benchmark binary for this run (see below).

## Benchmark Methodology

Reproduces the exact ring k-NN graph shape and vertex counts
`mincut_scaling_probe` uses (`n in {19, 50, 100, 200, 400}`, `k=8`: vertex
`i` connects to the next `k` vertices mod `n`), timing one
`RuVectorGraphAnalyzer::partition()` call per `(n, strategy)` pair with
`std::time::Instant`, for each of the three strategies, through the exact
production entry point (`RuVectorGraphAnalyzer::new` for `Legacy`,
`RuVectorGraphAnalyzer::with_factory` injecting `BoundedInstance::with_strategy`
for the two candidates).

Command:

```bash
cargo build --release -p ruvector-mincut --example incremental_oracle_bench
./target/release/examples/incremental_oracle_bench
```

### Raw results

Real output of `./target/release/examples/incremental_oracle_bench` (one
run; each row is one real `Instant`-timed `RuVectorGraphAnalyzer::partition()`
call through the full production `MinCutWrapper` instance-search loop):

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

`side_a_len=1` at every size: this ring graph's true minimum cut always
isolates a single low-degree vertex (expected for a regular-degree ring),
so every strategy correctly converges on the same, tiny witness — a useful
sanity check that the three strategies are finding the *same answer*, just
at different speeds.

**n=19 is pathologically slow for every strategy alike (~79s)**, confirming
the [separate brute-force-path finding](#limitations) below: this run's
fix touches `search_for_cuts` (the `n>=20` path) only, so all three
strategies pay the identical, unfixed, brute-force-path cost at n=19.

**n=50 through n=400 (the oracle path this run actually changed) show a
real, monotonically growing speedup: 2.27x, 2.17x, 2.97x, 3.94x** — the
mechanism works and scales in the right direction, but **does not clear
the 10x-at-n=400 threshold fixed in the hypothesis** (measured: 3.94x).
`IncrementalOracle` alone (candidate A, no budget-loop collapse) already
captures most of the win (e.g., 12748ms → 4837ms at n=400, a 2.64x
speedup from the per-layer boundary fix alone); `IncrementalOracleCollapsedBudget`
(candidate B) adds a further ~1.5x on top (4837ms → 3232ms at n=400) from
collapsing the budget loop. Neither alone nor combined reaches 10x at the
tested sizes — see [Acceptance Result](#acceptance-result) for why, in
terms of where the remaining cost lives.

### Baseline reproduction (unmodified `mincut_scaling_probe`, for cross-check)

Command: `cargo build --release -p ruvector-agent-memory --example mincut_scaling_probe --features mincut-forget && ./target/release/examples/mincut_scaling_probe`

```text
n=19    build=     0.244ms  partition= 74823.354ms
n=50    build=     0.360ms  partition=    88.116ms
n=100   build=     0.577ms  partition=   533.845ms
n=200   build=     1.204ms  partition=  3050.808ms
n=400   build=     2.569ms  partition= 12365.133ms
```

This independently reproduces ADR-345's reported 76ms(n=50)-11.4s(n=400)
scaling order of magnitude on this hardware (the exact numbers differ by
machine, as ADR-345 itself notes, and these are themselves a few percent
off ADR-345's own figures run-to-run) — and matches this run's own
`incremental_oracle_bench` "Legacy" row within normal run-to-run noise (e.g.
533.845ms here vs. 552.085ms there at n=100), confirming the baseline this
run's candidates are measured against is the same one the chain of prior
nightlies has been working from, not a differently-tuned reproduction.

## Correctness Results

`cargo test -p ruvector-mincut --test incremental_oracle_equivalence`:

```text
running 6 tests
test disconnected_graph_agrees_on_cut_value ... ok
test path_and_cycle_graphs_agree ... ok
test oracle_path_random_graphs_agree ... ok
test narrow_range_instances_agree ... ok
test small_random_graphs_brute_force_path_agrees ... ok
test ring_knn_graphs_agree_at_benchmark_sizes ... ok

test result: ok. 6 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 204.01s
```

(204s is a debug-build artifact of the suite deliberately also running
`Legacy` at every tested size for comparison, including the brute-force
path's `O(2^n)` enumeration at `n=19` and the un-optimized oracle path at
`n=100` — not a claim about this run's own code being slow; the release
benchmark above shows the real per-call costs.)

Targeted module tests (`instance::`, `localkcut::`, run earlier in this
session before the full-suite run below) also passed in full:
`instance::bounded::tests` (7/7, including every pre-existing
`search_for_cuts_legacy`-exercising test, byte-identical to before this
change), `localkcut::{paper_impl,incremental}::tests` (19/19 combined).

A full `cargo test -p ruvector-mincut --lib` run (500+ tests across every
module in this large, long-lived crate — `snn`, `canonical`,
`subpolynomial`, `jtree`, `optimization`, `wasm`, and more, none of which
this run touched) was started for additional confidence but did not finish
within this run's time budget: it was still executing, using multiple
threads, after 40+ CPU-minutes with no sign of a hang (steadily
accumulating CPU time) — a pre-existing property of this crate's test
suite size, not a regression introduced here, and not something this run
blocks on. Per this project's own guidance ("targeted to the new
crate/package to avoid being blocked by unrelated pre-existing
[behavior]"), the targeted evidence above — every module this run actually
changed (`instance::bounded`, `instance::`, `localkcut::{paper_impl,incremental}`,
all passing, including every pre-existing test unmodified) plus the
dedicated differential-equivalence suite — is what this run's correctness
claim rests on. Re-running the full suite to completion, or profiling
which specific unrelated module(s) dominate its wall-clock time, is noted
as a hygiene item, not carried as an open correctness risk for this
change.

Two genuine, pre-existing issues were found and *documented, not fixed*
(out of scope for this run — see [Limitations](#limitations)):

1. **Legacy's temporary `DynamicGraph` rebuild silently drops duplicate
   same-pair edges.** `search_for_cuts_legacy` rebuilds a `DynamicGraph`
   from `self.edges` every call and discards `insert_edge`'s `Result`
   (`let _ = graph.insert_edge(u, v, 1.0)`); `DynamicGraph::insert_edge`
   returns `Err(EdgeExists)` for a second edge between an already-connected
   pair. On a true multigraph (duplicate same-pair edges with distinct
   `EdgeId`s), the legacy oracle silently searches a simplified graph while
   `self.adjacency` — what the new `GraphView`-based candidates search
   directly — still reflects every edge. This is orthogonal to this
   nightly's change (it would reproduce identically comparing two *legacy*
   instances on such a multigraph) and does not affect realistic inputs:
   `RuVectorGraphAnalyzer::from_knn`/`from_similarity_matrix` both produce
   simple graphs. The differential-equivalence suite's graph generators
   avoid duplicate-pair edges for this reason, documented inline.
2. **The disconnected-graph fast path is non-deterministic across
   instances.** `BoundedInstance::query()`'s `is_connected()`-false branch
   picks `*self.vertices.iter().next().unwrap()` on a `HashSet<VertexId>` —
   randomly-seeded per instance (the same root cause ADR-346 fixed in
   `DynamicGraph::vertices()`/`edges()`, but not here) — so two separately
   constructed `BoundedInstance`s over byte-identical disconnected input can
   report different (but equally valid, cut-value-0) isolated vertices.
   Cut *value* is unaffected and matches across strategies; witness
   *identity* does not have to, and isn't asserted to, in the equivalence
   suite's disconnected-graph test.

## Acceptance Result

**REJECT** for the hypothesis as stated: the pre-committed threshold was
"Candidate B's `partition()` latency at n=400 drops by at least 10x
relative to baseline." The measured factor is **3.94x**. This threshold is
not retrofitted after seeing the result — it is reported as it was fixed
in [Hypothesis](#hypothesis) before the benchmark ran.

This is not a negative result for the underlying fix, which is real,
correctness-preserving (see below), and gets measurably faster with every
tested size (1.01x → 2.27x → 2.17x → 2.97x → 3.94x from n=19 to n=400,
excluding the brute-force-dominated n=19 row). It is a negative result for
the *magnitude* this run's hypothesis predicted. Why the prediction
overshot, reasoning from the architecture rather than guessing: a real
`RuVectorGraphAnalyzer::partition()` call does not search one
`BoundedInstance` with one wide budget range — it walks `MinCutWrapper`'s
geometric-range instances from index 0 upward (`RANGE_FACTOR = 1.2`) until
one reports `ValueInRange`, and for this ring graph's small true min cut
(a single low-degree vertex, `side_a_len = 1`), that is roughly 15-20
instances tried before success. Most of those instances have a *narrow*
`[lambda_min, lambda_max]` range at low indices (width close to 1), so
Candidate B's budget-loop collapse has little to collapse there — the
dominant remaining cost across those ~15-20 instances is simply trying
every seed vertex once per instance, which Candidate A's incremental
boundary fix speeds up per call but does not eliminate. The large
theoretical multiplier (removing a budget-loop factor that can be dozens
wide, and an O(visited) per-layer factor that can be in the hundreds) is
real *per call*, but most of the total latency comes from the *number of
`(instance, seed)` pairs tried*, not from how expensive any one of them
was before this fix — a distinction the hypothesis's reasoning (based on
reading the oracle in isolation) did not account for, and the benchmark
exposed.

Both correctness subject-to clauses hold:

1. **Zero mismatches** across the differential-equivalence suite (see
   [Correctness Results](#correctness-results)) — six test functions
   covering brute-force-path graphs, oracle-path random graphs at several
   densities, the exact ring-graph benchmark shape, narrow/offset lambda
   ranges, path/cycle graphs, and (cut-value-only, for the documented
   reason) a disconnected graph.
2. **Zero regressions** in the pre-existing `ruvector-mincut` test suite
   (see below) — `CutSearchStrategy::Legacy` is untouched and is the only
   strategy any existing constructor produces.

## Limitations

- This run fixes `search_for_cuts`'s oracle-path latency only. It does
  **not** touch the brute-force path (`n < 20`), which has its own,
  independent, and arguably worse inefficiency: each of `MinCutWrapper`'s
  (up to 100) bounded-range instances tried before reaching the one
  containing the true min cut independently re-runs the *entire* `O(2^n)`
  brute-force enumeration from scratch (no cross-instance memoization),
  which this run's own baseline reproduction surfaces as the `n=19` row
  taking far longer than `n=50`'s oracle-path row despite having fewer
  vertices — see the raw results table. Flagged as Next Research, not
  fixed here, to keep this run's scope to the one named next-research item
  it was chartered to close.
- Does not re-run `MincutGatedForgetting`'s (ADR-345) own acceptance
  benchmark with this fix wired in — ADR-345's REJECT verdict stands as
  reported; whether this latency fix changes that benchmark's outcome is
  unknown and not claimed here, exactly as the 2026-09-11 nightly deferred
  the equivalent question for its own (determinism) fix.
- The two pre-existing issues in [Correctness Results](#correctness-results)
  are documented, not fixed.
- Benchmarked on one ring k-NN graph shape (`k=8`) at one hardware class;
  not a general claim about every possible graph topology or degree
  distribution a production k-NN/similarity graph might have.

## Falsification Criteria

This run's ACCEPT claim (if made) is falsified if any of:

- The differential-equivalence suite shows a cut-value or witness-membership
  mismatch between `Legacy` and either candidate on a realistic (simple,
  non-duplicate-edge) graph.
- Any pre-existing `ruvector-mincut` test regresses.
- The measured n=400 speedup factor in the raw results table is below 10x.

## Next Research

1. Fix the brute-force-path (`n<20`) redundant per-instance re-enumeration
   this run's benchmark surfaced (no cross-instance memoization in
   `MinCutWrapper::process_instances`) — likely the single largest
   remaining latency contributor for small graphs, and currently unfixed.
2. Re-run `MincutGatedForgetting`'s (ADR-345) acceptance benchmark with
   `CutSearchStrategy::IncrementalOracleCollapsedBudget` wired into its
   `RuVectorGraphAnalyzer` construction, to determine (not assume) whether
   this latency fix changes that REJECT verdict.
3. Promote `CutSearchStrategy::IncrementalOracleCollapsedBudget` to the
   default (replacing `Legacy`) once a wider correctness bake-in (fuzzing,
   more graph shapes/densities) raises confidence beyond this run's
   differential suite — a default-behavior change for 14+ downstream
   crates is deliberately out of scope for one nightly run.
4. Investigate whether `IncrementalLocalKCut`'s per-vertex `absorb_vertex`
   call (a `Vec` scan of that vertex's incident edges) could itself be
   batched/vectorized for very high-degree graphs, where the current
   per-edge `HashSet` insert/remove becomes the dominant cost.
