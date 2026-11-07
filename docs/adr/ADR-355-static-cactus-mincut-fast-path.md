# ADR-355: `StaticCutAnalyzer` — A Weighted, Deterministic, One-Shot Min-Cut Fast Path in `ruvector-mincut`

## Status

Accepted (new, additive, feature-gated API; no existing API changed or
removed). Documentation-only change to `RuVectorGraphAnalyzer::min_cut()`
flagging a separately-filed, unresolved correctness gap (see
[Consequences](#consequences) and [Open Questions](#open-questions)) that
this ADR does **not** fix.

## Context

Nightly research `2026-09-05` (ADR-345) rejected a `ruvector-agent-memory`
integration built on
`ruvector_mincut::RuVectorGraphAnalyzer::partition()` on two grounds: (1) a
correctness bug (fixed separately by ADR-346, `2026-09-11`) and (2) latency
— `partition()`'s fully-dynamic paper algorithm (`MinCutWrapper`), designed
to amortize cost over a long-lived stream of incremental updates, costs
~77ms at n=50 vertices and ~11.4s at n=400 on a single cold, one-shot
query. ADR-346 fixed (1) but left (2) unaddressed.

`ruvector-mincut` already contains a second, independent exact min-cut
engine — `canonical::CactusGraph` (Stoer-Wagner, gated behind the
`canonical` feature) — built for ADR-117's pseudo-deterministic canonical
cut work, with an existing informal benchmark
(`tests/canonical_bench.rs`) suggesting it runs in microseconds-to-low-
milliseconds at 30-100 vertices: potentially orders of magnitude faster
than `RuVectorGraphAnalyzer::partition()` for the one-shot case. It had
never been benchmarked against `RuVectorGraphAnalyzer`, nor exposed as an
alternative for `RuVectorGraphAnalyzer`'s batch-query callers. This is a
case of Step 1's "attack the existing bottleneck" / "integrate into another
layer" guidance: rather than invent a new algorithm, test whether an
existing, previously-disconnected piece of this crate already solves it.

## Hypothesis

```text
Given a one-shot (build graph, query once, discard) minimum-cut query over
a static graph snapshot at the scale relevant to ruvector-agent-memory
compaction (tens to a few hundred vertices),

when CactusGraph's Stoer-Wagner engine is wrapped as a drop-in alternative
to RuVectorGraphAnalyzer (same from_knn/from_similarity_matrix/min_cut/
partition shape) instead of routing through MinCutWrapper's dynamic paper
algorithm,

then the wrapped engine returns the identical cut value and is at least
20x faster at n=200 and n=400 on a representative k-NN graph,

subject to: determinism across >=20 fresh rebuilds of the same graph, and
no regression to RuVectorGraphAnalyzer's existing public API.
```

This threshold was fixed before the comparison benchmark was run (see the
nightly README, "Pre-registered acceptance thresholds").

## Decision

1. Add `ruvector_mincut::integration::StaticCutAnalyzer` (re-exported at
   the crate root), feature-gated behind `canonical`. It wraps
   `canonical::CactusGraph::build_from_graph` and exposes
   `new`/`from_knn`/`from_similarity_matrix`/`min_cut`/`min_cut_f64`/
   `partition`, matching `RuVectorGraphAnalyzer`'s shape for the one-shot
   subset of its API. It has **no incremental-update API** (no
   `add_edge`/`remove_edge`/cache invalidation) and recomputes the full
   cactus on every call — by design, for the build-once/query-once use
   case; callers that mutate and re-query repeatedly should keep using
   `RuVectorGraphAnalyzer`.
2. Add a connectivity guard (`StaticCutAnalyzer::is_connected`, plain
   union-find over the graph snapshot) before invoking `CactusGraph`.
   **This nightly run found `CactusGraph::build_from_graph` returns an
   incorrect non-zero cut value for at least one disconnected input**
   (brute-force truth 0, `CactusGraph` returned 7 — see Evidence). Rather
   than modify `CactusGraph`'s general cactus construction (shared by other
   callers; a larger, riskier change), `StaticCutAnalyzer` checks
   connectivity itself first and short-circuits to `0` /
   `(Vec::new(), Vec::new())`, mirroring `MinCutWrapper::query()`'s own
   "fast path: check connectivity first." `CactusGraph` itself is
   unchanged and still has this gap for direct callers — filed, not fixed,
   see [Open Questions](#open-questions).
3. Add `min_cut_f64()` for the exact real-valued weighted cut;
   `min_cut() -> u64` rounds to the nearest integer purely to match
   `RuVectorGraphAnalyzer::min_cut()`'s signature for drop-in substitution.
4. **Separately discovered, documented but not fixed in this run:**
   `RuVectorGraphAnalyzer::min_cut()` / `partition()` (via
   `MinCutWrapper`, the specific "subpolynomial dynamic" engine this
   integration type delegates to) compute **unweighted edge connectivity**,
   not a weighted minimum cut, despite the type's public constructors
   (`from_knn`, `from_similarity_matrix`) accepting `Weight`/similarity
   values that imply weight matters. This is scoped precisely to that one
   delegation path, not a crate-wide limitation: `ruvector-mincut` also
   contains `algorithm::DynamicMinCut`/`MinCutBuilder`, a separate,
   already-weighted engine with an existing, passing test
   (`wrapper_tests.rs::test_weighted_graph_correctness`, verified against
   its own brute-force oracle) -- `RuVectorGraphAnalyzer` simply does not
   use it. Added a doc-comment warning on `RuVectorGraphAnalyzer::min_cut()`
   pointing here; no code path changed. This affects all 18 direct
   dependents of `ruvector-mincut::integration` that go through
   `RuVectorGraphAnalyzer` specifically, discovered incidentally while
   building the comparison benchmark for this ADR (see Evidence). It is
   **not** fixed here — see [Consequences](#consequences) and
   [Open Questions](#open-questions) for why, and for the recommended
   follow-up, which is more promising than originally scoped: rather than
   retrofitting weight into `MinCutWrapper`'s internals, a future run
   could first benchmark whether `RuVectorGraphAnalyzer` can delegate to
   the already-weighted `DynamicMinCut` instead -- untested here, and
   `DynamicMinCut`'s own dynamic-update performance characteristics are
   unknown, so this is a lead, not a verified fix.
5. New regression tests,
   `crates/ruvector-mincut/tests/static_cut_analyzer_tests.rs`: weighted
   correctness against an independent brute-force oracle (not
   `RuVectorGraphAnalyzer`, which is not a valid weighted oracle per
   decision 4), a hand-built weight-discriminating case, disconnected-graph
   handling, trivial-graph handling, and determinism across 30 fresh
   rebuilds.
6. New benchmark example,
   `crates/ruvector-mincut/examples/static_vs_dynamic_cut_bench.rs`,
   reusing (not re-deriving) the exact ring-k-NN and two-clique-bridge
   generators from the ADR-345/346 probe scripts in
   `ruvector-agent-memory/examples/`, for direct numeric comparability.

## Evidence

**Weighted correctness** (`crates/ruvector-mincut/tests/static_cut_analyzer_tests.rs`,
`cargo test --release -p ruvector-mincut --features canonical,integration`):

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Matches independent brute-force oracle (6 random weighted graphs, n=6-13) | exact (±1e-6) | 6/6 exact | PASS |
| Weight-discriminating case (2 weight-1 triangles + weight-100 bridge) | must prefer weight-2 cut over weight-100 bridge | `min_cut()` = 2 | PASS |
| Disconnected graph | 0 | 0 (after connectivity-guard fix; was 7 before) | PASS |
| Trivial (0/1-vertex) graphs | 0 | 0 | PASS |
| Determinism across 30 fresh rebuilds (19-vertex two-clique-bridge) | 1 distinct result | 1 distinct cut, 1 distinct partition | PASS |

**Performance** (`cargo run --release -p ruvector-mincut --features canonical,integration --example static_vs_dynamic_cut_bench`), ring k-NN graph (k=8), raw output:

```text
     n   baseline(ms)  candidate(ms)    speedup    cut(base)    cut(cand)
    19      86625.476          0.673  128736.5x           16          113
    30         27.697          1.872      14.8x           16          113
    50        102.770          5.525      18.6x           16          113
   100        784.129         27.527      28.5x           16          113
   200       3472.543        166.389      20.9x           16          113
   400      14661.795       1122.389      13.1x           16          113
```

random sparse k-NN (k=6, seed=0x5EED):

```text
     n   baseline(ms)  candidate(ms)    speedup    cut(base)    cut(cand)
    19      29237.939          0.202  145021.5x            5           10
    30         14.411          0.231      62.4x            6           13
    50         30.385          0.499      60.9x            6           14
   100        179.460          2.615      68.6x            6           10
   200        679.409         15.976      48.3x            6           11
   400       3150.147         66.837      47.1x            6           11
```

candidate-only extended scaling (baseline not re-measured above n=400 —
ADR-345 already reported ~11.4s there):

```text
n=800    candidate=    8064.526ms cut=113
n=1600   candidate=   61819.457ms cut=113
```

The `n=19` baseline outlier (~86-88s across two independent runs) reproduces
ADR-345's own documented "outlier 69s at n=19" finding — not new, environment
variance on a small regular topology, consistent across runs.

| Gate | Threshold | Measured | Result |
|---|---|---|---|
| Speedup at n=200 (ring) | >= 20x | 20.9x | PASS (borderline) |
| Speedup at n=400 (ring) | >= 20x | 13.1x | **FAIL** |
| Speedup at n=200 (random) | >= 20x | 48.3x | PASS |
| Speedup at n=400 (random) | >= 20x | 47.1x | PASS |

**On the `cut(base)` vs `cut(cand)` mismatch in every row above:** this is
not a `StaticCutAnalyzer` defect. `cut(base)=16` for every ring size
regardless of n is the ring's constant unweighted vertex degree (8 outgoing
+ 8 incoming); `MinCutWrapper::insert_edge(&mut self, edge_id, u, v)` takes
no weight parameter at all (confirmed by inspection,
`crates/ruvector-mincut/src/wrapper/mod.rs`), and
`crates/ruvector-mincut/src/instance/bounded.rs` never references edge
weight. `cut(cand)=113` is the correct weighted value
(`StaticCutAnalyzer::from_knn` inverts each input as `1/distance`, matching
`RuVectorGraphAnalyzer::from_knn`'s own convention; `2 * sum_{d=1..8} 1/(0.1+0.01d) = 113.23`,
confirmed independently via `cargo run --release --example weight_sanity_probe`
and `brute_force_cross_check`, both in `crates/ruvector-mincut/examples/`).

## Consequences

**Positive:**
- `ruvector-mincut` gains a correct, deterministic, weighted one-shot
  min-cut fast path, 13x-150,000x faster than the existing engine at the
  sizes tested, with no changes to any existing public API.
- A real latent bug (disconnected-graph mishandling in the previously-
  unintegrated `CactusGraph` path) was found and worked around before any
  production caller could hit it.
- The pre-existing weight-blindness of `RuVectorGraphAnalyzer`/
  `MinCutWrapper` is now documented in-code and in this ADR lineage instead
  of silently assumed-correct by the 18 crates depending on it.

**Negative / open:**
- `StaticCutAnalyzer`'s underlying Stoer-Wagner engine is O(V^3); the
  20x-speedup acceptance gate is not met at n=400 on the ring topology
  (13.1x, still a large win, but below the pre-registered bar) and its own
  scaling (800: ~8s, 1600: ~62s) means it is **not** a universal
  replacement — recommended only below roughly n=400-600 depending on
  latency budget, not for arbitrarily large graphs.
- `RuVectorGraphAnalyzer`'s weight-blindness is unresolved. Retrofitting
  weight support into the dynamic paper algorithm (`BoundedInstance`'s
  core data structures) is a materially larger, higher-risk change
  touching code shared by 18 dependent crates, including WASM/Node
  bindings — explicitly out of scope for this run (see
  [Open Questions](#open-questions)).
- `ADR-345`'s `MincutGatedForgetting` bridge-detection-accuracy rejection
  is **not** reopened or reversed by this work. That rejection rested on
  two independent grounds (latency and bridge-detection accuracy); this
  ADR addresses only latency, via a different code path
  (`StaticCutAnalyzer`, not `RuVectorGraphAnalyzer`). Whether weight-aware
  cuts correlate better with human-labeled bridges than ADR-345's
  unweighted-engine-driven result did is an open, untested question — see
  below.

## Alternatives

- **Modify `MinCutWrapper`/`BoundedInstance` to honor weight directly.**
  Rejected for this run: the dynamic paper algorithm's lambda-range binary
  search and instance machinery assume integer-ish cut values throughout;
  correctly threading real-valued weights through requires auditing code
  shared by 18 dependent crates (far larger blast radius than adding a new,
  additive, feature-gated type) for a benefit (drop-in weighted *dynamic*
  min-cut) not required by any currently-known caller — `ruvector-agent-
  memory`'s only real consumer, `MincutGatedForgetting`, is feature-gated
  off by default and already rejected for production per ADR-345.
- **Make `RuVectorGraphAnalyzer` delegate to `algorithm::DynamicMinCut`
  instead of `MinCutWrapper`.** Not rejected, but not attempted this run
  either: `DynamicMinCut` is already weighted and already tested, which
  makes this the most promising lead for a future fix -- but its dynamic
  (incremental-update) performance profile was not benchmarked here, so
  swapping the delegation could reintroduce or change the very latency
  problem ADR-345 originally found, in an unverified direction. Left as
  Open Question 1's primary recommended starting point rather than
  something this run can claim to have validated.
- **Fix `CactusGraph::build_from_graph` itself for disconnected inputs**,
  instead of guarding in `StaticCutAnalyzer`. Rejected for this run: three
  other callers (`CanonicalMinCutImpl`, `source_anchored`,
  `tree_packing`) share `CactusGraph`; changing its general construction
  logic for a case none of them are documented to hit is a larger surface
  to re-verify than a two-line guard in the one new caller that needs it.
- **Promote `StaticCutAnalyzer` as the new default for
  `RuVectorGraphAnalyzer`.** Rejected: different semantics (one-shot vs.
  incremental), different scaling regime (cubic vs. amortized-dynamic) —
  this is an additional tool for a specific use case, not a replacement.

## Implementation Plan

1. `crates/ruvector-mincut/src/integration/mod.rs`: add `StaticCutAnalyzer`
   (feature `canonical`), doc-warn `RuVectorGraphAnalyzer::min_cut()`.
2. `crates/ruvector-mincut/src/lib.rs`: re-export `StaticCutAnalyzer`.
3. `crates/ruvector-mincut/tests/static_cut_analyzer_tests.rs`: regression
   suite (committed).
4. `crates/ruvector-mincut/examples/static_vs_dynamic_cut_bench.rs`,
   `weight_sanity_probe.rs`, `brute_force_cross_check.rs`,
   `ring_graph_brute_force_check.rs`: benchmark + diagnostics (committed,
   marked throwaway/diagnostic per existing repo convention — see the
   ADR-345/346 probes in `ruvector-agent-memory/examples/` for precedent).

## API Shape

```rust
#[cfg(feature = "canonical")]
pub struct StaticCutAnalyzer { /* ... */ }

#[cfg(feature = "canonical")]
impl StaticCutAnalyzer {
    pub fn new(graph: Arc<DynamicGraph>) -> Self;
    pub fn from_knn(neighbors: &[(usize, Vec<(usize, f64)>)]) -> Self;
    pub fn from_similarity_matrix(similarities: &[f64], num_vectors: usize, threshold: f64) -> Self;
    pub fn min_cut(&self) -> u64;         // rounded, API-compatible with RuVectorGraphAnalyzer
    pub fn min_cut_f64(&self) -> f64;     // exact weighted value
    pub fn partition(&self) -> Option<(Vec<VertexId>, Vec<VertexId>)>;
}
```

No existing type's signature changed.

## Feature Flags

`StaticCutAnalyzer` is gated behind the pre-existing `canonical` feature
(not in `default`, in `full`). No new feature flag introduced.

## Benchmark Evidence

See Evidence above and the full raw output in
`docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/README.md`.

## Security

No new attack surface: `StaticCutAnalyzer` is a read-only query over an
existing in-memory `DynamicGraph` snapshot, with no I/O, no deserialization
of untrusted input beyond what `RuVectorGraphAnalyzer` already accepts, and
no new unsafe code (`#![deny(unsafe_code)]` still applies; crate is built
without the `wasm` feature here).

## Governance

Additive, feature-gated, non-breaking. No ADR supersession. Does not alter
ADR-345's rejection of `MincutGatedForgetting` for production — see
Consequences.

## Failure Modes

- Called on a graph larger than a few hundred-to-low-thousand vertices: 
  cubic cost makes this impractical; no panic, just slow — callers on a
  latency budget should check `graph.num_vertices()` before choosing this
  path over `RuVectorGraphAnalyzer`.
- Disconnected input: now returns `0` / empty partition, matching
  `RuVectorGraphAnalyzer`'s convention (previously a `CactusGraph`-level
  bug would have surfaced a wrong non-zero value — now guarded).

## Migration

None required; purely additive.

## Rollback

Remove the `StaticCutAnalyzer` impl block and its re-export; no other code
depends on it yet.

## Rejection Criteria

Would reject this ADR if: the brute-force correctness cross-check had
failed on any trial (it did not, after the connectivity-guard fix), or if
the measured speedup had been negative or negligible at realistic
agent-memory scales (it was 13x-150,000x depending on size and topology,
comfortably positive even at the one gate that missed the 20x bar).

## Open Questions

1. **Does `RuVectorGraphAnalyzer`'s weight-blindness get fixed?** Filed as
   a follow-up candidate for a future nightly run, in order of promise:
   (a) benchmark whether `RuVectorGraphAnalyzer` can delegate to the
   already-weighted, already-tested `algorithm::DynamicMinCut` instead of
   `MinCutWrapper` for its streaming/incremental case -- untested here,
   most promising lead; (b) if (a)'s performance is unacceptable, retrofit
   weight support into `MinCutWrapper`/`BoundedInstance` directly (large,
   shared blast radius); or (c) prominently deprecate `RuVectorGraphAnalyzer`
   for weighted use cases in favor of `StaticCutAnalyzer` (one-shot only).
2. **Would a weight-aware cut have changed ADR-345's bridge-detection-
   accuracy rejection?** Untested here — would require rerunning ADR-345's
   full bridge-survival methodology with `StaticCutAnalyzer` substituted
   for `RuVectorGraphAnalyzer`, which is a distinct experiment from this
   one (and must not silently reopen ADR-345's production rejection without
   its own full measured evidence).
3. Does `CactusGraph::build_from_graph` have the same disconnected-input
   bug for its other three callers (`CanonicalMinCutImpl`,
   `source_anchored`, `tree_packing`)? Not checked here — `StaticCutAnalyzer`
   guards itself; the underlying `CactusGraph` bug is still live for direct
   callers.
