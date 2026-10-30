# Why is our dynamic-mincut `partition()` slow, and is it the connectivity structure's fault?

## Problem

`ruvector-mincut` is RuVector's general dynamic minimum-cut engine, and
`RuVectorGraphAnalyzer::partition()` is its most-used entry point: 18
dependent crates, including WASM/Node bindings and an agent-memory
compaction policy. A prior research run measured `partition()` scaling
worse-than-quadratically with vertex count — 76ms at 50 vertices, 11.4
seconds at 400 — on a simple, regular ring k-NN graph, and rejected using
it as a per-compaction primitive on latency grounds alone. The follow-up
run that fixed an unrelated determinism bug in the same code path left one
open guess about the latency problem: swap the top-level connectivity
check (`DynamicConnectivity`) for a newer, in-tree, unused
`PolylogConnectivity` module implementing a 2025 dynamic-connectivity
paper.

Guesses about where time goes in a system you haven't profiled are cheap
and often wrong. This is the measurement.

## Hypothesis

If `partition()`'s latency growth is explained by the connectivity
check — called once per query, before anything expensive happens — then
the number of other internal structures touched per query should also
scale with latency, since both would be driven by the same per-query
setup cost. If it isn't, something else inside the query path should be
flat while latency isn't.

`MinCutWrapper::num_instances()` — the number of geometric "lambda-range"
sub-problems (`BoundedInstance`) actually instantiated and searched per
query — turned out to already be a public method nobody had used for this
purpose.

## Technical Design

`RuVectorGraphAnalyzer::partition()` calls `MinCutWrapper::query()`, which
(a) checks top-level connectivity once via `DynamicConnectivity`, then (b)
walks a geometric sequence of `BoundedInstance` cut-value ranges from a
binary-search-hinted starting point, lazily instantiating each one and
running a `DeterministicLocalKCut` local search (BFS from a bounded set of
seed vertices, radius-capped) until one reports the cut value is actually
inside its range. The radius cap was a hardcoded `20` with no way to
change it.

The fix under test: expose that radius as a real parameter.
`BoundedInstance::with_max_radius(lambda_min, lambda_max, max_radius)`
sits alongside the existing `new()` (which now just calls it with a new
`DEFAULT_MAX_RADIUS = 20` constant — byte-identical behavior for every
existing caller), and `RuVectorGraphAnalyzer::with_radius(graph,
max_radius)` / `::from_knn_with_radius(neighbors, max_radius)` build on
top of it using `MinCutWrapper`'s pre-existing (but previously unused for
this) `with_factory` constructor. Nothing about the default code path
changed; this is a new, parallel, opt-in entry point.

## Actual Implementation

```rust
// crates/ruvector-mincut/src/instance/bounded.rs
impl BoundedInstance {
    pub const DEFAULT_MAX_RADIUS: usize = 20;

    pub fn new(lambda_min: u64, lambda_max: u64) -> Self {
        Self::with_max_radius(lambda_min, lambda_max, Self::DEFAULT_MAX_RADIUS)
    }

    pub fn with_max_radius(lambda_min: u64, lambda_max: u64, max_radius: usize) -> Self {
        Self {
            /* ...unchanged fields... */
            oracle: DeterministicLocalKCut::new(max_radius),
            max_radius,
            /* ... */
        }
    }
}
```

```rust
// crates/ruvector-mincut/src/integration/mod.rs
impl RuVectorGraphAnalyzer {
    pub fn with_radius(graph: Arc<DynamicGraph>, max_radius: usize) -> Self {
        let mut wrapper = MinCutWrapper::with_factory(Arc::clone(&graph), move |_g, min, max| {
            Box::new(BoundedInstance::with_max_radius(min, max, max_radius)) as Box<_>
        });
        for edge in graph.edges() {
            wrapper.insert_edge(edge.id, edge.source, edge.target);
        }
        Self { graph, wrapper, cached_min_cut: None, cached_partition: None }
    }

    pub fn num_instances_touched(&self) -> usize {
        self.wrapper.num_instances()
    }
}
```

## Actual Benchmark Evidence

`cargo run --release -p ruvector-mincut --example mincut_radius_probe`,
ring k-NN graphs (k=8) at n = 50/100/200/400, 3 reps/cell:

```text
n      variant           partition_ms   instances
50     baseline_r20            86.252          16
50     candidate_a_r8          87.985          16
200    baseline_r20          2952.513          16
200    candidate_a_r8        1501.613          16
400    baseline_r20         12385.457          16
400    candidate_a_r8        3922.387          16
```

Instance count: flat at 16 across every size, at every radius. Latency:
grows ~112x from n=50 to n=400 at the fixed radius. That combination is
the direct evidence — the top-level connectivity check runs once
regardless of instance count, so a cost that scales with n while instance
count doesn't cannot be sitting in that check. Reducing the radius from
20 to 8 gives a real, measured 1.97x speedup at n=200 and 3.16x at n=400,
with zero change in the reported cut value on every one of 6 correctness
regression tests (path graph, cycle graph, two cliques joined by a single
bridge edge, all cross-checked against the radius-20 production default).

## Limitations

The ring/circulant graph family used for the root-cause measurement has a
property that matters: its true minimum cut value (`2k`, independent of
n) is why the instance count stays flat — a confound of this specific
graph shape, not a universal law. What that flatness *does* let us
isolate cleanly, on this workload, is that the n-scaling is 100%
per-instance search cost. Whether the radius/correctness tradeoff holds on
real (non-circulant) embedding k-NN graphs is untested here and is the
most important next step before anyone relies on radius=8 in production.

## Production Relevance

Every one of the 18 crates depending on `RuVectorGraphAnalyzer` inherits
this option with zero migration cost — it's a new constructor, not a
changed one. The specific motivating case, agent-memory compaction
(`MincutGatedForgetting`), was rejected by a prior run purely on latency
grounds; this is a precondition for revisiting that question, not a
guarantee it now passes.

## RuVector Ecosystem Implications

This is a narrow, internal performance fix to a shared engine, not a new
feature surface — so most of the broader ecosystem checklist (MCP tool,
RVF portability, RVM enforcement) legitimately comes back "not
applicable," and this document says so rather than forcing a connection.
The one live thread: a future ruFlo workflow could run this same
measurement against real production graphs on a schedule and
auto-select a safe radius, rather than a human picking 8 once and hoping
it generalizes.

## Future Direction

1. Re-run against a real agent-memory k-NN graph, not a synthetic ring.
2. Apply the same instrumentation to `find_bridges()`, which re-solves a
   fresh `MinCutWrapper` per edge and is a separate, likely larger,
   latency problem in the same file.
3. Replace the fixed radius with an adaptive one that only grows when a
   `BoundedInstance` actually reports `AboveRange`, trading today's
   measurement-based confidence for correctness-by-construction.

## References

- 2026-09-05 nightly, `mincut-gated-forgetting` (ADR-345) — the original
  latency rejection.
- 2026-09-11 nightly, `mincut-partition-determinism` — source of the
  PolylogConnectivity guess this run tested and did not confirm.
- ADR-346 — the 18-crate dependency audit of this code path.
- "Dynamic Connectivity with Expected Polylogarithmic Worst-Case Update
  Time", arXiv:2510.08297 (Oct. 2025) — the paper behind the in-tree,
  still-unused `PolylogConnectivity` module.
