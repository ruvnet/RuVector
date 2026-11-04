# Your Min-Cut Library Has Three Engines. One of Them Doesn't Look at Edge Weight.

## The setup

`ruvector-mincut` is a Rust crate implementing dynamic minimum-cut
algorithms — used across 18 other crates in the RuVector ecosystem for
graph connectivity analysis, community detection, and (experimentally)
agent-memory bridge detection. Its main integration-layer type,
`RuVectorGraphAnalyzer`, exposes constructors like `from_knn` and
`from_similarity_matrix` that explicitly take per-edge `Weight` values —
the whole point being that a strong similarity edge should behave
differently from a weak one when the library decides what to cut.

A prior research run (ADR-345, Sept 2026) measured this type's
`partition()` call costing 77ms at 50 vertices and 11.4 seconds at 400
vertices for a single cold query, and rejected an agent-memory integration
built on it partly for that reason. We went back to see if that latency
problem could be fixed.

## What we found instead

It can be fixed — but along the way we found something more interesting.
`MinCutWrapper::insert_edge`, the function `RuVectorGraphAnalyzer` syncs
through, has this signature:

```rust
pub fn insert_edge(&mut self, edge_id: EdgeId, u: VertexId, v: VertexId)
```

No weight parameter. At all. The dynamic algorithm underneath
(`BoundedInstance`, implementing a Thorup-style amortized dynamic
connectivity scheme) never references edge weight anywhere in its source.

We confirmed this isn't just a code-reading mistake with two direct tests:

- A fixed-degree weighted ring graph (every vertex connected to its next
  8 neighbors, weights `0.11` through `0.18`) returns the *exact same*
  minimum-cut value — `16` — at every size from 19 to 1600 vertices. `16`
  is the ring's constant unweighted vertex degree. A weighted cut should
  not be size-invariant like that; an unweighted edge-count cut is.
- Two weight-1 triangles joined by a single weight-100 "bridge" edge: a
  weight-aware minimum cut must prefer severing two cheap triangle edges
  (cost 2) over one very expensive bridge edge (cost 100). `RuVectorGraphAnalyzer`
  returns `1` — it cut the bridge, because in edge-*count* terms, one edge
  beats two, regardless of weight.

So the type's API promises weight-awareness it doesn't deliver. Eighteen
dependent crates have been calling a "weighted" API that's silently
unweighted underneath.

## The fix we shipped (narrower than it sounds)

`ruvector-mincut` already had a second exact min-cut engine sitting mostly
unused: `canonical::CactusGraph`, a Stoer-Wagner implementation built for
an earlier "canonical cut" research effort, with an informal benchmark
suggesting it ran in microseconds at 30-100 vertices — a completely
different performance profile from the dynamic engine. Nobody had ever
benchmarked the two against each other.

We wrapped it as `StaticCutAnalyzer`, a drop-in, API-compatible alternative
for one-shot (build-once, query-once) use, and ran it against
`RuVectorGraphAnalyzer` on two synthetic topologies from 19 to 1600
vertices.

```text
     n   baseline(ms)  candidate(ms)    speedup
    19      86625.476          0.673  128736.5x
    30         27.697          1.872      14.8x
    50        102.770          5.525      18.6x
   100        784.129         27.527      28.5x
   200       3472.543        166.389      20.9x
   400      14661.795       1122.389      13.1x
```

13x to 150,000x faster, correctly weighted (we verified the candidate
against an independent brute-force oracle across six random graphs and two
hand-built edge cases — not against `RuVectorGraphAnalyzer`, which we'd
just shown isn't a valid weighted oracle), deterministic across 30 fresh
rebuilds of the same graph. Along the way we also found and fixed a real
bug in the new path: it silently returned a wrong, non-zero cut value on
disconnected graphs until we added a connectivity guard.

The candidate's own cost grows cubically (classic Stoer-Wagner), so it's
not a universal replacement — recommended below roughly 400-600 vertices,
not for arbitrarily large graphs.

## What we didn't fix

The weight-blindness itself is still there in `RuVectorGraphAnalyzer`.
Fixing it properly has two paths: retrofit weight into the dynamic
algorithm's internals (large blast radius, touches code shared by 18
crates including WASM/Node bindings), or — more promising, and unverified
here — make `RuVectorGraphAnalyzer` delegate to a *third* engine already
in the crate, `algorithm::DynamicMinCut`, which is already weighted and
already has a passing correctness test. We didn't benchmark
`DynamicMinCut`'s own dynamic-update performance, so we're not claiming
that's a free fix — just a more promising lead than starting from scratch.

We also didn't reopen the original ADR-345 decision. That rejection had
two independent grounds — latency and a separate bridge-detection-accuracy
failure. We fixed latency, via a different code path entirely. Whether a
weight-aware cut would have changed the accuracy finding is a genuinely
open, untested question, and we're not claiming an answer we didn't
measure.

## The broader point

Three min-cut engines, in one crate, with meaningfully different
weight-handling and performance characteristics, had never been
benchmarked against each other before this run. The fastest fix to a
measured bottleneck wasn't a new algorithm — it was noticing that a
correct, fast implementation already existed in the same crate, unused,
and that the "slow" path everyone complained about had a second, more
fundamental problem nobody had actually checked for.

## Code

- `crates/ruvector-mincut/src/integration/mod.rs` — `StaticCutAnalyzer`
- `crates/ruvector-mincut/tests/static_cut_analyzer_tests.rs` — regression suite
- `crates/ruvector-mincut/examples/static_vs_dynamic_cut_bench.rs` — the benchmark above
- `crates/ruvector-mincut/examples/weight_sanity_probe.rs`,
  `brute_force_cross_check.rs` — the diagnostics that found the weight-blindness
  and validated the fix
- Full writeup: `docs/research/nightly/2026-11-04-canonical-mincut-static-fast-path/README.md`
- Decision record: `docs/adr/ADR-355-static-cactus-mincut-fast-path.md`

## References

- Stoer, M. and Wagner, F., "A Simple Min-Cut Algorithm," J. ACM 44(4), 1997.
- ADR-345 (`docs/adr/ADR-345-mincut-gated-forgetting.md`) and ADR-346
  (`docs/adr/ADR-346-deterministic-mincut-witness-partition.md`) — the
  prior nightly runs this one continues.
