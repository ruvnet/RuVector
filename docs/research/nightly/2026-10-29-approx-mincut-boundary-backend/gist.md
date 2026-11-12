# A stub `partition` field, and a 266,000x speedup hiding behind it

> **Update, 2026-11-12:** the main branch independently fixed the same
> stub bug this article describes, more rigorously than the fix below —
> and re-testing against that better fix found the "fast" half of this
> story doesn't hold up where it matters. The short version: the stub was
> real and worth fixing, but the fixed algorithm still can't reliably find
> the bridge on realistic, skewed-weight graphs — it reports a cut tens of
> thousands of times too large and misses the bridge in every one of 50
> trials. The speedup is real; the correctness it was supposed to come
> with is not. Full account in
> [ADR-357](https://github.com/ruvnet/ruvector/blob/main/docs/adr/ADR-357-approx-mincut-partition-fix-and-boundary-backend.md)
> and the nightly README's 2026-11-12 update. The bug-finding half of this
> article (below) still stands; its ending does not — see the corrected
> "What's next" at the bottom.

## Problem

`ruvector-mincut` is a Rust dynamic minimum-cut library with an exact
solver and an approximate one (`ApproxMinCut`, based on "Approximate
Min-Cut in All Cut Sizes," SODA 2025, arXiv:2412.15069). A downstream
crate, `ruvector-agent-memory`, wanted to use minimum cut as a structural
signal for agent-memory compaction: when deciding what to forget, don't
just score memories by recency/frequency/coherence — also check whether a
memory is a graph "bridge," the only link between two otherwise-disjoint
topic clusters, and protect it even if its scalar score is low.

A previous experiment with this idea (`MincutGatedForgetting`) was
rejected for production use for one simple reason: the only available
backend, an exact global min-cut solver, took 76ms to 11.4 seconds *per
call* on graphs of only 50-400 vertices. For a memory store doing
continuous compaction, that is not a background job, it's a wall.

The obvious next move — the one a previous research pass explicitly
flagged as unresolved — was: does the crate's own approximate algorithm
avoid this cost? It should, on paper: O(m log² n / ε²) preprocessing
instead of an exact global search.

## What we actually found

`ApproxMinCut` had 9 passing unit tests. All 9 checked `.value`,
`.lower_bound`, `.upper_bound`, or `.epsilon`. None checked `.partition`.

Here's why that mattered:

```rust
/// Compute partition achieving the approximate min-cut
fn compute_partition(&self, _cut_value: f64) -> Option<(Vec<VertexId>, Vec<VertexId>)> {
    // For now, return a simple partition based on BFS from first vertex
    ...
}
```

The `_cut_value` parameter — the actual computed minimum cut — is
unused. The function ignores it and returns an arbitrary bisection: walk
the graph breadth-first from whichever vertex happens to come first, stop
once you've visited half the vertices, call that side A and the rest side
B. This has no relationship to which edges the min-cut algorithm actually
selected as the cheapest cut. For any caller treating `partition` as "the
two sides of the minimum cut" — which is the only reasonable reading of a
field with that name, in a struct called `ApproxMinCutResult` — the
answer would be silently, structurally wrong on every call past a small
early-exit threshold.

This is a specific, nameable kind of bug: a placeholder that type-checks,
looks plausible in isolation, and would pass casual review, because
nothing *locally* looks broken. `self.vertices.len() <= 1 { return None }`
reads like a sensible guard clause. `visited.len() < target` reads like a
sensible stopping condition. The bug only shows up when you ask "does this
side actually correspond to a cheap cut," and nothing in the existing test
suite asked that question.

## The fix

Standard Stoer-Wagner minimum-cut computation already does almost all the
work needed to recover a partition — it just wasn't being asked to keep
it. Each phase of the algorithm contracts two vertices together and
tracks a "cut of the phase" value; the global minimum is the best value
seen across all phases. The partition achieving that value is exactly
"the supervertex that was about to be contracted" versus "everything
else" — you just have to remember, for each contraction, which original
vertices got merged into which supervertex, so you can unmerge the
winning phase's grouping back into real vertex IDs at the end.

```rust
let mut group_members: Vec<Vec<usize>> = (0..n).map(|i| vec![i]).collect();
// ...during contraction:
group_members[before_last].extend(std::mem::take(&mut group_members[last]));
// ...when a phase sets a new minimum:
best_group = group_members[last].clone();
```

That's the whole fix, conceptually. The rest is bookkeeping: threading it
through the small-graph exact path, the sparsifier path (recovering the
partition from whichever sparsifier sample produced the median cut value
— not from the full graph, which would defeat the point of
sparsification), the disconnected-graph case, and fixing a second, smaller
bug the same audit turned up: the cached-value fast path on repeated
`min_cut()` calls was unconditionally returning `partition: None`, even
when a partition had already been computed and cached alongside the
value.

## Proving it

The regression test built specifically to catch the original bug:

```rust
#[test]
fn partition_isolates_the_weak_bridge() {
    // Two 6-cliques (dense internal connectivity) joined by exactly one
    // weak edge. The true minimum cut is unambiguous: cut the bridge.
    let mut edges = Vec::new();
    for clique_base in [0u64, 100] {
        for i in 0..6u64 {
            for j in (i + 1)..6u64 {
                edges.push((clique_base + i, clique_base + j, 1.0));
            }
        }
    }
    edges.push((0, 100, 0.1)); // the weak bridge

    // ... insert edges, call min_cut() ...

    // The two cliques must land on opposite sides, because the only
    // minimum cut in this graph IS the bridge.
    assert_ne!(side_a_set.contains(&0), side_a_set.contains(&100));
}
```

An arbitrary BFS-order bisection has no mechanism to reliably produce this
split — it depends only on vertex ordering, not on where the graph is
actually weak. The fixed version passes this test because it is now
actually tracking the cut, not guessing at a bisection.

## Does it solve the original problem?

Yes, measurably. With the fix in place, we wired `ApproxMinCut` in as a
selectable backend for the original `MincutGatedForgetting` policy
(`MincutBackend::Approx { epsilon }`, alongside the existing
`MincutBackend::Exact`) and benchmarked both on the same topologies the
original rejected experiment used:

| n (vertices) | Exact    | Approx (ε=0.1) | Speedup      |
|-------------:|---------:|----------------:|-------------:|
| 19 (ring)    | 91,961ms | 0.35ms          | ~266,000x    |
| 50           | 139ms    | 2.6ms           | ~54x         |
| 100          | 679ms    | 10.5ms          | ~65x         |
| 200          | 3,851ms  | 57.6ms          | ~67x         |
| 400          | 14,804ms | 396ms           | ~37x         |

On the correctness side — the two-clique-plus-bridge topology, 50 repeated
calls per backend — both the exact and the fixed approximate backend flag
the bridge vertex as structurally load-bearing in 100% of trials, with 0%
empty or degenerate results for either.

(The n=19 outlier is itself a small finding: a regular, symmetric ring
graph is apparently close to a worst case for the exact solver's
underlying search, while a more realistic two-dense-clusters shape at the
same vertex count costs about 1.1 seconds, not 92. If you're budgeting
for an exact min-cut call, vertex count alone will mislead you — the
graph's shape matters more.)

## The part that's easy to miss

The original research question was "is the approximate algorithm fast
enough?" The answer turned out to be gated behind a different, more basic
question nobody had asked: "is the approximate algorithm's output
*correct*?" A fast wrong answer isn't a faster algorithm, it's a
different bug. The 9 pre-existing tests all independently verified the
*value* `ApproxMinCut` reports is a reasonable (1+ε)-approximation — and
they're right, the value computation was fine the whole time. It was
specifically the *partition* — a second output field riding alongside a
correct one — that had never been checked against anything, because
nothing in the test suite had ever needed it for anything until a
downstream crate came looking for a structural signal.

If your library returns two things from one function, and your tests only
exercise one of them, that's worth a second look — especially when the
untested one is the one a doc comment promises but a comment three lines
into its implementation quietly admits isn't really being computed yet
("For now, return a simple partition...").

## What's next (corrected, 2026-11-12)

The original ending here said the fix unblocked a fair re-trial of
`MincutGatedForgetting` at full scale, pending only a bigger benchmark
run. It doesn't. Main independently fixed the same stub bug in a more
careful way — always re-weighing the candidate partition on the real
graph instead of the sampler's own distorted weights — and that fix is
a genuine improvement on its own terms. But re-running this article's own
benchmark against it, the "fast and correct" story collapses on exactly
the graph shape this whole investigation cared about: two dense clusters
joined by one weak bridge, which is what a real embedding's nearest-neighbor
graph actually looks like. The algorithm samples edges weighted toward the
*heaviest* ones; the bridge is deliberately the lightest edge in the
graph, so it is one of the least likely edges to survive sampling, and no
amount of re-weighing candidates you never generated can recover a cut
you never built. Asking for more samples barely helps: 3 samples misses
by about 40,000x, 50 samples still misses by about 19,000x.

So the actual next step isn't "run the bigger benchmark." It's: either
find a sampling strategy that doesn't systematically discard the lightest
edges in a skewed-weight graph, or accept that this is the wrong tool for
this particular job and use the exact solver this repository already has
(slower, but it finds the bridge every time). That's still a well-defined
research result — a falsified hypothesis with a clear, specific reason it
failed is not a worse outcome than an accepted one, just a different one.

---

*Code: `ruvector-mincut` (`src/algorithm/approximate.rs`),
`ruvector-agent-memory` (`src/graph_forget.rs`,
`examples/mincut_backend_bench.rs`). Full research writeup, raw benchmark
output, and ADR:
`docs/research/nightly/2026-10-29-approx-mincut-boundary-backend/` and
`docs/adr/ADR-357-approx-mincut-partition-fix-and-boundary-backend.md` in
the [ruvector](https://github.com/ruvnet/ruvector) repository.*
