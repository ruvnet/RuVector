# A stub `partition` field, and a 266,000x speedup hiding behind it

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

## What's next

This fix unblocks, but does not itself complete, the original goal: the
2026-09-05 nightly's bridge-retention/Recall@10 benchmark — rejected
purely on the `Exact` backend's latency — needs to be re-run at its
originally-intended corpus scale with the now-fast, now-correct `Approx`
backend before `MincutGatedForgetting` can be called production-ready.
That's a well-defined next step, not an open-ended one, which is exactly
the kind of research result worth having: not "we made something faster,"
but "we found and fixed the specific thing standing between a rejected
experiment and a fair re-trial."

---

*Code: `ruvector-mincut` (`src/algorithm/approximate.rs`),
`ruvector-agent-memory` (`src/graph_forget.rs`,
`examples/mincut_backend_bench.rs`). Full research writeup, raw benchmark
output, and ADR:
`docs/research/nightly/2026-10-29-approx-mincut-boundary-backend/` and
`docs/adr/ADR-355-approx-mincut-partition-fix-and-boundary-backend.md` in
the [ruvector](https://github.com/ruvnet/ruvector) repository.*
