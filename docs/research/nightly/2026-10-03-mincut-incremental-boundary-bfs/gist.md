# Fixing an O(radius) tax hiding inside a dynamic minimum-cut search

## Problem

`ruvector-mincut` is a from-scratch Rust implementation of a December-2024
subpolynomial dynamic minimum-cut paper. Its local-search oracle explores
outward from a seed vertex via BFS, checking at each layer whether the
boundary (edges crossing from the explored set to the rest of the graph)
is small enough to count as a cut. A previous benchmark had measured this
getting dramatically slower as graphs grew — 77ms at 50 vertices, 11.4
seconds at 400 — fast enough to make a real consumer (an agent-memory
compaction policy that uses minimum cuts to detect "bridge" memories worth
protecting from eviction) impractical and rejected. Nobody had traced the
slowdown to a specific line of code.

## Hypothesis

If the BFS is recomputing something it could instead update incrementally,
fixing that should produce a measurable, provable-correct speedup without
changing what the algorithm finds — only how long it takes to find it.

## Technical design

Reading the BFS function line by line surfaces the culprit immediately:

```rust
for depth in 0..=radius {
    let boundary_size = self.calculate_boundary(graph, &visited);
    // ...
}
```

`calculate_boundary` walks every vertex in the current explored set and
every one of its edges — an O(edges touching the explored region)
operation — and it runs **once per BFS depth**, for up to `radius` (default
20) depths, per search call. Multiply that by the number of seed vertices
and budget values a caller tries, and by the number of range instances a
higher-level wrapper queries, and a cost that should scale with the final
explored region instead scales with `radius x` that region, repeated many
times over.

The fix is a standard incremental-maintenance trick, already used
elsewhere in the same crate for edge insert/delete caching: track the
boundary edge set across the whole BFS, and when a new layer of vertices
joins the explored set, touch only the edges incident to *those* vertices —
removing an edge if its other endpoint is now also explored (internal),
inserting it otherwise (crosses the cut). Edges untouched by the new layer
are left alone. Over the whole BFS this does the same total work the
from-scratch version did at a *single* depth, not `radius` times.

```rust
fn update_boundary_incremental(
    graph: &DynamicGraph,
    visited: &HashSet<VertexId>,      // already includes new_vertices
    new_vertices: &[VertexId],
    boundary_edges: &mut HashSet<EdgeId>,
) {
    for &v in new_vertices {
        for (neighbor, edge_id) in graph.neighbors(v) {
            if visited.contains(&neighbor) {
                boundary_edges.remove(&edge_id);
            } else {
                boundary_edges.insert(edge_id);
            }
        }
    }
}
```

Correctness hinges on one subtlety: two vertices joining in the *same*
layer, adjacent to each other. Calling this with `visited` already updated
to include the whole new layer (not just the vertex currently being
processed) handles it for free — both directions of the shared edge get
visited (once while processing each endpoint), and `remove` is a safe
no-op if the edge was never counted as boundary to begin with.

## Actual implementation

`crates/ruvector-mincut/src/localkcut/paper_impl.rs`: `deterministic_bfs`
rewritten to call `update_boundary_incremental` once for the initial seeds
and once per expanded layer, instead of calling the old from-scratch
`calculate_boundary` every depth. `calculate_boundary` itself is kept
as-is — one existing test calls it directly, and the new regression test
uses it as the independent correctness oracle to check the incremental
path against. No public API changed.

## Actual benchmark evidence

All runs on one machine, release builds, baseline and candidate measured
back-to-back via `git stash`/`pop` on exactly the one changed file.

**Isolated primitive** (worst case: a query budget chosen so the BFS always
runs its full radius, never short-circuiting):

| n | old | new | speedup |
|---|---|---|---|
| 50 | 2.80ms | 2.17ms | 1.3x |
| 400 | 353.24ms | 115.00ms | 3.1x |

**The exact prior benchmark that measured the original 11.4s**, re-run
unmodified:

| n | baseline | candidate | speedup |
|---|---|---|---|
| 50 | 77.3ms | 68.8ms | 1.12x |
| 400 | 11,420ms | 4,308ms | 2.65x |

**The actual rejected consumer workload** (an 84-memory compaction
benchmark, `--release`), re-run unmodified: compaction slowdown vs. a
non-structural baseline policy dropped from 2,726x/2,913x to
1,515x/1,564x — real, reproducible, and still an order of magnitude over
the 100x acceptance threshold that workload needs.

Correctness: a new test generates 20 random sparse graphs and checks every
cut the fixed code finds against an independent, from-scratch
recomputation of the same quantity — 100+ witnesses checked, zero
mismatches. All pre-existing tests (108 unit tests, 10 integration tests
across two test binaries) pass unchanged, including the determinism
regression suite from a prior fix to the same code path.

## Limitations

This does not make the rejected agent-memory policy viable. Two
independent problems caused its rejection: cuts were too slow to compute,
and the cuts found didn't reliably correspond to the memories a human would
call "bridges." This fixes only the first, partially — real speedup, but
not enough on its own to clear the practicality bar, and the second problem
is completely untouched, as expected from a pure performance fix.

## Production relevance

Shipped with no feature flag, no migration, and no public API change —
every current and future consumer of this code path (community detection,
graph partitioning for distributed indexing, and any future revival of
structural memory eviction) inherits the speedup automatically.

## RuVector ecosystem implications

This is a small, concrete instance of a more general discipline: when a
graph-native substrate's local-search primitives carry an avoidable
per-call tax, every higher-level feature built on them (agent memory,
community detection, future world-model graph traversal) pays it silently
and compounds it. Finding and removing the tax in the shared primitive is
higher-leverage than optimizing any one consumer, and is exactly the kind
of improvement worth re-measuring against a rejected hypothesis's own
unmodified benchmark rather than inventing a new one.

## Future direction

The remaining ~15-26x gap to the rejected policy's practicality threshold
is structural, not a per-call inefficiency: the number of searches
performed (across seed vertices, budget ranges, and wrapper instances)
still scales with graph size. Bounding that count — sampling a small,
representative set of seed vertices instead of trying every boundary
vertex — is the next lever, and is deliberately left as separate future
work rather than bundled into this narrower, provably behavior-preserving
fix.

## References

- ADR-345: Mincut-Gated Forgetting (original rejection, latency and
  bridge-survival findings)
- ADR-346: Deterministic, Non-Degenerate Witness Partitions in
  `ruvector-mincut` (prior fix to the same code path; left latency out of
  scope)
- ADR-352: Incremental Boundary Maintenance in `ruvector-mincut`'s
  `DeterministicLocalKCut` (this run)
- "Deterministic and Exact Fully-dynamic Minimum Cut of Superpolynomial
  Size" (arXiv:2512.13105) — the paper `ruvector-mincut` implements
- "Dynamic Connectivity with Expected Polylogarithmic Worst-Case Update
  Time" (arXiv:2510.08297) — `PolylogConnectivity`, evaluated and found to
  solve a different problem (connectivity, not minimum-cut search)
