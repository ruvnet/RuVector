# Why your dynamic min-cut library returns an empty set on a connected graph

## Problem

A nightly research run in the RuVector workspace
(`docs/research/nightly/2026-09-05-mincut-gated-forgetting`) tried to use an
existing dynamic minimum-cut engine, `ruvector-mincut`, to give an agent
memory system a structural "don't evict this, it's the only bridge between
two topic clusters" signal. The experiment was rejected, partly because the
library's `partition()` call — "give me the two sides of a minimum cut" —
returned an empty or degenerate result in about half of repeated calls
against the exact same graph. The write-up correctly identified the symptom
and correctly guessed at a plausible mechanism ("probably hash-map
iteration-order tie-breaking somewhere"), but didn't locate it.

This is a walkthrough of finding and fixing it, because the failure mode —
"a deterministic input to a supposedly-deterministic algorithm produces a
different, sometimes wrong, answer on every run" — is common enough in
graph/index libraries built on `HashMap`/`HashSet`/concurrent hash maps that
the diagnostic path is worth writing down.

## Hypothesis

Formalized before touching any code, so the fix couldn't be shaped by
whatever was found along the way:

```text
Given RuVectorGraphAnalyzer::partition() called on a fresh analyzer built
from a fixed, byte-identical connected k-NN graph,

when the graph's read-path ordering and the algorithm's internal
tie-breaking are made deterministic, and the partition-complement
calculation is fixed to use the graph's real vertex set,

then partition() returns a non-empty, non-degenerate, byte-identical result
across repeated calls, with no public API changes and the full existing
test suite still green.
```

## Technical Design: Finding Bug #1

The library's graph type stores vertices and edges in `DashMap` — a
concurrent hash map. Its public accessors looked like this:

```rust
pub fn vertices(&self) -> Vec<VertexId> {
    self.adjacency.iter().map(|entry| *entry.key()).collect()
}
```

`DashMap`, like `std::collections::HashMap`, uses a randomly-seeded hasher
per instance by default. That means: build the *same* graph, with the *same*
edges inserted in the *same* order, in two different process runs (or even
twice in the same run, since a new `DashMap` gets a fresh random seed each
time) — and `.iter()` can, and does, come back in a different order.

Three call sites downstream cared about that order in a way that changed
the *answer*, not just internal bookkeeping:

```rust
// BoundedInstance::brute_force_min_cut — used for small graphs (<20 vertices)
let vertex_vec: Vec<_> = self.vertices.iter().copied().collect();
// ... bitmask `mask` enumerates subsets by index into vertex_vec ...
if boundary < min_cut {   // strict <, so ties keep the FIRST subset found
    min_cut = boundary;
    best_set = subset;
}
```

If the graph has multiple equally-good minimum cuts (common with symmetric
or near-symmetric cluster data — exactly the shape of the reproduction
graph, two near-identical clusters joined by one bridge point), which
subset counts as "first" depends on which vertex got which bit position,
which depends on `vertex_vec`'s order, which depends on a randomly-seeded
hash map's iteration order. Two runs, same input, two different valid
answers.

**Fix**: sort the vector before using it as an ordering. One line, applied
at the two call sites that build these vectors from `HashSet` iteration,
plus the `DynamicGraph` accessors themselves (fixing it once at the read
boundary is cheaper than auditing every downstream consumer).

## Technical Design: Finding Bug #2 (the one that actually mattered more)

Sorting fixed the *which subset wins ties* question — verified by direct
instrumentation, the algorithm now returned the *same* 9-vertex subset on
every call. But the reported partition was still degenerate: `(9, 0)`
instead of `(9, 10)`. Fixing bug #1 in isolation made the *measured* rate of
bad results go from ~50% to 100% — worse, by the simple metric the original
probe measured, before bug #2 was found. That is worth sitting with for a
second: **a real, correct fix, measured with the same instrument, produced
a worse-looking number** — because the instrument's single metric couldn't
distinguish "sometimes wrong due to non-determinism" from "consistently
wrong due to a separate, deterministic bug that non-determinism had been
statistically masking."

The second bug was in how the library turns its internal, compact witness
representation (a seed vertex plus a bitmap of "which vertices are on this
side") back into the two explicit sides:

```rust
pub fn materialize_partition(&self) -> (HashSet<VertexId>, HashSet<VertexId>) {
    let u: HashSet<VertexId> = self.inner.membership.iter().map(|v| v as u64).collect();
    let max_vertex = self.inner.membership.max().unwrap_or(0) as u64;  // <-- bug
    let v_minus_u: HashSet<VertexId> = (0..=max_vertex)
        .filter(|&v| !self.inner.membership.contains(v as u32))
        .collect();
    (u, v_minus_u)
}
```

`max_vertex` is computed from the membership set `U`'s *own* maximum
element — not the graph's actual vertex count. If `U = {0..=8}` (a 9-vertex
cluster) in a 19-vertex graph whose other 10 vertices are numbered `9..=18`,
this function assumes the graph only has vertices `0..=8`, and "the
complement" comes back empty, because every vertex in that (wrong) range is
already in `U`. This isn't a rare edge case — it's the *default* case for
any cut that doesn't happen to include the graph's highest-numbered vertex,
which is most cuts, most of the time, for any reasonably-sized graph.

**Fix**: the one production call site that turns a witness into an explicit
partition (`RuVectorGraphAnalyzer::partition()`) doesn't need to guess the
graph's size from `U`. It already holds a reference to the actual graph:

```rust
let mut side_a = Vec::new();
let mut side_b = Vec::new();
for v in self.graph.vertices() {          // real vertex list, already sorted (bug #1's fix)
    if witness.contains(v) {              // O(1) bitmap check either way
        side_a.push(v);
    } else {
        side_b.push(v);
    }
}
```

Same asymptotic cost (`O(|V|)`), correct for any `U`. The buggy method
itself was left in place — two existing tests call it directly, and fixing
its signature to take the graph's real size is a breaking API change that
eighteen dependent crates would need auditing against, which is out of
scope for a same-night, non-breaking fix. Its doc comment now says exactly
what it will get wrong and points callers at the pattern above instead.

## Actual Benchmark Evidence

Reused, unmodified, the exact reproduction script the original nightly run
had already written
(`crates/ruvector-agent-memory/examples/mincut_determinism_probe.rs`) — the
only way to get an honest before/after number is to not touch the
instrument between runs.

```text
# Documented prior measurement (2026-09-05, same script):
# 15/30 (50%) empty/degenerate

# This run, both fixes applied, two independent 30-trial runs:
trials=30 elapsed=33.55s avg_per_call=1118.3ms empty_or_degenerate=0 (0%) bridge_detected_as_boundary=30 (100%)
trials=30 elapsed=33.88s avg_per_call=1129.5ms empty_or_degenerate=0 (0%) bridge_detected_as_boundary=30 (100%)
```

Plus a new regression test asserting a strictly stronger property —
byte-identical results across repeated calls, not merely "non-empty":

```text
running 3 tests
test graph_vertices_and_edges_are_sorted ... ok
test partition_is_never_degenerate_on_connected_graph ... ok
test partition_is_stable_across_repeated_calls ... ok

test result: ok. 3 passed; 0 failed; 0 ignored; 0 measured; 0 filtered out; finished in 34.17s
```

Per-call latency (~1.1s) is unchanged from the prior measurement (~841ms) —
the added sorting is microseconds on a 19-vertex graph; the difference is
environment noise, not a regression, and is reported as-is rather than
adjusted. This fix is about correctness, not speed — the library's
separately-documented latency scaling problem (77ms at 50 vertices to
11.4s at 400) is untouched.

## Limitations

- Only the small-graph (`<20`-vertex) brute-force code path was verified
  end-to-end with the reproduction script; a structurally similar
  seed-ordering fix was applied to the library's other internal search path
  (used for `>=20`-vertex graphs) by the same reasoning, but not
  independently re-benchmarked at that size this session.
- No concurrent-mutation stress test — correctness is established for
  single-threaded, from-scratch graph construction only.
- WASM and Node bindings that depend on this crate were not rebuilt or
  smoke-tested (no signature changes, so no code changes are required on
  their side, but this wasn't independently verified).

## Production Relevance

Three in-tree consumers (a graph partitioner, a community detector, and the
agent-memory integration that originally surfaced this) share this exact
code path, and eighteen crates in the workspace depend on the library
directly, including WASM and Node bindings. All of them get this fix for
free on next rebuild — no migration, no signature changes.

## RuVector Ecosystem Implications

This is a small, surgical fix, but it sits underneath a wider bet: RuVector
treats a general-purpose dynamic min-cut engine as a reusable primitive for
graph-structural signals across agent memory, community detection,
distributed index partitioning, and (longer horizon) proof-gated mutation
boundaries and coherence-domain isolation. Every one of those depends on
"the cut this library finds is a real cut of the real graph" being true
unconditionally, not true-most-of-the-time-on-the-demo-topology. Finding
and fixing this now, before a second downstream system builds on the same
buggy call, is more valuable than the size of the diff suggests.

## Future Direction

1. Extend the regression test to a `>=20`-vertex topology, to exercise the
   library's other internal search path end-to-end (only reasoned about,
   not independently re-benchmarked, this session).
2. Root-cause the separate, still-open latency scaling problem — likely by
   evaluating an existing-but-unwired, worst-case-bounded connectivity
   backend already in the codebase as a replacement for the current one.
3. Revisit whether the buggy `materialize_partition()` helper's signature
   should be fixed properly (a breaking change) in a dedicated follow-up,
   now that its failure mode is documented precisely rather than guessed at.
4. Re-attempt the originally-rejected agent-memory integration once the
   latency problem is also addressed — this fix removes one of its two
   rejection reasons, not both.

## References

- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/README.md` —
  the run that first measured this symptom.
- `docs/adr/ADR-345-mincut-gated-forgetting.md` — the corresponding ADR,
  rejected partly on this basis.
- `docs/adr/ADR-346-deterministic-mincut-witness-partition.md` — this fix's
  ADR.
- `crates/ruvector-mincut/tests/determinism_tests.rs` — the new regression
  test.
- arXiv:2512.13105 — the December 2024 bounded-range dynamic minimum-cut
  paper `ruvector-mincut`'s wrapper algorithm implements (unaffected by
  this fix; the bug was in this crate's own read/materialization boundary,
  not in the paper's algorithm).
