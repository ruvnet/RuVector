# Fixing a non-deterministic min-cut engine, and why the fix still wasn't enough

## Problem

`ruvector-mincut`, a from-scratch dynamic minimum-cut engine in the
RuVector Rust workspace, has a public "ask this graph's min-cut" entry
point: `RuVectorGraphAnalyzer::partition()`. A prior nightly experiment
(2026-09-05, ADR-345) tried to use it to give agent-memory compaction a
structural "don't evict the sole bridge between two topic clusters" signal,
on top of an existing scalar recency/frequency/coherence policy. It got
rejected on measured grounds: `partition()` was up to ~2,700x slower than
the scalar baseline on an 84-entry test corpus, and — filed as an
unresolved side finding — non-deterministic: on a fixed, byte-identical
19-vertex graph, repeated calls returned an unusable empty result roughly
half the time.

This write-up is the direct follow-up: root-cause the non-determinism, fix
it, and see whether the fix also solves the latency problem enough to
revisit the original (rejected) application.

## Root cause

`partition()` delegates to `MinCutWrapper`, which implements a bounded-range
dynamic-min-cut algorithm (the wrapper scheme from arXiv:2512.13105): up to
100 geometrically-scaled sub-instances, each of which — the first time it's
touched — replays the graph's *entire* current edge set to bootstrap itself.
That design amortizes well if you build one graph and mutate it many times.
Every actual call site in the workspace does the opposite: build a fresh
graph, ask once, throw it away. So every call pays full bootstrap cost with
nothing amortized.

Separately: `DynamicGraph` (the crate's core graph type) stores its edges in
`dashmap::DashMap`s. Each `DashMap::new()` gets an independently-randomized
hash seed, so two structurally-identical `DynamicGraph`s — as you get from
calling `from_knn(...)` twice on the same neighbor list — iterate their
edges in different orders. `MinCutWrapper` iterates `graph.edges()` directly
while bootstrapping each sub-instance, with no sort in between. That's
sufficient to explain the observed non-determinism without any intentional
randomness anywhere in the chain — it's an emergent property of routing a
hash-map-order-sensitive bootstrap through a type whose iteration order
isn't fixed.

## What we built

A second, separate code path: `ruvector_mincut::static_cut::stoer_wagner_min_cut`,
the classical Stoer-Wagner global min-cut algorithm (1997), O(V³), over a
dense weight matrix built once from the graph's edges — **sorted by
canonical endpoint and edge id before the algorithm ever runs**. Every
tie-break inside the algorithm resolves deterministically (lowest surviving
vertex index wins), so the whole thing's output depends only on graph
structure, never on any hash map's iteration order.

```rust
pub fn stoer_wagner_min_cut(graph: &DynamicGraph) -> Option<StaticCutResult> {
    let mut vertex_ids = graph.vertices();
    vertex_ids.sort_unstable();               // fixed order, not hash order
    // ... build dense weight matrix from edges sorted by (endpoints, id) ...
    // ... classical Stoer-Wagner min-cut-phase loop ...
}
```

Wired it into `RuVectorGraphAnalyzer` as `partition_static()`/`min_cut_static()`
(alongside, not replacing, the original dynamic `partition()`), and into the
agent-memory compaction policy as a new `MincutEngine::Static`, then re-ran
the *exact* prior benchmark, scaling probe, and determinism probe with both
engines side by side.

## Results

**Determinism: fixed, unambiguously.**

```
[Dynamic engine: partition()]
trials=50  avg_per_call=835.0ms   empty_or_degenerate=33 (66%)

[Static engine: partition_static()]
trials=50  avg_per_call=0.099ms   empty_or_degenerate=0 (0%)   distinct_partitions_seen=1
```

Fifty independently-rebuilt, structurally-identical graphs, one distinct
result. And the speedup on this exact fixture — a small (19-vertex) graph
where the dynamic engine happened to hit a pathological case — was **8,421x**.
The scaling probe found the same pathology again independently: on a
regular ring topology, the dynamic engine took **66.7 seconds** at n=19.

```
n=19     partition=66745.030ms   partition_static=0.069ms   (972,293x)
n=400    partition=10864.023ms   partition_static=30.030ms  (361.8x)
n=800    partition=skipped       partition_static=240.290ms
```

**Speed on the realistic benchmark: dramatically better, but not enough.**

We pre-registered a speed bar *before* running this: static engine
compaction should cost no more than 10x the scalar baseline (tighter than
the prior nightly's 100x "background job" allowance, because this
experiment's whole premise was "fast enough to be a foreground path now").

```
MincutGatedForgetting-Soft (Dynamic)   93455us   (3222.6x baseline)
MincutGatedForgetting-Soft (Static)     1270us     (43.8x baseline)
```

73.6x faster than the old engine. Still 4.4x over the 10x bar we set. O(V³),
even fully deterministic and two orders of magnitude faster in practice than
the alternative, is still asymptotically far more expensive than the O(n log n)
scalar sort it's competing against.

**Effectiveness: still flat — now confirmed twice, independently.**

The original hypothesis (structural bonus improves bridge-memory survival by
≥15 percentage points over the scalar baseline) failed again:

```
Bridge-survival gap over baseline, Dynamic engine: +0.0pp
Bridge-survival gap over baseline, Static engine:  +0.0pp
```

Same corpus, same result, on two independently-implemented min-cut engines.
That's stronger evidence than either engine alone could give: it's now much
more likely this is a property of the 84-memory corpus / dataset design than
a bug in either implementation.

## Verdict

Two pre-registered thresholds still fail (speed, effectiveness), so per the
hypothesis fixed before this run: **REJECT**, for the compaction-policy
application, a second time. That's not a wasted run — it's a materially
stronger negative result than before, because it now rules out "maybe it was
just this one buggy engine call" as an explanation.

But `static_cut` itself — correct (validated against known min-cut
properties on triangle/weighted/disconnected/bridge-topology fixtures),
deterministic (0/50 divergent trials, vs. 66% for the alternative), and
65x–8,421x faster depending on topology — is unambiguously useful
independent of that one failed application. It ships as new public API in
`ruvector-mincut`: any one-shot ("rebuild the graph, ask once") min-cut
query anywhere in the workspace can use it directly instead of paying the
dynamic engine's bootstrap cost for nothing.

## Limitations

Single run per benchmark (no averaged trials) within this nightly's
practical wall-clock budget — the dynamic engine's own numbers moved
noticeably between this run and the original ADR-345 run (1,800–2,700x
originally, 3,031–3,222x here), which is itself indirect confirmation of the
non-determinism finding rather than a discrepancy to explain away. Scaling
was only measured up to 800 vertices. No Darwin-style automated evolutionary
search was available in this environment (no resolvable `ruvector harness
darwin` CLI); the two variants compared here were defined and benchmarked
directly.

## Production relevance

`static_cut` is mergeable and useful today for any workspace code doing
one-shot structural graph queries — which, per this ADR's own audit, is
every current call site of `RuVectorGraphAnalyzer` in the repository. The
compaction application stays experimental and off by default, with the
bridge-survival question now more precisely scoped for whoever picks it up
next: not "is the engine broken" (no, fixed) but "does a min-cut-derived
signal help at any corpus size, or is a cheaper structural feature
(articulation points, local conductance) the better bet."

## RuVector ecosystem implications

A deterministic structural primitive is a prerequisite, not a nice-to-have,
for anything proof- or witness-gated (RVM coherence domains, proof-gated
writes): a non-deterministic gating signal means two honest observers can
legitimately disagree about the same query, which is disqualifying for
anything that needs to be independently verified. This work doesn't build
that integration, but it removes a concrete blocker for a future one.

## Future direction

1. Re-run the bridge-survival experiment at 10-25x this corpus's size — the
   original nightly wanted this and was blocked purely by the dynamic
   engine's latency; that blocker is now much weaker.
2. Try a structural signal that's cheaper than a global min cut in the first
   place — articulation points (O(V+E), one DFS) are a natural next
   candidate for clearing the 10x speed bar this run missed.
3. Point `ruvector-mincut`'s own `CommunityDetector`/`GraphPartitioner` — both
   still on the slow, non-deterministic path — at `partition_static()`, as
   its own scoped follow-up with a dedicated before/after benchmark.

## References

- Stoer, M. and Wagner, F. (1997). *A Simple Min-Cut Algorithm*. Journal of
  the ACM, 44(4), 585-591.
- `docs/research/nightly/2026-09-05-mincut-gated-forgetting/` and ADR-345 —
  this experiment's direct predecessor.
- Full methodology, ADR, and raw benchmark output:
  `docs/research/nightly/2026-09-08-static-mincut-forgetting/README.md`,
  `docs/adr/ADR-346-static-mincut-fast-path.md`.
