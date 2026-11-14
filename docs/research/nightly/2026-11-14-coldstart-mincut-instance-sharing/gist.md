# Why one `min_cut()` call took 64 seconds — and how to make it take 4

## Problem

`ruvector-mincut` implements a bounded-range dynamic minimum-cut wrapper
(following arXiv:2512.13105): to find a graph's global min cut, it tries
a geometric ladder of `[λ_min, λ_max]` ranges, instantiating a fresh
solver for each range until one reports "the cut is in my range." On a
**cold** query — no cached hint from a previous call, which is the only
scenario any benchmark of this code has ever actually measured — that
means visiting every range below the true answer, from scratch, every
time.

Two earlier research passes on this codebase (2026-09-05, 2026-09-11)
measured this path as unacceptably slow (tens of milliseconds to ~11
seconds depending on graph size, with an unexplained ~69-second outlier on
one specific 19-vertex graph) and left the cause unresolved, floating a
specific fix — swap in a different, already-in-tree "polylogarithmic
dynamic connectivity" data structure — as the likely next step.

## The suggested fix doesn't apply

That data structure (`PolylogConnectivity`, implementing arXiv:2510.08297)
answers one question: "are these two vertices connected?" It has no
notion of *how big* the minimum cut is, or *which* vertices sit on either
side of it. The slow code isn't the cheap connectivity check at all — it's
the actual cut-finding solver. No amount of swapping out the connectivity
checker touches that. This is worth stating plainly because it's a useful
negative result on its own: a specific, named, plausible-sounding fix,
checked against the actual code before being built, and ruled out in
minutes instead of after a wasted implementation attempt.

## The real cause

Reading the solver (`BoundedInstance`) directly: for graphs under 20
vertices, it answers every query by exhaustively checking **all `2^n`
non-trivial ways to split the vertex set** and picking the best — 2^19 ≈
524,000 subsets at n=19. This is a true, range-independent property of
the graph: all 2^19 subsets have to be checked once no matter which of the
wrapper's range-instances is asking. But the wrapper doesn't know that —
it builds ~12 separate solver instances before reaching the one whose
range happens to contain the true answer, and **each one independently
re-runs the entire 2^19 search from scratch**, because nothing shares the
result between them.

A parallel, smaller version of the same mistake exists for graphs over 50
vertices: each of those ~12 instances also independently rebuilds its own
copy of a hierarchical clustering structure used only to pick which
vertices to try as search seeds — again, identical input, computed from
scratch, up to a dozen times.

## The fix

Two small, additive, opt-in APIs, wired together behind an existing (but
previously unused for this purpose) extension point the wrapper already
exposed:

- Compute the exhaustive small-graph answer **once**, then hand every
  solver instance in the batch a pre-computed hint instead of letting each
  rediscover it.
- Compute the seed-selection clustering structure **once** per batch,
  sharing one `Arc`-wrapped copy across every instance instead of
  rebuilding it per instance.

Neither change touches the algorithm that decides *which* cut is correct
— only *how many times* a graph-dependent-but-range-independent quantity
gets recomputed for an identical input.

## Results (measured, release build, same hardware)

| Graph size | Before | After | Change |
|---|---|---|---|
| 19 vertices | 64.3 seconds | 4.0 seconds | **16.2x faster** |
| 50 vertices | 57ms | 57ms | no change (neither mechanism applies here) |
| 100 vertices | 303ms | 240ms | 21% faster |
| 200 vertices | 1,180ms | 964ms | 18% faster |
| 400 vertices | 3,856ms | 2,881ms | 25% faster |

Every variant returns the identical min-cut value on every graph size —
the fix changes only how much redundant work gets done, never which
answer comes out.

## A finding we didn't expect

An ablation that *skips* the shared clustering structure entirely (never
builds it, falls back to trying every vertex as a search seed) turned out
to be slightly *faster* than sharing it, at every size where it matters.
On the symmetric, evenly-connected test graph used here, the clustering
structure's "pick good seeds" benefit doesn't outweigh even a single
build's cost — every vertex looks roughly as promising a seed as any
other. That's a property of *this* test graph, not a general law: a graph
with real, lopsided community structure (the actual production scenario —
clustered agent-memory records) should make targeted seeds pay off far
more. We didn't guess at the answer for that case; it's an open, flagged
next step.

## What this doesn't fix

The small-graph path is still exponential — this removes ~12x redundant
*repetition* of a `2^19` search, not the `2^19` itself. A graph that grows
past roughly 20-25 vertices while still routing through this code path
would need the cutoff itself revisited, which is explicitly out of scope
here. And a previously-rejected downstream feature
(`ruvector-agent-memory`'s structural eviction policy) that cited this
exact latency problem as one of two independent reasons for its rejection
is not un-rejected by this change alone — the second reason (the
structural signal not reliably matching labeled ground truth on noisy
data) is untouched, and revisiting that promotion decision needs its own
fresh benchmark run.

## Why this belongs in RuVector

`ruvector-mincut`'s dynamic min-cut engine is a shared primitive behind
graph partitioning, community detection, and (experimentally) agent-memory
structural eviction — the kind of cross-cutting substrate RuVector is
meant to be. A latency cliff in the primitive is a latency cliff in every
consumer built on it; fixing it once, generically, at the primitive layer
is more valuable than any one consumer working around it locally — which
is exactly what the previously-rejected `MincutGatedForgetting` module
had already started doing (a local `mincut_trials` retry workaround for a
*different* bug in the same dependency) before this run instead closed
the gap upstream.

## References

- `docs/adr/ADR-361-coldstart-mincut-instance-sharing.md`
- `docs/research/nightly/2026-11-14-coldstart-mincut-instance-sharing/README.md` (full methodology and raw data)
- `docs/adr/ADR-345-mincut-gated-forgetting.md`, `docs/adr/ADR-346-deterministic-mincut-witness-partition.md` (prior work)
