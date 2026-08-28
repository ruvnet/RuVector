# When a "Local" Algorithm Isn't Local: A Negative Result on HNSW Self-Healing

## Problem

`ruvector-hnsw-repair` gives RuVector three ways to handle HNSW deletions:
tombstone-only (cheap, recall degrades), batch repair (amortised), and
eager repair (expensive full-graph scan per delete, best recall). All
three apply the same policy to every deleted node — a node sitting in a
dense, well-connected part of the graph gets the same expensive treatment
as one at a structural bottleneck.

RuVector separately ships `ruvector-mincut`, a large (55K-line) crate of
dynamic graph algorithms. One of them, `LocalKCut`, implements a real 2024
result (arXiv:2510.08297) for finding a small cut near a single vertex in
time bounded by that vertex's degree — not the whole graph. Its own module
docs name "self-healing networks" as the target use case. Composing the
two looked like an obvious win: ask `LocalKCut` whether a deleted node's
removal actually threatens connectivity, and only pay for eager repair
when it does.

## Hypothesis

On a 5,000-vector HNSW graph (dim 64) with 20% of nodes deleted, a
`LocalCutGuidedRepair` strategy — eager repair only for nodes where
`LocalKCut::find_cut` reports a local cut of size <= 2, tombstone
otherwise — should land within 1 percentage point of eager repair's
recall, at no more than 60% of its total repair cost, with connectivity
bookkeeping overhead under 25% of its own delete time.

## What Actually Happened

Recall landed identically for both strategies (0.9140), and
`LocalCutGuidedRepair` did fewer repairs than `EagerRepair` (0 vs. 176
edges) — on paper, two out of three criteria "passed." But at the sample
size the run could actually complete (3 deletions out of 5,000 vectors),
that's not a meaningful validation: all three deleted nodes were judged
"safe" by the cut-finder, so the strategy did nothing differently from
plain tombstoning, and 100 queries isn't enough to detect a recall
difference this small either way. The criterion that *is* meaningful at
any sample size — how much time the connectivity check itself costs,
relative to a 25% budget — failed by four orders of magnitude:
bookkeeping was **100.0%** of `LocalCutGuidedRepair`'s delete wall-clock
time. Three individual `find_cut` calls took 158.0s, 135.2s, and 24.8s.
An earlier attempt at the originally-planned 1,000 deletions (20%) never
finished in 10+ minutes; an intermediate attempt got through exactly one
call — 163.5 seconds — before a 200-second cap killed it. Three separate
runs, three consistent orders of magnitude.

A standalone diagnostic — five isolated `find_cut` calls on a *sparser*
test graph, no deletion involved — shows the same operation costs
222-418ms there, not tens of seconds:

```text
find_cut(0) = false in 387.9ms
find_cut(1) = false in 222.0ms
find_cut(2) = false in 418.1ms
find_cut(3) = false in 250.1ms
find_cut(4) = false in 257.2ms
```

Two things compound to cause the difference. First: `LocalKCut`'s
complexity bound, `O(k^{O(1)} · deg(v))`, is real and correctly
implemented for the setting its source paper targets — a bounded-degree,
low-expansion graph. HNSW is the opposite by design: a small-world graph
engineered so a handful of hops reach a large fraction of the index
(that's what makes HNSW fast). A depth-2 BFS isn't actually local on a
graph like that. Second, and more concretely: `LocalKCut::check_cut`
(`ruvector-mincut/src/localkcut/mod.rs:368`) resolves each crossing edge
by calling `graph.edges()` — which materializes a fresh list of *every*
edge in the graph — and linearly scanning it, instead of the O(1)
`get_edge(u, v)` lookup the graph already provides. That's a genuine,
fixable implementation defect, and it's why the cost scales with total
graph size rather than staying bounded by `k` and local degree the way
the complexity claim promises.

## Why This Is Still a Useful Result

This is exactly the kind of thing a nightly research process should catch
and record. `ruvector-mincut` is a genuinely sophisticated crate — several
of its algorithms are real, recent, correctly implemented results. But a
component being well-implemented in isolation doesn't mean it composes
cheaply with a different subsystem's topology. Recording *why* this
specific composition fails — small-world expansion, not a bug, not a
missing optimization — means the next engineer who has the same "let's use
mincut for self-healing" instinct doesn't have to re-run the same
five-minute benchmark to find out.

The negative result also narrows the search: a real self-healing-index
signal for HNSW would need to be based on something that stays cheap on a
small-world graph — node degree or local density, not a bounded local-BFS
cut search. That's now a documented open question (see the ADR) instead of
an unexamined assumption.

## Limitations

- One dataset size (n=5,000, dim=64), one `k` (2). The overhead's cause is
  architectural (small-world expansion vs. a bounded-degree assumption)
  and wouldn't improve at larger `n`, and a larger `k` provably makes
  `find_cut` more expensive per `LocalKCut`'s own `compute_radius(k)` — so
  neither was expected to change the conclusion, and neither was
  separately re-measured to confirm that expectation empirically.
- `LocalKCut` is not shown to be broken — only mismatched to HNSW's
  topology. Its cost profile on graphs closer to what its source paper
  targets (bounded-degree, expander-like) is a different, unanswered
  question.

## Production Relevance

None recommended for this specific design — see the ADR's rejection
criteria. The validated pieces that survive: `ruvector-hnsw-repair`'s
`repair_one` is now `pub`, reusable by any future repair-strategy
experiment without re-implementation; and the benchmark harness itself
(same dataset generator, same size, as the existing crate's own benchmark)
is a template for the next composability experiment.

## RuVector Ecosystem Implications

Three capabilities met directly: HNSW deletion repair (`ruvector-hnsw-repair`),
dynamic min-cut (`ruvector-mincut`), and self-healing indexes (an explicit
RuVector research-map item). No MCP, RVF, or RVM surface is recommended —
a rejected design has none to expose.

## Future Direction

A degree- or density-threshold fragility signal, benchmarked the same way,
to see whether a cheap heuristic can approximate what the expensive exact
local cut was trying to buy — without paying for a bounded BFS that isn't
actually bounded on this topology.

## References

- "Deterministic and Exact Fully-dynamic Minimum Cut of
  Superpolylogarithmic Size" (arXiv:2510.08297, Dec 2024) — the algorithm
  `LocalKCut` implements.
- Malkov & Yashunin, HNSW (the algorithm `ruvector-hnsw-repair`'s
  `HnswGraph` implements).
- Full methodology and raw benchmark output: `docs/research/nightly/2026-08-28-mincut-guided-hnsw-repair/README.md`
  in [ruvnet/ruvector](https://github.com/ruvnet/ruvector).
