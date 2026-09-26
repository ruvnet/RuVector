# Fixing a rejected min-cut agent-memory policy by swapping the min-cut engine, not the policy

## Problem

A prior experiment in the RuVector project (ADR-345) tried adding a
structural signal to agent-memory "forgetting" (compaction/eviction): use a
graph min-cut over a memory store's k-NN similarity graph to detect
"bridge" memories — the sole semantic link between two otherwise-disjoint
topic clusters — and protect them from eviction even when a purely scalar
recency/frequency score would drop them first.

It was rejected. Two of the three mandatory acceptance gates failed:

- Repeated min-cut computations on the *same, unchanged* graph returned
  different partitions — up to 50% of calls returned a degenerate empty
  result.
- Compaction using the signal was 1,800-2,700x slower than the scalar
  baseline, against a 100x "this is a background job, not a fast path"
  budget.

The natural question the original write-up left open: is this a property
of *min-cut as an idea*, or a bug in the specific min-cut engine used?

## Hypothesis

The codebase already ships a second, unused min-cut engine —
`ruvector-mincut`'s `canonical` feature, an implementation of
"pseudo-deterministic minimum cut" (Kenneth-Mordoch, 2026) that returns a
*canonical* cut chosen by an explicit, fixed tie-breaking rule instead of
whatever order the underlying hash-based graph storage happens to iterate
in. If the original failures were caused by the first engine's unordered
`DashMap`/`HashSet`-backed graph rather than by the min-cut approach
itself, swapping engines — no change to the policy logic, dataset, or
acceptance thresholds — should fix both gates.

## What was measured

Same 84-entry synthetic corpus, same random seed, same acceptance
thresholds as the original experiment, run back-to-back on the same
machine with only the min-cut backend swapped:

| Gate | Threshold | Old engine | New engine |
|---|---|---|---|
| Partition determinism (30 calls, identical input) | 100% identical | 40% agreement, 60% degenerate | 100% identical |
| Compaction slowdown vs. scalar baseline | ≤ 100x | 2003x / 1990x — fails | 67x / 62x — passes |
| Recall@10 delta | ≤ 2pp | 0.00pp — passes | 0.00pp — passes |
| Bridge-survival gap vs. baseline | ≥ 15pp | +0.0pp — fails | +0.0pp — fails |
| Tamper detection (eviction witness chain) | 100%/20 | 20/20 | 20/20 |

A separate scaling probe (ring k-NN graphs, 19 to 400 vertices) shows the
same pattern in isolation: the new engine is 110-200x faster end-to-end at
every size from 50 to 400 vertices, and the old engine's single-call
latency at just 400 vertices was over 14 seconds against roughly 100
milliseconds for the new one.

## What this does and does not show

It confirms the hypothesis on two of three axes: the non-determinism and
the extreme slowdown were properties of the *engine*, not of "min-cut as an
eviction signal" in general. Root-cause inspection backs this up — the old
engine's graph storage is a `DashMap<VertexId, HashSet<(VertexId, EdgeId)>>`
with no fixed visitation order, while the new engine explicitly sorts
vertices and applies a lexicographic tie-break before returning a result.

It does **not** rescue the policy. With the bug fixed, the exact same
experiment still shows zero measurable improvement in bridge survival over
a purely scalar policy. The most likely reason: a *global* minimum cut
finds the single cheapest way to split the *entire* graph in two — with 12
randomly-scattered bridge memories among 6 clusters, that one global cut
usually isn't anchored at any particular bridge. A *local* signal (a cut
per cluster pair, for which the codebase already has an unused
Gomory-Hu-tree implementation) is the more promising next step, not
attempted here.

## Why this is still a useful result

The nightly research process this ran under treats a well-evidenced null
result as equally valuable as a positive one. Here, the honest outcome is:
a previously-rejected experiment stays rejected, but for a narrower and now
better-understood reason — and a general-purpose fix (prefer the
deterministic engine for *any* future `ruvector-mincut` consumer that needs
a repeatable partition, not just this one policy) falls out of the
investigation for free, already tested against another use case
(the crate's own unit tests) at no extra cost.

## Reproduce it

```bash
cargo run --release -p ruvector-agent-memory --example mincut_canonical_probe --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_bench --features mincut-forget
cargo run --release -p ruvector-agent-memory --example mincut_gated_forgetting_bench_canonical --features mincut-forget
```

Full raw output, methodology, and the ecosystem/long-horizon analysis are
in the companion nightly research document
(`docs/research/nightly/2026-09-20-canonical-mincut-forgetting/README.md`)
and ADR-346.
