# Fixing a Rejected Experiment's Bottleneck Doesn't Always Save the Idea

## Problem

A prior experiment (ADR-345, in the same codebase) tried to make an
agent-memory compaction policy structure-aware: instead of scoring every
memory independently (recency, frequency, coherence), build a k-NN
similarity graph over the candidates and use a global minimum-cut to find
"bridge" memories — the sole semantic link between two otherwise-disjoint
topic clusters — so they aren't evicted just because they score low on
every scalar term.

That experiment was rejected, but not because the idea was wrong on its
merits: the specific min-cut implementation it called
(`RuVectorGraphAnalyzer::partition()`, a general dynamic min-cut wrapper)
was measured to be non-deterministic (50% of repeated calls on identical
input returned an unusable empty result) and slow (1,800-2,700x a plain
scalar-sort baseline, even on an 84-vertex test graph).

## Hypothesis

The same codebase separately ships a `canonical` feature on its min-cut
crate, built for exactly the determinism problem: a cactus-graph
representation that runs dense Stoer-Wagner to enumerate every global
minimum cut of a graph and deterministically picks the lexicographically
smallest one. Nobody had tried it against the rejected experiment's own
criteria. The question: does swapping backends — same policy logic, same
test corpus, same acceptance thresholds — fix the two measured blockers, and
is that enough to make the underlying idea viable?

## Technical Design

`CactusGatedForgetting` is a line-for-line mirror of the rejected policy,
with one substitution in its boundary-detection step:

```rust
// Before (rejected backend):
let mut analyzer = RuVectorGraphAnalyzer::from_knn(&neighbors);
let (side_a, side_b) = analyzer.partition().unwrap_or_default();

// After (this experiment):
let cactus = CactusGraph::build_from_graph(&graph);
let cut = cactus.canonical_cut();
let (side_a, side_b) = cut.partition;
```

Everything downstream — turning a cut partition into a "boundary vertex"
set, then either adding a scalar bonus (`Soft` mode) or reserving eviction
budget (`Hard` mode) for boundary vertices — is untouched.

## Actual Implementation

One real bug surfaced during implementation, worth calling out because it's
a general trap: an early version only inserted a k-NN graph edge `(i, j)`
when `i < j`, on the assumption that if `j` is among `i`'s nearest
neighbors, the relationship is roughly symmetric. It isn't, for exactly the
vertices this policy cares about most: a low-degree "bridge" vertex can have
a well-connected "gateway" vertex in its own short candidate list, while the
gateway's own list is dominated by its many same-cluster neighbors, pushing
the more-distant bridge out of its top-`k`. The `i < j` guard silently
dropped precisely the bridging edges the whole policy exists to detect —
caught immediately by two failing unit tests, fixed by inserting the edge
unconditionally from both endpoints and letting the graph's own undirected
deduplication handle the redundancy.

## Benchmark Evidence

All numbers below are from `cargo run --release` against a fixed-seed
synthetic corpus (identical to the rejected experiment's own: 6 clusters x
12 memories, 12 bridge memories, 32 dimensions), on `rustc 1.94.1`.

**Determinism** — 50 repeated calls on byte-identical input:

| Backend | Empty/degenerate results | Distinct partitions returned |
|---|---|---|
| Rejected (dynamic wrapper) | 50% | not applicable (non-deterministic) |
| This experiment (cactus) | **0%** | **1** |

**Speed** — same 84-vertex corpus, `Soft` policy:

| Backend | Compaction wall-clock |
|---|---|
| Rejected (dynamic wrapper) | 151,802 microseconds |
| This experiment (cactus) | **1,632 microseconds** |

That's a 93x speedup on top of full determinism. Both prior blockers: fixed.

**But the underlying idea doesn't hold up.** At the pre-registered seed,
*neither* backend beat a plain scalar-scoring baseline on bridge-memory
survival (16.7% for the baseline and the old backend; 8.3%, actually worse,
for the new one). That single data point could have been an unlucky seed —
so a follow-up swept 10 more seeds:

| Metric | Old backend | New (cactus) backend |
|---|---|---|
| Mean survival gap vs. baseline | -3.3 percentage points | -4.2 percentage points |
| Seeds meeting the pre-registered +15pp bar | 0 / 10 | 0 / 10 |

Zero out of ten. The original experiment's one positive result looks, in
hindsight, like a favorable-seed artifact rather than a reproducible
property of "protect the global min-cut boundary."

## Why This Happens

A single global minimum cut of a many-cluster graph finds *one* structurally
weakest point in the *entire* graph — generically whichever vertex or small
group has the least total edge weight. In a corpus deliberately constructed
with 12 separate bridge memories across 6 clusters, there is no guarantee
that "the one globally weakest link" coincides with any particular one of
those 12 engineered bridges. Making the min-cut computation faster and
deterministic doesn't change what it's fundamentally computing.

## Limitations

- Only a global cut was tested; a per-cluster or local-cut variant (which
  would more directly target "bridges between specific cluster pairs"
  rather than "the single weakest point anywhere") is untested and is the
  natural next experiment.
- Scaling: the cactus backend's dense Stoer-Wagner still grows roughly
  cubically (25ms at 100 vertices to 8.2 seconds at 800), so while it's a
  large constant-factor win, it doesn't reach the thousands-of-memories
  scale the original experiment wanted to test at either.
- Only the `Soft` policy variant was included in the 10-seed sweep, to keep
  its runtime bounded.

## Production Relevance

Two independently useful, validated facts survive a fully rejected
hypothesis: (1) the codebase's `canonical` cactus min-cut feature is now
proven correct, deterministic, and fast at small-to-medium graph sizes —
reusable by any future component that needs those properties without this
component's specific use case — and (2) "protect the global min-cut
boundary" is now a documented dead end for *this* eviction task, backed by
11 seeds of evidence rather than the 1 seed the original experiment had, so
a future engineer doesn't have to rediscover that the fast, correct version
of the same idea still doesn't work.

## RuVector Ecosystem Implications

This connects the min-cut crate, the agent-memory crate, and the existing
witness/provenance mechanism (re-verified, unchanged, 20/20 tamper
detections) — and demonstrates a useful pattern for this kind of iterative
research: when an experiment is rejected for an *implementation* reason
(slow, flaky) rather than a *design* reason, the right first move is
checking whether a fix already exists in-tree before either abandoning the
idea or reimplementing the fix from scratch. Here it did, and applying it
cleanly separated "was the implementation broken" (yes, now fixed) from "was
the idea correct" (no, now shown more convincingly than before).

## Future Direction

Community-aware or local minimum cuts — computed per topic cluster rather
than once globally, or via a sparsest-cut query if the min-cut crate's
`all-cut-queries` feature supports it at these graph sizes — are the
concrete next step this evidence points to, since they would target the
specific "bridge between these two clusters" structure the corpus actually
constructs, rather than one arbitrary global weak point.

## References

- Prior experiment and rejection: `docs/research/nightly/2026-09-05-mincut-gated-forgetting`.
- Stoer, M. and Wagner, F., 1997. "A simple min-cut algorithm." *Journal of the ACM*, 44(4).
