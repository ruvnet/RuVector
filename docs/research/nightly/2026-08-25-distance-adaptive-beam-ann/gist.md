# A real adaptive stopping signal for ANN search still lost to a fixed budget — here's the honest data

## Problem

Graph-based approximate nearest-neighbour search (HNSW and friends) traverses a proximity graph
with a fixed `ef_search` candidate budget, tuned offline. That's a systematic mismatch: easy queries
(near a dense cluster) finish long before the budget is spent; hard, out-of-distribution queries run
out of budget before finding their true neighbours. The right `ef` is per-query, and the graph
traversal itself is one of the only places that could supply a free, zero-calibration signal for it.

A previous experiment in this series tried Shannon entropy of the candidate heap as that signal and
found it didn't work: the entropy saturated to the same value for every query, so the "adaptive"
result was just a bigger fixed budget wearing a costume. That negative result cited, but didn't
implement, an alternative from the literature: a 2025 paper's *distance-ratio* stopping rule with an
actual proof behind it.

## Hypothesis

Implement that cited alternative for real, and hold it to the same bar that caught the entropy
signal's flaw: it has to beat a **matched-budget control**, not just look good on a recall table.

The rule: maintain the current top-k best-found distances during graph traversal; stop expanding as
soon as the next candidate in line is farther than `(1+γ)` times the current k-th-best distance,
for a tunable `γ` in `(0, 2]`. On a graph with a certain navigability property, this provably bounds
how far any undiscovered point can be. Pre-registered before any run on this dataset: `γ = 0.5`.

Three numeric bars, fixed before benchmarking:

1. The rule's per-query cost must vary substantially more on hard queries than easy ones (a direct
   test for "does it actually adapt," since the previous signal's own failure was that it didn't).
2. Recall can't drop more than 3 points below a generously-budgeted fixed baseline.
3. On hard queries, at a cost matched to its own average budget, it must beat a plain fixed-budget
   baseline by at least 2 recall points.

## What happened

Built the whole thing in Rust: a k-NN proximity graph, three search variants (fixed-budget baseline,
the new adaptive rule uncapped, and a capped production-safe version), a deterministic synthetic
benchmark, and — before trusting any of it — a correctness test suite that caught a real bug: an
early version used a single fixed graph entry point, which turned out to only be able to *reach*
about a fifth of the corpus, because a nearest-neighbour graph over separated clusters has no edges
between them. Fixed with a small deterministic set of routing candidates probed per query instead.

With that fixed, the actual experiment:

- **Bar 1 (does it adapt) — failed, in an interesting way.** The signal *does* vary a lot per query
  (that alone beats the previous entropy attempt, which didn't vary at all). But it varies backwards:
  hard, out-of-distribution queries cost *less* work than easy, cluster-core queries — a 0.915 ratio
  where the hypothesis needed at least 1.15. The mechanism, once you look at it, makes sense: the
  stopping threshold is a relative distance margin, and in a sparse region (where a hard query
  lands) there just aren't many points within any given margin, so the frontier runs dry fast. In a
  dense cluster, lots of points sit inside that margin, so expansion drags on. The rule is adapting
  to local crowding, not to how hard the query actually is.
- **Bar 2 (recall floor) — passed comfortably.**
- **Bar 3 (beats a matched-budget baseline) — failed, narrowly.** +0.014 recall advantage where +0.02
  was required. Close, but the pre-registered number is the pre-registered number.
- **The headline comparison** — cost needed to hit the same recall as a plain fixed budget — came out
  *negative*: the adaptive rule needed 6.6% *more* work for equal recall, the opposite of the
  10-50% reduction the source paper reports on real embedding benchmarks with real HNSW graphs.

## Why the mismatch with the source paper is not a contradiction

The paper's guarantee, and its reported wins, are on **navigable graphs** — the kind you get from an
incrementally-built HNSW or Vamana index. This experiment used a flat, exact k-nearest-neighbour
graph over a small synthetic dataset, which is not proven navigable (the entry-point bug above is
direct evidence it wasn't even fully reachable without a fix). So this is evidence about *this graph
construction*, not a refutation of the paper's own results. The natural next experiment is obvious:
run the same rule on a real incrementally-built HNSW graph over real embeddings before concluding
anything about the method itself.

## The useful part

Two independent per-query signals, tried on two separate nights, using two different mechanisms
(entropy of a distance distribution; a relative distance-ratio threshold), have now both been
observed to track *local density* instead of *task difficulty* on the same kind of synthetic
clustered dataset. That's either a real property of local/relative signals on this class of data, or
an artifact of how "hard" queries were generated (uniform random points, which are also just points
in sparse regions — density and difficulty are confounded by construction). Either way, it's a
specific, falsifiable thing to check before a third attempt at this problem: build a "hard" query set
that's genuinely difficult *without* also being in a sparse region, and see if either signal behaves
differently.

## What this is not

Not a claim that distance-adaptive stopping doesn't work — the cited paper's own results, on real
navigable graphs, say otherwise. Not a production recommendation either direction. It's a specific,
reproducible negative result on a specific graph construction, plus a concrete, falsifiable next
step, which is what a rejected hypothesis with real evidence is supposed to leave behind.

## Reproduce it

```bash
cargo test --release -p ruvector-dab-search
cargo run --release -p ruvector-dab-search --bin benchmark
```

Both are deterministic (fixed seeds throughout); rerunning reproduces the same numbers reported here.

## References

- Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor Search,
  arXiv:2505.15636 (NeurIPS 2025) — source of the stopping rule implemented here.
- The prior nightly this one follows up on: entropy-adaptive beam search (negative result), same
  repository, `docs/research/nightly/2026-08-13-entropy-adaptive-ann/`.
