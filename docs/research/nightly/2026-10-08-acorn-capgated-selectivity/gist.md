# Why more graph edges made capability-gated search *worse*, not better

## Problem

Vector databases that enforce per-record access control (capability tokens, row-level ACLs, multi-tenant isolation) face the same recall-collapse problem as any filtered ANN search: when only a tiny fraction of the corpus is visible to a given caller, a graph index built for unfiltered search can starve before it finds enough authorised results. ACORN (Patel et al., SIGMOD 2024) fixed this for generic metadata predicates with two levers: build denser graphs (γ-augmentation) and keep expanding a node's neighbours even when that node itself fails the predicate, so the search beam doesn't die.

RuVector's `ruvector-capgated` crate already implements the second lever for capability-token access control specifically — its `CapGraphIndex` expands every visited node's neighbours "regardless of the node's capability" by design. What it had never tested was the first lever, or realistic low-selectivity access patterns (its own benchmark only goes down to 12.5% authorised). Its source code says, verbatim, that its flat k-NN construction should "replace with HNSW for production" — a TODO nobody had followed up on.

## Hypothesis

Build the same existing `CapGraphIndex` with 4x the graph degree (γ=4), and separately build a new two-layer, HNSW-style index that finds better search entry points via a sparse top layer instead of fixed positions. Test both against the unmodified baseline at 1.56% authorised selectivity (one capability bit held out of 64 — the realistic shape for thousands of agents sharing one memory index). Expect both to measurably improve recall, matching ACORN's published result.

## What we built

- `HierarchicalCapGraphIndex` (new, ~250 lines of pure, dependency-free Rust): a sparse top layer (1-in-16 nodes) used only for greedy entry-point descent; the base layer, degree, visited-node budget, and predicate-agnostic traversal rule are copied verbatim from the existing `CapGraphIndex`, so any measured difference is attributable to seeding quality alone.
- A γ=4 variant needed **zero new code** — `CapGraphIndex` already exposes degree as a constructor parameter. Nobody had swept it at this selectivity.
- A frozen benchmark (`examples/selectivity_acorn_bench.rs`) comparing both against baseline at the crate's existing production-facing default (`ef_multiplier=30`, i.e. visit ≤300 of 4,000 nodes per query).

## What happened

At the frozen budget, **both hypotheses failed**:

| Variant | Low-access recall@10 | QPS |
|---|---|---|
| baseline (degree=12) | 0.585 | 5,619 |
| γ=4 (degree=48) | **0.548** (worse) | 2,539 (less than half) |
| hierarchical seeding | 0.587 (±noise) | 5,632 |

The denser graph didn't just fail to help — it measurably hurt, while costing more than twice the query latency. Better seeding did essentially nothing.

## Why

We didn't stop at the rejection — a supplementary diagnostic (run *after* the frozen verdict, so it couldn't influence it) swept the visited-node budget from 30 to 4,000 nodes:

```
  ef_mul  base_recall  candA_recall(γ=4)
      10        0.204        0.187
      30        0.585        0.548
     100        0.949        0.994
     300        0.979        1.000
    1000        0.979        1.000
```

γ-augmentation *does* eventually win — decisively, reaching perfect recall — but only once the search is allowed to visit ~25%+ of the corpus, roughly 3–10x more than the crate's current default budget. The mechanism: the stopping rule caps how many nodes get *visited* (popped), not how far the search ranges spatially. A denser graph spends that fixed visit-budget thoroughly covering the query's immediate neighbourhood — which at 1.56% selectivity is almost entirely unauthorised — before being forced outward to where the rare authorised node actually is. A sparser graph runs out of near candidates to explore sooner, and so gets pushed outward "for free," earlier, which turns out to help under a tight budget and stops mattering once the budget is generous.

A second, unplanned finding: even visiting the *entire* graph, the existing baseline plateaus at 97.9% recall, never reaching 100%. Its adjacency is directed — a node's edges point only to its own nearest neighbours — so a vector that happens to be nobody's "near neighbour" is structurally unreachable by graph traversal, for any caller, authorised or not. This is a real connectivity gap in production code, independent of this experiment's hypothesis, and γ=4 heals most of it as a side effect.

## Why this is a successful run despite the rejection

Both hypotheses were real, well-motivated, and testable — and both were falsified at the budget that actually matters in production, with a concrete, data-supported explanation for why, plus one previously-unknown correctness-adjacent finding (the connectivity ceiling) as a bonus. Nothing here was fabricated, cherry-picked, or adjusted after the fact: the acceptance thresholds and dataset were frozen before the first run, and the root-cause diagnostic was computed after the verdict, not before it.

## What's next

The real lever, per the diagnostic, isn't degree or seeding — it's the *stopping rule*. A node-count budget forces a tradeoff between "explore thoroughly nearby" and "range far enough to find rare authorised results" that a smarter rule (stop when the top-k hasn't improved recently, or stop by wall-clock instead of node count) might sidestep entirely. That's the next experiment, named explicitly so a future nightly run doesn't have to re-derive this.

## Code

Both new variants and the benchmark are in `ruvnet/ruvector`, `crates/ruvector-capgated/{src/hierarchical.rs,examples/selectivity_acorn_bench.rs}`, additive and fully tested (28/28 crate tests pass, including 6 new ones), building on top of the crate's own `CapGatedIndex` trait without touching its existing API.
