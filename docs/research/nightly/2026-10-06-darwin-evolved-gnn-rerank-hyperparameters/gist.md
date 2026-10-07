# Darwin: a bounded evolutionary search engine for RuVector, and what it found in 20 minutes of compute

## Problem

RuVector's GNN-based candidate reranker (`ruvector-gnn-rerank`) has three
hand-set hyperparameters: `alpha` (diffusion self-weight), `coherence_threshold`
(structural edge gate), `k_graph` (candidate-graph degree). They were set to
`(0.60, 0.50, 8)` by eyeballing one benchmark run in May 2026. The nightly
research doc that introduced them says, in its own words, that "optimal alpha
calibration for production embeddings is unknown" and lists adaptive tuning as
future work. It stayed future work. Two more nightly cycles hit the same wall
with different hand-set constants and said the same thing. By September, three
separate research reports were independently saying: we have no evolutionary
or adaptive tuning tool in this codebase at all.

That's the gap this run closes — not with a new reranking algorithm, but with
the thing that was missing underneath all of them.

## Hypothesis

A genuinely bounded evolutionary search — fixed budget, no surrogate model, no
hidden iteration — applied to those three hand-set numbers, evaluated honestly
(fitness measured on one slice of queries, final acceptance measured on a
disjoint slice the search never sees), should find hyperparameters that beat
the hand-set defaults on the held-out slice, not just on whatever it was
optimized against.

If it only wins on the slice it was scored against, that's overfitting, and
the honest answer is to say so.

## What was built

`ruvector-darwin`: a new, small (~390 lines including tests), dependency-light
Rust crate. It is not an AutoML framework. It is the smallest thing that can
honestly be called a bounded evolutionary search:

- A genome is a bounded vector of real numbers.
- Each generation mutates the current best candidate `candidates_per_generation`
  times (Gaussian perturbation, clamped back into bounds) and keeps the best.
- The budget is fixed before the run starts: 3 generations x 4 candidates x 1
  promotion max — 13 fitness evaluations total, win or lose.
- Every evaluated candidate — rejected, losing, or winning — is retained in a
  JSON lineage record. Nothing is thrown away.
- If nothing beats the parent, the parent is kept. That's a correct outcome,
  not a bug.

```rust
let report = run_evolution(parent, bounds, EvolutionConfig::default(), |g| {
    Evaluation::Fitness(fitness_on_fitness_set_only(g))
});
// report.promoted is None, or a genome that strictly beat the parent.
```

## What it found

Applied to `GnnMincutReranker` against the exact same synthetic benchmark
(5,000-point multi-Gaussian corpus, 100 queries, seed 42) the May nightly used,
with the 100 queries split 70/30 into a fitness set and a held-out set:

| | Fitness-set recall@10 | Held-out recall@10 |
|---|---|---|
| Hand-set defaults (0.60, 0.50, 8) | 38.29% | 38.67% |
| Evolved (0.317, 0.292, 8) | 47.14% | **51.00%** |

The held-out improvement (+12.3 points) is *larger* than the in-sample
improvement (+8.9 points). That's the opposite of what overfitting looks like
— if the search were gaming its own fitness function, the held-out number
would lag behind, not lead. Latency, measured as a side constraint rather than
optimized for, came in slightly faster on the evolved parameters (0.89x),
within the experiment's own pre-declared 1.5x regression ceiling.

Run three times as separate OS processes: identical recall numbers and
identical JSON lineage every time. Only wall-clock timing varied, as expected.

## Why this is the right size of contribution

It would have been easy to reach for something heavier — Bayesian
optimization, a surrogate model, an LLM-driven search over algorithm code
(RankEvolve, a February 2026 paper, does exactly that one level up, mutating
retrieval *algorithms* rather than their parameters). None of that was
warranted here. The problem was three numbers and a benchmark that runs in
milliseconds. The actual gap wasn't "we lack a sophisticated optimizer" — it
was "nobody wrote even a simple one, three times in a row, and said so each
time." Closing that gap with the simplest honest thing that works, and making
it reusable, is worth more than closing it with something impressive.

## What this doesn't claim

The evolved numbers beat the hand-set ones on this synthetic benchmark. They
are not claimed to be optimal, and they are not validated against real
(non-synthetic) embedding distributions — that's the natural next step, not
something this run skipped past quietly. The production reranker's actual
default is unchanged; this is a measured recommendation, not a silent
behavior change.

## Where it goes next

`ruvector-darwin` has no required dependents yet — it's deliberately generic.
Two other nightly cycles already named hyperparameters that want exactly this
treatment (`ruvector-coherence-hnsw`'s adaptation rate, `ruvector-agent-memory`'s
compaction weights). The natural follow-up — explicitly requested by name in
two of the nightlies that motivated this work — is a `ruFlo` workflow that
runs this same search continuously against live-but-held-out traffic instead
of a synthetic benchmark, with promotion gated the same way: beat a real
baseline on data the search never saw, or don't ship it.

## Evidence

Full research report: [`README.md`](./README.md). Raw benchmark output and
JSON lineage: [`evidence/`](./evidence/). ADR:
[ADR-352](../../../adr/ADR-352-darwin-bounded-evolutionary-search.md).
