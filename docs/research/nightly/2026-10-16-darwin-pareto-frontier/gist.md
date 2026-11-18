# Closing a Dependency Gap with a Benchmark That Disagreed With Me

## Problem

RuVector's self-tuning benchmark harness (`ruvector-sota-bench`) runs a
small evolutionary search — Darwin — over HNSW index parameters
(`ef_search`, `m`, `ef_construction`), looking for configurations that
improve recall or latency without regressing the other. When more than one
promising configuration survives a generation, deciding which ones are
*actually* improvements and which are just trade-offs requires Pareto
dominance: configuration A only beats configuration B if A is at least as
good on every tracked metric and strictly better on one.

The harness currently answers that question by calling an external npm
package, `@metaharness/darwin@^0.9.1`, for its Pareto-selection mode. An
existing architecture decision record in the repository (ADR-335) flagged
this explicitly as a dependency the project doesn't control, and asked for
either an in-repo implementation or an explicit decision to keep depending
on the external package. Nobody had done either yet — a grep across the
entire Rust workspace for "pareto" turns up nothing.

## Hypothesis

Build a small, dependency-free Rust module that computes Pareto frontiers
over multi-objective candidate data, verify it against the actual
mathematical definition of Pareto dominance (not against the external
package's unexamined internals), and measure whether a better algorithm is
actually better in practice at the scale this harness would use it.

## Technical Design

Three algorithms, one shared `dominates()` primitive:

```rust
pub fn dominates(objectives: &[Direction], a: &[f64], b: &[f64]) -> bool {
    // a dominates b iff a is >= b on every (oriented) objective
    // and strictly > on at least one.
}
```

- **Brute force** (`O(M·N²)`): for every point, scan all others; stop as
  soon as one dominator is found. This is the formal definition, used as
  ground truth for everything else.
- **NSGA-II's fast non-dominated sort** (Deb et al., 2002), restricted to
  front 0: compute a domination count for every point in one pass over all
  `N(N-1)/2` pairs, then keep the points with count zero. Textbook choice
  for multi-objective evolutionary algorithms.
- **Kung, Luccio & Preparata's 1975 divide-and-conquer skyline algorithm**,
  specialized to two objectives: sort by the first objective, scan while
  tracking the best second-objective value seen so far, keep anything that
  beats it. `O(N log N)`.

## What Actually Happened

The correctness story was clean: 200 random 2D instances, 200 more at
3 and 4 objectives, plus hand-built edge cases (duplicate points, total
antichains, a single dominating point, empty input) — all three algorithms
agreed with brute-force ground truth on every single case. Zero
disagreements across 407 generated and hand-written scenarios.

The performance story was not what the module's own doc comments initially
claimed. The first draft of this code said NSGA-II would be "the right
choice once a lineage accumulates many generations' candidates," reasoning
from its textbook complexity class. The benchmark said otherwise:

| N | brute force | NSGA-II | Kung 2D |
|---|---|---|---|
| 100 | 2.03 µs | 10.81 µs | 2.65 µs |
| 500 | 10.10 µs | 921.72 µs | 26.86 µs |
| 2,000 | 90.94 µs | 16.32 **ms** | 85.19 µs |
| 5,000 | 472.28 µs | 124.67 **ms** | 299.39 µs |

NSGA-II was slower than naive brute force at *every single measured size* —
by 5x at N=100, growing to 264x at N=5,000. The reason is almost
embarrassingly simple in hindsight: NSGA-II's domination-count pass visits
every pair unconditionally, while the "naive" brute-force scan exits the
moment it finds one dominator. For randomly distributed data, most points
get dominated fast — the Pareto frontier of N uniform random 2D points has
an expected size of only `O(log N)`. Brute force's early exit captures
almost all of the available savings for free; NSGA-II's textbook
optimization (counting dominations once instead of rescanning) doesn't
matter if you were never going to finish the rescan anyway.

The one real asymptotic win came from the algorithm that doesn't even
pretend to generalize: Kung's 2D-specialized divide-and-conquer approach.
It loses to brute force below roughly N=2,000 (sorting overhead isn't free),
crosses over between 2,000 and 5,000, and by 5,000 is 58% faster — then
keeps scaling near-linearithmically out to N=100,000 (15.4 ms), a regime
where running the `O(N²)` alternatives would take tens of seconds to
minutes.

The module's doc comments got rewritten to say this plainly, including the
specific ratios, rather than keeping the original "NSGA-II wins at scale"
framing that the benchmark had just disproved.

## Why This Is the Right Kind of Result

A complexity-class argument ("NSGA-II is `O(MN²)` just like brute force, but
with fewer effective comparisons") sounds right until you account for the
constant that early-exit buys the "worse" algorithm on exactly the kind of
sparse-frontier data this use case actually produces. The lesson generalizes
past this one module: an asymptotically-motivated choice between two
algorithms with the *same* big-O bound needs a benchmark, not a
blackboard, because the constant factor is where the real-world answer
lives.

## Limitations

- This module has no live integration yet. It was verified against the
  mathematical definition of Pareto dominance, not against
  `@metaharness/darwin`'s actual behavior — that package's internals were
  never inspected, so no equivalence claim is made.
- Only the two-objective case got a fast specialized algorithm. More than
  two objectives falls back to the NSGA-II implementation this benchmark
  just showed losing to brute force — which is fine for the small N a
  single Darwin generation produces (today's budget is 4 candidates per
  generation), but is exactly the kind of unresolved generalization that
  shouldn't be glossed over as "handled."
- The N=20,000/100,000 rows for brute force and NSGA-II are extrapolated
  from the N=5,000 measurement, not run directly — running NSGA-II at
  N=50,000 was estimated at roughly 12.5 seconds *per benchmark iteration*,
  which would make a full run take tens of minutes for a result the
  N=5,000 point already demonstrates clearly.

## Production Relevance

The module is additive, dependency-free, and has zero live callers today —
it does not yet replace the external npm dependency the triggering ADR
flagged. The concrete next step, explicitly scoped out of this run, is
wiring it into the TypeScript harness's actual Darwin promotion-selection
call site. What this run does provide is a verified, benchmarked primitive
to wire in, plus — maybe more durably useful — a documented example of a
plausible-sounding algorithmic choice that a five-minute benchmark
overturned before it shipped as an unexamined assumption.

## References

- Deb, K., Pratap, A., Agarwal, S., & Meyarivan, T. (2002). "A Fast and
  Elitist Multiobjective Genetic Algorithm: NSGA-II." *IEEE Transactions on
  Evolutionary Computation*, 6(2), 182-197.
- Kung, H. T., Luccio, F., & Preparata, F. P. (1975). "On Finding the
  Maxima of a Set of Vectors." *Journal of the ACM*, 22(4), 469-476.
- Börzsönyi, S., Kossmann, D., & Stocker, K. (2001). "The Skyline Operator."
  *Proceedings of the 17th International Conference on Data Engineering
  (ICDE)*.
- `docs/adr/ADR-335-repo-native-optimization-manifest.md` (this repository)
  — the dependency gap this run addresses.
- `schemas/optimization-manifest-v1.json` (this repository) — the per-repo
  objective/promotion schema this module's API mirrors.
