# Does sparsifying a min-cut admission graph to K nearest centroids preserve its quality gain? No — but the cost ceiling it was meant to fix is real and closes cleanly.

## Problem

Streaming agent-memory systems have to decide, for every incoming vector,
whether it joins an existing cluster or starts a new one. A fixed
cosine-similarity threshold against the nearest centroid is the standard
approach, but it only looks at one edge of the similarity graph: it
cannot distinguish "a legitimate but distant member of a naturally spread
cluster" from "something weakly attached to everything."

`MincutGatedAdmission`, built in an earlier nightly, fixes that by
building a small weighted graph over *every* existing cluster centroid
plus the incoming point and running a global minimum cut (Stoer-Wagner)
over it — the point is admitted to an existing cluster only if it isn't
isolated on the weak side of the graph's single weakest link. It
measurably beats the threshold baseline on cluster purity and retrieval
recall at the same cluster-count budget. The catch: Stoer-Wagner is
O(V^3), and V is the current cluster count C, so the mechanism's own cost
grows cubically as an agent's memory accumulates topics over a long
session.

## Hypothesis

Build the same mechanism, but restrict the graph to each point's `K`
nearest existing centroids instead of all `C`. If quality survives and
cost drops, this is a clean fix for the cubic ceiling. Formally:

> Given `MincutGatedAdmission`'s existing admission mechanism (full graph
> over all C centroids), when the graph is instead restricted to the
> incoming point's K nearest centroids (K ∈ {4, 8, 16, 32}), then mean
> admission latency should stay well under the write-path's 500µs budget
> at cluster counts where the full graph exceeds it, subject to purity
> and recall@10 staying within 1 percentage point of the full graph at
> the same matched cluster count.

## Technical Design

`BoundedDegreeAdmission` reuses the exact same spawn/merge decision logic
as the full-graph mechanism (`should_spawn`, `merge_target`) — the only
change is which nodes go into the graph before Stoer-Wagner runs:

1. Select the `K` centroids most similar (cosine) to the incoming point
   — an O(C·K) bounded insertion into a size-K running list, not a full
   sort.
2. Build a `(K+1)`-node weighted graph over just those centroids plus the
   point — O(K²).
3. Run the identical Stoer-Wagner global min cut — O(K³) instead of
   O(C³).

When `K ≥ C`, step 1 selects every centroid, and the result is
mathematically identical to the full graph — a correctness regression
test checks this directly, 40 admissions, decision-by-decision.

## Implementation

Rust, in `ruvector-memory-admission` (an existing, previously-accepted
crate). Additive only: one new module (`bounded_degree.rs`, ~230 lines,
7 unit tests), one new benchmark binary, and three `pub(crate)` visibility
changes on existing shared helper functions so the decision logic isn't
duplicated a third time. No new dependency. No change to the existing
accepted mechanism or its benchmark's acceptance gates.

## Actual Benchmark Evidence

**Scaling** (exact cluster count, 80 timed decisions per point, release
build): the full graph crosses a 500µs write-path latency budget between
C=64 and C=96 (two independent runs landed in that band). Every tested
`K` stays under budget there. At C=256, K=4 is ~2,335x faster than the
full graph (6.3µs vs 14,709.5µs mean) while doing 126x fewer similarity
computations. **This half of the hypothesis holds cleanly.**

**Quality**, at a matched cluster budget on a realistic synthetic stream
(6,000 points, 24 ground-truth clusters, 20% noisy boundary points): at
K=16 and K=32, results are bit-identical to the full graph — but only
because this corpus's natural cluster count (15) never exceeds either K,
so the "approximation" never actually approximates anything. At K=4 and
K=8 — the values small enough to matter for cost — purity and recall
diverge from the full graph by 8.9–11.4 and 2.0–2.6 percentage points
respectively, both outside the pre-registered ±1pp band. (They happened
to diverge *upward* — better purity/recall, not worse — on this
particular corpus; that direction is not claimed to generalize, and a
residual cluster-count-matching imperfection in the calibration search
means even these numbers aren't perfectly clean.) **This half of the
hypothesis fails for every tested K.**

A bounded Darwin-style sweep scored all four `K` values on a fitness
function weighting latency heavily; K=4 scored highest — and is exactly
the kind of fast-but-wrong candidate a hard promotion gate exists to
block. No candidate was promoted; the full-graph mechanism remains the
production-relevant default.

**Overall acceptance: REJECT**, with the exit code of the benchmark
binary itself carrying that verdict, not a separately maintained claim.

## Limitations

Synthetic corpus only; one corpus configuration; no concurrent-writer or
delete-interaction testing; no edge/ARM hardware measurement; the
matched-budget calibration could not hit the exact target cluster count
for K=4/K=8, leaving a small residual confound disclosed in the full
write-up.

## Production Relevance

Negative, but useful: operators of a similar graph-gated admission
mechanism now have a measured crossover band (not a guess) for when cubic
cost becomes a real write-path problem, and a concrete warning that the
obvious cheap fix — just look at K nearest neighbors — is not safe to
reach for without re-validating quality on their own data. That is
immediately actionable for anyone building on `ruvector-agent-memory` or
a similar streaming-clustering admission layer.

## RuVector Ecosystem Implications

Answers an explicitly-named open question from the prior nightly that
built `MincutGatedAdmission`, closing it with measurement rather than
leaving it open indefinitely. Retains the rejected mechanism, its tests,
and its benchmark as lineage — a future nightly exploring adaptive or
locally-conditioned degree bounds (the natural next step this result
points to) starts from this evidence instead of from zero.

## Future Direction

A flat `K` cannot adapt to local graph density, which is plausibly
exactly what's needed to keep the cost win without losing quality. The
most direct next experiment: condition the degree bound (or the
admission threshold) on local similarity structure rather than a global
constant, and separately, fix the residual calibration confound with a
joint (tau, K) search instead of calibrating tau alone per fixed K.

## References

1. Stoer, M., Wagner, F. "A Simple Min-Cut Algorithm." *Journal of the ACM*, 1997.
2. Cover, T., Hart, P. "Nearest Neighbor Pattern Classification." *IEEE Trans. Information Theory*, 1967.
3. RuVector nightly research, 2026-09-02, "Global-Min-Cut Gated Streaming Memory Admission" (the parent result this nightly extends).
