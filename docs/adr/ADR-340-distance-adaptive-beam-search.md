# ADR-340: Distance-Adaptive Beam (DAB) Search for ANN Graph Traversal

**Date**: 2026-08-25
**Status**: Closed — negative result (documented; not recommended for production)
**Deciders**: Nightly research agent (revised after measured review)
**Tags**: ann, hnsw, beam-search, ruvector-dab-search, negative-result, adr-303-followup

---

## Context

ADR-303 (`ruvector-entropy-ann`) tested whether the Shannon entropy of the candidate-heap distance
distribution could serve as a live, per-query stopping signal for HNSW-style beam search, replacing
the fixed `ef_search` budget. It measured a negative result: heap-distance entropy saturates near
`ln(n)` for every query on that PoC's data, so `EntropyScaledEf`'s apparent recall gain was entirely
explained by a larger effective search budget, not by any real per-query adaptivity. Its prior-art
table cited "Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest Neighbor
Search" (arXiv:2505.15636) as a scalar-distance-threshold alternative but did not implement or
measure it.

This nightly implements and measures that cited alternative, against the exact methodological trap
that sank the entropy signal.

---

## Hypothesis

```text
Given a 2,000-vector synthetic corpus at dimension 16, clustered into 10 groups (identical
construction to ADR-303's benchmark), indexed by a single-layer k-NN proximity graph
(k=16 neighbours/node) with query-time entry routing through 40 deterministic seed nodes,

when beam-search traversal uses the distance-adaptive stopping rule
d(q,x) >= (1+gamma) * d(q,x_k) (gamma=0.5, pre-registered) instead of a fixed ef_search budget,

then (1) the rule's per-query work (distance computations) should vary measurably more on hard
queries than easy queries (ratio >= 1.15) — the direct test for "does it actually adapt", contrasting
with ADR-303's measured EntropyScaledEf, whose effective budget was constant (~124) for every query,

and (2) recall@10 should stay within 3 points of a FixedEf(100) high-recall reference on every
query set,

and (3) on hard queries specifically, it should beat a FixedEf baseline whose ef is calibrated (on a
disjoint query set) to match its own average distance-computation budget, by >= 2 recall points —
proving adaptive reallocation beats flat allocation at equal average cost.
```

**Result: REJECT.** Test (1) and test (3) failed as measured; test (2) passed. See
[Evidence](#evidence).

---

## Decision

**Do not adopt the distance-adaptive stopping rule in this PoC form.** The rule is real — unlike
ADR-303's entropy signal, `AdaptiveGamma`'s per-query distance-computation count has substantial,
genuine spread (stddev 96–172 vs FixedEf's 19–61 at comparable means) — but it adapts to the wrong
thing for this use case, and it does not beat a matched-budget baseline.

### What was measured

On the pre-registered `gamma = 0.5`:

| Query set | FixedEf(50) recall | FixedEf(100) recall | AdaptiveGamma(0.5) recall | AdaptiveGamma mean dist_comp |
|---|---|---|---|---|
| easy | 0.811 | 0.846 | **0.903** | 346.7 (sd 153.6) |
| hard | 0.706 | 0.722 | **0.756** | 317.3 (sd 130.4) |
| mixed | 0.635 | 0.663 | **0.678** | 291.5 (sd 95.7) |

Recall is higher than both FixedEf baselines on every query set — but at a materially higher
distance-computation cost (291.5–346.7 vs 215.9–276.8), so this is not evidence of a better
recall/cost trade-off by itself; see the matched-budget and matched-recall tests below, which
control for that.

**Test 1 — adaptivity direction (FAIL).** The pre-registered hypothesis was that harder
(out-of-distribution) queries would need *more* work. Measured: `hard/easy` distance-computation
ratio = **0.915** — hard queries cost *less*, not more. Mechanistically, the `(1+gamma)*d_k`
threshold is a relative distance margin: in the sparse region around an out-of-distribution query,
few graph nodes fall within that margin at all, so the frontier is exhausted quickly; in a dense
cluster interior, many nodes fall within the margin, so expansion continues longer. The signal is
real and query-dependent (unlike ADR-303's constant `ef_actual`), but it tracks **local point
density**, not **task difficulty** — a different flavour of the same failure ADR-303 named: "the
softmin entropy... measures the local density of the neighbourhood the search has landed in, not
the ambiguity of routing to it." Two independent per-query signals on two different mechanisms have
now both been observed to track density instead of difficulty on this synthetic dataset — worth
treating as a standing caution for the next attempt at this problem, not a coincidence to ignore.

**Test 2 — recall floor (PASS).** AdaptiveGamma(0.5) beat the FixedEf(100) reference by +0.033 to
+0.057 recall points on every query set.

**Test 3 — matched-budget control (FAIL, narrowly).** A `FixedEf(150)` baseline, calibrated on the
*mixed* query set only to match AdaptiveGamma(0.5)'s mean distance-computation cost there (291.0 vs
291.5 — a tight match), scored 0.741 recall on the *hard* set, vs AdaptiveGamma's 0.756: an advantage
of **+0.014**, short of the pre-registered **+0.02** threshold. This is the test ADR-303's
`EntropyScaledEf` could not even approach (it tied its matched-budget control to four decimal
places); AdaptiveGamma comes measurably closer to real value but still falls short of the bar set in
advance.

**Headline metric (cost at matched recall — the source paper's own primary metric).** At recall
matched to AdaptiveGamma(0.5)'s 0.678 on the mixed set, `FixedEf(122)` needs 273.4 distance
computations/query vs AdaptiveGamma's 291.5 — a **-6.6%** change, i.e. AdaptiveGamma needs *more*
work for the same recall on this dataset, the reverse of the 10–50% reduction the source paper
reports on SIFT1M/DEEP/GloVe/GIST/MNIST. The most likely explanation (untested further this run,
listed under Open Questions) is that this crate's flat exact-k-NN graph and small synthetic corpus
lack the degree heterogeneity of a real incrementally-built HNSW/Vamana graph, which is exactly the
structure the source paper's theorem assumes ("navigable graphs").

### Candidate B — capped variant

`AdaptiveGamma(gamma=0.5, max_expansions=40)`, a production safety bound fixed before any run
(chosen to roughly match `FixedEf(50)`'s typical expansion count, not tuned to results), behaves
almost identically to `FixedEf(50)` on every metric (recall 0.625–0.801 vs FixedEf(50)'s 0.635–0.811;
cost 198–202 vs 215–218). It does not blow up on hard/sparse queries — a real, useful property if
this direction is revisited — but it also forfeits essentially all of the uncapped variant's recall
gain, confirming the gain is concentrated in the (potentially unbounded) tail the cap removes.

---

## A structural finding, orthogonal to the gamma hypothesis

While building this crate's benchmark harness, an earlier design used a single fixed traversal
entry point (the node nearest the corpus centroid, computed once at build time) instead of
ADR-303's per-query O(n) brute-force entry scan — deliberately, to avoid an O(n) entry cost swamping
the O(dozens–hundreds) traversal-cost metric this crate measures. That design measured ~19% recall
across the board, regardless of stopping rule or gamma value. Root cause: an exact k-NN graph over
well-separated clusters has few or no edges *between* clusters, so a single fixed entry point can
only reach the ~1/10 of the corpus in its own cluster (10 clusters in the benchmark; ~19% ≈ close to
2/10, consistent with adjacent-cluster noise overlap at this dataset's noise level). The fix used in
the shipped version — `FlatGraph::entry_seeds`, a small (40-node) deterministic sample probed at
query time (`O(entry_seeds)`, not `O(n)`) to select the nearest as the entry point, approximating a
coarse HNSW upper-layer routing step — restores realistic recall (0.625–0.917 across variants) at a
small, constant, cross-variant-equal cost. This is graph-construction plumbing, not part of the
gamma hypothesis, but it is exactly the kind of confound an attack pass is required to catch, and it
is retained in the crate (`graph.rs` docs, `search::tests::loose_gamma_achieves_high_recall_on_majority_of_self_queries`)
as a documented pitfall for any future flat-graph PoC in this repository.

---

## Alternatives Considered

| Alternative | Notes |
|-------------|-------|
| Fixed ef (status quo) | Remains the recommendation; beats AdaptiveGamma at matched recall on this dataset |
| Heap-distance entropy (ADR-303) | Already rejected; a different but related density-not-difficulty failure |
| Ada-ef (arXiv:2512.06636) | Requires an offline-trained regressor; not attempted this run |
| Min-expansion floor + gamma hybrid | Untested; see Open Questions |

---

## Consequences

### What the merge provides

- A self-contained, zero-dependency Rust harness (`ruvector-dab-search`) implementing the DAB
  stopping rule faithfully to its source paper's stated inequality, with a matched-budget control
  and a matched-recall headline number as permanent benchmark columns.
- A corrected, reusable entry-routing pattern (`entry_seeds`) for any future flat-graph ANN PoC in
  this repository, with the failure mode it fixes documented in both code and this ADR.
- A second, independently-measured data point (after ADR-303) that a natural per-query stopping
  signal on this synthetic clustered dataset tracks local density rather than task difficulty —
  useful negative evidence for whoever attempts this problem next.

### Costs / trade-offs measured

- AdaptiveGamma(0.5) costs 291.5–346.7 distance computations/query vs FixedEf(50)'s 215.9–218.1 and
  FixedEf(100)'s 256.3–276.8 — 8–61% more work than the baselines it is compared against, for a
  recall gain of +1.5 to +5.7 points, a worse cost/recall trade-off than simply raising `ef`
  (confirmed by the -6.6% matched-recall headline number).
- The capped variant (candidate B) removes the tail-latency risk but also removes essentially all
  of the recall gain, landing within noise of `FixedEf(50)`.

### If this is ever revisited

1. Test on a real incrementally-built HNSW or Vamana graph (this PoC's flat exact-k-NN graph is not
   proven navigable, and the source paper's guarantee assumes navigability).
2. Test on real embeddings (SIFT1M/GloVe/GIST, matching the source paper) rather than synthetic
   clustered data, where local density and task difficulty may correlate differently.
3. Try a hybrid: a minimum-expansion floor (preventing premature termination in sparse regions)
   combined with the gamma rule (preventing over-expansion in dense regions) — this PoC's evidence
   suggests the two failure directions are somewhat separable.
4. Always report both a matched-budget control and a matched-recall headline number; this run's
   narrow test-3 miss (+0.014 vs required +0.02) would have looked like a clean win on recall
   numbers alone.

---

## Implementation Status

**PoC**: `crates/ruvector-dab-search` v0.1.0 — merged as negative result
**Tests**: 17 assertions, all pass (`cargo test --release -p ruvector-dab-search`)
**Benchmark**: `cargo run --release -p ruvector-dab-search --bin benchmark` — includes the
matched-budget control (Test 3) and matched-recall headline number as permanent output

No production integration is planned.

---

## Open Questions

- Would the same rule show a positive matched-budget result on a real incrementally-built HNSW
  graph, where degree heterogeneity and true navigability hold by construction?
- Does the density-vs-difficulty confound observed here (and, differently, in ADR-303) generalize
  to real embedding distributions, or is it an artifact of this repository's synthetic
  cluster-plus-noise dataset generator?
- Is a minimum-expansion floor + gamma hybrid (item 3 above) sufficient to fix the observed
  under-search-on-hard-queries direction without reintroducing a fixed-budget confound?

---

## References

- arXiv:2505.15636 — Distance Adaptive Beam Search for Provably Accurate Graph-Based Nearest
  Neighbor Search (source of the `(1+gamma)*d_k` stopping rule implemented here)
- ADR-303 — Entropy-Adaptive Beam Search for ANN Graph Traversal (prior nightly; this ADR's
  matched-budget-control methodology and density-vs-difficulty framing both build directly on it)
- Research README: `docs/research/nightly/2026-08-25-distance-adaptive-beam-ann/README.md`
