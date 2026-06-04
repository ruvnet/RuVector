---
adr: 201
title: "Region-Pruned IVF for Filtered ANN vs ACORN: Qualified NO-GO"
status: proposed
date: 2026-06-04
authors: [ofershaal, claude-flow]
related: [ADR-193, ADR-196, ADR-199, ADR-200]
tags: [ruvector, retrieval, ann, filtered-search, acorn, ivf, region-pruning, no-go]
---

# ADR-201 — Region-Pruned IVF for Filtered ANN vs ACORN: Qualified NO-GO

## Status

**Proposed — qualified NO-GO at the pre-registered bar (2026-06-04).** BET 2 ⊗ BET 4 of the
SepRAG exploration (issue #534): does region-pruned IVF search beat the in-repo `ruvector-acorn`
incumbent on *correlated* filtered queries? Pre-registration:
[`docs/plans/bet2-filtered-ann/PRE-REGISTRATION.md`](../plans/bet2-filtered-ann/PRE-REGISTRATION.md).

Region-pruning beats *vanilla* ACORN by 6–48× distance-evals (and 4.7–26× wall-clock) at
selectivity ≤ 1%. **But the pre-registered ≥5× WIN does not survive the mandatory adversarial
check (protocol rule #5):** giving ACORN a *predicate-aware entry* — a simple, known enhancement
— collapses the advantage to **~2× at high correlation (ρ=1), below the 5× bar.** A retains a
real but **narrow, conditional** edge at *moderate* correlation (ρ≈0.7, 6–39×) and very low
selectivity, plus an at-scale metric caveat that favours it. Net: the bet **does not cleanly
pay**; the clean win was an artifact of an under-equipped incumbent.

## Context

Filtered ANN ("nearest among items matching predicate X") is a real flat-ANN weakness: a
post-filter graph walk starves at low selectivity. `ruvector-acorn` (SIGMOD 2024,
arXiv:2403.04871) fixes this with a denser γ·M graph + predicate-agnostic traversal, and is the
strong in-repo incumbent. The hypothesis (BET 2 ⊗ BET 4): when the predicate **correlates** with
embedding-cluster structure (the production metadata-filter case — `tenant`, `doc_type`, `year`,
`category`), an IVF hierarchy can **skip whole clusters with zero matches** and beat ACORN on
cost. On embeddings the pruning kernel cannot use graph separators (high treewidth, [ADR-199]),
so the substrate is the treewidth-immune IVF hierarchy (`ruvector-rairs`, [ADR-193]) — BET 4 is
the mechanism, BET 2 the benchmark.

## Method

Self-contained crate `ruvector-filtered-bench` (depends only on `ruvector-acorn` +
`ruvector-rairs`; independent of [ADR-200]/PR #535). Real ogbn-arxiv (n=20k slice, 128-d, 40
subject labels). Ground truth = `ruvector-acorn::exact_filtered_knn`. Cost = **distance-evals/
query** (hardware-independent), with wall-clock as an honesty guard. Predicates built by a
ρ-correlation knob holding selectivity *exactly* constant across ρ (shuffle a fraction 1−ρ of a
structured label-class set), so cost deltas are attributable to correlation, not set size.

Contenders, all scored against the same oracle, all reporting **exact** distance-evals (ACORN
was instrumented with additive, result-preserving `*_counted` search variants):
- **A** — region-pruned IVF (`prune::RegionPruneIvf`): k-means partition + two stacked prunings
  — skip zero-match clusters (predicate) and a triangle-inequality branch-and-bound on cluster
  radius (exact). The salvaged separator-tree B&B kernel ([ADR-196]) on the IVF hierarchy.
- **B** — tuned vanilla ACORN (γ=2, ef swept; ef=512 ≈ 92% recall at sel=1%).
- **C** — post-filter floor (retrieve top-pool unfiltered, then filter).
- **D** — ACORN with predicate-aware entry (the rule-#5 "tune harder" adversary): sample probes,
  predicate-test free, distance-eval only matching probes, seed the beam from the nearest match.

## Evidence

### The benchmark has teeth (negative control, M1)

Post-filter (C) vs agnostic ACORN (B) on the *same* graph, ρ=1, recall@10:

| sel | B (agnostic) | C (post-filter) |
|---|---|---|
| 0.1% | 73.7% | **22.7%** |
| 0.5% | 90.4% | **59.7%** |
| 1% | 92.6% | 79.3% |
| ≥5% | (converge) | (fine) |

A 50+ point swing at low selectivity → the benchmark can distinguish methods (it is not
insensitive). Tuned ACORN reaches ~92.6% recall @ ~1622 evals/query at sel=1%; its eval count is
~flat in ef (early-termination-bound), so "tuned" = crank ef for recall at near-constant cost.

### A vs vanilla ACORN — large win (M3 sweep, nclusters=64, cost at matched recall)

| ρ | sel | ACORN-B evals | A evals | ev-ratio | wall-clock ratio |
|---|---|---|---|---|---|
| 1.0 | 0.1% | 3753 | 145 | 25.9× | 22.5× |
| 1.0 | 0.5% | 2152 | 164 | 13.1× | 8.3× |
| 1.0 | 1% | 1622 | 264 | 6.1× | 4.7× |
| 1.0 | 5% | 955 | 628 | 1.5× | 1.6× |
| 0.7 | 1% | 1710 | 189 | 9.0× | 6.4× |

A's exact B&B has recall ≥ ACORN (≈1.0). Win is monotonic in selectivity and **selectivity-
driven** (it also holds at ρ=0 in the sparse regime — partially refuting the pre-registered
*correlation* mechanism: correlation governs recall quality, not the eval win). sel=5% already
misses the ≥2× sub-bar.

### A vs **predicate-aware-entry** ACORN — the win collapses (M3 adversarial, rule #5)

| ρ | sel | vanilla B | **tuned D** | A | A vs **best ACORN** |
|---|---|---|---|---|---|
| 1.0 | 0.1% | 3753 | **203** | 84 | **2.4× — MISS** |
| 1.0 | 0.5% | 2152 | **377** | 164 | **2.3× — MISS** |
| 1.0 | 1% | 1622 | **508** | 264 | **1.9× — MISS** |
| 0.7 | 0.1% | 4009 | 3100 | 80 | 38.8× — WIN |
| 0.7 | 1% | 1769 | 1388 | 214 | 6.5× — WIN |

**Predicate-aware entry cuts ACORN's cost up to ~18× at high correlation** (3753→203 evals),
because seeding the beam at any matching node lands it inside the tight match cluster, finishing
in a few hops. A and D then exploit the *same* structure and converge to within ~2×. The win
**inverts with correlation**: A beats D decisively (6–39×) only at *moderate* ρ=0.7, where D's
sampled seed often lands on a scattered random match and the walk still wanders.

## Decision / Finding

**Qualified NO-GO at the pre-registered ≥5× bar.** Region-pruned IVF does *not* cleanly beat a
properly-tuned ACORN. The headline 6–48× win is against *vanilla* ACORN; once ACORN is given a
predicate-aware entry (a simple, standard enhancement), the gap at high correlation falls to
~2×, below the bar. The pre-registered WIN required ≥5× at sel≤1% for ρ≥0.7 — met at ρ=0.7,
**failed at ρ=1.0** — so the conjunction does not hold.

What *did* hold, honestly:
- A's **exact** recall (1.0) dominates ACORN's ~92% — a quality, not cost, advantage.
- A retains a **6–39× cost edge at moderate correlation (ρ≈0.7) and sel≤1%**, where ACORN's
  predicate-aware seeding is ineffective.
- **At-scale caveat (favours A):** D's seeding leans on predicate-testing ~16k nodes that the
  distance-eval metric counts as free (O(1) predicate vs 128-d distance). At billion-scale a near-
  full predicate scan per query is *not* free; that cost would partially restore A's edge. The
  metric flatters D in exactly the regime where D wins.

## Consequences

- **Do not productionize region-pruned IVF as a general ACORN replacement.** The clean win was an
  artifact of benchmarking an under-equipped incumbent — caught only by the rule-#5 adversarial
  check, which is the central lesson: *a filtered-ANN cost claim is meaningless without a
  predicate-aware-entry baseline.*
- The B&B region-pruning kernel is **correct and exact** (validated vs the oracle) and remains a
  reusable asset; its cost advantage is real but narrow and regime-dependent.
- The honest open question worth a follow-up: at **large n**, where D's per-query predicate scan
  is genuinely costly, does A's edge re-open? That is the only condition under which this bet
  could flip to a WIN, and it is not yet tested.

## Boundaries / not proven

- Single dataset (ogbn-arxiv), n=20k, k=10, 200 queries (per-point noise ~±1%).
- Label-derived correlation as a proxy for production metadata filters.
- ACORN's lite single-layer graph is weak in dense regions (recall non-monotonic at high
  selectivity); the comparison is fair (both use it) but absolute recalls are modest there.
- D's seed-finding is one realization of predicate-aware entry; a smarter one could differ.
- The at-scale (large-n) regime — where the verdict might flip — is unmeasured.

## Next steps

1. **Large-n re-test** (n ≥ 10⁵–10⁶, ≥500 queries): the one condition that could flip the
   verdict (D's predicate scan stops being free). If A's edge re-opens there, revisit.
2. Otherwise, close BET 2 ⊗ BET 4 as a qualified NO-GO and retain the exact B&B kernel as a
   validated asset for the narrow ρ≈0.7 / very-low-selectivity regime.

## Alternatives considered

- **Region-pruning on graph separators** (not IVF) — rejected upstream ([ADR-199]: embedding
  graphs are high-treewidth).
- **Believing the vanilla-ACORN win** — rejected: it does not survive the adversarial check.
