# ADR-350: Value-of-Information Routing Between Baseline and Coherence-Gated Search

**Status**: Proposed (primary hypothesis REJECTED; a secondary finding is promoted to a follow-up recommendation)
**Date**: 2026-09-27
**Relates to**: `ruvector-coherence-hnsw` (2026-06-16 nightly, accepted), ADR-331 (VoI/cost-aware routing, proposed)
**Crate**: `ruvector-voi-router` (new, experimental)

## Context

`ruvector-coherence-hnsw` (accepted 2026-06-16) showed that gating a beam
search's neighbor expansion on "traversal coherence" — whether a candidate
lies roughly toward the query from the search entry point — cuts expansions
and latency while keeping recall within its acceptance band. That benchmark
always used one fixed, distant entry point for every query, so it never
tested what happens when the entry point is *already close* to the query —
a case every real workload with a shared entry point (e.g. one HNSW
layer-0 descent result reused across similar queries) will contain.

ADR-331 (Pandora pattern, proposed) argues for cost-aware / value-of-information
routing between cheap and expensive paths generally. No prior nightly run
connects that idea to `ruvector-coherence-hnsw` specifically. This ADR
records that experiment.

## Hypothesis

> Given a query workload with heterogeneous entry-to-query distance,
> routing each query to plain baseline search when a free signal (the
> entry→query squared L2 distance, `d0` — already computed as beam search's
> first step) predicts the coherence gate has little to prune, and to
> coherence-gated search otherwise, with the distance threshold calibrated
> once from a disjoint calibration query set, achieves mean latency within
> noise of always-on gating while remaining safe on recall — and beats
> always-on gating specifically on the subset of queries near the entry
> point.
>
> Falsified by: routed mean latency materially exceeding always-gated mean
> latency on the evaluation set, or routed recall falling outside 1
> percentage point of always-gated recall.

## Decision

**Do not promote `ruvector-voi-router` as a latency optimization.** The
primary hypothesis is **REJECTED**: across every measured run (see Evidence),
the router's mean evaluation latency exceeded always-on gating's by 5–12%,
because always-on coherence gating turns out to be faster than plain
baseline search across the *entire* measured distance range, not only on
far-entry queries as hypothesized — there is no interior threshold on `d0`
where routing beats picking one fixed policy for the whole workload, so a
router built on this signal cannot add latency value here.

**Do promote the underlying discovery as a follow-up item for
`ruvector-coherence-hnsw`.** While building the evaluation breakdown for the
rejected hypothesis, the same runs surfaced a real, reproducible defect in
the *already-accepted* fixed-threshold coherence gate: on the near-entry
("easy") query subgroup, `CoherenceGatedSearch(threshold=0.50)` recall
collapses to 73.2%, a 21-percentage-point drop from its own far-entry
("hard") subgroup recall (92.2%) and from baseline's near-entry recall
(94.3%) — see Evidence. The 2026-06-16 acceptance benchmark could not have
caught this: it never varied entry-to-query distance within one run. This
ADR does not change `ruvector-coherence-hnsw`'s shipped default (out of this
experiment's scope — that crate's own acceptance thresholds and tests are
unaffected and still pass), but it flags the risk and recommends a specific,
already-validated mitigation: route queries with `d0` below a calibrated
floor to plain baseline search, exactly what `ruvector-voi-router` does.

**Keep `ruvector-voi-router` in the tree as an experimental crate**,
because it is the working, tested implementation of that mitigation, and
because the promotion gate for a *recall-safety* fallback is different from
(and easier to satisfy than) the promotion gate for a *latency* optimization
that this ADR rejects.

## Evidence

Benchmark: `cargo run --release -p ruvector-voi-router --bin benchmark`
(raw output for repeated runs: `docs/research/nightly/2026-09-27-voi-gated-coherence-routing/raw-runs.txt`).

Dataset: 8 clusters × 250 vectors = 2000 vectors, D=32 (identical
construction to `ruvector-coherence-hnsw`'s own accepted benchmark, seed
`0xDEAD_BEEF`); flat navigable-small-world graph, M=16 local + 6 long-jump
neighbors; k=10, ef=80, fixed entry = node 0. 300 calibration queries (seed
`0x0C4B_CA11`) and 400 evaluation queries (seed `0xCAFE_BABE`), fully
disjoint. Each reported latency is pooled over 7 timed passes per query
after 1 untimed warmup pass.

Representative evaluation run:

| Policy | Recall@10 | Mean (µs) |
|---|---|---|
| Baseline | 92.8% | 82.21 |
| AlwaysGated | 90.1% | 72.63 |
| VoiRouted | 92.8% | 80.71 |

| Policy | Group | Recall@10 | Mean (µs) |
|---|---|---|---|
| Baseline | easy (n=44) | 94.3% | 44.83 |
| Baseline | hard (n=356) | 92.6% | 85.70 |
| AlwaysGated | easy | **73.2%** | 2.58 |
| AlwaysGated | hard | 92.2% | 91.56 |
| VoiRouted | easy | 94.3% | 42.65 |
| VoiRouted | hard | 92.6% | 77.30 |

The `AlwaysGated`/easy recall of 73.2% and its near-instant latency (~1.3–2.6µs,
consistent with the search's early-stop condition firing after only a
handful of pops) reproduced identically across 6 independent runs — this is
deterministic given the fixed seeds, not noise. `VoiRouted` recovers
baseline's 94.3% on that subgroup by routing it away from the gate.

The calibration threshold itself is **not** stably reproducible: across
repeated runs with identical seeds, the promoted percentile ranged from the
17th to the 95th, because the bounded search's fitness function is computed
from wall-clock latency, whose run-to-run noise (a few percent) is
comparable to the fitness differences between candidates. This is reported
as a limitation, not hidden — see Consequences. It does not change the
accept/reject verdict: **every** run rejected the primary hypothesis on the
same criterion (routed mean latency exceeding always-gated mean latency by
more than a 2% noise margin), regardless of which threshold calibration
happened to pick.

## Alternatives Considered

1. **A shallow secondary probe as the routing signal** (run a low-`ef`
   `BaselineSearch` pass first, route on its pop count) instead of the free
   `d0` distance. Rejected for tonight's scope: it spends real search work
   to decide whether to spend more search work, which only pays off if the
   probe is much cheaper than the full search it's deciding about — `d0` is
   strictly cheaper (already computed, zero marginal cost) and the
   evaluation shows the discriminating problem isn't signal quality, it's
   that the underlying latency/recall trade-off doesn't reverse direction
   across the distance range for this gate configuration.
2. **Beam-width entropy as the signal** — already falsified for a related
   purpose in the 2026-08-13 nightly (`entropy-adaptive-ann`); not
   re-attempted here without a materially different mechanism, per that
   run's own recommendation.
3. **Redefine the fitness function to force an interior optimum.** Rejected:
   the degenerate "push to an extreme" behavior of the bounded search is
   itself evidence (the recall/latency trade-off is workload-global here,
   not query-conditional on `d0`), and reshaping the objective after seeing
   that would be moving the goalposts rather than reporting the finding.

## Consequences

- No default behavior in the ecosystem changes. `ruvector-coherence-hnsw`'s
  existing acceptance tests, thresholds, and shipped `CoherenceGatedSearch`
  default are untouched and still pass.
- **Recommended follow-up (not performed here):** re-run
  `ruvector-coherence-hnsw`'s own acceptance benchmark with a query mix that
  includes near-entry queries (it currently does not), to decide whether
  that crate's default threshold or its degenerate-direction handling in
  `traversal_coherence` needs hardening independent of routing.
- `ruvector-voi-router` ships as an experimental crate: a tested, documented
  implementation of the "route near-entry queries to baseline" mitigation,
  available for that follow-up to reuse, but not wired into any default
  path.
- The calibration procedure's noise-sensitivity (percentile 17–95 across
  identical-seed runs) means it should not be trusted to pick a
  latency-optimal threshold from a single calibration pass in production;
  any future use needs either many more repetitions or a fitness function
  that isn't dominated by single-digit-percent latency noise.

## Implementation Plan

Not applicable for production adoption — this ADR does not promote a
production change. The experimental crate (`crates/ruvector-voi-router`)
is complete as: `router::VoiRoutedSearch` (implements
`ruvector_coherence_hnsw::search::Searcher`), `calibrate::percentile_threshold`
(pure, unit-tested), and `src/bin/benchmark.rs` (the experiment above).

## API Shape

```rust
pub struct VoiRoutedSearch {
    pub distance_threshold: f32,
    pub gate_threshold: f32,
}

impl VoiRoutedSearch {
    pub fn new(distance_threshold: f32, gate_threshold: f32) -> Self;
    pub fn search_routed(&self, graph: &FlatGraph, query: &[f32], k: usize, ef: usize, entry_id: usize) -> RoutedResult;
}
impl Searcher for VoiRoutedSearch { /* delegates to search_routed */ }
```

## Feature Flags

None. The crate is additive and depends only on `ruvector-coherence-hnsw`
as a path dependency; nothing existing changed behavior.

## Benchmark Evidence

See Evidence above and `docs/research/nightly/2026-09-27-voi-gated-coherence-routing/`
for the full research report, raw multi-run output, and public gist.

## Security

No new attack surface: pure computation over caller-provided float vectors,
no I/O, no unsafe code, no new dependencies beyond the existing
`ruvector-coherence-hnsw` (already in-tree) and workspace `rand`/`rand_distr`.
The routing decision (`d0 <= threshold`) is not attacker-influenced in any
way that matters — at worst a crafted query forces the more expensive path,
which is already the unconditional behavior of `ruvector-coherence-hnsw`'s
shipped default.

## Governance

No governance mechanism changes. No witness chain, proof-gate, or capability
surface is touched.

## Failure Modes

- Empty graph: `search_routed` returns an empty result without panicking
  (tested).
- Degenerate/duplicate calibration distances: `percentile_threshold` panics
  on an empty distance slice or an out-of-range percentile (tested,
  intentional — these are caller bugs, not runtime conditions).
- Noise-dominated calibration (see Consequences): documented, not silently
  swallowed — the benchmark prints every calibration candidate's fitness
  and hard-constraint result.

## Migration

None; nothing existing is changed or replaced.

## Rollback

Remove the `ruvector-voi-router` workspace member; nothing else references
it.

## Rejection Criteria

This ADR's primary hypothesis is already rejected by its own evidence. The
crate remains in-tree under the secondary rationale (documented mitigation
for a real discovered defect). That secondary rationale would itself be
rejected if a follow-up run showed the near-entry recall collapse does not
reproduce on non-synthetic data, or reproduces but at a magnitude too small
to matter in practice.

## Open Questions

1. Does `ruvector-coherence-hnsw`'s near-entry recall collapse reproduce on
   non-synthetic (e.g. real embedding) data, and at what threshold values?
2. Is the collapse specific to the fixed threshold (0.50) or does
   `AdaptiveCoherenceSearch` (not evaluated tonight) also exhibit it?
3. Is there a per-query signal cheaper than a second search pass that
   predicts *this specific* failure mode well enough to route around it
   without a separately-calibrated threshold?
