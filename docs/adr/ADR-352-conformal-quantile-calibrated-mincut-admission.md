# ADR-352: Conformal-Style Quantile-Calibrated Mincut Admission (Rejected)

## Status

**Rejected** (experiment complete, hypothesis falsified). Experimental
addition to the existing research-status crate `ruvector-memory-admission`
(ADR-344), not wired into any production write path. Standalone research
PoC, evaluated at synthetic-benchmark scale only. Recorded as a rejected
candidate, not deleted, per the nightly process's rule that a falsified
path with good evidence is a successful (not a failed) nightly run.

## Context

ADR-344 (2026-09-02) established `ruvector-memory-admission`: three online
cluster-admission policies for streaming agent memory, behind one
`AdmissionPolicy` trait. Its candidate A (`MincutGatedAdmission`, global
min-cut gated on a fixed `tau`) was accepted, beating a matched-budget
baseline on purity and recall@10. Its candidate B
(`AdaptiveMincutAdmission`, `tau` set from a lifetime running mean/std of
observed cut weights) was a documented negative result: it drifted to the
48-cluster safety-valve cap and lost 12.3 percentage points of recall@10.
ADR-344's own analysis proposed a specific next step: condition the
self-calibrating statistic on cluster count or on local (not whole-graph)
similarity, because a *lifetime* statistic doesn't track the *local*
admission-relevant threshold once the cluster graph grows.

This ADR implements and evaluates the cheapest concrete instantiation of
"condition on recency instead of lifetime": replace the lifetime mean/std
with an empirical quantile over a fixed-size *sliding window* of recently
observed cut weights — conformal-prediction-style quantile calibration.

## Hypothesis

```text
Given the identical streaming workload and matched-budget baseline from
ADR-344 (4,000 synthetic agent-memory vectors, 8 ground-truth clusters, 64
dimensions, 20% drift noise, interleaved arrival order),

when cluster admission uses ConformalMincutAdmission (tau_t set to the
empirical alpha-quantile, alpha = 0.15, of a sliding window of the last
200 observed average-cut weights, instead of a fixed constant or a
lifetime running mean-minus-std),

then the final cluster count should stay within the 3x-K_true acceptance
bound (<= 24) without external calibration, specifically not repeating
candidate B's runaway drift to the 48-cluster safety valve,

subject to: recall@10 regression vs. the matched baseline staying within
2 percentage points (the same tolerance applied to candidate B), and mean
insertion latency staying under the same 500us write-path budget.

alpha and window were fixed before running the benchmark, not tuned
against the evaluation metrics.
```

## Decision

Implement `ConformalMincutAdmission` as a fourth policy in the existing
`ruvector-memory-admission` crate (no new crate — a direct, scoped
extension of ADR-344's PoC):

- Identical graph construction and Stoer-Wagner cut mechanism to candidates
  A and B (`src/mincut.rs`, unchanged).
- `tau_t` computed as the linear-interpolated empirical `alpha`-quantile of
  a `VecDeque<f32>` sliding window (`src/policy.rs::ConformalMincutAdmission`),
  bootstrapped from a fixed constant until the window fills to
  `min(window, 10)` observations.
- Wired into the existing four-way benchmark harness
  (`src/bin/benchmark.rs`) with its own acceptance-criteria block, same
  shape and thresholds as candidate B's.
- 6 new unit tests plus one new integration test, extending (not replacing)
  the existing test suites for this crate.

**Result: REJECTED.** Candidate C saturates the same 48-cluster safety
valve as candidate B, at the pre-registered configuration and at three
additional configurations swept afterward specifically to probe the
mechanism (`alpha=0.02`, `alpha=0.005`, `window=800`) — all four saturate
identically. The windowing fix does not work, and the evidence rules out
"wrong alpha/window" as the explanation (see Evidence).

## Evidence

Matched-budget run (baseline and candidate A at 17 clusters, reference
only — unchanged from ADR-344, re-measured here to confirm reproducibility
on this build):

| Variant | Clusters | Purity | Recall@10 | Mean insert (µs) |
|---|---|---|---|---|
| NearestCentroidThreshold (calibrated) | 17 | 0.8285 | 0.7840 | 0.01 |
| MincutGatedAdmission (A, reference) | 17 | 0.8735 | 0.8623 | 17.18 |
| AdaptiveMincutAdmission (B, parent negative result) | 48 | 0.8615 | 0.6610 | 36.18 |
| **ConformalMincutAdmission (C, this ADR)** | **48** | **0.8588** | **0.7063** | **7.84** |

**Candidate C: FAIL on 2 of 3 pre-registered criteria.** Final cluster
count (48) exceeds the 24-cluster bound — the primary falsification
condition — and recall@10 regression (7.77pp) exceeds the 2pp tolerance.
Mean latency (7.84us) passes, but is a side effect of early saturation to
the cheap nearest-centroid fallback path, not an independent efficiency
gain (see nightly doc).

**Root cause, evidence-backed** (full derivation in the nightly doc):
`should_spawn` has two independent triggers — a weight threshold
(`avg_cut < tau`, the only one self-calibration touches) and a structural
signal (`group.is_empty()`, independent of `tau`). Candidate A's `tau =
0.005` is low enough that the weight-threshold path almost never fires;
A's effectiveness comes mostly from the structural signal. Self-calibration
(B's lifetime mean/std or C's windowed quantile alike) necessarily targets
a value drawn from the *observed cut-weight distribution*, which — even at
`alpha=0.005`, numerically matching A's constant — never realizes values as
low as A's hand-picked threshold in this workload. The weight-threshold
path therefore fires far more often under B or C than under A, compounding
with the always-present structural trigger into runaway growth. This is
corroborated, not merely asserted: all four swept configurations
(including `alpha=0.005`, which should reproduce A's behavior under the
"wrong summary statistic" theory and does not) saturate identically.

**Correctness**: 25/25 tests pass (`cargo test --release -p
ruvector-memory-admission`, up from 20/20 in ADR-344 — 6 new unit tests on
`ConformalMincutAdmission` plus 1 new integration test, no existing test
modified). `cargo clippy --release -p ruvector-memory-admission -- -D
warnings`: clean.

## Consequences

**Positive**:
- Closes off an entire family of "recalibrate the weight-distribution
  statistic" self-calibration attempts for this policy with concrete,
  swept evidence, saving a future nightly from re-attempting a plausible-
  sounding variant (different window size, different alpha, different
  parametric form) blind.
- Produces a specific, falsifiable, more useful open question than ADR-344
  left: the blocker is not *which* statistic of the cut-weight distribution
  to calibrate, but that *no* such statistic represents the structural
  signal the working candidate actually relies on.
- Extends the existing crate's test coverage (6 new unit tests, 1 new
  integration test) without weakening or removing any existing test.

**Negative / costs**:
- No self-calibrating variant of this admission policy is closer to
  production-viable than before this ADR; candidate A's hand-tuned `tau`
  remains the only path with a production trajectory, unchanged from
  ADR-344.
- The root-cause mechanism is inferred from aggregate benchmark behavior
  across four configurations, not from direct per-decision instrumentation
  of which spawn trigger fired — named as the first item of future work,
  not silently assumed proven beyond the evidence presented.

## Alternatives

- **Genuine online/adaptive conformal inference** (Gibbs & Candès, 2021 —
  a feedback-aware quantile-tracking update targeting a fixed miscoverage
  rate under distribution shift, rather than a static sliding window): the
  more faithful approach for this non-exchangeable, feedback-driven
  setting. Not attempted in this ADR — scope was deliberately narrowed to
  "does windowing alone fix candidate B" as a cheaper question to answer
  first, now answered (no). Named as follow-up work, not substituted after
  the negative result.
- **Condition on cluster count directly** (e.g., normalize `avg_cut` by a
  function of `C` before thresholding) or **condition on local similarity**
  (candidate's own best-single-centroid cosine): ADR-344's other named
  directions. Not attempted here, to keep this experiment isolated to one
  variable (recency via windowing).
- **Calibrate against the structural signal directly** (e.g., the margin
  between the realized min cut and the next-best cut): suggested by this
  ADR's own root-cause analysis as the more promising direction; not
  implemented here, named as future work.

## Implementation Plan

Not applicable — this candidate is rejected, not scheduled for promotion.
If a future nightly pursues one of the Alternatives above against this same
crate, it should start from the root-cause analysis in this ADR and the
2026-10-01 nightly doc rather than re-deriving it.

## API Shape

```rust
pub struct ConformalMincutAdmission { /* alpha, window, max_clusters, bootstrap_tau, ... */ }
impl AdmissionPolicy for ConformalMincutAdmission { /* decide / commit / n_clusters / centroid */ }
```

Same `AdmissionPolicy` trait as ADR-344, no new production API surface
proposed — this candidate is rejected, so no API shape is proposed for
promotion.

## Feature Flags

None. Not wired into any existing crate beyond the experimental
`ruvector-memory-admission` crate itself.

## Benchmark Evidence

`cargo run --release -p ruvector-memory-admission --bin benchmark` (default
env for the pre-registered result; `CONFORMAL_ALPHA` / `CONFORMAL_WINDOW`
overrides for the exploratory follow-up sweep). Full raw output for the
pre-registered run, a determinism-check second run, and the three
exploratory sweep configurations are preserved in
`docs/research/nightly/2026-10-01-conformal-admission-calibration/README.md`
under "Raw Evidence."

## Security

No new surface versus ADR-344: no untrusted deserialization, no network or
filesystem I/O beyond benchmark diagnostics, no secrets. The sliding-window
calibration buffer is bounded at `window` entries by construction (a
`VecDeque` with an explicit pop-on-overflow), so it introduces no new
unbounded-growth risk.

## Governance

Research PoC only; additive to an existing non-production crate. No
governance action required. This ADR narrows (removes one avenue from) the
production-readiness case for self-calibrating admission in this crate
rather than expanding it.

## Failure Modes

See the nightly doc's "Failure Modes" section: the primary measured failure
(safety-valve saturation, root-caused above), cold-start bootstrap
(unchanged from A/B, unit-tested), a window-size degeneracy caught and
fixed during implementation (`min_observations` capped at `window` so a
small window can still exit bootstrap), and the same disclosed-not-silently-
assumed gaps ADR-344 already named (concurrency, deletes, scale,
cross-platform determinism).

## Migration

None; no existing behavior changes. This candidate is rejected.

## Rollback

None required beyond what ADR-344 already describes: deleting
`crates/ruvector-memory-admission` and its workspace-member entry removes
this ADR's footprint along with ADR-344's.

## Rejection Criteria

Already met — this is the record of that rejection, not a forward-looking
criterion. Per the nightly process, the candidate and its code are
retained (not deleted) specifically so a future nightly does not
re-implement and re-measure the same negative result.

## Open Questions

1. Does direct per-decision instrumentation (counting which of
   `should_spawn`'s two trigger paths fired, per policy, over the same
   stream) confirm the root-cause mechanism proposed here, or reveal a
   different explanation consistent with the same aggregate numbers?
2. Does calibrating against a structural quantity (cut-weight margin, or
   best-single-centroid similarity) rather than the cut-weight distribution
   succeed where both weight-distribution-based attempts (B and C) failed?
3. Does a genuine feedback-aware online/adaptive conformal update (rather
   than a static sliding window) change this result?
4. ADR-344's four open questions (real-corpus replication, O(C^3) cost
   ceiling, cross-platform determinism) remain open and unaffected by this
   ADR.
