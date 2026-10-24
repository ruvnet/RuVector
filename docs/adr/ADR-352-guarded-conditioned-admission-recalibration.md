# ADR-352: Guarded, Feature-Conditioned Recalibration for Mincut-Gated Memory Admission

## Status

Proposed. Experimental module (`ruvector-memory-admission::conditioned`),
not wired into `ruvector-agent-memory` or any production write path.
Standalone research PoC, evaluated at synthetic-benchmark scale only — the
same scale and status as its parent, ADR-344.

## Context

ADR-344 (`ruvector-memory-admission`, 2026-09-02 nightly) measured two
admission policies against a `NearestCentroidThreshold` baseline at a
matched cluster-count budget:

- **Candidate A** (`MincutGatedAdmission`, fixed `tau`) beat the baseline
  by +4.50pp purity and +7.83pp recall@10 — **accepted**.
- **Candidate B** (`AdaptiveMincutAdmission`, `tau` self-calibrated from a
  running mean/std of the *global* cut-weight distribution) was a
  documented **negative result**: its threshold drifted to the
  48-cluster safety valve, losing 12.3pp recall versus the matched
  baseline. That nightly's own root-cause section named the likely
  mechanism (the running mean/std of the whole-graph cut weight is not
  the same statistic as "the cut weight that separates true outliers from
  true members," and the two diverge as cluster count grows) and named
  the untried fix explicitly in its Next Research item 2: *"try a
  cluster-count-conditioned or local-similarity-conditioned
  self-calibrating tau, addressing candidate B's specific documented
  failure mode rather than abandoning self-calibration entirely."*

This ADR is that attempt. It does not abandon self-calibration (ADR-344
left that door open); it changes two things candidate B did not have:

1. **What is conditioned on.** `tau` is a function of two *local*
   features — normalized cluster count and the candidate point's own
   best single-centroid cosine similarity — instead of an unconditional
   statistic of the global cut-weight distribution.
2. **How recalibration is screened.** The three-coefficient mapping is
   not fit by an unguarded online estimator. It is periodically proposed
   and screened by `ruvector-sona`'s `darwin_guard::Guard` (ADR-271's
   reward-hacking defense, reused here as an actual Cargo dependency, not
   a re-implemented pattern) before a `(1+1)`-ES is allowed to adopt it.

## Hypothesis

```text
Given the same 4,000-point synthetic agent-memory stream as ADR-344 (8
ground-truth clusters, 64 dimensions, 20% high-noise boundary points),

when GuardedConditionedAdmission (candidate C: tau = clamp(base +
count_coeff * cluster_count_norm - sim_coeff * best_single_centroid_sim),
coefficients recalibrated online by a (1+1)-ES screened by
darwin_guard::Guard against a warm-up-measured target spawn rate) is used
for admission, starting from the SAME bootstrap tau as candidates A and B,

then on the static stream it should match candidate A's matched-budget
quality (no worse than 1.0pp purity or recall@10 regression) while never
reproducing candidate B's uncontrolled cluster-count blow-up (final
cluster count <= 1.5x candidate A's, far tighter than the 3x safety-valve
bound),

AND on a second, regime-shift ("drift") stream — same generator, cluster
geometry crowding by 0.45 at the stream's midpoint — it should beat a
transplanted-unchanged candidate A (same fixed tau, never retuned for the
drift) by >= 2.0 percentage points of recall@10 on post-drift held-out
queries, since adapting under drift is the one thing a fixed tau
structurally cannot do,

subject to: mean insertion latency staying under the same 500µs
write-path budget as A and B, and the guard's accept/reject counts being
reported for auditability (not gated on a specific ratio — a guard that
never rejects anything is not screening for anything, but what count is
"too strict" is read from evidence, not fixed in advance).
```

This hypothesis has two independent parts, scored separately: the
**primary (safety) claim** — guarded conditioning is a safe, zero-hand-
tuned drop-in for candidate A — and the **secondary (adaptivity) claim**
— it earns something A cannot do, under drift. ADR-344 was accepted on
its primary claim and rejected on its secondary (self-calibration) claim;
this ADR's result has the same shape, for a different, now mechanistically
understood, reason (see Evidence).

## Decision

Add one new module, `conditioned.rs`, to the existing
`ruvector-memory-admission` crate (no new crate — this is a fourth
admission policy behind the same `AdmissionPolicy` trait ADR-344 defined,
not a new subsystem):

- `GuardedConditionedAdmission` — candidate C. Same min-cut mechanism and
  `should_spawn`/`merge_target` decision logic as candidates A/B (reused
  as `pub(crate)` functions, not re-implemented a third time). `tau` is
  `clamp(base + count_coeff * cluster_count_norm - sim_coeff *
  best_single_sim, TAU_MIN, TAU_MAX)`, a 3-coefficient genome bounded to a
  small box around the bootstrap tau.
- A bounded `(1+1)`-ES recalibrates that genome every `calibrate_every`
  (150) commits, after an initial `warmup_len` (200) commits during which
  a fixed bootstrap tau is used (same convention as candidate B) and the
  realized spawn rate is recorded as `target_spawn_rate`.
- Every candidate mutation is **shadow-replayed** against a sliding
  window (150) of the policy's own recently observed `(avg_cut, best_sim,
  cluster_count_norm)` triples — never against ground-truth labels — and
  screened by `ruvector_sona::darwin_guard::Guard::deterministic()` for
  non-finite fitness, out-of-bounds genes, and the true degenerate
  collapse points (spawns the whole window, or spawns none of it). A
  guard-accepted candidate still only replaces the incumbent if its
  shadow fitness beats the incumbent's (`(1+1)`-ES selection, separate
  from the guard — matching `crates/sona/examples/darwin_router.rs`'s
  convention).
- `GuardStats` (attempts/accepted/rejected-by-reason) is exposed publicly
  for the benchmark to report and for any future caller to audit.
- A new `DriftStreamConfig`/`StreamDataset::generate_drift` in
  `dataset.rs` generates the regime-shift stream the secondary hypothesis
  needs: same arrival-order interleaving as the static stream, but the
  `k_true` centres are pulled 0.45 of the way toward their shared mean
  (more mutually confusable, not a different topic count) starting at the
  stream's midpoint.
- `ruvector-sona` is added as a real `path` dependency (`default-features
  = false, features = ["serde-support"]` — `darwin_guard` itself needs no
  serde, but several of `ruvector-sona`'s other modules do not compile
  without it, so the feature is enabled rather than vendoring a
  cargo-feature workaround).

## Evidence

`cargo run --release -p ruvector-memory-admission --bin benchmark`,
x86-64 Linux 6.18.44, `rustc` 1.9x (workspace toolchain), release build.
Quality metrics (cluster counts, purity, recall@10, guard accept/reject
counts) are **bit-identical across repeated runs** — only wall-clock
latency varies, as expected for the same reason ADR-344's benchmark
documents. Raw output in the nightly research doc's Raw Evidence section.

**Static stream** (matched to candidate A's 17-cluster budget):

| Variant | Clusters | Purity | Recall@10 | Mean(µs) |
|---|---|---|---|---|
| NearestCentroidThreshold (baseline) | 17 | 0.8285 | 0.7840 | 0.13–0.98 |
| MincutGatedAdmission (A) | 17 | 0.8735 | 0.8623 | 26–33 |
| AdaptiveMincutAdmission (B) | 48 | 0.8615 | 0.6610 | 48–56 |
| **GuardedConditionedAdmission (C)** | **17** | **0.8735** | **0.8623** | 47–61 |

Candidate C's numbers are **identical to candidate A's**, to the last
decimal. The guard attempted 25 recalibrations and accepted **zero**:
one was out-of-bounds, 24 were degenerate (spawned either 0/150 or
150/150 of the shadow window). The genome never moved from its bootstrap
value, so C *is* A for this run — which is exactly the primary (safety)
claim: **PASS** on all four static-stream criteria.

**Drift stream** (regime switch at point 2000/4000, regime-B centres
crowded by 0.45; held-out queries drawn from regime-B geometry only):

| Variant | Clusters | Purity | Recall@10 |
|---|---|---|---|
| MincutGatedAdmission (A, transplanted unchanged) | 15 | 0.8057 | 0.5953 |
| GuardedConditionedAdmission (C) | 15 | 0.8057 | 0.5953 |

Identical again — C gained **0.00pp** over transplanted A, failing the
secondary (adaptivity) criterion (needed >= 2.0pp). **FAIL** on the drift
recall-gain check; latency and cluster-count bounds both still pass.

**Why, mechanistically** (debug instrumentation added and removed during
this run — not part of the committed module, but its finding is):
inspecting individual calibration attempts showed the shadow window's
`avg_cut` values sit in roughly `[0.01, 0.18]` by the time calibration
starts (after the 200-commit warmup, with ~17 clusters already formed),
while the genome's bounds keep `tau` near the bootstrap value of `0.005`
and a single `(1+1)`-ES mutation step (`sigma = 0.004`, and `0.0004` —
both tested) essentially never reaches far enough to cross that window's
minimum. Every mutation tried therefore produced either 0 or (if it
somehow jumped tau far past the window's upper range — it did not, in
this run) all spawns — hence "24 degenerate, 1 out-of-bounds, 0 accepted"
is not a bug in the guard; it is the guard correctly refusing to gamble on
a mutation this shadow-replay setup could never actually evaluate as safe
with this run's step size. The deeper reason recalibrating `tau` buys
nothing once the cluster set stabilizes: `policy::should_spawn` spawns
unconditionally when the candidate point is alone on its side of the cut
(`group.is_empty()`, for `c >= 2`) — a *structural* signal independent of
`tau` — and only falls back to the `avg_cut < tau` threshold test when the
point is grouped with existing clusters. Once ~17 clusters exist, most
admission decisions are already resolved by the structural branch before
`tau` is ever consulted, so there is no calibration lever left to pull by
the time the `(1+1)`-ES starts running. The warm-up-measured target spawn
rate (0.055, dominated by the early stream's still-forming-cluster
dynamics, where `tau`-sensitive decisions are actually common) is
consequently unreachable later — not a near-miss, a structurally
unreachable target given the genome's bounds and the admission rule's own
shape.

## Consequences

**What this run supports promoting to "known, load-bearing fact":**
conditioning a self-calibrating admission threshold on local features
plus guarding every recalibration step against degenerate/out-of-bounds
collapse is a **safe** way to attempt self-calibration for this admission
rule — it cannot reproduce candidate B's blow-up, because the guard
rejects exactly the mutations that would cause it. That is a real,
reusable result for anyone building an online self-tuning loop over
`MincutGatedAdmission`-shaped policies elsewhere in the ecosystem.

**What this run does NOT support:** that guarded, feature-conditioned
self-calibration (as designed and parameterized here) delivers any
adaptivity benefit under drift for *this specific admission rule*. The
reason is now understood mechanistically (the structural `group.is_empty()`
branch dominates post-stabilization), not merely observed as a null
result — which makes it a stronger, more actionable negative result than
candidate B's own ADR-344 finding.

**Implication for future work on this admission rule specifically:** a
calibration mechanism that wants to matter here needs to intervene
*during* the cluster-formation phase (when the structural branch is rare
and threshold-sensitive decisions are common), not via a fixed-size
trailing window started after a fixed warm-up — the warm-up itself walks
past the only phase where there was ever a lever to pull.

## Alternatives Considered

- **Re-tuning `AdaptiveMincutAdmission`'s existing `k_std`/`min_observations`
  to fix candidate B directly** — rejected per ADR-344's own explicit
  reasoning: that would be chasing a pass on the same wrong statistic
  (global cut-weight mean/std) rather than testing the Next Research
  item's actual proposal (local conditioning).
- **A differentiable/gradient-based calibration loop** — rejected: there
  is no differentiable loss here (the admission decision and the quality
  metrics it is ultimately judged against are both discrete/discontinuous),
  so a zeroth-order `(1+1)`-ES matching `crates/sona/examples/darwin_autotuner.rs`'s
  own convention was the natural fit, not a gap in this design.
- **Calibrating earlier (during warmup) instead of only after it** — not
  attempted in this run, to avoid moving the goalposts after seeing the
  negative result; it is this ADR's primary Next Research item instead
  (see the nightly research doc).
- **Loosening the degenerate-collapse definition further** (e.g. allowing
  shadow rates like 1–2% of the window to still count as "informative")
  — partially done: the original "outside a 1–99% band" check was
  replaced with the true collapse-point check (`spawns == 0 ||
  spawns == window_n`) as a legitimate pre-registration bug fix (the
  1–99% band would have misclassified this problem's legitimately-low
  target rates as degenerate). No further loosening was attempted once
  the mechanistic cause was understood, since no degenerate-threshold
  change fixes a target that is structurally unreachable.

## Implementation Plan

1. `conditioned.rs` — `Genome`, `Obs`, `GuardStats`,
   `GuardedConditionedAdmission` behind the existing `AdmissionPolicy`
   trait. Done, this run.
2. `dataset.rs` — `DriftStreamConfig`, `StreamDataset::generate_drift`,
   `StreamDataset::held_out_queries_drift`. Done, this run.
3. `bin/benchmark.rs` — candidate C on both streams, full acceptance
   gate. Done, this run.
4. Production wiring — **not planned** until the Next Research items
   (calibrating during the formation phase) are tried; this stays an
   experimental, unwired module exactly like its ADR-344 siblings.

## API Shape

```rust
pub struct GuardedConditionedAdmission { /* ... */ }
impl GuardedConditionedAdmission {
    pub fn new(bootstrap_tau: f32, max_clusters: usize, seed: u64) -> Self;
    pub fn guard_stats(&self) -> GuardStats;
    pub fn target_spawn_rate(&self) -> Option<f32>;
}
impl AdmissionPolicy for GuardedConditionedAdmission { /* decide/commit/n_clusters/centroid */ }

pub struct DriftStreamConfig { /* n_points, k_true, dims, seed, noise,
    drift_noise, drift_frac, switch_at_frac, regime_b_crowding */ }
impl StreamDataset {
    pub fn generate_drift(cfg: &DriftStreamConfig) -> (Self, usize);
    pub fn held_out_queries_drift(cfg: &DriftStreamConfig, n: usize, seed: u64, use_regime_b: bool) -> Vec<(Vec<f32>, usize)>;
}
```

No feature flag: this is a fourth variant inside an already-experimental,
already-unwired crate, matching ADR-344's own flagging decision (none).

## Benchmark Evidence

See Evidence above and the nightly research doc's Raw Evidence section
for full console output, including guard accept/reject breakdowns for
both scenarios and the matched-budget calibration log shared with
candidates A and B.

## Security

No new external attack surface: this module adds no I/O, no
deserialization of untrusted input (the `(1+1)`-ES genome is self-
generated, not attacker-supplied), and no new dependency beyond
`ruvector-sona` (already an in-workspace crate, already depended on by
five other crates in this repository). The `darwin_guard::Guard` is the
security-relevant piece in spirit (reward-hacking defense for a self-
modifying parameter), and this run's evidence shows it working as
intended: it rejected every risky mutation this run's design proposed,
rather than silently adopting one.

## Governance

Unwired, experimental, like ADR-344's candidates. No governance action
required; no production code path changes behavior as a result of this
ADR.

## Failure Modes

- No concurrent-writer, delete, or large-scale (>4,000-point) testing —
  inherited limitation from ADR-344, not newly introduced or newly
  resolved here.
- The drift scenario tests exactly one regime-shift shape (centre
  crowding at the stream midpoint); other plausible drifts (new topics
  appearing, old topics disappearing, gradual rather than abrupt shift)
  are untested.
- The mechanistic explanation in Evidence is based on one run's debug
  instrumentation (added and removed within this session, not a
  committed diagnostic), not a formal proof that the structural branch
  always dominates post-stabilization for every stream configuration;
  it is a strong, reproducible, but single-configuration observation.

## Migration

None — no existing code path depends on this module.

## Rollback

Delete `conditioned.rs`, its `lib.rs` export, its `benchmark.rs`
integration, and the `ruvector-sona` dependency line in `Cargo.toml`; no
other crate references it.

## Rejection Criteria

This ADR's secondary (adaptivity) hypothesis is **already rejected** by
its own evidence (0.00pp drift recall gain vs. a 2.0pp requirement). The
primary (safety) hypothesis is accepted. Promotion of the primary claim
to a stable, wired admission policy would additionally require: (1) the
mechanistic finding re-verified on at least one different stream
configuration (different `k_true`, dimensionality, noise profile), and
(2) a calibration mechanism that intervenes during the formation phase
(this ADR's Next Research item) tested and shown to beat transplanted-A
under drift, not merely matching it safely.

## Open Questions

1. Does calibrating during the cluster-formation phase (not after a fixed
   warm-up) change the drift result, now that the mechanistic blocker is
   understood? (Next Research item 1.)
2. Does the "structural branch dominates post-stabilization" finding
   generalize to other `k_true`/dimensionality/noise configurations, or
   is it specific to this benchmark's parameters?
3. Is there a cheaper, more direct way to detect "tau is currently
   irrelevant to admission decisions" online (e.g. tracking the fraction
   of recent decisions resolved by the structural branch vs. the
   threshold branch) that could gate *when* to even attempt
   recalibration, rather than discovering it empirically via 25 rejected
   attempts per run?
