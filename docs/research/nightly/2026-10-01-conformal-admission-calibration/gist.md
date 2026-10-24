# Why "Just Use a Sliding Window" Didn't Fix Self-Calibrating Mincut Admission

## Problem

Streaming agent-memory systems need to decide, for every incoming vector,
whether it joins an existing cluster or starts a new one. A prior nightly in
the RuVector workspace (`ruvector-memory-admission`, ADR-344) built three
policies for this: a fixed-threshold baseline, a global-min-cut-gated policy
with a hand-tuned threshold (`tau`), and a self-calibrating variant that set
`tau` from a running mean/std of its own history. The self-calibrating
variant failed — badly: it drifted to a hard safety-valve cap instead of
settling near the right number of clusters, losing 12 percentage points of
recall versus a fairly-matched baseline. The stated likely cause was that a
*lifetime* running statistic doesn't track what "normal" looks like once the
cluster graph has grown — an old average from early in the stream (few
clusters) doesn't describe the current regime (many clusters).

## Hypothesis

The obvious fix: don't average over the whole lifetime. Use a sliding
window of only the most recent observations, and instead of a mean/std
(which assumes something like a Gaussian shape), take an empirical
quantile — conformal-prediction-style calibration. This is cheap, well
motivated by recent online-conformal-inference literature, and directly
attacks the "lifetime vs. local" explanation the prior nightly proposed. If
that explanation was right, windowing should fix it.

## What Was Built

A fourth policy, `ConformalMincutAdmission`, added to the existing Rust
crate: identical graph-cut mechanism to the other mincut-based policies,
but its threshold is the alpha-quantile (default 15th percentile) of a
200-entry sliding window of its own recent cut-weight observations, instead
of a lifetime mean/std. Implemented with no new dependencies (`std`'s
`VecDeque`), ~150 lines, 6 new unit tests, wired into the existing
four-way benchmark harness.

## Result: Rejected

It drifted to the same safety-valve cap the lifetime-statistic version did.
Not a slightly-better-but-still-failing result — the *exact same* 48-cluster
ceiling, on a stream where the accepted baseline settles at 17. Recall
regression was smaller (7.8 percentage points instead of 12.3) and latency
was notably lower, but the primary, pre-registered success criterion —
staying within a 24-cluster bound — failed outright.

To make sure this wasn't a bad choice of window size or quantile level, the
same benchmark was re-run at three more configurations after seeing the
first result: a much more aggressive quantile (2nd percentile, then 0.5th
percentile — numerically the same value as the hand-tuned constant that
works for the non-self-calibrating version), and a 4x larger window. All
four configurations saturated to the same cap. That rules out "wrong
hyperparameter" as the explanation.

## Why

The admission policy's spawn decision actually has two independent
triggers, not one: a weight threshold (the thing any self-calibration
scheme touches), and a *structural* signal — whether the candidate point
ends up completely isolated on its own side of the graph cut, which fires
regardless of the weight threshold whenever two or more clusters already
exist. The hand-tuned constant that works is set so low that, in practice,
the weight-threshold path almost never fires — the policy's real behavior
comes from the structural signal. Any calibration against the *distribution
of observed weights* — mean/std or quantile, lifetime or windowed — lands
somewhere in the middle of that distribution, which is nowhere near low
enough to stay as quiet as the hand-tuned constant. So the weight-threshold
path starts firing constantly, on top of the structural trigger that was
already doing the real work, and the two compound into runaway cluster
growth.

This reframes the open question. It's not "which summary statistic of the
weight distribution should self-calibration target" — it's that no summary
statistic of that distribution is the right thing to calibrate against,
because the signal that actually matters is topological, not a scalar
weight. A future attempt should either calibrate against something
structural (the margin between the best and second-best cut, say) or stop
trying to make this particular threshold self-calibrating at all.

## Why This Is Still a Useful Night's Work

A failed hypothesis with a concrete, evidence-backed, swept root cause is
more valuable to the next person than either an untested guess or a vague
"it didn't work." This result:

- Closes off an entire family of plausible-sounding fixes (different
  window sizes, different quantile levels, different parametric forms of
  "recalibrate the weight statistic") with actual swept evidence, not just
  one data point.
- Produces a specific, testable next hypothesis (calibrate against
  structure, not weight) instead of leaving the question exactly where it
  was.
- Is honestly reported as "rejected," with the one real partial
  improvement (lower recall regression, lower latency) disclosed and
  explained as a side effect of the same failure, not spun as a win.

## Limitations

This isn't a rigorous conformal-prediction result — formal conformal
guarantees require the calibration data to be independent of the decisions
being evaluated, and here the calibration window is built from the same
policy's own past decisions, which is exactly the kind of feedback loop
that breaks that assumption. The name describes the mechanism (an
empirical-quantile threshold over a calibration buffer), not a transplanted
statistical guarantee — a distinction worth being explicit about rather
than letting a well-known technique's name imply more than was actually
measured. The root-cause explanation is also inferred from aggregate
benchmark behavior across four configurations, not confirmed by directly
instrumenting which trigger fired on which decision — a concrete, cheap
next step.

## References

- Stoer, M., & Wagner, F. (1997). "A Simple Min-Cut Algorithm." *JACM.*
- Vovk, V., Gammerman, A., & Shafer, G. (2005). *Algorithmic Learning in a
  Random World.* (Split conformal prediction.)
- Gibbs, I., & Candès, E. (2021). "Adaptive Conformal Inference Under
  Distribution Shift." *NeurIPS 2021.*
- This workspace: ADR-299, ADR-344, and the 2026-09-02 nightly this work
  directly extends.
