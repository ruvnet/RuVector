# Why guarding a self-calibration loop isn't enough: a negative result from RuVector's memory-admission research

## Problem

RuVector's `ruvector-memory-admission` crate answers a write-time
question agent-memory systems face on every insertion: should this new
vector merge into an existing cluster, or does it need one of its own? A
prior experiment (ADR-344) showed that gating this decision on a global
minimum cut over the cluster graph beats the standard fixed-threshold
baseline — but that min-cut gate still needs its own threshold, `tau`.
Hand-tuning `tau` per deployment is exactly the kind of parameter nobody
wants to own. A first attempt at self-calibrating it online (tracking a
running mean/std of the cut-weight distribution) failed: the threshold
drifted until the system hit its hard-coded 48-cluster safety valve,
losing 12 points of recall in the process.

## Hypothesis

The original experiment's own diagnosis was specific: tracking a *global*
statistic of the cut-weight distribution is the wrong thing to track,
because that distribution's shape changes as the cluster count grows,
independent of whether the system is actually miscalibrated. The fix it
proposed but didn't try: condition the threshold on *local* features
(how many clusters exist, how similar the incoming point is to its best
match) instead.

This run built that fix, plus a second, independent safeguard the first
attempt didn't have: every proposed recalibration is screened by a
reward-hacking guard — `ruvector-sona`'s `darwin_guard`, originally built
to keep an evolutionary search over SONA's own learning-rate config from
gaming its own fitness function — before a small `(1+1)` evolutionary
search is allowed to adopt it. The guard lives outside the parameters it
governs, cannot be edited by what it screens, and rejects non-finite
fitness, out-of-bounds parameters, and degenerate collapse (a mutation
that would make the policy spawn a new cluster for literally every point,
or for none of them).

The pre-registered test had two independent parts. First, a safety
claim: does the guarded, conditioned version at least match the original
min-cut policy, without the blow-up? Second, an adaptivity claim — the
actual point of self-calibration — does it beat a policy with a fixed
threshold once the underlying data distribution drifts?

## What happened

The safety claim passed cleanly: across a 4,000-point synthetic stream,
the guarded policy produced numbers **identical to the last decimal**
to the fixed-threshold policy it was built to replace — 17 clusters,
0.8735 purity, 0.8623 recall@10, both runs. The guard attempted 25
recalibrations over the course of the stream and accepted none of them:
one proposal fell outside its allowed parameter bounds, and the other 24
would have caused the policy to spawn either on every point in a
look-back window or none of them — exactly the collapse the guard exists
to catch.

The adaptivity claim failed. A second benchmark stream, identical to the
first except that the underlying clusters are pulled closer together
(more mutually confusable) starting halfway through, is where a
fixed-threshold policy should start losing ground to one that can adapt.
It didn't: the guarded policy matched the fixed one exactly again, a
0.00-percentage-point gain against a pre-registered requirement of at
least 2.

## Why, specifically

Instrumentation added to trace individual recalibration attempts (and
removed again before this code shipped — it was a diagnostic step, not
a feature) found the mechanism precisely. By the time the policy's
200-insertion warm-up period ends and recalibration starts trying, the
window of recent cut-weight observations it shadow-replays candidates
against already sits far above the bootstrap threshold — one to two
orders of magnitude, in the runs measured. No small evolutionary step
can close that gap without passing straight through the "spawns
everything" collapse zone the guard correctly flags.

But there's a second, more structural reason underneath that one: the
admission rule itself has a branch that bypasses the threshold entirely.
If an incoming point ends up completely isolated on its own side of the
minimum cut — structurally distinct from every existing cluster, not
just numerically dissimilar — the policy spawns a new cluster
unconditionally, with no reference to `tau` at all. Once enough clusters
exist to make that branch common (which happens well within the
warm-up period in this benchmark), most admission decisions never
consult the threshold in the first place. There's no lever left for any
calibration mechanism — a better statistic, a cleverer online estimator,
or this run's guarded evolutionary search — to pull.

## Why this is useful anyway

A failed self-calibration attempt with a located, mechanistic cause is
worth more than a vague one. The original experiment's diagnosis implied
a better statistic might fix candidate B's drift. This run's evidence
says the real constraint is earlier and harder: by the time any
after-warm-up calibration mechanism gets to run, the variable it's
trying to calibrate has already stopped being decisive. That reframes
the actual next experiment — not "try a smarter estimator" but "try
calibrating during cluster formation, while the threshold still matters,"
which is now this project's next research item instead of a repeat of
this one.

It's also a clean, positive demonstration of the guard doing its job.
Every one of the 25 proposed changes this run tried was either genuinely
unsafe or evaluated against evidence too thin to justify adopting it, and
the guard rejected all of them — rather than the alternative failure
mode, where a self-tuning loop silently adopts a change that looks fine
on paper and blows up in production weeks later. Zero accepted mutations
and zero regressions is the correct, boring outcome when there's nothing
safe worth changing.

## Limitations

This result is specific to one admission rule's internal structure (the
`group.is_empty()` branch) and one drift shape (clusters becoming more
confusable, not new clusters appearing or old ones vanishing). It doesn't
claim that no guarded self-calibration mechanism could ever add
adaptivity to this system — only that recalibrating after a fixed
warm-up, against a trailing window of recent decisions, cannot, for a
reason now pinned down precisely enough to design the next attempt
around.

## Production relevance

Nothing here ships to a production write path yet — this stays an
unwired, experimental module, same as its predecessor. What is reusable
today: the guarded-recalibration pattern itself (now exercised outside
its original crate for the first time), the drift-benchmarking
infrastructure, and the specific, falsified claim, so the next person
who reaches for "just add a smarter self-calibrating threshold" to this
particular admission rule can skip straight to why that alone won't be
enough.

## References

- RuVector `ruvector-memory-admission` crate and ADR-344 (prior
  experiment establishing the min-cut-gated admission policy and its
  rejected self-calibration attempt).
- ADR-352 (this experiment).
- `ruvector-sona`'s `darwin_guard` module (ADR-271) — the reward-hacking
  defense reused here.
- Lu et al., "Autonomous Concept Drift Threshold Determination," AAAI
  2026 — on adaptive thresholds generally beating fixed ones, in a
  different setting (drift detectors, not clustering admission) than
  this run's negative result.
- Skalse et al. (2022) on the theoretical unavoidability of reward
  hacking for non-constant proxies — the broader argument for guarding
  self-modifying parameters that this run's guard design follows.
