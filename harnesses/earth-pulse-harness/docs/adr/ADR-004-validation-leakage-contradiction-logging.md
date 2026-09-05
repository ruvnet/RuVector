# ADR-004: Validation — Leakage Control, Contradiction Logging, and the Discovery Ladder

## Status

Accepted

## Context

A scientific harness can fool itself in three classic ways: it can
**leak** the answer from test data into training data, it can manufacture
a **fake correlation** by only ever evaluating on convenient conditions,
and it can quietly **bury contradictions** that would falsify its
favored hypothesis. The 26-second pulse is especially prone to fake
correlation because both the pulse and ocean swell have strong seasonal
structure — almost anything correlates with almost anything else if you
only look at stormy months.

We therefore need a validator that (a) constructs honest held-out splits
across *opposite* ocean regimes, (b) requires beating naive baselines
*out-of-sample*, and (c) logs every contradiction it finds so the
science stays auditable. Bruland & Hadziioannou (2023) report gliding
tremors associated with the 26 s microseism; a credible harness must be
able to say honestly when and how often those tremors *fail* to follow
the favored narrative.

Validation logic lives in `src/validate.ts` and consumes the scores
from `src/score-hypotheses.ts`. The output contract is
`ValidationReport` in `src/types.ts`.

## Decision

### Held-out splits across opposite regimes

The validator holds out **storm weeks AND calm-sea weeks** as separate
test partitions, never training/fitting on either:

- Holding out *storm weeks* prevents the harness from learning only the
  high-amplitude regime where the pulse is easy.
- Holding out *calm-sea weeks* is the key anti-fake-correlation control:
  if a hypothesis claims swell drives the pulse, it must still predict
  correctly when the sea is calm. A correlation that only exists in
  storms is exactly the artifact we are guarding against.

Test windows and training windows are kept structurally disjoint, and
`src/validate.ts` sets `leakageDetected = true` if any feature fitted on
a training window was derived using information from a test window
(this hard-fails the promotion gate, ADR-003).

### Baselines that must be beaten out-of-sample

No hypothesis or child harness is credited unless it beats **all** of
these naive baselines on the held-out partitions:

- **Seasonal average** — predict the pulse from time-of-year alone.
- **Swell-only** — predict from swell height/period/direction alone.
- **Tide-only** — predict from tide phase alone.

If a hypothesis cannot beat "it's just the season" or "it's just the
swell" out-of-sample, it has explained nothing. `ValidationReport`
records `baselineError` (best baseline) and `heldOutError` (the
hypothesis), with `improvementPct` between them.

### Contradiction logging

The validator emits a **contradiction log** — explicit, auditable
statements of where the data disagrees with the favored hypothesis, for
example:

> "Gliding tremors appear during 18% of calm windows" (illustrative
> target/hypothesis, not a measured finding)

Each contradiction is attached to the relevant `HypothesisScore` and
feeds the `contradictionSurvival` component (ADR-003). Contradictions
are never silently dropped; suppressing a contradiction is a validation
failure. This is what keeps the science honest: the harness must
publish its own counter-evidence.

### The discovery ladder

Hypotheses are assessed against a six-level **discovery ladder**. Each
level has an explicit acceptance test that must pass on held-out data
before the next level is attempted. Numbers below are *targets /
acceptance thresholds*, not measured results.

| Level | Claim | Acceptance test |
|------:|-------|-----------------|
| **1 — Detection** | A stable ~26 s pulse exists and is reliably detected. | `pulse_detection_f1` beats seasonal baseline out-of-sample; FPR within budget. |
| **2 — Source stability** | The source is spatially stable (Gulf of Guinea). | Beamforming azimuth/coherence stable across held-out months; `sourceStability` above threshold. |
| **3 — Ocean coupling** | The pulse couples to ocean swell/tide. | Beats swell-only and tide-only baselines out-of-sample on *both* storm and calm partitions (no fake correlation). |
| **4 — Resonance mechanism** | A specific seafloor-resonance mechanism explains the period. | Mechanistic prediction matches held-out spectra better than a non-mechanistic fit; consistent with microseism theory. |
| **5 — Glide explanation** | The mechanism explains the gliding tremors (Bruland & Hadziioannou 2023). | Predicts glide-slope sign/magnitude on held-out glide events better than chance; contradictions logged and survived. |
| **6 — Causal ranking** | Competing causes are ranked by causal contribution. | Contrastive-pair causality estimates (ADR-002) are stable and beat baselines; ranking robust to leakage and holdout choice. |

A hypothesis may only *claim* a level if every lower level's acceptance
test passes on held-out data. Reports state the highest honestly
achieved level, plus the contradiction log for that level.

## Consequences

### Positive

- The calm-sea holdout directly kills the most likely false-positive
  story (storm-only correlation).
- Forcing out-of-sample wins over seasonal/swell/tide baselines means a
  "discovery" is a real lift over the obvious explanation.
- The contradiction log makes the harness self-auditing and gives
  reviewers something concrete to challenge.
- The discovery ladder turns vague "we found a mechanism" claims into a
  graded, testable progression.

### Negative

- Honest holdouts across opposite regimes shrink the usable training
  data, especially for rare calm/glide windows.
- Maintaining and replaying the contradiction log adds bookkeeping cost.
- The ladder is strict: a genuinely promising Level-5 idea is blocked
  until Levels 1-4 pass on held-out data, which can feel slow.

## Alternatives considered

1. **Random k-fold split.** Rejected: seasonal autocorrelation means
   random folds leak regime information; the storm/calm split is the
   point.
2. **No explicit baselines.** Rejected: without seasonal/swell/tide
   baselines, ordinary seasonality masquerades as discovery.
3. **Track only confirming evidence.** Rejected: this is how harnesses
   fool themselves; contradiction logging is mandatory and scored
   (ADR-003 `contradictionSurvival`).

## References

- `src/validate.ts`, `src/score-hypotheses.ts`, `src/types.ts`
  (`ValidationReport`, `HypothesisScore`)
- Bruland, A. & Hadziioannou, C. (2023). Gliding tremors associated with
  the 26 s microseism.
- Microseism theory: ocean-wave / seafloor coupling.
- ADR-003 (promotion gate consumes leakage + baseline results),
  ADR-002 (contrastive causality for Level 6)
