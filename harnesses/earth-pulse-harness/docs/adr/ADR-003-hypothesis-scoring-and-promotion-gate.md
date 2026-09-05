# ADR-003: Hypothesis Scoring Function and Promotion Gate

## Status

Accepted

## Context

The harness entertains several competing explanations for the 26-second
pulse and its gliding tremors — for example, a stable seafloor
resonance excited by Gulf-of-Guinea swell, a tide-modulated coupling, a
volcanic/hydrothermal contribution, or some combination. We need a
**single, transparent, reproducible** way to score how well each
hypothesis is supported by evidence, and a **hard gate** that decides
when a *child harness* produced by Darwin Mode (ADR-001) may replace its
parent.

Two failure modes must be designed out:

1. **Overfitting to in-sample fit.** A hypothesis can look great on the
   data it was tuned against and fail to predict anything new.
2. **Ungrounded confidence.** A high score must never be achievable by
   inventing citations or by ignoring contradictions.

Scoring lives in `src/score-hypotheses.ts`; the promotion gate is
enforced by `src/validate.ts`. The relevant contracts are
`Hypothesis`, `HypothesisEvidence`, `HypothesisScore`, and
`ValidationReport` in `src/types.ts`.

## Decision

### Weighted discovery-score function

Each hypothesis receives a scalar **discovery score** in `[0,1]`, a
weighted sum of six evidence components (all components in `[0,1]`,
weights sum to 1.0). Default weights live in
`.metaharness/objective.json` and may be overridden per hypothesis via
`Hypothesis.weights`:

| Component                  | Weight | Meaning |
|----------------------------|:------:|---------|
| `sourceStability`          | 0.25   | Consistency of the inferred source location/geometry over time (Gulf of Guinea persistence; beamforming azimuth stability). |
| `environmentalCorrelation` | 0.20   | Strength of the hypothesized ocean/tide/barometric coupling, measured via contrastive pairs (ADR-002). |
| `outOfSamplePrediction`    | 0.20   | How well the hypothesis predicts *held-out* windows it was not fit on. |
| `contradictionSurvival`    | 0.15   | Fraction of logged contradictions the hypothesis survives without special-casing (ADR-004). |
| `mechanisticPlausibility`  | 0.10   | Consistency with established microseism physics (ocean-wave / seafloor coupling); parsimony. |
| `citationGrounding`        | 0.10   | Fraction of the hypothesis's cited claims that map to real source documents in the corpus. |

```
discoveryScore =
    0.25 * sourceStability
  + 0.20 * environmentalCorrelation
  + 0.20 * outOfSamplePrediction
  + 0.15 * contradictionSurvival
  + 0.10 * mechanisticPlausibility
  + 0.10 * citationGrounding
```

`src/score-hypotheses.ts` returns a `HypothesisScore` carrying the
scalar `score`, the per-component `components` breakdown (for audit),
and the list of `contradictions` encountered. The breakdown is
mandatory: a bare score with no component decomposition is treated as
invalid output.

Note that two of the six components are *integrity* components:
`citationGrounding` rewards real references, and `contradictionSurvival`
penalizes hypotheses that only hold by ignoring inconvenient windows.
These weights make integrity part of the objective, not an
afterthought.

### The promotion gate

A child harness emitted by Darwin Mode is promoted over its parent
**only if ALL of the following hold** (evaluated by `src/validate.ts`,
producing a `ValidationReport`). This is an AND gate — any single
failure blocks promotion:

1. **`pulse_detection_f1` improves by >= 3%** versus the parent on the
   held-out evaluation set.
2. **`false_positive_rate` does not increase** (strictly: child FPR
   <= parent FPR). Better recall must not be bought with more false
   alarms.
3. **`held_out_prediction_error` improves by >= 5%** out-of-sample.
   This is the anti-overfitting clause, tied to `outOfSamplePrediction`.
4. **Every cited claim maps to a source document.** Any unmapped
   citation fails the gate outright (and is independently forbidden by
   ADR-001 / ADR-005).
5. **No leakage from test windows into training windows.** Structural
   leakage check in `src/validate.ts`; if `leakageDetected` is true the
   gate fails regardless of every other metric (ADR-004).

In pseudocode:

```
promote =
     (childF1 - parentF1) / parentF1 >= 0.03
  && childFPR <= parentFPR
  && (parentHeldOutErr - childHeldOutErr) / parentHeldOutErr >= 0.05
  && allCitationsMapped(child)
  && !leakageDetected(child)
```

`ValidationReport.passed` is the conjunction above;
`improvementPct`, `heldOutError`, `baselineError`, and
`leakageDetected` are populated for the audit trail, and human-readable
reasons are appended to `notes`.

## Consequences

### Positive

- One transparent number per hypothesis, fully decomposable into audited
  components — no black-box ranking.
- The promotion gate makes "an improvement" a precise, multi-criteria,
  out-of-sample claim rather than a hopeful in-sample number.
- Integrity (citations, contradiction survival) is baked into both the
  score and the gate, so a harness cannot win by cutting corners.
- The weights themselves are an evolvable surface (ADR-001 #4), so the
  objective can be refined under the same gated discipline.

### Negative

- Weight choices are inherently judgment calls; different weightings
  could rank close hypotheses differently. Mitigated by versioning
  weights in `.metaharness/objective.json` and recording them per run.
- The strict AND gate is conservative: a child that is much better on
  F1 but flat on held-out error is rejected. This is intentional, but
  it slows accepted progress.
- Computing `outOfSamplePrediction` and `contradictionSurvival`
  honestly is expensive (requires held-out splits and replaying the
  contradiction log).

## Alternatives considered

1. **Single metric (e.g., detection F1 only).** Rejected: rewards
   overfitting and ignores source stability, environmental causality,
   and integrity.
2. **Learned/auto-tuned scoring weights.** Rejected here: would couple
   the scoring criterion to the data and undermine reproducibility;
   weight changes instead go through the gated Darwin process, logged.
3. **OR-style gate (promote if any metric improves a lot).** Rejected:
   lets a child trade away false-positive rate or out-of-sample
   accuracy for a flashy single-metric win.

## References

- `src/score-hypotheses.ts`, `src/validate.ts`, `src/types.ts`
  (`Hypothesis`, `HypothesisEvidence`, `HypothesisScore`,
  `ValidationReport`)
- `.metaharness/objective.json` (default weights + gate thresholds)
- ADR-001 (forbidden mutations: no unmapped citations, no promotion
  without beating baseline), ADR-002 (contrastive evidence),
  ADR-004 (baselines, leakage, contradiction logs)
