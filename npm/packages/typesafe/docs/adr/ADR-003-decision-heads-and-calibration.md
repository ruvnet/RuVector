# ADR-003: Decision heads and calibrated confidence

## Status
Proposed

## Date
2026-09-21

## Context

Three question types must be answered from embeddings:

- `choice`: one of up to 255 options, each described by a string or
  `{what, not_for, examples}`; answer is `{choice, probabilities (sum 1), confidence}`.
- `score`: an ordinal legend (`["Calm", "Irritated", "Angry"]`); answer is
  `{score, legend, probabilities}`.
- `noul`: a 0–1 value for a predicate ("the sender needs a response soon").

What the evidence says:

- Zero-shot cosine similarity between `state` and option descriptions reaches
  roughly 35–48 % (1-shot) to 75–87 % (5–10-shot) on banking77 / CLINC150 /
  HWU64 — below Jev's measured 85–90 %. A linear probe on frozen embeddings
  outranks cosine-to-prototype, which outranks kNN, in an intent-classification
  probing study; SetFit-style contrastive fine-tuning with ~8 examples/class is
  competitive with full-data RoBERTa-large (arXiv:2209.11055).
- Jev's `confidence` is saturated near 1.0 with ECE 0.07 on our test split. The
  likely cause is structural, not a missing calibration layer: forcing
  probabilities to sum to 1 over the options asserts they are exhaustive, while
  `not_for` describes an open world. Whatever the model, a state that fits no
  option still gets a confident answer.
- Jev's `noul` for urgency scored 60 % against a 70 % majority baseline. A raw
  similarity has no decision threshold and no base-rate correction; that is the
  expected outcome, not a tuning problem.

## Decision

### `choice`

1. **Head, in order of available labels.**
   - *No labeled examples for this question:* nearest-prototype. Each option's
     prototype is the mean of the embeddings of its `what`/description and its
     `examples`; `not_for` text becomes a **hard-negative prototype** for that
     option, and the option's score is `sim(state, proto) − λ·sim(state, not_for)`.
   - *≥ 4 labeled examples per option (or the bank crosses that after
     ADR-004's loop admits examples):* a **linear probe** (multinomial logistic
     regression on frozen embeddings, L2-regularized, trained in the Rust core in
     milliseconds), with `not_for` examples as negatives for that class. The
     engine switches heads per question automatically and records which head
     answered in the receipt.
   - *v2:* SetFit contrastive fine-tune per deployment once labels accumulate
     (with negative sampling at high class counts), behind the same promotion
     gate as everything else.

2. **An explicit `none` mass.** Probabilities are computed over the options
   *plus* an abstain bucket whose logit is the best `not_for` match and the
   distance to the nearest prototype. The reported `probabilities` are the
   option masses renormalized (to keep Jev's sum-to-1 contract), and the
   abstain mass is exposed as `abstain` alongside `confidence`. This is the
   design-level fix for saturated confidence: an out-of-scope `state` produces
   low `confidence` and high `abstain`, not a confident wrong answer.

3. **Calibration = temperature scaling per question**, fitted on a held-out
   calibration slice disjoint from the head's training examples. Temperature
   never changes the argmax; it only makes `confidence` honest. `confidence` is
   the calibrated top-1 probability, so it is comparable across deployments and
   its ECE is reportable.

4. **Conformal candidate sets (v2, additive).** Split-conformal RAPS over the
   calibrated probabilities yields a set of options with a coverage guarantee
   (e.g. 90 %). Exposed as `candidates`, never replacing `choice`. Valuable at
   high option counts, where a single argmax hides a near-tie.

### `score`

Legend buckets are ordinal. v1 treats them as classes through the same
`choice` machinery (honest fallback, reuses the head and calibration). v2
replaces the head with CORN (rank-consistent ordinal regression,
arXiv:2111.08851) so adjacent-bucket confusions cost less than distant ones
and boundary probabilities are calibrated. `score` is the expected bucket index
under the calibrated distribution rounded to the nearest bucket, with
`probabilities` per bucket.

### `noul`

`noul` is a **logistic head** on the embedding, trained on labeled
positive/negative examples for the predicate, with Platt calibration and a
class-balanced loss. With zero labels it falls back to a similarity-to-predicate
score that is *flagged uncalibrated* in the receipt and the response
(`calibrated: false`). Raw similarity is never reported as a probability.

### What every answer carries

```
{ ..., "confidence": 0.81, "abstain": 0.09, "calibrated": true,
  "head": "linear-probe", "model": "bge-small-en-v1.5@<hash>", "temperature": 1.7 }
```

The extra fields are additive to Jev's shape; a Jev-shape-only mode strips them.

## Consequences

- Reaching Jev-level accuracy requires labeled examples; the package must make
  that path first-class (`typesafe train`, and the loop in ADR-004) and must not
  imply zero-label parity.
- Calibration needs a held-out slice, which is the scarce resource in a
  few-example deployment. The engine reserves a fixed fraction of admitted
  examples for calibration and reports `calibrated: false` until the slice is
  large enough (ADR-006 sets the floor).
- The abstain bucket changes semantics relative to Jev for out-of-scope
  states: callers get a low-confidence answer plus `abstain`, which is the
  behaviour they should have wanted.

## Alternatives considered

- **kNN vote over the HNSW index as the head.** Roughly ties the linear probe
  on frozen features in adjacent literature but is noisier at high class counts
  and pays a search per query; the index is still used for example retrieval
  and for the loop's active learning, just not as the classifier.
- **Evidential (Dirichlet) heads** for uncertainty. Attractive for abstention,
  but need a custom loss and training loop, and the literature is mixed on
  in-distribution top-1 calibration versus plain temperature scaling. Deferred.
- **Isotonic regression** instead of temperature scaling. More flexible, but it
  needs more calibration data per class than a 255-option question can supply.

## Evidence

- Few-shot baselines: SetFit (arXiv:2209.11055); probing study ranking linear
  probe > cosine-linear > cosine-prototype > raw prototype; TK-KNN
  (arXiv:2310.11607) for centroid-vs-kNN; "All Labels Together"
  (arXiv:2309.03563) for encoding all label descriptions jointly.
- Calibration: temperature scaling (Guo et al., ICML 2017); APS (Romano et al.
  2020), RAPS (Angelopoulos et al. 2021); evidential DL (arXiv:1806.01768).
- Ordinal heads: CORAL (Cao et al. 2020), CORN (arXiv:2111.08851).
- Jev calibration/noul measurements: `bench/jev-baseline-2026-09-21.json`.
