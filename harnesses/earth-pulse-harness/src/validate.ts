// SPDX-License-Identifier: MIT
//
// Promotion gate. A harness change is only promoted when it demonstrably
// improves detection without leakage or uncited claims.

import type { ValidationReport } from './types.js';

/** Minimum F1 improvement required to promote. */
const MIN_F1_DELTA = 0.03;
/** Minimum held-out error improvement (fraction) required to promote. */
const MIN_IMPROVEMENT_PCT = 5;

/**
 * Evaluate the promotion gate.
 *
 * Promote ONLY if ALL hold:
 *   - pulseDetectionF1 improves by >= 0.03
 *   - falsePositiveDelta <= 0 (no increase in false positives)
 *   - held-out prediction error improves by >= 5%
 *   - no data leakage
 *   - zero uncited claims
 */
export function validate(opts: {
  baselineError: number;
  heldOutError: number;
  pulseDetectionF1Delta: number;
  falsePositiveDelta: number;
  leakage: boolean;
  uncitedClaims: number;
}): ValidationReport {
  const {
    baselineError,
    heldOutError,
    pulseDetectionF1Delta,
    falsePositiveDelta,
    leakage,
    uncitedClaims,
  } = opts;

  // Improvement fraction of held-out vs baseline error (positive = better).
  const improvementPct =
    baselineError > 0
      ? ((baselineError - heldOutError) / baselineError) * 100
      : heldOutError <= 0
        ? 0
        : -Infinity;

  const notes: string[] = [];

  const f1Pass = pulseDetectionF1Delta >= MIN_F1_DELTA;
  notes.push(
    f1Pass
      ? `PASS: F1 improved by ${pulseDetectionF1Delta.toFixed(3)} (>= ${MIN_F1_DELTA}).`
      : `FAIL: F1 delta ${pulseDetectionF1Delta.toFixed(3)} < required ${MIN_F1_DELTA}.`,
  );

  const fpPass = falsePositiveDelta <= 0;
  notes.push(
    fpPass
      ? `PASS: false positives did not increase (delta ${falsePositiveDelta}).`
      : `FAIL: false positives increased by ${falsePositiveDelta}.`,
  );

  const improvementPass = improvementPct >= MIN_IMPROVEMENT_PCT;
  notes.push(
    improvementPass
      ? `PASS: held-out error improved by ${improvementPct.toFixed(2)}% (>= ${MIN_IMPROVEMENT_PCT}%).`
      : `FAIL: held-out error improvement ${improvementPct.toFixed(2)}% < required ${MIN_IMPROVEMENT_PCT}%.`,
  );

  const leakagePass = !leakage;
  notes.push(
    leakagePass
      ? 'PASS: no data leakage detected.'
      : 'FAIL: data leakage detected — train/test contamination.',
  );

  const citationPass = uncitedClaims === 0;
  notes.push(
    citationPass
      ? 'PASS: all claims are cited.'
      : `FAIL: ${uncitedClaims} uncited claim(s) present.`,
  );

  const passed =
    f1Pass && fpPass && improvementPass && leakagePass && citationPass;

  notes.push(
    passed
      ? 'PROMOTE: all gate criteria satisfied.'
      : 'REJECT: one or more gate criteria failed.',
  );

  return {
    passed,
    leakageDetected: leakage,
    heldOutError,
    baselineError,
    improvementPct,
    notes,
  };
}
