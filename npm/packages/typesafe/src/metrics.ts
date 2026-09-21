/**
 * Evaluation metrics computed in JS from a batch of decisions (ADR-006):
 * accuracy, macro-F1, ECE (10 equal-width bins), Brier, mean confidence, mean
 * abstain, and p50/p95/mean latency. Pure functions — no binding, no I/O.
 */

/** One scored decision for a single question. */
export interface EvalItem {
  /** Predicted class label: a choice key, a legend index as a string, or "yes"/"no". */
  predicted: string;
  /** Ground-truth label, in the same space as `predicted`. */
  truth: string;
  /** Calibrated top-1 probability the engine reported. */
  confidence: number;
  /** Abstain mass the engine reported. */
  abstain: number;
  /** Wall-clock latency of the decision, in milliseconds. */
  latencyMs: number;
}

export interface EvalOptions {
  /** Reliability bins for ECE (default 10). */
  bins?: number;
}

export interface EvalMetrics {
  n: number;
  accuracy: number;
  macroF1: number;
  ece: number;
  brier: number;
  meanConfidence: number;
  meanAbstain: number;
  latencyMs: { p50: number; p95: number; mean: number };
}

/** Linear-interpolation percentile over `values` (0–100). Empty → 0. */
export function percentile(values: readonly number[], p: number): number {
  if (values.length === 0) return 0;
  const sorted = [...values].sort((a, b) => a - b);
  if (sorted.length === 1) return sorted[0];
  const rank = (p / 100) * (sorted.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return sorted[lo];
  return sorted[lo] + (sorted[hi] - sorted[lo]) * (rank - lo);
}

/** Fraction of `true` entries. Empty → 0. */
export function accuracy(correct: readonly boolean[]): number {
  if (correct.length === 0) return 0;
  return correct.filter(Boolean).length / correct.length;
}

/** Unweighted mean of per-class F1 over the labels present in `pairs`. */
export function macroF1(
  pairs: ReadonlyArray<{ predicted: string; truth: string }>,
): number {
  if (pairs.length === 0) return 0;
  const labels = new Set<string>();
  for (const { predicted, truth } of pairs) {
    labels.add(predicted);
    labels.add(truth);
  }
  let sum = 0;
  for (const label of labels) {
    let tp = 0;
    let fp = 0;
    let fn = 0;
    for (const { predicted, truth } of pairs) {
      if (predicted === label && truth === label) tp++;
      else if (predicted === label && truth !== label) fp++;
      else if (predicted !== label && truth === label) fn++;
    }
    const precision = tp + fp === 0 ? 0 : tp / (tp + fp);
    const recall = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = precision + recall === 0 ? 0 : (2 * precision * recall) / (precision + recall);
    sum += f1;
  }
  return sum / labels.size;
}

/** Expected Calibration Error over equal-width confidence bins. */
export function ece(
  confidences: readonly number[],
  corrects: readonly boolean[],
  bins = 10,
): number {
  const n = confidences.length;
  if (n === 0 || n !== corrects.length) return 0;
  const counts = new Array<number>(bins).fill(0);
  const confSum = new Array<number>(bins).fill(0);
  const accSum = new Array<number>(bins).fill(0);
  for (let i = 0; i < n; i++) {
    const c = Math.min(Math.max(confidences[i], 0), 1);
    let b = Math.floor(c * bins);
    if (b === bins) b = bins - 1;
    counts[b]++;
    confSum[b] += c;
    accSum[b] += corrects[i] ? 1 : 0;
  }
  let total = 0;
  for (let b = 0; b < bins; b++) {
    if (counts[b] === 0) continue;
    const avgConf = confSum[b] / counts[b];
    const avgAcc = accSum[b] / counts[b];
    total += (counts[b] / n) * Math.abs(avgConf - avgAcc);
  }
  return total;
}

/** Brier score of the top-1 confidence against the 0/1 correctness outcome. */
export function brier(
  confidences: readonly number[],
  corrects: readonly boolean[],
): number {
  const n = confidences.length;
  if (n === 0 || n !== corrects.length) return 0;
  let sum = 0;
  for (let i = 0; i < n; i++) {
    const outcome = corrects[i] ? 1 : 0;
    sum += (confidences[i] - outcome) ** 2;
  }
  return sum / n;
}

/** Roll a batch of scored decisions up into the ADR-006 metric set. */
export function evaluate(items: readonly EvalItem[], opts: EvalOptions = {}): EvalMetrics {
  const correct = items.map((it) => it.predicted === it.truth);
  const confidences = items.map((it) => it.confidence);
  const latencies = items.map((it) => it.latencyMs);
  const abstains = items.map((it) => it.abstain);
  const mean = (xs: readonly number[]): number =>
    xs.length === 0 ? 0 : xs.reduce((a, b) => a + b, 0) / xs.length;
  return {
    n: items.length,
    accuracy: accuracy(correct),
    macroF1: macroF1(items.map((it) => ({ predicted: it.predicted, truth: it.truth }))),
    ece: ece(confidences, correct, opts.bins ?? 10),
    brier: brier(confidences, correct),
    meanConfidence: mean(confidences),
    meanAbstain: mean(abstains),
    latencyMs: {
      p50: percentile(latencies, 50),
      p95: percentile(latencies, 95),
      mean: mean(latencies),
    },
  };
}
