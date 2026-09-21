// Scoring metrics for the typesafe benchmark (ADR-006 §protocol 5-6).
// Pure functions over plain arrays so each is unit-testable against
// hand-computed values. No I/O, no dependencies.

/** accuracy = fraction of correct predictions. */
export function accuracy(correct) {
  if (correct.length === 0) return 0;
  let n = 0;
  for (const c of correct) if (c) n++;
  return n / correct.length;
}

/**
 * Macro-F1 over a fixed label set. `pred` and `truth` are parallel arrays of
 * label strings. Precision/recall are computed per label and averaged
 * unweighted (a label with no predictions and no truths contributes F1 0).
 */
export function macroF1(pred, truth, labels) {
  const set = labels ?? [...new Set([...pred, ...truth])].sort();
  let sum = 0;
  for (const lab of set) {
    let tp = 0,
      fp = 0,
      fn = 0;
    for (let i = 0; i < pred.length; i++) {
      const p = pred[i] === lab;
      const t = truth[i] === lab;
      if (p && t) tp++;
      else if (p && !t) fp++;
      else if (!p && t) fn++;
    }
    const prec = tp + fp === 0 ? 0 : tp / (tp + fp);
    const rec = tp + fn === 0 ? 0 : tp / (tp + fn);
    const f1 = prec + rec === 0 ? 0 : (2 * prec * rec) / (prec + rec);
    sum += f1;
  }
  return set.length === 0 ? 0 : sum / set.length;
}

/**
 * Expected Calibration Error with 10 equal-width bins over [0,1].
 * confidences[i] in [0,1], correct[i] boolean. A confidence of exactly 1.0
 * lands in the top bin (index = min(floor(conf*bins), bins-1)).
 * Returns { ece, bins:[{lo,hi,n,conf,acc,sparse}] }. Bins with n<10 are
 * flagged `sparse` (ADR-006: "sparse bins flagged"). Empty bins do not
 * contribute to the weighted error.
 */
export function ece(confidences, correct, bins = 10) {
  const buckets = Array.from({ length: bins }, (_, i) => ({
    lo: i / bins,
    hi: (i + 1) / bins,
    n: 0,
    confSum: 0,
    accSum: 0,
  }));
  const N = confidences.length;
  for (let i = 0; i < N; i++) {
    const c = confidences[i];
    let idx = Math.floor(c * bins);
    if (idx >= bins) idx = bins - 1;
    if (idx < 0) idx = 0;
    const b = buckets[idx];
    b.n++;
    b.confSum += c;
    b.accSum += correct[i] ? 1 : 0;
  }
  let err = 0;
  const reliability = buckets.map((b) => {
    const conf = b.n ? b.confSum / b.n : 0;
    const acc = b.n ? b.accSum / b.n : 0;
    if (b.n && N) err += (b.n / N) * Math.abs(acc - conf);
    return { lo: b.lo, hi: b.hi, n: b.n, conf, acc, sparse: b.n > 0 && b.n < 10 };
  });
  return { ece: err, bins: reliability };
}

/**
 * Multiclass Brier score: mean over items of sum_k (p_k - y_k)^2, where y is
 * the one-hot of the true label. `probs[i]` is an object {label: prob};
 * missing labels count as 0. Lower is better; range [0, 2].
 */
export function brier(probs, truth, labels) {
  const set = labels ?? [...new Set([...truth, ...probs.flatMap((p) => Object.keys(p))])];
  const N = probs.length;
  if (N === 0) return 0;
  let total = 0;
  for (let i = 0; i < N; i++) {
    let s = 0;
    for (const lab of set) {
      const p = probs[i][lab] ?? 0;
      const y = truth[i] === lab ? 1 : 0;
      s += (p - y) * (p - y);
    }
    total += s;
  }
  return total / N;
}

/**
 * Binary AUROC via the rank-sum (Mann–Whitney U) identity, tie-corrected with
 * average ranks. `scores[i]` numeric, `positive[i]` boolean. Returns 0.5 when
 * one class is absent (undefined discrimination, reported not thrown).
 */
export function auroc(scores, positive) {
  const n = scores.length;
  const idx = [...Array(n).keys()].sort((a, b) => scores[a] - scores[b]);
  // average ranks (1-based) over tied score groups
  const ranks = new Array(n);
  let i = 0;
  while (i < n) {
    let j = i;
    while (j + 1 < n && scores[idx[j + 1]] === scores[idx[i]]) j++;
    const avg = (i + j) / 2 + 1; // average of ranks i+1..j+1
    for (let k = i; k <= j; k++) ranks[idx[k]] = avg;
    i = j + 1;
  }
  let nPos = 0,
    nNeg = 0,
    sumRankPos = 0;
  for (let k = 0; k < n; k++) {
    if (positive[k]) {
      nPos++;
      sumRankPos += ranks[k];
    } else nNeg++;
  }
  if (nPos === 0 || nNeg === 0) return 0.5;
  return (sumRankPos - (nPos * (nPos + 1)) / 2) / (nPos * nNeg);
}

/** Mean of a numeric array (0 for empty). */
export function mean(xs) {
  if (xs.length === 0) return 0;
  let s = 0;
  for (const x of xs) s += x;
  return s / xs.length;
}

/**
 * Percentile via linear interpolation on the sorted sample (same convention as
 * NumPy's default). p in [0,100].
 */
export function percentile(xs, p) {
  if (xs.length === 0) return 0;
  const s = [...xs].sort((a, b) => a - b);
  if (s.length === 1) return s[0];
  const rank = (p / 100) * (s.length - 1);
  const lo = Math.floor(rank);
  const hi = Math.ceil(rank);
  if (lo === hi) return s[lo];
  return s[lo] + (rank - lo) * (s[hi] - s[lo]);
}

/** Latency summary from wall-clock per-decision samples (ms). */
export function latency(samplesMs) {
  return {
    n: samplesMs.length,
    p50: percentile(samplesMs, 50),
    p95: percentile(samplesMs, 95),
    mean: mean(samplesMs),
  };
}

/**
 * Throughput. `embedCount` / `decisionCount` over `wallMs` milliseconds.
 * Returns per-second rates; 0 when wall time is non-positive.
 */
export function throughput({ embedCount = 0, decisionCount = 0, wallMs = 0 }) {
  const s = wallMs / 1000;
  return {
    embeds_per_s: s > 0 ? embedCount / s : 0,
    decisions_per_s: s > 0 ? decisionCount / s : 0,
  };
}

export const majorityBaselineRate = (positive) =>
  positive.length === 0 ? 0 : Math.max(mean(positive.map(Number)), 1 - mean(positive.map(Number)));
