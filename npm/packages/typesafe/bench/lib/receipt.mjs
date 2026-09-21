// Receipt assembly + per-arm scoring (ADR-006 §protocol 5, §Where it lives:
// "one receipt per run"). A receipt records what was measured and against what
// — arms, split hashes, fixture hashes, metrics per split, latency, the
// binding's statsJson, and host info — and NEVER the text of any item (ADR-005:
// state is never logged; receipts store hashes and lengths, not text).

import os from 'node:os';
import { createHash } from 'node:crypto';
import { accuracy, macroF1, ece, brier, auroc, mean, latency, throughput, majorityBaselineRate } from './metrics.mjs';

const sha256hex = (s) => createHash('sha256').update(String(s)).digest('hex');

/**
 * Score a set of normalised records (from arms.mjs) into a metrics block.
 * `departments` fixes the label set so macro-F1/Brier are stable.
 */
export function scoreRecords(records, { departments, wallMs } = {}) {
  const scored = records.filter((r) => r.choice !== null && r.choice !== undefined);
  const choicePred = scored.map((r) => r.choice);
  const choiceTrue = scored.map((r) => r.trueChoice);
  const choiceCorrect = scored.map((r) => r.correctChoice);
  const confidences = scored.map((r) => r.confidence);
  const probs = scored.map((r) => r.probabilities);
  const latencies = records.map((r) => r.latencyMs).filter((x) => x > 0);

  const block = {
    n: records.length,
    n_scored: scored.length,
    choice_accuracy: accuracy(choiceCorrect),
    macro_f1: macroF1(choicePred, choiceTrue, departments),
    ece: ece(confidences, choiceCorrect),
    brier: brier(probs, choiceTrue, departments),
    mean_confidence: mean(confidences),
    mean_abstain: mean(records.map((r) => r.abstain).filter((x) => typeof x === 'number')),
    latency_ms: latency(latencies),
  };

  // Out-of-scope AUROC (ADR-006): when the suite carries an OOS flag on its
  // items, score how well the abstain mass separates OOS from in-scope. Higher
  // abstain on an OOS item is the correct behaviour, so OOS is the positive
  // class and abstain is the ranking score.
  const oosFlagged = records.filter((r) => typeof r.oos === 'boolean');
  if (oosFlagged.length) {
    const scores = oosFlagged.map((r) => r.abstain);
    const positive = oosFlagged.map((r) => r.oos === true);
    block.oos_auroc = auroc(scores, positive);
    block.n_oos = positive.filter(Boolean).length;
    block.oos_majority_rate = majorityBaselineRate(positive.map(Number));
  }

  // Secondary questions, when present.
  const urgentRecs = records.filter((r) => r.urgent);
  if (urgentRecs.length) {
    block.urgent_accuracy = accuracy(urgentRecs.map((r) => r.urgent.correct));
    block.urgent_majority_rate = majorityBaselineRate(urgentRecs.map((r) => r.urgent.label));
    // AUROC only when the arm exposes a continuous noul score (local arm).
    const withScore = urgentRecs.filter((r) => typeof r.urgent.score === 'number');
    if (withScore.length === urgentRecs.length && withScore.length) {
      block.urgent_auroc = auroc(withScore.map((r) => r.urgent.score), withScore.map((r) => !!r.urgent.label));
    } else {
      block.urgent_auroc = null; // Jev exposes no noul probability
    }
  }
  const frRecs = records.filter((r) => r.frustration);
  if (frRecs.length) {
    block.frustration_accuracy = accuracy(frRecs.map((r) => r.frustration.correct));
  }

  if (typeof wallMs === 'number' && wallMs > 0) {
    block.throughput = throughput({ embedCount: records.length, decisionCount: records.length, wallMs });
  }
  return block;
}

/** Safe statsJson call — not part of the required binding contract. */
export function readStats(engine) {
  if (!engine || typeof engine.statsJson !== 'function') return null;
  try {
    const s = engine.statsJson();
    return typeof s === 'string' ? JSON.parse(s) : s;
  } catch {
    return null;
  }
}

/** Host + runtime provenance for reproducibility (no secrets, no paths). */
export function hostInfo() {
  return {
    node: process.version,
    platform: process.platform,
    arch: process.arch,
    cpus: os.cpus()?.length ?? null,
    cpu_model: os.cpus()?.[0]?.model ?? null,
    os_release: os.release(),
    total_mem_mb: Math.round(os.totalmem() / 1024 / 1024),
  };
}

/**
 * Assemble the receipt. `metricsBySplitByArm` is { arm: { split: block } }.
 * `questions` is hashed, never embedded (champion criteria carry ticket text).
 */
export function buildReceipt({
  suite,
  arms,
  embedder,
  regime,
  shots,
  limit,
  splitsHash,
  fixtureHashes,
  counts,
  questions,
  metrics,
  gates,
  binding,
  training,
  vocabGuard,
  stats,
  extra,
}) {
  return {
    schema: 'ruvector-typesafe-bench/receipt@1',
    generated_at: new Date().toISOString(),
    suite,
    arms,
    embedder,
    regime,
    shots: regime === 'few-shot' ? shots : null,
    limit: limit ?? null,
    binding: binding ?? null, // { version, backend } or { unavailable, error }
    fixtures: {
      hashes: fixtureHashes,
      splits_hash: splitsHash,
      counts,
      questions_hash: questions ? sha256hex(JSON.stringify(questions)) : null,
    },
    vocab_guard: vocabGuard ?? null,
    training: training ?? null,
    metrics, // { arm: { split: block } } — no item text, only aggregates
    gates: gates ?? null,
    stats_json: stats ?? null,
    host: hostInfo(),
    ...(extra ? { extra } : {}),
  };
}
