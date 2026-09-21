// Benchmark arms (ADR-006 §protocol 1-2). Each arm scores the *identical*
// request JSON per item and emits normalised per-item records that metrics.mjs
// consumes uniformly.
//
//   arm 'jev'   — replays the frozen 2026-09-21 baseline JSON. NO NETWORK: the
//                 answers were measured once, at a paid call, and committed.
//                 Only the `test` split has per-item rows; other splits carry
//                 aggregate accuracy only.
//   arm 'local' — the ruvector binding. Builds one request per item, times the
//                 wall-clock at decideJson, and supports SetFit-style few-shot
//                 (train on the train split, N examples/class) and zero-shot.

import { createHash } from 'node:crypto';
import { buildQuestions } from './fixture.mjs';

const sha256hex = (s) => createHash('sha256').update(String(s)).digest('hex');

/**
 * Normalised per-item scoring record. `choice`/`probabilities`/`confidence`
 * describe the `department` choice question (the recomputed parity metric).
 * `urgent`/`frustration` carry pred/label/correct for the secondary questions.
 */
function record({ id, choice, probabilities, trueChoice, confidence, abstain, oos, latencyMs, tokens, urgent, frustration }) {
  return {
    id,
    choice,
    probabilities: probabilities ?? {},
    trueChoice,
    confidence: confidence ?? 0,
    // abstain mass (ADR-003). Falls back to 1-confidence when the arm does not
    // report it explicitly (the Jev replay). Used for the OOS AUROC gate.
    abstain: typeof abstain === 'number' ? abstain : 1 - (confidence ?? 0),
    oos: typeof oos === 'boolean' ? oos : undefined,
    correctChoice: choice === trueChoice,
    latencyMs: latencyMs ?? 0,
    tokens: tokens ?? null,
    urgent: urgent ?? null,
    frustration: frustration ?? null,
  };
}

// ---------------------------------------------------------------------------
// jev arm — replay the frozen baseline
// ---------------------------------------------------------------------------

/**
 * Replay Jev's frozen per-item answers for the tickets `test` split.
 * @param jevBaseline parsed jev-baseline-2026-09-21.json
 * @param testItems   the fixture items assigned to the `test` split
 * @param opts.arm    'baseline' (gen-0, apples-to-apples with our gen-0
 *                    criteria) or 'champion' (mutated criteria — informational).
 * @param opts.limit  restrict to the first N ids (kept identical across arms).
 */
export function replayJev(jevBaseline, testItems, { arm = 'baseline', limit } = {}) {
  const rows = jevBaseline.test_rows?.[arm];
  if (!Array.isArray(rows)) {
    return { available: false, reason: `no test_rows.${arm} in baseline`, records: [] };
  }
  const byId = new Map(rows.map((r) => [r.id, r]));
  let items = testItems;
  if (typeof limit === 'number') items = items.slice(0, limit);

  const records = [];
  const missing = [];
  for (const it of items) {
    const r = byId.get(it.id);
    if (!r || !r.ok) {
      missing.push(it.id);
      continue;
    }
    records.push(
      record({
        id: it.id,
        choice: r.pred.department,
        probabilities: r.probabilities,
        trueChoice: r.label.department,
        confidence: r.confidence,
        latencyMs: r.latencyMs,
        tokens: r.tokens,
        // Jev exposes no noul/score probability, only its own correctness flags.
        urgent: { pred: r.pred.urgent, label: r.label.urgent, correct: r.correct.urgent, score: null },
        frustration: {
          pred: r.pred.frustration,
          label: r.label.frustration,
          correct: r.correct.frustration,
          score: null,
        },
      }),
    );
  }
  return {
    available: true,
    network: false,
    arm,
    note: `replayed frozen jev-baseline (${arm}); no network call`,
    idsMissing: missing,
    records,
  };
}

// ---------------------------------------------------------------------------
// local arm — the ruvector binding
// ---------------------------------------------------------------------------

/**
 * Pick N examples/class deterministically (sorted by sha256(id)) from the
 * train items for a choice question keyed by `labelKey`. Records the effective
 * shot count per class (a class may hold < N).
 */
export function fewShotExamples(trainItems, labelKey, shots) {
  const byClass = new Map();
  for (const it of trainItems) {
    const lab = it.label[labelKey];
    if (!byClass.has(lab)) byClass.set(lab, []);
    byClass.get(lab).push(it);
  }
  const examples = [];
  const effective = {};
  for (const [lab, items] of [...byClass.entries()].sort((a, b) => String(a[0]).localeCompare(String(b[0])))) {
    const picked = [...items].sort((a, b) => sha256hex(a.id).localeCompare(sha256hex(b.id))).slice(0, shots);
    effective[lab] = picked.length;
    for (const it of picked) examples.push({ text: it.text, label: String(lab) });
  }
  return { examples, effective };
}

/**
 * Train the engine for the few-shot regime, if it exposes trainJson. Trains the
 * `department` choice question on the train split. Returns training metadata
 * (never throws for a missing trainJson — zero-shot engines simply skip).
 */
export function trainFewShot(engine, trainItems, { shots = 8, question = 'department', labelKey = 'department' } = {}) {
  if (typeof engine.trainJson !== 'function') {
    return { trained: false, reason: 'engine has no trainJson' };
  }
  const { examples, effective } = fewShotExamples(trainItems, labelKey, shots);
  let out;
  try {
    out = engine.trainJson(JSON.stringify({ question, examples }));
  } catch (e) {
    return { trained: false, error: String(e && e.message ? e.message : e) };
  }
  let parsed = null;
  try {
    parsed = typeof out === 'string' ? JSON.parse(out) : out;
  } catch {
    /* trainJson may return a non-JSON status string */
  }
  if (parsed && parsed.error) return { trained: false, error: parsed.error };
  return { trained: true, shots, shotsEffective: effective, examplesUsed: examples.length };
}

/**
 * Score items with the local binding. Builds one request per item and times
 * decideJson. Returns { available:false, error } when the engine reports the
 * "not implemented yet" error on the FIRST item (the harness then reports
 * "engine unavailable" for the arm rather than crashing).
 */
export function runLocal(engine, items, questionDefs, { labelKey = 'department' } = {}) {
  const wireQuestions = buildQuestions(questionDefs);
  const records = [];
  const wallStart = performance.now();
  for (let i = 0; i < items.length; i++) {
    const it = items[i];
    const req = JSON.stringify({ state: it.text, questions: wireQuestions });
    let raw;
    const t0 = performance.now();
    try {
      raw = engine.decideJson(req);
    } catch (e) {
      return { available: false, error: `decideJson threw: ${e && e.message ? e.message : e}` };
    }
    const latencyMs = performance.now() - t0;
    let resp;
    try {
      resp = typeof raw === 'string' ? JSON.parse(raw) : raw;
    } catch (e) {
      return { available: false, error: `decideJson returned non-JSON: ${String(raw).slice(0, 120)}` };
    }
    if (resp && resp.error) {
      // Engine unavailable / not implemented: report on the first item.
      if (i === 0) return { available: false, error: resp.error.message ?? JSON.stringify(resp.error), errorKind: resp.error.kind };
      // A mid-run per-item error: record as an incorrect abstention, keep going.
      records.push(record({ id: it.id, choice: null, trueChoice: it.label[labelKey], confidence: 0, latencyMs }));
      continue;
    }
    records.push(normalizeLocal(it, resp, latencyMs, labelKey));
  }
  const wallMs = performance.now() - wallStart;
  return { available: true, network: false, records, wallMs, embedCount: items.length };
}

/** Convert a DecisionResponse into a normalised record. */
function normalizeLocal(item, resp, latencyMs, labelKey) {
  const answers = resp.answers ?? {};
  const dept = answers[labelKey] ?? answers.department ?? {};
  const urgentA = answers.urgent;
  const frA = answers.frustration;
  let urgent = null;
  if (urgentA && typeof urgentA.noul === 'number') {
    const pred = urgentA.noul >= 0.5;
    urgent = { pred, score: urgentA.noul, label: item.label.urgent, correct: pred === item.label.urgent };
  }
  let frustration = null;
  if (frA && typeof frA.score === 'number') {
    frustration = {
      pred: frA.score,
      score: frA.score,
      label: item.label.frustration,
      correct: frA.score === item.label.frustration,
    };
  }
  const confidence = typeof dept.confidence === 'number' ? dept.confidence : dept.meta?.confidence ?? 0;
  const abstain = typeof dept.abstain === 'number' ? dept.abstain : dept.meta?.abstain;
  return record({
    id: item.id,
    choice: dept.choice ?? null,
    probabilities: dept.probabilities ?? {},
    trueChoice: item.label[labelKey],
    confidence,
    abstain,
    oos: typeof item.oos === 'boolean' ? item.oos : undefined,
    latencyMs,
    urgent,
    frustration,
  });
}
