// Benchmark-harness unit + integration tests (node --test). No network, no
// child processes: the local arm is driven by a fake in-process binding.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { accuracy, macroF1, ece, brier, auroc, percentile, latency } from '../bench/lib/metrics.mjs';
import { loadTickets, assertDisjoint, verifyFixtureHashes, FIXTURE_DIR, BENCH_DIR } from '../bench/lib/fixture.mjs';
import { generatorWordsFromCorpus, checkVocabDisjoint, assertVocabDisjoint, ACCEPTED_OVERLAPS } from '../bench/lib/vocab-guard.mjs';
import { fewShotExamples } from '../bench/lib/arms.mjs';
import { scoreRecords } from '../bench/lib/receipt.mjs';
import { main } from '../bench/run.mjs';
import { runChecks } from '../scripts/check-security.mjs';

// ---------------------------------------------------------------------------
// metrics — hand-computed values
// ---------------------------------------------------------------------------

test('accuracy', () => {
  assert.equal(accuracy([true, true, false, true]), 0.75);
  assert.equal(accuracy([]), 0);
});

test('macro-F1 on a tiny 2-class case', () => {
  // pred:  a a b b ; truth: a b a b
  // class a: tp1 fp1 fn1 -> P=.5 R=.5 F1=.5 ; class b: same -> .5 ; macro=.5
  const f1 = macroF1(['a', 'a', 'b', 'b'], ['a', 'b', 'a', 'b'], ['a', 'b']);
  assert.ok(Math.abs(f1 - 0.5) < 1e-12, `got ${f1}`);
  // perfect prediction -> 1.0
  assert.equal(macroF1(['a', 'b'], ['a', 'b'], ['a', 'b']), 1);
});

test('ECE with a hand-computed 2-bin example', () => {
  // conf 0.2 (wrong), 0.8 (right), 0.8 (wrong): bins [0.2..0.3) and [0.8..0.9)
  // bin1: n1 conf .2 acc 0 -> |0-.2|=.2 weight 1/3
  // bin8: n2 conf .8 acc .5 -> |.5-.8|=.3 weight 2/3
  // ECE = 1/3*.2 + 2/3*.3 = .0667 + .2 = .26667
  const { ece: e, bins } = ece([0.2, 0.8, 0.8], [false, true, false]);
  assert.ok(Math.abs(e - (1 / 3) * 0.2 - (2 / 3) * 0.3) < 1e-12, `got ${e}`);
  // sparse flag: every populated bin here has n<10
  assert.ok(bins.filter((b) => b.n > 0).every((b) => b.sparse));
});

test('ECE reproduces the frozen Jev baseline (external ground truth)', () => {
  const j = JSON.parse(readFileSync(join(BENCH_DIR, 'jev-baseline-2026-09-21.json'), 'utf8'));
  for (const arm of ['baseline', 'champion']) {
    const rows = j.test_rows[arm];
    const { ece: e } = ece(rows.map((r) => r.confidence), rows.map((r) => r.correct.department));
    assert.ok(Math.abs(e - j[arm].calibration_test.ece) < 1e-9, `${arm}: ${e} vs ${j[arm].calibration_test.ece}`);
  }
});

test('Brier: perfect vs worst', () => {
  // one item, 2 classes, correct prob mass -> 0 ; wrong -> 2
  assert.ok(Math.abs(brier([{ a: 1, b: 0 }], ['a'], ['a', 'b']) - 0) < 1e-12);
  assert.ok(Math.abs(brier([{ a: 0, b: 1 }], ['a'], ['a', 'b']) - 2) < 1e-12);
  // 0.7/0.3 on the true class: (0.7-1)^2 + (0.3-0)^2 = .09+.09 = .18
  assert.ok(Math.abs(brier([{ a: 0.7, b: 0.3 }], ['a'], ['a', 'b']) - 0.18) < 1e-12);
});

test('AUROC: sklearn example, perfect, inverted, tied', () => {
  // sklearn roc_auc_score([0,0,1,1], [.1,.4,.35,.8]) == 0.75 (one misordered pair)
  assert.equal(auroc([0.1, 0.4, 0.35, 0.8], [false, false, true, true]), 0.75);
  assert.equal(auroc([0.1, 0.2, 0.35, 0.8], [false, false, true, true]), 1); // perfect sep
  assert.equal(auroc([0.9, 0.6, 0.35, 0.2], [false, false, true, true]), 0); // fully inverted
  assert.equal(auroc([0.5, 0.5, 0.5, 0.5], [false, true, false, true]), 0.5); // all tied
  assert.equal(auroc([0.1, 0.9], [true, true]), 0.5); // one class absent
});

test('percentile + latency', () => {
  assert.equal(percentile([1, 2, 3, 4], 50), 2.5);
  assert.equal(percentile([10], 95), 10);
  const l = latency([5, 1, 3]);
  assert.equal(l.n, 3);
  assert.equal(l.p50, 3);
});

// ---------------------------------------------------------------------------
// fixture: hashes, splits, disjointness
// ---------------------------------------------------------------------------

test('fixture loads, splits are disjoint and cover every item', () => {
  const t = loadTickets();
  assert.equal(assertDisjoint(t.bySplit).total, 500);
  const total = Object.values(t.counts).reduce((a, b) => a + b, 0);
  assert.equal(total, 500);
  assert.ok(t.counts.test === 150 && t.counts.validation === 150);
});

test('hash mismatch → refuses to run', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-hash-'));
  try {
    // A HASHES.json whose hashes are wrong for the real bench files.
    writeFileSync(
      join(dir, 'HASHES.json'),
      JSON.stringify({ files: { 'jev-baseline-2026-09-21.json': 'deadbeef', 'ruvector-router-2026-09-21.json': 'x', 'fixtures/tickets-corpus.json': 'x', 'fixtures/tickets-decisions.json': 'x', 'fixtures/tickets-truth.json': 'x' } }),
    );
    assert.throws(() => verifyFixtureHashes({ benchDir: BENCH_DIR, fixtureDir: dir }), /hash mismatch/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// vocab guard
// ---------------------------------------------------------------------------

test('vocab guard passes the real fixture and catches a planted leak', () => {
  const t = loadTickets();
  assert.doesNotThrow(() => assertVocabDisjoint(t.questions));

  const corpus = JSON.parse(readFileSync(join(FIXTURE_DIR, 'tickets-corpus.json'), 'utf8'));
  const gw = generatorWordsFromCorpus(corpus);
  const planted = JSON.parse(JSON.stringify(t.questions));
  planted.department.criteria.billing = 'Charges and astronomy observations';
  const res = checkVocabDisjoint(gw, planted, { accepted: ACCEPTED_OVERLAPS });
  assert.equal(res.ok, false);
  assert.ok(res.leaks.includes('astronomy'));
});

test('few-shot picks N/class deterministically', () => {
  const t = loadTickets();
  const a = fewShotExamples(t.bySplit.train, 'department', 8);
  const b = fewShotExamples(t.bySplit.train, 'department', 8);
  assert.deepEqual(a.examples, b.examples); // deterministic
  for (const n of Object.values(a.effective)) assert.ok(n <= 8);
});

test('scoreRecords computes OOS AUROC + mean_abstain when items carry an oos flag', () => {
  // in-scope items with LOW abstain, oos items with HIGH abstain -> perfect sep
  const recs = [
    { choice: 'a', trueChoice: 'a', correctChoice: true, confidence: 0.9, abstain: 0.1, oos: false, probabilities: { a: 0.9 }, latencyMs: 1 },
    { choice: 'b', trueChoice: 'b', correctChoice: true, confidence: 0.8, abstain: 0.2, oos: false, probabilities: { b: 0.8 }, latencyMs: 1 },
    { choice: 'a', trueChoice: 'oos', correctChoice: false, confidence: 0.2, abstain: 0.9, oos: true, probabilities: { a: 0.2 }, latencyMs: 1 },
    { choice: 'b', trueChoice: 'oos', correctChoice: false, confidence: 0.3, abstain: 0.8, oos: true, probabilities: { b: 0.3 }, latencyMs: 1 },
  ];
  const b = scoreRecords(recs, { departments: ['a', 'b'] });
  assert.equal(b.oos_auroc, 1); // abstain perfectly separates oos from in-scope
  assert.equal(b.n_oos, 2);
  assert.ok(Math.abs(b.mean_abstain - 0.5) < 1e-12);
});

test('scoreRecords omits OOS metrics when no item carries an oos flag', () => {
  const recs = [{ choice: 'a', trueChoice: 'a', correctChoice: true, confidence: 0.9, abstain: 0.1, probabilities: { a: 0.9 }, latencyMs: 1 }];
  const b = scoreRecords(recs, { departments: ['a'] });
  assert.equal(b.oos_auroc, undefined);
});

// ---------------------------------------------------------------------------
// run.mjs end-to-end with a fake binding
// ---------------------------------------------------------------------------

function keywordEngine(alwaysWrong = false) {
  const depts = ['billing', 'technical', 'shipping', 'returns', 'account', 'sales', 'legal', 'feedback'];
  const KW = {
    billing: /charg|invoice|refund|bill|payment/i,
    technical: /error|bug|broken|crash|software|login failing/i,
    shipping: /ship|parcel|deliver|track|package/i,
    returns: /return|wrong size|exchange|swap/i,
    account: /locked out|password|sign-in|account access/i,
    sales: /price|cost|tier|enterprise|purchas/i,
    legal: /subpoena|contract|complian|legal|gdpr/i,
    feedback: /love|hate|opinion|experience|feedback/i,
  };
  return class Engine {
    constructor() {}
    trainJson() {
      return JSON.stringify({ ok: true });
    }
    statsJson() {
      return JSON.stringify({ backend: 'fake', trained: true });
    }
    decideJson(reqJson) {
      const req = JSON.parse(reqJson);
      let dept = depts[0];
      for (const [d, re] of Object.entries(KW)) if (re.test(req.state)) { dept = d; break; }
      if (alwaysWrong) dept = 'legal'; // a single class, ~always wrong
      const probabilities = Object.fromEntries(depts.map((d) => [d, d === dept ? 0.9 : 0.1 / 7]));
      return JSON.stringify({
        answers: {
          department: { choice: dept, probabilities, confidence: 0.9, abstain: 0, calibrated: true, head: 'nearest-prototype', model: 'fake@0', temperature: 1 },
          urgent: { noul: /urgent|asap|immediately|now/i.test(req.state) ? 0.8 : 0.2, abstain: 0, calibrated: false, head: 'similarity-uncalibrated', model: 'fake@0', temperature: 1 },
          frustration: { score: 0, legend: 'Neutral or calm', probabilities: [0.8, 0.15, 0.05], abstain: 0, calibrated: true, head: 'linear-probe', model: 'fake@0', temperature: 1 },
        },
        usage: { embed_calls: 1, texts_embedded: 1, state_bytes: req.state.length },
      });
    }
  };
}

const fakeBinding = (wrong = false) => ({ Engine: keywordEngine(wrong), version: () => 'fake-1.0.0', backend: 'fake' });

test('run.mjs end-to-end: fake binding produces a well-formed receipt; --gate --report-only exits 0', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'ts-run-')), 'receipt.json');
  const { code, receipt } = await main(
    ['--suite', 'tickets', '--arm', 'both', '--embedder', 'hash', '--gate', '--report-only', '--limit', '60', '--out', out],
    { binding: fakeBinding(false) },
  );
  assert.equal(code, 0, 'report-only never fails the exit code');
  // receipt schema + no item text
  assert.equal(receipt.schema, 'ruvector-typesafe-bench/receipt@1');
  assert.ok(receipt.fixtures.splits_hash && receipt.fixtures.questions_hash);
  assert.ok(receipt.fixtures.hashes['jev-baseline-2026-09-21.json']);
  assert.ok(receipt.metrics.jev.test.choice_accuracy > 0);
  assert.ok(receipt.metrics.local.test.n_scored > 0);
  assert.ok(receipt.gates && Array.isArray(receipt.gates.rows));
  assert.equal(receipt.binding.backend, 'fake');
  // no raw ticket text anywhere in the receipt
  const blob = JSON.stringify(receipt);
  assert.ok(!blob.includes('Received the wrong size'), 'receipt must not embed item text');
});

test('run.mjs --gate (strict) exits non-zero with an always-wrong engine', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'ts-run-')), 'receipt.json');
  const { code, receipt } = await main(
    ['--suite', 'tickets', '--arm', 'both', '--embedder', 'hash', '--gate', '--limit', '60', '--out', out],
    { binding: fakeBinding(true) },
  );
  assert.equal(code, 1, 'a failing accuracy gate must fail the run in strict mode');
  const acc = receipt.gates.rows.find((r) => r.name === 'accuracy_vs_jev');
  assert.equal(acc.status, 'FAIL');
});

test('run.mjs reports "engine unavailable" without crashing when decideJson errors', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'ts-run-')), 'receipt.json');
  const notImpl = {
    Engine: class {
      decideJson() {
        return JSON.stringify({ error: { kind: 'invalid', message: 'engine not implemented yet' } });
      }
    },
    version: () => '0',
    backend: 'stub',
  };
  const { code, receipt } = await main(
    ['--suite', 'tickets', '--arm', 'local', '--out', out],
    { binding: notImpl },
  );
  assert.equal(code, 0);
  assert.ok(receipt.binding.unavailable);
  assert.match(receipt.binding.error, /not implemented/);
});

// ---------------------------------------------------------------------------
// check-security.mjs
// ---------------------------------------------------------------------------

test('check-security passes on the current tree', () => {
  const res = runChecks();
  assert.equal(res.ok, true, JSON.stringify(res.violations));
  assert.ok(res.scanned.decision_files > 0);
});

test('check-security fails on a temp file using child_process', () => {
  const root = mkdtempSync(join(tmpdir(), 'ts-sec-'));
  try {
    mkdirSync(join(root, 'src'));
    writeFileSync(join(root, 'src', 'bad.js'), "const cp = require('child_process'); cp.execSync('ls');\n");
    const res = runChecks({ root });
    assert.equal(res.ok, false);
    assert.ok(res.violations.some((v) => /child_process/.test(v.pattern)));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
