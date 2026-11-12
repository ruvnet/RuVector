import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { loadTickets, assertDisjoint, excludeHeldOutText, majorityLabelFromTrain } from '../lib/fixture.mjs';
import { trainTicketQuestions } from '../lib/arms.mjs';
import { scoreRecords } from '../lib/receipt.mjs';
import { evaluateGates, loadGates } from '../lib/gates.mjs';
import { main } from '../run.mjs';

test('few-shot trains all three heads from the frozen train split with a per-class cap', () => {
  const fixture = loadTickets();
  assertDisjoint(fixture.bySplit);
  const trainPool = excludeHeldOutText(fixture.bySplit);
  assert.equal(trainPool.excluded, 1, 'frozen fixture has one exact text collision');
  const batches = [];
  const engine = {
    trainJson(json) {
      batches.push(JSON.parse(json));
      return JSON.stringify({ accepted: 1 });
    },
  };
  const result = trainTicketQuestions(engine, trainPool.trainItems, fixture.questions, { shots: 8 });
  assert.equal(result.trained, true);
  assert.deepEqual(batches.map((b) => b.question), ['department', 'urgent', 'frustration']);

  const trainTexts = new Set(fixture.bySplit.train.map((it) => it.text));
  const heldOutTexts = new Set(
    ['calibration', 'validation', 'transfer', 'test']
      .flatMap((split) => fixture.bySplit[split].map((it) => it.text)),
  );
  const legend = fixture.questions.frustration.criteria;
  for (const batch of batches) {
    for (const example of batch.examples) {
      assert.ok(trainTexts.has(example.text), `not in frozen train: ${example.text}`);
      assert.ok(!heldOutTexts.has(example.text), 'held-out text appeared in training');
    }
    const counts = result.questions[batch.question].shotsEffective;
    assert.ok(Object.values(counts).every((n) => n > 0 && n <= 8));
    assert.equal(batch.examples.length, Object.values(counts).reduce((sum, n) => sum + n, 0));
  }
  assert.deepEqual(new Set(batches[1].examples.map((e) => e.label)), new Set(['yes', 'no']));
  assert.deepEqual(new Set(batches[2].examples.map((e) => e.label)), new Set(legend));
  assert.equal(result.examplesUsed, batches.reduce((n, batch) => n + batch.examples.length, 0));

  const repeated = [];
  trainTicketQuestions({ trainJson: (json) => (repeated.push(JSON.parse(json)), '{}') },
    trainPool.trainItems, fixture.questions, { shots: 8 });
  assert.deepEqual(repeated, batches, 'selection and label conversion must be deterministic');
});

test('CLINC OOS carries both classes to AUROC while choice scores in-scope rows only', async () => {
  const sample = [
    { id: 'in-a', text: 'known a', label: 'a', oos: false },
    { id: 'in-b', text: 'known b', label: 'b', oos: false },
    { id: 'out-a', text: 'outside a', label: 'oos', oos: true },
    { id: 'out-b', text: 'outside b', label: 'oos', oos: true },
  ];
  const ds = {
    trainItems: [{ id: 'train-a', text: 'training', label: 'a' }],
    testItems: sample,
    labels: ['a', 'b'],
    questions: { intent: { type: 'choice', criteria: { a: 'class a', b: 'class b' } } },
    counts: { train: 1, test_in_scope: 2, test_oos: 2 },
    hasOos: true,
    splitsHash: 'frozen-test-split',
  };
  const binding = {
    version: () => 'fake', backend: 'native',
    Engine: class {
      trainJson() { return '{}'; }
      statsJson() { return '{}'; }
      decideJson(json) {
        const state = JSON.parse(json).state;
        const oos = state.startsWith('outside');
        const choice = state.endsWith('b') ? 'b' : 'a';
        return JSON.stringify({ answers: {
          intent: { choice, probabilities: { a: choice === 'a' ? 0.9 : 0.1, b: choice === 'b' ? 0.9 : 0.1 },
            confidence: oos ? 0.1 : 0.9, abstain: oos ? 0.9 : 0.1 },
        } });
      }
    },
  };
  const dir = mkdtempSync(join(tmpdir(), 'typesafe-clinc-'));
  try {
    const { receipt } = await main(['--suite', 'clinc150', '--arm', 'local', '--out', join(dir, 'receipt.json')],
      { binding, loadDataset: async () => ds });
    const metrics = receipt.metrics.local.test;
    assert.equal(metrics.n, 4);
    assert.equal(metrics.n_scored, 2);
    assert.equal(metrics.choice_accuracy, 1);
    assert.equal(metrics.n_oos, 2);
    assert.equal(metrics.n_in_scope, 2);
    assert.equal(metrics.oos_auroc, 1);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('ticket receipt gates secondary heads against train-selected constant baselines', async () => {
  const trainCalls = [];
  const binding = {
    version: () => 'fake', backend: 'native',
    Engine: class {
      trainJson(json) { trainCalls.push(JSON.parse(json)); return '{}'; }
      statsJson() { return '{}'; }
      decideJson() {
        return JSON.stringify({ answers: {
          department: { choice: 'billing', probabilities: { billing: 1 }, confidence: 1, abstain: 0 },
          urgent: { noul: 0.1 },
          frustration: { score: 0 },
        } });
      }
    },
  };
  const dir = mkdtempSync(join(tmpdir(), 'typesafe-tickets-'));
  try {
    const { receipt } = await main([
      '--suite', 'tickets', '--arm', 'both', '--limit', '10', '--gate', '--report-only',
      '--out', join(dir, 'receipt.json'),
    ], { binding });
    assert.deepEqual(trainCalls.map((b) => b.question), ['department', 'urgent', 'frustration']);
    assert.equal(receipt.training.excludedHeldOutTexts, 1);
    const testMetrics = receipt.metrics.local.test;
    const trainPool = excludeHeldOutText(loadTickets().bySplit);
    assert.equal(testMetrics.urgent_train_majority_label,
      majorityLabelFromTrain(trainPool.trainItems, 'urgent'));
    assert.equal(testMetrics.frustration_train_majority_label,
      majorityLabelFromTrain(trainPool.trainItems, 'frustration'));
    for (const name of ['urgent_vs_train_majority', 'frustration_vs_train_majority']) {
      const gate = receipt.gates.rows.find((row) => row.name === name);
      assert.equal(gate.status, 'PASS', `${name}: ${gate.detail}`);
    }
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('majority baseline chooses from train even when test majority is opposite', () => {
  const train = [
    { label: { urgent: true, frustration: 2 } },
    { label: { urgent: true, frustration: 2 } },
    { label: { urgent: false, frustration: 0 } },
  ];
  const majorityLabels = {
    urgent: majorityLabelFromTrain(train, 'urgent'),
    frustration: majorityLabelFromTrain(train, 'frustration'),
  };
  const records = [false, false, false, true].map((urgent, i) => ({
    choice: 'a', trueChoice: 'a', correctChoice: true, probabilities: { a: 1 },
    confidence: 1, abstain: 0, latencyMs: 1,
    urgent: { label: urgent, correct: i >= 1 },
    frustration: { label: i === 3 ? 2 : 0, correct: i >= 1 },
  }));
  const block = scoreRecords(records, { departments: ['a'], majorityLabels });
  assert.equal(block.urgent_train_majority_label, true);
  assert.equal(block.frustration_train_majority_label, 2);
  assert.equal(block.urgent_train_majority_accuracy, 0.25);
  assert.equal(block.frustration_train_majority_accuracy, 0.25);
  assert.equal(block.urgent_majority_rate, 0.75, 'oracle test majority is informational only');
  const gates = evaluateGates({
    urgent_accuracy_test: block.urgent_accuracy,
    urgent_train_majority_accuracy_test: block.urgent_train_majority_accuracy,
    frustration_accuracy_test: block.frustration_accuracy,
    frustration_train_majority_accuracy_test: block.frustration_train_majority_accuracy,
  }, { hasSecondary: true, engineAvailable: true, embedderTarget: 'hash' }, loadGates());
  for (const name of ['urgent_vs_train_majority', 'frustration_vs_train_majority']) {
    assert.equal(gates.rows.find((row) => row.name === name).status, 'PASS');
  }
});

test('OOS gate fails when a flagged suite has no negative examples', () => {
  const onlyOos = scoreRecords([
    { choice: 'a', trueChoice: 'oos', correctChoice: false, confidence: 0.1,
      abstain: 0.9, probabilities: { a: 0.1 }, oos: true, latencyMs: 1 },
  ], { departments: ['a'] });
  assert.equal(onlyOos.oos_auroc, null);
  const gates = evaluateGates({ oos_auroc: onlyOos.oos_auroc },
    { hasOos: true, engineAvailable: true, embedderTarget: 'hash' }, loadGates());
  assert.equal(gates.rows.find((g) => g.name === 'oos_auroc').status, 'FAIL');
});
