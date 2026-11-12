// End-to-end tests for the optimize campaign + bank persistence over the REAL
// hash binding (index.js → the native addon). The hash embedder is a
// deterministic test double, so these are reproducible and network-free.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, writeFileSync, readFileSync, existsSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);
const { createTypesafe } = require('../dist/index.js');
const { main } = require('../dist/cli/main.js');

// The loader needs a built backend. CI builds native and wasm in their own
// jobs, so skip rather than fail when neither artifact is present here.
const pkgDir = join(import.meta.dirname ?? new URL('.', import.meta.url).pathname, '..');
const backendBuilt =
  ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc'].some((t) =>
    existsSync(join(pkgDir, 'native', `typesafe.${t}.node`)),
  ) || existsSync(join(pkgDir, 'wasm', 'ruvector_typesafe_wasm.js'));
const skip = backendBuilt ? false : 'no native or wasm backend built';
const realBinding = backendBuilt ? require('../index.js') : null;

const QUESTION = {
  type: 'choice',
  instructions: '',
  criteria: {
    weather: 'weather forecast rain sun storm',
    sports: 'sports match goal court score',
  },
};

// A tiny separable, split-tagged dataset.
function rows() {
  const w = [
    'rain storm clouds today',
    'sunny warm forecast tomorrow',
    'humid temperature rising fast',
    'snow cold winter freeze hard',
    'wind gust breeze strong gale',
    'fog mist morning damp grey',
    'thunder lightning heavy downpour',
    'hail sleet icy slick roads',
  ];
  const s = [
    'football goal striker match win',
    'tennis serve racket court ace',
    'basketball dunk hoop score fast',
    'cricket bat wicket over run',
    'hockey puck rink slapshot net',
    'golf putt fairway birdie green',
    'rugby scrum tackle tryline maul',
    'baseball pitch home run base',
  ];
  const splits = ['train', 'train', 'train', 'train', 'calibration', 'validation', 'transfer', 'test'];
  const out = [];
  for (let i = 0; i < splits.length; i++) {
    out.push({ text: w[i], label: 'weather', split: splits[i] });
    out.push({ text: s[i], label: 'sports', split: splits[i] });
  }
  return out;
}

test('ts.optimize runs a gated campaign and scores test exactly twice', { skip }, async () => {
  const ts = createTypesafe({ binding: realBinding });
  const report = await ts.optimize({
    question: 'q',
    question_def: QUESTION,
    rows: rows(),
    budget_per_day: 64,
    day_key: 'd0',
  });
  assert.equal(report.test_scorings, 2);
  assert.ok(report.arms.length > 0, 'default grid produced arms');
  assert.ok(Array.isArray(report.receipts.receipts));
  assert.ok(report.receipts.receipts.length >= report.arms.length);
  // Champion never trails the baseline on validation (gate is one-directional).
  assert.ok(
    report.champion_val.champion_accuracy >= report.baseline_val.baseline_accuracy - 1e-9,
  );
});

test('typesafe optimize CLI writes a hash-chained receipts JSONL', { skip }, async () => {
  const tmp = mkdtempSync(join(tmpdir(), 'ts-opt-'));
  const qpath = join(tmp, 'q.json');
  const dpath = join(tmp, 'data.jsonl');
  const rpath = join(tmp, 'receipts.jsonl');
  writeFileSync(qpath, JSON.stringify(QUESTION));
  writeFileSync(dpath, rows().map((r) => JSON.stringify(r)).join('\n') + '\n');

  const lines = [];
  const res = await main(
    ['optimize', '--questions', qpath, '--dataset', dpath, '--receipts', rpath, '--budget', '32'],
    { binding: realBinding, stdout: (s) => lines.push(s), stderr: (s) => lines.push(s) },
  );
  assert.equal(res.code, 0);
  const summary = JSON.parse(lines.join('\n'));
  assert.equal(summary.test_scorings, 2);
  assert.ok(existsSync(rpath));
  const receipts = readFileSync(rpath, 'utf8').trim().split('\n').filter(Boolean).map((l) => JSON.parse(l));
  assert.ok(receipts.length > 0);
  // Each receipt links to its predecessor's hash.
  for (let i = 1; i < receipts.length; i++) {
    assert.equal(receipts[i].prev_hash, receipts[i - 1].hash);
  }
});

test('train --bank persists and re-imports the example bank', { skip }, async () => {
  const tmp = mkdtempSync(join(tmpdir(), 'ts-bank-'));
  const bankPath = join(tmp, 'bank.json');
  const exPath = join(tmp, 'examples.jsonl');
  const examples = rows().map((r) => ({ text: r.text, label: r.label }));
  writeFileSync(exPath, examples.map((e) => JSON.stringify(e)).join('\n') + '\n');

  const lines = [];
  const res = await main(
    ['train', '--question', 'q', '--examples', exPath, '--bank', bankPath],
    { binding: realBinding, stdout: (s) => lines.push(s), stderr: (s) => lines.push(s) },
  );
  assert.equal(res.code, 0);
  assert.ok(existsSync(bankPath), 'bank file written');

  // Re-import the persisted bank into a fresh engine.
  const ts = createTypesafe({ binding: realBinding });
  ts.importBank(readFileSync(bankPath, 'utf8'));
  const stats = ts.stats();
  assert.equal(stats.examples, examples.length);
});
