#!/usr/bin/env node
// Release-only guard for the frozen, real-ONNX tickets benchmark. The benchmark
// itself can emit a receipt for an unavailable binding or SKIP a missing metric;
// neither is evidence for publishing a native decision model.

import { createHash } from 'node:crypto';
import { readFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { fileURLToPath } from 'node:url';
import { evaluateGates, loadGates } from '../bench/lib/gates.mjs';
import { loadTickets, excludeHeldOutText, majorityLabelFromTrain } from '../bench/lib/fixture.mjs';

const ROOT = resolve(fileURLToPath(new URL('..', import.meta.url)));
const MODEL_NAME = 'bge-small-en-v1.5';
const MODEL_SHA = '828e1496d7fabb79cfa4dcd84fa38625c0d3d21da474a00f08db0f559940cf35';
const TOKENIZER_SHA = 'd241a60d5e8f04cc1b2b3e9ef7a4921b27bf526d9f6050ab90f9267a1f9e5c66';
const BASELINE_SHA = '098f1eeedf3cec56464b5d1efcce24a3cb99181d79e6929e51206cc34e74341f';
const GATES_SHA = 'f4ab640322b01c6a71b9a5e0f658a5787276631887dbabbbadd4a98d13b0fe77';
const BASELINE = resolve(ROOT, 'bench/results/tickets-onnx-bge-2026-09-21.json');
const GATES = resolve(ROOT, 'bench/gates.json');
const MANIFEST = resolve(ROOT, 'models/manifest.json');
const FIXTURE_HASHES = resolve(ROOT, 'bench/fixtures/HASHES.json');
const TICKET_PASSES = new Set([
  'accuracy_vs_jev', 'calibration_ece', 'urgent_vs_train_majority',
  'frustration_vs_train_majority', 'latency_native_p95', 'transfer_regression',
]);
const TICKET_SKIPS = new Set(['oos_auroc', 'latency_wasm_p95']);

function insist(ok, message) {
  if (!ok) throw new Error(message);
}

function digest(bytes) {
  return createHash('sha256').update(bytes).digest('hex');
}

function readJson(path) {
  return JSON.parse(readFileSync(path, 'utf8'));
}

function assertSameJson(actual, expected, what) {
  insist(JSON.stringify(actual) === JSON.stringify(expected), `${what} differs from frozen baseline`);
}

function finite(value, what) {
  insist(typeof value === 'number' && Number.isFinite(value), `${what} was not measured`);
  return value;
}

function main(receiptPath, advisory = false) {
  insist(receiptPath,
    'usage: node scripts/verify-release-bench.mjs <strict-onnx-receipt.json>|--preflight-models');
  insist(digest(readFileSync(BASELINE)) === BASELINE_SHA, 'frozen ONNX baseline receipt changed');
  insist(digest(readFileSync(GATES)) === GATES_SHA, 'release thresholds changed; review and re-pin explicitly');

  const baseline = readJson(BASELINE);
  const models = readJson(MANIFEST).models;
  insist(Array.isArray(models) && models.length && models.every((m) =>
    /^[0-9a-f]{64}$/.test(m.sha256) && /^[0-9a-f]{64}$/.test(m.tokenizer_sha256)),
  'every downloaded model and tokenizer must have a fixed SHA-256 before fetch');
  const model = models.find((m) => m.name === MODEL_NAME);
  insist(model && model.sha256 === MODEL_SHA && model.tokenizer_sha256 === TOKENIZER_SHA,
    'bge model or tokenizer hash differs from release pin');
  if (receiptPath === '--preflight-models') {
    console.log('release model, tokenizer, baseline and gate pins verified before download');
    return;
  }
  const receipt = readJson(resolve(receiptPath));
  const fixtureHashes = readJson(FIXTURE_HASHES).files;

  insist(receipt.schema === 'ruvector-typesafe-bench/receipt@1', 'wrong receipt schema');
  insist(receipt.suite === 'tickets' && receipt.embedder === 'onnx', 'release benchmark must use ONNX tickets');
  assertSameJson(receipt.arms, ['jev', 'local'], 'benchmark arms');
  insist(receipt.regime === 'few-shot' && receipt.shots === baseline.shots && receipt.limit === null,
    'release benchmark must use the full frozen few-shot test');
  insist(receipt.binding?.backend === 'native' && receipt.binding?.unavailable !== true,
    'the shipped native binary was not loaded');
  insist(receipt.stats_json?.embedderId === `${MODEL_NAME}@${MODEL_SHA.slice(0, 12)}` &&
    receipt.stats_json?.dims === model.dims, 'the shipped binary did not load the pinned ONNX model');
  insist(receipt.vocab_guard?.ok === true && receipt.training?.trained === true,
    'vocabulary guard or training did not complete');
  insist(['department', 'urgent', 'frustration'].every((q) => receipt.training.questions?.[q]?.trained === true),
    'one or more typed decision heads were not trained');
  assertSameJson(fixtureHashes, baseline.fixtures.hashes, 'frozen fixture pin');
  assertSameJson(receipt.fixtures?.hashes, fixtureHashes, 'fixture content hashes');
  assertSameJson(receipt.fixtures?.splits_hash, baseline.fixtures.splits_hash, 'fixture splits');
  assertSameJson(receipt.fixtures?.counts, baseline.fixtures.counts, 'fixture split counts');
  assertSameJson(receipt.fixtures?.questions_hash, baseline.fixtures.questions_hash, 'question hash');

  const local = receipt.metrics?.local;
  const jev = receipt.metrics?.jev?.test;
  insist(local?.test?.n === baseline.fixtures.counts.test &&
    local?.test?.n_scored === baseline.fixtures.counts.test &&
    local?.transfer?.n === baseline.fixtures.counts.transfer &&
    jev?.n === baseline.fixtures.counts.test && jev?.n_scored === baseline.fixtures.counts.test,
    'a frozen test, Jev reference, or transfer split was not fully scored');
  insist(jev.choice_accuracy === baseline.metrics.jev.test.choice_accuracy,
    'Jev replay differs from frozen baseline');
  const trainItems = excludeHeldOutText(loadTickets().bySplit).trainItems;
  insist(local.test.urgent_train_majority_label === majorityLabelFromTrain(trainItems, 'urgent') &&
    local.test.frustration_train_majority_label === majorityLabelFromTrain(trainItems, 'frustration'),
  'a secondary-head baseline was chosen from held-out labels');
  const bag = {
    jev_choice_accuracy_test: finite(jev.choice_accuracy, 'Jev test accuracy'),
    choice_accuracy_test: finite(local.test.choice_accuracy, 'native ONNX test accuracy'),
    ece_test: finite(local.test.ece?.ece, 'native ONNX test ECE'),
    urgent_accuracy_test: finite(local.test.urgent_accuracy, 'native ONNX urgency accuracy'),
    urgent_train_majority_accuracy_test: finite(local.test.urgent_train_majority_accuracy, 'train-selected urgency baseline'),
    frustration_accuracy_test: finite(local.test.frustration_accuracy, 'native ONNX frustration accuracy'),
    frustration_train_majority_accuracy_test: finite(local.test.frustration_train_majority_accuracy, 'train-selected frustration baseline'),
    latency_p95_test: finite(local.test.latency_ms?.p95, 'native ONNX p95 latency'),
    transfer_regression_pp:
      (finite(baseline.metrics.local.transfer.choice_accuracy, 'baseline transfer accuracy') -
        finite(local.transfer.choice_accuracy, 'native ONNX transfer accuracy')) * 100,
  };
  const computed = evaluateGates(bag, {
    embedderKind: 'onnx', embedderTarget: 'native', hasOos: false, hasSecondary: true,
    hasBaselineReceipt: true, engineAvailable: true,
  }, loadGates(GATES));
  insist(receipt.gates?.report_only === advisory && receipt.gates?.pass === computed.pass &&
    receipt.gates?.anyFail === computed.anyFail,
  'recorded ONNX gate mode or result differs from recalculated gates');
  if (!advisory) insist(computed.pass, 'strict ONNX gates did not pass');

  const seen = new Set();
  for (const row of computed.rows) {
    const expected = TICKET_SKIPS.has(row.name) ? 'SKIP' :
      TICKET_PASSES.has(row.name) ? (advisory ? row.status : 'PASS') : null;
    insist(expected && ['PASS', 'FAIL', 'SKIP'].includes(expected) &&
      (TICKET_SKIPS.has(row.name) || expected !== 'SKIP') &&
      row.status === expected, `${row.name}: expected ${expected}, got ${row.status}`);
    const recorded = receipt.gates.rows?.find((r) => r.name === row.name);
    insist(recorded?.status === expected, `${row.name}: receipt status is not ${expected}`);
    seen.add(row.name);
  }
  insist(seen.size === TICKET_PASSES.size + TICKET_SKIPS.size &&
    receipt.gates.rows.length === seen.size, 'release gates added, missing, or not checked');
  for (const [file, hash] of [[model.file, MODEL_SHA], [model.tokenizer_file, TOKENIZER_SHA]]) {
    const bytes = readFileSync(resolve(ROOT, 'models', file));
    insist(digest(bytes) === hash, `model asset hash mismatch: ${file}`);
  }
  console.log(`${advisory ? 'advisory' : 'strict'} frozen ONNX tickets ${computed.pass ? 'PASS' : 'FAIL'}: shipped native binary, pinned model and tokenizer, full splits, measured gates`);
  console.log('OOS AUROC and WASM latency remain SKIP on this native tickets suite; they require separate evidence.');
}

try {
  main(process.argv[2], process.argv.includes('--advisory'));
} catch (error) {
  console.error(`release benchmark rejected: ${error.message}`);
  process.exitCode = 1;
}
