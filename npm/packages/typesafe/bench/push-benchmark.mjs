#!/usr/bin/env node
// Bounded exploratory selection using the existing guarded --no-test harness.
import assert from 'node:assert/strict';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { spawnSync } from 'node:child_process';
import { dirname, join, resolve } from 'node:path';
import { fileURLToPath } from 'node:url';

const here = dirname(fileURLToPath(import.meta.url));
const root = dirname(here);
const out = resolve(process.argv[2] ?? join(here, 'results', 'push-2026-10-10'));
mkdirSync(out, { recursive: true });
const base = { probeIterations: 5000, choiceInstructions: true };
const variants = [
  ['baseline', {}], ['l2-low', { probeL2: 0.0001 }], ['l2-high', { probeL2: 0.01 }],
  ['balanced', { probeClassBalanced: true }],
  ['balanced-l2-low', { probeClassBalanced: true, probeL2: 0.0001 }],
  ['logit-4', { logitScale: 4 }],
];
const models = ['bge-small-en-v1.5', 'bge-small-en-v1.5-int8'];
const plan = {
  schema: 'typesafe-bench/selection-plan@1', created_at: new Date().toISOString(),
  test_policy: 'Test was reported previously; no test scoring during selection. One selected candidate will be evaluated subsequently. Exploratory, not a fresh confirmatory holdout.',
  selection: 'Eligible: validation department and mean accuracy >= FP32 baseline; validation ECE <= baseline; transfer mean accuracy >= baseline - 0.01. Rank by validation mean accuracy, department accuracy, lower ECE, then model size. If no candidate qualifies, retain baseline.',
  configs: models.flatMap(model => variants.map(([name, options]) => ({
    id: `${model.endsWith('int8') ? 'int8' : 'fp32'}-${name}`, model, engine_options: { ...base, ...options },
  }))),
};
const planText = JSON.stringify(plan, null, 2) + '\n';
writeFileSync(join(out, 'selection-plan.json'), planText, { flag: 'wx' });
const meanAcc = m => (m.choice_accuracy + m.urgent_accuracy + m.frustration_accuracy) / 3;
const summarize = m => ({ department: m.choice_accuracy, urgent: m.urgent_accuracy,
  frustration: m.frustration_accuracy, mean_accuracy: meanAcc(m), ece: m.ece.ece });
const results = [];
for (const config of plan.configs) {
  const path = join(out, `${config.id}.json`);
  const args = [join(here, 'run.mjs'), '--suite', 'tickets', '--arm', 'local', '--embedder', 'onnx',
    '--model', config.model, '--shots', '10000', '--engine-options', JSON.stringify(config.engine_options),
    '--no-test', '--out', path];
  const run = spawnSync(process.execPath, args, { cwd: root, encoding: 'utf8', timeout: 180000 });
  writeFileSync(join(out, `${config.id}.log`), (run.stdout ?? '') + (run.stderr ?? ''));
  assert.equal(run.status, 0, `${config.id}: ${run.error ?? run.stderr}`);
  const receipt = JSON.parse(readFileSync(path));
  assert(receipt.no_test && !receipt.metrics.local.test && !receipt.metrics.jev);
  assert(receipt.training.trained);
  const result = { ...config, validation: summarize(receipt.metrics.local.validation),
    transfer: summarize(receipt.metrics.local.transfer), receipt: `${config.id}.json` };
  results.push(result);
  console.log(JSON.stringify(result));
}
const baseline = results[0];
for (const r of results) r.eligible =
  r.validation.department >= baseline.validation.department &&
  r.validation.mean_accuracy >= baseline.validation.mean_accuracy &&
  r.validation.ece <= baseline.validation.ece &&
  r.transfer.mean_accuracy >= baseline.transfer.mean_accuracy - 0.01;
const ranked = results.filter(r => r.eligible).sort((a,b) =>
  b.validation.mean_accuracy - a.validation.mean_accuracy ||
  b.validation.department - a.validation.department || a.validation.ece - b.validation.ece ||
  Number(b.model.endsWith('int8')) - Number(a.model.endsWith('int8')));
const selected = ranked[0] ?? baseline;
const summary = { schema: 'typesafe-bench/selection@1',
  plan_sha256: createHash('sha256').update(planText).digest('hex'),
  plan, results, selected, completed_at: new Date().toISOString() };
writeFileSync(join(out, 'selection.json'), JSON.stringify(summary, null, 2) + '\n');
writeFileSync(join(out, 'selected-config.json'), JSON.stringify({ model: selected.model,
  engine_options: selected.engine_options, selection_plan_sha256: summary.plan_sha256 }, null, 2) + '\n');
console.log(`Selected ${selected.id}; test remains unscored by this script.`);
