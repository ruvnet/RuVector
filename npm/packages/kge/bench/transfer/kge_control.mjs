// Target-trained native KGE baseline on the same explicit splits as ULTRA.
// Usage: node kge_control.mjs INPUT.json /absolute/path/to/kge/index.js OUTPUT.json
import { readFileSync, writeFileSync } from 'node:fs';
import { createHash } from 'node:crypto';
import { createRequire } from 'node:module';
import { resolve } from 'node:path';
import { performance } from 'node:perf_hooks';
import assert from 'node:assert/strict';

const [input, bindingPath, output] = process.argv.slice(2);
assert(input && bindingPath && output, 'input, explicit binding path, output required');
const hash = value => createHash('sha256').update(value).digest('hex');
const bytes = readFileSync(input);
const payload = JSON.parse(bytes);
const require = createRequire(import.meta.url);
const binding = require(resolve(bindingPath));
assert.equal(binding.backend, 'native', 'native baseline required; no fallback');
const rows = payload.triples;
assert(Array.isArray(rows) && rows.length > 0 && rows.length <= 100000);
const seen = new Set();
for (const row of rows) {
  assert(['train', 'valid', 'test'].includes(row.split));
  for (const key of ['s', 'r', 'o']) assert(typeof row[key] === 'string' && row[key].length > 0);
  const key = JSON.stringify([row.s, row.r, row.o]);
  assert(!seen.has(key), 'duplicate / cross-split overlap');
  seen.add(key);
}
const parse = raw => {
  const value = JSON.parse(raw);
  assert(!value.error, JSON.stringify(value.error));
  return value;
};
const report = { schema: 'ruvector.kge-transfer-control.v1', graph: payload.name,
  inputSha256: hash(bytes), runnerSha256: hash(readFileSync(new URL(import.meta.url))),
  engineVersion: binding.version(), backend: binding.backend,
  scorer:'hole', config: {dims:64, epochs:50, batch_size:256, lr:0.1,
    optimizer:{kind:'adagrad', epsilon:1e-8}, loss:{kind:'one_vs_all'}, n3_lambda:0.001},
  arms: [], promotion:false,
  limits:'Target-only KGE scratch control. Not equal total pretraining compute; no source replay.' };
for (const seed of [101, 202, 303]) {
  const model = new binding.Model(JSON.stringify({scorer:'hole', dims:64, seed}));
  parse(model.addTriplesJson(JSON.stringify(rows)));
  const before = parse(model.statsJson());
  assert.equal(before.splits.test, rows.filter(x => x.split === 'test').length);
  const started = performance.now();
  const training = parse(model.trainJson(JSON.stringify({...report.config,seed})));
  assert.equal(training.epochs, 50);
  assert.equal(training.batches, Math.ceil(rows.filter(x=>x.split==='train').length / 256));
  assert(Number.isFinite(training.loss));
  const trainingMs = performance.now()-started;
  const frozen = hash(model.toJson());
  const evalStart = performance.now();
  const evaluation = parse(model.evalJson(JSON.stringify({split:'test', tieBreak:'bottom'})));
  const m = evaluation.report?.combined;
  assert(m && Number.isFinite(m.mrr) && m.mrr > 0 && m.mrr <= 1);
  assert.equal(m.count, 2 * before.splits.test);
  assert.equal(evaluation.filtered, true);
  assert.equal(hash(model.toJson()), frozen, 'evaluation mutated checkpoint');
  report.arms.push({seed,training,trainingMs,evaluationMs:performance.now()-evalStart,
    checkpointSha256:frozen,evaluation});
}
writeFileSync(output, JSON.stringify(report,null,2)+'\n');
console.log(JSON.stringify({output, mrr:report.arms.map(x=>x.evaluation.report.combined.mrr)}));
