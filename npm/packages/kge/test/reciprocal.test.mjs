// Plan M3 binding tests: the `complex` scorer, reciprocal models and the
// recipe fields of trainJson — against every built backend (native, wasm).
// Skips cleanly when neither has been built.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const pkgDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const indexPath = join(pkgDir, 'index.js');
const distIndex = join(pkgDir, 'dist', 'index.js');

function loadBackend(forced) {
  for (const key of Object.keys(require.cache)) delete require.cache[key];
  process.env.KGE_BACKEND = forced;
  const mod = require(indexPath);
  delete process.env.KGE_BACKEND;
  return mod;
}

const nativeBuilt = ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc']
  .some((t) => existsSync(join(pkgDir, 'native', `kge.${t}.node`)));
const wasmBuilt = existsSync(join(pkgDir, 'wasm', 'ruvector_kge_wasm.js'));

const RECIPE = {
  loss: { kind: 'one_vs_all' },
  n3_form: 'moduli',
  loss_reduction: 'mean',
  init: { kind: 'normal', scale: 0.001 },
  optimizer: { kind: 'adagrad' },
  optim_state: 'dense',
  n3_lambda: 0.01,
  rp_weight: 0.05,
  one_n_kernel: 'gemm',
  epochs: 3,
  batch_size: 16,
  lr: 0.1,
  seed: 3,
};

function graph(ne, nr, nt) {
  const out = [];
  for (let i = 0; i < nt; i++) {
    const s = i % ne;
    const r = Math.floor(i / ne) % nr;
    out.push({ s: `e${s}`, r: `r${r}`, o: `e${(s * 7 + r * 13 + 1) % ne}` });
  }
  return JSON.stringify(out);
}

const kindOf = (json) => JSON.parse(json).error?.kind;

function runContract(mod) {
  const m = new mod.Model('{"scorer":"complex","dims":8,"seed":5,"reciprocal":true}');
  assert.equal(JSON.parse(m.addTriplesJson(graph(30, 3, 90))).added, 90);
  const stats = JSON.parse(m.statsJson());
  assert.equal(stats.scorer, 'complex');
  assert.equal(stats.reciprocal, true);
  assert.equal(stats.relations, 3, 'relations count labels, not 2R rows');

  // trainJson takes the recipe fields; `reciprocal` must match the model.
  const trained = JSON.parse(m.trainJson(JSON.stringify(RECIPE)));
  assert.equal(trained.status, 'trained', JSON.stringify(trained));
  assert.ok(Number.isFinite(trained.loss));
  assert.equal(kindOf(m.trainJson('{"reciprocal":false,"epochs":1}')), 'invalid');

  // Eval uses the reciprocal protocol and reports it.
  const ev = JSON.parse(m.evalJson('{"tieBreak":"bottom","seed":1}'));
  assert.equal(ev.reciprocal, true);
  assert.equal(ev.report.combined.count, 180);
  assert.ok(ev.report.combined.mrr > 0 && ev.report.combined.mrr <= 1);

  // Head predict (answered as (o, r⁻¹, ?)) and similarRelations over labels only.
  const head = JSON.parse(m.predictJson('{"r":"r1","o":"e4","k":3,"useIndex":false}'));
  assert.equal(head.candidates.length, 3);
  const sim = JSON.parse(m.similarRelationsJson('{"r":"r0","k":10}'));
  assert.deepEqual(sim.relations.map((r) => r.relation).sort(), ['r1', 'r2']);
  assert.equal(kindOf(m.composeJson('{"r1":"r0","r2":"r1","s":"e1"}')), 'unsupported');
  assert.equal(kindOf(m.optimizeJson('{}')), 'unsupported');

  // Round-trips with the flag; a non-reciprocal model's envelope never carries it.
  const back = mod.Model.fromJson(m.toJson());
  assert.equal(JSON.parse(back.statsJson()).reciprocal, true);
  const plain = new mod.Model('{"scorer":"complex","dims":8,"seed":5}');
  plain.addTriplesJson(graph(10, 2, 20));
  assert.ok(!plain.toJson().includes('reciprocal'));
  assert.equal(kindOf(plain.trainJson('{"reciprocal":true,"epochs":1}')), 'invalid');
  // Backwards compatible: an old-style config still trains a plain model.
  assert.equal(JSON.parse(plain.trainJson('{"epochs":1,"lr":0.05}')).status, 'trained');
}

test('native: complex + reciprocal contract', { skip: !nativeBuilt && 'native not built' }, () => {
  runContract(loadBackend('native'));
});

test('wasm: complex + reciprocal contract', { skip: !wasmBuilt && 'wasm not built' }, () => {
  runContract(loadBackend('wasm'));
});

test('toOptionsJson: reciprocal sent only when set', { skip: !existsSync(distIndex) && 'run npm run build first' }, async () => {
  const { toOptionsJson } = await import(distIndex);
  assert.equal(toOptionsJson({ scorer: 'complex', dims: 8, seed: 1 }), '{"scorer":"complex","dims":8,"seed":1}');
  assert.equal(
    toOptionsJson({ scorer: 'complex', dims: 8, seed: 1, reciprocal: true }),
    '{"scorer":"complex","dims":8,"seed":1,"reciprocal":true}',
  );
});
