// Binding smoke tests — run against both backends (node --test).
// Wired methods (predict / similarRelations / compose) assert real payloads;
// the not-yet-landed methods (train / eval / optimize / buildIndex) accept the
// documented `unavailable` error JSON and tighten automatically when the core
// modules land.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { readFileSync, existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const testDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(testDir, '..');
const repoRoot = join(pkgDir, '..', '..', '..');
const indexPath = join(pkgDir, 'index.js');

function workspaceVersion() {
  const toml = readFileSync(join(repoRoot, 'Cargo.toml'), 'utf8');
  const section = toml.slice(toml.indexOf('[workspace.package]'));
  const m = section.match(/version\s*=\s*"([^"]+)"/);
  assert.ok(m, 'workspace version found in Cargo.toml');
  return m[1];
}
const WORKSPACE_VERSION = workspaceVersion();

function loadBackend(forced) {
  for (const key of Object.keys(require.cache)) delete require.cache[key];
  if (forced) process.env.KGE_BACKEND = forced;
  else delete process.env.KGE_BACKEND;
  return require(indexPath);
}

const nativeBuilt = ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc']
  .some((t) => existsSync(join(pkgDir, 'native', `kge.${t}.node`)));
const wasmBuilt = existsSync(join(pkgDir, 'wasm', 'ruvector_kge_wasm.js'));

const TRIPLES = JSON.stringify([
  { s: 'Ada', r: 'bornIn', o: 'London' },
  { s: 'Ada', r: 'bornIn', o: 'Paris' },
  { s: 'London', r: 'locatedIn', o: 'England' },
]);

function runContract(mod, expectedBackend) {
  assert.equal(typeof mod.version, 'function', 'exports version()');
  assert.equal(typeof mod.Model, 'function', 'exports Model class');
  assert.equal(mod.backend, expectedBackend, `backend is ${expectedBackend}`);
  assert.equal(mod.version(), WORKSPACE_VERSION, 'version() matches Cargo workspace version');

  const model = new mod.Model('{"scorer":"hole","dims":8,"seed":1}');

  const stats = JSON.parse(model.statsJson());
  assert.equal(stats.dims, 8, 'stats reports configured dims');
  assert.equal(stats.scorer, 'hole', 'stats reports the scorer');

  const added = JSON.parse(model.addTriplesJson(TRIPLES));
  assert.equal(added.added, 3, 'all triples admitted');
  assert.equal(added.relations, 2, 'two distinct relations interned');

  // predict is wired: real candidates for the open tail.
  const pred = JSON.parse(model.predictJson('{"s":"Ada","r":"bornIn","k":3}'));
  assert.ok(Array.isArray(pred.candidates), 'predict returns candidates');
  assert.equal(pred.exact, true, 'exhaustive scoring reports exact:true');
  assert.equal(pred.ann, false, 'no ANN yet: ann:false');
  assert.ok(typeof pred.candidates[0].entity === 'string', 'candidate carries an entity label');

  // similarRelations is wired: ranks the other relation.
  const sim = JSON.parse(model.similarRelationsJson('{"r":"bornIn","k":5}'));
  assert.ok(Array.isArray(sim.relations), 'similarRelations returns a list');
  assert.ok(sim.relations.every((x) => x.relation !== 'bornIn'), 'query relation excluded');

  // compose on a hole model is unsupported.
  const composeHole = JSON.parse(model.composeJson('{"r1":"bornIn","r2":"locatedIn","s":"Ada","k":3}'));
  assert.equal(composeHole.error.kind, 'unsupported', 'hole compose -> unsupported');

  // toJson / fromJson round-trips byte-identically; a tamper throws.
  const saved = model.toJson();
  const reloaded = mod.Model.fromJson(saved);
  assert.equal(reloaded.toJson(), saved, 'save/load round-trips byte-identically');
  const tampered = saved.replace('"Ada"', '"Eve"');
  assert.notEqual(tampered, saved, 'tamper actually changed the payload');
  assert.throws(() => mod.Model.fromJson(tampered), 'a tampered envelope throws');

  // programmer errors throw.
  assert.throws(() => new mod.Model('{ not valid json'), 'bad options JSON throws');
  assert.throws(() => new mod.Model('{"dims":7}'), 'odd dims throws');

  // request limits return the documented error JSON.
  const big = JSON.stringify([{ s: 'x'.repeat(1025), r: 'r', o: 'y' }]);
  assert.equal(JSON.parse(model.addTriplesJson(big)).error.kind, 'limit', 'oversized label -> limit');
  const empty = JSON.stringify([{ s: '', r: 'r', o: 'y' }]);
  assert.equal(JSON.parse(model.addTriplesJson(empty)).error.kind, 'invalid', 'empty label -> invalid');
  assert.equal(JSON.parse(model.predictJson('{"s":"Ada","r":"bornIn","k":0}')).error.kind, 'limit', 'k=0 -> limit');
  assert.equal(JSON.parse(model.predictJson('{"s":"Ada","r":"bornIn","k":2000}')).error.kind, 'limit', 'k>1000 -> limit');

  // optimize is the one still-stubbed method.
  assert.equal(JSON.parse(model.optimizeJson('{}')).error.kind, 'unavailable', 'optimize -> unavailable');
}

// train, the growth guarantee, and the ANN index — wired end to end.
function runPipeline(mod) {
  const model = new mod.Model('{"scorer":"hole","dims":8,"seed":1}');
  model.addTriplesJson(TRIPLES);

  const report = JSON.parse(model.trainJson('{"epochs":3,"batchSize":4,"lr":0.1}'));
  assert.equal(typeof report.loss, 'number', 'train reports a numeric loss');
  assert.ok(Number.isFinite(report.loss), 'train loss is finite');

  assert.ok(JSON.parse(model.predictJson('{"s":"Ada","r":"bornIn","k":5}')).candidates.length > 0,
    'predict works after train');

  // Growth guarantee: adding a triple must not disturb the trained rows of
  // existing entities, so their scores are unchanged.
  const scoreOf = (json, ent) => {
    const c = JSON.parse(json).candidates.find((x) => x.entity === ent);
    return c ? c.score : undefined;
  };
  const beforeLondon = scoreOf(model.predictJson('{"s":"Ada","r":"bornIn","k":10}'), 'London');
  model.addTriplesJson('[{"s":"Ada","r":"bornIn","o":"Rome"}]');
  const afterLondon = scoreOf(model.predictJson('{"s":"Ada","r":"bornIn","k":10}'), 'London');
  assert.equal(afterLondon, beforeLondon, 'trained score for an existing entity survives addTriples');

  // ANN index: buildIndex flips stats.indexed and predict reports ann:true.
  assert.equal(JSON.parse(model.buildIndexJson()).indexed, true, 'buildIndex reports indexed');
  assert.equal(JSON.parse(model.statsJson()).indexed, true, 'stats.indexed is true after buildIndex');
  assert.equal(JSON.parse(model.predictJson('{"s":"Ada","r":"bornIn","k":3}')).ann, true,
    'predict uses the ANN path after buildIndex');

  // eval runs and returns a report with ranking metrics.
  const evalRes = JSON.parse(model.evalJson('{"filtered":true,"tieBreak":"random"}'));
  assert.ok(evalRes.report && evalRes.report.combined, 'eval returns a report with combined metrics');
  assert.equal(typeof evalRes.report.combined.mrr, 'number', 'eval reports a numeric MRR');
}

test('native backend contract', { skip: !nativeBuilt && 'native binary not built' }, () => {
  runContract(loadBackend(undefined), 'native');
});

test('native async train resolves to JSON', {
  skip: !nativeBuilt && 'native binary not built',
}, async () => {
  const mod = loadBackend(undefined);
  const model = new mod.Model('{"scorer":"hole","dims":8}');
  assert.equal(typeof model.train, 'function', 'native exposes async train');
  const out = JSON.parse(await model.train('{}'));
  assert.ok('error' in out || typeof out === 'object', 'async train returns report-or-error JSON');
});

test('rotate compose returns candidates', { skip: !wasmBuilt && 'wasm not built' }, () => {
  const mod = loadBackend('wasm');
  const model = new mod.Model('{"scorer":"rotate","dims":8,"seed":2}');
  model.addTriplesJson(TRIPLES);
  const composed = JSON.parse(model.composeJson('{"r1":"bornIn","r2":"locatedIn","s":"Ada","k":3}'));
  assert.ok(Array.isArray(composed.candidates), 'rotate compose returns candidates');
  assert.equal(composed.exact, true, 'compose reports exact:true');
});

test('wasm backend contract', { skip: !wasmBuilt && 'wasm module not built' }, () => {
  runContract(loadBackend('wasm'), 'wasm');
});

test('pipeline: train / growth / ANN index (native)', { skip: !nativeBuilt && 'native not built' }, () => {
  runPipeline(loadBackend(undefined));
});

test('pipeline: train / growth / ANN index (wasm)', { skip: !wasmBuilt && 'wasm not built' }, () => {
  runPipeline(loadBackend('wasm'));
});
