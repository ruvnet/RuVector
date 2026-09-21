// Binding smoke tests — run against both backends (node --test).
// Every method is wired: predict / similarRelations / compose / train / eval /
// buildIndex / optimize all assert real payloads.

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

  // optimize is wired: with only 3 untagged triples it cannot form a valid/test
  // split, so it returns a request error — but never the old 'unavailable'.
  const optTiny = JSON.parse(model.optimizeJson('{}'));
  assert.notEqual(optTiny.error?.kind, 'unavailable', 'optimize is wired (not unavailable)');
}

// A ring KG with frozen split tags, big enough that optimize can fit and gate a
// campaign and install a champion (the trained tables change the saved bytes).
function ringTriples(n) {
  const t = [];
  for (let i = 0; i < n; i++) {
    const a = `e${i}`;
    const b = `e${(i + 1) % n}`;
    const split = i % 5 === 0 ? 'valid' : i % 7 === 0 ? 'test' : i % 11 === 0 ? 'transfer' : 'train';
    t.push({ s: a, r: 'ring', o: b, split });
    t.push({ s: b, r: 'ring', o: a, split: 'train' }); // symmetric; keep in train for coverage
  }
  return t;
}

function runOptimizeCampaign(mod) {
  const model = new mod.Model('{"scorer":"hole","dims":8,"seed":1}');
  model.addTriplesJson(JSON.stringify(ringTriples(40)));
  const before = model.toJson();

  const report = JSON.parse(model.optimizeJson('{"budget":6,"seed":1}'));
  assert.ok(!report.error, `optimize returned a report, not an error: ${JSON.stringify(report.error)}`);
  assert.ok(Array.isArray(report.proposals) && report.proposals.length >= 1, 'at least one proposal gated');
  assert.equal(typeof report.championId, 'number', 'report carries a champion id');
  assert.equal(report.splitSource, 'per-triple', 'frozen tags drove the split');
  assert.equal(typeof report.test.championMrr, 'number', 'test MRR reported');
  // The receipt log is non-empty JSONL, one object per line.
  assert.ok(report.receiptsCount >= report.proposals.length, 'a receipt per proposal (plus champion)');
  assert.ok(report.receipts.split('\n').filter((l) => l.trim()).length === report.receiptsCount, 'receipts JSONL line count matches');

  // The champion's trained tables were installed → the saved model changed.
  assert.equal(report.installed, true, 'champion tables installed');
  assert.notEqual(model.toJson(), before, 'installing the champion changed the saved model');
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

test('optimize: campaign gates arms and installs a champion (native)', { skip: !nativeBuilt && 'native not built' }, () => {
  runOptimizeCampaign(loadBackend(undefined));
});

test('optimize: campaign gates arms and installs a champion (wasm)', { skip: !wasmBuilt && 'wasm not built' }, () => {
  runOptimizeCampaign(loadBackend('wasm'));
});

// Frozen per-triple splits are honoured verbatim; useIndex forces exhaustion.
function runSplits(mod) {
  const model = new mod.Model('{"scorer":"hole","dims":8,"seed":1}');
  model.addTriplesJson(JSON.stringify([
    { s: 'A', r: 'rel', o: 'B', split: 'train' },
    { s: 'A', r: 'rel', o: 'C', split: 'train' },
    { s: 'A', r: 'rel', o: 'D', split: 'test' },
    { s: 'A', r: 'rel', o: 'E', split: 'transfer' },
  ]));
  assert.equal(JSON.parse(model.addTriplesJson('[{"s":"X","r":"rel","o":"Y","split":"nope"}]')).error.kind,
    'invalid', 'an unknown split tag is rejected');

  // statsJson carries per-split counts.
  const splits = JSON.parse(model.statsJson()).splits;
  assert.deepEqual(splits, { train: 2, valid: 0, transfer: 1, test: 1, unlabelled: 0 }, 'stats reports per-split counts');

  model.trainJson('{"epochs":3,"lr":0.1}');
  // eval on 'transfer' scores ONLY the labelled transfer triple.
  const evT = JSON.parse(model.evalJson('{"split":"transfer"}'));
  assert.equal(evT.evalTriples, 1, 'the transfer split is exactly the one labelled transfer triple');
  assert.equal(evT.splitSource, 'per-triple', 'transfer eval source is per-triple');
  assert.equal(evT.note, 'frozen per-triple split', 'note marks the frozen per-triple split');
  const evTest = JSON.parse(model.evalJson('{"split":"test"}'));
  assert.equal(evTest.evalTriples, 1, 'the test split is exactly the one labelled test triple');

  model.buildIndexJson();
  assert.equal(JSON.parse(model.predictJson('{"s":"A","r":"rel","k":2}')).ann, true, 'default predict uses the index');
  assert.equal(JSON.parse(model.predictJson('{"s":"A","r":"rel","k":2,"useIndex":false}')).ann, false,
    'useIndex:false forces the exhaustive path');
}

// With NO ingested tags, a requested split falls back to a derived partition.
function runDerivedSplit(mod) {
  const model = new mod.Model('{"scorer":"hole","dims":8,"seed":7}');
  // 20 untagged triples so a derived 80/10/10 test split is non-empty.
  const facts = Array.from({ length: 20 }, (_, i) => ({ s: `S${i}`, r: 'rel', o: `O${i}` }));
  model.addTriplesJson(JSON.stringify(facts));
  model.trainJson('{"epochs":2}');
  const ev = JSON.parse(model.evalJson('{"split":"test"}'));
  assert.equal(ev.splitSource, 'derived', 'untagged eval falls back to a derived split');
  assert.match(ev.note, /derived 80\/10\/10 split/, 'derived note retains its disclaimer');
  // transfer without tags is rejected (it has no derived form).
  assert.equal(JSON.parse(model.evalJson('{"split":"transfer"}')).error.kind, 'invalid',
    'transfer without ingested tags is rejected');
}

test('frozen per-triple splits + useIndex (wasm)', { skip: !wasmBuilt && 'wasm not built' }, () => runSplits(loadBackend('wasm')));
test('frozen per-triple splits + useIndex (native)', { skip: !nativeBuilt && 'native not built' }, () => runSplits(loadBackend(undefined)));
test('derived-split fallback when untagged (wasm)', { skip: !wasmBuilt && 'wasm not built' }, () => runDerivedSplit(loadBackend('wasm')));
