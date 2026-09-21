// TypeScript client tests, run against the compiled dist/ and the real binding
// (native or wasm). Skips cleanly if the package has not been built yet.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const testDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(testDir, '..');
const distIndex = join(pkgDir, 'dist', 'index.js');
const built = existsSync(distIndex);

const load = () => import(distIndex);

const FACTS = [
  { s: 'Ada', r: 'bornIn', o: 'London' },
  { s: 'Ada', r: 'bornIn', o: 'Paris' },
  { s: 'London', r: 'locatedIn', o: 'England' },
];

test('createKge: add / predict / similar', { skip: !built && 'run npm run build first' }, async () => {
  const { createKge, defineSchema } = await load();
  const schema = defineSchema({ relations: ['bornIn', 'locatedIn'] });
  const kge = createKge({ scorer: 'hole', dims: 8, seed: 1, schema });
  assert.equal(typeof kge.version, 'string', 'version present');
  assert.ok(['native', 'wasm'].includes(kge.backend), 'backend is native or wasm');

  const report = kge.addTriples(FACTS);
  assert.equal(report.added, 3, 'all facts admitted');

  const pred = kge.predict({ s: 'Ada', r: 'bornIn', k: 3 });
  assert.ok(pred.candidates.length > 0, 'predict returns candidates');
  assert.equal(pred.exact, true, 'exhaustive path reports exact');

  const sim = kge.similarRelations({ r: 'bornIn', k: 5 });
  assert.ok(Array.isArray(sim.relations), 'similarRelations returns a list');
});

test('compose: rotate works, hole is unsupported', { skip: !built && 'not built' }, async () => {
  const { createKge, KgeError } = await load();
  const rot = createKge({ scorer: 'rotate', dims: 8, seed: 2 });
  rot.addTriples(FACTS);
  const composed = rot.compose({ r1: 'bornIn', r2: 'locatedIn', s: 'Ada', k: 3 });
  assert.ok(composed.candidates.length > 0, 'rotate compose returns candidates');

  const hole = createKge({ scorer: 'hole', dims: 8 });
  hole.addTriples(FACTS);
  assert.throws(
    () => hole.compose({ r1: 'bornIn', r2: 'locatedIn', s: 'Ada' }),
    (e) => e instanceof KgeError && e.kind === 'unsupported',
    'hole compose throws KgeError{unsupported}',
  );
});

test('save / loadKge round-trips', { skip: !built && 'not built' }, async () => {
  const { createKge, loadKge } = await load();
  const kge = createKge({ scorer: 'hole', dims: 8, seed: 3 });
  kge.addTriples(FACTS);
  const envelope = kge.save();
  const reloaded = loadKge(envelope);
  assert.equal(reloaded.save(), envelope, 'reloaded model saves identically');
  const a = kge.predict({ s: 'Ada', r: 'bornIn', k: 2 });
  const b = reloaded.predict({ s: 'Ada', r: 'bornIn', k: 2 });
  assert.deepEqual(b.candidates, a.candidates, 'predictions match after reload');
});

test('train / evaluate / buildIndex work; optimize is unavailable', { skip: !built && 'not built' }, async () => {
  const { createKge, KgeError } = await load();
  const kge = createKge({ scorer: 'hole', dims: 8, seed: 4 });
  kge.addTriples(FACTS);

  const report = await kge.train({ epochs: 2, lr: 0.1 });
  assert.equal(typeof report.loss, 'number', 'train returns a numeric loss');

  const evalReport = kge.evaluate({ filtered: true });
  assert.ok(evalReport.report, 'evaluate returns a report');

  const built2 = kge.buildIndex();
  assert.equal(built2.indexed, true, 'buildIndex reports indexed');
  assert.equal(kge.predict({ s: 'Ada', r: 'bornIn', k: 2 }).ann, true, 'predict uses ANN after buildIndex');

  assert.throws(
    () => kge.optimize({}),
    (e) => e instanceof KgeError && e.kind === 'unavailable',
    'optimize throws KgeError{unavailable}',
  );
});
