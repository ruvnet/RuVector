// F1: embedding-table size limits enforced at the binding surface — on
// construction, on addTriples growth and on load. Runs against every built
// backend; skips cleanly when none is built.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const pkgDir = join(dirname(fileURLToPath(import.meta.url)), '..');
const indexPath = join(pkgDir, 'index.js');

function loadBackend(forced) {
  for (const key of Object.keys(require.cache)) delete require.cache[key];
  process.env.KGE_BACKEND = forced;
  return require(indexPath);
}

const nativeBuilt = ['linux-x64-gnu', 'linux-arm64-gnu', 'darwin-x64', 'darwin-arm64', 'win32-x64-msvc']
  .some((t) => existsSync(join(pkgDir, 'native', `kge.${t}.node`)));
const wasmBuilt = existsSync(join(pkgDir, 'wasm', 'ruvector_kge_wasm.js'));

/** The kind of a `{"error":{"kind"}}` envelope, from a string or a throw. */
function kindOf(value) {
  const text = value instanceof Error ? value.message : String(value);
  try {
    return JSON.parse(text)?.error?.kind;
  } catch {
    return undefined;
  }
}

function throwsKind(fn, kind) {
  let caught;
  try {
    fn();
  } catch (e) {
    caught = e;
  }
  assert.ok(caught !== undefined, 'expected a throw');
  assert.equal(kindOf(caught), kind, `thrown: ${caught}`);
}

function runLimits(mod) {
  // 50,000 dims: rejected at construction, before any table exists.
  throwsKind(() => new mod.Model('{"dims":50000}'), 'limit');
  assert.ok(new mod.Model('{"dims":4096}'));

  // Growth past a (lowered) byte cap: the whole batch is rejected, no mutation.
  const m = new mod.Model('{"dims":64,"maxTableBytes":4096}');
  assert.equal(kindOf(m.addTriplesJson('[{"s":"a","r":"r","o":"b"}]')), undefined);
  const before = m.toJson();
  const batch = Array.from({ length: 20 }, (_, i) => ({ s: `e${i}`, r: 'r', o: `f${i}` }));
  assert.equal(kindOf(m.addTriplesJson(JSON.stringify(batch))), 'limit');
  assert.equal(m.toJson(), before, 'rejected batch must not mutate the model');

  // Load of an oversized model: rejected by the size gate, before the hash.
  const oversized = JSON.stringify({
    sha256: '00',
    model: {
      version: 1,
      config: { scorer: 'hole', dims: 50000, seed: 1 },
      entities: { labels: ['a', 'b'] },
      relations: { labels: ['r'] },
      triples: [],
      tables: null,
    },
  });
  throwsKind(() => mod.Model.fromJson(oversized), 'limit');

  // In-range models still round-trip byte-identically.
  const ok = new mod.Model('{"dims":8}');
  ok.addTriplesJson('[{"s":"a","r":"r","o":"b"}]');
  const saved = ok.toJson();
  assert.equal(mod.Model.fromJson(saved).toJson(), saved);
}

test('F1 limits (native)', { skip: !nativeBuilt && 'native addon not built' }, () => {
  runLimits(loadBackend('native'));
});

test('F1 limits (wasm)', { skip: !wasmBuilt && 'wasm fallback not built' }, () => {
  runLimits(loadBackend('wasm'));
});
