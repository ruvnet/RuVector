#!/usr/bin/env node
/**
 * RaBitQ (@ruvector/rabitq-wasm) is an opt-in extra.
 *
 * (a) manifest: optional peer only, never a hard dependency; rvlite (used by
 *     the rvlite_* MCP tools via dynamic import) is declared the same way.
 * (b) missing: the compiled wrapper, copied where no node_modules is reachable,
 *     degrades with an error pointing at `npx ruvector add rabitq`.
 * (c) mock: wrapper maps numeric hits back to caller ids and frees wasm objects.
 * (d) real: when @ruvector/rabitq-wasm resolves (devDependency), end-to-end.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const ROOT = path.join(__dirname, '..');
const WRAPPER = path.join(ROOT, 'dist', 'core', 'rabitq-wrapper.js');

let passed = 0;
let failed = 0;
function test(name, fn) {
  try {
    fn();
    passed++;
    console.log(`  PASS  ${name}`);
  } catch (e) {
    failed++;
    console.log(`  FAIL  ${name}\n        ${e.message}`);
  }
}

function isolatedWrapper(withMock) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-rabitq-'));
  fs.copyFileSync(WRAPPER, path.join(dir, 'rabitq-wrapper.js'));
  if (withMock) {
    const modDir = path.join(dir, 'node_modules', '@ruvector', 'rabitq-wasm', 'node');
    fs.mkdirSync(modDir, { recursive: true });
    fs.writeFileSync(path.join(modDir, 'ruvector_rabitq_wasm.js'), `
      const log = (globalThis.__rabitqMockLog = { builds: [], freed: 0 });
      class Hit { constructor(id, d) { this.id = id; this.distance = d; } free() { log.freed++; } }
      class RabitqIndex {
        static build(vectors, dim, seed, rerank) {
          log.builds.push({ n: vectors.length / dim, dim, seed, rerank, isF32: vectors instanceof Float32Array });
          const i = new RabitqIndex(); i.len = vectors.length / dim; return i;
        }
        search(q, k) { return [new Hit(2, 0.5), new Hit(0, 1.5)].slice(0, k); }
        free() { log.freed++; }
      }
      module.exports = { RabitqIndex, version: () => '0.1.1-mock' };
    `);
  }
  return require(path.join(dir, 'rabitq-wrapper.js'));
}

console.log('\nRaBitQ optional extra');

test('manifest: rabitq-wasm and rvlite are optional peers, never dependencies', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  for (const name of ['@ruvector/rabitq-wasm', 'rvlite']) {
    assert.ok(!(name in (pkg.dependencies || {})), `${name} must not be a hard dependency`);
    assert.ok(name in (pkg.peerDependencies || {}), `${name} must be a peer dependency`);
    assert.strictEqual(pkg.peerDependenciesMeta[name].optional, true, `${name} peer must be optional`);
  }
});

test('manifest: no new hard dependencies were added for extras', () => {
  const pkg = JSON.parse(fs.readFileSync(path.join(ROOT, 'package.json'), 'utf8'));
  assert.deepStrictEqual(Object.keys(pkg.dependencies).sort(), [
    '@modelcontextprotocol/sdk', '@ruvector/attention', '@ruvector/core', '@ruvector/gnn',
    '@ruvector/sona', 'chalk', 'commander', 'ora',
  ]);
});

test('missing package: isRabitqAvailable() is false and errors point at ruvector add', () => {
  const w = isolatedWrapper(false);
  assert.strictEqual(w.isRabitqAvailable(), false);
  assert.throws(
    () => w.RabitqIndex.build(['a'], [[1, 2]], { dim: 2 }),
    (e) => /npx ruvector add rabitq/.test(e.message) && /@ruvector\/rabitq-wasm/.test(e.message),
  );
});

test('mock package: ids round-trip, inputs flattened to Float32Array, hits freed', () => {
  const w = isolatedWrapper(true);
  assert.strictEqual(w.isRabitqAvailable(), true);
  const idx = w.RabitqIndex.build(['a', 'b', 'c'], [[1, 0], [0, 1], [1, 1]], { dim: 2, seed: 7 });
  const log = globalThis.__rabitqMockLog;
  assert.deepStrictEqual(log.builds[0], { n: 3, dim: 2, seed: 7n, rerank: 20, isF32: true });
  assert.strictEqual(idx.size, 3);
  assert.strictEqual(idx.dim, 2);
  const hits = idx.search([1, 1], 2);
  assert.deepStrictEqual(hits, [{ id: 'c', distance: 0.5 }, { id: 'a', distance: 1.5 }]);
  assert.strictEqual(log.freed, 2, 'wasm SearchResult objects must be freed');
  idx.free();
  assert.strictEqual(log.freed, 3);
});

test('input validation happens before touching the wasm module', () => {
  const w = isolatedWrapper(true);
  assert.throws(() => w.RabitqIndex.build(['a'], [[1, 2]], { dim: 0 }), /dim/);
  assert.throws(() => w.RabitqIndex.build(['a', 'b'], [[1, 2]], { dim: 2 }), /ids/);
  assert.throws(() => w.RabitqIndex.build(['a'], [[1, 2, 3]], { dim: 2 }), /dim/);
  assert.throws(() => w.RabitqIndex.build([], [], { dim: 2 }), /empty/);
  const idx = w.RabitqIndex.build(['a'], [[1, 2]], { dim: 2 });
  assert.throws(() => idx.search([1, 2, 3], 1), /dim/);
  assert.throws(() => idx.search([1, 2], 0), /k/);
});

let realAvailable = false;
try { require.resolve('@ruvector/rabitq-wasm/node/ruvector_rabitq_wasm.js', { paths: [ROOT] }); realAvailable = true; } catch {}

if (realAvailable) {
  test('real @ruvector/rabitq-wasm: nearest neighbour of a stored vector is itself', () => {
    const { RabitqIndex } = require(WRAPPER);
    const dim = 64;
    const n = 500;
    const ids = [];
    const rows = [];
    for (let i = 0; i < n; i++) {
      ids.push(`doc-${i}`);
      rows.push(Array.from({ length: dim }, (_, j) => Math.sin(i * 0.37 + j * 1.13) + Math.cos(i * j * 0.01)));
    }
    const idx = RabitqIndex.build(ids, rows, { dim, seed: 42 });
    assert.strictEqual(idx.size, n);
    for (const probe of [0, 123, 499]) {
      const hits = idx.search(rows[probe], 5);
      assert.strictEqual(hits[0].id, `doc-${probe}`);
      assert.ok(hits[0].distance <= hits[1].distance);
    }
    idx.free();
  });
} else {
  console.log('  SKIP  real @ruvector/rabitq-wasm not installed');
}

console.log(`\nResults: ${passed} passed, ${failed} failed`);
process.exit(failed ? 1 : 0);
