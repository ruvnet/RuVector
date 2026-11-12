// `kge demo` tests, run in-process against the compiled dist/ CLI and the real
// binding (native or wasm), like api.test.mjs. Verifies the bundled demo
// actually predicts facts it never trained on: every held-out triple is absent
// from the training set, and the trained model ranks its true tail first.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const testDir = dirname(fileURLToPath(import.meta.url));
const pkgDir = join(testDir, '..');
const mainPath = join(pkgDir, 'dist', 'cli', 'main.js');
const demoPath = join(pkgDir, 'dist', 'cli', 'demo.js');
const built = existsSync(mainPath);

const require = createRequire(import.meta.url);

async function runDemo(argv) {
  const { main } = require(mainPath);
  const out = [];
  const err = [];
  const result = await main(['demo', ...argv], {
    stdout: (s) => out.push(s),
    stderr: (s) => err.push(s),
  });
  return { code: result.code, out: out.join('\n'), err: err.join('\n') };
}

test('demo: held-out facts are not in the training triples', { skip: !built && 'run npm run build first' }, () => {
  assert.ok(existsSync(demoPath), 'dist/cli/demo.js exists');
  const { DEMO_TRIPLES, HELD_OUT } = require(demoPath);
  assert.ok(HELD_OUT.length >= 2, 'at least two held-out queries');
  for (const h of HELD_OUT) {
    const leaked = DEMO_TRIPLES.some((t) => t.s === h.s && t.r === h.r);
    assert.equal(leaked, false, `(${h.s}, ${h.r}, ?) has no training triple`);
  }
});

test('demo --json: the trained model ranks every held-out answer first', { skip: !built && 'run npm run build first' }, async () => {
  const { code, out } = await runDemo(['--json']);
  assert.equal(code, 0, 'demo exits 0');
  const report = JSON.parse(out);
  assert.ok(['native', 'wasm'].includes(report.backend), 'backend reported');
  assert.equal(report.scorer, 'hole');
  assert.ok(report.queries.length >= 2, 'reports every held-out query');
  for (const q of report.queries) {
    assert.equal(q.rank, 1, `(${q.s}, ${q.r}, ?) -> ${q.answer} ranked first`);
    assert.equal(q.candidates[0].entity, q.answer);
    assert.ok(q.candidates.length <= 3, 'default k is 3');
  }
});

test('demo: human-readable output names each query and its held-out answer', { skip: !built && 'run npm run build first' }, async () => {
  const { code, out } = await runDemo(['-k', '2']);
  assert.equal(code, 0, 'demo exits 0');
  assert.match(out, /\(Tim, citizenOf, \?\)/);
  assert.match(out, /held-out answer: UK, rank 1/);
  assert.doesNotMatch(out, /^\s+3\. /m, '-k 2 prints two candidates per query');
});

test('demo: an invalid -k is rejected', { skip: !built && 'run npm run build first' }, async () => {
  const { code, err } = await runDemo(['-k', '0']);
  assert.equal(code, 1);
  assert.match(err, /-k/);
});
