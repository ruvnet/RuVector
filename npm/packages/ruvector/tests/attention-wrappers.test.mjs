/**
 * Issue #1375: the attention wrappers in src/core/attention-fallbacks.ts and
 * `ruvector attention compute -t flash|linear` passed argument shapes the
 * native @ruvector/attention binding does not accept. Every case here runs the
 * real native module (no mocks) and, where a direct native call is possible,
 * compares the wrapper against it as the oracle.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
// Hard dependency: fail, don't skip, if the native module is missing.
const native = require('@ruvector/attention');
const w = require('../dist/core/attention-fallbacks.js');

const DIM = 16;
let seed = 7;
const rand = () => ((seed = (seed * 16807) % 2147483647) / 2147483647 - 0.5) * 0.2;
const vec = () => Float32Array.from({ length: DIM }, rand);
const q = vec();
const K = [vec(), vec(), vec(), vec()];
const V = [vec(), vec(), vec(), vec()];

const assertVector = (out, len = DIM) => {
  assert.ok(Array.isArray(out), `expected number[], got ${Object.prototype.toString.call(out)}`);
  assert.equal(out.length, len);
  for (const x of out) assert.ok(Number.isFinite(x), `non-finite value ${x}`);
};
const assertClose = (a, b, eps = 1e-5) => {
  assert.equal(a.length, b.length);
  for (let i = 0; i < a.length; i++) assert.ok(Math.abs(a[i] - b[i]) <= eps, `index ${i}: ${a[i]} vs ${b[i]}`);
};

test('GraphRoPeAttention builds from (dim, numHeads, maxSeqLen) and computes', () => {
  const attn = new w.GraphRoPeAttention(DIM, 4, 64);
  const out = attn.compute(q, K, V);
  assertVector(out.values);
  const oracle = new native.GraphRoPeAttention({ dim: DIM, maxPosition: 64 });
  assertClose(out.values, Array.from(oracle.compute(q, K, V)));
  // Graph positions route to computeWithPositions.
  const pos = attn.compute(q, K, V, [1, 2, 3, 4], 0);
  assertClose(pos.values, Array.from(oracle.computeWithPositions(q, K, V, 0, [1, 2, 3, 4])));
});

test('DualSpaceAttention builds from (dim, curvature, alpha) and computes', () => {
  const out = new w.DualSpaceAttention(DIM, 1.0, 0.7).compute(q, K, V);
  assertVector(out.values);
  const oracle = native.DualSpaceAttention.withWeights(DIM, 1.0, 0.7, 0.3);
  assertClose(out.values, Array.from(oracle.compute(q, K, V)));
});

test('EdgeFeaturedAttention computes with and without edge features', () => {
  const attn = new w.EdgeFeaturedAttention(DIM, 4);
  assertVector(attn.compute(q, K, V).values);
  const edges = K.map(() => Float32Array.from({ length: 4 }, rand));
  assertVector(attn.compute(q, K, V, edges).values);
});

test('computeFlashAttentionAsync resolves to the native result', async () => {
  const out = await w.computeFlashAttentionAsync(q, K, V, 2);
  assertVector(out);
  assertClose(out, Array.from(await native.computeFlashAttentionAsync(q, K, V, DIM, 2)));
});

test('computeHyperbolicAttentionAsync resolves to the native result', async () => {
  const out = await w.computeHyperbolicAttentionAsync(q, K, V, 1.0);
  assertVector(out);
  assertClose(out, Array.from(await native.computeHyperbolicAttentionAsync(q, K, V, DIM, 1.0)));
});

test('infoNceLoss returns the native loss', () => {
  const loss = w.infoNceLoss(q, K[0], K.slice(1), 0.1);
  assert.ok(Number.isFinite(loss));
  assert.ok(Math.abs(loss - new native.InfoNceLoss(0.1).compute(q, K[0], K.slice(1))) < 1e-6);
});

test('mineHardNegatives returns k candidate vectors', () => {
  const out = w.mineHardNegatives(q, K, 2);
  assert.equal(out.length, 2);
  for (const v of out) {
    assertVector(v);
    assert.ok(K.some((k) => Array.from(k).every((x, i) => x === v[i])), 'mined vector must be one of the candidates');
  }
});

for (const type of ['dot', 'dot_product', 'flash', 'linear', 'hyperbolic', 'local-global']) {
  test(`parallelAttentionCompute(${type}) returns one output per query`, async () => {
    const out = await w.parallelAttentionCompute([q, K[1]], K, V, type);
    assert.equal(out.length, 2);
    out.forEach((o) => assertVector(o));
  });
}

test('parallelAttentionCompute dot matches DotProductAttention', async () => {
  const [out] = await w.parallelAttentionCompute([q], K, V, 'dot');
  assertClose(out, Array.from(new native.DotProductAttention(DIM).compute(q, K, V)));
});

test('parallelAttentionCompute rejects a type with no native batch kernel', async () => {
  await assert.rejects(() => w.parallelAttentionCompute([q], K, V, 'multi-head'), /multi-head/);
});

test('batchAttentionCompute computes each {query, keys, values} set', async () => {
  const out = await w.batchAttentionCompute(
    [{ query: q, keys: K, values: V }, { query: Array.from(K[2]), keys: K.slice(0, 2).map((k) => Array.from(k)), values: V.slice(0, 2) }],
    'dot_product'
  );
  assert.equal(out.length, 2);
  out.forEach((o) => assertVector(o));
  assertClose(out[0], Array.from(new native.DotProductAttention(DIM).compute(q, K, V)));
});

test('hyperbolic helpers return full-length vectors', () => {
  assertVector(w.projectToPoincareBall(q));
  assertVector(w.mobiusAddition(q, K[0]));
  assertVector(w.expMap(new Float32Array(DIM), q));
  assertVector(w.logMap(new Float32Array(DIM), q));
});

test('AdamOptimizer.step(gradients, params) matches native Adam', () => {
  const params = [0.5, 0.5, 0.5, 0.5];
  const grads = [1, 1, -1, -1];
  const out = new w.AdamOptimizer(0.1).step(grads, params);
  // Native oracle: lr 0.1, first step moves each param by lr against the gradient sign.
  const oracle = Array.from(new native.AdamOptimizer(4, 0.1).step(Float32Array.from(params), Float32Array.from(grads)));
  assertClose(out, oracle);
  assertClose(out, [0.4, 0.4, 0.6, 0.6], 1e-4);
  assert.deepEqual(params, [0.5, 0.5, 0.5, 0.5], 'caller params must not be mutated');
});

test('benchmarkAttention returns timings per attention type', async () => {
  const res = await w.benchmarkAttention(DIM, 8, 2);
  assert.ok(Object.keys(res).length > 0);
  for (const r of Object.values(res)) {
    assert.ok(Number.isFinite(r.avgMs) && Number.isFinite(r.minMs) && Number.isFinite(r.maxMs));
  }
});

for (const type of ['dot', 'multi-head', 'flash', 'hyperbolic', 'linear']) {
  test(`CLI attention compute -t ${type}`, (t) => {
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rv-attn-'));
    t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
    const keysFile = path.join(dir, 'keys.json');
    const outFile = path.join(dir, 'out.json');
    fs.writeFileSync(keysFile, JSON.stringify(K.map((k) => Array.from(k))));
    execFileSync(process.execPath, [CLI, 'attention', 'compute', '-t', type, '-q', JSON.stringify(Array.from(q)),
      '-k', keysFile, '-h', '4', '-o', outFile], { encoding: 'utf8', env: { ...process.env, NO_COLOR: '1' } });
    const { result } = JSON.parse(fs.readFileSync(outFile, 'utf8'));
    assertVector(result);
  });
}
