'use strict';

// Package tests for @ruvector/cnn, run against the built WASM artifact.
// Usage: npm test   (node test.js)

const assert = require('assert');
const cnn = require('./index.js');

const tests = [];
const test = (name, fn) => tests.push({ name, fn });
const close = (a, b, eps = 1e-5) => Math.abs(a - b) < eps;

test('CnnEmbedder constructs and reports embeddingDim', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 64 });
  assert.strictEqual(e.embeddingDim, 64);
});

test('extract returns an L2-normalised vector of embeddingDim', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 64 });
  const img = new Uint8Array(16 * 16 * 3);
  for (let i = 0; i < img.length; i++) img[i] = (i * 7) & 255;
  const v = e.extract(img, 16, 16);
  assert.strictEqual(v.length, 64);
  assert.ok(close(Math.hypot(...v), 1, 1e-4));
});

test('extract rejects wrong buffer length', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 16 });
  assert.throws(() => e.extract(new Uint8Array(10), 4, 4), /length/);
});

test('extract rejects zero dimensions instead of returning NaN', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 16 });
  assert.throws(() => e.extract(new Uint8Array(0), 0, 0), /dimension|zero/i);
});

test('extract rejects width*height*3 overflow (u32 wrap)', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 16 });
  // 3 * 2 * 715827883 = 2^32 + 2 -> wraps to 2 on u32
  assert.throws(() => e.extract(new Uint8Array(2), 2, 715827883), /overflow|length/i);
});

test('CnnEmbedder rejects absurd embeddingDim', () => {
  assert.throws(() => new cnn.CnnEmbedder({ embeddingDim: 0xffffffff }), /embedding/i);
});

test('SimdOps.dotProduct rejects mismatched lengths', () => {
  assert.throws(() => cnn.SimdOps.dotProduct(new Float32Array(8), new Float32Array(5)), /length|match/i);
  assert.ok(close(cnn.SimdOps.dotProduct(new Float32Array([1, 2, 3]), new Float32Array([1, 1, 1])), 6));
});

test('SimdOps.relu / relu6 / l2Normalize modify in place', () => {
  const a = new Float32Array([-1, 2, -3, 7, 0.5]);
  cnn.SimdOps.relu(a);
  assert.deepStrictEqual(Array.from(a), [0, 2, 0, 7, 0.5]);
  const b = new Float32Array([-1, 2, -3, 7, 0.5]);
  cnn.SimdOps.relu6(b);
  assert.deepStrictEqual(Array.from(b), [0, 2, 0, 6, 0.5]);
  const c = new Float32Array([3, 4]);
  cnn.SimdOps.l2Normalize(c);
  assert.ok(close(c[0], 0.6) && close(c[1], 0.8));
});

test('LayerOps.batchNorm modifies in place', () => {
  // NHWC, 2 channels, 2 spatial positions
  const x = new Float32Array([1, 2, 3, 4]);
  const one = new Float32Array([1, 1]);
  const zero = new Float32Array([0, 0]);
  cnn.LayerOps.batchNorm(x, new Float32Array([2, 2]), new Float32Array([1, 1]), zero, one, 0);
  assert.deepStrictEqual(Array.from(x), [3, 5, 7, 9]);
});

test('LayerOps.batchNorm rejects mismatched parameter lengths', () => {
  const x = new Float32Array(8);
  assert.throws(() => cnn.LayerOps.batchNorm(x, new Float32Array(2), new Float32Array(1),
    new Float32Array(1), new Float32Array(1)), /length|match/i);
  assert.throws(() => cnn.LayerOps.batchNorm(x, new Float32Array(0), new Float32Array(0),
    new Float32Array(0), new Float32Array(0)), /channel|length/i);
});

test('LayerOps.globalAvgPool averages each channel (NHWC)', () => {
  // 2 channels x 3 spatial positions: [c0,c1, c0,c1, c0,c1]
  const x = new Float32Array([1, 10, 2, 20, 3, 30]);
  const out = cnn.LayerOps.globalAvgPool(x, 2, 3);
  assert.strictEqual(out.length, 2);
  assert.ok(close(out[0], 2) && close(out[1], 20));
});

test('LayerOps.globalAvgPool rejects short input and zero spatial size', () => {
  assert.throws(() => cnn.LayerOps.globalAvgPool(new Float32Array(3), 4, 4), /length/i);
  assert.throws(() => cnn.LayerOps.globalAvgPool(new Float32Array(0), 4, 0), /zero|dimension/i);
});

test('InfoNCELoss rejects non-positive temperature instead of trapping', () => {
  assert.throws(() => new cnn.InfoNCELoss(0), /temperature/i);
  assert.throws(() => new cnn.InfoNCELoss(NaN), /temperature/i);
});

test('InfoNCELoss treats (i, i+N) as the positive pair', () => {
  // N = 2, D = 2. Row i and row i+N are identical, the two samples orthogonal.
  // Layout [a, b, a', b'] with a = a' = e0, b = b' = e1.
  const loss = new cnn.InfoNCELoss(0.1);
  const pairs = new Float32Array([1, 0, 0, 1, 1, 0, 0, 1]);
  const lossPaired = loss.forward(pairs, 2, 2);
  // Same rows, but now (i, i+N) are orthogonal: [a, a', b, b'].
  const wrong = new Float32Array([1, 0, 1, 0, 0, 1, 0, 1]);
  const lossWrong = loss.forward(wrong, 2, 2);
  assert.ok(lossPaired < lossWrong, `paired ${lossPaired} should be < unpaired ${lossWrong}`);
  assert.ok(lossPaired < 0.01, `matching pairs should give ~0 loss, got ${lossPaired}`);
});

test('InfoNCELoss rejects size overflow, zero sizes and NaN input', () => {
  const loss = new cnn.InfoNCELoss(0.1);
  // 2 * 0x40000000 * 2 = 2^32 wraps to 0 on wasm32
  assert.throws(() => loss.forward(new Float32Array(0), 0x40000000, 2), /overflow|size|length/i);
  assert.throws(() => loss.forward(new Float32Array(0), 4, 0), /zero|dimension|size/i);
  assert.throws(() => loss.forward(new Float32Array([NaN, 0, 0, 1, 1, 0, 0, 1]), 2, 2), /NaN/);
});

test('TripletLoss rejects dim 0 and invalid margin instead of trapping', () => {
  const t = new cnn.TripletLoss(1);
  const z = new Float32Array(4);
  assert.throws(() => t.forward(z, z, z, 0), /dim|zero/i);
  assert.throws(() => new cnn.TripletLoss(-1), /margin/i);
  assert.throws(() => new cnn.TripletLoss(NaN), /margin/i);
});

test('TripletLoss computes hinge on squared distance', () => {
  const t = new cnn.TripletLoss(1);
  const a = new Float32Array([0, 0]);
  const p = new Float32Array([0, 0]);
  const n = new Float32Array([0.5, 0]);
  // max(0, 0 - 0.25 + 1) = 0.75
  assert.ok(close(t.forward(a, p, n, 2), 0.75));
});

test('module survives a rejected call', () => {
  const e = new cnn.CnnEmbedder({ embeddingDim: 8 });
  try { e.extract(new Uint8Array(1), 1, 1); } catch (_) { /* expected */ }
  assert.strictEqual(e.extract(new Uint8Array(3), 1, 1).length, 8);
});

(async () => {
  await cnn.init();
  let failed = 0;
  for (const { name, fn } of tests) {
    try {
      fn();
      console.log(`ok   ${name}`);
    } catch (err) {
      failed++;
      console.log(`FAIL ${name}\n     ${String(err && err.message).split('\n')[0]}`);
    }
  }
  console.log(`\n${tests.length - failed}/${tests.length} passed`);
  process.exit(failed ? 1 : 0);
})();
