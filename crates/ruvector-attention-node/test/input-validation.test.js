// Invalid configuration and mismatched input lengths must surface as JavaScript
// errors, never as a process abort.
//
// Run against the local build:      node --test test/
// Run against a published package:  ATTENTION_BINDING=@ruvector/attention node --test test/
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

function loadBinding() {
  if (process.env.ATTENTION_BINDING) {
    return require(require.resolve(process.env.ATTENTION_BINDING, { paths: [process.cwd()] }));
  }
  const root = path.join(__dirname, '..');
  if (fs.existsSync(path.join(root, 'index.js'))) {
    return require(root);
  }
  const node = fs.readdirSync(root).find((f) => f.endsWith('.node'));
  if (!node) throw new Error('no local build found; run `napi build --platform --release` first');
  return require(path.join(root, node));
}

const a = loadBinding();
// napi-rs exports this class as `GraphRoPeAttention`.
const RoPE = a.GraphRoPeAttention || a.GraphRoPEAttention;
const vec = (n, v = 0.5) => new Float32Array(n).fill(v);

test('MultiHeadAttention rejects short keys and values', () => {
  const mh = new a.MultiHeadAttention(8, 2);
  assert.throws(() => mh.compute(vec(8), [vec(2)], [vec(8)]));
  assert.throws(() => mh.compute(vec(8), [vec(8)], [vec(2)]));
  assert.throws(() => mh.compute(vec(8), [vec(8), vec(8)], [vec(8)]));
  assert.strictEqual(mh.compute(vec(8), [vec(8)], [vec(8)]).length, 8);
});

test('MultiHeadAttention rejects zero or non-dividing heads', () => {
  assert.throws(() => new a.MultiHeadAttention(8, 0));
  assert.throws(() => new a.MultiHeadAttention(10, 3));
  assert.throws(() => new a.MultiHeadAttention(0, 1));
});

test('FlashAttention rejects zero block size', () => {
  assert.throws(() => new a.FlashAttention(4, 0));
  const flash = new a.FlashAttention(4, 2);
  assert.throws(() => flash.compute(vec(4), [vec(4), vec(4)], [vec(2), vec(8)]));
  assert.strictEqual(flash.compute(vec(4), [vec(4)], [vec(4)]).length, 4);
});

test('EdgeFeaturedAttention rejects zero heads and bad inputs', () => {
  assert.throws(() => a.EdgeFeaturedAttention.simple(8, 4, 0));
  assert.throws(() => new a.EdgeFeaturedAttention({ nodeDim: 8, edgeDim: 4, numHeads: 0 }));
  const ef = a.EdgeFeaturedAttention.simple(8, 4, 2);
  assert.throws(() => ef.computeWithEdges(vec(8), [vec(8), vec(8)], [vec(8)], [vec(4), vec(4)]));
  assert.throws(() => ef.computeWithEdges(vec(8), [vec(64)], [vec(8)], [vec(4)]));
  assert.throws(() => ef.computeWithEdges(vec(8), [vec(8)], [vec(8)], [vec(1)]));
  assert.throws(() => ef.compute(vec(8), [vec(3)], [vec(8)]));
  assert.ok(ef.computeWithEdges(vec(8), [vec(8)], [vec(8)], [vec(4)]).length > 0);
});

test('GraphRoPEAttention rejects bad config and short vectors', () => {
  assert.throws(() => RoPE.simple(8, 0));
  assert.throws(() => RoPE.simple(7, 16));
  assert.throws(() => new RoPE({ dim: 8, maxPosition: 0 }));
  const rope = RoPE.simple(8, 16);
  assert.throws(() => rope.applyRotary(vec(3), 1));
  assert.throws(() => rope.compute(vec(8), [vec(3)], [vec(8)]));
  assert.throws(() => rope.computeWithPositions(vec(8), [vec(8)], [vec(8)], 0, [1, 2]));
  assert.strictEqual(rope.applyRotary(vec(8), 1).length, 8);
  assert.strictEqual(RoPE.distanceToPosition(100, 4), 8);
});

test('other mechanisms reject invalid config and ragged inputs', () => {
  assert.throws(() => new a.DotProductAttention(0));
  assert.throws(() => new a.LinearAttention(8, 0));
  assert.throws(() => new a.LocalGlobalAttention(8, 0, 1));
  assert.throws(() => a.MoEAttention.simple(8, 0, 1));
  assert.throws(() => a.MoEAttention.simple(8, 2, 3));
  assert.throws(() => RoPE.simple(1 << 16, 1 << 20));
  const dp = new a.DotProductAttention(4);
  assert.throws(() => dp.compute(vec(4), [vec(2)], [vec(4)]));
  assert.throws(() => dp.computeWithMask(vec(4), [vec(4)], [vec(4)], [true, true]));
  const ds = a.DualSpaceAttention.simple(4, 1.0);
  assert.throws(() => ds.compute(vec(4), [vec(9)], [vec(4)]));
  assert.throws(() => ds.getSpaceContributions(vec(4), [vec(9)]));
  assert.throws(() => a.poincareDistance(vec(4, 0.1), vec(2, 0.1), 1.0));
  const sp = new a.StreamProcessor(4, 8);
  assert.throws(() => sp.push(vec(2)));
});

test('a panic inside a sync binding becomes a JS error', () => {
  // AdamOptimizer sizes its state from paramCount; a mismatched step used to
  // index out of bounds and abort the process.
  const opt = new a.AdamOptimizer(2, 0.01);
  assert.throws(() => opt.step(vec(8), vec(2)));
});

test('the process survives every trigger above', () => {
  // Reaching this test at all means none of the calls above aborted Node.
  const mh = new a.MultiHeadAttention(4, 2);
  assert.strictEqual(mh.compute(vec(4), [vec(4)], [vec(4)]).length, 4);
});
