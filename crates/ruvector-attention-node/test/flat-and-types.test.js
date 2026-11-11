// computeFlat must match compute, reject malformed flat input, and every
// native export must be declared in the shipped index.d.ts.
//
// Run against the local build:      node --test test/*.test.js
// Run against a published package:  ATTENTION_BINDING=@ruvector/attention node --test test/*.test.js
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');
const fs = require('node:fs');

function bindingRoot() {
  if (process.env.ATTENTION_BINDING) {
    return path.dirname(
      require.resolve(`${process.env.ATTENTION_BINDING}/package.json`, { paths: [process.cwd()] })
    );
  }
  return path.join(__dirname, '..');
}

const root = bindingRoot();
const a = require(root);

const dim = 16;
let seed = 7;
const rnd = () => {
  seed = (seed * 1103515245 + 12345) >>> 0;
  return seed / 4294967296 - 0.5;
};
const vec = (n) => Float32Array.from({ length: n }, rnd);
const flat = (rows) => {
  const out = new Float32Array(rows.length * rows[0].length);
  rows.forEach((r, i) => out.set(r, i * r.length));
  return out;
};

const mechanisms = () => [
  ['DotProductAttention', new a.DotProductAttention(dim)],
  ['MultiHeadAttention', new a.MultiHeadAttention(dim, 4)],
  ['HyperbolicAttention', new a.HyperbolicAttention(dim, 1.0)],
  ['FlashAttention', new a.FlashAttention(dim, 4)],
  ['LinearAttention', new a.LinearAttention(dim, 8)],
  ['LocalGlobalAttention', new a.LocalGlobalAttention(dim, 2, 1)],
  ['MoEAttention', a.MoEAttention.simple(dim, 2, 1)],
];

test('computeFlat matches compute for every mechanism', () => {
  const q = vec(dim);
  const keys = Array.from({ length: 9 }, () => vec(dim));
  const values = Array.from({ length: 9 }, () => vec(dim));
  for (const [name, m] of mechanisms()) {
    const nested = m.compute(q, keys, values);
    const fromFlat = m.computeFlat(q, flat(keys), flat(values));
    assert.strictEqual(fromFlat.length, nested.length, name);
    for (let i = 0; i < nested.length; i++) {
      assert.ok(Math.abs(nested[i] - fromFlat[i]) < 1e-6, `${name}[${i}]`);
    }
  }
});

test('computeFlat rejects malformed flat input', () => {
  const q = vec(dim);
  for (const [name, m] of mechanisms()) {
    assert.throws(() => m.computeFlat(q, new Float32Array(0), new Float32Array(0)), name);
    assert.throws(() => m.computeFlat(q, vec(dim + 1), vec(dim)), name);
    assert.throws(() => m.computeFlat(q, vec(dim * 2), vec(dim * 2 + 1)), name);
    assert.throws(() => m.computeFlat(vec(dim - 1), vec(dim), vec(dim)), name);
  }
});

test('nested compute rejects non-Float32Array rows', () => {
  const flash = new a.FlashAttention(dim, 4);
  const q = vec(dim);
  assert.throws(() => flash.compute(q, [Array.from(vec(dim))], [vec(dim)]), /Float32Array/);
  assert.throws(() => flash.compute(q, [new Float64Array(dim)], [vec(dim)]), /Float32Array/);
  assert.throws(() => flash.compute(q, vec(dim), [vec(dim)]), /array/);
});

test('index.d.ts declares every native export', () => {
  const dts = fs.readFileSync(path.join(root, 'index.d.ts'), 'utf8');
  const declared = new Set(
    [...dts.matchAll(/export (?:declare )?(?:const enum|function|class|enum|interface|type|const) (\w+)/g)].map(
      (m) => m[1]
    )
  );
  const missing = Object.keys(a).filter((name) => !declared.has(name));
  assert.deepStrictEqual(missing, []);
  for (const cls of ['FlashAttention', 'MultiHeadAttention', 'DotProductAttention']) {
    assert.match(dts, new RegExp(`class ${cls}[\\s\\S]*?computeFlat\\(`));
  }
});
