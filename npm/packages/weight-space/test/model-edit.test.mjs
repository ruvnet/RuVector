import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LIMITS, WeightSpaceError, createMlp, diff, getLayer, modelHash, perturb, setLayer, topologyOf, validateModel,
} from '../dist/esm/index.js';

const model = () => createMlp([3, 4, 2], { seed: 11 });

test('LIMITS mirror ruv-fann 0.1.0 exactly', async () => {
  const fann = await import('ruv-fann');
  assert.deepEqual({ ...LIMITS }, { ...fann.LIMITS });
});

test('validateModel rejects bad schema, shape, adjacency, non-finite and oversize models', () => {
  const m = model();
  const bad = (mut, code) => {
    const copy = JSON.parse(JSON.stringify(m));
    mut(copy);
    assert.throws(() => validateModel(copy), (e) => e instanceof WeightSpaceError && (!code || e.code === code));
  };
  bad((c) => { c.schema = 'other'; });
  bad((c) => { c.layers[0].weights.pop(); }, 'ESHAPE');
  bad((c) => { c.layers[1].inputs = 5; c.layers[1].weights = new Array(10).fill(0); }, 'ESHAPE');
  bad((c) => { c.layers[0].weights[0] = Infinity; });
  bad((c) => { c.layers[0].activation = 'Softmax'; });
  bad((c) => { c.layers = []; }, 'ERANGE');
  assert.throws(() => validateModel({ schema: 'ruv-fann.dense-layers/1', layers: [{ inputs: 1001, outputs: 1000, weights: [], bias: null, activation: 'Linear', steepness: 1 }] }), /parameters/);
});

test('modelHash is content-addressed and key-order independent', () => {
  const m = model();
  const reordered = { layers: m.layers.map((l) => ({ steepness: l.steepness, activation: l.activation, bias: l.bias, weights: l.weights, outputs: l.outputs, inputs: l.inputs })), schema: m.schema };
  assert.equal(modelHash(m), modelHash(validateModel(reordered)));
  assert.notEqual(modelHash(m), modelHash(perturb(m, { scale: 1e-6 })));
});

test('getLayer returns a detached copy', () => {
  const m = model();
  const view = getLayer(m, 0);
  assert.equal(view.weights.length, 12);
  view.weights[0] = 999;
  assert.notEqual(m.layers[0].weights[0], 999);
  assert.throws(() => getLayer(m, 2), /layer index/);
});

test('setLayer returns a new model, validates shape, and leaves the input untouched', () => {
  const m = model();
  const before = JSON.stringify(m);
  const next = setLayer(m, 1, { weights: new Array(8).fill(0.5), activation: 'Sigmoid' });
  assert.equal(JSON.stringify(m), before);
  assert.equal(next.layers[1].weights[3], 0.5);
  assert.equal(next.layers[1].activation, 'Sigmoid');
  assert.deepEqual(next.layers[0], m.layers[0]);
  assert.throws(() => setLayer(m, 1, { weights: [1, 2, 3] }), (e) => e.code === 'ESHAPE');
  assert.throws(() => setLayer(m, 0, { bias: [1] }), (e) => e.code === 'ESHAPE');
  assert.throws(() => setLayer(m, 0, { inputs: 9 }), /unknown patch field/);
});

test('perturb is seeded, scoped to chosen layers, and never mutates', () => {
  const m = model();
  const a = perturb(m, { scale: 0.1, seed: 5, layers: [1] });
  const b = perturb(m, { scale: 0.1, seed: 5, layers: [1] });
  assert.equal(modelHash(a), modelHash(b));
  assert.deepEqual(a.layers[0], m.layers[0]);
  assert.notDeepEqual(a.layers[1].weights, m.layers[1].weights);
  assert.throws(() => perturb(m, { scale: -1 }), /scale/);
});

test('diff reports per-layer deltas and incompatibility', () => {
  const m = model();
  const zero = diff(m, m);
  assert.equal(zero.compatible, true);
  assert.equal(zero.l2, 0);
  assert.equal(zero.changed, 0);
  const edited = setLayer(m, 0, { weights: m.layers[0].weights.map((w, i) => (i === 0 ? w + 3 : w)) });
  const d = diff(m, edited);
  assert.equal(d.changed, 1);
  assert.ok(Math.abs(d.l2 - 3) < 1e-12);
  assert.equal(d.layers[0].maxAbs, 3);
  const other = diff(m, createMlp([3, 5, 2]));
  assert.equal(other.compatible, false);
  assert.deepEqual(other.b.sizes, [3, 5, 2]);
  assert.deepEqual(topologyOf(m).sizes, [3, 4, 2]);
});
