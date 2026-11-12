import assert from 'node:assert/strict';
import { spawnSync } from 'node:child_process';
import { test } from 'node:test';
import { init, fit, predict } from '../e-means.mjs';

test('numerical operations require completed initialization', () => {
  const moduleUrl = new URL('../e-means.mjs', import.meta.url).href;
  const child = spawnSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { fit, predict } from ${JSON.stringify(moduleUrl)};
    assert.throws(() => fit(new Float32Array([0]), 1, { clusters: 1 }), /await init/);
    assert.throws(() => predict(new Float32Array([0]), 1, new Float32Array([0])), /await init/);
  `], { encoding: 'utf8' });
  assert.equal(child.status, 0, child.stderr);
});

test('initialization failure is retryable and concurrent calls share the runtime', async () => {
  const moduleUrl = new URL('../e-means.mjs', import.meta.url).href;
  const child = spawnSync(process.execPath, ['--input-type=module', '-e', `
    import assert from 'node:assert/strict';
    import { init, fit } from ${JSON.stringify(moduleUrl)};
    await assert.rejects(init(new Uint8Array([0, 1, 2])), /Could not initialize/);
    await Promise.all([init(), init()]);
    assert.equal(fit(new Float32Array([1, 3]), 1, { clusters: 1 }).centroids[0], 2);
  `], { encoding: 'utf8' });
  assert.equal(child.status, 0, child.stderr);
});

await init();

test('last-epoch mean uses every row, including a partial final batch', () => {
  const data = new Float32Array([1, 2, 3, 4]);
  const original = data.slice();
  const model = fit(data, 1, { clusters: 1, epochs: 1, batchSize: 3, peakStep: 1 });
  assert.deepEqual(data, original);
  assert.deepEqual(model.centroids, new Float32Array([2.5]));
  assert.deepEqual(model.assignments, new Uint32Array([0, 0, 0, 0]));
  assert.equal(model.mse, 1.25);
  assert.equal(model.updates, 2);
  assert.equal(model.batchSize, 3);
  assert.equal(model.dimensions, 1);
});

test('shared native/WASM golden case agrees on the exact codebook and diagnostics', () => {
  const model = fit(new Float32Array([1, 2, 3, 4, 5, 6, 7, 8, 9, 10]), 2,
    { clusters: 1, epochs: 2, batchSize: 2, seed: 42 });
  assert.deepEqual([...model.centroids], [5, 6]);
  assert.deepEqual([...model.assignments], [0, 0, 0, 0, 0]);
  assert.equal(model.mse, 16);
  assert.equal(model.updates, 6);
  assert.equal(model.batchSize, 2);
});

test('EMA maintains decayed sums and counts across unequal batch sizes', () => {
  const model = fit(new Float32Array([0, 1, 9]), 1,
    { clusters: 1, epochs: 1, batchSize: 2, peakStep: 1, finalization: 'ema', seed: 42 });
  // At t=1 of M=2, alpha=.55. Denominator is .45*2+.55*1=1.45.
  // Enumerate the three possible final rows so the oracle is independent of shuffle order.
  const possible = [3.103448275862069, 3.1724137931034484, 3.724137931034483];
  const center = model.centroids[0];
  assert.ok(possible.some((expected) => Math.abs(expected - center) < 1e-6), `center ${center}`);
  assert.ok(Math.abs(model.mse - ([0, 1, 9].reduce((sum, x) => sum + (x - center) ** 2, 0) / 3)) < 1e-10);
});

test('separated clusters reconstruct accurately and output labels match the final codebook', () => {
  const data = new Float32Array(256 * 2);
  for (let i = 0; i < 256; ++i) {
    const center = i < 128 ? -10 : 10;
    data[i * 2] = center + (i % 5) * 0.01;
    data[i * 2 + 1] = center - (i % 7) * 0.01;
  }
  const options = { clusters: 2, epochs: 4, batchSize: 32, seed: 7 };
  const model = fit(data, 2, options);
  assert.ok(model.mse < 0.01, `MSE ${model.mse}`);
  assert.deepEqual(model.assignments, predict(data, 2, model.centroids));
  assert.notEqual(model.assignments[0], model.assignments[128]);
  assert.deepEqual(fit(data, 2, options), model, 'same seed reproduces the same fit');
  assert.equal(model.centroids.length, 4);
  for (const v of model.centroids) assert.ok(Number.isFinite(v));
});

test('nearest centroid handles ties, empty queries and f32 extreme values', () => {
  const centroids = new Float32Array([-1, 1]);
  assert.deepEqual(predict(new Float32Array([-2, 0, 2]), 1, centroids), new Uint32Array([0, 0, 1]));
  assert.deepEqual(predict(new Float32Array(), 1, centroids), new Uint32Array());
  const large = new Float32Array([-3e38, 3e38]);
  const model = fit(large, 1, { clusters: 1, epochs: 1 });
  assert.equal(model.centroids[0], 0);
  assert.ok(Number.isFinite(model.mse));
});

test('duplicate and zero vectors never create nonfinite centroids', () => {
  for (const finalization of ['lastEpochMean', 'ema']) {
    const data = new Float32Array(96);
    const model = fit(data, 3, { clusters: 8, epochs: 3, batchSize: 2, peakStep: 1, finalization });
    assert.equal(model.centroids.length, 24);
    assert.equal(model.mse, 0);
    for (const v of model.centroids) assert.ok(Number.isFinite(v));
  }
});

test('invalid shapes, nonfinite input and malformed config reject before WASM coercion', () => {
  const data = new Float32Array([0, 1]);
  const config = { clusters: 1 };
  for (const bad of [0, -1, 1.5, NaN, Infinity, 2 ** 32, '1']) {
    assert.throws(() => fit(data, bad, config), /dimensions/);
  }
  for (const bad of [[], [0, 1], new Float64Array([0, 1]), null]) {
    assert.throws(() => fit(bad, 1, config), /Float32Array/);
  }
  assert.throws(() => fit(new Float32Array(), 1, config), /complete/);
  assert.throws(() => fit(new Float32Array([1, 2, 3]), 2, config), /complete/);
  for (const bad of [NaN, Infinity, -Infinity]) {
    assert.throws(() => fit(new Float32Array([0, bad]), 1, config), /finite/);
    assert.throws(() => predict(data, 1, new Float32Array([bad])), /finite/);
  }
  for (const key of ['clusters', 'epochs', 'batchSize']) {
    for (const bad of [0, -1, 0.5, NaN, Infinity, 2 ** 32, '1', null]) {
      assert.throws(() => fit(data, 1, { ...config, [key]: bad }), new RegExp(key));
    }
  }
  assert.throws(() => fit(data, 1, { clusters: 3 }), /number of vectors/);
  for (const bad of [0, -1, 2, NaN, Infinity, '1', null]) {
    assert.throws(() => fit(data, 1, { ...config, peakStep: bad }), /peakStep/);
  }
  for (const bad of [-1, 1.5, Number.MAX_SAFE_INTEGER + 1, NaN, Infinity, '1', null]) {
    assert.throws(() => fit(data, 1, { ...config, seed: bad }), /seed/);
  }
  for (const bad of [undefined, null, [], 1, 'config']) {
    assert.throws(() => fit(data, 1, bad), /options/);
  }
  assert.throws(() => fit(data, 1, { ...config, finalization: 'slowEma' }), /finalization/);
  assert.throws(() => fit(data, 1, { ...config, epohs: 3 }), /Unknown option/);
  assert.throws(() => predict(data, 1, new Float32Array()), /complete/);
});
