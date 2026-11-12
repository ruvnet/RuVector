import assert from 'node:assert/strict';
import { test } from 'node:test';
import { init, fit } from '../e-means.mjs';
import initWasm, { fitJson, fitTyped, FittedModel } from '../wasm/ruvector_e_means.js';

await init();
const runtime = await initWasm();
const samples = new Float32Array([0, 1, 0, 2, 10, 12, 10, 13, -9, -10, -9, -11]);

test('typed result transport matches JSON transport for both finalizers', () => {
  for (const finalization of ['lastEpochMean', 'ema']) {
    const options = { clusters: 3, epochs: 3, batchSize: 2, seed: 7, finalization };
    const json = JSON.parse(fitJson(samples, 2, JSON.stringify(options)));
    assert.deepEqual(fit(samples, 2, options), {
      ...json,
      centroids: Float32Array.from(json.centroids),
      assignments: Uint32Array.from(json.assignments),
    });
  }
});

test('consumed outputs are caller-owned and survive handle release and WASM memory growth', () => {
  const handle = fitTyped(samples, 2, JSON.stringify({ clusters: 3, epochs: 2 }));
  let centroids, assignments;
  try {
    centroids = handle.takeCentroids();
    assignments = handle.takeAssignments();
    assert.deepEqual(handle.takeCentroids(), new Float32Array());
    assert.deepEqual(handle.takeAssignments(), new Uint32Array());
    assert.notEqual(centroids.buffer, runtime.memory.buffer);
    assert.notEqual(assignments.buffer, runtime.memory.buffer);
  } finally {
    handle.free();
  }
  const expectedCentroids = centroids.slice(), expectedAssignments = assignments.slice();
  runtime.memory.grow(1);
  fit(samples, 2, { clusters: 2 });
  assert.deepEqual(centroids, expectedCentroids);
  assert.deepEqual(assignments, expectedAssignments);
});

test('the wrapper frees its model handle after success and output-copy failure', () => {
  const originalTake = FittedModel.prototype.takeAssignments;
  const originalFree = FittedModel.prototype.free;
  let frees = 0;
  FittedModel.prototype.free = function () { frees++; return originalFree.call(this); };
  try {
    assert.equal(fit(samples, 2, { clusters: 2 }).assignments.length, 6);
    assert.equal(frees, 1);
    FittedModel.prototype.takeAssignments = function () { throw new Error('injected output-copy failure'); };
    assert.throws(() => fit(samples, 2, { clusters: 2 }), /injected output-copy failure/);
    assert.equal(frees, 2);
  } finally {
    FittedModel.prototype.takeAssignments = originalTake;
    FittedModel.prototype.free = originalFree;
  }
  assert.equal(fit(samples, 2, { clusters: 2 }).assignments.length, 6);
});

test('repeated fits release WASM model allocations instead of accumulating them', () => {
  const data = Float32Array.from({ length: 4096 }, (_, i) => (i % 17) / 17);
  const config = { clusters: 1, epochs: 1, batchSize: 128 };
  for (let i = 0; i < 8; i++) fit(data, 1, config);
  const afterWarmup = runtime.memory.buffer.byteLength;
  for (let i = 0; i < 128; i++) {
    const model = fit(data, 1, config);
    assert.equal(model.assignments.length, data.length);
  }
  assert.equal(runtime.memory.buffer.byteLength, afterWarmup, 'WASM allocation high-water mark must plateau');
});
