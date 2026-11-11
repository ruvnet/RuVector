import test from 'node:test';
import assert from 'node:assert/strict';
import { init, fit, predict } from '@ruvector/e-means';
import { partition, postingList } from '@ruvector/e-means/ivf';

await init();

const data = new Float32Array([-10, -9, 10, 9, -9, -10, 9, 10, -11, -9, 11, 9]);
const options = { clusters: 2, epochs: 6, batchSize: 3, seed: 42 };

test('IVF partitions use shared fit results and stable exact assignment posting lists', () => {
  const before = data.slice();
  const result = partition(data, 2, options);
  const shared = fit(data, 2, options);
  assert.equal(result.schema, 'ruvector-e-means-ivf-v1');
  assert.deepEqual(result.model.centroids, shared.centroids);
  assert.deepEqual(result.model.assignments, predict(data, 2, shared.centroids));
  assert.equal(result.offsets.length, 3);
  assert.equal(result.offsets[0], 0);
  assert.equal(result.offsets[2], 6);
  assert.deepEqual(Array.from(result.indices).sort((a, b) => a - b), [0, 1, 2, 3, 4, 5]);
  for (let cluster = 0; cluster < 2; cluster++) {
    const expected = Array.from({ length: 6 }, (_, row) => row).filter(row => shared.assignments[row] === cluster);
    assert.deepEqual(Array.from(postingList(result, cluster)), expected);
  }
  assert.deepEqual(data, before);
});

test('IVF handles empty clusters and returns independent posting-list storage', () => {
  const result = partition(new Float32Array(16).fill(1), 2, { clusters: 4, epochs: 1, batchSize: 8 });
  assert.deepEqual(Array.from(postingList(result, 0)), [0, 1, 2, 3, 4, 5, 6, 7]);
  for (let cluster = 1; cluster < 4; cluster++) assert.deepEqual(postingList(result, cluster), new Uint32Array());
  const list = postingList(result, 0);
  list.fill(99);
  assert.equal(result.indices[0], 0);
  assert.deepEqual(postingList(structuredClone(result), 0), postingList(result, 0));
});

test('IVF rejects malformed training data and mutated partition records', () => {
  assert.throws(() => partition(new Float32Array([1, NaN]), 2, { clusters: 1 }));
  assert.throws(() => partition(new Float32Array([1]), 2, { clusters: 1 }));
  assert.throws(() => partition(data, 2, { clusters: 0 }));
  const result = partition(data, 2, options);
  for (const cluster of [-1, 2, 0.5, NaN]) assert.throws(() => postingList(result, cluster));
  for (const mutation of [
    value => { value.schema = 'unknown'; },
    value => { value.offsets = value.offsets.slice(1); },
    value => { value.offsets[0] = 1; },
    value => { value.offsets[1] = 99; },
    value => { value.indices[0] = 99; },
    value => { value.indices[1] = value.indices[0]; },
    value => { value.model.assignments[value.indices[0]] = 99; },
    value => { value.model.centroids[0] = NaN; },
    value => { value.model.mse = NaN; },
  ]) {
    const bad = structuredClone(result);
    mutation(bad);
    assert.throws(() => postingList(bad, 0));
  }
});

test('posting validation reads actual centroid storage despite an overridden iterator', () => {
  const result = partition(data, 2, options);
  result.model.centroids[0] = NaN;
  result.model.centroids[Symbol.iterator] = function* () { yield 0; };
  assert.throws(() => postingList(result, 0), /centroids must be finite/);
});
