import test from 'node:test';
import assert from 'node:assert/strict';
import { init, fit, predict } from '@ruvector/e-means';
import {
  trainProductQuantizer, encodeProductQuantizer, decodeProductQuantizer,
  productQuantizerToJSON, productQuantizerFromJSON,
} from '@ruvector/e-means/quantization';

await init();

const data = new Float32Array([1, 2, 10, 11, 1, 2, 20, 21, 3, 4, 10, 11, 3, 4, 20, 21]);
const options = { clusters: 2, epochs: 8, batchSize: 4, peakStep: 0.5, seed: 42 };

test('product quantization trains exact shared codebooks and reconstructs representable vectors', () => {
  const before = data.slice();
  const model = trainProductQuantizer(data, 4, 2, options);
  assert.equal(model.schema, 'ruvector-e-means-pq-v1');
  assert.equal(model.centroids.length, 8);
  const codes = encodeProductQuantizer(model, data);
  assert.ok(codes instanceof Uint8Array);
  assert.equal(codes.length, 8);
  assert.deepEqual(decodeProductQuantizer(model, codes), data);
  assert.deepEqual(data, before);
  assert.deepEqual(trainProductQuantizer(data, 4, 2, options).centroids, model.centroids);
  for (let subspace = 0; subspace < 2; subspace++) {
    const subvectors = Float32Array.from({ length: 8 }, (_, index) => data[Math.floor(index / 2) * 4 + subspace * 2 + index % 2]);
    const shared = fit(subvectors, 2, options);
    const start = subspace * 4;
    assert.deepEqual(model.centroids.slice(start, start + 4), shared.centroids);
    const assignments = predict(subvectors, 2, shared.centroids);
    for (let row = 0; row < 4; row++) assert.equal(codes[row * 2 + subspace], assignments[row]);
  }
});

test('one-cluster subspaces have arithmetic-mean codebooks and support empty codecs', () => {
  const model = trainProductQuantizer(data, 4, 2, { clusters: 1, batchSize: 3 });
  assert.deepEqual(Array.from(model.centroids), [2, 3, 15, 16]);
  assert.deepEqual(encodeProductQuantizer(model, new Float32Array()), new Uint8Array());
  assert.deepEqual(decodeProductQuantizer(model, new Uint8Array()), new Float32Array());
  assert.deepEqual(Array.from(decodeProductQuantizer(model, new Uint8Array([0, 0]))), [2, 3, 15, 16]);
});

test('all subspaces use one snapshot of caller configuration', () => {
  let reads = 0;
  const model = trainProductQuantizer(data, 4, 2, {
    get clusters() { reads++; return reads === 1 ? 1 : 2; },
    batchSize: 3,
  });
  assert.equal(reads, 1);
  assert.equal(model.clusters, 1);
  assert.deepEqual(Array.from(model.centroids), [2, 3, 15, 16]);
});

test('codebooks survive JSON and structured clone with independent ownership', () => {
  const model = trainProductQuantizer(data, 4, 2, options);
  const json = productQuantizerToJSON(model);
  const restored = productQuantizerFromJSON(JSON.parse(JSON.stringify(json)));
  assert.deepEqual(restored.centroids, model.centroids);
  assert.notEqual(restored.centroids, model.centroids);
  const clone = structuredClone(model);
  assert.deepEqual(encodeProductQuantizer(clone, data), encodeProductQuantizer(model, data));
  restored.centroids.fill(0);
  json.centroids.fill(0);
  assert.ok(model.centroids.some(value => value !== 0));
});

test('256-centroid codebooks preserve the full byte range', () => {
  const model = {
    schema: 'ruvector-e-means-pq-v1', dimensions: 1, subspaces: 1, clusters: 256,
    centroids: Float32Array.from({ length: 256 }, (_, index) => index),
  };
  const input = new Float32Array([0, 255, 254]);
  const codes = encodeProductQuantizer(model, input);
  assert.deepEqual(Array.from(codes), [0, 255, 254]);
  assert.deepEqual(decodeProductQuantizer(model, codes), input);
});

test('product quantization rejects invalid shapes, values, configuration and byte codes', () => {
  for (const [dimensions, subspaces, clusters] of [[0, 1, 1], [4, 0, 1], [4, 3, 1], [4, 5, 1], [4, 2, 0], [4, 2, 257], [4, 2, 5]]) {
    assert.throws(() => trainProductQuantizer(data, dimensions, subspaces, { clusters }));
  }
  assert.throws(() => trainProductQuantizer(data, 4, 2, { clusters: 2, epochs: 0 }));
  assert.throws(() => trainProductQuantizer(data, 4, 2, { ...options, unknown: true }));
  assert.throws(() => trainProductQuantizer([1, 2, 3, 4], 4, 2, { clusters: 1 }));
  assert.throws(() => trainProductQuantizer(new Float32Array([1, NaN, 3, 4]), 4, 2, { clusters: 1 }));
  const model = trainProductQuantizer(data, 4, 2, options);
  assert.throws(() => encodeProductQuantizer(model, new Float32Array([1])));
  assert.throws(() => encodeProductQuantizer(model, new Float32Array([1, 2, 3, Infinity])));
  assert.throws(() => decodeProductQuantizer(model, new Uint32Array([0, 0])));
  assert.throws(() => decodeProductQuantizer(model, new Uint8Array([0])));
  assert.throws(() => decodeProductQuantizer(model, new Uint8Array([0, 2])));
});

test('every operation revalidates caller-owned codebooks and JSON values', () => {
  const model = trainProductQuantizer(data, 4, 2, options);
  for (const mutation of [
    value => { value.schema = 'unknown'; },
    value => { value.subspaces = 3; },
    value => { value.clusters = 257; },
    value => { value.centroids = value.centroids.slice(1); },
    value => { value.centroids[0] = NaN; },
  ]) {
    const bad = structuredClone(model);
    mutation(bad);
    assert.throws(() => encodeProductQuantizer(bad, data));
    assert.throws(() => decodeProductQuantizer(bad, new Uint8Array([0, 0])));
    assert.throws(() => productQuantizerToJSON(bad));
  }
  for (const coordinate of ['1', NaN, Infinity, 1e100, undefined]) {
    const json = productQuantizerToJSON(model);
    json.centroids[0] = coordinate;
    assert.throws(() => productQuantizerFromJSON(json));
  }
});

test('codecs validate indexed typed-array storage despite overridden iterators', () => {
  const model = { schema: 'ruvector-e-means-pq-v1', dimensions: 1, subspaces: 1,
    clusters: 1, centroids: new Float32Array([3]) };
  const hiddenNonfinite = new Float32Array([NaN]);
  hiddenNonfinite[Symbol.iterator] = function* () { yield 0; };
  const corrupt = { ...model, centroids: hiddenNonfinite };
  assert.throws(() => decodeProductQuantizer(corrupt, new Uint8Array([0])), /centroids must be finite/);
  assert.throws(() => productQuantizerToJSON(corrupt), /centroids must be finite/);
  const hiddenInvalidCode = new Uint8Array([255]);
  hiddenInvalidCode[Symbol.iterator] = function* () { yield 0; };
  assert.throws(() => decodeProductQuantizer(model, hiddenInvalidCode), /code exceeds/);
});

test('codebook JSON serialization reads stored coordinates rather than a custom iterator', () => {
  const centroids = new Float32Array([3, 7]);
  centroids[Symbol.iterator] = function* () { yield 0; };
  const model = { schema: 'ruvector-e-means-pq-v1', dimensions: 2, subspaces: 1,
    clusters: 1, centroids };
  assert.deepEqual(productQuantizerToJSON(model).centroids, [3, 7]);
  assert.deepEqual(Array.from(decodeProductQuantizer(model, new Uint8Array([0]))), [3, 7]);
});
