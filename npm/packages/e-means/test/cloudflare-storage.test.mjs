import assert from 'node:assert/strict';
import { test } from 'node:test';
import { persistCodebook, loadCodebook, exportCentroids, CentroidExportError } from '../integrations/cloudflare.mjs';

const model = { dimensions: 2, centroids: new Float32Array([1, 2, 3, 4]), assignments: new Uint32Array([0, 1]), mse: 1 };
function stored(body) { return { size: new TextEncoder().encode(body).length, body: new Response(body).body }; }

test('R2 binding helpers persist only centroid codebooks and restore independent typed output', async () => {
  let written;
  const bucket = { async put(key, body, options) { written = { key, body, options }; return {}; }, async get() { return stored(written.body); } };
  const receipt = await persistCodebook(bucket, 'private/codebook.json', model);
  assert.equal(receipt.clusters, 2);
  const record = JSON.parse(written.body);
  assert.deepEqual(Object.keys(record).sort(), ['centroids', 'clusters', 'dimensions', 'schema']);
  assert.equal(written.options.httpMetadata.contentType, 'application/json');
  const loaded = await loadCodebook(bucket, 'private/codebook.json');
  assert.deepEqual(loaded.centroids, model.centroids);
  loaded.centroids[0] = 99;
  assert.equal(model.centroids[0], 1);
  assert.equal(await loadCodebook({ get: async () => null }, 'missing'), null);
});

test('R2 writes snapshot caller-owned centroid data before the first asynchronous boundary', async () => {
  const input = { dimensions: 1, centroids: new Float32Array([3, 7]) };
  let payload;
  const promise = persistCodebook({ async put(key, body) { payload = body; await Promise.resolve(); return {}; } }, 'snapshot', input);
  input.centroids.fill(100);
  await promise;
  assert.deepEqual(JSON.parse(payload).centroids, [3, 7]);
});

test('centroid validation inspects copied numeric storage even with an overridden iterator', async () => {
  const centroids = new Float32Array([1, NaN]);
  centroids[Symbol.iterator] = function* () { yield 0; };
  const input = { dimensions: 1, centroids };
  let writes = 0;
  await assert.rejects(persistCodebook({ put: async () => { writes++; return {}; } }, 'bad', input), /finite/);
  await assert.rejects(exportCentroids({ upsert: async () => { writes++; return { mutationId: 'bad' }; } }, input, { prefix: 'bad' }), /finite/);
  assert.equal(writes, 0);
});

test('PQ envelopes and accessor schemas cannot be mistaken for full-dimensional codebooks', async () => {
  let writes = 0, reads = 0;
  const bucket = { put: async () => { writes++; return {}; } };
  const index = { upsert: async () => { writes++; return { mutationId: 'unexpected' }; } };
  const pq = { schema: 'ruvector-e-means-pq-v1', dimensions: 2, centroids: new Float32Array([1, 2, 3, 4]) };
  const accessor = { dimensions: 2, centroids: new Float32Array([1, 2]) };
  Object.defineProperty(accessor, 'schema', { get() { reads++; return 'ruvector.e-means-codebook/1'; } });
  for (const input of [pq, accessor]) {
    await assert.rejects(persistCodebook(bucket, 'book', input), /single row-major codebook/);
    await assert.rejects(exportCentroids(index, input, { prefix: 'book' }), /single row-major codebook/);
  }
  assert.equal(writes, 0);
  assert.equal(reads, 0);
});

test('corrupt, oversized, nonfinite and malformed stored codebooks reject', async () => {
  for (const record of [
    { schema: 'other', dimensions: 1, clusters: 1, centroids: [0] },
    { schema: 'ruvector.e-means-codebook/1', dimensions: 2, clusters: 1, centroids: [0] },
    { schema: 'ruvector.e-means-codebook/1', dimensions: 1, clusters: 2, centroids: [0] },
    { schema: 'ruvector.e-means-codebook/1', dimensions: 1, clusters: 1, centroids: ['1'] },
    { schema: 'ruvector.e-means-codebook/1', dimensions: 1, clusters: 1, centroids: [1e40] },
  ]) await assert.rejects(loadCodebook({ get: async () => stored(JSON.stringify(record)) }, 'bad'));
  await assert.rejects(loadCodebook({ get: async () => ({ size: 10_000_000, body: new Response('x').body }) }, 'large'));
  await assert.rejects(persistCodebook({ put: async () => null }, 'failed', model), /acknowledge/);
  await assert.rejects(persistCodebook({ put: async () => undefined }, 'failed', model), /acknowledge/);
  let writes = 0;
  await assert.rejects(persistCodebook({ put: async () => { writes++; } }, 'invalid', { dimensions: 1, centroids: new Float32Array([Infinity]) }));
  assert.equal(writes, 0);
});

test('Vectorize binding export batches static centroid records and propagates explicit namespace', async () => {
  const batches = [];
  const index = { async upsert(vectors) { batches.push(structuredClone(vectors)); return { mutationId: `m${batches.length}` }; } };
  const receipt = await exportCentroids(index, { dimensions: 1, centroids: Float32Array.from({ length: 2001 }, (_, i) => i) },
    { prefix: 'codebook', namespace: 'owned-namespace' });
  assert.deepEqual(batches.map(batch => batch.length), [1000, 1000, 1]);
  assert.deepEqual(receipt, { submitted: 2001, mutationIds: ['m1', 'm2', 'm3'] });
  assert.deepEqual(batches[2][0], { id: 'codebook:2000', values: [2000], metadata: { cluster: 2000 }, namespace: 'owned-namespace' });
  assert.equal(new Set(batches.flat().map(v => v.id)).size, 2001);
});

test('Vectorize validates every record before writes and reports acknowledged batches on failure', async () => {
  let calls = 0;
  const index = { async upsert() { if (++calls === 2) throw new Error('binding failure'); return { mutationId: 'first' }; } };
  for (const [input, options] of [
    [model, { prefix: '😀'.repeat(16) }], [model, { prefix: 'x', namespace: '😀'.repeat(17) }],
    [model, { prefix: 'x', batchSize: 1001 }],
    [{ dimensions: 1537, centroids: new Float32Array(1537) }, { prefix: 'x' }],
    [{ dimensions: 1, centroids: new Float32Array([NaN]) }, { prefix: 'x' }],
  ]) await assert.rejects(exportCentroids(index, input, options));
  assert.equal(calls, 0);
  await assert.rejects(exportCentroids(index, { dimensions: 1, centroids: new Float32Array(1001) }, { prefix: 'x' }), error => {
    assert.ok(error instanceof CentroidExportError);
    assert.equal(error.submitted, 1000);
    assert.deepEqual(error.mutationIds, ['first']);
    return true;
  });
  for (const acknowledgement of [undefined, {}, { mutationId: '' }, { mutationId: 'x'.repeat(257) }]) {
    await assert.rejects(exportCentroids({ upsert: async () => acknowledgement }, model, { prefix: 'x' }), error => {
      assert.ok(error instanceof CentroidExportError);
      assert.equal(error.submitted, 0);
      assert.deepEqual(error.mutationIds, []);
      return true;
    });
  }
});
