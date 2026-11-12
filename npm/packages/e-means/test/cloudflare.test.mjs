import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import { test } from 'node:test';
import { createClusteringWorker, WORKER_LIMITS } from '../integrations/cloudflare.mjs';

const wasmModule = new WebAssembly.Module(await readFile(new URL('../wasm/ruvector_e_means_bg.wasm', import.meta.url)));
const token = 'e-means-test-token-not-for-production';
const valid = { data: [1, 2, 3, 4, 5, 6, 7, 8, 9, 10], dimensions: 2,
  options: { clusters: 1, epochs: 2, batchSize: 2, seed: 42 } };
const handler = createClusteringWorker({ wasmModule, bearerToken: token });
function request(path, body, authorization = `Bearer ${token}`, headers = {}) {
  return new Request(`https://worker.test${path}`, { method: 'POST',
    headers: { 'content-type': 'application/json', authorization, ...headers }, body: JSON.stringify(body) });
}

test('Worker fits and predicts with the real precompiled Rust WASM module', async () => {
  const response = await handler.fetch(request('/fit', valid));
  assert.equal(response.status, 200);
  const fitted = await response.json();
  assert.deepEqual(fitted.centroids, [5, 6]);
  assert.deepEqual(fitted.assignments, [0, 0, 0, 0, 0]);
  assert.equal(fitted.mse, 16);
  assert.equal(fitted.updates, 6);
  const predicted = await handler.fetch(request('/predict', { data: [2, 2, 8, 8], dimensions: 2, centroids: [1, 1, 9, 9] }));
  assert.deepEqual(await predicted.json(), { assignments: [0, 1] });
  const empty = await handler.fetch(request('/predict', { data: [], dimensions: 2, centroids: [1, 1] }));
  assert.deepEqual(await empty.json(), { assignments: [] });
  assert.equal(response.headers.get('cache-control'), 'no-store');
  assert.equal(response.headers.get('x-content-type-options'), 'nosniff');
});

test('authentication fails closed and request fields cannot bypass it', async () => {
  const unavailable = createClusteringWorker({ wasmModule });
  assert.equal((await unavailable.fetch(request('/fit', valid))).status, 503);
  assert.equal((await handler.fetch(request('/fit', valid, 'Bearer incorrect'))).status, 401);
  const health = await unavailable.fetch(new Request('https://worker.test/health'));
  assert.equal(health.status, 200);
  assert.equal((await health.json()).authenticatedOperations, false);
  assert.equal((await handler.fetch(request('/fit', { ...valid, allowUnauthenticated: true }))).status, 422);
  const local = createClusteringWorker({ wasmModule, allowUnauthenticated: true });
  assert.equal((await local.fetch(request('/fit', valid, ''))).status, 200);
});

test('shape, numeric, option, method and conservative work limits reject requests', async () => {
  for (const body of [
    { ...valid, data: [] }, { ...valid, data: [1, 2, 3] },
    { ...valid, data: ['1', 2] }, { ...valid, data: [1e40, 2] },
    { ...valid, dimensions: 0 }, { ...valid, dimensions: WORKER_LIMITS.maxDimensions + 1 },
    { ...valid, options: { clusters: 0 } }, { ...valid, options: { clusters: 6 } },
    { ...valid, options: { clusters: 1, epochs: 0 } },
    { ...valid, options: { clusters: 1, batchSize: 0 } },
    { ...valid, options: { clusters: 1, peakStep: 0 } },
    { ...valid, options: { clusters: 1, seed: -1 } },
    { ...valid, options: { clusters: 1, finalization: 'slow' } },
    { ...valid, options: { clusters: 1, unknown: 1 } },
  ]) assert.equal((await handler.fetch(request('/fit', body))).status, 422, JSON.stringify(body));
  const limited = createClusteringWorker({ wasmModule, bearerToken: token, maxWork: 1 });
  assert.equal((await limited.fetch(request('/fit', valid))).status, 413);
  assert.equal((await limited.fetch(request('/predict', { data: [1, 2], dimensions: 1, centroids: [0, 1] }))).status, 413);
  // Assignment-only accounting would allow this ~4M-unit request; donor
  // revival at every fifth batch makes the actual upper bound much larger.
  assert.equal((await handler.fetch(request('/fit', { data: Array(256).fill(0), dimensions: 1,
    options: { clusters: 256, epochs: 32, batchSize: 1 } }))).status, 413);
  assert.equal((await handler.fetch(new Request('https://worker.test/fit'))).status, 405);
  assert.equal((await handler.fetch(request('/fit?namespace=another-user', valid))).status, 400);
  assert.equal((await handler.fetch(request('/persist', valid))).status, 404);
  assert.equal((await handler.fetch(request('/fit', valid, `Bearer ${token}`, { 'content-type': 'text/plain' }))).status, 415);
});

test('streamed bodies have actual byte limits, strict UTF-8 parsing and read deadlines', async () => {
  const small = createClusteringWorker({ wasmModule, bearerToken: token, maxRequestBytes: 32 });
  let cancelled = false;
  const oversized = new ReadableStream({ start(controller) { controller.enqueue(new Uint8Array(33)); }, cancel() { cancelled = true; } });
  const response = await small.fetch(new Request('https://worker.test/fit', { method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: oversized, duplex: 'half' }));
  assert.equal(response.status, 413);
  assert.equal(cancelled, true);
  const malformed = await handler.fetch(new Request('https://worker.test/fit', { method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: new Uint8Array([0xff]) }));
  assert.equal(malformed.status, 400);
  const quick = createClusteringWorker({ wasmModule, bearerToken: token, bodyTimeoutMs: 20 });
  const stalled = new ReadableStream({ start() {}, cancel() {} });
  const timeout = await quick.fetch(new Request('https://worker.test/fit', { method: 'POST',
    headers: { 'content-type': 'application/json', authorization: `Bearer ${token}` }, body: stalled, duplex: 'half' }));
  assert.equal(timeout.status, 408);
  const contentLength = await handler.fetch(request('/fit', valid, `Bearer ${token}`, { 'content-length': '999999999' }));
  assert.equal(contentLength.status, 413);
});

test('Worker configuration is explicit and bounded', () => {
  for (const options of [
    {}, { wasmModule: new Uint8Array() }, { wasmModule, bearerToken: 'short' },
    { wasmModule, maxWork: Infinity }, { wasmModule, maxRequestBytes: WORKER_LIMITS.maxRequestBytes + 1 },
    { wasmModule, allowUnauthenticated: 'true' }, { wasmModule, bodyTimeoutMs: 0 },
  ]) assert.throws(() => createClusteringWorker(options));
});
