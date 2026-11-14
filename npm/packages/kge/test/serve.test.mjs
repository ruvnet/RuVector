// `kge serve` request guards, run against the compiled dist/ with the fake binding.
// Skips cleanly if the package has not been built yet.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { existsSync } from 'node:fs';
import { request } from 'node:http';
import { join, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';
import { createRequire } from 'node:module';

const testDir = dirname(fileURLToPath(import.meta.url));
const distIndex = join(testDir, '..', 'dist', 'index.js');
const built = existsSync(distIndex);
const require = createRequire(import.meta.url);
const fakeBinding = require('./fixtures/fake-binding.cjs');

function send(port, { method = 'POST', path = '/v1/predict', host, type, body }) {
  return new Promise((resolve, reject) => {
    const headers = { host: host ?? `127.0.0.1:${port}` };
    if (type) headers['content-type'] = type;
    const req = request({ host: '127.0.0.1', port, method, path, headers }, (res) => {
      res.resume();
      res.on('end', () => resolve(res.statusCode));
    });
    req.on('error', reject);
    req.end(body);
  });
}

async function withServer(fn) {
  const { createKge, serve } = await import(distIndex);
  const kge = createKge({ scorer: 'hole', dims: 8, binding: fakeBinding });
  let predicted = 0;
  const predict = kge.predict.bind(kge);
  kge.predict = (q) => { predicted++; return predict(q); };
  const handle = await serve(kge, { port: 0, log: () => {} });
  try {
    await fn(handle.port, () => predicted);
  } finally {
    await handle.close();
  }
}

const QUERY = JSON.stringify({ s: 'Ada', r: 'bornIn', k: 1 });

// A browser page can send a cross-origin POST without a CORS preflight only
// with a "simple" content type, so POST bodies must be declared JSON.
test('serve: POST without a JSON content type is rejected', { skip: !built && 'run npm run build first' }, async () => {
  await withServer(async (port, predicted) => {
    for (const type of [undefined, 'text/plain', 'application/x-www-form-urlencoded']) {
      assert.equal(await send(port, { type, body: QUERY }), 415, `content-type ${type}`);
    }
    assert.equal(predicted(), 0, 'the model was never queried');
    const ok = await send(port, { type: 'application/json; charset=utf-8', body: QUERY });
    assert.ok(ok !== 415 && ok !== 403, `JSON request accepted (got ${ok})`);
  });
});

// DNS rebinding: a loopback-bound server must only answer its own Host names.
test('serve: a foreign Host header is rejected on loopback', { skip: !built && 'run npm run build first' }, async () => {
  await withServer(async (port, predicted) => {
    assert.equal(await send(port, { host: `attacker.example:${port}`, type: 'application/json', body: QUERY }), 403);
    assert.equal(await send(port, { method: 'GET', path: '/healthz', host: 'attacker.example' }), 403);
    assert.equal(predicted(), 0, 'the model was never queried');
    for (const host of [`127.0.0.1:${port}`, `localhost:${port}`, 'localhost']) {
      assert.equal(await send(port, { method: 'GET', path: '/healthz', host }), 200, host);
    }
  });
});
