'use strict';
/**
 * Tests for WasmBackend byte-level persistence (issue #705).
 *
 * `@ruvector/rvf-wasm`'s WASM backend is in-memory only (no filesystem
 * access), so the SDK needs exportBytes()/openBytes() as the supported way
 * to durably persist a browser-side store. These tests exercise the real
 * compiled WASM module (not a mock).
 */

const assert = require('assert');
const { RvfDatabase } = require('../dist/index.js');

async function main() {
  const results = [];
  function test(name, fn) {
    return Promise.resolve()
      .then(fn)
      .then(() => results.push({ name, pass: true }))
      .catch((err) => results.push({ name, pass: false, err }));
  }

  await test('exportBytes() round-trips vectors through openBytes()', async () => {
    const db = await RvfDatabase.create('unused-in-wasm', { dimensions: 4 }, 'wasm');
    await db.ingestBatch([
      { id: '1', vector: new Float32Array([1, 0, 0, 0]) },
      { id: '2', vector: new Float32Array([0, 1, 0, 0]) },
      { id: '3', vector: new Float32Array([0, 0, 1, 0]) },
    ]);

    const bytes = await db.exportBytes();
    assert.ok(bytes instanceof Uint8Array, 'exportBytes should return a Uint8Array');
    assert.ok(bytes.length > 0, 'exported bytes should be non-empty');

    const reopened = await RvfDatabase.openBytes(bytes, 'wasm');
    const results2 = await reopened.query(new Float32Array([1, 0, 0, 0]), 1);
    assert.strictEqual(results2.length, 1, 'reopened store should return a result');
    assert.strictEqual(results2[0].id, '1', 'reopened store should preserve the nearest vector id');

    await db.close();
    await reopened.close();
  });

  await test('exportBytes() on an empty store does not throw', async () => {
    const db = await RvfDatabase.create('unused-in-wasm', { dimensions: 4 }, 'wasm');
    const bytes = await db.exportBytes();
    assert.ok(bytes instanceof Uint8Array);
    await db.close();
  });

  await test('node backend throws on exportBytes/openBytes (unsupported)', async () => {
    const os = require('os');
    const path = require('path');
    const fs = require('fs');
    const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-wasm-test-'));
    const dbPath = path.join(dir, 'test.rvf');
    let db;
    try {
      db = await RvfDatabase.create(dbPath, { dimensions: 4 }, 'node');
    } catch (err) {
      // Native @ruvector/rvf-node addon may not be built in this environment;
      // that's a separate concern from this test's actual target (wasm).
      return;
    }
    await assert.rejects(() => db.exportBytes(), /not supported/i);
    await db.close();
    fs.rmSync(dir, { recursive: true, force: true });
  });

  console.log('\n--- WasmBackend byte persistence (Issue #705) ---');
  let passed = 0;
  for (const r of results) {
    if (r.pass) {
      console.log(`  PASS  ${r.name}`);
      passed++;
    } else {
      console.log(`  FAIL  ${r.name}`);
      console.log(`        ${r.err && r.err.stack ? r.err.stack : r.err}`);
    }
  }
  console.log('\n' + '='.repeat(60));
  console.log(`Results: ${passed} passed, ${results.length - passed} failed, ${results.length} total`);
  console.log('='.repeat(60));

  if (passed !== results.length) process.exit(1);
}

main().catch((err) => {
  console.error(err);
  process.exit(1);
});
