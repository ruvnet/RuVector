'use strict';
/**
 * Issue #704 end to end against the real native backend: metadata written at
 * ingest is returned on hits, survives close/reopen, survives derive, and is
 * filterable by field name. Skips when @ruvector/rvf-node is not loadable.
 *
 *   node tests/test-metadata-durable.js
 *   RVF_SDK=@ruvector/rvf node test-metadata-durable.js   # against an install
 */
const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');

const { RvfDatabase, RvfErrorCode } = require(process.env.RVF_SDK || '../dist/index');

let passed = 0, failed = 0;
async function test(name, fn) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-meta-durable-'));
  try {
    await fn(dir);
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err && err.stack || err}`);
    failed++;
  } finally {
    fs.rmSync(dir, { recursive: true, force: true });
  }
}

const ids = (hits) => hits.map((h) => h.id).sort();

(async () => {
  try {
    require.resolve('@ruvector/rvf-node');
  } catch {
    console.log('SKIP: @ruvector/rvf-node is not installed');
    return;
  }
  console.log('--- Durable per-vector metadata (Issue #704, native) ---');

  await test('issue repro: metadata on hits, after reopen, and in filters', async (dir) => {
    const file = path.join(dir, 'm.rvf');
    let db = await RvfDatabase.create(file, { dimensions: 2, metric: 'cosine' }, 'node');
    const res = await db.ingestBatch([
      { id: 'red-item', vector: new Float32Array([1, 0]), metadata: { color: 'red', storedAt: 1700000000000, tags: ['a', 'b'] } },
      { id: 'blue-item', vector: new Float32Array([0, 1]), metadata: { color: 'blue', storedAt: 1700000000500.5 } },
      { id: 'plain', vector: new Float32Array([0.7, 0.7]) },
    ]);
    assert.strictEqual(res.accepted, 3);
    await db.close();

    db = await RvfDatabase.open(file, 'node');
    const hits = await db.query(new Float32Array([1, 0]), 3);
    const byId = Object.fromEntries(hits.map((h) => [h.id, h]));
    assert.deepStrictEqual({ ...byId['red-item'].metadata }, { color: 'red', storedAt: 1700000000000, tags: ['a', 'b'] });
    assert.deepStrictEqual({ ...byId['blue-item'].metadata }, { color: 'blue', storedAt: 1700000000500.5 });
    assert.strictEqual(byId.plain.metadata, undefined);

    const red = await db.query(new Float32Array([0, 1]), 3, { filter: { op: 'eq', field: 'color', value: 'red' } });
    assert.deepStrictEqual(ids(red), ['red-item']);
    // Integer literal against an f64 field must still match.
    const recent = await db.query(new Float32Array([0, 1]), 3, {
      filter: { op: 'ge', field: 'storedAt', value: 1700000000000 },
    });
    assert.deepStrictEqual(ids(recent), ['blue-item', 'red-item']);
    const none = await db.query(new Float32Array([0, 1]), 3, { filter: { op: 'eq', field: 'missing', value: 'x' } });
    assert.deepStrictEqual(none, []);
    await db.close();
  });

  await test('later batches add fields; a conflicting type is rejected, not dropped', async (dir) => {
    const file = path.join(dir, 'c.rvf');
    let db = await RvfDatabase.create(file, { dimensions: 2 }, 'node');
    await db.ingestBatch([{ id: 'a', vector: [1, 0], metadata: { n: 1 } }]);
    await db.close();
    db = await RvfDatabase.open(file, 'node');
    await db.ingestBatch([{ id: 'b', vector: [0, 1], metadata: { n: 2, label: 'x' } }]);
    await assert.rejects(
      () => db.ingestBatch([{ id: 'c', vector: [1, 1], metadata: { n: 'three' } }]),
      (err) => err.code === RvfErrorCode.MetadataTypeConflict,
    );
    const hits = await db.query(new Float32Array([0, 1]), 5, { filter: { op: 'eq', field: 'label', value: 'x' } });
    assert.deepStrictEqual(ids(hits), ['b']);
    await db.close();
  });

  await test('derived stores keep metadata and the field registry', async (dir) => {
    const db = await RvfDatabase.create(path.join(dir, 'p.rvf'), { dimensions: 2 }, 'node');
    await db.ingestBatch([{ id: 'a', vector: [1, 0], metadata: { kind: 'parent' } }]);
    const child = await db.derive(path.join(dir, 'c.rvf'));
    await child.ingestBatch([{ id: 'b', vector: [0, 1], metadata: { kind: 'child' } }]);
    const hits = await child.query(new Float32Array([0, 1]), 5, { filter: { op: 'eq', field: 'kind', value: 'child' } });
    assert.deepStrictEqual(ids(hits), ['b']);
    await child.close();
    await db.close();
  });

  await test('stores without metadata still open and query', async (dir) => {
    const file = path.join(dir, 'old.rvf');
    let db = await RvfDatabase.create(file, { dimensions: 2 }, 'node');
    await db.ingestBatch([{ id: 'x', vector: [1, 0] }]);
    await db.close();
    db = await RvfDatabase.open(file, 'node');
    const hits = await db.query(new Float32Array([1, 0]), 1);
    assert.strictEqual(hits[0].id, 'x');
    assert.strictEqual(hits[0].metadata, undefined);
    await db.close();
  });

  console.log(`\nResults: ${passed} passed, ${failed} failed, ${passed + failed} total`);
  process.exit(failed > 0 ? 1 : 0);
})();
