// node --test tests/async.test.mjs  (after building the addon)
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadAddon, vectors } from './load-addon.mjs';

const { RvfDatabase } = loadAddon();
const DIM = 32;
const N = 3000;

function freshDir() {
  return mkdtempSync(join(tmpdir(), 'rvf-node-async-'));
}

async function makeDb(dir) {
  const db = RvfDatabase.create(join(dir, 'a.rvf'), { dimension: DIM });
  const ids = Array.from({ length: N }, (_, i) => i);
  const metadata = ids.map((i) => ({ fieldId: 0, valueType: 'u64', value: String(i % 10) }));
  const res = await db.ingestBatchAsync(vectors(N, DIM, 7), ids, metadata);
  assert.equal(res.accepted, N);
  return db;
}

const filter = JSON.stringify({ op: 'lt', fieldId: 0, valueType: 'u64', value: '5' });

test('queryAsync resolves to exactly what query returns', async () => {
  const dir = freshDir();
  try {
    const db = await makeDb(dir);
    const qs = vectors(20, DIM, 99);
    for (let i = 0; i < 20; i++) {
      const q = qs.subarray(i * DIM, (i + 1) * DIM);
      assert.deepEqual(await db.queryAsync(q, 10), db.query(q, 10));
      assert.deepEqual(await db.queryAsync(q, 10, { filter }), db.query(q, 10, { filter }));
    }
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('concurrent queryAsync calls match sequential results', async () => {
  const dir = freshDir();
  try {
    const db = await makeDb(dir);
    const qs = vectors(64, DIM, 5);
    const query = (i) => qs.subarray(i * DIM, (i + 1) * DIM);
    const expected = Array.from({ length: 64 }, (_, i) => db.query(query(i), 10));
    const got = await Promise.all(Array.from({ length: 64 }, (_, i) => db.queryAsync(query(i), 10)));
    assert.deepEqual(got, expected);
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('deleteAsync and compactAsync mutate under the write lock', async () => {
  const dir = freshDir();
  try {
    const db = await makeDb(dir);
    const doomed = Array.from({ length: N / 2 }, (_, i) => i * 2);
    // Queries queued alongside the mutations must neither crash nor see a
    // half-applied state: each sees the store before or after the delete.
    const q = vectors(1, DIM, 3);
    const [, del] = await Promise.all([db.queryAsync(q, 5), db.deleteAsync(doomed)]);
    assert.equal(del.deleted, N / 2);
    const after = await db.queryAsync(q, 50);
    assert.ok(after.every((r) => r.id % 2 === 1));
    const compacted = await db.compactAsync();
    assert.ok(compacted.bytesReclaimed > 0);
    assert.equal(db.status().totalVectors, N / 2);
    db.close();
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

test('errors reject the promise instead of throwing', async () => {
  const dir = freshDir();
  try {
    const db = await makeDb(dir);
    const q = vectors(1, DIM, 1);
    await assert.rejects(db.queryAsync(q, 10, { filter: '{not json' }));
    await assert.rejects(db.queryAsync(new Float32Array(DIM + 1), 10), /RVF error/);
    await assert.rejects(db.ingestBatchAsync(new Float32Array(5), [1, 2]), /divisible/);
    db.close();
    await assert.rejects(db.queryAsync(q, 10), /Store is closed/);
    await assert.rejects(db.compactAsync(), /Store is closed/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
