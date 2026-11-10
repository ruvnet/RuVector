/**
 * Runtime contract with the ruvector peer range (^0.3.14): the VectorDB
 * calls the exporters and examples rely on.
 */

import { test } from 'node:test';
import assert from 'node:assert';
import { VectorDB } from 'ruvector';

test('ruvector VectorDB insert/insertBatch/get/search round-trip', async () => {
  const db = new VectorDB({ dimensions: 3 });
  await db.insert({ id: 'a', vector: new Float32Array([1, 0, 0]), metadata: { tag: 'x' } });
  await db.insertBatch([
    { id: 'b', vector: new Float32Array([0, 1, 0]) },
    { id: 'c', vector: new Float32Array([0, 0, 1]) },
  ]);
  assert.strictEqual(await db.len(), 3);
  const got = await db.get('a');
  assert.ok(got, 'get returns the entry');
  assert.strictEqual(got.id, 'a');
  const hits = await db.search({ vector: new Float32Array([1, 0.1, 0]), k: 2 });
  assert.strictEqual(hits.length, 2);
  assert.strictEqual(hits[0].id, 'a');
  assert.ok(typeof hits[0].score === 'number');
});
