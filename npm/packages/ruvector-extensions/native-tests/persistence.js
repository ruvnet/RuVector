/** Run with RUVECTOR_NATIVE_TEST_PREFIX pointing to an npm project containing ruvector. */
import test from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtemp, readFile, writeFile, rm } from 'node:fs/promises';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { DatabasePersistence } from '../src/persistence.js';

const prefix = process.env.RUVECTOR_NATIVE_TEST_PREFIX;
assert.ok(prefix, 'Set RUVECTOR_NATIVE_TEST_PREFIX to a project with the real native RuVector packages installed.');
test('native persistence restores arbitrary IDs, float vectors and metadata', async () => {
  const require = createRequire(join(prefix, 'package.json'));
  const { VectorDB, isNative } = require('ruvector');
  assert.equal(isNative(), true, 'a memory fallback cannot satisfy this test');
  const dir = await mkdtemp(join(tmpdir(), 'extension-native-'));
  try {
    const ids = ['ticket:alpha', 'non-numeric/beta'];
    const db = new VectorDB({ dimensions: 3, storagePath: join(dir, 'source.db') });
    await db.insertBatch([
      { id: ids[0], vector: [1, 0, 0], metadata: { department: 'support', nested: { count: 2 } } },
      { id: ids[1], vector: [0, 1, 0], metadata: { urgent: false } },
    ]);
    const persistence = new DatabasePersistence(db, { baseDir: dir, dimensions: 3, getVectorIds: () => ids });
    const saved = await persistence.save();
    const state = JSON.parse(await readFile(saved, 'utf8'));
    assert.deepEqual(state.vectors.map(v => v.id), ids);
    assert.ok(state.vectors.every(v => Array.isArray(v.vector)));
    const destination = new VectorDB({ dimensions: 3, storagePath: join(dir, 'destination.db') });
    await destination.insert({ id: 'stale', vector: [0, 0, 1] });
    let currentIds = ['stale'];
    const restored = new DatabasePersistence(destination, { baseDir: dir, dimensions: 3, getVectorIds: () => currentIds });
    await restored.load({ path: saved, verifyChecksum: true });
    currentIds = ids;
    assert.equal(await destination.len(), 2);
    assert.equal(await destination.get('stale'), null);
    assert.deepEqual((await destination.get(ids[0])).metadata, state.vectors[0].metadata);
    assert.deepEqual(Array.from((await destination.get(ids[1])).vector), [0, 1, 0]);
    const snapshot = await restored.createSnapshot('roundtrip');
    assert.equal(snapshot.vectorCount, 2);
    assert.equal(snapshot.dimension, 3);
    const reopened = new VectorDB({ dimensions: 3, storagePath: join(dir, 'destination.db') });
    assert.equal((await reopened.search({ vector: [1, 0, 0], k: 1 }))[0].id, ids[0]);
    const incomplete = new DatabasePersistence(db, { baseDir: dir, dimensions: 3, getVectorIds: () => [ids[0]] });
    await assert.rejects(incomplete.save(), /every current vector ID/);
    const wrongDimension = new VectorDB({ dimensions: 2, storagePath: join(dir, 'wrong-dimension.db') });
    await wrongDimension.insert({ id: 'keep', vector: [1, 0] });
    const incompatible = new DatabasePersistence(wrongDimension, { baseDir: dir, dimensions: 2, getVectorIds: () => ['keep'] });
    await assert.rejects(incompatible.load({ path: saved }), /dimension does not match/);
    assert.equal(await wrongDimension.len(), 1, 'an incompatible restore must not clear existing data');
    await assert.rejects(incompatible.import({ path: saved, clear: true }), /dimension does not match/);
    assert.deepEqual(Array.from((await wrongDimension.get('keep')).vector), [1, 0], 'invalid import must preserve the existing vector');
    const corrupt = join(dir, 'corrupt.json');
    await writeFile(corrupt, JSON.stringify({ ...state, checksum: 'invalid-checksum' }));
    await assert.rejects(restored.import({ path: corrupt, clear: true, verifyChecksum: true }), /Checksum verification failed/);
    assert.equal(await destination.len(), 2, 'checksum rejection must preserve existing data');
  } finally {
    await rm(dir, { recursive: true, force: true });
  }
});
