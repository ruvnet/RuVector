#!/usr/bin/env node

const assert = require('node:assert/strict');
const { rm, mkdtemp } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const test = require('node:test');

const { VectorDb, VectorIndex } = require('../dist/index.js');

async function removeIndexStores(paths) {
  for (const path of paths) {
    if (path) await rm(path, { force: true });
  }
}

test('VectorIndex.clear preserves the requested Euclidean ranking', async () => {
  const index = new VectorIndex({ dimension: 2, metric: 'euclidean', indexType: 'hnsw' });
  const paths = [index._storagePath];
  const rows = [
    { id: 'far-collinear', values: [10, 0] },
    { id: 'near-angled', values: [0.8, 0.6] },
  ];

  try {
    await index.insertBatch(rows);
    const before = await index.search([1, 0], { k: 2 });

    await index.clear();
    paths.push(index._storagePath);
    await index.insertBatch(rows);
    const after = await index.search([1, 0], { k: 2 });

    assert.deepEqual(before.map(result => result.id), ['near-angled', 'far-collinear']);
    assert.deepEqual(after.map(result => result.id), before.map(result => result.id));
  } finally {
    await removeIndexStores(paths);
  }
});

test('explicit ID inserts and batches remain single-value upserts', async () => {
  const index = new VectorIndex({ dimension: 3, metric: 'cosine', indexType: 'hnsw' });
  const paths = [index._storagePath];

  try {
    await index.insert({ id: 'same-id', values: [1, 0, 0] });
    await index.insert({ id: 'same-id', values: [0, 1, 0] });

    assert.deepEqual(await index.stats(), { vectorCount: 1, dimension: 3 });
    assert.deepEqual((await index.get('same-id')).values, [0, 1, 0]);

    const currentMatches = await index.search([0, 1, 0], { k: 10 });
    assert.deepEqual(currentMatches.map(result => result.id), ['same-id']);

    const staleMatches = await index.search([1, 0, 0], { k: 10 });
    assert.deepEqual(staleMatches.map(result => result.id), ['same-id']);
    assert.ok(staleMatches[0].score > 0.9, 'search must score the current vector, not a stale node');

    const ids = await index.insertBatch([
      { id: 'same-id', values: [1, 0, 0] },
      { id: 'same-id', values: [0, 0, 1] },
    ]);
    assert.deepEqual(ids, ['same-id', 'same-id']);
    assert.deepEqual(await index.stats(), { vectorCount: 1, dimension: 3 });
    assert.deepEqual((await index.get('same-id')).values, [0, 0, 1]);

    const batchMatches = await index.search([0, 0, 1], { k: 10 });
    assert.deepEqual(batchMatches.map(result => result.id), ['same-id']);

    await Promise.all([
      index.insert({ id: 'same-id', values: [1, 0, 0] }),
      index.insert({ id: 'same-id', values: [0, 1, 0] }),
      index.insert({ id: 'same-id', values: [0, 0, 1] }),
    ]);
    const concurrentCurrent = await index.get('same-id');
    const concurrentMatches = await index.search(concurrentCurrent.values, { k: 10 });
    assert.deepEqual(concurrentMatches.map(result => result.id), ['same-id']);
    assert.deepEqual(await index.stats(), { vectorCount: 1, dimension: 3 });
  } finally {
    await removeIndexStores(paths);
  }
});

test('storage identity is explicit and implicit schema collisions are diagnosed', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ruvector-storage-identity-'));
  const explicitPath = join(directory, 'sixteen.db');
  const originalDirectory = process.cwd();

  try {
    const explicit = new VectorDb({ dimensions: 16, storagePath: explicitPath });
    await explicit.insert({ id: 'sixteen', vector: new Float32Array(16).fill(1) });
    assert.equal(await explicit.len(), 1);

    process.chdir(directory);
    const first = new VectorDb({ dimensions: 3 });
    await first.insert({ id: 'preserve-me', vector: [1, 0, 0] });

    const conflicting = new VectorDb({ dimensions: 16 });
    await assert.rejects(
      conflicting.search({ vector: new Float32Array(3), k: 1 }),
      /Vector dimension mismatch: expected 16, got 3/i
    );
    await assert.rejects(
      conflicting.search({ vector: new Float32Array(16), k: 1 }),
      /schema collision at implicit persistent store "\.\/ruvector\.db"/i
    );
    await assert.rejects(
      conflicting.insert({ id: 'preserve-me', vector: new Float32Array(16).fill(1) }),
      error => {
        assert.match(error.message, /schema collision at implicit persistent store "\.\/ruvector\.db"/i);
        assert.match(error.message, /requested 16 dimensions/i);
        assert.match(error.message, /expects 3/i);
        assert.match(error.message, /explicit `storagePath`/i);
        return true;
      }
    );

    const preserved = await first.get('preserve-me');
    assert.deepEqual(Array.from(preserved.vector), [1, 0, 0]);
  } finally {
    process.chdir(originalDirectory);
    await rm(directory, { recursive: true, force: true });
  }
});
