#!/usr/bin/env node

const assert = require('node:assert/strict');
const { access, rm, mkdtemp } = require('node:fs/promises');
const { tmpdir } = require('node:os');
const { join } = require('node:path');
const test = require('node:test');

// Published @ruvector/core 0.1.32 predates the authoritative getOptions()
// binding. Keep local regression runs meaningful with a narrow constructor
// proxy; CI and the coordinated core release exercise the real native getter.
const coreModulePath = require.resolve('@ruvector/core');
const loadedCore = require(coreModulePath);
if (typeof loadedCore.VectorDb.prototype.getOptions !== 'function') {
  const RealNativeVectorDb = loadedCore.VectorDb;
  const effectiveByStorage = new Map();
  const canonicalMetric = metric => {
    const value = String(metric ?? 'Cosine').toLowerCase().replace(/[_\s-]/g, '');
    if (value === 'l2') return 'euclidean';
    if (value === 'dot' || value === 'innerproduct') return 'dotproduct';
    if (value === 'l1') return 'manhattan';
    return value;
  };

  class NativeVectorDbWithOptions {
    constructor(options) {
      this.inner = new RealNativeVectorDb(options);
      const storagePath = options.storagePath ?? './ruvector.db';
      const requested = {
        dimensions: options.dimensions,
        distanceMetric: canonicalMetric(options.distanceMetric),
        storagePath,
        hnswConfig: options.hnswConfig === undefined
          ? undefined
          : {
              m: options.hnswConfig.m ?? 32,
              efConstruction: options.hnswConfig.efConstruction ?? 200,
              efSearch: options.hnswConfig.efSearch ?? 100,
              maxElements: options.hnswConfig.maxElements ?? 10_000_000,
            },
        indexType: options.hnswConfig === undefined ? 'flat' : 'hnsw',
      };
      this.effective = effectiveByStorage.get(storagePath) ?? requested;
      effectiveByStorage.set(storagePath, this.effective);
    }

    getOptions() { return { ...this.effective }; }
    insert(entry) { return this.inner.insert(entry); }
    insertBatch(entries) { return this.inner.insertBatch(entries); }
    search(query) { return this.inner.search(query); }
    delete(id) { return this.inner.delete(id); }
    get(id) { return this.inner.get(id); }
    len() { return this.inner.len(); }
    isEmpty() { return this.inner.isEmpty(); }
  }

  require.cache[coreModulePath].exports = {
    ...loadedCore,
    VectorDb: NativeVectorDbWithOptions,
    VectorDB: NativeVectorDbWithOptions,
  };
}

const { NativeVectorDb, VectorDb, VectorIndex } = require('../dist/index.js');

async function removeIndexStores(paths) {
  for (const path of paths) {
    if (path) await rm(path, { force: true });
  }
}

async function pathExists(path) {
  try {
    await access(path);
    return true;
  } catch {
    return false;
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

test('native HNSW containment survives 100 replacements and delete cycles', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ruvector-hnsw-rebuild-'));
  try {
    const db = new VectorDb({
      dimensions: 2,
      storagePath: join(directory, 'vectors.db'),
      indexType: 'hnsw',
    });
    assert.deepEqual(db.getIndexInfo(), {
      indexType: 'hnsw',
      dimensions: 2,
      distanceMetric: 'cosine',
      storagePath: join(directory, 'vectors.db'),
      mutationMode: 'rebuild',
      configurationVerified: true,
    });

    await db.insert({ id: 'mutable', vector: [1, 0], metadata: { revision: -1 } });
    for (let revision = 0; revision < 100; revision++) {
      const vector = revision % 2 === 0 ? [1, 0] : [0, 1];
      await db.insert({ id: 'mutable', vector, metadata: { revision } });
      const matches = await db.search({ vector, k: 10 });
      assert.deepEqual(matches.map(result => result.id), ['mutable'], `replacement ${revision}`);
      assert.equal((await db.get('mutable')).metadata.revision, revision);
      assert.equal(await db.len(), 1);
    }

    for (let revision = 100; revision < 200; revision++) {
      const vector = revision % 2 === 0 ? [1, 0] : [0, 1];
      assert.equal(await db.delete('mutable'), true);
      await db.insert({ id: 'mutable', vector, metadata: { revision } });
      const matches = await db.search({ vector, k: 10 });
      assert.deepEqual(matches.map(result => result.id), ['mutable'], `delete cycle ${revision}`);
      assert.equal(await db.len(), 1);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('flat selector supports 100 in-place mutable updates', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ruvector-flat-upsert-'));
  try {
    const db = new VectorDb({
      dimensions: 2,
      storagePath: join(directory, 'vectors.db'),
      indexType: 'flat',
    });
    assert.deepEqual(db.getIndexInfo(), {
      indexType: 'flat',
      dimensions: 2,
      distanceMetric: 'cosine',
      storagePath: join(directory, 'vectors.db'),
      mutationMode: 'in-place',
      configurationVerified: true,
    });

    await db.insert({ id: 'mutable', vector: [1, 0], metadata: { revision: -1 } });
    for (let revision = 0; revision < 100; revision++) {
      const vector = revision % 2 === 0 ? [1, 0] : [0, 1];
      await db.insert({ id: 'mutable', vector, metadata: { revision } });
      const matches = await db.search({ vector, k: 10 });
      assert.deepEqual(matches.map(result => result.id), ['mutable'], `flat replacement ${revision}`);
      assert.equal(await db.len(), 1);
    }

    const compatibility = new VectorDb({
      dimensions: 2,
      storagePath: join(directory, 'compatibility.db'),
      hnswConfig: null,
    });
    assert.equal(compatibility.indexType, 'flat');
    await compatibility.insert({ id: 'flat', vector: [1, 0] });
    assert.deepEqual(
      (await compatibility.search({ vector: [1, 0], k: 1 })).map(result => result.id),
      ['flat']
    );

    const index = new VectorIndex({ dimension: 2, indexType: 'flat' });
    const indexPath = index._storagePath;
    try {
      assert.equal(index.indexType, 'flat');
      assert.equal(index.getIndexInfo().mutationMode, 'in-place');
    } finally {
      await removeIndexStores([indexPath]);
    }

    const normalized = new VectorIndex({ dimension: 2, indexType: 'FLAT' });
    const normalizedPath = normalized._storagePath;
    try {
      assert.equal(normalized.indexType, 'flat');
      assert.equal(normalized.getIndexInfo().indexType, 'flat');
    } finally {
      await removeIndexStores([normalizedPath]);
    }

    assert.throws(
      () => new VectorIndex({ dimension: 2, indexType: 'tree' }),
      /Invalid indexType: expected "hnsw" or "flat"/i
    );
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('authoritative native options reject persisted index and metric mismatch', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ruvector-config-contract-'));
  const storagePath = join(directory, 'vectors.db');
  try {
    const original = new VectorDb({
      dimensions: 2,
      distanceMetric: 'euclidean',
      storagePath,
      indexType: 'hnsw',
    });
    await original.insert({ id: 'preserved', vector: [1, 0] });

    const reopened = new VectorDb({
      dimensions: 2,
      distanceMetric: 'EUCLIDEAN',
      storagePath,
      indexType: 'HNSW',
    });
    assert.deepEqual(reopened.getIndexInfo(), {
      indexType: 'hnsw',
      dimensions: 2,
      distanceMetric: 'euclidean',
      storagePath,
      mutationMode: 'rebuild',
      configurationVerified: true,
    });

    assert.throws(
      () => new VectorDb({
        dimensions: 2,
        distanceMetric: 'cosine',
        storagePath,
        indexType: 'flat',
      }),
      error => {
        assert.match(error.message, /Vector configuration mismatch/i);
        assert.match(error.message, /native reports dimensions=2, distanceMetric=euclidean, indexType=hnsw/i);
        assert.match(error.message, /requested dimensions=2, distanceMetric=cosine, indexType=flat/i);
        return true;
      }
    );

    assert.deepEqual(
      (await reopened.search({ vector: [1, 0], k: 10 })).map(result => result.id),
      ['preserved']
    );

    const descriptor = Object.getOwnPropertyDescriptor(NativeVectorDb.prototype, 'getOptions');
    Object.defineProperty(NativeVectorDb.prototype, 'getOptions', {
      ...descriptor,
      value() {
        const effective = descriptor.value.call(this);
        return effective.storagePath === storagePath
          ? { ...effective, distanceMetric: 'cosine' }
          : effective;
      },
    });
    try {
      await assert.rejects(
        reopened.insert({ id: 'preserved', vector: [0, 1] }),
        /HNSW containment rebuild failed.*effective vector configuration mismatch/is
      );
    } finally {
      Object.defineProperty(NativeVectorDb.prototype, 'getOptions', descriptor);
    }

  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('native bindings without getOptions never certify flat indexes', async () => {
  const directory = await mkdtemp(join(tmpdir(), 'ruvector-legacy-config-'));
  const storagePath = join(directory, 'legacy.db');
  try {
    const legacy = new NativeVectorDb({
      dimensions: 2,
      distanceMetric: 'Euclidean',
      storagePath,
    });
    await legacy.insert({ id: 'legacy', vector: new Float32Array([1, 0]) });
    const descriptor = Object.getOwnPropertyDescriptor(NativeVectorDb.prototype, 'getOptions');
    assert.equal(typeof descriptor?.value, 'function', 'test requires the upgraded native getter');
    Object.defineProperty(NativeVectorDb.prototype, 'getOptions', {
      ...descriptor,
      value: undefined,
    });
    try {
      assert.throws(
        () => new VectorDb({
          dimensions: 2,
          distanceMetric: 'euclidean',
          storagePath,
          indexType: 'flat',
        }),
        /does not expose getOptions/i
      );

      const conservative = new VectorDb({
        dimensions: 2,
        distanceMetric: 'cosine',
        storagePath,
        indexType: 'hnsw',
      });
      assert.deepEqual(conservative.getIndexInfo(), {
        indexType: 'hnsw',
        dimensions: 2,
        distanceMetric: 'cosine',
        storagePath,
        mutationMode: 'rebuild',
        configurationVerified: false,
      });

      await conservative.insert({ id: 'legacy', vector: [0, 1] });
      assert.deepEqual(
        (await conservative.search({ vector: [0, 1], k: 10 })).map(result => result.id),
        ['legacy']
      );
    } finally {
      Object.defineProperty(NativeVectorDb.prototype, 'getOptions', descriptor);
    }
  } finally {
    await rm(directory, { recursive: true, force: true });
  }
});

test('implicit storage identity remains pinned across chdir and HNSW rebuild', async () => {
  const firstDirectory = await mkdtemp(join(tmpdir(), 'ruvector-cwd-first-'));
  const secondDirectory = await mkdtemp(join(tmpdir(), 'ruvector-cwd-second-'));
  const originalDirectory = process.cwd();

  try {
    process.chdir(firstDirectory);
    const db = new VectorDb({ dimensions: 2, indexType: 'hnsw' });
    await db.insert({ id: 'mutable', vector: [1, 0] });

    process.chdir(secondDirectory);
    await db.insert({ id: 'mutable', vector: [0, 1] });
    assert.deepEqual(
      (await db.search({ vector: [0, 1], k: 1 })).map(result => result.id),
      ['mutable']
    );
    assert.equal(await pathExists(join(firstDirectory, 'ruvector.db')), true);
    assert.equal(await pathExists(join(secondDirectory, 'ruvector.db')), false);
    assert.equal(db.getIndexInfo().storagePath, join(firstDirectory, 'ruvector.db'));
  } finally {
    process.chdir(originalDirectory);
    await rm(firstDirectory, { recursive: true, force: true });
    await rm(secondDirectory, { recursive: true, force: true });
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

    await assert.rejects(
      first.search({ vector: new Float32Array(16), k: 1 }),
      /Vector dimension mismatch: expected 3, got 16/i
    );
    assert.throws(
      () => new VectorDb({ dimensions: 16 }),
      error => {
        assert.match(error.message, /effective vector configuration mismatch at implicit persistent store "\.\/ruvector\.db"/i);
        assert.match(error.message, /native reports dimensions=3/i);
        assert.match(error.message, /requested dimensions=16/i);
        assert.match(error.message, /new storagePath/i);
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
