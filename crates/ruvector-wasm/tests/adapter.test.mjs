/**
 * Tests for RuvectorWasmAdapter.
 *
 * Uses a FakeVectorDB that reproduces the three problematic behaviours of the
 * real WASM build:
 *   1. flat (no HNSW) — `usesHnsw === false`
 *   2. `score` is a cosine *distance* (lower is better)
 *   3. metadata does not round-trip (search/get return `{}`)
 *
 * The adapter must hide all three: similarity higher-is-better with correct
 * ordering, and metadata round-tripped via the sidecar.
 *
 * Run: node --test crates/ruvector-wasm/tests/adapter.test.mjs
 */

import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  RuvectorWasmAdapter,
  distanceToSimilarity,
  WASM_HNSW_AVAILABLE,
} from '../src/adapter.js';

function cosineDistance(a, b) {
  let dot = 0;
  let na = 0;
  let nb = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    na += a[i] * a[i];
    nb += b[i] * b[i];
  }
  const denom = Math.sqrt(na) * Math.sqrt(nb);
  return denom === 0 ? 1 : 1 - dot / denom;
}

/** Mimics the real WASM VectorDB: flat index, distance score, no metadata round-trip. */
class FakeVectorDB {
  constructor() {
    this.store = new Map();
    this._auto = 0;
  }

  insert(vector, id /*, metadata */) {
    const key = id ?? `auto-${this._auto++}`;
    // Note: metadata is intentionally dropped — reproduces the WASM bug.
    this.store.set(key, Float32Array.from(vector));
    return key;
  }

  insertBatch(entries) {
    return entries.map((e) => this.insert(e.vector, e.id, e.metadata));
  }

  search(vector, k /*, filter */) {
    const results = [];
    for (const [id, vec] of this.store) {
      results.push({ id, score: cosineDistance(vector, vec), metadata: {} });
    }
    results.sort((a, b) => a.score - b.score); // flat scan, ascending distance
    return results.slice(0, k);
  }

  get(id) {
    const vec = this.store.get(id);
    return vec ? { id, vector: vec, metadata: {} } : null;
  }

  delete(id) {
    return this.store.delete(id);
  }

  len() {
    return this.store.size;
  }

  isEmpty() {
    return this.store.size === 0;
  }
}

test('distanceToSimilarity: cosine distance -> higher-is-better similarity', () => {
  assert.equal(distanceToSimilarity('cosine', 0), 1);
  assert.equal(distanceToSimilarity('cosine', 0.25), 0.75);
  assert.ok(
    distanceToSimilarity('cosine', 0.1) > distanceToSimilarity('cosine', 0.4)
  );
});

test('finding #2: search returns similarity (higher is better) with a, b before c', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 3, metric: 'cosine' });

  // a and b are close to the query [1,0,0]; c is orthogonal.
  adapter.insert({ id: 'a', vector: [1, 0, 0] });
  adapter.insert({ id: 'b', vector: [0.9, 0.1, 0] });
  adapter.insert({ id: 'c', vector: [0, 0, 1] });

  const results = adapter.search({ vector: [1, 0, 0], k: 3 });
  assert.deepEqual(
    results.map((r) => r.id),
    ['a', 'b', 'c']
  );

  // Higher is better, and the best result outscores the worst.
  assert.ok(results[0].similarity >= results[1].similarity);
  assert.ok(results[1].similarity >= results[2].similarity);
  assert.ok(results[0].similarity > results[2].similarity);
  // `.score` honours the documented "higher is better" contract.
  assert.equal(results[0].score, results[0].similarity);
  // Raw distance preserved (lower is better) and consistent with similarity.
  assert.ok(results[0].distance <= results[2].distance);
  assert.ok(Math.abs(results[0].similarity - (1 - results[0].distance)) < 1e-6);
});

test('finding #3: metadata round-trips via the sidecar', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 3, metric: 'cosine' });

  const meta = { title: 'doc-a', tags: ['x', 'y'] };
  adapter.insert({ id: 'a', vector: [1, 0, 0], metadata: meta });
  adapter.insert({ id: 'b', vector: [0, 1, 0], metadata: { title: 'doc-b' } });

  // Raw WASM would return {}; the adapter restores the real metadata.
  const [top] = adapter.search({ vector: [1, 0, 0], k: 1 });
  assert.deepEqual(top.metadata, meta);

  const got = adapter.get('a');
  assert.deepEqual(got.metadata, meta);
});

test('insertBatch round-trips metadata in order', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 2, metric: 'cosine' });

  const ids = adapter.insertBatch([
    { id: 'one', vector: [1, 0], metadata: { n: 1 } },
    { id: 'two', vector: [0, 1], metadata: { n: 2 } },
  ]);
  assert.deepEqual(ids, ['one', 'two']);
  assert.deepEqual(adapter.get('two').metadata, { n: 2 });
});

test('filter is applied against sidecar metadata', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 2, metric: 'cosine' });

  adapter.insert({ id: 'a', vector: [1, 0], metadata: { kind: 'fruit' } });
  adapter.insert({ id: 'b', vector: [0.95, 0.05], metadata: { kind: 'veg' } });
  adapter.insert({ id: 'c', vector: [0.9, 0.1], metadata: { kind: 'fruit' } });

  const results = adapter.search({ vector: [1, 0], k: 2, filter: { kind: 'fruit' } });
  assert.deepEqual(
    results.map((r) => r.id),
    ['a', 'c']
  );
});

test('finding #1: index type reports flat (HNSW not active in WASM build)', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 2 });
  assert.equal(WASM_HNSW_AVAILABLE, false);
  assert.equal(adapter.usesHnsw, false);
  assert.equal(adapter.indexType, 'flat');
});

test('delete drops sidecar metadata and updates length', () => {
  const db = new FakeVectorDB();
  const adapter = new RuvectorWasmAdapter(db, { dimensions: 2 });
  adapter.insert({ id: 'a', vector: [1, 0], metadata: { keep: false } });
  assert.equal(adapter.len(), 1);
  assert.equal(adapter.delete('a'), true);
  assert.equal(adapter.len(), 0);
  assert.equal(adapter.get('a'), null);
  assert.equal(adapter.isEmpty(), true);
});
