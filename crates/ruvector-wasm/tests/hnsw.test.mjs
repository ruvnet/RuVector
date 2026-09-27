import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { RuvectorWasmAdapter } from '../src/adapter.js';
import { vectors, exactIds, recall, readResults } from './fixtures.mjs';
const { VectorDB, hnswAvailable } = createRequire(import.meta.url)('../pkg-node/ruvector_wasm.js');
const vec = (...xs) => Float32Array.from(xs);

test('real WASM exposes the selected index, including flat opt out', async () => {
  assert.equal(hnswAvailable(), true);
  for (const useHnsw of [true, false]) {
    const db = new VectorDB(3, 'cosine', useHnsw);
    try {
      assert.equal(db.indexType, useHnsw ? 'hnsw' : 'flat');
      const adapter = new RuvectorWasmAdapter(db, { dimensions: 3 });
      assert.equal(adapter.usesHnsw, useHnsw);
      assert.equal(adapter.indexType, db.indexType);
    } finally { db.free(); }
  }
});

test('seeded self and held-out recall use an independent exhaustive oracle', () => {
  const data = vectors(1200, 24);
  const queries = vectors(40, 24, 0x9001);
  const graph = new VectorDB(24, 'cosine', true);
  const flat = new VectorDB(24, 'cosine', false);
  try {
    for (let i = 0; i < data.length; i++) {
      graph.insert(data[i], String(i));
      flat.insert(data[i], String(i));
    }
    for (let i = 0; i < data.length; i += 31)
      assert.equal(readResults(graph, data[i], 1)[0].id, String(i));
    let total = 0;
    for (const query of queries) {
      const expected = exactIds(data, query, 10);
      const exact = readResults(flat, query, 10);
      assert.deepEqual(exact.map(r => r.id), expected);
      const result = readResults(graph, query, 10);
      assert.equal(result.length, 10);
      assert.equal(new Set(result.map(r => r.id)).size, 10);
      assert.ok(result.every((r, i) => i === 0 || r.score >= result[i - 1].score));
      total += recall(result.map(r => r.id), expected);
    }
    const mean = total / queries.length;
    assert.ok(mean >= 0.95, `held-out recall@10 ${mean} below 0.95`);
  } finally { graph.free(); flat.free(); }
});

for (const useHnsw of [true, false]) {
  test(`${useHnsw ? 'hnsw' : 'flat'} mutation, sparse filter, metadata and score contracts`, () => {
    const db = new VectorDB(3, 'cosine', useHnsw);
    try {
      for (let i = 0; i < 100; i++) db.insert(vec(1, i / 100, 0), `common-${i}`, { kind: 'common' });
      db.insert(vec(0, 1, 0), 'rare', { kind: 'rare', nested: { value: 7 } });
      const filtered = readResults(db, vec(1, 0, 0), 10, { kind: 'rare' });
      assert.deepEqual(filtered.map(r => r.id), ['rare']);
      assert.deepEqual(filtered[0].metadata, { kind: 'rare', nested: { value: 7 } });
      assert.equal(readResults(db, vec(1, 0, 0), 10, { kind: 'absent' }).length, 0);
      assert.equal(db.delete('rare'), true);
      assert.equal(db.delete('rare'), false);
      assert.equal(db.get('rare'), undefined);
      assert.ok(!readResults(db, vec(0, 1, 0), 101).some(r => r.id === 'rare'));
      db.insert(vec(0, 0, 1), 'rare', { kind: 'replacement' });
      assert.equal(readResults(db, vec(0, 0, 1), 1)[0].id, 'rare');
      db.insert(vec(-1, 0, 0), 'rare', { kind: 'updated' });
      assert.equal(db.len(), 101);
      const updated = readResults(db, vec(-1, 0, 0), 1)[0];
      assert.equal(updated.id, 'rare');
      assert.ok(Math.abs(updated.score) < 1e-6);
      assert.deepEqual(updated.metadata, { kind: 'updated' });
      const adapter = new RuvectorWasmAdapter(db, { dimensions: 3 });
      const result = adapter.search({ vector: [-1, 0, 0], k: 2 });
      assert.ok(result[0].score >= result[1].score);
      assert.equal(result[0].distance, updated.score);
      assert.equal(readResults(db, vec(1, 0, 0), 0).length, 0);
      assert.equal(readResults(db, vec(1, 0, 0), 1000).length, 101);
    } finally { db.free(); }
  });
  test(`${useHnsw ? 'hnsw' : 'flat'} rejects invalid inputs without corrupting the index`, () => {
    assert.throws(() => new VectorDB(0, 'cosine', useHnsw));
    assert.throws(() => new VectorDB(65537, 'cosine', useHnsw));
    const db = new VectorDB(3, 'cosine', useHnsw);
    try {
      for (const invalid of [vec(1, 2), vec(1, NaN, 0), vec(Infinity, 0, 1), vec(3e38, 0, 1)]) {
        assert.throws(() => db.insert(invalid, 'bad'));
        assert.throws(() => db.search(invalid, 1));
      }
      assert.throws(() => db.insertBatch([
        { id: 'valid', vector: vec(1, 0, 0) },
        { id: 'invalid', vector: vec(1, 2) },
      ]));
      assert.equal(db.len(), 0);
      db.insert(vec(1, 0, 0), 'ok');
      assert.equal(readResults(db, vec(1, 0, 0), 1)[0].id, 'ok');
    } finally { db.free(); }
  });
  test(`${useHnsw ? 'hnsw' : 'flat'} metadata-only update keeps the index intact`, () => {
    const data = vectors(200, 8);
    const queries = vectors(10, 8, 0x9001);
    const db = new VectorDB(8, 'cosine', useHnsw);
    try {
      data.forEach((v, i) => db.insert(v, String(i), { version: 1 }));
      const snapshot = () => queries.map(q => readResults(db, q, 10).map(r => `${r.id}:${r.score}`));
      const before = snapshot();
      db.insert(data[42], '42', { version: 2 });
      assert.equal(db.len(), 200);
      assert.deepEqual(snapshot(), before);
      const hit = readResults(db, data[42], 1)[0];
      assert.equal(hit.id, '42');
      assert.deepEqual(hit.metadata, { version: 2 });
      assert.deepEqual(readResults(db, data[42], 1, { version: 2 }).map(r => r.id), ['42']);
    } finally { db.free(); }
  });
}

for (const metric of ['euclidean', 'manhattan', 'dotproduct']) {
  test(`${metric} graph and flat share ascending distance semantics`, () => {
    const indexes = [true, false].map(enabled => new VectorDB(2, metric, enabled));
    try {
      const output = indexes.map(db => {
        db.insert(vec(1, 0), 'best');
        db.insert(vec(.5, .5), 'middle');
        db.insert(vec(-1, 0), 'worst');
        const result = readResults(db, vec(1, 0), 3);
        assert.deepEqual(result.map(r => r.id), ['best', 'middle', 'worst']);
        assert.ok(result[0].score < result[1].score);
        assert.ok(result[1].score < result[2].score);
        const adapter = new RuvectorWasmAdapter(db, { dimensions: 2, metric });
        const converted = adapter.search({ vector: [1, 0], k: 3 });
        assert.ok(converted[0].score > converted[1].score);
        assert.ok(converted[1].score > converted[2].score);
        return result.map(r => r.score);
      });
      assert.deepEqual(output[0], output[1]);
    } finally { indexes.forEach(db => db.free()); }
  });
}
