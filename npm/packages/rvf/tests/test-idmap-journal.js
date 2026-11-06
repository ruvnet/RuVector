'use strict';
/**
 * Tests for the append-only id-map journal (`<store>.idmap.json.journal`).
 *
 * Uses a mock native handle like test-id-mapping.js, so no N-API addon is
 * needed. Run after `tsc`: node tests/test-idmap-journal.js
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');

const { NodeBackend } = require('../dist/backend');
const { replayJournal } = require('../dist/idmap-journal');
const { RvfError, RvfErrorCode } = require('../dist/errors');

// Sidecar exactly as written by @ruvector/rvf 0.4.2 (origin/main) with the
// real native backend: 7 ids ingested, 'to-delete' (label 7) deleted.
const OLD_FORMAT_SIDECAR =
  '{"idToLabel":{"42":5,"alpha":1,"beta":2,"gamma":3,"déjà-vu ✓":4,"with\\"quote":6},' +
  '"labelToId":{"1":"alpha","2":"beta","3":"gamma","4":"déjà-vu ✓","5":"42","6":"with\\"quote"},' +
  '"nextLabel":8}';

class MockNativeHandle {
  constructor() { this.vectors = new Map(); this.calls = []; }
  ingestBatch(flat, ids) {
    this.calls.push('ingestBatch');
    const dim = flat.length / ids.length;
    ids.forEach((label, i) => this.vectors.set(label, flat.slice(i * dim, (i + 1) * dim)));
    return { accepted: ids.length, rejected: 0, epoch: 1 };
  }
  query(vector, k) {
    this.calls.push('query');
    const out = [];
    for (const [id, v] of this.vectors) {
      let d = 0;
      for (let i = 0; i < vector.length; i++) d += (vector[i] - v[i]) ** 2;
      out.push({ id, distance: d });
    }
    return out.sort((a, b) => a.distance - b.distance).slice(0, k);
  }
  delete(ids) {
    this.calls.push('delete');
    let deleted = 0;
    for (const id of ids) if (this.vectors.delete(id)) deleted++;
    return { deleted, epoch: 1 };
  }
  compact() { this.calls.push('compact'); return { segmentsCompacted: 0, bytesReclaimed: 0, epoch: 1 }; }
  status() { return { total_vectors: this.vectors.size }; }
  close() {}
}

class AsyncMockNativeHandle extends MockNativeHandle {
  async ingestBatchAsync(flat, ids) { return this.viaSync('ingestBatchAsync', () => super.ingestBatch(flat, ids)); }
  async queryAsync(v, k) { return this.viaSync('queryAsync', () => super.query(v, k)); }
  async compactAsync() { return this.viaSync('compactAsync', () => super.compact()); }
  async deleteAsync(ids) { return this.viaSync('deleteAsync', () => super.delete(ids)); }
  viaSync(name, fn) { const r = fn(); this.calls.pop(); this.calls.push(name); return r; }
}

/** A backend opened through the real open() path with a mock native layer. */
async function openBackend(storePath, Handle = MockNativeHandle, readonly = false) {
  if (!fs.existsSync(storePath)) fs.writeFileSync(storePath, '');
  const backend = new NodeBackend();
  const handle = new Handle();
  backend['native'] = { open: () => handle, openReadonly: () => handle };
  if (readonly) await backend.openReadonly(storePath);
  else await backend.open(storePath);
  return backend;
}

const vec = (x) => new Float32Array([x, 0, 0, 0]);
const sidecarOf = (store) => `${store}.idmap.json`;
const journalOf = (store) => `${store}.idmap.json.journal`;
const HEADER = '{"rvfIdmapJournal":1}\n';

let passed = 0, failed = 0;
async function test(name, fn) {
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-journal-test-'));
  try {
    await fn(tmp);
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.stack || err.message}`);
    failed++;
  } finally {
    try { fs.rmSync(tmp, { recursive: true, force: true }); } catch {}
  }
}

(async () => {
  console.log('--- NodeBackend id-map journal ---');

  await test('old-format sidecar (0.4.2) opens and every id resolves', async (tmp) => {
    const store = path.join(tmp, 'old.rvf');
    fs.writeFileSync(sidecarOf(store), OLD_FORMAT_SIDECAR);
    const b = await openBackend(store);
    const expected = { alpha: 1, beta: 2, gamma: 3, 'déjà-vu ✓': 4, '42': 5, 'with"quote': 6 };
    for (const [id, label] of Object.entries(expected)) {
      assert.strictEqual(b['idToLabel'].get(id), label, id);
      assert.strictEqual(b['labelToId'].get(label), id);
    }
    assert.strictEqual(b['idToLabel'].has('to-delete'), false);
    assert.strictEqual(b['nextLabel'], 8, 'nextLabel preserved, deleted label 7 not reused');
    await b.ingestBatch([{ id: 'fresh', vector: vec(1) }]);
    assert.strictEqual(b['idToLabel'].get('fresh'), 8);
    await b.close();
  });

  await test('ingest appends to the journal instead of rewriting the snapshot', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }, { id: 'b', vector: vec(2) }]);
    await b.ingestBatch([{ id: 'c', vector: vec(3) }]);
    await b.ingestBatch([{ id: 'a', vector: vec(4) }]); // upsert: no new label, no op
    assert.strictEqual(fs.existsSync(sidecarOf(store)), false, 'no snapshot rewrite per ingest');
    assert.strictEqual(
      fs.readFileSync(journalOf(store), 'utf-8'),
      HEADER + '["a","a",1]\n["a","b",2]\n["a","c",3]\n',
    );
    await b.close();
  });

  await test('journal replay after a crash (no compaction) restores ids', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    await b.ingestBatch([{ id: 'x', vector: vec(1) }, { id: 'y', vector: vec(2) }]);
    // Simulated crash: the handle is dropped without close().
    const r = await openBackend(store);
    assert.strictEqual(r['idToLabel'].get('x'), 1);
    assert.strictEqual(r['idToLabel'].get('y'), 2);
    assert.strictEqual(r['nextLabel'], 3);
    await r.close();
  });

  await test('torn last line is dropped and cut off before the next append', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(sidecarOf(store), OLD_FORMAT_SIDECAR);
    fs.writeFileSync(journalOf(store), HEADER + '["a","j1",8]\n["a","j2",9]\n["a","to');
    const b = await openBackend(store);
    assert.strictEqual(b['idToLabel'].get('j1'), 8);
    assert.strictEqual(b['idToLabel'].get('j2'), 9);
    assert.strictEqual(b['nextLabel'], 10);
    assert.strictEqual(
      fs.readFileSync(journalOf(store), 'utf-8'),
      HEADER + '["a","j1",8]\n["a","j2",9]\n',
      'torn tail truncated on read-write open',
    );
    await b.ingestBatch([{ id: 'j3', vector: vec(1) }]);
    assert.ok(fs.readFileSync(journalOf(store), 'utf-8').endsWith('["a","j2",9]\n["a","j3",10]\n'));
    // Crash again and reopen: the journal is still well-formed.
    const r = await openBackend(store);
    assert.strictEqual(r['idToLabel'].get('j3'), 10);
    await r.close();
  });

  await test('a torn header alone leaves an empty, usable journal', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(journalOf(store), '{"rvfIdm');
    const b = await openBackend(store);
    assert.strictEqual(b['idToLabel'].size, 0);
    assert.strictEqual(fs.existsSync(journalOf(store)), false);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }]);
    assert.strictEqual(fs.readFileSync(journalOf(store), 'utf-8'), HEADER + '["a","a",1]\n');
    await b.close();
  });

  await test('corruption before the tail fails loud with SidecarCorrupt', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(journalOf(store), HEADER + '["a","a",1]\nGARBAGE\n["a","b",2]\n');
    await assert.rejects(
      () => openBackend(store),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.SidecarCorrupt,
    );
    assert.strictEqual(fs.existsSync(journalOf(store)), false, 'quarantined, not overwritten');
    assert.strictEqual(
      fs.readdirSync(tmp).filter((f) => f.includes('.journal.corrupt-')).length, 1,
    );
  });

  await test('a damaged header followed by ops is corruption, not a torn tail', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(journalOf(store), '{"rvfIdmapJournal":9}\n["a","a",1]\n');
    await assert.rejects(
      () => openBackend(store),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.SidecarCorrupt,
    );
  });

  await test('query and create options reach the binding as camelCase keys', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    let createOpts;
    const b = new NodeBackend();
    b['native'] = { create: (_p, o) => { createOpts = o; return new MockNativeHandle(); } };
    await b.create(store, { dimensions: 4, efConstruction: 321 });
    assert.strictEqual(createOpts.efConstruction, 321);
    let queryOpts;
    b['handle'].query = (_v, _k, o) => { queryOpts = o; return []; };
    await b.query(vec(1), 1, { efSearch: 555, timeoutMs: 7 });
    assert.strictEqual(queryOpts.efSearch, 555);
    assert.strictEqual(queryOpts.timeoutMs, 7);
    assert.ok(!('ef_search' in queryOpts) && !('ef_construction' in createOpts));
    await b.close();
  });

  await test('a label owned by two ids is corruption', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(sidecarOf(store), OLD_FORMAT_SIDECAR);
    fs.writeFileSync(journalOf(store), HEADER + '["a","intruder",3]\n');
    await assert.rejects(
      () => openBackend(store),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.SidecarCorrupt,
    );
  });

  await test('delete ops in the journal are honoured on replay', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    await b.ingestBatch([{ id: 'keep', vector: vec(1) }, { id: 'gone', vector: vec(2) }]);
    assert.strictEqual((await b.delete(['gone', 'never-existed'])).deleted, 1);
    assert.ok(fs.readFileSync(journalOf(store), 'utf-8').endsWith('["d","gone"]\n'));
    const r = await openBackend(store); // crash: replay only
    assert.strictEqual(r['idToLabel'].get('keep'), 1);
    assert.strictEqual(r['idToLabel'].has('gone'), false);
    assert.strictEqual(r['labelToId'].has(2), false);
    assert.strictEqual(r['nextLabel'], 3, 'deleted label is not reissued');
    await r.ingestBatch([{ id: 'gone', vector: vec(3) }]);
    assert.strictEqual(r['idToLabel'].get('gone'), 3, 're-added id gets a fresh label');
    await r.close();
  });

  await test('close compacts: old-format snapshot, journal removed, reopen works', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }, { id: 'b', vector: vec(2) }, { id: 'c', vector: vec(3) }]);
    await b.delete(['b']);
    await b.close();
    assert.strictEqual(fs.existsSync(journalOf(store)), false);
    const snap = JSON.parse(fs.readFileSync(sidecarOf(store), 'utf-8'));
    assert.deepStrictEqual(Object.keys(snap).sort(), ['idToLabel', 'labelToId', 'nextLabel']);
    assert.deepStrictEqual(snap, {
      idToLabel: { a: 1, c: 3 },
      labelToId: { 1: 'a', 3: 'c' },
      nextLabel: 4,
    });
    const r = await openBackend(store);
    assert.strictEqual(r['idToLabel'].get('c'), 3);
    await r.close();
  });

  await test('compact() folds the journal into the snapshot', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }]);
    await b.compact();
    assert.strictEqual(fs.existsSync(journalOf(store)), false);
    assert.strictEqual(JSON.parse(fs.readFileSync(sidecarOf(store), 'utf-8')).nextLabel, 2);
    await b.ingestBatch([{ id: 'b', vector: vec(2) }]);
    assert.strictEqual(fs.readFileSync(journalOf(store), 'utf-8'), HEADER + '["a","b",2]\n');
    await b.close();
  });

  await test('journal compacts automatically past the size threshold', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    b['journalCompactMinBytes'] = 256;
    for (let i = 0; i < 100; i++) await b.ingestBatch([{ id: `id-${i}`, vector: vec(i) }]);
    const snap = JSON.parse(fs.readFileSync(sidecarOf(store), 'utf-8'));
    assert.ok(snap.nextLabel > 1, 'a snapshot was written mid-run');
    const journalBytes = fs.existsSync(journalOf(store)) ? fs.statSync(journalOf(store)).size : 0;
    assert.ok(journalBytes <= Math.max(256, fs.statSync(sidecarOf(store)).size) + 64);
    const r = await openBackend(store); // crash: snapshot + journal tail
    assert.strictEqual(r['idToLabel'].size, 100);
    assert.strictEqual(r['nextLabel'], 101);
    await r.close();
  });

  await test('replaying a journal already folded into the snapshot is a no-op', async () => {
    // Crash between snapshot rename and journal removal.
    const state = {
      idToLabel: new Map([['x', 3]]),
      labelToId: new Map([[3, 'x']]),
      nextLabel: 4,
    };
    const buf = Buffer.from(HEADER + '["a","x",1]\n["a","y",2]\n["d","x"]\n["d","y"]\n["a","x",3]\n');
    const res = replayJournal(buf, state);
    assert.strictEqual(res.ops, 5);
    assert.deepStrictEqual([...state.idToLabel], [['x', 3]]);
    assert.deepStrictEqual([...state.labelToId], [[3, 'x']]);
    assert.strictEqual(state.nextLabel, 4);
  });

  await test('journal append failure rolls back the new labels and skips native ingest', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store);
    fs.mkdirSync(journalOf(store)); // appending to a directory fails
    await assert.rejects(
      () => b.ingestBatch([{ id: 'a', vector: vec(1) }]),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.SidecarWriteFailed,
    );
    assert.strictEqual(b['idToLabel'].has('a'), false);
    assert.strictEqual(b['nextLabel'], 1);
    assert.strictEqual(b['handle'].vectors.size, 0, 'native ingest not attempted');
    fs.rmdirSync(journalOf(store));
    await b.ingestBatch([{ id: 'a', vector: vec(1) }]);
    assert.strictEqual(b['idToLabel'].get('a'), 1);
    await b.close();
  });

  await test('read-only open replays but never writes', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const torn = HEADER + '["a","a",1]\n["a","b';
    fs.writeFileSync(journalOf(store), torn);
    const b = await openBackend(store, MockNativeHandle, true);
    assert.strictEqual(b['idToLabel'].get('a'), 1);
    await b.close();
    assert.strictEqual(fs.readFileSync(journalOf(store), 'utf-8'), torn);
    assert.strictEqual(fs.existsSync(sidecarOf(store)), false);
  });

  await test('create() refuses a leftover journal unless overwrite is set', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    fs.writeFileSync(journalOf(store), HEADER + '["a","stale",1]\n');
    const b = new NodeBackend();
    b['native'] = { create: () => new MockNativeHandle() };
    await assert.rejects(
      () => b.create(store, { dimensions: 4 }),
      (e) => e instanceof RvfError && e.code === RvfErrorCode.FileExists,
    );
    await b.create(store, { dimensions: 4, overwrite: true });
    assert.strictEqual(fs.existsSync(journalOf(store)), false, 'stale journal replaced');
    await b.ingestBatch([{ id: 'fresh', vector: vec(1) }]);
    assert.strictEqual(b['idToLabel'].get('fresh'), 1);
    assert.strictEqual(b['idToLabel'].has('stale'), false);
    await b.close();
  });

  await test('async native methods are used when present', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store, AsyncMockNativeHandle);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }]);
    const res = await b.query(vec(1), 1);
    await b.compact();
    await b.delete(['a']);
    assert.strictEqual(res[0].id, 'a');
    assert.deepStrictEqual(b['handle'].calls, ['ingestBatchAsync', 'queryAsync', 'compactAsync', 'deleteAsync']);
    assert.strictEqual(b['idToLabel'].has('a'), false);
    await b.close();
  });

  await test('sync native methods are used when async ones are absent', async (tmp) => {
    const store = path.join(tmp, 's.rvf');
    const b = await openBackend(store, MockNativeHandle);
    await b.ingestBatch([{ id: 'a', vector: vec(1) }]);
    const res = await b.query(vec(1), 1);
    await b.compact();
    await b.delete(['a']);
    assert.strictEqual(res[0].id, 'a');
    assert.deepStrictEqual(b['handle'].calls, ['ingestBatch', 'query', 'compact', 'delete']);
    await b.close();
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`);
  console.log('='.repeat(60));
  process.exit(failed > 0 ? 1 : 0);
})();
