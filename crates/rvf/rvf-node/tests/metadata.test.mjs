// node --test tests/metadata.test.mjs  (after building the addon)
// Issue #704: per-vector metadata is durable, returned on hits, filterable.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadAddon } from './load-addon.mjs';

const { RvfDatabase } = loadAddon();

function withDir(fn) {
  const dir = mkdtempSync(join(tmpdir(), 'rvf-node-meta-'));
  return Promise.resolve(fn(dir)).finally(() => rmSync(dir, { recursive: true, force: true }));
}

const s = (fieldId, value) => ({ fieldId, valueType: 'string', value });
const f = (fieldId, value) => ({ fieldId, valueType: 'f64', value: String(value) });
const b = (fieldId, value) => ({ fieldId, valueType: 'bool', value: String(value) });
const eqRed = JSON.stringify({ op: 'eq', fieldId: 0, valueType: 'string', value: 'red' });

function fields(entries) {
  return Object.fromEntries((entries ?? []).map((e) => [e.fieldId, `${e.valueType}:${e.value}`]));
}

test('records survive close/reopen, ride on hits, and filter', () =>
  withDir(async (dir) => {
    const path = join(dir, 'm.rvf');
    let db = RvfDatabase.create(path, { dimension: 2, metric: 'cosine' });
    const res = await db.ingestBatchWithMetadataAsync(
      new Float32Array([1, 0, 0, 1, 0.7, 0.7]),
      [101, 102, 103],
      [
        { id: 101, fields: [s(0, 'red'), f(1, 5), b(2, true)] },
        { id: 102, fields: [s(0, 'blue'), f(1, 2.5)] },
        // 103 deliberately has no record
      ],
    );
    assert.equal(res.accepted, 3);
    db.close();

    db = RvfDatabase.open(path);
    assert.deepEqual(fields(db.getMetadata(101)), { 0: 'string:red', 1: 'f64:5.0', 2: 'bool:true' });
    assert.deepEqual(fields(db.getMetadata(102)), { 0: 'string:blue', 1: 'f64:2.5' });
    assert.equal(db.getMetadata(103), null);

    const hits = db.query(new Float32Array([1, 0]), 3);
    const byId = Object.fromEntries(hits.map((h) => [h.id, h]));
    assert.deepEqual(fields(byId[101].metadata), { 0: 'string:red', 1: 'f64:5.0', 2: 'bool:true' });
    assert.equal(byId[103].metadata ?? null, null);

    const red = db.query(new Float32Array([0, 1]), 3, { filter: eqRed });
    assert.deepEqual(red.map((h) => h.id), [101]);
    const big = db.query(new Float32Array([0, 1]), 3, {
      filter: JSON.stringify({ op: 'ge', fieldId: 1, valueType: 'f64', value: '5' }),
    });
    assert.deepEqual(big.map((h) => h.id), [101]);
    db.close();
  }));

test('file metadata is durable and rvf.* keys are reserved', () =>
  withDir((dir) => {
    const path = join(dir, 'f.rvf');
    let db = RvfDatabase.create(path, { dimension: 2 });
    assert.equal(db.getFileMetadata('sdk.test'), null);
    db.setFileMetadata('sdk.test', '{"a":1}');
    assert.throws(() => db.setFileMetadata('rvf.embedding.space_id', 'x'));
    db.close();
    db = RvfDatabase.open(path);
    assert.equal(db.getFileMetadata('sdk.test'), '{"a":1}');
    db.close();
  }));

test('null, bytes and out-of-range field ids', () =>
  withDir((dir) => {
    const db = RvfDatabase.create(join(dir, 'n.rvf'), { dimension: 2 });
    db.ingestBatchWithMetadata(new Float32Array([1, 0]), [1], [
      { id: 1, fields: [{ fieldId: 0, valueType: 'null', value: '' }, { fieldId: 1, valueType: 'bytes', value: '00ff' }] },
    ]);
    assert.deepEqual(fields(db.getMetadata(1)), { 0: 'null:', 1: 'bytes:00ff' });
    assert.throws(
      () => db.ingestBatchWithMetadata(new Float32Array([0, 1]), [2], [{ id: 2, fields: [s(70000, 'x')] }]),
      /exceeds 65535/,
    );
    db.close();
  }));
