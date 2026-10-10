const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { RvfDatabase } = require('../dist/index.js');
const { NodeBackend } = require('../dist/backend.js');

for (const [name, filter, survivor] of [
  ['equality', { op: 'eq', fieldId: 0, value: 'red' }, '2'],
  ['conjunction', { op: 'and', exprs: [
    { op: 'eq', fieldId: 0, value: 'red' }, { op: 'ne', fieldId: 0, value: 'blue' },
  ] }, '2'],
  ['negation', { op: 'not', expr: { op: 'eq', fieldId: 0, value: 'red' } }, '1'],
  ['set membership', { op: 'in', fieldId: 0, values: ['blue'] }, '1'],
  ['range', { op: 'range', fieldId: 0, low: 'blue', high: 'red' }, '1'],
]) {
  test(`deleteByFilter maps a public ${name} filter to the real native parser`, async () => {
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-sdk-filter-'));
    const backend = new NodeBackend();
    await backend.create(path.join(scratch, 'store.rvf'), { dimensions: 2 });
    // Seed metadata via the actual native API: SDK metadata ingestion is deliberately unsupported.
    // Access the initialized native handle only for fixture setup; the operation under test is
    // the unchanged public RvfDatabase facade and real N-API filter parser/runtime.
    backend.handle.ingestBatch(new Float32Array([1, 0, 0, 1]), [1, 2], [
      { fieldId: 0, valueType: 'string', value: 'red' },
      { fieldId: 0, valueType: 'string', value: 'blue' },
    ]);
    const db = RvfDatabase.fromBackend(backend);
    try {
      const result = await db.deleteByFilter(filter);
      assert.equal(result.deleted, 1);
      const hits = await db.query([1, 0], 2);
      assert.deepEqual(hits.map(hit => hit.id), [survivor]);
      assert.equal((await db.deleteByFilter(filter)).deleted, 0);
    } finally { await db.close(); fs.rmSync(scratch, { recursive: true, force: true }); }
  });
}
