const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { RvfDatabase, RvfErrorCode } = require('../dist/index.js');

for (const backend of ['node', 'wasm']) {
  test(`${backend} rejects inconsistent batch dimensions before accepting any row`, async () => {
    const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-batch-dim-'));
    const db = await RvfDatabase.create(path.join(scratch, 'store.rvf'), { dimensions: 2 }, backend);
    try {
      for (const invalid of [[2], [2, 0, 3], []]) {
        await assert.rejects(() => db.ingestBatch([
          { id: '1', vector: [1, 0] }, { id: '2', vector: invalid },
        ]), error => error.code === RvfErrorCode.InvalidArgument);
        assert.equal((await db.status()).totalVectors, 0);
      }
      // Consistent dimensions within the batch still must match the store.
      await assert.rejects(() => db.ingestBatch([
        { id: '1', vector: [1] }, { id: '2', vector: [2] },
      ]), error => error.code === RvfErrorCode.InvalidArgument);
      assert.equal((await db.status()).totalVectors, 0);
      assert.equal((await db.ingestBatch([])).accepted, 0);
      const valid = await db.ingestBatch([
        { id: '1', vector: new Float32Array([1, 0]) }, { id: '2', vector: [2, 0] },
      ]);
      assert.equal(valid.accepted, 2);
      const hits = await db.query([2, 0], 2);
      assert.equal(hits[0].id, '2');
      assert.equal(hits[0].distance, 0);
    } finally { await db.close(); fs.rmSync(scratch, { recursive: true, force: true }); }
  });
}
