const assert = require('node:assert/strict');
const { test } = require('node:test');
const { RvfDatabase } = require('../dist/index.js');

// Uses the real @ruvector/rvf-wasm microkernel, not an emulated distance function.
for (const [metric, firstId, firstDistance] of [
  ['l2', '1', 1], ['cosine', '2', 0], ['dotproduct', '2', -20],
]) {
  test(`WASM query honors the store's ${metric} metric`, async () => {
    const db = await RvfDatabase.create('in-memory', { dimensions: 2, metric }, 'wasm');
    try {
      await db.ingestBatch([
        { id: '1', vector: [1, 0] }, { id: '2', vector: [10, 10] },
      ]);
      const positional = await db.query([1, 1], 2);
      const options = await db.query([1, 1], { k: 2 });
      assert.equal(positional[0].id, firstId);
      assert.ok(Math.abs(positional[0].distance - firstDistance) < 1e-6);
      assert.deepEqual(options, positional);
      assert.equal(positional.length, 2);
    } finally { await db.close(); }
  });
}
