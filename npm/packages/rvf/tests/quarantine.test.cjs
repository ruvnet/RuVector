const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { RvfDatabase, RvfErrorCode } = require('../dist/index.js');

test('retry after sidecar quarantine stays refused until a valid mapping is restored', async () => {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-quarantine-retry-'));
  const target = path.join(scratch, 'store.rvf');
  const sidecar = `${target}.idmap.json`;
  try {
    const db = await RvfDatabase.create(target, { dimensions: 2 }, 'node');
    await db.ingestBatch([{ id: 'a', vector: [1, 0] }]); await db.close();
    const validMapping = fs.readFileSync(sidecar);
    const broken = '{broken JSON'; fs.writeFileSync(sidecar, broken);
    for (const method of ['open', 'open', 'openReadonly']) {
      await assert.rejects(async () => {
        const unexpected = await RvfDatabase[method](target, 'node');
        await unexpected.close();
      }, error => error.code === RvfErrorCode.SidecarCorrupt);
      assert.equal(fs.existsSync(sidecar), false);
    }
    const quarantines = fs.readdirSync(scratch).filter(name => name.startsWith('store.rvf.idmap.json.corrupt-'));
    assert.equal(quarantines.length, 1);
    assert.equal(fs.readFileSync(path.join(scratch, quarantines[0]), 'utf8'), broken);
    // A restored valid mapping takes precedence over the preserved quarantine.
    fs.writeFileSync(sidecar, validMapping);
    const recovered = await RvfDatabase.open(target, 'node');
    try {
      await recovered.ingestBatch([{ id: 'b', vector: [0, 1] }]);
      assert.deepEqual((await recovered.query([1, 0], 2)).map(hit => hit.id).sort(), ['a', 'b']);
    } finally { await recovered.close(); }
    // An ordinary fresh empty store with no quarantine remains valid.
    const fresh = await RvfDatabase.create(path.join(scratch, 'fresh.rvf'), { dimensions: 2 }, 'node');
    await fresh.close();
  } finally { fs.rmSync(scratch, { recursive: true, force: true }); }
});
