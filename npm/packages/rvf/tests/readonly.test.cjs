const assert = require('node:assert/strict');
const { test } = require('node:test');
const fs = require('node:fs');
const os = require('node:os');
const path = require('node:path');
const { RvfDatabase } = require('../dist/index.js');

test('closing a stale read-only reader preserves newer durable IDs and the next label', async () => {
  const scratch = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-readonly-map-'));
  const target = path.join(scratch, 'store.rvf');
  const handles = [];
  try {
    const writer = await RvfDatabase.create(target, { dimensions: 2 }, 'node'); handles.push(writer);
    await writer.ingestBatch([{ id: 'a', vector: [1, 0] }]);
    const reader = await RvfDatabase.openReadonly(target, 'node'); handles.push(reader);
    await writer.ingestBatch([{ id: 'b', vector: [0, 1] }]);
    await writer.close();
    const latest = fs.readFileSync(`${target}.idmap.json`);
    await reader.close();
    assert.deepEqual(fs.readFileSync(`${target}.idmap.json`), latest);
    const reopened = await RvfDatabase.open(target, 'node'); handles.push(reopened);
    await reopened.ingestBatch([{ id: 'c', vector: [2, 0] }]);
    const hits = await reopened.query([0, 1], 5);
    assert.equal(hits[0].id, 'b');
    assert.deepEqual(hits.map(hit => hit.id).sort(), ['a', 'b', 'c']);
    await reopened.close();
    const finalReader = await RvfDatabase.openReadonly(target, 'node'); handles.push(finalReader);
    assert.equal((await finalReader.query([0, 1], 1))[0].id, 'b');
    const finalMap = fs.readFileSync(`${target}.idmap.json`);
    await finalReader.close();
    assert.deepEqual(fs.readFileSync(`${target}.idmap.json`), finalMap);
  } finally {
    for (const handle of handles) await handle.close();
    fs.rmSync(scratch, { recursive: true, force: true });
  }
});
