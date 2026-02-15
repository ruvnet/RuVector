/**
 * Smoke test: Verify the patched @ruvector/rvf TS SDK exposes putMeta/getMeta/listMetaKeys/deleteMeta
 * through the NodeBackend -> N-API chain.
 *
 * Run from: cd /Users/danielalberttis/Desktop/Projects/claude-flow-local && node /path/to/this/file
 */
'use strict';

const path = require('path');
const fs = require('fs');
const os = require('os');

async function main() {
    // Load the patched @ruvector/rvf from claude-flow-local's node_modules
    const rvf = require('@ruvector/rvf');
    const { RvfDatabase } = rvf;

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sdk-meta-'));
    const storePath = path.join(tmpDir, 'test.rvf');

    console.log('--- Session 1: Create store and write META_SEG entries ---');
    const db = await RvfDatabase.create(storePath, { dimensions: 4, metric: 'cosine' });

    // Test putMeta
    await db.putMeta('e:entry1', Buffer.from(JSON.stringify({ id: 'entry1', key: 'hello', namespace: 'default', content: 'world' })));
    await db.putMeta('e:entry2', Buffer.from(JSON.stringify({ id: 'entry2', key: 'foo', namespace: 'test', content: 'bar' })));
    console.log('putMeta: 2 entries written');

    // Test getMeta
    const raw1 = await db.getMeta('e:entry1');
    const obj1 = JSON.parse(Buffer.from(raw1).toString());
    console.log('getMeta e:entry1:', obj1.key, '=', obj1.content);

    // Test listMetaKeys
    const keys = await db.listMetaKeys();
    console.log('listMetaKeys:', keys);

    // Test deleteMeta
    const deleted = await db.deleteMeta('e:entry2');
    console.log('deleteMeta e:entry2:', deleted);

    const keysAfterDelete = await db.listMetaKeys();
    console.log('listMetaKeys after delete:', keysAfterDelete);

    await db.close();

    console.log('\n--- Session 2: Reopen and verify persistence ---');
    const db2 = await RvfDatabase.open(storePath);

    const keys2 = await db2.listMetaKeys();
    console.log('listMetaKeys:', keys2);

    const raw1b = await db2.getMeta('e:entry1');
    const obj1b = JSON.parse(Buffer.from(raw1b).toString());
    console.log('getMeta e:entry1:', obj1b.key, '=', obj1b.content);

    const raw2b = await db2.getMeta('e:entry2');
    console.log('getMeta e:entry2 (deleted):', raw2b);

    await db2.close();

    // Cleanup
    fs.rmSync(tmpDir, { recursive: true });

    console.log('\n=== ALL TESTS PASSED ===');
}

main().catch(err => {
    console.error('FAILED:', err);
    process.exit(1);
});
