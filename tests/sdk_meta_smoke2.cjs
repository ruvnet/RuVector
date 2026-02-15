'use strict';
const { createRequire } = require('module');
const path = require('path');
const fs = require('fs');
const os = require('os');

// Load @ruvector/rvf from claude-flow-local's node_modules
const cfRequire = createRequire('/Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@ruvector/rvf/dist/database.js');
const rvfNodePath = '/Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@ruvector/rvf-node';

async function main() {
    // Directly load the patched files
    const backend = require('/Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@ruvector/rvf/dist/backend.js');
    const database = require('/Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@ruvector/rvf/dist/database.js');
    const { RvfDatabase } = database;

    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'sdk-meta-'));
    const storePath = path.join(tmpDir, 'test.rvf');

    console.log('--- Session 1: Create, write, verify META_SEG ---');
    const db = await RvfDatabase.create(storePath, { dimensions: 4, metric: 'cosine' });

    await db.putMeta('e:entry1', Buffer.from(JSON.stringify({ id: 'entry1', key: 'hello', namespace: 'default', content: 'world' })));
    await db.putMeta('e:entry2', Buffer.from(JSON.stringify({ id: 'entry2', key: 'foo', namespace: 'test', content: 'bar' })));
    console.log('putMeta: 2 entries written');

    const raw1 = await db.getMeta('e:entry1');
    console.log('getMeta e:entry1:', JSON.parse(Buffer.from(raw1).toString()).content);

    const keys = await db.listMetaKeys();
    console.log('listMetaKeys:', keys);

    const deleted = await db.deleteMeta('e:entry2');
    console.log('deleteMeta e:entry2:', deleted);

    await db.close();

    console.log('\n--- Session 2: Reopen and verify persistence ---');
    const db2 = await RvfDatabase.open(storePath);
    const keys2 = await db2.listMetaKeys();
    console.log('listMetaKeys:', keys2);

    const raw1b = await db2.getMeta('e:entry1');
    console.log('getMeta e:entry1:', JSON.parse(Buffer.from(raw1b).toString()).content);

    const raw2b = await db2.getMeta('e:entry2');
    console.log('getMeta e:entry2 (deleted):', raw2b);

    await db2.close();

    fs.rmSync(tmpDir, { recursive: true });
    console.log('\n=== ALL TESTS PASSED ===');
}

main().catch(err => { console.error('FAILED:', err); process.exit(1); });
