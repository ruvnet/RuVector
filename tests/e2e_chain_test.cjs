'use strict';
/**
 * E2E chain test: Simulates what memory-initializer.js does through the TS SDK.
 * Tests: create → putMeta → listMetaKeys → getMeta → close → reopen → verify
 *
 * Loads backend.js and database.js from claude-flow-local's patched copies,
 * but the N-API resolves from ruvector's node_modules (same binary).
 */
const path = require('path');
const fs = require('fs');
const os = require('os');

// Override require resolution: when backend.js does require('@ruvector/rvf-node'),
// it should find our local copy (same binary, same adapter layer)
const Module = require('module');
const origResolve = Module._resolveFilename;
Module._resolveFilename = function(request, parent, isMain, options) {
    // Redirect @ruvector/rvf-node to our local copy
    if (request === '@ruvector/rvf-node') {
        return origResolve.call(this, request, parent, isMain, options);
    }
    // Redirect @ruvector/rvf/* to claude-flow-local's patched copies
    if (request.startsWith('./') && parent && parent.filename &&
        parent.filename.includes('claude-flow-local/node_modules/@ruvector/rvf/dist/')) {
        return origResolve.call(this, request, parent, isMain, options);
    }
    return origResolve.call(this, request, parent, isMain, options);
};

// Load the patched database.js from claude-flow-local
const cfBase = '/Users/danielalberttis/Desktop/Projects/claude-flow-local/node_modules/@ruvector/rvf/dist';
const { RvfDatabase } = require(path.join(cfBase, 'database.js'));

async function main() {
    const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'e2e-chain-'));
    const storePath = path.join(tmpDir, 'test.rvf');

    console.log('=== E2E Chain Test: TS SDK → NodeBackend → N-API ===\n');

    // Session 1: Create, write entries, close
    console.log('--- Session 1: Create + Write ---');
    const db = await RvfDatabase.create(storePath, { dimensions: 4, metric: 'cosine' });

    // Simulate what memory-initializer.js does in addToHNSWIndex
    const entry1 = { id: 'abc-123', key: 'test-key', namespace: 'default', content: 'Hello World' };
    await db.putMeta('e:' + entry1.id, Buffer.from(JSON.stringify(entry1)));
    console.log('putMeta e:abc-123 OK');

    const entry2 = { id: 'def-456', key: 'another-key', namespace: 'patterns', content: 'Pattern data' };
    await db.putMeta('e:' + entry2.id, Buffer.from(JSON.stringify(entry2)));
    console.log('putMeta e:def-456 OK');

    // Verify getMeta returns proper Buffer that .toString() works on
    const raw = await db.getMeta('e:abc-123');
    const parsed = JSON.parse(raw.toString());
    console.log('getMeta → toString() → JSON.parse:', parsed.key, '=', parsed.content);
    if (parsed.key !== 'test-key' || parsed.content !== 'Hello World') {
        throw new Error('getMeta round-trip mismatch');
    }

    // listMetaKeys
    const keys = await db.listMetaKeys();
    console.log('listMetaKeys:', keys);
    if (!keys.includes('e:abc-123') || !keys.includes('e:def-456')) {
        throw new Error('listMetaKeys missing entries');
    }

    await db.close();
    console.log('closed\n');

    // Session 2: Reopen, verify persistence
    console.log('--- Session 2: Reopen + Verify ---');
    const db2 = await RvfDatabase.open(storePath);

    // Simulate what memory-initializer.js does in getHNSWIndex (META_SEG load)
    const metaKeys = await db2.listMetaKeys();
    const entryKeys = metaKeys.filter(k => k.startsWith('e:'));
    console.log('Entry keys found:', entryKeys);

    const entries = new Map();
    for (const mk of entryKeys) {
        const rawVal = await db2.getMeta(mk);
        if (!rawVal) continue;
        const obj = JSON.parse(rawVal.toString());
        const entryId = obj.id || mk.slice(2);
        entries.set(entryId, obj);
    }
    console.log('Loaded entries:', entries.size);

    // Verify
    const e1 = entries.get('abc-123');
    const e2 = entries.get('def-456');
    if (!e1 || e1.content !== 'Hello World') throw new Error('entry1 missing or corrupted');
    if (!e2 || e2.content !== 'Pattern data') throw new Error('entry2 missing or corrupted');
    console.log('Entry abc-123:', e1.key, '=', e1.content);
    console.log('Entry def-456:', e2.key, '=', e2.content);

    await db2.close();

    // Cleanup
    fs.rmSync(tmpDir, { recursive: true });

    console.log('\n=== ALL E2E CHAIN TESTS PASSED ===');
}

main().catch(err => { console.error('FAILED:', err); process.exit(1); });
