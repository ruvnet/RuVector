'use strict';
/**
 * Tests for issue #704: RVF SDK silently drops metadata; filter
 * serialization omits the native parser's required `valueType`.
 *
 * Rather than silently accepting and dropping `RvfIngestEntry.metadata`
 * (the original bug — metadata vanished with no error), NodeBackend and
 * WasmBackend now reject it loudly until the SDK has a real field-name to
 * native-fieldId mapping design. Separately, query() filter serialization
 * now infers and includes the `valueType` the native parser requires.
 */

const assert = require('assert');
const fs = require('fs');
const path = require('path');
const os = require('os');

class MockNativeHandle {
  constructor() {
    this.vectors = new Map();
    this.lastQueryOptions = null;
  }
  ingestBatch(flat, ids) {
    const dim = flat.length / ids.length;
    for (let i = 0; i < ids.length; i++) {
      this.vectors.set(ids[i], flat.slice(i * dim, (i + 1) * dim));
    }
    return { accepted: ids.length, rejected: 0, epoch: 1 };
  }
  query(vector, k, options) {
    this.lastQueryOptions = options;
    return [];
  }
  close() {}
  dimension() { return 4; }
}

const { NodeBackend, WasmBackend } = require('../dist/backend');
const { RvfErrorCode } = require('../dist/errors');

async function createNodeTestBackend(tmpDir) {
  const storePath = path.join(tmpDir, 'test.rvf');
  const backend = new NodeBackend();
  backend['native'] = { create: () => new MockNativeHandle(), open: () => new MockNativeHandle() };
  backend['handle'] = new MockNativeHandle();
  backend['storePath'] = storePath;
  backend['idToLabel'] = new Map();
  backend['labelToId'] = new Map();
  backend['nextLabel'] = 1;
  return backend;
}

let passed = 0, failed = 0;

async function test(name, fn) {
  const tmpDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-metadata-test-'));
  try {
    await fn(tmpDir);
    console.log(`  PASS  ${name}`);
    passed++;
  } catch (err) {
    console.log(`  FAIL  ${name}: ${err.message}`);
    failed++;
  } finally {
    try { fs.rmSync(tmpDir, { recursive: true, force: true }); } catch {}
  }
}

(async () => {
  console.log('--- Metadata safety + filter serialization (Issue #704) ---');

  await test('ingestBatch without metadata still works', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    const result = await backend.ingestBatch([
      { id: 'a', vector: new Float32Array([1, 0, 0, 0]) },
    ]);
    assert.strictEqual(result.accepted, 1);
  });

  await test('NodeBackend rejects (not silently drops) non-empty metadata', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    await assert.rejects(
      () => backend.ingestBatch([
        { id: 'a', vector: new Float32Array([1, 0, 0, 0]), metadata: { color: 'red' } },
      ]),
      (err) => {
        assert.strictEqual(err.code, RvfErrorCode.MetadataNotSupported);
        return true;
      },
    );
    // Confirm it actually rejected before ingest, not after silently accepting.
    assert.strictEqual(backend['handle'].vectors.size, 0);
  });

  await test('NodeBackend accepts entries with an empty metadata object', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    const result = await backend.ingestBatch([
      { id: 'a', vector: new Float32Array([1, 0, 0, 0]), metadata: {} },
    ]);
    assert.strictEqual(result.accepted, 1);
  });

  await test('WasmBackend rejects non-empty metadata the same way', async () => {
    const backend = new WasmBackend();
    // Force a handle without loading the real wasm module — ingestBatch's
    // metadata check runs before any wasm calls.
    backend['handle'] = 1;
    backend['dim'] = 4;
    await assert.rejects(
      () => backend.ingestBatch([
        { id: 'a', vector: new Float32Array([1, 0, 0, 0]), metadata: { color: 'red' } },
      ]),
      (err) => {
        assert.strictEqual(err.code, RvfErrorCode.MetadataNotSupported);
        return true;
      },
    );
  });

  await test('query() filter serialization infers valueType for strings', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'eq', fieldId: 0, value: 'red' },
    });
    const sent = JSON.parse(backend['handle'].lastQueryOptions.filter);
    assert.deepStrictEqual(sent, { op: 'eq', fieldId: 0, valueType: 'string', value: 'red' });
  });

  await test('query() filter serialization infers u64/i64/f64 for numbers', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);

    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'ge', fieldId: 1, value: 42 },
    });
    assert.deepStrictEqual(
      JSON.parse(backend['handle'].lastQueryOptions.filter),
      { op: 'ge', fieldId: 1, valueType: 'u64', value: '42' },
    );

    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'lt', fieldId: 1, value: -5 },
    });
    assert.deepStrictEqual(
      JSON.parse(backend['handle'].lastQueryOptions.filter),
      { op: 'lt', fieldId: 1, valueType: 'i64', value: '-5' },
    );

    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'lt', fieldId: 1, value: 3.14 },
    });
    assert.deepStrictEqual(
      JSON.parse(backend['handle'].lastQueryOptions.filter),
      { op: 'lt', fieldId: 1, valueType: 'f64', value: '3.14' },
    );
  });

  await test('query() filter serialization infers bool and recurses through and/or/not', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: {
        op: 'and',
        exprs: [
          { op: 'eq', fieldId: 0, value: 'tech' },
          { op: 'not', expr: { op: 'eq', fieldId: 2, value: true } },
        ],
      },
    });
    const sent = JSON.parse(backend['handle'].lastQueryOptions.filter);
    assert.deepStrictEqual(sent, {
      op: 'and',
      children: [
        { op: 'eq', fieldId: 0, valueType: 'string', value: 'tech' },
        { op: 'not', child: { op: 'eq', fieldId: 2, valueType: 'bool', value: 'true' } },
      ],
    });
  });

  await test('query() filter serialization handles "in" and "range"', async (tmp) => {
    const backend = await createNodeTestBackend(tmp);
    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'in', fieldId: 0, values: [1, 2, 3] },
    });
    assert.deepStrictEqual(
      JSON.parse(backend['handle'].lastQueryOptions.filter),
      { op: 'in', fieldId: 0, valueType: 'u64', values: ['1', '2', '3'] },
    );

    await backend.query(new Float32Array([1, 0, 0, 0]), 5, {
      filter: { op: 'range', fieldId: 1, low: 10, high: 50 },
    });
    assert.deepStrictEqual(
      JSON.parse(backend['handle'].lastQueryOptions.filter),
      { op: 'range', fieldId: 1, valueType: 'u64', low: '10', high: '50' },
    );
  });

  console.log(`\n${'='.repeat(60)}`);
  console.log(`Results: ${passed} passed, ${failed} failed, ${passed + failed} total`);
  console.log('='.repeat(60));
  process.exit(failed > 0 ? 1 : 0);
})();
