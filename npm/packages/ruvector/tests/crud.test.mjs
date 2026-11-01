/**
 * Full CRUD, end to end: the VectorDB library, the CLI (vector stores, RVF,
 * hook memories) and the MCP server over stdio, including the rejection paths
 * (outside-root paths, prototype-polluting metadata, bad vectors, oversize
 * batches) and the upsert ghost regression in the native HNSW index.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawn } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const require = createRequire(import.meta.url);
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
const SERVER = path.join(PKG, 'bin', 'mcp-server.js');
const { VectorDB } = require('../dist/index.js');
const memoryStore = require('../bin/memory-store.js');

function tmpdir(t, prefix) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), prefix));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  return dir;
}

function cli(cwd, args, env = {}) {
  try {
    const out = execFileSync(process.execPath, [CLI, ...args], {
      cwd, env: { ...process.env, HOME: cwd, NO_COLOR: '1', ...env }, encoding: 'utf8', stdio: ['ignore', 'pipe', 'pipe'],
    });
    return { code: 0, out };
  } catch (e) {
    return { code: e.status ?? 1, out: `${e.stdout || ''}${e.stderr || ''}` };
  }
}

/** Sequential MCP client: one request in flight at a time, like a real agent turn. */
async function mcpSession(cwd, env = {}) {
  const p = spawn(process.execPath, [SERVER], {
    cwd, stdio: ['pipe', 'pipe', 'pipe'], env: { ...process.env, HOME: cwd, ...env },
  });
  let buf = '';
  let nextId = 0;
  const waiting = new Map();
  p.stdout.on('data', (d) => {
    buf += d;
    let i;
    while ((i = buf.indexOf('\n')) >= 0) {
      const line = buf.slice(0, i);
      buf = buf.slice(i + 1);
      let m;
      try { m = JSON.parse(line); } catch { continue; }
      if (waiting.has(m.id)) { waiting.get(m.id)(m); waiting.delete(m.id); }
    }
  });
  const request = (method, params) => new Promise((resolve, reject) => {
    const id = nextId++;
    const timer = setTimeout(() => reject(new Error(`MCP ${method} timed out`)), 60000);
    waiting.set(id, (m) => { clearTimeout(timer); resolve(m); });
    p.stdin.write(JSON.stringify({ jsonrpc: '2.0', id, method, params }) + '\n');
  });
  await request('initialize', { protocolVersion: '2024-11-05', capabilities: {}, clientInfo: { name: 'crud-test', version: '1' } });
  p.stdin.write(JSON.stringify({ jsonrpc: '2.0', method: 'notifications/initialized' }) + '\n');
  return {
    async call(name, args) {
      const m = await request('tools/call', { name, arguments: args });
      const text = m.result?.content?.[0]?.text ?? JSON.stringify(m.error);
      let body;
      try { body = JSON.parse(text); } catch { body = { raw: text }; }
      return { isError: !!m.result?.isError, body };
    },
    async tools() { return (await request('tools/list', {})).result.tools.map(t => t.name); },
    close() { p.kill(); },
  };
}

// ── Library ──────────────────────────────────────────────────────────────────

test('VectorDB: re-inserting an id replaces it (no stale HNSW ghost)', async (t) => {
  const dir = tmpdir(t, 'ruvector-crud-lib-');
  const db = new VectorDB({ dimensions: 3, storagePath: path.join(dir, 'a.db') });
  for (let i = 0; i < 40; i++) await db.insert({ id: `n${i}`, vector: [Math.random(), Math.random(), Math.random()] });
  await db.insert({ id: 'a', vector: [1, 0, 0] });
  await db.insert({ id: 'a', vector: [0, 0, 1], metadata: { v: 2 } });
  await db.insertBatch([{ id: 'b', vector: [1, 0, 0] }, { id: 'b', vector: [0, 1, 0] }]);
  const hits = await db.search({ vector: [1, 0, 0], k: 50 });
  assert.equal(hits.filter(h => h.id === 'a').length, 1, 'id a returned more than once');
  assert.equal(hits.filter(h => h.id === 'b').length, 1, 'id b returned more than once');
  assert.equal(await db.len(), 42);
  assert.deepEqual(Array.from((await db.get('b')).vector), [0, 1, 0], 'last entry in a batch must win');
});

test('VectorDB: update and deleteMany', async (t) => {
  const dir = tmpdir(t, 'ruvector-crud-lib-');
  const db = new VectorDB({ dimensions: 3, storagePath: path.join(dir, 'b.db') });
  await db.insert({ id: 'x', vector: [1, 0, 0], metadata: { a: 1 } });
  assert.equal(await db.update('missing', { metadata: { a: 2 } }), null, 'update must not create');
  assert.equal(await db.get('missing'), null);
  await db.update('x', { metadata: { b: 2 }, mergeMetadata: true });
  assert.deepEqual((await db.get('x')).metadata, { a: 1, b: 2 });
  await db.update('x', { vector: [0, 1, 0] });
  const rec = await db.get('x');
  assert.deepEqual(Array.from(rec.vector), [0, 1, 0]);
  assert.deepEqual(rec.metadata, { a: 1, b: 2 }, 'vector-only update keeps metadata');
  const hits = await db.search({ vector: [1, 0, 0], k: 5 });
  assert.equal(hits.filter(h => h.id === 'x').length, 1);
  assert.deepEqual(await db.deleteMany(['x', 'nope', 'x']), { deleted: ['x'], missing: ['nope'] });
  assert.equal(await db.len(), 0);
});

// ── CLI ──────────────────────────────────────────────────────────────────────

test('CLI: vector create/insert/get/update/delete round trip', (t) => {
  const dir = tmpdir(t, 'ruvector-crud-cli-');
  fs.writeFileSync(path.join(dir, 'v.json'), JSON.stringify([
    { id: 'a', vector: [1, 0, 0], metadata: { k: 1 } }, { id: 'b', vector: [0, 1, 0] },
  ]));
  assert.equal(cli(dir, ['create', 'db', '-d', '3']).code, 0);
  assert.equal(cli(dir, ['insert', 'db', 'v.json']).code, 0);
  const got = JSON.parse(cli(dir, ['get', 'db', 'a']).out);
  assert.deepEqual(got, { id: 'a', vector: [1, 0, 0], metadata: { k: 1 } });
  assert.equal(cli(dir, ['update', 'db', 'a', '-v', '[0,0,1]', '-m', '{"z":2}', '--merge']).code, 0);
  const after = JSON.parse(cli(dir, ['get', 'db', 'a']).out);
  assert.deepEqual(after.vector, [0, 0, 1]);
  assert.deepEqual(after.metadata, { k: 1, z: 2 });
  assert.equal(cli(dir, ['update', 'db', 'nope', '-m', '{}']).code, 1, 'update must not create');
  const del = cli(dir, ['delete', 'db', 'a', 'missing']);
  assert.equal(del.code, 0);
  assert.match(del.out, /Deleted 1/);
  assert.match(del.out, /Not found: missing/);
  assert.equal(cli(dir, ['get', 'db', 'a']).code, 1);
  // Rejections: wrong dimension, non-finite, prototype keys, store not created by accident.
  assert.match(cli(dir, ['update', 'db', 'b', '-v', '[1,2]']).out, /dimension 2, store expects 3/);
  assert.match(cli(dir, ['update', 'db', 'b', '-v', '[1,"x",2]']).out, /not a finite number/);
  assert.match(cli(dir, ['update', 'db', 'b', '-m', '{"a":{"__proto__":{"p":1}}}']).out, /not allowed/);
  assert.equal(cli(dir, ['get', 'nosuch.db', 'a']).code, 1);
  assert.ok(!fs.existsSync(path.join(dir, 'nosuch.db')), 'get must not create a store');
});

test('CLI: rvf delete', (t) => {
  let rvfAvailable = true;
  try { require('../dist/core/rvf-wrapper.js').isRvfAvailable() || (rvfAvailable = false); } catch { rvfAvailable = false; }
  if (!rvfAvailable) return t.skip('@ruvector/rvf not installed');
  const dir = tmpdir(t, 'ruvector-crud-rvf-');
  fs.writeFileSync(path.join(dir, 'in.json'), JSON.stringify([{ id: '1', vector: [1, 0, 0] }, { id: '2', vector: [0, 1, 0] }]));
  assert.equal(cli(dir, ['rvf', 'create', 's.rvf', '-d', '3']).code, 0);
  assert.equal(cli(dir, ['rvf', 'ingest', 's.rvf', '-i', 'in.json']).code, 0);
  assert.match(cli(dir, ['rvf', 'delete', 's.rvf', '--ids', '1']).out, /Deleted 1/);
  assert.doesNotMatch(cli(dir, ['rvf', 'query', 's.rvf', '-v', '1,0,0', '-k', '5']).out, /id=1\b/);
});

test('CLI: hook memory remember/recall/get/list/update/forget', (t) => {
  const dir = tmpdir(t, 'ruvector-crud-mem-');
  fs.writeFileSync(path.join(dir, 'package.json'), '{}');
  const env = { RUVECTOR_EMBEDDER: 'hash' };
  const r1 = JSON.parse(cli(dir, ['hooks', 'remember', '-t', 'project', 'postgres connection pooling'], env).out);
  const r2 = JSON.parse(cli(dir, ['hooks', 'remember', '-t', 'note', 'redis cache warmup'], env).out);
  assert.ok(r1.id && r2.id && r1.id !== r2.id, 'ids must be unique');
  const recall = JSON.parse(cli(dir, ['hooks', 'recall', 'postgres'], env).out);
  assert.ok(recall.results.some(r => r.id === r1.id), 'recall must expose ids');
  const got = JSON.parse(cli(dir, ['hooks', 'memory-get', r1.id], env).out);
  assert.equal(got.memory.content, 'postgres connection pooling');
  assert.equal(got.memory.type, 'project');
  const upd = JSON.parse(cli(dir, ['hooks', 'memory-update', r1.id, '-c', 'mysql replication lag', '-t', 'ops'], env).out);
  assert.equal(upd.success, true);
  const store = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  const stored = store.memories.find(m => m.id === r1.id);
  assert.equal(stored.content, 'mysql replication lag');
  assert.equal(stored.memory_type, 'ops');
  assert.equal(stored.embedding.length > 0, true);
  const list = JSON.parse(cli(dir, ['hooks', 'memory-list', '-t', 'ops'], env).out);
  assert.deepEqual(list.items.map(i => i.id), [r1.id]);
  assert.match(cli(dir, ['hooks', 'memory-update', r1.id, '-m', '{"__proto__":{"x":1}}'], env).out, /not allowed/);
  const forget = JSON.parse(cli(dir, ['hooks', 'forget', r1.id, 'mem_missing'], env).out);
  assert.deepEqual(forget.deleted, [r1.id]);
  assert.deepEqual(forget.missing, ['mem_missing']);
  assert.equal(cli(dir, ['hooks', 'memory-get', r1.id], env).code, 1);
  assert.equal(JSON.parse(cli(dir, ['hooks', 'memory-list'], env).out).total, 1);
});

test('memory ids: legacy entries get deterministic, unique ids', () => {
  const legacy = () => ({ memories: [
    { content: 'a', type: 'general', created: '2026-01-01T00:00:00Z' },
    { id: 'mem_1', content: 'b' },
    { id: 'mem_1', content: 'c' },
  ] });
  const d1 = legacy();
  const d2 = legacy();
  assert.equal(memoryStore.ensureMemoryIds(d1), 2);
  memoryStore.ensureMemoryIds(d2);
  assert.deepEqual(d1.memories.map(m => m.id), d2.memories.map(m => m.id), 'two processes must derive the same ids');
  assert.equal(new Set(d1.memories.map(m => m.id)).size, 3);
  assert.equal(memoryStore.ensureMemoryIds(d1), 0, 'idempotent');
});

// ── MCP over stdio ───────────────────────────────────────────────────────────

test('MCP: vectordb_* CRUD and rejections', async (t) => {
  const dir = tmpdir(t, 'ruvector-crud-mcp-');
  const outside = tmpdir(t, 'ruvector-crud-outside-');
  const s = await mcpSession(dir);
  t.after(() => s.close());
  const names = await s.tools();
  for (const n of ['vectordb_create', 'vectordb_insert', 'vectordb_get', 'vectordb_update', 'vectordb_delete', 'vectordb_search', 'vectordb_stats',
    'hooks_memory_get', 'hooks_memory_list', 'hooks_memory_update', 'hooks_forget']) {
    assert.ok(names.includes(n), `missing tool ${n}`);
  }

  assert.equal((await s.call('vectordb_create', { path: 'v.db', dimension: 3 })).isError, false);
  let r = await s.call('vectordb_insert', { path: 'v.db', entries: [{ id: 'a', vector: [1, 0, 0], metadata: { t: 'x' } }, { id: 'b', vector: [0, 1, 0] }] });
  assert.equal(r.body.total, 2);
  r = await s.call('vectordb_get', { path: 'v.db', id: 'a' });
  assert.deepEqual(r.body.record, { id: 'a', metadata: { t: 'x' }, vector: [1, 0, 0] });
  r = await s.call('vectordb_update', { path: 'v.db', id: 'a', vector: [0, 0, 1], metadata: { u: 1 }, merge_metadata: true });
  assert.equal(r.isError, false);
  r = await s.call('vectordb_search', { path: 'v.db', vector: [0, 0, 1], k: 5 });
  assert.equal(r.body.results.filter(x => x.id === 'a').length, 1);
  assert.deepEqual(r.body.results.find(x => x.id === 'a').metadata, { t: 'x', u: 1 });
  r = await s.call('vectordb_delete', { path: 'v.db', ids: ['a', 'zz'] });
  assert.deepEqual([r.body.deleted, r.body.missing, r.body.total], [['a'], ['zz'], 1]);
  r = await s.call('vectordb_stats', { path: 'v.db' });
  assert.equal(r.body.count, 1);
  assert.equal(r.body.dimension, 3);

  // Rejections
  const rejected = async (name, args, pattern) => {
    const res = await s.call(name, args);
    assert.equal(res.isError, true, `${name} should be rejected`);
    if (pattern) assert.match(res.body.error, pattern);
  };
  await rejected('vectordb_create', { path: path.join(outside, 'x.db'), dimension: 3 }, /outside the working directory/);
  await rejected('vectordb_get', { path: '../escape.db', id: 'a' }, /outside the working directory/);
  await rejected('vectordb_update', { path: 'v.db', id: 'b', metadata: { n: { __proto__: null, constructor: { prototype: { polluted: 1 } } } } }, /not allowed/);
  await rejected('vectordb_insert', { path: 'v.db', entries: [{ id: 'c', vector: [1, 2] }] }, /expects 3/);
  await rejected('vectordb_insert', { path: 'v.db', entries: [{ id: 'c', vector: [1, null, 2] }] }, /finite/);
  await rejected('vectordb_insert', { path: 'v.db', entries: Array.from({ length: 10001 }, (_, i) => ({ id: `e${i}`, vector: [1, 0, 0] })) }, /exceeds 10000/);
  await rejected('vectordb_get', { path: 'v.db', id: 'a\u0000b' }, /control characters/);
  await rejected('vectordb_create', { path: 'v.db', dimension: 3 }, /Already exists/);
  await rejected('vectordb_update', { path: 'v.db', id: 'missing', metadata: {} }, /not found/);
  assert.equal(({}).polluted, undefined, 'prototype polluted');
  assert.ok(!fs.existsSync(path.join(dir, 'ruvector.db')), 'no default ./ruvector.db may be created');
  assert.ok(!fs.existsSync(path.join(outside, 'x.db')));
});

test('MCP readonly profile exposes memory reads but no CRUD writes', async (t) => {
  const dir = tmpdir(t, 'ruvector-crud-ro-');
  const s = await mcpSession(dir, { RUVECTOR_MCP_PROFILE: 'readonly' });
  t.after(() => s.close());
  const names = await s.tools();
  assert.ok(names.includes('hooks_memory_get') && names.includes('hooks_memory_list'));
  for (const w of ['hooks_memory_update', 'hooks_forget', 'vectordb_create', 'vectordb_insert', 'vectordb_update', 'vectordb_delete']) {
    assert.ok(!names.includes(w), `readonly profile exposes write tool ${w}`);
  }
  assert.equal((await s.call('hooks_forget', { ids: ['mem_x'] })).isError, true, 'denied tool must not be callable');
});

test('MCP: hook memory CRUD is coherent within one server process', async (t) => {
  const dir = tmpdir(t, 'ruvector-crud-mcpmem-');
  fs.writeFileSync(path.join(dir, 'package.json'), '{}');
  const s = await mcpSession(dir, { RUVECTOR_EMBEDDER: 'hash' });
  t.after(() => s.close());
  const a = (await s.call('hooks_remember', { content: 'kubernetes pod eviction', type: 'ops' })).body;
  const b = (await s.call('hooks_remember', { content: 'terraform state locking', type: 'ops' })).body;
  assert.ok(a.id && b.id && a.id !== b.id);
  let recall = (await s.call('hooks_recall', { query: 'kubernetes pod eviction', top_k: 5 })).body;
  assert.ok(recall.results.some(r => r.id === a.id), 'recall must expose ids');

  let r = await s.call('hooks_memory_update', { id: a.id, content: 'nginx upstream timeout', metadata: { src: 'test' } });
  assert.equal(r.isError, false);
  assert.equal(r.body.memory.content, 'nginx upstream timeout');
  recall = (await s.call('hooks_recall', { query: 'kubernetes pod eviction', top_k: 5 })).body;
  assert.ok(!recall.results.some(x => x.content === 'kubernetes pod eviction'), 'stale content served after update');

  r = await s.call('hooks_forget', { ids: [b.id] });
  assert.deepEqual(r.body.deleted, [b.id]);
  recall = (await s.call('hooks_recall', { query: 'terraform state locking', top_k: 5 })).body;
  assert.ok(!recall.results.some(x => x.id === b.id), 'forgotten memory still recalled in the same process');

  r = await s.call('hooks_memory_list', {});
  assert.deepEqual(r.body.items.map(i => i.id), [a.id]);
  assert.equal((await s.call('hooks_memory_get', { id: b.id })).isError, true);
  assert.equal((await s.call('hooks_memory_update', { id: a.id, metadata: { constructor: { prototype: { x: 1 } } } })).isError, true);
  assert.equal((await s.call('hooks_forget', { ids: [] })).isError, true);

  // A CLI process edits the store: the running server must pick it up.
  execFileSync(process.execPath, [CLI, 'hooks', 'forget', a.id], { cwd: dir, env: { ...process.env, HOME: dir, RUVECTOR_EMBEDDER: 'hash' }, stdio: 'ignore' });
  r = await s.call('hooks_memory_list', {});
  assert.equal(r.body.total, 0);
  recall = (await s.call('hooks_recall', { query: 'nginx upstream timeout', top_k: 5 })).body;
  assert.equal(recall.results.length, 0, 'memory deleted by another process still recalled');
});
