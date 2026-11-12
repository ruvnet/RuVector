// Tests run against the compiled dist/ (npm test builds first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { RvfDatabase } from '@ruvector/rvf';
import { RvfMcpServer } from '../dist/index.js';

const tmpDirs = [];
function tempDataDir() {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-mcp-test-'));
  tmpDirs.push(dir);
  return dir;
}
process.on('exit', () => {
  for (const d of tmpDirs) fs.rmSync(d, { recursive: true, force: true });
});

async function connect(options) {
  const server = new RvfMcpServer({ dataDir: tempDataDir(), ...options });
  const [clientT, serverT] = InMemoryTransport.createLinkedPair();
  await server.connect(serverT);
  const client = new Client({ name: 'test', version: '1.0.0' });
  await client.connect(clientT);
  return { server, client };
}

function text(res) {
  return res.content[0].text;
}

test('prompts advertise named arguments and render', { timeout: 10_000 }, async () => {
  const { client } = await connect();
  const { prompts } = await client.listPrompts();
  const search = prompts.find((p) => p.name === 'rvf-search');
  assert.deepEqual(search.arguments.map((a) => a.name).sort(), ['description', 'storeId']);
  const ingest = prompts.find((p) => p.name === 'rvf-ingest');
  assert.deepEqual(ingest.arguments.map((a) => a.name).sort(), ['data', 'storeId']);

  const got = await client.getPrompt({ name: 'rvf-search', arguments: { storeId: 's1', description: 'cats' } });
  assert.match(got.messages[0].content.text, /"s1".*"cats"/);
  const got2 = await client.getPrompt({ name: 'rvf-ingest', arguments: { storeId: 's2', data: 'hello' } });
  assert.match(got2.messages[0].content.text, /"s2": hello/);
});

test('serverInfo reports the configured version', { timeout: 10_000 }, async () => {
  const { client } = await connect({ version: '9.9.9' });
  assert.equal(client.getServerVersion().version, '9.9.9');
});

test('ingest + query round trip still works', { timeout: 10_000 }, async () => {
  const { client } = await connect();
  const created = JSON.parse(text(await client.callTool({
    name: 'rvf_create_store', arguments: { path: 'mem.rvf', dimensions: 2, metric: 'l2' },
  })));
  const storeId = created.storeId;
  await client.callTool({
    name: 'rvf_ingest',
    arguments: { storeId, entries: [{ id: 'a', vector: [0, 0] }, { id: 'b', vector: [5, 5] }] },
  });
  const q = JSON.parse(text(await client.callTool({
    name: 'rvf_query', arguments: { storeId, vector: [1, 1], k: 1 },
  })));
  assert.equal(q.results[0].id, 'a');
});

test('oversized dimensions, k and batches are rejected', { timeout: 10_000 }, async () => {
  const { client } = await connect();
  const big = await client.callTool({
    name: 'rvf_create_store', arguments: { path: 'x.rvf', dimensions: 1_000_000_000 },
  });
  assert.equal(big.isError, true);

  const { storeId } = JSON.parse(text(await client.callTool({
    name: 'rvf_create_store', arguments: { path: 'y.rvf', dimensions: 1 },
  })));
  const hugeK = await client.callTool({
    name: 'rvf_query', arguments: { storeId, vector: [0], k: 1e12 },
  });
  assert.equal(hugeK.isError, true);

  const entries = Array.from({ length: 10_001 }, (_, i) => ({ id: String(i), vector: [i] }));
  const hugeBatch = await client.callTool({ name: 'rvf_ingest', arguments: { storeId, entries } });
  assert.equal(hugeBatch.isError, true);
});

async function call(client, name, args) {
  const res = await client.callTool({ name, arguments: args });
  const t = text(res);
  return t.startsWith('Error:') ? { error: t } : JSON.parse(t);
}

test('stores persist on disk across a server restart', { timeout: 30_000 }, async () => {
  const dataDir = tempDataDir();
  let { server, client } = await connect({ dataDir });
  const { storeId } = await call(client, 'rvf_create_store', { path: 'nested/p.rvf', dimensions: 3, metric: 'l2' });
  await call(client, 'rvf_ingest', {
    storeId,
    entries: [
      { id: 'red', vector: [1, 0, 0], metadata: { color: 'red', n: 1 } },
      { id: 'green', vector: [0, 1, 0], metadata: { color: 'green', n: 2 } },
      { id: 'blue', vector: [0, 0, 1] },
    ],
  });
  await call(client, 'rvf_delete', { storeId, ids: ['blue'] });
  assert.ok(fs.existsSync(path.join(dataDir, 'nested', 'p.rvf')), 'real .rvf file on disk');
  await server.close();

  ({ server, client } = await connect({ dataDir }));
  const opened = await call(client, 'rvf_open_store', { path: 'nested/p.rvf' });
  assert.equal(opened.dimensions, 3);
  assert.equal(opened.metric, 'l2');
  const sid = opened.storeId;
  const status = await call(client, 'rvf_status', { storeId: sid });
  assert.equal(status.totalVectors, 2);
  const q = await call(client, 'rvf_query', { storeId: sid, vector: [0.9, 0.1, 0], k: 3 });
  assert.deepEqual(q.results.map((r) => r.id), ['red', 'green']);
  const filtered = await call(client, 'rvf_query', { storeId: sid, vector: [1, 0, 0], k: 3, filter: { color: 'green' } });
  assert.deepEqual(filtered.results.map((r) => r.id), ['green']);
  const df = await call(client, 'rvf_delete_filter', { storeId: sid, filter: { n: 1 } });
  assert.equal(df.deleted, 1);
  assert.equal(df.remaining, 1);
  await call(client, 'rvf_close_store', { storeId: sid });

  // Read-only reopen sees the delete and refuses writes.
  const ro = await call(client, 'rvf_open_store', { path: 'nested/p.rvf', readOnly: true });
  const left = await call(client, 'rvf_query', { storeId: ro.storeId, vector: [1, 0, 0], k: 3 });
  assert.deepEqual(left.results.map((r) => r.id), ['green']);
  const denied = await call(client, 'rvf_ingest', { storeId: ro.storeId, entries: [{ id: 'x', vector: [1, 1, 1] }] });
  assert.match(denied.error, /read-only/);
  await server.close();
});

test('an index built outside the server reports the metric it was created with', { timeout: 30_000 }, async () => {
  // A portable .rvf made directly with @ruvector/rvf has no .mcp-meta.jsonl sidecar.
  const dataDir = tempDataDir();
  for (const [file, metric] of [['cos.rvf', 'cosine'], ['dot.rvf', 'dotproduct']]) {
    const db = await RvfDatabase.create(path.join(dataDir, file), { dimensions: 2, metric }, 'node');
    await db.ingestBatch([{ id: 'a', vector: new Float32Array([1, 0]) }]);
    await db.close();
  }
  const { server, client } = await connect({ dataDir });
  const cos = await call(client, 'rvf_open_store', { path: 'cos.rvf', readOnly: true });
  assert.equal(cos.metric, 'cosine');
  const q = await call(client, 'rvf_query', { storeId: cos.storeId, vector: [0, 1], k: 1 });
  assert.equal(q.metric, 'cosine');
  const dot = await call(client, 'rvf_open_store', { path: 'dot.rvf' });
  assert.equal(dot.metric, 'dotproduct');
  await call(client, 'rvf_close_store', { storeId: dot.storeId });
  // A writable open records the metric in the sidecar it creates.
  const header = JSON.parse(fs.readFileSync(path.join(dataDir, 'dot.rvf.mcp-meta.jsonl'), 'utf-8').split('\n')[0]);
  assert.equal(header.metric, 'dotproduct');
  await server.close();
});

test('re-ingesting an id replaces its vector and metadata', { timeout: 10_000 }, async () => {
  const { server, client } = await connect();
  const { storeId } = await call(client, 'rvf_create_store', { path: 'u.rvf', dimensions: 2 });
  await call(client, 'rvf_ingest', { storeId, entries: [{ id: 'a', vector: [0, 0], metadata: { tag: 'old' } }] });
  const res = await call(client, 'rvf_ingest', { storeId, entries: [{ id: 'a', vector: [9, 9] }, { id: 'b', vector: [1] }] });
  assert.deepEqual([res.accepted, res.rejected, res.totalVectors], [1, 1, 1]);
  const q = await call(client, 'rvf_query', { storeId, vector: [9, 9], k: 1 });
  assert.equal(q.results[0].id, 'a');
  assert.equal(q.results[0].distance, 0);
  const tagged = await call(client, 'rvf_query', { storeId, vector: [9, 9], k: 1, filter: { tag: 'old' } });
  assert.deepEqual(tagged.results, []);
  await server.close();
});

test('store paths cannot escape the data dir', { timeout: 10_000 }, async () => {
  const dataDir = tempDataDir();
  const outside = tempDataDir();
  fs.symlinkSync(outside, path.join(dataDir, 'link'));
  fs.symlinkSync(path.join(outside, 'dangling.rvf'), path.join(dataDir, 'dangle.rvf'));
  const { server, client } = await connect({ dataDir });
  const evil = ['../evil.rvf', 'a/../../evil.rvf', '/tmp/evil.rvf', 'link/evil.rvf',
    'link/sub/deeper/evil.rvf', 'dangle.rvf', 'a\0b.rvf', ''];
  for (const p of evil) {
    const created = await call(client, 'rvf_create_store', { path: p, dimensions: 2 });
    assert.ok(created.error, `create accepted ${JSON.stringify(p)}`);
    const opened = await call(client, 'rvf_open_store', { path: p });
    assert.ok(opened.error, `open accepted ${JSON.stringify(p)}`);
  }
  assert.deepEqual(fs.readdirSync(outside), [], 'nothing created outside the data dir');
  const missing = await call(client, 'rvf_open_store', { path: 'no/such/dir/x.rvf' });
  assert.match(missing.error, /does not exist/);
  assert.ok(!fs.existsSync(path.join(dataDir, 'no')), 'open does not create directories');
  assert.ok(!fs.existsSync(path.join(dataDir, '..', 'evil.rvf')));
  await server.close();
});

test('creating over an existing store is refused', { timeout: 10_000 }, async () => {
  const { server, client } = await connect();
  const first = await call(client, 'rvf_create_store', { path: 'dup.rvf', dimensions: 2 });
  await call(client, 'rvf_close_store', { storeId: first.storeId });
  const second = await call(client, 'rvf_create_store', { path: 'dup.rvf', dimensions: 2 });
  assert.ok(second.error);
  await server.close();
});

test("many single-vector ingests are auto-compacted so the file stays bounded", { timeout: 120_000 }, async () => {
  const dataDir = tempDataDir();
  const { client } = await connect({ dataDir, autoCompact: { minBytes: 256 * 1024, ratio: 8 } });
  const { storeId } = JSON.parse(text(await client.callTool({
    name: "rvf_create_store", arguments: { path: "grow.rvf", dimensions: 16, metric: "l2" },
  })));
  let compactions = 0;
  for (let i = 0; i < 300; i++) {
    const vector = Array.from({ length: 16 }, (_, d) => (d === 0 ? i : ((i * 31 + d * 7) % 97) / 97));
    const r = JSON.parse(text(await client.callTool({
      name: "rvf_ingest", arguments: { storeId, entries: [{ id: `v${i}`, vector }] },
    })));
    if (r.autoCompacted) compactions++;
  }
  const st = JSON.parse(text(await client.callTool({ name: "rvf_status", arguments: { storeId } })));
  assert.equal(st.totalVectors, 300);
  assert.ok(compactions > 0, "expected at least one auto-compaction");
  const size = fs.statSync(path.join(dataDir, "grow.rvf")).size;
  assert.ok(size < 2 * 1024 * 1024, `file grew to ${size} bytes`);
  const q = JSON.parse(text(await client.callTool({
    name: "rvf_query", arguments: { storeId, vector: Array.from({ length: 16 }, (_, d) => (d === 0 ? 299 : ((299 * 31 + d * 7) % 97) / 97)), k: 1 },
  })));
  assert.equal(q.results[0].id, "v299");
});

test('concurrent rvf_query requests all resolve with correct results', { timeout: 60_000 }, async () => {
  const { client } = await connect();
  const { storeId } = JSON.parse(text(await client.callTool({
    name: 'rvf_create_store', arguments: { path: 'conc.rvf', dimensions: 8, metric: 'l2' },
  })));
  const vec = (i) => Array.from({ length: 8 }, (_, d) => (d === 0 ? i : Math.sin(i * 0.37 + d)));
  const entries = Array.from({ length: 2000 }, (_, i) => ({ id: `c${i}`, vector: vec(i) }));
  await client.callTool({ name: 'rvf_ingest', arguments: { storeId, entries } });
  const ids = Array.from({ length: 64 }, (_, i) => i * 31);
  const replies = await Promise.all(ids.map((i) => client.callTool({
    name: 'rvf_query', arguments: { storeId, vector: vec(i), k: 1 },
  })));
  replies.forEach((r, j) => assert.equal(JSON.parse(text(r)).results[0].id, `c${ids[j]}`));
});

// A portable store is <store>.rvf plus sidecars; the string-id map is the one
// most easily left behind when copying it.
async function seedMovedStore() {
  const src = tempDataDir();
  const dst = tempDataDir();
  const { server, client } = await connect({ dataDir: src });
  const { storeId } = await call(client, 'rvf_create_store', { path: 'kb.rvf', dimensions: 3, metric: 'cosine' });
  await call(client, 'rvf_ingest', {
    storeId,
    entries: [
      { id: 'hnsw-tuning', vector: [1, 0, 0], metadata: { topic: 'search' } },
      { id: 'filtered-knn', vector: [0.9, 0.1, 0], metadata: { topic: 'search' } },
      { id: 'pasta', vector: [0, 0, 1], metadata: { topic: 'food' } },
      { id: 'stale', vector: [0, 1, 0] },
    ],
  });
  // Embedded ids must survive a delete + compaction rewrite of the file.
  await call(client, 'rvf_delete', { storeId, ids: ['stale'] });
  await call(client, 'rvf_compact', { storeId });
  await server.close();
  for (const f of ['kb.rvf', 'kb.rvf.mcp-meta.jsonl']) fs.copyFileSync(path.join(src, f), path.join(dst, f));
  return dst;
}

test('a store copied without its .idmap.json still answers by string id', { timeout: 30_000 }, async () => {
  const { server, client } = await connect({ dataDir: await seedMovedStore() });
  const { storeId } = await call(client, 'rvf_open_store', { path: 'kb.rvf', readOnly: true });
  const filtered = await call(client, 'rvf_query', { storeId, vector: [1, 0, 0], k: 2, filter: { topic: 'search' } });
  assert.deepEqual(filtered.results.map((r) => r.id), ['hnsw-tuning', 'filtered-knn']);
  const plain = await call(client, 'rvf_query', { storeId, vector: [0, 0, 1], k: 1 });
  assert.deepEqual(plain.results.map((r) => r.id), ['pasta']);
  await server.close();
});

test('a writable open without the .idmap.json is refused', { timeout: 30_000 }, async () => {
  const { server, client } = await connect({ dataDir: await seedMovedStore() });
  const opened = await call(client, 'rvf_open_store', { path: 'kb.rvf' });
  assert.match(opened.error, /kb\.rvf\.idmap\.json/);
  await server.close();
});

test('a store without embedded ids or its .idmap.json fails loudly instead of returning []', { timeout: 30_000 }, async () => {
  // Shape of a store written by rvf-mcp-server <= 0.2.3: ids only in the id map.
  const dataDir = tempDataDir();
  const file = path.join(dataDir, 'old.rvf');
  const db = await RvfDatabase.create(file, { dimensions: 2, metric: 'l2' }, 'node');
  await db.ingestBatch([{ id: 'a', vector: new Float32Array([0, 0]) }, { id: 'b', vector: new Float32Array([5, 5]) }]);
  await db.close();
  fs.writeFileSync(`${file}.mcp-meta.jsonl`, '{"rvfMcpMeta":1,"dimensions":2,"metric":"l2"}\n["s","a",{"tag":"x"}]\n');
  fs.rmSync(`${file}.idmap.json`);
  const { server, client } = await connect({ dataDir });
  const { storeId } = await call(client, 'rvf_open_store', { path: 'old.rvf', readOnly: true });
  const filtered = await call(client, 'rvf_query', { storeId, vector: [0, 0], k: 1, filter: { tag: 'x' } });
  assert.match(filtered.error, /old\.rvf\.idmap\.json/);
  const plain = await call(client, 'rvf_query', { storeId, vector: [0, 0], k: 1 });
  assert.match(plain.error, /old\.rvf\.idmap\.json/);
  await server.close();
});
