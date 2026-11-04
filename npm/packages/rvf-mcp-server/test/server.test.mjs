// Tests run against the compiled dist/ (npm test builds first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';
import { RvfMcpServer } from '../dist/index.js';

async function connect(options) {
  const server = new RvfMcpServer(options);
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
