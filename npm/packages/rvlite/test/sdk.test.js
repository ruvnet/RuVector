// SDK smoke tests against the built dist/ (run `npm run build` first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtemp, readFile, rm, writeFile } from 'node:fs/promises';
import { createRequire } from 'node:module';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

const require = createRequire(import.meta.url);

async function exercise(RvLite) {
  const db = new RvLite({ dimensions: 3, distanceMetric: 'euclidean' });
  const id = await db.insert([1, 2, 3], { tag: 'a' });
  assert.equal(typeof id, 'string');
  const hits = await db.search([1, 2, 3], 1);
  assert.equal(hits.length, 1);
  assert.equal(hits[0].id, id);
  assert.equal(await db.len(), 1);

  await db.sql('CREATE TABLE docs (id TEXT, embedding VECTOR(3))');
  await db.sql("INSERT INTO docs (id, embedding) VALUES ('d1', [-1.0, 0.5, 0.25])");
  const rows = await db.sql('SELECT * FROM docs ORDER BY embedding <-> [-1.0, 0.5, 0.25] LIMIT 1');
  assert.equal(rows.rows.length, 1);

  await db.cypher("CREATE (n:Person {name: 'Alice'})");
  assert.equal((await db.cypherStats()).node_count, 1);

  await db.addTriple('<http://ex.org/a>', '<http://ex.org/knows>', '<http://ex.org/b>');
  const res = await db.sparql('SELECT ?o WHERE { <http://ex.org/a> <http://ex.org/knows> ?o }');
  assert.equal(res.bindings.length, 1);

  // Malformed input must surface as an error, not trap the wasm instance.
  await assert.rejects(db.cypher('$x'));
  await assert.rejects(db.sql(`SELECT * FROM docs WHERE ${'NOT '.repeat(10000)}id = 'a'`));
  await assert.rejects(db.sparql(`SELECT ?s WHERE ${'{ '.repeat(10000)}?s ?p ?o${' }'.repeat(10000)}`));
  await assert.rejects(db.addTriple('<http://s>', '<http://p>', '"unterminated'));
  assert.equal(await db.len(), 1, 'instance still usable after rejected queries');
}

test('ESM build works in Node', async () => {
  const { RvLite } = await import('../dist/index.mjs');
  await exercise(RvLite);
});

test('CommonJS build works in Node', async () => {
  const { RvLite } = require('../dist/index.cjs');
  await exercise(RvLite);
});

test('JSON RVF round trip preserves metadata and graph properties', async (t) => {
  const { RvLite } = await import('../dist/index.mjs');
  const dir = await mkdtemp(join(tmpdir(), 'rvlite-sdk-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const rvfPath = join(dir, 'data', 'db.rvf');
  const dimensions = 3;
  const db = new RvLite({ dimensions });
  const metadata = { tag: 'x', nested: { items: [{ label: 'y' }, false], empty: null } };
  await db.insertWithId('v1', [1, 0, 0], metadata);
  await db.insertWithId('v2', [0, 1, 0]);
  await db.cypher("CREATE (a:T {name: 'A'}), (b:T), (a)-[:R {weight: 2}]->(b)");
  await db.addTriple('<http://ex.org/a>', '<http://ex.org/knows>', '<http://ex.org/b>');

  const before = await db.exportJson();
  const vector = await db.get('v1');
  const otherVector = await db.get('v2');
  await db.saveToRvf(rvfPath);

  const restored = await RvLite.createWithRvf({ dimensions, rvfPath });
  const after = await restored.exportJson();
  // Cypher exports nodes in hash-map order.
  const graphById = ({ nodes, ...graph }) => ({
    ...graph,
    nodes: new Map(nodes.map(node => [node.id, node])),
  });
  assert.deepEqual(
    { vector: await restored.get('v1'), graph: graphById(after.graph) },
    { vector, graph: graphById(before.graph) },
  );
  assert.deepEqual(await restored.get('v2'), otherVector);
  assert.deepEqual(after.triples, before.triples);

  const envelope = JSON.parse(await readFile(rvfPath, 'utf8'));
  assert.equal(envelope.magic, 'RVF1');
  assert.equal(envelope.rvf_version, 1);
  assert.deepEqual(envelope.payload.vectors.entries.find(({ id }) => id === 'v1').metadata, metadata);
});

test('JSON RVF loads existing files with omitted metadata and empty properties', async (t) => {
  const { RvLite } = await import('../dist/index.mjs');
  const dir = await mkdtemp(join(tmpdir(), 'rvlite-sdk-'));
  t.after(() => rm(dir, { recursive: true, force: true }));
  const rvfPath = join(dir, 'db.rvf');
  await writeFile(rvfPath, JSON.stringify({
    rvf_version: 1,
    magic: 'RVF1',
    created_at: '2026-01-01T00:00:00.000Z',
    dimensions: 3,
    distance_metric: 'cosine',
    payload: {
      version: 1,
      saved_at: 1767225600000,
      vectors: {
        entries: [{ id: 'v1', vector: [1, 0, 0] }],
        dimensions: 3,
        distance_metric: 'cosine',
        next_id: 0,
      },
      graph: {
        nodes: [{ id: 'n0', labels: ['T'], properties: {} }],
        edges: [],
        next_node_id: 1,
        next_edge_id: 0,
      },
      triples: { triples: [], named_graphs: {}, default_graph: [], next_id: 0 },
      sql_schemas: [],
    },
  }));

  const db = new RvLite({ dimensions: 3 });
  await db.loadFromRvf(rvfPath);
  assert.deepEqual(await db.get('v1'), { id: 'v1', vector: [1, 0, 0], metadata: undefined });
  assert.deepEqual((await db.exportJson()).graph.nodes, [
    { id: 'n0', labels: ['T'], properties: new Map() },
  ]);
});
