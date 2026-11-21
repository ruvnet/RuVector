'use strict';

const test = require('node:test');
const assert = require('node:assert');
const { GraphDatabase } = require('../index.js');

async function createGraph() {
  const db = new GraphDatabase({ distanceMetric: 'Cosine', dimensions: 3 });
  const embedding = new Float32Array([1, 0, 0]);
  for (const id of ['n1', 'n2', 'n3', 'n4', 'n5']) {
    await db.createNode({ id, embedding, labels: ['T'], properties: { id } });
  }
  return { db, embedding };
}

test('plain MATCH returns all five nodes', async () => {
  const { db } = await createGraph();
  const result = await db.query('MATCH (n:T) RETURN n');
  assert.deepStrictEqual(result.nodes.map(node => node.id).sort(), ['n1', 'n2', 'n3', 'n4', 'n5']);
  assert.strictEqual(result.edges.length, 0);
});

for (const [name, query, error] of [
  ['LIMIT', 'MATCH (n:T) RETURN n LIMIT 2', /Unsupported Cypher: .*LIMIT/],
  ['SKIP', 'MATCH (n:T) RETURN n SKIP 4', /Unsupported Cypher: .*SKIP/],
  ['ORDER BY', 'MATCH (n:T) RETURN n ORDER BY n.id DESC', /Unsupported Cypher: .*ORDER BY/],
  ['ORDER BY with LIMIT', 'MATCH (n:T) RETURN n ORDER BY n.id DESC LIMIT 1',
    /Unsupported Cypher: .*LIMIT.*ORDER BY/],
  ['count(n)', 'MATCH (n:T) RETURN count(n)', /Unsupported Cypher: .*RETURN/],
  ['nested count(*)', 'MATCH (n:T) RETURN count(*) + 1 AS total', /Unsupported Cypher: .*RETURN/],
  ['WITH', 'MATCH (n:T) WITH n LIMIT 1 RETURN n', /Unsupported Cypher: .*WITH/]
]) {
  test(`query and querySync reject unsupported ${name}`, async () => {
    const { db } = await createGraph();
    await assert.rejects(async () => {
      const { nodes, edges } = await db.query(query);
      assert.fail(`Expected Unsupported Cypher, got ${nodes.length} nodes and ${edges.length} edges`);
    }, error);
    assert.throws(() => db.querySync(query), error);
  });
}

test('type(r) filters by the stored relationship type', async () => {
  const { db, embedding } = await createGraph();
  for (const [from, to] of [['n1', 'n2'], ['n2', 'n3'], ['n3', 'n4']]) {
    await db.createEdge({ from, to, description: 'R', embedding });
  }

  const all = await db.query('MATCH (a)-[r]->(b) RETURN a, b');
  assert.strictEqual(all.edges.length, 3);
  assert.strictEqual(all.nodes.length, 4);
  const ids = all.edges.map(edge => edge.id).sort();

  const query = "MATCH (a)-[r]->(b) WHERE type(r) = 'R' RETURN a, b";
  const matching = await db.query(query);
  assert.deepStrictEqual(matching.edges.map(edge => edge.id).sort(), ids);
  assert.deepStrictEqual(db.querySync(query).edges.map(edge => edge.id).sort(), ids);

  const other = await db.createEdge({
    from: 'n4', to: 'n5', description: 'OTHER', embedding, metadata: { type: 'R' }
  });
  const filtered = await db.query(query);
  assert.deepStrictEqual(filtered.edges.map(edge => edge.id).sort(), ids);
  assert.strictEqual(filtered.nodes.length, 4);

  const mixed = db.querySync("MATCH (a)-[r]->(b) WHERE TyPe(r) = 'OTHER' RETURN a, b");
  assert.deepStrictEqual(mixed.edges.map(edge => edge.id), [other]);
  const missing = await db.query("MATCH (a)-[r]->(b) WHERE type(r) = 'missing' RETURN a, b");
  assert.strictEqual(missing.edges.length, 0);
  assert.strictEqual(missing.nodes.length, 0);
});
