'use strict';

const test = require('node:test');
const assert = require('node:assert/strict');
const { mkdtempSync, rmSync } = require('node:fs');
const { tmpdir } = require('node:os');
const path = require('node:path');

const binding = process.env.GRAPH_NODE_BINDING
  ? require(require.resolve(process.env.GRAPH_NODE_BINDING, { paths: [process.cwd()] }))
  : require(path.join(__dirname, '..', 'index.js'));
const { GraphDatabase } = binding;

const options = { distanceMetric: 'Cosine', dimensions: 3 };
const first = { id: 'n1', labels: ['A'], properties: { v: 'first' } };
const second = { id: 'n1', labels: ['B'], properties: { v: 'second' } };
const original = () => ({ ...first, embedding: new Float32Array([1, 0, 0]) });
const duplicate = () => ({ ...second, embedding: new Float32Array([0, 1, 0]) });
const duplicateError = /node.*n1.*already exists/i;

test('createNode rejects a duplicate ID and preserves the original node', async () => {
  const db = new GraphDatabase(options);
  assert.equal(await db.createNode(original()), 'n1');
  assert.equal((await db.stats()).totalNodes, 1);

  const [attempt] = await Promise.allSettled([db.createNode(duplicate())]);
  const { nodes } = await db.query('MATCH (n) RETURN n');
  assert.equal(attempt.status, 'rejected', `Duplicate ID resolved with graph nodes ${JSON.stringify(nodes)}`);
  assert.match(attempt.reason.message, duplicateError);
  assert.equal((await db.stats()).totalNodes, 1);
  assert.deepEqual(nodes, [first]);
  assert.deepEqual((await db.query('MATCH (n:A) RETURN n')).nodes, [first]);
  assert.deepEqual((await db.query('MATCH (n:B) RETURN n')).nodes, []);
});

test('concurrent createNode calls for one ID have a single winner', async () => {
  const db = new GraphDatabase(options);
  const results = await Promise.allSettled([db.createNode(original()), db.createNode(duplicate())]);
  const winner = results.findIndex(result => result.status === 'fulfilled');
  assert.notEqual(winner, -1);
  assert.equal(results[winner].value, 'n1');
  assert.equal(results[1 - winner].status, 'rejected');
  assert.match(results[1 - winner].reason.message, duplicateError);
  assert.deepEqual((await db.query('MATCH (n) RETURN n')).nodes, [winner === 0 ? first : second]);
});

test('createNode rejects duplicates across persistent handles', async t => {
  const dir = mkdtempSync(path.join(tmpdir(), 'ruvector-graph-node-'));
  t.after(() => rmSync(dir, { recursive: true, force: true }));
  const storagePath = path.join(dir, 'graph.redb');
  const writer = new GraphDatabase({ ...options, storagePath });
  const stale = GraphDatabase.open(storagePath);
  assert.equal((await stale.stats()).totalNodes, 0);
  assert.equal(await writer.createNode(original()), 'n1');

  await assert.rejects(stale.createNode(duplicate()), duplicateError);
  assert.equal((await stale.stats()).totalNodes, 0);
  const reopened = GraphDatabase.open(storagePath);
  await assert.rejects(reopened.createNode(duplicate()), duplicateError);
  assert.deepEqual((await reopened.query('MATCH (n) RETURN n')).nodes, [first]);
});

test('batchInsert can replace an ID and createNode can reuse a deleted ID', async () => {
  const db = new GraphDatabase(options);
  const inserted = await db.batchInsert({ nodes: [original()], edges: [] });
  assert.deepEqual(inserted.nodeIds, ['n1']);
  await assert.rejects(db.createNode(duplicate()), duplicateError);

  const replaced = await db.batchInsert({ nodes: [duplicate()], edges: [] });
  assert.deepEqual(replaced.nodeIds, ['n1']);
  assert.deepEqual((await db.query('MATCH (n:A) RETURN n')).nodes, []);
  assert.deepEqual((await db.query('MATCH (n:B) RETURN n')).nodes, [second]);

  assert.equal((await db.deleteNode('n1')).deletedNode, true);
  assert.equal(await db.createNode(original()), 'n1');
  assert.deepEqual((await db.query('MATCH (n) RETURN n')).nodes, [first]);
});
