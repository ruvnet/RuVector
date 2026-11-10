// Deeply nested Cypher must be rejected with an Error, not crash the process.
//
// Local build:        node --test test/
// Published package:  GRAPH_NODE_BINDING=@ruvector/graph-node node --test test/
'use strict';

const test = require('node:test');
const assert = require('node:assert');
const path = require('node:path');

const binding = process.env.GRAPH_NODE_BINDING
  ? require(require.resolve(process.env.GRAPH_NODE_BINDING, { paths: [process.cwd()] }))
  : require(path.join(__dirname, '..', 'index.js'));
const { GraphDatabase } = binding;

const db = () => new GraphDatabase({ distanceMetric: 'Cosine', dimensions: 3 });

test('5k nested parentheses are rejected', async () => {
  const n = 5000;
  const q = `MATCH (n) WHERE ${'('.repeat(n)}n.x = 1${')'.repeat(n)} RETURN n`;
  await assert.rejects(db().query(q), /nesting/i);
  assert.throws(() => db().querySync(q), /nesting/i);
});

test('200k chained NOT is rejected', async () => {
  const q = `MATCH (n) WHERE ${'NOT '.repeat(200000)}n.x RETURN n`;
  await assert.rejects(db().query(q), /nesting/i);
  assert.throws(() => db().querySync(q), /nesting/i);
});

test('ordinary nesting still parses', async () => {
  const q = `MATCH (n) WHERE ${'('.repeat(50)}n.x = 1${')'.repeat(50)} RETURN n`;
  await db().query(q);
});
