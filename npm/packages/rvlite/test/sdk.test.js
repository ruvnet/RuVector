// SDK smoke tests against the built dist/ (run `npm run build` first).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';

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
