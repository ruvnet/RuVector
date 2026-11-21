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

test('SQL SELECT filters before limiting', async () => {
  const { RvLite } = await import('../dist/index.mjs');
  const db = new RvLite({ dimensions: 3 });
  await db.sql('CREATE TABLE docs (id TEXT, grp TEXT, embedding VECTOR(3))');
  assert.deepEqual((await db.sql('SELECT * FROM docs')).rows, []);

  for (let i = 0; i < 1500; i++) {
    const group = i % 5 === 0 ? 'rare' : 'common';
    await db.sql(`INSERT INTO docs (id, grp, embedding) VALUES ('d${i}', '${group}', [1, 0, 0])`);
  }

  const explicit = await db.sql("SELECT * FROM docs WHERE grp = 'common' LIMIT 2000");
  assert.equal(explicit.rows.length, 1200);

  const uncapped = await db.sql("SELECT * FROM docs WHERE grp = 'common'");
  assert.equal(uncapped.rows.length, 1200, 'WHERE without LIMIT must return every matching row');
  assert.equal((await db.sql('SELECT * FROM docs;')).rows.length, 1500);

  const limited = await db.sql("SELECT id FROM docs WHERE grp = 'rare' LIMIT 3");
  assert.equal(limited.rows.length, 3);
  for (const row of limited.rows) {
    assert.deepEqual(Object.keys(row), ['id']);
    assert.equal(Number(row.id.Text.slice(1)) % 5, 0);
  }

  const point = await db.sql("SELECT id, embedding FROM docs WHERE id = 'd1495' AND grp = 'rare'");
  assert.deepEqual(point.rows, [{ id: { Text: 'd1495' }, embedding: { Vector: [1, 0, 0] } }]);
  assert.deepEqual((await db.sql("SELECT * FROM docs WHERE grp = 'missing' LIMIT 3")).rows, []);
  assert.deepEqual((await db.sql("SELECT * FROM docs WHERE grp = 'rare' LIMIT 0")).rows, []);
  assert.deepEqual((await db.sql('SELECT * FROM docs LIMIT 0')).rows, []);

  const ranked = await db.sql("SELECT id FROM docs WHERE grp = 'rare' ORDER BY embedding <-> [1, 0, 0] LIMIT 3");
  assert.equal(ranked.rows.length, 3);
  for (const row of ranked.rows) {
    assert.deepEqual(Object.keys(row), ['id']);
    assert.equal(Number(row.id.Text.slice(1)) % 5, 0);
  }
  const nearest = await db.sql('SELECT * FROM docs ORDER BY embedding <-> [1, 0, 0] LIMIT 1');
  assert.equal(nearest.rows.length, 1);
  assert.deepEqual(nearest.rows[0]._distance, { Real: 0 });
  const scores = await db.sql('SELECT _distance FROM docs ORDER BY embedding <-> [1, 0, 0]');
  assert.deepEqual(scores.rows, Array.from({ length: 10 }, () => ({ _distance: { Real: 0 } })));

  await db.sql("INSERT INTO docs (id, embedding) VALUES ('no-group', [1, 0, 0])");
  const missingGroup = await db.sql("SELECT grp FROM docs WHERE id = 'no-group'");
  assert.deepEqual(missingGroup.rows, [{ grp: 'Null' }]);

  await assert.rejects(db.sql('SELECT missing FROM docs LIMIT 0'), {
    kind: 'SqlError',
    message: /Column 'missing'/,
  });
  await assert.rejects(db.sql('SELECT grp FROM docs GROUP BY grp LIMIT 2000'), {
    kind: 'SqlError',
    message: /GROUP/i,
  });
});
