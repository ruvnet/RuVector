// Regression tests for untrusted input reaching the wasm build.
//
// Every case here once trapped the instance (a panic or stack overflow leaves
// the graph locks held, so the GraphDB never answers again), lost data, or let
// stored text run as Cypher. Build the web target first, then from the crate
// directory:
//   wasm-pack build --target web --release
//   node tests-js/hardening.mjs
//
// Set GRAPH_WASM_PKG to point at another build directory.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgDir = process.env.GRAPH_WASM_PKG ?? path.resolve(here, '..', 'pkg');

const wasm = await import(pathToFileURL(path.join(pkgDir, 'ruvector_graph_wasm.js')).href);
wasm.initSync({ module: fs.readFileSync(path.join(pkgDir, 'ruvector_graph_wasm_bg.wasm')) });
const { GraphDB } = wasm;

let failures = 0;
async function check(name, fn) {
  try {
    await fn();
    console.log(`ok   ${name}`);
  } catch (e) {
    failures += 1;
    console.log(`FAIL ${name}\n     ${String(e?.message ?? e).split('\n')[0]}`);
  }
}

async function rejects(promise, pattern) {
  try {
    await promise;
  } catch (e) {
    const msg = typeof e === 'string' ? e : e?.message ?? JSON.stringify(e);
    assert.match(msg, pattern);
    return;
  }
  assert.fail('expected an error');
}

async function stillAnswers(db) {
  const r = await db.query('RETURN 1 AS one');
  assert.deepEqual(r.data, [{ one: 1 }]);
}

await check('deeply nested queries are rejected and the instance survives', async () => {
  const db = new GraphDB();
  const n = 5000;
  const queries = [
    `RETURN ${'('.repeat(n)}1${')'.repeat(n)} AS x`,
    `RETURN ${'['.repeat(n)}${']'.repeat(n)} AS x`,
    `RETURN ${'{a:'.repeat(n)}1${'}'.repeat(n)} AS x`,
    `RETURN ${'NOT '.repeat(n)}true AS x`,
    `RETURN ${Array(n).fill('1').join(' + ')} AS x`,
    `MATCH (a)${'-[:R]->()'.repeat(n)} RETURN a`,
  ];
  for (const q of queries) {
    await rejects(db.query(q), /nesting/);
    await stillAnswers(db);
  }
  const ok = await db.query(`RETURN ${'('.repeat(200)}7${')'.repeat(200)} AS x`);
  assert.deepEqual(ok.data, [{ x: 7 }]);
});

await check('createHyperedge with mismatched node embeddings errors cleanly', async () => {
  const db = new GraphDB();
  const a = db.createNode([], { embedding: [1, 2] });
  const b = db.createNode([], { embedding: [1, 2, 3] });
  assert.throws(() => db.createHyperedge([a, b], 'mixed'), /dimension/);
  await stillAnswers(db);
});

await check('deep or cyclic property objects are rejected', async () => {
  const db = new GraphDB();
  let deep = 1;
  for (let i = 0; i < 20000; i++) deep = { a: deep };
  assert.throws(() => db.createNode([], { deep }), /nest/);
  const cyclic = { name: 'x' };
  cyclic.self = cyclic;
  assert.throws(() => db.createNode([], cyclic), /nest/);
  const id = db.createNode(['T'], { embedding: [0.5, 0.25], tags: ['a', { k: [1] }] });
  assert.deepEqual(Array.from(db.getNode(id).embedding), [0.5, 0.25]);
  await stillAnswers(db);
});

await check('integers beyond 2^53 do not null the result', async () => {
  const db = new GraphDB();
  const r = await db.query('RETURN 9007199254740993 AS big, 2 AS small');
  assert.deepEqual(r.data, [{ big: 9007199254740992, small: 2 }]);
});

await check('deleteNode removes incident edges and hyperedges everywhere', async () => {
  const db = new GraphDB();
  const a = db.createNode(['A'], { embedding: [1, 0] });
  const b = db.createNode(['B'], { embedding: [0, 1] });
  db.createEdge(a, b, 'R', {});
  db.createEdge(b, b, 'SELF', {});
  const h = db.createHyperedge([a, b], 'pair');
  assert.equal(db.deleteNode(b), true);
  assert.equal(db.getHyperedge(h), undefined);
  const stats = db.stats();
  assert.equal(stats.edgeCount, 0);
  assert.equal(stats.hyperedgeCount, 0);
  assert.equal(stats.hypergraphEdges, 0);
  const r = await db.query('MATCH (a:A)-[r]-() RETURN count(r) AS c');
  assert.deepEqual(r.data, [{ c: 0 }]);
});

await check('exportCypher round-trips edges and cannot inject clauses', async () => {
  const db = new GraphDB();
  const a = db.createNode(['X) CREATE (pwn:Evil'], { "k: 1}) CREATE (pwn2:Evil {z": "v'}) CREATE (pwn3:Evil {a:'" });
  const b = db.createNode(['Person'], { name: "O'Brien\nline two", n: 1.5e-7 });
  db.createEdge(a, b, 'KNOWS', { since: 2020 });
  const copy = new GraphDB();
  await copy.importCypher(db.exportCypher().split('\n'));
  assert.equal(copy.stats().nodeCount, 2);
  assert.equal(copy.stats().edgeCount, 1);
  const r = await copy.query('MATCH (x)-[k:KNOWS]->(p:Person) RETURN labels(x) AS l, k.since AS s, p.name AS name');
  assert.deepEqual(r.data, [{ l: ['X) CREATE (pwn:Evil'], s: 2020, name: "O'Brien\nline two" }]);
  const evil = await copy.query('MATCH (e:Evil) RETURN count(e) AS c');
  assert.deepEqual(evil.data, [{ c: 0 }]);
});

if (failures > 0) {
  console.log(`\n${failures} check(s) failed`);
  process.exit(1);
}
console.log('\nall hardening checks passed');
