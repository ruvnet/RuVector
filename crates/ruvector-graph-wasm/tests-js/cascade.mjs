// Acceptance test for the graph-wasm Cypher path.
//
// Build the web target first, then run from the crate directory:
//   wasm-pack build --target web --release
//   node tests-js/cascade.mjs
//
// Set GRAPH_WASM_PKG to point at another build directory.
//
// The graph is built through createNode / createEdge. Every expected row is derived
// in this file from the topology arrays below, not hard-coded.

import assert from 'node:assert/strict';
import fs from 'node:fs';
import path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const here = path.dirname(fileURLToPath(import.meta.url));
const pkgDir = process.env.GRAPH_WASM_PKG ?? path.resolve(here, '..', 'pkg');

const wasm = await import(pathToFileURL(path.join(pkgDir, 'ruvector_graph_wasm.js')).href);
wasm.initSync({ module: fs.readFileSync(path.join(pkgDir, 'ruvector_graph_wasm_bg.wasm')) });
const { GraphDB } = wasm;

// ---- Topology -------------------------------------------------------------

const PORTS = ['LA', 'SEA'];
const CONTAINER_PORT = [0, 0, 1, 0, 1]; // C1, C2 and C4 arrive at LA; C3 and C5 at SEA
const PO_CONTAINER = [0, 1, 2, 3, 4, 0, 1, 2]; // container index per PO
const PO_VALUE = [12000, 48000, 51000, 7000, 90000, 33000, 60500, 25000];
const CUSTOMERS = ['Acme', 'Globex', 'Initech'];
const PO_CUSTOMER = [0, 0, 1, 1, 1, 2, 2, 1]; // customer index per PO (ORDERED)
const DCS = ['DC-East', 'DC-West'];
const PO_DC = [0, 1, 0, 1, 0, 1, 0, 1]; // DC index per PO (DELIVERS_TO)

const NODES = PORTS.length + CONTAINER_PORT.length + PO_VALUE.length + CUSTOMERS.length + DCS.length;
const EDGES = CONTAINER_PORT.length + PO_CONTAINER.length + PO_CUSTOMER.length + PO_DC.length;

// ---- Build ----------------------------------------------------------------

const db = new GraphDB();
const allNodeIds = [];
const node = (labels, props) => {
  const id = db.createNode(labels, props);
  allNodeIds.push(id);
  return id;
};

const port = PORTS.map((name) => node(['Port'], { name }));
const container = CONTAINER_PORT.map((p, i) => {
  const c = node(['Container'], { cid: `C${i + 1}` });
  db.createEdge(c, port[p], 'ARRIVES_AT', {});
  return c;
});
const po = PO_VALUE.map((value, i) => {
  const p = node(['PO'], { po: `PO-${i + 1}`, value });
  db.createEdge(p, container[PO_CONTAINER[i]], 'SHIPS_IN', {});
  return p;
});
const customer = CUSTOMERS.map((name) => node(['Customer'], { name }));
PO_CUSTOMER.forEach((c, i) => db.createEdge(customer[c], po[i], 'ORDERED', {}));
const dc = DCS.map((name) => node(['DC'], { name }));
PO_DC.forEach((d, i) => db.createEdge(po[i], dc[d], 'DELIVERS_TO', {}));

assert.equal(db.stats().nodeCount, NODES, 'fixture node count');
assert.equal(db.stats().edgeCount, EDGES, 'fixture edge count');

// Sample output: the 3-hop customer query on the untouched fixture, as a caller sees it.
const sample = await db.query(
  "MATCH (cu:Customer)-[:ORDERED]->(o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN cu.name, o.po",
);
console.log('\nSAMPLE query():');
console.log(
  JSON.stringify(
    { count: sample.count, isEmpty: sample.isEmpty(), nodes: sample.nodes.length, edges: sample.edges.length, data: sample.data },
    null,
    2,
  ),
);

// ---- Expected values, derived from the topology ---------------------------

const LA = PORTS.indexOf('LA');
const laContainers = CONTAINER_PORT.map((p, i) => i).filter((i) => CONTAINER_PORT[i] === LA);
const laPoIdx = PO_VALUE.map((_, i) => i).filter((i) => laContainers.includes(PO_CONTAINER[i]));
const cmp = (a, b) => (a < b ? -1 : a > b ? 1 : 0);
const sorted = (xs) => [...xs].sort(cmp);
const sum = (xs) => xs.reduce((a, b) => a + b, 0);

// ---- Checks ---------------------------------------------------------------

let passed = 0;
const check = async (name, fn) => {
  try {
    await fn();
    passed += 1;
    console.log(`PASS  ${name}`);
  } catch (err) {
    console.log(`FAIL  ${name}\n${err.message}`);
    process.exitCode = 1;
  }
};

const isPlainObject = (v) => v !== null && typeof v === 'object' && Object.getPrototypeOf(v) === Object.prototype;

await check('MATCH (n) RETURN n returns all 20 nodes', async () => {
  const r = await db.query('MATCH (n) RETURN n');
  const rows = r.data;
  assert.equal(rows.length, NODES);
  assert.equal(r.nodes.length, NODES);
  assert.equal(r.count, NODES);
  assert.equal(r.isEmpty(), false);
  assert.deepEqual(sorted(rows.map((row) => row.n.id)), sorted(allNodeIds));
  for (const row of rows) {
    assert.ok(isPlainObject(row), 'row is a plain object, not a Map');
    assert.ok(Array.isArray(row.n.labels) && row.n.labels.length === 1);
  }
});

await check('MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: LA}) RETURN c', async () => {
  const r = await db.query("MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN c");
  const cids = sorted(r.data.map((row) => row.c.properties.cid));
  assert.deepEqual(cids, sorted(laContainers.map((i) => `C${i + 1}`)));
  assert.equal(r.nodes.length, laContainers.length);
});

await check('2-hop PO -> Container -> Port LA returns o.po, o.value, c.cid', async () => {
  const r = await db.query(
    "MATCH (o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN o.po, o.value, c.cid",
  );
  const got = sorted(r.data.map((row) => JSON.stringify([row['o.po'], row['o.value'], row['c.cid']])));
  const want = sorted(
    laPoIdx.map((i) => JSON.stringify([`PO-${i + 1}`, PO_VALUE[i], `C${PO_CONTAINER[i] + 1}`])),
  );
  assert.deepEqual(got, want);
  assert.equal(r.count, laPoIdx.length);
});

await check('3-hop Customer -> PO -> Container -> Port LA returns cu.name, o.po', async () => {
  const r = await db.query(
    "MATCH (cu:Customer)-[:ORDERED]->(o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN cu.name, o.po",
  );
  const got = sorted(r.data.map((row) => JSON.stringify([row['cu.name'], row['o.po']])));
  const want = sorted(laPoIdx.map((i) => JSON.stringify([CUSTOMERS[PO_CUSTOMER[i]], `PO-${i + 1}`])));
  assert.deepEqual(got, want);
});

await check('WHERE o.value > 50000 filters rows', async () => {
  const r = await db.query('MATCH (o:PO) WHERE o.value > 50000 RETURN o.po, o.value');
  const got = sorted(r.data.map((row) => row['o.po']));
  const want = sorted(PO_VALUE.map((v, i) => (v > 50000 ? `PO-${i + 1}` : null)).filter(Boolean));
  assert.deepEqual(got, want);
  assert.ok(r.data.every((row) => row['o.value'] > 50000));
});

await check('count(o) and sum(o.value), all POs and LA only', async () => {
  const all = await db.query('MATCH (o:PO) RETURN count(o), sum(o.value)');
  assert.equal(all.data.length, 1);
  assert.equal(all.data[0]['count(o)'], PO_VALUE.length);
  assert.equal(all.data[0]['sum(o.value)'], sum(PO_VALUE));

  const la = await db.query(
    "MATCH (o:PO)-[:SHIPS_IN]->(c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN count(o) AS n, sum(o.value) AS total",
  );
  assert.equal(la.data.length, 1);
  assert.equal(la.data[0].n, laPoIdx.length);
  assert.equal(la.data[0].total, sum(laPoIdx.map((i) => PO_VALUE[i])));
});

await check('ORDER BY ... DESC LIMIT returns the top values', async () => {
  const r = await db.query('MATCH (o:PO) RETURN o.po, o.value ORDER BY o.value DESC LIMIT 3');
  const want = [...PO_VALUE].sort((a, b) => b - a).slice(0, 3);
  assert.deepEqual(r.data.map((row) => row['o.value']), want);
});

await check('RETURN DISTINCT collapses duplicate rows', async () => {
  const r = await db.query(
    'MATCH (cu:Customer)-[:ORDERED]->(o:PO)-[:SHIPS_IN]->(c:Container) RETURN DISTINCT cu.name',
  );
  assert.deepEqual(sorted(r.data.map((row) => row['cu.name'])), sorted(CUSTOMERS));
});

await check('CREATE relationship between two MATCHed nodes', async () => {
  const before = db.stats();
  const w = await db.query(
    "MATCH (c:Container {cid: 'C3'}), (p:Port {name: 'LA'}) CREATE (c)-[:ARRIVES_AT]->(p)",
  );
  assert.equal(w.data.length, 0, 'CREATE without RETURN returns no rows');
  const after = db.stats();
  assert.equal(after.edgeCount, before.edgeCount + 1, 'one relationship added');
  assert.equal(after.nodeCount, before.nodeCount, 'no node added');

  const r = await db.query("MATCH (c:Container)-[:ARRIVES_AT]->(p:Port {name: 'LA'}) RETURN c.cid");
  const want = sorted([...laContainers.map((i) => `C${i + 1}`), 'C3']);
  assert.deepEqual(sorted(r.data.map((row) => row['c.cid'])), want);
});

await check('CREATE with RETURN reports the bound variables', async () => {
  const r = await db.query(
    "MATCH (c:Container {cid: 'C5'}), (p:Port {name: 'SEA'}) CREATE (c)-[:ARRIVES_AT]->(p) RETURN c.cid, p.name",
  );
  assert.deepEqual(r.data, [{ 'c.cid': 'C5', 'p.name': 'SEA' }]);
});

await check('unsupported clause rejects with a message', async () => {
  await assert.rejects(db.query('MATCH (n) SET n.x = 1'), (e) => /not supported/.test(e.message));
  await assert.rejects(db.query('MATCH (a)-[*1..3]->(b) RETURN a'), (e) => /Variable-length/.test(e.message));
});

await check('JsNode.properties is a plain object', async () => {
  const r = await db.query("MATCH (p:Port {name: 'SEA'}) RETURN p");
  assert.equal(r.nodes.length, 1);
  assert.ok(isPlainObject(r.nodes[0].properties));
  assert.equal(r.nodes[0].properties.name, 'SEA');
});

console.log(`\n${passed} checks passed${process.exitCode ? ' (with failures)' : ''}`);
