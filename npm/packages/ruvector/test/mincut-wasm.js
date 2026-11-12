#!/usr/bin/env node

/**
 * MinCut WASM wrapper tests.
 *
 * @ruvector/mincut-wasm is OPTIONAL, so this suite must pass both with and
 * without it installed. Capability tests are skipped when it is absent; the
 * fallback and degradation tests always run.
 */

const assert = require('assert');
const {
  isMinCutWasmAvailable,
  minCutFast,
  DynamicMinCut,
  RoadRouter,
} = require('../dist/core/mincut-wasm-wrapper.js');
const { buildGraph, minCut } = require('../dist/core/graph-algorithms.js');

console.log('MinCut WASM Wrapper Test\n' + '='.repeat(50));

const available = isMinCutWasmAvailable();
console.log(`\n@ruvector/mincut-wasm available: ${available}`);

// A 5/1/5 triangle: the true global minimum cut isolates the light-edge
// vertex for a weight of 6 (not 1 -- cutting one edge leaves a connected path).
const nodes = ['a', 'b', 'c'];
const edges = [
  { from: 'a', to: 'b', weight: 5 },
  { from: 'b', to: 'c', weight: 1 },
  { from: 'c', to: 'a', weight: 5 },
];
const graph = buildGraph(nodes, edges);

console.log('\n1. minCutFast agrees with the pure-TS solver...');
{
  const ts = minCut(graph);
  const fast = minCutFast(graph);
  assert.strictEqual(ts.cutWeight, 6, 'TS solver should find the exact cut');
  assert.strictEqual(fast.cutWeight, ts.cutWeight, 'fast path must match TS exactly');
  // Whichever path ran, the reported groups must induce the reported weight.
  const side = new Set(fast.groups[0] || []);
  const crossing = edges
    .filter((e) => side.has(e.from) !== side.has(e.to))
    .reduce((s, e) => s + e.weight, 0);
  assert.strictEqual(crossing, fast.cutWeight, 'groups must induce the cut weight');
  console.log(`   ✓ both report ${fast.cutWeight}, groups consistent`);
}

console.log('\n2. A declared-but-edgeless vertex is its own component (cut 0)...');
{
  // Regression: the native solver only sees vertices that appear in an edge,
  // so an isolated vertex must not be silently dropped.
  const g = buildGraph(['a', 'b', 'c', 'lonely'], [
    { from: 'a', to: 'b', weight: 5 },
    { from: 'b', to: 'c', weight: 5 },
  ]);
  assert.strictEqual(minCutFast(g).cutWeight, 0, 'isolated vertex means a zero cut');
  assert.strictEqual(minCutFast(g).cutWeight, minCut(g).cutWeight, 'must match TS');
  console.log('   ✓ isolated vertex handled, matches TS');
}

console.log('\n3. Fallback works when the optional dep is missing...');
{
  // minCutFast never throws: without the native module it uses the TS solver.
  assert.strictEqual(typeof minCutFast(graph).cutWeight, 'number');
  console.log(`   ✓ minCutFast returned a value (native=${available})`);
}

if (!available) {
  console.log('\n4-5. Capability tests SKIPPED (@ruvector/mincut-wasm not installed)');
  console.log('\n' + '='.repeat(50));
  console.log('MinCut WASM wrapper: PASS (fallback mode)');
  process.exit(0);
}

console.log('\n4. DynamicMinCut maintains the cut across edits...');
{
  const d = new DynamicMinCut();
  d.insertEdge('a', 'b', 5);
  d.insertEdge('b', 'c', 1);
  assert.strictEqual(d.insertEdge('c', 'a', 5), 6, 'triangle cut is 6');
  assert.strictEqual(d.numVertices(), 3);
  assert.strictEqual(d.numEdges(), 3);
  assert.strictEqual(d.isConnected(), true);
  // Removing the light edge leaves a 5/5 path; the cheapest cut is now 5.
  assert.strictEqual(d.deleteEdge('b', 'c'), 5, 'cut drops to 5 after deletion');
  const groups = d.partition();
  assert.strictEqual(groups.flat().length, 3, 'partition covers every vertex');
  console.log('   ✓ insert/delete maintained the cut (6 -> 5)');
}

console.log('\n5. RoadRouter computes exact directed routes...');
{
  const r = new RoadRouter(3, [
    { from: 0, to: 1, cost: 10 },
    { from: 1, to: 2, cost: 10 },
  ]);
  const route = r.route(0, 2);
  assert.ok(route, 'route should exist');
  assert.strictEqual(route.cost, 20);
  assert.deepStrictEqual(route.nodes, [0, 1, 2]);
  // Arcs are directed: there is no path back.
  assert.strictEqual(r.route(2, 0), undefined, 'unreachable target yields undefined');
  console.log('   ✓ route 0->2 cost 20; reverse correctly unreachable');
}

console.log('\n' + '='.repeat(50));
console.log('MinCut WASM wrapper: PASS (native mode)');
