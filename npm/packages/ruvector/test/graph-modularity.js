// Simple undirected graphs: each input edge appears once, adjacency is symmetric.
const assert = require('node:assert/strict');
const modulePath = require.resolve('../dist/core/graph-algorithms');
const { buildGraph, calculateModularity, minCut } = require(modulePath);

// Independent edge-list definition: Q = sum_c[L_c/m - (d_c/(2m))^2].
function oracle(nodes, edges, groups) {
  const total = edges.reduce((n, e) => n + (e.weight ?? 1), 0);
  if (total === 0) return 0;
  return groups.reduce((score, group) => {
    const members = new Set(group);
    let inside = 0, degree = 0;
    for (const edge of edges) {
      const weight = edge.weight ?? 1;
      const from = members.has(edge.from), to = members.has(edge.to);
      if (from && to) inside += weight;
      degree += weight * (Number(from) + Number(to));
    }
    return score + inside / total - (degree / (2 * total)) ** 2;
  }, 0);
}
const cases = [
  { name: 'single edge, one community', nodes: ['a','b'], edges: [{from:'a',to:'b'}], groups: [['a','b']], expected: 0 },
  { name: 'single edge, split communities', nodes: ['a','b'], edges: [{from:'a',to:'b'}], groups: [['a'],['b']], expected: -0.5 },
  { name: 'disconnected equal pairs', nodes: ['a','b','c','d'], edges: [{from:'a',to:'b'},{from:'c',to:'d'}], groups: [['a','b'],['c','d']], expected: 0.5 },
  { name: 'weighted disconnected pairs', nodes: ['a','b','c','d'], edges: [{from:'a',to:'b',weight:2},{from:'c',to:'d',weight:1}], groups: [['a','b'],['c','d']], expected: 4/9 },
  { name: 'triangle with isolated node', nodes: ['a','b','c','d'], edges: [{from:'a',to:'b'},{from:'b',to:'c'},{from:'c',to:'a'}], groups: [['a','b','c'],['d']], expected: 0 },
  { name: 'edgeless control', nodes: ['a','b'], edges: [], groups: [['a'],['b']], expected: 0 },
];
let failures = [];
for (const item of cases) {
  try {
    const graph = buildGraph(item.nodes, item.edges);
    assert.ok(Math.abs(oracle(item.nodes,item.edges,item.groups)-item.expected) < 1e-12, `${item.name}: oracle control`);
    assert.ok(Math.abs(calculateModularity(graph,item.groups)-item.expected) < 1e-12, `${item.name}: modularity`);
    const cut = minCut(graph);
    assert.ok(Math.abs(cut.modularity-oracle(item.nodes,item.edges,cut.groups)) < 1e-12, `${item.name}: minCut score`);
  } catch (error) { failures.push(error.message); }
}
assert.deepEqual(failures, []);
console.log('Graph modularity and minCut scoring controls passed');
