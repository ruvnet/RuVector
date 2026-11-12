const { test } = require('node:test');
const assert = require('node:assert/strict');
const { WasmMinCut } = require('@ruvector/mincut-wasm');

test('typed batches, failure atomicity, and stable weight replacement', () => {
  const graph = new WasmMinCut();
  try {
    assert.equal(graph.batchInsertTyped(new Uint32Array([0, 1, 1, 2, 0, 2]), new Float64Array([2, 3, 4])), 5);
    assert.throws(() => graph.batchInsert([[2, 3, 1], [3, 4, -1]]));
    assert.equal(graph.numVertices(), 3);
    assert.throws(() => graph.batchDelete([[0, 1], [9, 10]]));
    assert.equal(graph.numEdges(), 3);
    assert.throws(() => graph.updateEdge(0n, 1n, NaN));
    assert.equal(graph.minCutValue(), 5);
    assert.equal(graph.updateEdge(0n, 1n, 0), 3);
    assert.equal(graph.cutEdges().reduce((sum, edge) => sum + edge.weight, 0), 3);
    assert.equal(graph.batchDelete([[0, 1], [1, 2]]), 0);
    assert.throws(() => graph.batchInsertTyped(new Uint32Array([7]), new Float64Array([1])));
  } finally { graph.free(); }
});

test('array APIs reject lossy vertex IDs', () => {
  for (const id of [-1, 0.5, NaN, Infinity, Number.MAX_SAFE_INTEGER + 1]) {
    assert.throws(() => WasmMinCut.fromEdges([[id, 2, 1]]));
  }
});

test('Node ESM import exposes the same class', async () => {
  const module = await import('@ruvector/mincut-wasm');
  assert.equal(module.WasmMinCut, WasmMinCut);
});

test('web entry initializes from bytes without a bundler', async () => {
  const { readFileSync } = require('node:fs');
  const web = await import('@ruvector/mincut-wasm/web');
  await web.default({ module_or_path: readFileSync(require.resolve('@ruvector/mincut-wasm/wasm')) });
  const graph = web.WasmMinCut.fromEdges([[0, 1, 7]]);
  try { assert.equal(graph.minCutValue(), 7); } finally { graph.free(); }
});
