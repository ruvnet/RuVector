const { test } = require('node:test');
const assert = require('node:assert/strict');
const { MinCut } = require(process.env.MINCUT_ADDON);
test('native typed batch and safe replacement', () => {
  const graph = new MinCut();
  assert.equal(graph.batchInsertTyped(new Uint32Array([0, 1, 1, 2, 0, 2]), new Float64Array([2, 3, 4])), 5);
  assert.throws(() => graph.batchInsertTyped(new Uint32Array([2, 3, 3, 4]), new Float64Array([1, -1])));
  assert.equal(graph.numVertices, 3);
  assert.throws(() => graph.updateEdge(0, 1, NaN));
  assert.equal(graph.minCutValue, 5);
  assert.equal(graph.updateEdge(0, 1, 0), 3);
  assert.equal(graph.minCut().value, 3);
  assert.throws(() => graph.batchDeleteTyped(new Uint32Array([0, 1, 9, 10])));
  assert.equal(graph.numEdges, 3);
  assert.equal(graph.batchDeleteTyped(new Uint32Array([0, 1, 1, 2])), 0);
  graph.clear();
  assert.equal(graph.numVertices, 0);
  assert.equal(graph.numEdges, 0);
  assert.equal(graph.stats.insertions, 0);
  assert.equal(graph.minCutValue, Infinity);
  assert.equal(graph.insertEdge(7, 8, 2), 2);
});
test('invalid approximate configuration throws instead of panicking', () => {
  for (const epsilon of [0, -1, NaN, Infinity, 2]) {
    assert.throws(() => new MinCut({ approximate: true, epsilon }));
    assert.throws(() => MinCut.fromEdges([], { approximate: true, epsilon }));
  }
});

test('clear preserves solver configuration', () => {
  const graph = new MinCut({ approximate: true, epsilon: 0.25 });
  graph.insertEdge(0, 1, 1);
  graph.clear();
  graph.insertEdge(2, 3, 2);
  assert.equal(graph.minCut().approximationRatio, 1.25);
  graph.clear();
});
