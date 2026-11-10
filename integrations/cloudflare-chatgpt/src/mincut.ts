import { initSync, WasmMinCut } from '@ruvector/mincut-wasm/web';

let initialized = false;

export function calculateMinCut(module: WebAssembly.Module, edges: Array<[number, number, number]>) {
  if (edges.length < 1 || edges.length > 100 || edges.some(([u, v, weight]) =>
    !Number.isSafeInteger(u) || !Number.isSafeInteger(v) || u < 0 || v < 0 ||
    u > 1000 || v > 1000 || u === v || !Number.isFinite(weight) || weight <= 0 || weight > 10000
  )) throw new RangeError('invalid_edges');
  if (!initialized) {
    initSync({ module });
    initialized = true;
  }
  const graph = WasmMinCut.fromEdges(edges);
  try {
    return { value: graph.minCutValue(), partition: graph.partition(), edgeCount: graph.numEdges() };
  } finally {
    graph.free();
  }
}
