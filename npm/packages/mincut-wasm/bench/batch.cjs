// Same graph and exact solver, comparing per-edge FFI calls against one batch.
const { performance } = require('node:perf_hooks');
const assert = require('node:assert/strict');
const native = process.env.MINCUT_ADDON;
const Class = native ? require(native).MinCut : require('@ruvector/mincut-wasm').WasmMinCut;
const endpoints = [], weights = [];
for (let u = 0; u < 40; u++) for (let v = u + 1; v < 40; v++) {
  endpoints.push(u, v); weights.push(1 + ((u * 17 + v) % 11));
}
const uv = new Uint32Array(endpoints), w = new Float64Array(weights);
function run(batch) {
  const graph = new Class();
  const start = performance.now();
  if (batch) graph.batchInsertTyped(uv, w);
  else for (let i = 0; i < w.length; i++) {
    const u = uv[2 * i], v = uv[2 * i + 1];
    graph.insertEdge(native ? u : BigInt(u), native ? v : BigInt(v), w[i]);
  }
  const ms = performance.now() - start;
  const value = native ? graph.minCutValue : graph.minCutValue();
  if (!native) graph.free();
  return { ms, value };
}
run(true); run(false);
const single = [], batch = [];
for (let i = 0; i < 5; i++) {
  const a = run(false), b = run(true);
  assert.equal(a.value, b.value);
  single.push(a.ms); batch.push(b.ms);
}
const median = a => a.sort((x, y) => x - y)[Math.floor(a.length / 2)];
console.log(JSON.stringify({runtime: native ? 'native' : 'wasm', vertices: 40, edges: w.length, repetitions: 5, singleMedianMs: median(single), batchMedianMs: median(batch)}));
