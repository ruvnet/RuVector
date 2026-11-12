// Real topology, unit capacities, controlled delete/restore replay (not historical).
const { readFileSync } = require('node:fs');
const { join } = require('node:path');
const { performance } = require('node:perf_hooks');
const assert = require('node:assert/strict');
const native = !!process.env.MINCUT_ADDON;
const Class = native ? require(process.env.MINCUT_ADDON).MinCut : require(process.env.MINCUT_WASM).WasmMinCut;
const root = process.argv[2];
const quantile = (a, p) => a.slice().sort((x, y) => x - y)[Math.min(a.length - 1, Math.ceil(a.length * p) - 1)];
for (const item of JSON.parse(readFileSync(join(root, 'manifest.json')))) {
  const lines = readFileSync(join(root, item.file), 'utf8').trim().split('\n');
  const edges = lines.slice(1).map(l => l.split(' ').map(Number));
  const endpoints = Uint32Array.from(edges.flatMap(e => e.slice(0, 2)));
  const weights = Float64Array.from(edges.map(e => e[2]));
  const build = [], queries = [], updates = [];
  for (let repetition = 0; repetition < 3; repetition++) {
    const graph = new Class();
    try {
      let start = performance.now();
      assert.equal(graph.batchInsertTyped(endpoints, weights), item.oracle_cut);
      build.push(performance.now() - start);
      const value = () => native ? graph.minCutValue : graph.minCutValue();
      const verify = (expected, omitted) => {
        assert.equal(value(), expected);
        const {s, t} = graph.partition();
        const side = new Set(s);
        assert.equal(side.size + new Set(t).size, item.vertices);
        assert(s.length && t.length && !t.some(v => side.has(v)));
        const cut = edges.reduce((sum, [u, v, w]) =>
          sum + (omitted && ((u === omitted.u && v === omitted.v) || (v === omitted.u && u === omitted.v)) ? 0 : side.has(u) !== side.has(v) ? w : 0), 0);
        assert.equal(cut, expected);
      };
      verify(item.oracle_cut);
      for (let i = 0; i < 200; i++) {
        start = performance.now(); value(); queries.push(performance.now() - start);
      }
      for (const edge of item.trace) {
        const u = native ? edge.u : BigInt(edge.u), v = native ? edge.v : BigInt(edge.v);
        start = performance.now(); graph.deleteEdge(u, v); updates.push(performance.now() - start);
        verify(edge.after_delete, edge);
        start = performance.now(); graph.insertEdge(u, v, edge.weight); updates.push(performance.now() - start);
        verify(item.oracle_cut);
      }
    } finally { if (native) graph.clear(); else graph.free(); }
  }
  console.log(JSON.stringify({name:item.name, runtime:native?'native':'wasm', vertices:item.vertices, edges:item.edges,
    buildMedianMs:quantile(build,.5), queryP50Ms:quantile(queries,.5), queryP95Ms:quantile(queries,.95), queryP99Ms:quantile(queries,.99),
    updateSamples:updates.length, updateP50Ms:updates.length?quantile(updates,.5):null, updateP95Ms:updates.length?quantile(updates,.95):null,
    processPeakRssKiB:process.resourceUsage().maxRSS}));
}
