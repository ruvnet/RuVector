#!/usr/bin/env node
// Latency (p50/p95), memory and before/after numbers for the hot paths.
// "before" = the straightforward implementation; "after" = what the package ships.
//
// Usage: node bench/bench.mjs [--out results.json] [--n 10000]
import { writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import {
  FlatVectorStore, RuvectorVectorStore, WeightRegistry, createMlp, loadFann, trainWithinBudget,
} from '../dist/esm/index.js';
import { QueryReshaper } from '../dist/esm/reshape.js';

const argv = process.argv.slice(2);
const opt = (name, def) => (argv.includes(`--${name}`) ? argv[argv.indexOf(`--${name}`) + 1] : def);
const N = Number(opt('n', 10000));
const DIM = 384;

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
const rand = rng(42);
const vec = (d = DIM) => Array.from({ length: d }, () => rand() * 2 - 1);

function stats(samples) {
  const s = samples.slice().sort((a, b) => a - b);
  const q = (p) => s[Math.min(s.length - 1, Math.floor(p * s.length))];
  return { n: s.length, p50: +q(0.5).toFixed(4), p95: +q(0.95).toFixed(4), mean: +(s.reduce((a, b) => a + b, 0) / s.length).toFixed(4) };
}
async function time(fn, iters, warmup = Math.min(50, iters)) {
  for (let i = 0; i < warmup; i++) await fn(i);
  const out = [];
  for (let i = 0; i < iters; i++) {
    const t = performance.now();
    await fn(i);
    out.push(performance.now() - t);
  }
  return stats(out);
}

// Naive baseline store: array of number[] + full sort per query.
class NaiveStore {
  rows = [];
  upsert(id, v) { this.rows.push({ id, v: Array.from(v) }); }
  search(q, k) {
    const nq = Math.hypot(...q);
    return this.rows
      .map(({ id, v }) => { let d = 0, n = 0; for (let i = 0; i < v.length; i++) { d += q[i] * v[i]; n += v[i] * v[i]; } return { id, similarity: d / (nq * Math.sqrt(n)) }; })
      .sort((a, b) => b.similarity - a.similarity).slice(0, k);
  }
}

const results = { node: process.version, platform: `${process.platform}-${process.arch}`, dims: DIM, n: N };
const rt = await loadFann();
results.fannBackend = rt.backend;

// ── nearest ────────────────────────────────────────────────────────────────
const embeddings = Array.from({ length: N }, () => vec());
const queries = Array.from({ length: 200 }, () => vec());
const naive = new NaiveStore();
const flat = new FlatVectorStore(DIM);
for (let i = 0; i < N; i++) { naive.upsert(String(i), embeddings[i]); await flat.upsert(String(i), embeddings[i]); }
results.nearestStore = {
  'before-naive-array-sort': await time((i) => naive.search(queries[i % 200], 5), 200, 20),
  'after-flat-f32-topk': await time((i) => flat.search(queries[i % 200], 5), 1000),
};
try {
  const core = await RuvectorVectorStore.open(DIM);
  const tIns = performance.now();
  for (let i = 0; i < N; i++) await core.upsert(String(i), embeddings[i]);
  results.nearestStore['ruvector-core-flat'] = await time((i) => core.search(queries[i % 200], 5), 1000);
  results.nearestStore['ruvector-core-flat'].insertMsPerItem = +((performance.now() - tIns) / N).toFixed(4);
  await core.close();
  const hnsw = await RuvectorVectorStore.open(DIM, { hnsw: { m: 16, efConstruction: 100, efSearch: 64, maxElements: N } });
  for (let i = 0; i < N; i++) await hnsw.upsert(String(i), embeddings[i]);
  results.nearestStore['ruvector-core-hnsw'] = await time((i) => hnsw.search(queries[i % 200], 5), 1000);
  let agree = 0;
  for (const q of queries) {
    const a = new Set((await flat.search(q, 5)).map((h) => h.id));
    agree += (await hnsw.search(q, 5)).filter((h) => a.has(h.id)).length;
  }
  results.nearestStore['ruvector-core-hnsw'].recallAt5VsExact = +(agree / (queries.length * 5)).toFixed(4);
  await hnsw.close();
} catch (err) {
  results.nearestStore['ruvector-core'] = `skipped: ${err.message}`;
}

// Full registry.nearest() (index search + record lookup), N snapshots of a [16,32,4] model.
const memBefore = process.memoryUsage();
const reg = await WeightRegistry.open({ dimensions: DIM, store: 'flat' }); // in memory: save excludes the log append
const tSave = [];
for (let i = 0; i < N; i++) {
  const t = performance.now();
  await reg.save(createMlp([16, 32, 4], { seed: i + 1 }), embeddings[i], { task: `t${i}` });
  tSave.push(performance.now() - t);
}
results.registry = {
  snapshots: N, model: [16, 32, 4],
  save: stats(tSave),
  nearestK5: await time((i) => reg.nearest(queries[i % 200], 5), 1000),
  heapDeltaMB: +((process.memoryUsage().heapUsed - memBefore.heapUsed) / 1048576).toFixed(1),
  rssDeltaMB: +((process.memoryUsage().rss - memBefore.rss) / 1048576).toFixed(1),
};
await reg.close();

// ── open (replay the log + index), the per-invocation cost of the ruvector CLI ──
results.open = {};
{
  const { mkdtempSync, rmSync } = await import('node:fs');
  const { tmpdir } = await import('node:os');
  const { join } = await import('node:path');
  for (const n of [1000, N]) {
    for (const store of ['flat', 'ruvector-core']) {
      const dir = mkdtempSync(join(tmpdir(), 'ws-bench-'));
      try {
        const path = join(dir, 'w.jsonl');
        const w = await WeightRegistry.open({ path, store });
        for (let i = 0; i < n; i++) await w.save(createMlp([16, 32, 4], { seed: i + 1 }), embeddings[i], { task: `t${i}` });
        await w.close();
        results.open[`${store} n=${n}`] = await time(async () => { const r = await WeightRegistry.open({ path, store }); await r.nearest(queries[0], 5); await r.close(); }, 5, 1);
      } catch (err) {
        results.open[`${store} n=${n}`] = `skipped: ${err.message}`;
      } finally {
        rmSync(dir, { recursive: true, force: true });
      }
    }
  }
}

// ── predict ────────────────────────────────────────────────────────────────
results.predict = {};
for (const sizes of [[16, 32, 4], [64, 128, 64]]) {
  const m = createMlp(sizes, { seed: 3 });
  const x = Float64Array.from(vec(sizes[0]));
  const prepared = rt.prepare(m, { maxRows: 1 });
  const out = new Float64Array(sizes.at(-1));
  results.predict[sizes.join('-')] = {
    'before-prepare-per-call': await time(() => { const p = rt.prepare(m, { maxRows: 1 }); p.predict(x, 1); p.dispose(); }, 500),
    'after-reused-predictInto': await time(() => prepared.predictInto(x, out, 1), 5000, 500),
  };
  const batch = Float64Array.from({ length: 256 * sizes[0] }, () => rand());
  const bp = rt.prepare(m, { maxRows: 256 });
  const bout = new Float64Array(256 * sizes.at(-1));
  results.predict[sizes.join('-')]['batch256-predictInto'] = await time(() => bp.predictInto(batch, bout, 256), 500);
  prepared.dispose();
  bp.dispose();
}

// ── reshape ────────────────────────────────────────────────────────────────
results.reshape = {};
for (const hidden of [16, 32]) {
  const rs = await QueryReshaper.create(DIM, { hidden, runtime: rt });
  const q = vec();
  const out = new Float64Array(DIM);
  const model = rs.model;
  results.reshape[`384-${hidden}-384`] = {
    'before-prepare-per-call': await time(() => { const p = rt.prepare(model, { maxRows: 1 }); p.predict(Float64Array.from(q), 1); p.dispose(); }, 300),
    'after-reshape': await time(() => rs.reshape(q), 3000, 300),
    'after-reshapeInto': await time(() => rs.reshapeInto(q, out), 3000, 300),
  };
  rs.dispose();
}

// ── train ──────────────────────────────────────────────────────────────────
results.train = {};
{
  const m = createMlp([16, 32, 4], { seed: 5 });
  const rows = 256;
  const x = Float64Array.from({ length: rows * 16 }, () => rand() - 0.5);
  const y = Float64Array.from({ length: rows * 4 }, () => rand() - 0.5);
  results.train['16-32-4 rows=256 epochs=50'] = {
    'before-one-call-per-epoch': await time(() => { let mm = m; for (let e = 0; e < 50; e++) mm = rt.train(mm, { inputs: x, targets: y, rows, epochs: 1, learningRate: 0.05 }).model; }, 20, 3),
    'after-trainWithinBudget': await time(() => trainWithinBudget(rt, m, x, y, rows, { epochs: 50, learningRate: 0.05 }), 20, 3),
  };
  const rm = (await QueryReshaper.create(DIM, { hidden: 16, runtime: rt })).model;
  const rrows = 550;
  const rx = Float64Array.from({ length: rrows * DIM }, () => rand() - 0.5);
  const ry = Float64Array.from({ length: rrows * DIM }, () => (rand() - 0.5) * 0.1);
  results.train['384-16-384 rows=550 epochs=1 (mini-batched)'] = await time(() => trainWithinBudget(rt, rm, rx, ry, rrows, { epochs: 1, learningRate: 5 }), 10, 2);
}

console.log(JSON.stringify(results, null, 2));
const out = opt('out', null);
if (out) writeFileSync(resolve(out), JSON.stringify(results, null, 2) + '\n');
