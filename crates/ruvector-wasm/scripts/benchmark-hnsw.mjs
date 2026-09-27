// Bounded, seeded benchmark. Timings are observations, never a release gate.
import { createRequire } from 'node:module';
import { performance } from 'node:perf_hooks';
import { cpus } from 'node:os';
import { vectors, exactIds, recall, readResults } from '../tests/fixtures.mjs';
const { VectorDB } = createRequire(import.meta.url)('../pkg-node/ruvector_wasm.js');
const count = Number(process.env.BENCH_COUNT ?? 2000);
if (!Number.isInteger(count) || count < 100 || count > 10000) throw new Error('BENCH_COUNT must be 100..10000');
const dimensions = 32, k = 10, queryCount = 100, seed = 0x51a7;
const data = vectors(count, dimensions, seed);
const queries = vectors(queryCount, dimensions, 0x9001);
const expected = queries.map(query => exactIds(data, query, k));
const results = [];
for (const enabled of [false, true]) {
  const db = new VectorDB(dimensions, 'cosine', enabled);
  try {
    const insertStart = performance.now();
    for (let i = 0; i < data.length; i++) db.insert(data[i], String(i));
    const insertMs = performance.now() - insertStart;
    for (const query of queries.slice(0, 10)) readResults(db, query, k);
    const times = [], recalls = [];
    for (let i = 0; i < queries.length; i++) {
      const start = performance.now();
      const found = readResults(db, queries[i], k);
      times.push(performance.now() - start);
      recalls.push(recall(found.map(row => row.id), expected[i]));
    }
    times.sort((a, b) => a - b);
    results.push({ indexType: db.indexType, insertMs,
      queryP50Ms: times[Math.floor(times.length * .5)],
      queryP95Ms: times[Math.floor(times.length * .95)],
      meanRecallAt10: recalls.reduce((a, b) => a + b) / recalls.length });
  } finally { db.free(); }
}
console.log(JSON.stringify({ runtime: process.version, platform: process.platform,
  architecture: process.arch, cpu: cpus()[0]?.model, count, dimensions, k, queryCount,
  corpusSeed: seed, heldOutQuerySeed: 0x9001, results,
  limitations: 'Synthetic uniform vectors; single process; warm queries; no persistence or concurrent load; delete/update rebuild not timed. Repeat on production embeddings before choosing HNSW.' }, null, 2));
