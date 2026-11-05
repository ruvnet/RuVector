// Event-loop stall and concurrency: sync vs *Async methods.
//
//   node bench/async-bench.mjs [n=20000] [dim=128] [queries=400]
//   UV_THREADPOOL_SIZE=8 node bench/async-bench.mjs
//
// (a) stall: a 1 ms setInterval records how late each tick fires while a
//     batch of queries or a compaction runs, sync vs async.
// (b) throughput: `queries` queries run sequentially with sync `query`
//     vs all at once through `queryAsync` (bounded by UV_THREADPOOL_SIZE).
import { mkdtempSync, rmSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { loadAddon, vectors } from '../tests/load-addon.mjs';

const { RvfDatabase } = loadAddon();
const [n = 20000, dim = 128, queries = 400] = process.argv.slice(2).map(Number);
const pool = process.env.UV_THREADPOOL_SIZE ?? '4 (default)';
const dir = mkdtempSync(join(tmpdir(), 'rvf-node-bench-'));

function pct(sorted, p) {
  return sorted.length ? sorted[Math.min(sorted.length - 1, Math.round((sorted.length - 1) * p))] : 0;
}

/** Run `work` while sampling event-loop lag; returns { ms, maxLag, p99Lag }. */
async function withLagProbe(work) {
  const lags = [];
  let last = performance.now();
  const timer = setInterval(() => {
    const now = performance.now();
    lags.push(Math.max(0, now - last - 1));
    last = now;
  }, 1);
  await new Promise((r) => setTimeout(r, 20)); // let the probe settle
  last = performance.now();
  const t0 = performance.now();
  await work();
  const ms = performance.now() - t0;
  // The sync case blocks the loop for its whole duration; the tick that
  // finally fires afterwards carries that stall, so sample once more.
  await new Promise((r) => setTimeout(r, 5));
  clearInterval(timer);
  lags.sort((a, b) => a - b);
  return { ms, maxLag: lags.at(-1) ?? 0, p99Lag: pct(lags, 0.99) };
}

function makeDb(name, count) {
  const db = RvfDatabase.create(join(dir, name), { dimension: dim });
  const ids = Array.from({ length: count }, (_, i) => i);
  db.ingestBatch(vectors(count, dim, 11), ids);
  return db;
}

const fmt = (r) => `${r.ms.toFixed(1)} ms total, loop lag max ${r.maxLag.toFixed(1)} ms / p99 ${r.p99Lag.toFixed(1)} ms`;

try {
  console.log(`n=${n} dim=${dim} queries=${queries} UV_THREADPOOL_SIZE=${pool} node=${process.version}`);
  const db = makeDb('q.rvf', n);
  const qs = vectors(queries, dim, 77);
  const q = (i) => qs.subarray(i * dim, (i + 1) * dim);
  let t0 = performance.now();
  db.query(q(0), 10); // builds the HNSW index (lazy, first query)
  console.log(`index build: ${(performance.now() - t0).toFixed(0)} ms\n`);

  // (a) stall while a batch of queries runs
  const syncQ = await withLagProbe(async () => {
    for (let i = 0; i < queries; i++) db.query(q(i), 10);
  });
  const asyncQ = await withLagProbe(async () => {
    await Promise.all(Array.from({ length: queries }, (_, i) => db.queryAsync(q(i), 10)));
  });
  console.log(`queries  sync : ${fmt(syncQ)}`);
  console.log(`queries  async: ${fmt(asyncQ)}`);

  // (a) stall while a compaction runs (two identical stores of 5n vectors,
  // half deleted; no index needed)
  const doomed = Array.from({ length: (5 * n) / 2 }, (_, i) => i * 2);
  const c1 = makeDb('c1.rvf', 5 * n);
  const c2 = makeDb('c2.rvf', 5 * n);
  c1.delete(doomed);
  c2.delete(doomed);
  const syncC = await withLagProbe(async () => {
    c1.compact();
  });
  const asyncC = await withLagProbe(async () => {
    await c2.compactAsync();
  });
  console.log(`compact  sync : ${fmt(syncC)}`);
  console.log(`compact  async: ${fmt(asyncC)}`);
  c1.close();
  c2.close();

  // (b) throughput: sequential sync vs concurrent async, 5 alternating
  // rounds, median of each (the host may be shared).
  const seq = [];
  const par = [];
  for (let round = 0; round < 5; round++) {
    t0 = performance.now();
    await Promise.all(Array.from({ length: queries }, (_, i) => db.queryAsync(q(i), 10)));
    par.push(performance.now() - t0);
    t0 = performance.now();
    for (let i = 0; i < queries; i++) db.query(q(i), 10);
    seq.push(performance.now() - t0);
  }
  const seqMs = seq.sort((a, b) => a - b)[2];
  const parMs = par.sort((a, b) => a - b)[2];
  console.log(
    `\nthroughput: sync sequential ${(queries / (seqMs / 1000)).toFixed(0)} q/s, ` +
      `queryAsync concurrent ${(queries / (parMs / 1000)).toFixed(0)} q/s ` +
      `(${(seqMs / parMs).toFixed(2)}x)`,
  );
  db.close();
} finally {
  rmSync(dir, { recursive: true, force: true });
}
