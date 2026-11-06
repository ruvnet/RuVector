// One-vector-per-call ingest benchmark for the NodeBackend string-id map.
//
// Every ingestBatch() call persists the id map. Before the journal this
// rewrote the whole `.idmap.json` per call (O(n) per insert, O(n^2) total);
// with the append-only journal each call appends one line.
//
// Run: node scripts/bench-idmap-ingest.mjs [count=10000] [dim=32] [distDir=dist]
// Prints JSON: total time, per-insert mean for each 10% slice of progress,
// and the reopen time (snapshot load + journal replay).
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const count = Number(process.argv[2] ?? 10_000);
const dim = Number(process.argv[3] ?? 32);
const distDir = path.resolve(process.argv[4] ?? 'dist');
const { RvfDatabase } = await import(pathToFileURL(path.join(distDir, 'index.js')).href);

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-bench-idmap-'));
const target = path.join(tmp, 'bench.rvf');
const vec = new Float32Array(dim);
const slices = 10;
const sliceSize = Math.ceil(count / slices);
const perInsertUs = [];

const db = await RvfDatabase.create(target, { dimensions: dim }, 'node');
const t0 = process.hrtime.bigint();
let sliceStart = t0;
for (let i = 0; i < count; i++) {
  for (let d = 0; d < dim; d++) vec[d] = Math.sin(i * 31 + d);
  await db.ingestBatch([{ id: `doc-${i}`, vector: vec }]);
  if ((i + 1) % sliceSize === 0 || i + 1 === count) {
    const now = process.hrtime.bigint();
    const n = (i % sliceSize) + 1;
    perInsertUs.push(Number(now - sliceStart) / 1e3 / n);
    sliceStart = now;
    // Progress on stderr so a long baseline run can be extrapolated if cut short.
    console.error(JSON.stringify({ done: i + 1, elapsedMs: Math.round(Number(now - t0) / 1e6), sliceUs: Math.round(perInsertUs.at(-1)) }));
  }
}
const totalMs = Number(process.hrtime.bigint() - t0) / 1e6;
const tc = process.hrtime.bigint();
await db.close();
const closeMs = Number(process.hrtime.bigint() - tc) / 1e6;

const tr = process.hrtime.bigint();
const reopened = await RvfDatabase.open(target, 'node');
const reopenMs = Number(process.hrtime.bigint() - tr) / 1e6;
let hit = [];
try { hit = await reopened.query(new Float32Array(dim).map((_, d) => Math.sin(7 * 31 + d)), 1); } catch { /* stub native */ }
await reopened.close();

console.log(JSON.stringify({
  count,
  dim,
  dist: distDir,
  totalMs: Math.round(totalMs),
  closeMs: Math.round(closeMs),
  reopenMs: Math.round(reopenMs),
  perInsertUsBySlice: perInsertUs.map((u) => Math.round(u)),
  sanityTop1: hit[0]?.id,
}));
fs.rmSync(tmp, { recursive: true, force: true });
