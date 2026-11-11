#!/usr/bin/env node
/**
 * CRUD latency benchmark for the VectorDB wrapper and hook-memory operations.
 *
 *   node scripts/bench/crud-bench.mjs [--n 10000] [--dim 384]
 *
 * Reports per-op latency (p50/p95, µs) for insert into a fresh store, upsert of
 * an existing id, get, update, delete and search, and the cost the upsert check
 * adds to bulk loads (empty vs populated store). Not part of `npm test`.
 */
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';

const require = createRequire(import.meta.url);
const { VectorDB } = require('../../dist/index.js');
const memoryStore = require('../../bin/memory-store.js');

const arg = (name, dflt) => {
  const i = process.argv.indexOf(`--${name}`);
  return i > 0 ? parseInt(process.argv[i + 1], 10) : dflt;
};
const N = arg('n', 10000);
const DIM = arg('dim', 384);
const OPS = Math.min(1000, N);

const rand = () => { const v = new Float32Array(DIM); for (let i = 0; i < DIM; i++) v[i] = Math.random() * 2 - 1; return v; };
const pct = (xs, p) => { const s = [...xs].sort((a, b) => a - b); return s[Math.min(s.length - 1, Math.floor(p * s.length))]; };
async function timeOps(label, count, fn) {
  const t = [];
  for (let i = 0; i < count; i++) { const s = performance.now(); await fn(i); t.push((performance.now() - s) * 1000); }
  return { op: label, n: count, p50_us: +pct(t, 0.5).toFixed(1), p95_us: +pct(t, 0.95).toFixed(1) };
}

const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-crud-bench-'));
try {
  const vecs = Array.from({ length: N }, rand);
  const rows = [];

  // Bulk load: fresh store (upsert check skipped) vs. populated store (checked).
  const fresh = new VectorDB({ dimensions: DIM, storagePath: path.join(dir, 'fresh.db') });
  let s = performance.now();
  for (let i = 0; i < N; i += 1000) await fresh.insertBatch(vecs.slice(i, i + 1000).map((v, j) => ({ id: `v${i + j}`, vector: v })));
  const freshMs = performance.now() - s;
  const extra = vecs.slice(0, 1000).map(rand);
  s = performance.now();
  await fresh.insertBatch(extra.map((v, j) => ({ id: `w${j}`, vector: v })));
  const populatedNewMs = performance.now() - s;
  s = performance.now();
  await fresh.insertBatch(extra.map((v, j) => ({ id: `w${j}`, vector: rand() })));
  const populatedReplaceMs = performance.now() - s;

  const db = fresh;
  rows.push(await timeOps('insert (new id)', OPS, i => db.insert({ id: `n${i}`, vector: rand() })));
  rows.push(await timeOps('insert (existing id → upsert)', OPS, i => db.insert({ id: `n${i}`, vector: rand() })));
  rows.push(await timeOps('get', OPS, i => db.get(`v${i}`)));
  rows.push(await timeOps('update (vector)', OPS, i => db.update(`v${i}`, { vector: rand() })));
  rows.push(await timeOps('update (metadata merge)', OPS, i => db.update(`v${i}`, { metadata: { k: i }, mergeMetadata: true })));
  rows.push(await timeOps('search k=10', OPS, () => db.search({ vector: rand(), k: 10 })));
  rows.push(await timeOps('delete', OPS, i => db.delete(`n${i}`)));

  // Hook memories: in-process ops on a 5000-memory store (the store's cap).
  const data = { memories: Array.from({ length: 5000 }, (_, i) => ({ content: `memory ${i}`, type: 'general', created: new Date(1e12 + i).toISOString(), embedding: [] })) };
  s = performance.now();
  memoryStore.ensureMemoryIds(data);
  const backfillMs = performance.now() - s;
  const ids = data.memories.map(m => m.id);
  rows.push(await timeOps('memory get (5k store)', OPS, i => memoryStore.getMemory(data, ids[i])));
  rows.push(await timeOps('memory list page=50 (5k store)', 200, () => memoryStore.listMemories(data, { limit: 50 })));
  rows.push(await timeOps('memory forget (5k store)', 200, i => memoryStore.deleteMemories(data, [ids[i]])));

  console.log(JSON.stringify({
    node: process.version, platform: `${process.platform}-${process.arch}`, n: N, dim: DIM,
    bulk: {
      fresh_insert_ms_per_1k: +(freshMs / (N / 1000)).toFixed(1),
      populated_insert_new_ids_ms_per_1k: +populatedNewMs.toFixed(1),
      populated_replace_existing_ms_per_1k: +populatedReplaceMs.toFixed(1),
    },
    memory_id_backfill_5k_ms: +backfillMs.toFixed(2),
    ops: rows,
  }, null, 2));
} finally {
  fs.rmSync(dir, { recursive: true, force: true });
}
