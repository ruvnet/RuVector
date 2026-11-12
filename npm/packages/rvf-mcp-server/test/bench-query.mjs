// Query-latency benchmark through the MCP tool path (in-memory transport).
//
// Run: npm run build && node test/bench-query.mjs [n=10000] [dim=128] [queries=200] [dist=../dist]
// Prints JSON with ingest time, the first (cold) query, warm rvf_query
// p50/p95/mean latency (ms) and recall@10 against an exact scan.
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';
import { Client } from '@modelcontextprotocol/sdk/client/index.js';
import { InMemoryTransport } from '@modelcontextprotocol/sdk/inMemory.js';

const n = Number(process.argv[2] ?? 10_000);
const dim = Number(process.argv[3] ?? 128);
const queries = Number(process.argv[4] ?? 200);
const dist = path.resolve(path.dirname(new URL(import.meta.url).pathname), process.argv[5] ?? '../dist');
const { RvfMcpServer } = await import(pathToFileURL(path.join(dist, 'index.js')).href);

const dataDir = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-mcp-bench-'));
const server = new RvfMcpServer({ dataDir });
const [ct, st] = InMemoryTransport.createLinkedPair();
await server.connect(st);
const client = new Client({ name: 'bench', version: '1.0.0' });
await client.connect(ct);
const call = async (name, args) => JSON.parse((await client.callTool({ name, arguments: args })).content[0].text);

// Deterministic pseudo-random vectors (mulberry32: 32-bit integer state, full
// 2^32 period). A float LCG like (seed * 1103515245 + 12345) % 2 ** 31 loses
// precision past 2^53 and cycles after ~10k draws, which turns the dataset into
// ~80 repeated vectors and makes recall against an exact scan meaningless.
let seed = 42;
const rnd = () => {
  seed = (seed + 0x6d2b79f5) | 0;
  let t = Math.imul(seed ^ (seed >>> 15), 1 | seed);
  t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
  return ((t ^ (t >>> 14)) >>> 0) / 2 ** 32 - 0.5;
};
const vector = () => Array.from({ length: dim }, rnd);

const all = [];
const { storeId } = await call('rvf_create_store', { path: 'bench.rvf', dimensions: dim, metric: 'l2' });
const t0 = performance.now();
for (let i = 0; i < n; i += 10_000) {
  const entries = [];
  for (let j = i; j < Math.min(n, i + 10_000); j++) {
    const v = vector();
    all.push(v);
    entries.push({ id: `v${j}`, vector: v });
  }
  await call('rvf_ingest', { storeId, entries });
}
const ingestMs = performance.now() - t0;

const exactTop10 = (q) => all
  .map((v, i) => [v.reduce((s, x, d) => s + (x - q[d]) ** 2, 0), i])
  .sort((a, b) => a[0] - b[0]).slice(0, 10).map(([, i]) => `v${i}`);

const tc = performance.now();
await call('rvf_query', { storeId, vector: vector(), k: 10 });
const firstQueryMs = performance.now() - tc;

const lat = [];
let hits = 0;
let checked = 0;
for (let i = 0; i < queries; i++) {
  const v = vector();
  const s = performance.now();
  const r = await call('rvf_query', { storeId, vector: v, k: 10 });
  lat.push(performance.now() - s);
  if (r.results.length !== 10) throw new Error(`expected 10 results, got ${r.results.length}`);
  if (i < 20) {
    const truth = new Set(exactTop10(v));
    hits += r.results.filter((x) => truth.has(x.id)).length;
    checked += 10;
  }
}
lat.sort((a, b) => a - b);
const pct = (p) => lat[Math.min(lat.length - 1, Math.floor(p * lat.length))];
console.log(JSON.stringify({
  dist,
  n,
  dim,
  queries,
  ingestMs: Math.round(ingestMs),
  firstQueryMs: Math.round(firstQueryMs),
  queryP50Ms: +pct(0.5).toFixed(3),
  queryP95Ms: +pct(0.95).toFixed(3),
  queryMeanMs: +(lat.reduce((a, b) => a + b, 0) / lat.length).toFixed(3),
  recallAt10: +(hits / checked).toFixed(3),
}));
await server.close();
fs.rmSync(dataDir, { recursive: true, force: true });
