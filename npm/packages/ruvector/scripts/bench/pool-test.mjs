/**
 * Worker-pool correctness + throughput test (issue #523 SOTA).
 *
 *  - cosine-equivalence: pool embeddings must match the single-thread path
 *    (min cosine >= 0.9999) — proves "quality-neutral speed".
 *  - throughput: parallel embeds/sec vs sequential embeds/sec.
 *
 * Exits non-zero if cosine-equivalence fails.
 */
import * as path from 'node:path';
import { performance } from 'node:perf_hooks';
import * as os from 'node:os';

const rv = await import(path.resolve('dist/index.js'));

const N = 48;
const corpus = Array.from({ length: N }, (_, i) =>
  `Parallel embedder correctness sentence ${i}: semantic vector for retrieval workload ${i % 13} indexing tokens and phrases ${i}.`
);

await rv.initOnnxEmbedder();
// warmup
for (let i = 0; i < 5; i++) await rv.embed(corpus[i]);

// --- sequential reference + timing ---
const seq = [];
const sStart = performance.now();
for (const t of corpus) seq.push((await rv.embed(t)).embedding);
const seqElapsed = (performance.now() - sStart) / 1000;
const seqEps = N / seqElapsed;

// --- parallel pool ---
await rv.initParallelEmbedder();
const workers = rv.getParallelWorkerCount();
// warmup the pool
await rv.embedBatchParallel(corpus.slice(0, workers));
const pStart = performance.now();
const par = await rv.embedBatchParallel(corpus);
const parElapsed = (performance.now() - pStart) / 1000;
const parEps = N / parElapsed;

// --- cosine-equivalence ---
let minCos = 1;
for (let i = 0; i < N; i++) {
  const c = rv.cosineSimilarity(seq[i], par[i]);
  if (c < minCos) minCos = c;
}

await rv.shutdownParallelEmbedder();

const report = {
  cpus: os.cpus().length,
  workers,
  n: N,
  seq_eps: +seqEps.toFixed(2),
  parallel_eps: +parEps.toFixed(2),
  speedup: +(parEps / seqEps).toFixed(2),
  min_cosine_vs_single: +minCos.toFixed(6),
  cosine_ok: minCos >= 0.9999,
};
console.log('POOL_REPORT ' + JSON.stringify(report));

if (!report.cosine_ok) {
  console.error('FAIL: pool embeddings diverge from single-thread (min cosine ' + minCos + ')');
  process.exit(1);
}
