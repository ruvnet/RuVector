/**
 * ONNX embedder SOTA benchmark + cosine-equivalence harness (issue #523 follow-up).
 *
 * Measures, in one process:
 *   - single embed() latency: p50 / p95 / mean (ms)
 *   - batch-32 throughput (embeds/sec)
 *   - min cosine similarity of every embedding vs the committed baseline
 *     (proves "quality-neutral speed": optimizations must not change vectors)
 *
 * First run writes baseline-embeddings.json and treats min-cosine as 1.0.
 * Every run appends one record to onnx-bench-results.json.
 *
 * Usage: node scripts/bench/onnx-bench.mjs [labelForThisIteration]
 */
import * as path from 'node:path';
import * as fs from 'node:fs';
import { fileURLToPath } from 'node:url';
import { performance } from 'node:perf_hooks';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const pkgRoot = path.resolve(__dirname, '..', '..');
const rv = await import(path.join(pkgRoot, 'dist', 'index.js'));

const label = process.argv[2] || 'baseline';
const N = 64;
const corpus = Array.from({ length: N }, (_, i) =>
  `Vector database sentence ${i}: the quick brown fox number ${i % 9} jumps over ${i % 7} lazy dogs near the river while indexing semantic embeddings for retrieval task ${i}.`
);

function pct(sorted, p) { return sorted[Math.min(sorted.length - 1, Math.floor(sorted.length * p))]; }

await rv.initOnnxEmbedder();

// Warmup (exclude cold-start + JIT from measurement).
for (let i = 0; i < 8; i++) await rv.embed(corpus[i % N]);

// --- single embed() latency ---
const times = [];
for (const t of corpus) {
  const s = performance.now();
  await rv.embed(t);
  times.push(performance.now() - s);
}
times.sort((a, b) => a - b);
const p50 = pct(times, 0.5), p95 = pct(times, 0.95);
const mean = times.reduce((a, b) => a + b, 0) / times.length;

// --- batch-32 throughput ---
const REPS = 8;
const batch = corpus.slice(0, 32);
let embeds = 0;
const bStart = performance.now();
for (let r = 0; r < REPS; r++) {
  const res = await rv.embedBatch(batch);
  embeds += res.length;
}
const bElapsed = (performance.now() - bStart) / 1000;
const throughput = embeds / bElapsed;

// --- cosine-equivalence vs committed baseline ---
const vecs = [];
for (const t of corpus) vecs.push((await rv.embed(t)).embedding);
const baseFile = path.join(__dirname, 'baseline-embeddings.json');
let minCos = 1, isBaseline = false;
if (fs.existsSync(baseFile)) {
  const base = JSON.parse(fs.readFileSync(baseFile, 'utf8'));
  for (let i = 0; i < vecs.length; i++) {
    const c = rv.cosineSimilarity(vecs[i], base[i]);
    if (c < minCos) minCos = c;
  }
} else {
  fs.writeFileSync(baseFile, JSON.stringify(vecs));
  isBaseline = true;
}

const record = {
  iter: 0,
  label,
  node: process.version,
  n: N,
  p50_ms: +p50.toFixed(4),
  p95_ms: +p95.toFixed(4),
  mean_ms: +mean.toFixed(4),
  batch32_throughput_eps: +throughput.toFixed(2),
  min_cosine_vs_baseline: isBaseline ? 'baseline-established' : +minCos.toFixed(6),
  stats: rv.getStats(),
  ts: new Date().toISOString(),
};

const resultsFile = path.join(__dirname, 'onnx-bench-results.json');
const results = fs.existsSync(resultsFile) ? JSON.parse(fs.readFileSync(resultsFile, 'utf8')) : [];
record.iter = results.length + 1;
results.push(record);
fs.writeFileSync(resultsFile, JSON.stringify(results, null, 2));

console.log('=== onnx-bench iteration ' + record.iter + ' (' + label + ') ===');
console.log(JSON.stringify(record, null, 2));
