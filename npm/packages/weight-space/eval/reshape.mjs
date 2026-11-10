#!/usr/bin/env node
// Does learned query reshaping improve paraphrase retrieval on held-out queries?
//
// Corpus/queries come from eval/adr-corpus.mjs (real ADR titles; body sentences as queries).
// Two protocols, 3 seeds each:
//   query-split: queries split 60/20/20 train/val/test; documents are shared (feedback on known docs)
//   doc-split:   ADRs split 60/20/20; test documents never appear in training (generalisation)
// Hyper-parameters are chosen on val only. Test is touched once per seed.
// Ship gate (pre-registered): pooled test recall@5 delta > 0 with a 95% bootstrap CI excluding 0.
//
// Usage: node eval/reshape.mjs --corpus corpus.json [--out results.json] [--quick]
import { readFileSync, writeFileSync } from 'node:fs';
import { resolve } from 'node:path';
import { FlatVectorStore, loadFann } from '../dist/esm/index.js';
import { QueryReshaper } from '../dist/esm/reshape.js';

const args = Object.fromEntries(process.argv.slice(2).reduce((acc, a, i, all) => {
  if (a.startsWith('--')) acc.push([a.slice(2), all[i + 1] === undefined || all[i + 1].startsWith('--') ? 'true' : all[i + 1]]);
  return acc;
}, []));
const corpus = JSON.parse(readFileSync(resolve(args.corpus ?? 'corpus.json'), 'utf8'));
const SEEDS = [1, 2, 3];
const KS = [1, 5, 10];
const GRID = args.quick
  ? [{ hidden: 16, epochs: 20, learningRate: 20, negativeWeight: 0.5 }]
  : [16, 32].flatMap((hidden) => [20, 60].flatMap((epochs) => [5, 20, 50].flatMap((learningRate) =>
    [0, 0.5].map((negativeWeight) => ({ hidden, epochs, learningRate, negativeWeight })))));

function rng(seed) {
  let s = seed >>> 0 || 1;
  return () => ((s = (Math.imul(s, 1664525) + 1013904223) >>> 0) / 4294967296);
}
function shuffle(arr, rand) {
  const a = arr.slice();
  for (let i = a.length - 1; i > 0; i--) {
    const j = Math.floor(rand() * (i + 1));
    [a[i], a[j]] = [a[j], a[i]];
  }
  return a;
}

const index = new FlatVectorStore(corpus.dimensions, corpus.docs.length);
for (const d of corpus.docs) await index.upsert(String(d.id), d.vector);

async function ranks(queries, transform) {
  // Rank (1-based) of the relevant doc, or Infinity beyond top-10.
  const out = [];
  for (const q of queries) {
    const v = transform ? transform(q.vector) : q.vector;
    const hits = await index.search(v, 10);
    const r = hits.findIndex((h) => Number(h.id) === q.doc);
    out.push(r < 0 ? Infinity : r + 1);
  }
  return out;
}
const recall = (rs, k) => rs.filter((r) => r <= k).length / rs.length;

async function hardNegatives(q, n = 3) {
  const hits = await index.search(q.vector, n + 1);
  return hits.filter((h) => Number(h.id) !== q.doc).slice(0, n).map((h) => corpus.docs[Number(h.id)].vector);
}

async function trainReshaper(train, cfg, runtime, seed) {
  const pairs = [];
  for (const q of train) {
    pairs.push({ query: q.vector, relevant: corpus.docs[q.doc].vector, irrelevant: cfg.negativeWeight > 0 ? await hardNegatives(q) : [] });
  }
  const rs = await QueryReshaper.create(corpus.dimensions, { hidden: cfg.hidden, seed, runtime });
  const t0 = performance.now();
  const fit = rs.train(pairs, { epochs: cfg.epochs, learningRate: cfg.learningRate, negativeWeight: cfg.negativeWeight, seed });
  return { rs, fit, trainMs: performance.now() - t0 };
}

function split(seed, protocol) {
  const rand = rng(seed * 7919);
  if (protocol === 'query-split') {
    const qs = shuffle(corpus.queries, rand);
    const a = Math.floor(qs.length * 0.6), b = Math.floor(qs.length * 0.8);
    return { train: qs.slice(0, a), val: qs.slice(a, b), test: qs.slice(b) };
  }
  const ids = shuffle(corpus.docs.map((d) => d.id), rand);
  const a = Math.floor(ids.length * 0.6), b = Math.floor(ids.length * 0.8);
  const part = (set) => corpus.queries.filter((q) => set.has(q.doc));
  return { train: part(new Set(ids.slice(0, a))), val: part(new Set(ids.slice(a, b))), test: part(new Set(ids.slice(b))) };
}

function bootstrapCI(deltas, iters = 5000, seed = 99) {
  const rand = rng(seed);
  const means = [];
  for (let i = 0; i < iters; i++) {
    let s = 0;
    for (let j = 0; j < deltas.length; j++) s += deltas[Math.floor(rand() * deltas.length)];
    means.push(s / deltas.length);
  }
  means.sort((x, y) => x - y);
  const mean = deltas.reduce((x, y) => x + y, 0) / deltas.length;
  return { mean, lo: means[Math.floor(iters * 0.025)], hi: means[Math.floor(iters * 0.975)] };
}

const runtime = await loadFann();
const report = { corpus: { docs: corpus.docs.length, queries: corpus.queries.length, embedder: corpus.embedder, dimensions: corpus.dimensions }, grid: GRID.length, protocols: {} };

for (const protocol of ['query-split', 'doc-split']) {
  const perSeed = [];
  const pooled = Object.fromEntries(KS.map((k) => [k, []]));
  for (const seed of SEEDS) {
    const { train, val, test } = split(seed, protocol);
    const valBase = recall(await ranks(val), 5);
    let best = null;
    for (const cfg of GRID) {
      const { rs } = await trainReshaper(train, cfg, runtime, seed);
      const v = recall(await ranks(val, (x) => rs.reshape(x)), 5);
      if (!best || v > best.val) best = { cfg, val: v };
      rs.dispose();
    }
    const { rs, fit, trainMs } = await trainReshaper(train, best.cfg, runtime, seed);
    const baseRanks = await ranks(test);
    const reshapedRanks = await ranks(test, (x) => rs.reshape(x));
    const row = { seed, sizes: { train: train.length, val: val.length, test: test.length }, chosen: best.cfg, valRecall5: { base: valBase, reshaped: best.val }, trainMs: Math.round(trainMs), trainLoss: fit, test: {} };
    for (const k of KS) {
      row.test[`recall@${k}`] = { base: recall(baseRanks, k), reshaped: recall(reshapedRanks, k) };
      baseRanks.forEach((r, i) => pooled[k].push((reshapedRanks[i] <= k ? 1 : 0) - (r <= k ? 1 : 0)));
    }
    rs.dispose();
    perSeed.push(row);
    console.log(protocol, 'seed', seed, JSON.stringify(row.test), 'chosen', JSON.stringify(best.cfg));
  }
  const summary = {};
  for (const k of KS) {
    const base = perSeed.map((r) => r.test[`recall@${k}`].base);
    const resh = perSeed.map((r) => r.test[`recall@${k}`].reshaped);
    summary[`recall@${k}`] = {
      base: base.reduce((a, b) => a + b, 0) / base.length,
      reshaped: resh.reduce((a, b) => a + b, 0) / resh.length,
      delta95: bootstrapCI(pooled[k]),
    };
  }
  const ci = summary['recall@5'].delta95;
  summary.gate = { metric: 'pooled test recall@5 delta', passes: ci.lo > 0, ci };
  report.protocols[protocol] = { perSeed, summary };
  console.log(protocol, 'summary', JSON.stringify(summary));
}

report.ship = report.protocols['query-split'].summary.gate.passes;
if (args.out) writeFileSync(resolve(args.out), JSON.stringify(report, null, 2) + '\n');
console.log('SHIP reshape:', report.ship);
