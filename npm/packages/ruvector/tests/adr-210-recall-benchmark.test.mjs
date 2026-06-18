/**
 * ADR-210 acceptance gate 8 — the ANN recall benchmark runs on a REAL text
 * fixture (D2), not only uniform-random vectors.
 *
 * The committed fixture (tests/fixtures/text-corpus.json, 600 short English
 * sentences + 60 held-out queries across 10 topics) is embedded with MiniLM
 * and searched through the default VectorDB path this package actually
 * exposes; recall@10 is measured against exact brute-force cosine over the
 * same vectors and must be >= 0.9 on the default path. An efSearch sweep is
 * reported alongside (efSearch is the one ANN knob the JS API exposes).
 *
 * HONEST SCOPE: the RaBitQ oversample floor and the rvf-runtime ef_search
 * floor re-tuning are Rust-side (rvf-runtime) and are NOT reachable from
 * this npm package's API — they are reported as a remaining Rust-side
 * follow-up, not faked here.
 *
 * Model-dependent: skips gracefully when the MiniLM model cannot load
 * (offline / restricted CI). VectorDB-dependent half skips when no native
 * binding is available.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath, pathToFileURL } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const FIXTURE_PATH = path.join(__dirname, 'fixtures', 'text-corpus.json');

test('fixture: text-corpus.json is reproducible from its committed generator', async () => {
  const { generateFixture } = await import(pathToFileURL(path.join(__dirname, 'fixtures', 'generate-text-corpus.mjs')).href);
  const onDisk = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf8'));
  const regenerated = generateFixture(onDisk.seed);
  assert.equal(onDisk.corpus.length, 600);
  assert.equal(onDisk.queries.length, 60);
  assert.deepEqual(regenerated.corpus, onDisk.corpus, 'committed corpus must match the seeded generator');
  assert.deepEqual(regenerated.queries, onDisk.queries, 'committed queries must match the seeded generator');
  // Queries are held out of the corpus (recall@10 is meaningful, not identity)
  const corpusSet = new Set(onDisk.corpus);
  for (const q of onDisk.queries) assert.ok(!corpusSet.has(q), 'query leaked into corpus');
});

test('gate 8: recall@10 >= 0.9 on the text corpus through the default search path', { timeout: 600000 }, async (t) => {
  const rv = await import(pathToFileURL(path.join(PKG, 'dist', 'index.js')).href);

  try {
    await rv.initOnnxEmbedder();
  } catch {
    t.skip('ONNX model could not be loaded (offline) — gate 8 needs MiniLM');
    return;
  }

  const fixture = JSON.parse(fs.readFileSync(FIXTURE_PATH, 'utf8'));
  const { corpus, queries } = fixture;

  // ---- D3 bulk path: embed the corpus through the parallel-fp32 pool ------
  // (a bounded worker count keeps the test footprint sane; the pool default
  // is min(cpus-2, 16))
  let poolWorkers = 0;
  try {
    await rv.initParallelEmbedder(4);
    poolWorkers = rv.getParallelWorkerCount();
  } catch {
    // worker_threads/SAB unavailable — embedBulk falls back single-threaded
  }

  const bulkStart = performance.now();
  const corpusVecs = await rv.embedBulk(corpus);
  const bulkMs = performance.now() - bulkStart;
  assert.equal(corpusVecs.length, corpus.length);
  assert.equal(corpusVecs[0].length, 384);

  // Throughput comparison: single-threaded batch on a 64-text slice.
  const seqStart = performance.now();
  const seqSlice = await rv.embedBatch(corpus.slice(0, 64));
  const seqMs = performance.now() - seqStart;

  // Pool output must be identical to the single-thread path (no quality drift).
  assert.deepEqual(seqSlice[0].embedding, corpusVecs[0], 'bulk pool vectors must match single-thread vectors');

  // Queries go through the query entry point (D4; MiniLM applies no prefix).
  const queryVecs = [];
  for (const q of queries) queryVecs.push((await rv.embedQuery(q)).embedding);

  // ---- Exact ground truth: brute-force cosine top-10 ----------------------
  const K = 10;
  const cosine = (a, b) => {
    let dot = 0;
    for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
    return dot; // unit-norm vectors (gate 5): dot === cosine
  };
  const exactTopK = queryVecs.map(qv =>
    corpusVecs
      .map((cv, i) => ({ i, s: cosine(qv, cv) }))
      .sort((x, y) => y.s - x.s)
      .slice(0, K)
      .map(r => `t${r.i}`)
  );

  // ---- ANN path: the default VectorDB search this package exposes ---------
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-gate8-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  let db;
  try {
    db = new rv.VectorDB({ dimensions: 384, storagePath: path.join(dir, 'gate8.db') });
    await db.insertBatch(corpusVecs.map((v, i) => ({ id: `t${i}`, vector: v })));
  } catch (e) {
    t.skip(`VectorDB implementation unavailable: ${e.message}`);
    return;
  }
  if (rv.getImplementationType && rv.getImplementationType() === 'wasm') {
    t.skip('only the stub VectorDb is available — no real ANN path to measure');
    return;
  }

  async function measureRecall(efSearch) {
    let hits = 0;
    const start = performance.now();
    for (let q = 0; q < queryVecs.length; q++) {
      const query = { vector: queryVecs[q], k: K };
      if (efSearch !== undefined) query.efSearch = efSearch;
      const results = await db.search(query);
      const got = new Set(results.map(r => r.id));
      for (const id of exactTopK[q]) if (got.has(id)) hits++;
    }
    return { recall: hits / (queryVecs.length * K), ms: performance.now() - start };
  }

  const dflt = await measureRecall(undefined);
  const sweep = {};
  for (const ef of [16, 32, 64, 128, 256]) {
    sweep[`ef=${ef}`] = (await measureRecall(ef)).recall;
  }

  // Report measured numbers (the honest, regime-appropriate record D2 asks for).
  console.log('[gate 8] text-corpus recall benchmark (MiniLM, 600 docs, 60 queries, k=10)');
  console.log(`[gate 8]   default search path recall@10 = ${dflt.recall.toFixed(4)} (${dflt.ms.toFixed(0)}ms for 60 queries)`);
  console.log(`[gate 8]   efSearch sweep: ${JSON.stringify(sweep)}`);
  console.log(`[gate 8]   bulk embed: ${corpus.length} texts in ${bulkMs.toFixed(0)}ms ` +
    `(${(corpus.length / (bulkMs / 1000)).toFixed(1)} texts/s, pool workers: ${poolWorkers}) vs ` +
    `single-thread ${(64 / (seqMs / 1000)).toFixed(1)} texts/s`);
  console.log('[gate 8]   RaBitQ oversample / rvf-runtime ef floors: Rust-side, not reachable from the npm API (reported as follow-up)');

  // Gate 8: the default path must reach recall@10 >= 0.9 on real text.
  assert.ok(
    dflt.recall >= 0.9,
    `recall@10 on the text corpus is ${dflt.recall.toFixed(4)}, below the 0.9 gate`
  );

  // Release pool worker threads so the test process can exit.
  await rv.shutdownParallelEmbedder();
});
