#!/usr/bin/env node
/**
 * IntelligenceEngine.recall uses a RaBitQ candidate index (optional
 * @ruvector/rabitq-wasm) above RUVECTOR_RABITQ_THRESHOLD memories and the
 * exact brute-force scan otherwise.
 *
 *   1. With the real wasm module: after the first (brute-force) eligible
 *      query, recall is served by RaBitQ, its top-k agrees with brute force
 *      (recall@k >= 0.9, scores are exact cosine), and inserts/deletes after
 *      the build are reflected immediately.
 *   2. With the module hidden (child process + --require shim): recall falls
 *      back to brute force and returns exactly the brute-force list.
 *
 * Uses the deterministic hash embedder so no ONNX model is needed.
 */

const assert = require('assert');
const fs = require('fs');
const os = require('os');
const path = require('path');
const { execFileSync } = require('child_process');

process.env.RUVECTOR_EMBEDDER = 'hash';
const ROOT = path.join(__dirname, '..');
const { default: IntelligenceEngine } = require(path.join(ROOT, 'dist', 'core', 'intelligence-engine.js'));
const { isRabitqAvailable } = require(path.join(ROOT, 'dist', 'core', 'rabitq-wrapper.js'));

const N = 1200;
const THRESHOLD = 500;
const K = 10;
const TOPICS = ['database migration', 'react component', 'rust borrow checker', 'kubernetes deploy',
  'vector search', 'oauth token refresh', 'css grid layout', 'python asyncio', 'git rebase conflict', 'wasm memory'];
const QUERIES = ['how do I fix a git rebase conflict', 'vector search recall', 'oauth refresh token expired',
  'rust borrow checker error', 'kubernetes deploy rollout', 'react component state'];

function makeEngine() {
  const engine = new IntelligenceEngine({ enableSona: false, enableAttention: false, enableOnnx: false });
  const memories = [];
  for (let i = 0; i < N; i++) {
    const content = `${TOPICS[i % TOPICS.length]} note ${i} variant ${(i * 7919) % 97}`;
    memories.push({ id: `m${i}`, content, type: 'note', embedding: engine.embed(content), created: '2026-01-01T00:00:00Z', accessed: 0 });
  }
  engine.import({ memories });
  return engine;
}

async function recallWith(engine, query, threshold) {
  process.env.RUVECTOR_RABITQ_THRESHOLD = String(threshold);
  const hits = await engine.recall(query, K);
  return { hits, engine: engine.getStats().recallEngine };
}

async function withWasm() {
  if (!isRabitqAvailable()) {
    console.log('  SKIP  @ruvector/rabitq-wasm not installed (index path)');
    return;
  }
  const engine = makeEngine();
  // First eligible query in a process is brute force (one-shot hooks never pay the build).
  const first = await recallWith(engine, QUERIES[0], THRESHOLD);
  assert.strictEqual(first.engine, 'brute-force', 'first eligible query is brute force');

  let overlap = 0;
  for (const q of QUERIES) {
    const bf = await recallWith(engine, q, 0);
    assert.strictEqual(bf.engine, 'brute-force');
    const rq = await recallWith(engine, q, THRESHOLD);
    assert.strictEqual(rq.engine, 'rabitq', `"${q}" should be served by RaBitQ`);
    assert.strictEqual(rq.hits.length, K);
    const bfIds = new Set(bf.hits.map((h) => h.id));
    overlap += rq.hits.filter((h) => bfIds.has(h.id)).length;
    // Scores are exact cosine and sorted descending.
    for (let i = 1; i < rq.hits.length; i++) assert.ok(rq.hits[i - 1].score >= rq.hits[i].score);
    const byId = new Map(bf.hits.map((h) => [h.id, h.score]));
    for (const h of rq.hits) if (byId.has(h.id)) assert.ok(Math.abs(byId.get(h.id) - h.score) < 1e-9);
  }
  const recallAtK = overlap / (QUERIES.length * K);
  assert.ok(recallAtK >= 0.9, `recall@${K} vs brute force ${recallAtK}`);
  console.log(`  PASS  RaBitQ recall@${K} vs brute force = ${recallAtK.toFixed(3)} over ${QUERIES.length} queries (N=${N})`);

  // Delta tracking: a memory added after the build is found; a deleted one is gone.
  const fresh = await engine.remember('zebra quasar marmalade unique phrase', 'note', 'fresh');
  let r = await recallWith(engine, 'zebra quasar marmalade unique phrase', THRESHOLD);
  assert.strictEqual(r.engine, 'rabitq');
  assert.strictEqual(r.hits[0].id, fresh.id, 'post-build insert is recalled');
  const victim = r.hits[1].id;
  assert.ok(engine.forget(victim));
  r = await recallWith(engine, 'zebra quasar marmalade unique phrase', THRESHOLD);
  assert.ok(!r.hits.some((h) => h.id === victim), 'deleted memory is not returned');
  console.log('  PASS  inserts and deletes after the build are reflected without a rebuild');

  engine.clear();
  r = await recallWith(engine, 'anything', THRESHOLD);
  assert.deepStrictEqual(r.hits, [], 'clear() empties recall');
}

function withoutWasm() {
  const work = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-rabitq-recall-'));
  const shim = path.join(work, 'hide-rabitq.js');
  fs.writeFileSync(shim,
    "const M = require('module'); const orig = M._resolveFilename;\n" +
    "M._resolveFilename = function (req, ...rest) {\n" +
    "  if (req.startsWith('@ruvector/rabitq-wasm')) { const e = new Error(\"Cannot find module '\" + req + \"'\"); e.code = 'MODULE_NOT_FOUND'; throw e; }\n" +
    '  return orig.call(this, req, ...rest);\n};\n');
  try {
    const out = execFileSync(process.execPath, [__filename, '--child-fallback'], {
      env: { ...process.env, NODE_OPTIONS: `${process.env.NODE_OPTIONS || ''} --require ${shim}`.trim() },
      encoding: 'utf8',
    });
    process.stdout.write(out);
  } finally {
    fs.rmSync(work, { recursive: true, force: true });
  }
}

async function childFallback() {
  assert.strictEqual(isRabitqAvailable(), false, 'shim hides the module');
  const engine = makeEngine();
  for (const q of QUERIES) {
    const bf = await recallWith(engine, q, 0);
    const fb = await recallWith(engine, q, THRESHOLD);
    assert.strictEqual(fb.engine, 'brute-force');
    assert.deepStrictEqual(fb.hits.map((h) => [h.id, h.score]), bf.hits.map((h) => [h.id, h.score]));
  }
  console.log('  PASS  without @ruvector/rabitq-wasm recall falls back to the exact brute-force list');
}

(async () => {
  if (process.argv.includes('--child-fallback')) return childFallback();
  await withWasm();
  withoutWasm();
})().catch((e) => {
  console.error(`  FAIL  ${e.stack || e.message}`);
  process.exitCode = 1;
});
