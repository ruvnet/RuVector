/**
 * ADR-210 acceptance gates 1–5 + 7 (gate 8 / wave 2 is the text-corpus
 * recall benchmark, deliberately deferred; gates 6–7's pure-prefix unit
 * coverage lives in embedding-provenance.test.mjs).
 *
 *   1. stats().embedderKind === 'onnx-minilm' when the model loads
 *      (model-dependent — skipped gracefully offline).
 *   2. Fallback emits exactly ONE warning per process (deterministic child
 *      process with an unloadable model id — no network involved).
 *   3. A legacy store (vectors, no provenance) opens read-only for vector
 *      writes; `hooks reembed` unlocks it.
 *   4. Mixed-provenance insert fails with a clear error naming both sides
 *      (hooks store + .db sidecar paths).
 *   5. Normalized embedding L2 norm ∈ [0.999, 1.001] (model-dependent).
 *   7. MiniLM applies no prefix: embed/embedQuery/embedPassage agree
 *      (model-dependent bitwise check).
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { execFileSync, spawnSync } from 'node:child_process';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const PKG = path.join(__dirname, '..');
const CLI = path.join(PKG, 'bin', 'cli.js');
const require = createRequire(import.meta.url);

/** Env without inherited rollout flags, plus explicit overrides. */
function cleanEnv(overrides = {}) {
  const env = { ...process.env, FORCE_COLOR: '0', NO_COLOR: '1' };
  delete env.RUVECTOR_EMBEDDER;
  delete env.RUVECTOR_ONNX;
  delete env.RUVECTOR_REEMBED;
  return { ...env, ...overrides };
}

function cli(cwd, args, envOverrides = {}) {
  const r = spawnSync(process.execPath, [CLI, ...args], {
    cwd,
    encoding: 'utf8',
    timeout: 120000,
    env: cleanEnv(envOverrides),
  });
  return { code: r.status, stdout: r.stdout || '', stderr: r.stderr || '' };
}

function makeProject(intelligence) {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-adr210-'));
  fs.mkdirSync(path.join(dir, '.ruvector'), { recursive: true });
  fs.writeFileSync(
    path.join(dir, '.ruvector', 'intelligence.json'),
    JSON.stringify(intelligence ?? {}, null, 2)
  );
  return dir;
}

const LEGACY_256_STORE = {
  memories: [
    { id: 'm1', memory_type: 'note', content: 'legacy hash-era memory one', embedding: Array(256).fill(0.0625), timestamp: 1 },
    { id: 'm2', memory_type: 'note', content: 'legacy hash-era memory two', embedding: Array(256).fill(0.05), timestamp: 2 },
  ],
  stats: { total_memories: 2 },
};

// ---------------------------------------------------------------------------
// Gate 1 — embedderKind reporting (model-dependent half is skippable)
// ---------------------------------------------------------------------------
test('gate 1: stats().embedderKind is onnx-minilm when loaded, hash-fallback otherwise, hash when disabled', async (t) => {
  const { IntelligenceEngine } = require(path.join(PKG, 'dist', 'core', 'intelligence-engine.js'));

  // RUVECTOR_EMBEDDER=hash forces the legacy embedder (D5) — offline-safe.
  process.env.RUVECTOR_EMBEDDER = 'hash';
  try {
    const hashEngine = new IntelligenceEngine({ enableSona: false, enableAttention: false });
    assert.equal(hashEngine.getStats().embedderKind, 'hash');
    assert.equal(hashEngine.getStats().memoryDimensions, 256);
  } finally {
    delete process.env.RUVECTOR_EMBEDDER;
  }

  // Default (auto): ONNX enabled, 384-dim space, honest readiness reporting.
  const engine = new IntelligenceEngine({ enableSona: false, enableAttention: false });
  assert.equal(engine.getStats().memoryDimensions, 384, 'default-on ONNX uses the 384-dim space');
  const ready = await engine.awaitOnnx();
  if (!ready) {
    assert.equal(engine.getStats().embedderKind, 'hash-fallback',
      'when the model cannot load, the fallback must be reported, not hidden');
    t.skip('ONNX model could not be loaded (offline) — onnx-minilm half skipped');
    return;
  }
  assert.equal(engine.getStats().embedderKind, 'onnx-minilm');
  const v = await engine.embedAsync('semantic embedding check');
  assert.equal(v.length, 384);
});

// ---------------------------------------------------------------------------
// Gate 2 — exactly ONE fallback warning per process (deterministic, offline)
// ---------------------------------------------------------------------------
test('gate 2: hash fallback warns exactly once per process', () => {
  // Unknown model id makes init fail instantly inside ModelLoader — no
  // network, no dependence on the local model cache.
  const script = `
    const { IntelligenceEngine } = require(${JSON.stringify(path.join(PKG, 'dist', 'core', 'intelligence-engine.js'))});
    (async () => {
      const e = new IntelligenceEngine({ enableSona: false, enableAttention: false, onnxConfig: { modelId: 'adr210-no-such-model' } });
      const a = await e.embedAsync('first fallback embed');
      const b = await e.embedAsync('second fallback embed');
      if (a.length !== 384 || b.length !== 384) process.exit(4);
      if (e.getStats().embedderKind !== 'hash-fallback') process.exit(5);
      console.log('CHILD-OK');
    })().catch((err) => { console.error('CHILD-ERR', err && err.message); process.exit(3); });
  `;
  const r = spawnSync(process.execPath, ['-e', script], {
    cwd: PKG,
    encoding: 'utf8',
    timeout: 60000,
    env: cleanEnv(),
  });
  assert.equal(r.status, 0, `child failed (${r.status}): ${r.stderr}`);
  assert.match(r.stdout, /CHILD-OK/);
  const warnings = (r.stderr.match(/ONNX semantic embedder unavailable/g) || []).length;
  assert.equal(warnings, 1, `expected exactly 1 fallback warning, got ${warnings}:\n${r.stderr}`);
});

test('D5: RUVECTOR_EMBEDDER=minilm hard-requires the model (init failure -> error, no fallback)', () => {
  const script = `
    const { IntelligenceEngine } = require(${JSON.stringify(path.join(PKG, 'dist', 'core', 'intelligence-engine.js'))});
    (async () => {
      const e = new IntelligenceEngine({ enableSona: false, enableAttention: false, onnxConfig: { modelId: 'adr210-no-such-model' } });
      try {
        await e.embedAsync('must not fall back');
        console.log('FELL-BACK');
      } catch (err) {
        console.log('THREW:' + err.message);
      }
    })();
  `;
  const r = spawnSync(process.execPath, ['-e', script], {
    cwd: PKG,
    encoding: 'utf8',
    timeout: 60000,
    env: cleanEnv({ RUVECTOR_EMBEDDER: 'minilm' }),
  });
  assert.equal(r.status, 0, r.stderr);
  assert.match(r.stdout, /THREW:.*hard-requires/);
  assert.doesNotMatch(r.stdout, /FELL-BACK/);
});

// ---------------------------------------------------------------------------
// Gate 3 — legacy store is read-only for vector writes; reembed unlocks it
// ---------------------------------------------------------------------------
test('gate 3: a 256-dim legacy store refuses vector writes until `hooks reembed`', (t) => {
  const dir = makeProject(LEGACY_256_STORE);
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  // Vector write refused (read-only), with an actionable message.
  const refused = cli(dir, ['hooks', 'remember', '-t', 'test', 'new memory']);
  assert.notEqual(refused.code, 0, 'legacy vector write must fail');
  const res = JSON.parse(refused.stdout);
  assert.equal(res.success, false);
  assert.equal(res.code, 'ERR_LEGACY_STORE_READONLY');
  assert.match(res.error, /read-only/);
  assert.match(res.error, /hooks reembed/);

  // Non-vector intelligence (routing) still works on a read-only store.
  const route = cli(dir, ['hooks', 'route', 'fix a failing test', '--file', 'src/index.ts']);
  assert.equal(route.code, 0, route.stderr);
  assert.equal(JSON.parse(route.stdout).recommended, 'typescript-developer');

  // Reembed with the (offline-safe) forced hash embedder unlocks the store.
  const reembed = cli(dir, ['hooks', 'reembed'], { RUVECTOR_EMBEDDER: 'hash' });
  assert.equal(reembed.code, 0, reembed.stdout + reembed.stderr);
  const rr = JSON.parse(reembed.stdout);
  assert.equal(rr.success, true);
  assert.equal(rr.reembedded, 2);
  assert.equal(rr.provenance.embedderKind, 'hash');

  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.ok(saved.embeddingProvenance, 'provenance stamped after reembed');
  assert.equal(saved.memories[0].embedding.length, saved.embeddingProvenance.dimension);

  // Vector writes now succeed (matching hash provenance).
  const ok = cli(dir, ['hooks', 'remember', '-t', 'test', 'post-reembed memory']);
  assert.equal(ok.code, 0, ok.stdout + ok.stderr);
  assert.equal(JSON.parse(ok.stdout).success, true);
});

test('gate 3 corollary: a fresh store stamps provenance on first write and stays writable', (t) => {
  const dir = makeProject({});
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const first = cli(dir, ['hooks', 'remember', '-t', 'test', 'first memory']);
  assert.equal(first.code, 0, first.stdout + first.stderr);
  assert.equal(JSON.parse(first.stdout).success, true);

  const saved = JSON.parse(fs.readFileSync(path.join(dir, '.ruvector', 'intelligence.json'), 'utf8'));
  assert.ok(saved.embeddingProvenance, 'first vector write stamps provenance');
  assert.equal(saved.embeddingProvenance.embedderKind, 'hash');

  const second = cli(dir, ['hooks', 'remember', '-t', 'test', 'second memory']);
  assert.equal(second.code, 0);
  assert.equal(JSON.parse(second.stdout).success, true);
});

// ---------------------------------------------------------------------------
// Gate 4 — mixed-provenance insert fails with a clear error (both stores)
// ---------------------------------------------------------------------------
test('gate 4: hooks store stamped onnx-minilm/384 refuses a hash write, naming both sides', (t) => {
  const dir = makeProject({
    embeddingProvenance: { embedderKind: 'onnx-minilm', modelId: 'all-MiniLM-L6-v2', dimension: 384, normalize: true, prefixPolicy: 'none' },
    memories: [
      { id: 'm1', memory_type: 'note', content: 'semantic memory', embedding: Array(384).fill(0.05), timestamp: 1 },
    ],
    stats: { total_memories: 1 },
  });
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));

  const r = cli(dir, ['hooks', 'remember', '-t', 'test', 'hash-embedded write']);
  assert.notEqual(r.code, 0, 'mixed-provenance write must fail');
  const res = JSON.parse(r.stdout);
  assert.equal(res.success, false);
  assert.equal(res.code, 'ERR_EMBEDDING_PROVENANCE');
  assert.match(res.error, /onnx-minilm/, 'names the store embedder');
  assert.match(res.error, /hash/, 'names the active embedder');
  assert.match(res.error, /384/);
  assert.match(res.error, /dimension/);
});

test('gate 4: provenance-stamped .db sidecar refuses mismatched inserts (dimension + prefixPolicy)', (t) => {
  const dir = fs.mkdtempSync(path.join(os.tmpdir(), 'ruvector-adr210-db-'));
  t.after(() => fs.rmSync(dir, { recursive: true, force: true }));
  const dbPath = path.join(dir, 'prov.db');

  // Probe: native VectorDB needed; skip gracefully when unavailable.
  const create = cli(dir, ['create', dbPath, '-d', '4']);
  if (create.code !== 0) {
    t.skip(`VectorDB implementation unavailable: ${create.stderr.slice(-200)}`);
    return;
  }
  const seed = path.join(dir, 'seed.json');
  fs.writeFileSync(seed, JSON.stringify([{ id: 'v1', vector: [1, 0, 0, 0] }]));
  const seedIns = cli(dir, ['insert', dbPath, seed]);
  assert.equal(seedIns.code, 0, seedIns.stderr);

  // Stamp the sidecar with hash/4 provenance (as an embedding-path writer would).
  const storeProv = { embedderKind: 'hash', modelId: null, dimension: 4, normalize: false, prefixPolicy: 'none' };
  fs.writeFileSync(`${dbPath}.meta.json`, JSON.stringify({ dimension: 4, metric: 'cosine', provenance: storeProv }, null, 2));

  // (a) dimension mismatch: 8-dim vectors into the 4-dim hash store.
  const wideFile = path.join(dir, 'wide.json');
  fs.writeFileSync(wideFile, JSON.stringify([{ id: 'v2', vector: [1, 0, 0, 0, 0, 0, 0, 0] }]));
  const wide = cli(dir, ['insert', dbPath, wideFile]);
  assert.notEqual(wide.code, 0, 'mismatched-dimension insert must fail');
  const wideOut = wide.stdout + wide.stderr;
  assert.match(wideOut, /Insert refused/);
  assert.match(wideOut, /embedder=hash/, 'names the store side');
  assert.match(wideOut, /8-dimensional/, 'names the incoming side');

  // (b) declared-provenance prefixPolicy mismatch (same dimension).
  const prefFile = path.join(dir, 'pref.json');
  fs.writeFileSync(prefFile, JSON.stringify({
    provenance: { ...storeProv, prefixPolicy: 'required' },
    vectors: [{ id: 'v3', vector: [0, 1, 0, 0] }],
  }));
  const pref = cli(dir, ['insert', dbPath, prefFile]);
  assert.notEqual(pref.code, 0, 'prefixPolicy-mismatched insert must fail');
  const prefOut = pref.stdout + pref.stderr;
  assert.match(prefOut, /Insert refused/);
  assert.match(prefOut, /prefixPolicy/);

  // Matching declared provenance still inserts fine.
  const okFile = path.join(dir, 'ok.json');
  fs.writeFileSync(okFile, JSON.stringify({ provenance: storeProv, vectors: [{ id: 'v4', vector: [0, 0, 1, 0] }] }));
  const ok = cli(dir, ['insert', dbPath, okFile]);
  assert.equal(ok.code, 0, ok.stdout + ok.stderr);
});

// ---------------------------------------------------------------------------
// Gates 5 + 7 (model-dependent): unit-norm vectors; MiniLM prefix identity
// ---------------------------------------------------------------------------
test('gates 5+7: normalized minilm embeddings (norm in [0.999,1.001]); no prefix on either entry point', async (t) => {
  const rv = await import(new URL('../dist/index.js', import.meta.url).href);
  try {
    await rv.initOnnxEmbedder();
  } catch {
    t.skip('ONNX model could not be loaded (offline)');
    return;
  }

  const norm = (v) => Math.sqrt(v.reduce((s, x) => s + x * x, 0));
  const text = 'the quick brown fox jumps over the lazy dog';

  // Gate 5: normalize defaults to true — every emitted vector is unit norm.
  const plain = await rv.embed(text);
  const query = await rv.embedQuery(text);
  const passage = await rv.embedPassage(text);
  for (const r of [plain, query, passage]) {
    assert.equal(r.dimension, 384);
    const n = norm(r.embedding);
    assert.ok(n >= 0.999 && n <= 1.001, `L2 norm ${n} outside [0.999, 1.001]`);
  }

  // Gate 7: MiniLM applies NO prefix — query/passage/plain are bit-identical.
  assert.deepEqual(query.embedding, plain.embedding, 'embedQuery must not prefix MiniLM input');
  assert.deepEqual(passage.embedding, plain.embedding, 'embedPassage must not prefix MiniLM input');

  // D0: the loaded embedder reports its provenance record.
  const provRecord = rv.getEmbedderProvenance();
  assert.deepEqual(provRecord, {
    embedderKind: 'onnx-minilm',
    modelId: 'all-MiniLM-L6-v2',
    dimension: 384,
    normalize: true,
    prefixPolicy: 'none',
  });
});
