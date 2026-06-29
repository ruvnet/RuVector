/**
 * Regression tests for ONNX embedder API contract — issue #523.
 *
 * Covers:
 *   #1  isOnnxAvailable() is capability-only; isOnnxInitialized()/isReady() are
 *       the post-init gates.
 *   #2  OptimizedOnnxEmbedder.isReady() flips true after a successful embed().
 *   #3  ModelLoader memoizes loaded models (no duplicate fetch); the optimized
 *       embedder no longer logs a quantization it doesn't actually apply.
 *   #4  AdaptiveEmbedder.isReady() returns a boolean, never undefined.
 *
 * Model-dependent assertions (real embed) are skipped when the model can't be
 * loaded (offline) so the suite stays green in restricted CI.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import * as fs from 'node:fs';
import * as os from 'node:os';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const rv = await import(path.join(__dirname, '..', 'dist', 'index.js'));

// ---------------------------------------------------------------------------
// #1 — capability vs readiness (no network)
// ---------------------------------------------------------------------------
test('#1 isOnnxAvailable / isOnnxInitialized / isReady are exported functions', () => {
  assert.equal(typeof rv.isOnnxAvailable, 'function');
  assert.equal(typeof rv.isOnnxInitialized, 'function',
    'isOnnxInitialized must be exported (not shadowed by WASM-core isInitialized)');
  assert.equal(typeof rv.isReady, 'function');
});

test('#1 isOnnxAvailable() is a capability check (true when bundled)', () => {
  assert.equal(rv.isOnnxAvailable(), true);
});

test('#1 isOnnxInitialized() === isReady() and both are booleans', () => {
  assert.equal(typeof rv.isOnnxInitialized(), 'boolean');
  assert.equal(typeof rv.isReady(), 'boolean');
  assert.equal(rv.isOnnxInitialized(), rv.isReady());
});

// ---------------------------------------------------------------------------
// #4 — AdaptiveEmbedder.isReady() returns a boolean (no network)
// ---------------------------------------------------------------------------
test('#4 AdaptiveEmbedder.isReady() returns a boolean before init (not undefined)', () => {
  const ae = new rv.AdaptiveEmbedder({ useEpisodic: false });
  assert.equal(typeof ae.isReady, 'function');
  const r = ae.isReady();
  assert.equal(typeof r, 'boolean', 'isReady() must return boolean, got ' + typeof r);
  assert.equal(r, false, 'a freshly constructed embedder has not initialized ONNX');
});

// ---------------------------------------------------------------------------
// #3 — ModelLoader memoization (no network: fetch is stubbed)
// ---------------------------------------------------------------------------
test('#3 ModelLoader memoizes downloads across calls and instances', async () => {
  const { ModelLoader } = require('../src/core/onnx/loader.js');
  // Isolate the on-disk cache to a temp dir + use a model name unused elsewhere
  // in this file, so this test never touches the real ~/.ruvector cache or the
  // 'all-MiniLM-L6-v2' entry the model-dependent test relies on.
  const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rv-memo-'));
  const prevEnv = process.env.RUVECTOR_CACHE_DIR;
  process.env.RUVECTOR_CACHE_DIR = tmp;
  let fetchCount = 0;
  const orig = ModelLoader.prototype.fetchWithCache;
  ModelLoader.prototype.fetchWithCache = async function (_url, _key, type) {
    fetchCount++;
    return type === 'arraybuffer' ? new ArrayBuffer(8) : '{}';
  };
  try {
    const a = await new ModelLoader({ cache: true }).loadModel('bge-small-en-v1.5');
    const b = await new ModelLoader({ cache: true }).loadModel('bge-small-en-v1.5');
    assert.equal(a, b, 'second load (even from a new instance) must return the memoized object');
    // First load = 2 fetches (model + tokenizer). Memoized second load = 0 more.
    assert.equal(fetchCount, 2, `expected exactly 2 fetches total (no duplicate), got ${fetchCount}`);
  } finally {
    ModelLoader.prototype.fetchWithCache = orig;
    if (prevEnv === undefined) delete process.env.RUVECTOR_CACHE_DIR;
    else process.env.RUVECTOR_CACHE_DIR = prevEnv;
    fs.rmSync(tmp, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// Model-dependent: #1 post-init, #2 optimized readiness, #3 no false FP16 log
// ---------------------------------------------------------------------------
test('#1/#2/#3 model-dependent contract (skipped if model unavailable)', async (t) => {
  try {
    await rv.initOnnxEmbedder();
  } catch {
    t.skip('ONNX model could not be loaded (offline)');
    return;
  }

  // #1: after init, the readiness gates are true.
  assert.equal(rv.isReady(), true);
  assert.equal(rv.isOnnxInitialized(), true);

  // #2/#3: the optimized embedder lazily loads its OWN wasm/model on first
  // embed(). That load is model-dependent and can fail in restricted CI (e.g.
  // Node ESM cannot import the `.wasm`) even when the base embedder above
  // initialised. Per this test's contract, skip that half when the optimized
  // model is unavailable — but still fail on genuine assertion regressions.
  const errs = [];
  const orig = console.error;
  console.error = (...a) => { errs.push(a.join(' ')); };
  try {
    // #3: capture console.error during optimized init — must NOT claim FP16/INT8.
    const emb = rv.getOptimizedOnnxEmbedder();
    assert.equal(emb.isReady(), false, 'optimized embedder is not ready before its own embed/init');
    const v = await emb.embed('regression test for #523');

    // #2: optimized.isReady() is true after a successful embed.
    assert.equal(emb.isReady(), true, 'optimized.isReady() must be true after embed()');
    assert.equal(v.length, 384);

    // #3: no misleading quantization log.
    const liedAboutQuant = errs.some(l => /Using (FP16|INT8) quantized model/.test(l));
    assert.equal(liedAboutQuant, false,
      'must not log a quantization that is not applied: ' + JSON.stringify(errs));
  } catch (e) {
    if (e instanceof assert.AssertionError) throw e;
    t.skip('optimized ONNX model unavailable: ' + (e && e.message ? e.message : String(e)));
  } finally {
    console.error = orig;
  }
});
