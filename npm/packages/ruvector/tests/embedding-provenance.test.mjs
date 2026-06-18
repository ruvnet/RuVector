/**
 * ADR-210 — embedding-provenance module contract (offline, no model needed).
 *
 * Covers:
 *   - D4 prefix policies: prefixText() per model card (acceptance gates 6, 7)
 *   - registry consistency: onnx/loader.js MODELS mirrors the policy table
 *   - D0 compare/refuse: mismatched provenance (incl. prefixPolicy) is
 *     refused with an error naming both sides (acceptance gate 4, unit part)
 *   - D0 legacy derivation: stores without provenance default to hash
 *   - D5 env resolution: RUVECTOR_EMBEDDER > RUVECTOR_ONNX > config;
 *     RUVECTOR_REEMBED defaults to refuse
 *   - D1 warning latch: warnHashFallbackOnce fires exactly once
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const require = createRequire(import.meta.url);
const prov = require(path.join(__dirname, '..', 'dist', 'core', 'embedding-provenance.js'));

// ---------------------------------------------------------------------------
// Gate 6 — embedQuery()/embedPassage() prefixes for E5/BGE (pure helper)
// ---------------------------------------------------------------------------
test('gate 6: e5-small-v2 requires query:/passage: prefixes', () => {
  assert.equal(prov.prefixText('e5-small-v2', 'query', 'find docs'), 'query: find docs');
  assert.equal(prov.prefixText('e5-small-v2', 'passage', 'a document'), 'passage: a document');
  assert.equal(prov.getModelPrefixSpec('e5-small-v2').prefixPolicy, 'required');
});

test('gate 6: bge-small-en-v1.5 applies the documented query instruction, passages unprefixed', () => {
  assert.equal(
    prov.prefixText('bge-small-en-v1.5', 'query', 'find docs'),
    'Represent this sentence for searching relevant passages: find docs'
  );
  assert.equal(prov.prefixText('bge-small-en-v1.5', 'passage', 'a document'), 'a document');
  assert.equal(prov.getModelPrefixSpec('bge-small-en-v1.5').prefixPolicy, 'query-recommended');
});

// ---------------------------------------------------------------------------
// Gate 7 — MiniLM applies NO prefix on either entry point
// ---------------------------------------------------------------------------
test('gate 7: MiniLM applies no prefix on either entry point', () => {
  for (const model of ['all-MiniLM-L6-v2', 'all-MiniLM-L12-v2']) {
    assert.equal(prov.prefixText(model, 'query', 'find docs'), 'find docs');
    assert.equal(prov.prefixText(model, 'passage', 'find docs'), 'find docs');
    assert.equal(prov.getModelPrefixSpec(model).prefixPolicy, 'none');
  }
});

test('unknown models default to the no-prefix policy', () => {
  assert.equal(prov.prefixText('some-future-model', 'query', 'x'), 'x');
  assert.equal(prov.prefixText(null, 'passage', 'x'), 'x');
});

// ---------------------------------------------------------------------------
// D4 — loader.js model registry mirrors the policy table
// ---------------------------------------------------------------------------
test('onnx/loader.js MODELS carries prefixPolicy/queryPrefix/passagePrefix consistent with the policy table', async () => {
  const { MODELS } = await import(
    new URL('../dist/core/onnx/loader.js', import.meta.url).href
  );
  for (const [id, entry] of Object.entries(MODELS)) {
    const spec = prov.getModelPrefixSpec(id);
    assert.equal(entry.prefixPolicy, spec.prefixPolicy, `${id} prefixPolicy`);
    assert.equal(entry.queryPrefix, spec.queryPrefix, `${id} queryPrefix`);
    assert.equal(entry.passagePrefix, spec.passagePrefix, `${id} passagePrefix`);
  }
});

// ---------------------------------------------------------------------------
// Gate 4 (unit) — mismatched provenance refused, error names both sides
// ---------------------------------------------------------------------------
const MINILM_PROV = {
  embedderKind: 'onnx-minilm',
  modelId: 'all-MiniLM-L6-v2',
  dimension: 384,
  normalize: true,
  prefixPolicy: 'none',
};

test('gate 4: 256-dim hash store vs 384-dim minilm insert is refused, naming both sides', () => {
  const store = prov.legacyHashProvenance(256);
  assert.throws(
    () => prov.assertProvenanceMatch(store, MINILM_PROV, 'test-store'),
    (err) => {
      assert.equal(err.code, 'ERR_EMBEDDING_PROVENANCE');
      assert.match(err.message, /hash/, 'must name the store embedder');
      assert.match(err.message, /onnx-minilm/, 'must name the active embedder');
      assert.match(err.message, /256/, 'must name the store dimension');
      assert.match(err.message, /384/, 'must name the active dimension');
      assert.ok(err.mismatches.includes('embedderKind'));
      assert.ok(err.mismatches.includes('dimension'));
      return true;
    }
  );
});

test('gate 4: prefixPolicy mismatch alone is refused', () => {
  const stored = { ...MINILM_PROV, modelId: 'e5-small-v2', embedderKind: 'onnx', prefixPolicy: 'required' };
  const active = { ...stored, prefixPolicy: 'none' };
  assert.throws(
    () => prov.assertProvenanceMatch(stored, active, 'test-store'),
    (err) => {
      assert.deepEqual(err.mismatches, ['prefixPolicy']);
      assert.match(err.message, /prefixPolicy/);
      return true;
    }
  );
});

test('matching provenance passes (no throw)', () => {
  prov.assertProvenanceMatch({ ...MINILM_PROV }, { ...MINILM_PROV }, 'test-store');
  assert.deepEqual(prov.compareProvenance(MINILM_PROV, { ...MINILM_PROV }), []);
});

// ---------------------------------------------------------------------------
// D0 — legacy derivation
// ---------------------------------------------------------------------------
test('legacy stores derive { hash, recorded dimension, normalize:false, none }', () => {
  const legacy = prov.legacyHashProvenance(256);
  assert.deepEqual(legacy, {
    embedderKind: 'hash',
    modelId: null,
    dimension: 256,
    normalize: false,
    prefixPolicy: 'none',
  });
});

// ---------------------------------------------------------------------------
// D5 — env flag resolution (RUVECTOR_EMBEDDER wins over RUVECTOR_ONNX)
// ---------------------------------------------------------------------------
test('RUVECTOR_EMBEDDER wins over RUVECTOR_ONNX; ONNX=0/1 map to hash/minilm; default auto', () => {
  const r = (env) => prov.resolveEmbedderSelection(env);
  assert.equal(r({}), 'auto');
  assert.equal(r({ RUVECTOR_EMBEDDER: 'hash' }), 'hash');
  assert.equal(r({ RUVECTOR_EMBEDDER: 'minilm' }), 'minilm');
  assert.equal(r({ RUVECTOR_EMBEDDER: 'AUTO' }), 'auto');
  assert.equal(r({ RUVECTOR_ONNX: '0' }), 'hash');
  assert.equal(r({ RUVECTOR_ONNX: '1' }), 'minilm');
  // precedence: EMBEDDER beats ONNX when both set
  assert.equal(r({ RUVECTOR_EMBEDDER: 'hash', RUVECTOR_ONNX: '1' }), 'hash');
  assert.equal(r({ RUVECTOR_EMBEDDER: 'minilm', RUVECTOR_ONNX: '0' }), 'minilm');
  // unrecognized values fall back to auto
  assert.equal(r({ RUVECTOR_EMBEDDER: 'bogus' }), 'auto');
  assert.equal(r({ RUVECTOR_ONNX: '2' }), 'auto');
});

test('RUVECTOR_REEMBED resolves refuse|warn|auto, default refuse', () => {
  const r = (env) => prov.resolveReembedPolicy(env);
  assert.equal(r({}), 'refuse');
  assert.equal(r({ RUVECTOR_REEMBED: 'warn' }), 'warn');
  assert.equal(r({ RUVECTOR_REEMBED: 'auto' }), 'auto');
  assert.equal(r({ RUVECTOR_REEMBED: 'nonsense' }), 'refuse');
});

// ---------------------------------------------------------------------------
// D1 — warning latch (full once-per-process behavior is gate 2, child test)
// ---------------------------------------------------------------------------
test('warnHashFallbackOnce fires once per process (latch)', () => {
  prov.resetHashFallbackWarningForTests();
  const writes = [];
  const orig = process.stderr.write;
  process.stderr.write = (chunk) => { writes.push(String(chunk)); return true; };
  try {
    assert.equal(prov.warnHashFallbackOnce('test reason'), true);
    assert.equal(prov.warnHashFallbackOnce('second call'), false);
    assert.equal(prov.warnHashFallbackOnce(), false);
  } finally {
    process.stderr.write = orig;
    prov.resetHashFallbackWarningForTests();
  }
  assert.equal(writes.length, 1, 'exactly one stderr warning');
  assert.match(writes[0], /hash-fallback/);
  assert.match(writes[0], /test reason/);
});
