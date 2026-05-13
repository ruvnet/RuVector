#!/usr/bin/env node
// Regression guard for IntelligenceEngine behaviour:
//   #315  — after `export()` → `import()` on a fresh engine, `recall()` must
//           still return the indexed memories (the import path used to skip
//           re-inserting into the HNSW VectorDB, and recall() returned [] in
//           silence instead of falling back to the brute-force scan).
//   #316  — sync `embed()` must NOT return a hash vector while an ONNX
//           embedder is configured — that produces silent dimension/space
//           mismatches with `embedAsync()`'s ONNX vectors.

import assert from 'node:assert/strict';
import { resolve, dirname } from 'node:path';
import { fileURLToPath } from 'node:url';

const REPO_ROOT = resolve(dirname(fileURLToPath(import.meta.url)), '..', '..');
const ENTRY = resolve(REPO_ROOT, 'npm/packages/ruvector/dist/core/intelligence-engine.js');

const mod = await import(ENTRY);
const IntelligenceEngine = mod.IntelligenceEngine || mod.default;
if (!IntelligenceEngine) {
  console.error('✗ IntelligenceEngine export not found at', ENTRY);
  process.exit(1);
}

// ── #315 — import() must repopulate enough state for recall() to return hits.
{
  const e1 = new IntelligenceEngine({ embeddingDim: 64, useOnnx: false });
  await e1.remember('the cat sat on the mat', 'general');
  await e1.remember('quick brown fox jumps over lazy dog', 'general');
  const exported = e1.export();

  const e2 = new IntelligenceEngine({ embeddingDim: 64, useOnnx: false });
  e2.import(exported);
  const results = await e2.recall('cat mat', 5);

  assert.ok(
    Array.isArray(results) && results.length > 0,
    `#315 regression: recall() after import() returned ${JSON.stringify(results)} — ` +
      'import() must repopulate the vector index (or recall() must fall back to brute force).',
  );
  console.log(`✓ #315 — import()→recall() returned ${results.length} hit(s)`);
}

// ── #316 — sync embed() must refuse when an ONNX embedder is configured.
{
  const e = new IntelligenceEngine({ embeddingDim: 384, useOnnx: false });

  // Force the "ONNX is configured" state without actually loading ONNX.
  e.onnxEmbedder = {
    init: async () => {},
    embed: async () => new Array(384).fill(0.1),
  };
  e.onnxReady = true;

  let threw = false;
  try {
    e.embed('hello');
  } catch (err) {
    threw = true;
    assert.match(
      String(err.message),
      /embedAsync|ONNX/i,
      `#316 regression: embed() threw, but the message doesn't direct the caller to embedAsync(): ${err.message}`,
    );
  }
  assert.ok(threw, '#316 regression: sync embed() did NOT throw while an ONNX embedder is configured.');
  console.log('✓ #316 — sync embed() throws when an ONNX embedder is configured');

  // And embedAsync() must STILL work (graceful fallback to attention/hash).
  const vec = await e.embedAsync('hello');
  assert.ok(Array.isArray(vec) && vec.length === 384, '#316 follow-up: embedAsync() should still return a vector');
  console.log(`✓ #316 — embedAsync() still returns a ${vec.length}-dim vector when ONNX embed succeeds`);
}

console.log('\n✓ IntelligenceEngine invariants OK');
