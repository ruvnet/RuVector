/**
 * Worker-pool CI regression test (issue #523 SOTA).
 *
 * Enforces the core guarantee of the bundled parallel embedder: its output
 * vectors are cosine-equivalent to the single-thread path (quality-neutral
 * speed). Also exercises the hardening: shutdown rejects nothing left hanging.
 *
 * Skipped (not failed) when the model can't be loaded (offline + no cache),
 * so the suite stays green in restricted CI.
 */
import { test } from 'node:test';
import assert from 'node:assert/strict';
import * as path from 'node:path';
import { fileURLToPath } from 'node:url';

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const rv = await import(path.join(__dirname, '..', 'dist', 'index.js'));

test('worker pool is cosine-equivalent to single-thread (skipped offline)', async (t) => {
  try {
    await rv.initOnnxEmbedder();
  } catch {
    t.skip('ONNX model unavailable (offline + no cache)');
    return;
  }

  const texts = Array.from({ length: 20 }, (_, i) =>
    `Pool equivalence sentence ${i}: semantic vector for retrieval workload ${i % 6}.`
  );

  // Single-thread reference.
  const ref = [];
  for (const s of texts) ref.push((await rv.embed(s)).embedding);

  // Parallel pool.
  await rv.initParallelEmbedder();
  assert.ok(rv.getParallelWorkerCount() >= 1, 'pool should report >= 1 worker');
  const par = await rv.embedBatchParallel(texts);

  try {
    assert.equal(par.length, texts.length);
    let minCos = 1;
    for (let i = 0; i < texts.length; i++) {
      assert.equal(par[i].length, ref[i].length, 'dimension mismatch');
      const c = rv.cosineSimilarity(ref[i], par[i]);
      if (c < minCos) minCos = c;
    }
    assert.ok(minCos >= 0.9999, `pool diverges from single-thread (min cosine ${minCos})`);
  } finally {
    await rv.shutdownParallelEmbedder();
  }

  // After shutdown the worker count resets.
  assert.equal(rv.getParallelWorkerCount(), 0, 'pool should report 0 workers after shutdown');
});

test('empty batch returns empty array without starting workers issues', async (t) => {
  try {
    await rv.initOnnxEmbedder();
  } catch {
    t.skip('ONNX model unavailable (offline + no cache)');
    return;
  }
  const out = await rv.embedBatchParallel([]);
  assert.deepEqual(out, []);
  await rv.shutdownParallelEmbedder();
});
