// Real-binding smoke for three rvf fixes (needs @ruvector/rvf-node):
//
//  1. rvf:ingest-dup-id-invalid-manifest: a batch naming an id twice keeps
//     the last write (vector and metadata) instead of failing with
//     InvalidManifest; later writes, queries and reopen keep working.
//  2. rvf:public-metric: RvfDatabase.metric() reports the create() metric.
//  3. rvf:upsert-relink-recall: after 300 in-place upserts on 3,000 x 384
//     cosine, recall@10 is not below delete + insert on the same churn
//     (0.9675 vs 0.9955 on rvf 0.4.9). Same data as agentdb's paired test.
//
// Run: node scripts/smoke-batch-upsert.mjs  (after `tsc` emits dist/), or
//      RVF_SDK=@ruvector/rvf node smoke-batch-upsert.mjs  (installed package)
import assert from 'node:assert';
import fs from 'node:fs';
import os from 'node:os';
import path from 'node:path';
import { pathToFileURL } from 'node:url';

const sdk = process.env.RVF_SDK ?? pathToFileURL(path.resolve('dist/index.js')).href;
const { RvfDatabase } = await import(sdk);

try {
  await import('@ruvector/rvf-node');
} catch {
  console.log('  skip - no @ruvector/rvf-node resolvable');
  process.exit(0);
}

const tmp = fs.mkdtempSync(path.join(os.tmpdir(), 'rvf-batch-upsert-'));
const ok = (name) => console.log(`  ok - ${name}`);
try {
  // 1. Repeated id in one batch.
  {
    const file = path.join(tmp, 'dup.rvf');
    const v = (s) => new Float32Array([s, 1, 2, 3]);
    const db = await RvfDatabase.create(file, { dimensions: 4, metric: 'l2' });
    const r = await db.ingestBatch([
      { id: 'x', vector: v(10), metadata: { tag: 'first' } },
      { id: 'y', vector: v(20), metadata: { tag: 'y' } },
      { id: 'x', vector: v(30), metadata: { tag: 'last' } },
    ]);
    assert.strictEqual(r.accepted, 2);
    await db.ingestBatch([{ id: 'z', vector: v(40) }]);
    const check = async (store) => {
      const [hit] = await store.query(v(30), 1);
      assert.strictEqual(hit.id, 'x');
      assert.ok(hit.distance < 1e-6);
      assert.deepStrictEqual({ ...hit.metadata }, { tag: 'last' });
      assert.strictEqual((await store.status()).totalVectors, 3);
    };
    await check(db);
    await db.close();
    const reopened = await RvfDatabase.open(file);
    await check(reopened);
    await reopened.close();
    ok('a batch that repeats an id keeps the last write and the store keeps working');
  }

  // 2. Public metric.
  for (const metric of ['l2', 'cosine', 'dotproduct']) {
    const file = path.join(tmp, `metric-${metric}.rvf`);
    const db = await RvfDatabase.create(file, { dimensions: 4, metric });
    assert.strictEqual(await db.metric(), metric);
    await db.close();
    const reopened = await RvfDatabase.openReadonly(file);
    assert.strictEqual(await reopened.metric(), metric);
    await reopened.close();
  }
  ok('metric() reports l2 / cosine / dotproduct, before and after reopen');

  // 3. Upsert vs delete + insert recall (agentdb's generator, seed 12345).
  {
    const D = 384, N = 3000, CHURN = 300, Q = 200;
    let seed = 12345;
    const rnd = () => (seed = (seed * 1103515245 + 12345) >>> 0) / 2 ** 32 - 0.5;
    const unit = () => {
      const v = new Float32Array(D);
      let s = 0;
      for (let i = 0; i < D; i++) { v[i] = rnd(); s += v[i] * v[i]; }
      s = Math.sqrt(s);
      for (let i = 0; i < D; i++) v[i] /= s;
      return v;
    };
    const base = new Map();
    for (let i = 0; i < N; i++) base.set(`v${i}`, unit());
    const churn = Array.from({ length: CHURN }, (_, k) => [`v${(k * 37) % N}`, unit()]);
    const queries = Array.from({ length: Q }, unit);
    const latest = new Map(base);
    for (const [id, v] of churn) latest.set(id, v);

    const up = await RvfDatabase.create(path.join(tmp, 'up.rvf'), { dimensions: D, metric: 'cosine' });
    const re = await RvfDatabase.create(path.join(tmp, 're.rvf'), { dimensions: D, metric: 'cosine' });
    const all = [...base].map(([id, vector]) => ({ id, vector }));
    for (let i = 0; i < N; i += 500) {
      await up.ingestBatch(all.slice(i, i + 500));
      await re.ingestBatch(all.slice(i, i + 500));
    }
    await up.query(queries[0], 10);
    await re.query(queries[0], 10);
    for (const [id, vector] of churn) {
      await up.ingestBatch([{ id, vector }]);
      await re.delete([id]);
      await re.ingestBatch([{ id, vector }]);
    }
    const exactTop = (q) =>
      [...latest]
        .map(([id, v]) => { let d = 0; for (let i = 0; i < D; i++) d += v[i] * q[i]; return [id, d]; })
        .sort((a, b) => b[1] - a[1]).slice(0, 10).map(([id]) => id);
    const recall = async (db) => {
      let hit = 0;
      for (const q of queries) {
        const ids = (await db.query(q, 10)).map((h) => h.id);
        assert.strictEqual(new Set(ids).size, ids.length);
        const truth = exactTop(q);
        hit += ids.filter((id) => truth.includes(id)).length;
      }
      return hit / (Q * 10);
    };
    const rUp = await recall(up);
    const rRe = await recall(re);
    await up.close();
    await re.close();
    console.log(`  recall@10 after ${CHURN} updates: upsert ${rUp}, delete+insert ${rRe}`);
    assert.ok(rUp >= rRe - 0.005, `upsert ${rUp} vs delete+insert ${rRe}`);
    ok('in-place upsert recall matches delete + insert');
  }
} finally {
  fs.rmSync(tmp, { recursive: true, force: true });
}
