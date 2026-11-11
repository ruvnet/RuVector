// Retrieval correctness through the published native binding (issues #1310, #1313).
// - 10k vectors: a filtered top-10 is full at 1% and 10% selectivity (HNSW and flat);
// - re-inserting an existing id is an upsert with no ghost of the old vector;
// - NaN / ±Inf vectors are rejected for insert, insertBatch and search.
const assert = require('assert');
const { VectorDB } = require('./index.js');

const DIM = 32;
const N = 10000;

function rng(seed) {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

function vectors(n, seed) {
  const next = rng(seed);
  return Array.from({ length: n }, () => Float32Array.from({ length: DIM }, next));
}

function open(hnsw) {
  return new VectorDB({
    dimensions: DIM,
    distanceMetric: 'Euclidean',
    storagePath: 'memory://',
    hnswConfig: hnsw ? { m: 16, efConstruction: 100, efSearch: 64, maxElements: 2 * N } : undefined,
  });
}

const entry = (id, vector, i) => ({
  id,
  vector,
  metadata: JSON.stringify({ pct: i % 100, decile: i % 10 }),
});

async function populate(db, base) {
  await db.insertBatch(base.map((v, i) => entry(`v${i}`, v, i)));
}

function distinct(results) {
  const ids = results.map((r) => r.id);
  assert.strictEqual(new Set(ids).size, ids.length, `duplicate ids in top-k: ${ids}`);
}

async function filteredTopK() {
  const base = vectors(N, 11);
  const queries = vectors(20, 12);
  for (const hnsw of [true, false]) {
    const db = open(hnsw);
    await populate(db, base);
    for (const [field, mod] of [['pct', 100], ['decile', 10]]) {
      for (const q of queries) {
        const got = await db.search({ vector: q, k: 10, filter: JSON.stringify({ [field]: 0 }) });
        assert.strictEqual(got.length, 10, `hnsw=${hnsw} ${field}: ${N / mod} eligible, got ${got.length}`);
        distinct(got);
        for (const r of got) assert.strictEqual(Number(r.id.slice(1)) % mod, 0, `${r.id} not eligible`);
      }
    }
  }
  console.log('ok filtered top-10 full at 1% and 10% selectivity (hnsw + flat, 10k)');
}

async function upsertWithoutGhosts() {
  const base = vectors(2000, 21);
  const far = vectors(200, 22);
  for (const hnsw of [true, false]) {
    const db = open(hnsw);
    await populate(db, base);
    for (let i = 0; i < far.length; i++) {
      await db.insert(entry(`v${i}`, far[i].map((x) => x + 10), i));
    }
    await db.insertBatch(far.map((v, i) => entry(`v${i}`, v.map((x) => x + 20), i)));
    assert.strictEqual(await db.len(), base.length, `hnsw=${hnsw}: upsert changed len`);
    for (let i = 0; i < far.length; i++) {
      const got = await db.search({ vector: base[i], k: 10 });
      assert.strictEqual(got.length, 10, `hnsw=${hnsw}: underfilled after upsert`);
      distinct(got);
      assert.ok(!got.some((r) => r.id === `v${i}`), `hnsw=${hnsw}: ghost of v${i} at its old position`);
    }
  }
  console.log('ok duplicate-id insert is an upsert with no ghosts (hnsw + flat)');
}

async function rejectsNonFinite() {
  for (const hnsw of [true, false]) {
    const db = open(hnsw);
    await populate(db, vectors(100, 31));
    for (const bad of [NaN, Infinity, -Infinity]) {
      const v = new Float32Array(DIM).fill(0.5);
      v[7] = bad;
      await assert.rejects(db.insert(entry('bad', v, 0)), `hnsw=${hnsw}: insert accepted ${bad}`);
      await assert.rejects(
        db.insertBatch([entry('ok', new Float32Array(DIM).fill(0.1), 0), entry('bad', v, 0)]),
        `hnsw=${hnsw}: insertBatch accepted ${bad}`,
      );
      assert.ok((await db.get('ok')) == null, `hnsw=${hnsw}: batch partially applied`);
      await assert.rejects(db.search({ vector: v, k: 5 }), `hnsw=${hnsw}: search accepted ${bad}`);
    }
    assert.strictEqual(await db.len(), 100);
  }
  console.log('ok NaN/Inf rejected for insert, insertBatch and search');
}

(async () => {
  const failures = [];
  for (const t of [filteredTopK, upsertWithoutGhosts, rejectsNonFinite]) {
    try {
      await t();
    } catch (err) {
      failures.push(t.name);
      console.error(`FAIL ${t.name}: ${err.message}`);
    }
  }
  if (failures.length) process.exit(1);
  console.log('retrieval correctness: all passed');
})();
