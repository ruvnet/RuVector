// Benchmark-harness unit + integration tests (node --test). No network, no child
// processes: the local arm is driven by an in-process fake Model binding.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { createRequire } from 'node:module';
import { mkdtempSync, writeFileSync, mkdirSync, rmSync, readFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';
import { fileURLToPath } from 'node:url';
import { dirname } from 'node:path';

import { filteredRank, rankMetrics, recallAtK, percentile, latency, mulberry32 } from '../bench/lib/metrics.mjs';
import {
  generateSynthetic, serializeKg, loadSynthetic, assertDisjoint, verifyFixtureHashes,
  hashSplits, allTagged, entitySet, BENCH_DIR, FIXTURE_DIR, SYNTHETIC_PATH,
} from '../bench/lib/fixture.mjs';
import { detectSymmetricRelations, detectInversePairs, generateSymmetryDecoys, pickTargets } from '../bench/lib/adversarial.mjs';
import { evaluateGates, loadGates } from '../bench/lib/gates.mjs';
import { main } from '../bench/run.mjs';
import { runChecks } from '../scripts/check-security.mjs';

const require = createRequire(import.meta.url);
const HERE = dirname(fileURLToPath(import.meta.url));
const FAKE = require('./fixtures/fake-binding.cjs');

// A binding whose Model injects a dev knob (e.g. `_annDegraded`) regardless of
// the config the harness builds — so a single test can drive one gate to FAIL.
function knobbed(knob) {
  return {
    Model: class extends FAKE.Model {
      constructor(cfgJson) {
        const c = JSON.parse(cfgJson || '{}');
        c[knob] = true;
        super(JSON.stringify(c));
      }
    },
    version: FAKE.version,
    backend: 'fake',
  };
}

// ---------------------------------------------------------------------------
// metrics — hand-computed values
// ---------------------------------------------------------------------------

test('filteredRank: TOP / BOTTOM / RANDOM over a tied block', () => {
  // true=B(5); A(9) strictly higher; B,C,D all tied at 5. filter removes D.
  const cands = [{ entity: 'A', score: 9 }, { entity: 'B', score: 5 }, { entity: 'C', score: 5 }, { entity: 'D', score: 5 }];
  // filtered: A higher (1), C tied (1). tied block size after filter = {B,C} = 2.
  assert.equal(filteredRank(cands, 'B', { filter: ['D'], tieBreak: 'top' }), 2); // higher+1
  assert.equal(filteredRank(cands, 'B', { filter: ['D'], tieBreak: 'bottom' }), 3); // higher+tied+1
  // RANDOM: rank in {2,3}; mean over many seeds ≈ 2.5
  let sum = 0;
  const N = 4000;
  for (let i = 0; i < N; i++) sum += filteredRank(cands, 'B', { filter: ['D'], tieBreak: 'random', rng: mulberry32(i) });
  assert.ok(Math.abs(sum / N - 2.5) < 0.1, `mean random rank ${sum / N}`);
});

test('filteredRank: no ties, unique scores', () => {
  const cands = [{ entity: 'A', score: 3 }, { entity: 'B', score: 2 }, { entity: 'C', score: 1 }];
  assert.equal(filteredRank(cands, 'A', { tieBreak: 'random' }), 1);
  assert.equal(filteredRank(cands, 'C', { tieBreak: 'top' }), 3);
});

test('rankMetrics: MRR / MR / Hits hand-computed', () => {
  const m = rankMetrics([1, 2, 4, 10, 11]); // 1/1+1/2+1/4+1/10+1/11 = 1.8409...
  assert.ok(Math.abs(m.mrr - (1 + 0.5 + 0.25 + 0.1 + 1 / 11) / 5) < 1e-12);
  assert.equal(m.mr, (1 + 2 + 4 + 10 + 11) / 5);
  assert.equal(m.hits[1], 1 / 5);
  assert.equal(m.hits[3], 2 / 5); // ranks 1,2 ≤ 3
  assert.equal(m.hits[10], 4 / 5); // ranks 1,2,4,10 ≤ 10
});

test('recallAtK: overlap of two ranked lists', () => {
  assert.equal(recallAtK(['a', 'b', 'c', 'd'], ['a', 'b', 'x', 'y'], 4), 0.5);
  assert.equal(recallAtK(['a', 'b'], ['a', 'b'], 2), 1);
  assert.equal(recallAtK(['z'], ['a'], 1), 0);
});

test('percentile + latency', () => {
  assert.equal(percentile([1, 2, 3, 4], 50), 2.5);
  const l = latency([5, 1, 3]);
  assert.equal(l.n, 3);
  assert.equal(l.p50, 3);
});

// ---------------------------------------------------------------------------
// fixture: generation determinism, hashes, disjointness
// ---------------------------------------------------------------------------

test('synthetic fixture regenerates byte-identically to the committed file', () => {
  const regenerated = serializeKg(generateSynthetic());
  const committed = readFileSync(SYNTHETIC_PATH, 'utf8');
  assert.equal(regenerated, committed, 'generator drifted from the committed fixture bytes');
});

test('synthetic splits load, are disjoint, and cover every triple once', () => {
  const kg = loadSynthetic();
  const total = Object.values(kg.counts).reduce((a, b) => a + b, 0);
  assert.equal(assertDisjoint(kg.splits).total, total);
  assert.ok(kg.counts.test > 0 && kg.counts.valid > 0 && kg.counts.transfer > 0);
  assert.equal(kg.nEntities, 300);
});

test('every eval-split entity is seen in train (filtered ranking is meaningful)', () => {
  const kg = loadSynthetic();
  const trained = entitySet(kg.splits.train.map((t) => [t.s, t.r, t.o]));
  for (const s of ['valid', 'transfer', 'test']) {
    for (const t of kg.splits[s]) assert.ok(trained.has(t.s) && trained.has(t.o), `unseen entity in ${s}`);
  }
});

test('fixture hash mismatch → refuses to run', () => {
  const dir = mkdtempSync(join(tmpdir(), 'kge-hash-'));
  try {
    writeFileSync(join(dir, 'HASHES.json'), JSON.stringify({ files: { 'fixtures/synthetic-kg.json': 'deadbeef' } }));
    assert.throws(() => verifyFixtureHashes({ benchDir: BENCH_DIR, fixtureDir: dir }), /hash mismatch/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});

// ---------------------------------------------------------------------------
// adversarial: symmetry detection + decoys
// ---------------------------------------------------------------------------

test('adversarial detects the synthetic symmetric relation and the inverse pair', () => {
  const kg = loadSynthetic();
  const all = allTagged(kg.splits).map((t) => ({ s: t.s, r: t.r, o: t.o }));
  const sym = detectSymmetricRelations(all).filter((r) => r.symmetric);
  assert.equal(sym.length, 1, 'exactly one symmetric relation');
  assert.ok(sym[0].fraction >= 0.99);
  const inv = detectInversePairs(all);
  assert.ok(inv.length >= 1, 'inverse pair detected');
  // directed relations are NOT flagged symmetric
  const symRels = new Set(sym.map((r) => r.relation));
  assert.ok(!symRels.has('works_with') && !symRels.has('type_of'));
});

test('symmetry decoys are new triples disjoint from the graph', () => {
  const kg = loadSynthetic();
  const all = allTagged(kg.splits).map((t) => ({ s: t.s, r: t.r, o: t.o }));
  const sym = detectSymmetricRelations(all);
  const targets = pickTargets(kg.splits.test, sym, 10);
  assert.ok(targets.length > 0);
  const decoys = generateSymmetryDecoys(targets, kg.entities, all, { perTarget: 3 });
  assert.ok(decoys.length > 0);
  const present = new Set(all.map((t) => `${t.s} ${t.r} ${t.o}`));
  for (const d of decoys) assert.ok(!present.has(`${d.s} ${d.r} ${d.o}`), 'decoy already in graph');
});

// ---------------------------------------------------------------------------
// run.mjs end-to-end with the fake binding
// ---------------------------------------------------------------------------

test('run.mjs e2e: fake binding → well-formed receipt; --gate --report-only exits 0', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'kge-run-')), 'receipt.json');
  const { code, receipt } = await main(
    ['--suite', 'synthetic', '--tie-check', '--ann', '--adversarial', '--gate', '--report-only', '--out', out],
    { binding: FAKE },
  );
  assert.equal(code, 0, 'report-only never fails the exit code');
  assert.equal(receipt.schema, 'ruvector-kge-bench/receipt@1');
  assert.ok(receipt.dataset.splits_hash && receipt.dataset.hashes['fixtures/synthetic-kg.json']);
  assert.ok(receipt.metrics.test.mrr > 0 && receipt.metrics.test.hits);
  assert.ok(receipt.ann.recall_at_10 >= 0 && receipt.adversarial.drop !== undefined);
  assert.ok(receipt.tie_check.available && Array.isArray(receipt.gates.rows));
  assert.equal(receipt.binding.backend, 'fake');
  // no entity id or relation name anywhere in the receipt (ADR-005)
  const blob = JSON.stringify(receipt);
  const kg = loadSynthetic();
  assert.ok(!blob.includes(`"${kg.splits.test[0].s}"`), 'receipt must not embed entity ids');
  assert.ok(!blob.includes('sibling_of'), 'receipt must not embed relation names');
});

test('run.mjs --gate (strict) exits non-zero when the ANN recall gate fails', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'kge-run-')), 'receipt.json');
  const { code, receipt } = await main(
    ['--suite', 'synthetic', '--ann', '--gate', '--out', out],
    { binding: knobbed('_annDegraded') },
  );
  assert.equal(code, 1, 'a failing ANN recall gate must fail the run in strict mode');
  const row = receipt.gates.rows.find((r) => r.name === 'ann_recall10');
  assert.equal(row.status, 'FAIL');
});

test('run.mjs --gate (strict) exits non-zero when RANDOM tie-break is wrong', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'kge-run-')), 'receipt.json');
  const { code, receipt } = await main(
    ['--suite', 'synthetic', '--tie-check', '--gate', '--out', out],
    { binding: knobbed('_badTieBreak') },
  );
  assert.equal(code, 1, 'a TOP-biased tie-break must fail the tie_break gate');
  const row = receipt.gates.rows.find((r) => r.name === 'tie_break_random');
  assert.equal(row.status, 'FAIL');
});

test('run.mjs reports "engine unavailable" without crashing when the Model errors', async () => {
  const out = join(mkdtempSync(join(tmpdir(), 'kge-run-')), 'receipt.json');
  const notImpl = {
    Model: class {
      addTriplesJson() { return JSON.stringify({ added: 0 }); }
      trainJson() { return JSON.stringify({ epochs: 0 }); }
      evalJson() { return JSON.stringify({ error: { kind: 'invalid', message: 'evaluator not implemented yet' } }); }
    },
    version: () => '0',
    backend: 'stub',
  };
  const { code, receipt } = await main(['--suite', 'synthetic', '--gate', '--report-only', '--out', out], { binding: notImpl });
  assert.equal(code, 0);
  assert.ok(receipt.binding.unavailable);
  assert.match(receipt.binding.error, /not implemented/);
});

// ---------------------------------------------------------------------------
// gate evaluator: latency gate activates from the spike, tie-break, SKIP logic
// ---------------------------------------------------------------------------

test('latency gate: PASS under native threshold, FAIL over it, SKIP off-backend', () => {
  const gates = loadGates(); // the committed spike file exists, so the gate is live
  const base = { suite: 'synthetic', engineAvailable: true, hasAnn: true };
  const pass = evaluateGates({}, { ...base, backend: 'native', latencyP95: 5 }, gates).rows.find((r) => r.name === 'latency_predict_p95');
  assert.equal(pass.status, 'PASS');
  const fail = evaluateGates({}, { ...base, backend: 'native', latencyP95: 999 }, gates).rows.find((r) => r.name === 'latency_predict_p95');
  assert.equal(fail.status, 'FAIL');
  const wasmSkip = evaluateGates({}, { ...base, backend: 'wasm', latencyP95: 5 }, gates).rows.find((r) => r.name === 'latency_predict_p95');
  assert.equal(wasmSkip.status, 'SKIP'); // wasm threshold still pending
  const noLat = evaluateGates({}, { ...base, backend: 'native' }, gates).rows.find((r) => r.name === 'latency_predict_p95');
  assert.equal(noLat.status, 'SKIP'); // no --ann latency measured
});

// ---------------------------------------------------------------------------
// check-security.mjs
// ---------------------------------------------------------------------------

test('check-security passes on the current tree', () => {
  const res = runChecks();
  assert.equal(res.ok, true, JSON.stringify(res.violations));
  assert.ok(res.scanned.js_files > 0);
});

test('check-security fails on a temp file using child_process', () => {
  const root = mkdtempSync(join(tmpdir(), 'kge-sec-'));
  try {
    mkdirSync(join(root, 'src'));
    writeFileSync(join(root, 'src', 'bad.js'), "const cp = require('child_process'); cp.execSync('ls');\n");
    const res = runChecks({ root });
    assert.equal(res.ok, false);
    assert.ok(res.violations.some((v) => /child_process/.test(v.pattern)));
  } finally {
    rmSync(root, { recursive: true, force: true });
  }
});
