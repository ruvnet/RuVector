// OpenJev v0 statistics layer (ADR-007 §1b, §2): the shared norm() contract,
// the JS port of the Rust PairedSequentialTest, the frozen novel-composition
// slice, and the vs-jev tiering. No network, no binding.

import { test } from 'node:test';
import assert from 'node:assert/strict';
import { mkdtempSync, writeFileSync, readFileSync, rmSync, mkdirSync, copyFileSync } from 'node:fs';
import { tmpdir } from 'node:os';
import { join } from 'node:path';

import { norm, sha256Norm } from '../lib/norm.mjs';
import { PairedSequentialTest } from '../lib/paired.mjs';
import { loadTickets, loadDerivedFixture, BENCH_DIR, FIXTURE_DIR } from '../lib/fixture.mjs';
import { computeNovelSlice, novelSliceDocument, sentences, NOVEL_SLICE_FILE } from '../lib/novel-slice.mjs';
import { compareToJev, loadSlices, recordsFrom, main as vsJevMain } from '../vs-jev.mjs';

const HERE = new URL('.', import.meta.url).pathname;
const readJson = (p) => JSON.parse(readFileSync(p, 'utf8'));
const JEV = readJson(join(BENCH_DIR, 'jev-baseline-2026-09-21.json'));

// ---------------------------------------------------------------------------
// norm — golden vectors shared with the Rust trainer
// ---------------------------------------------------------------------------

test('norm golden vectors: every case reproduces norm and sha256 exactly', () => {
  const golden = readJson(join(HERE, 'norm-golden.json'));
  assert.ok(golden.cases.length >= 40, `need >= 40 golden cases, have ${golden.cases.length}`);
  for (const c of golden.cases) {
    assert.equal(norm(c.input), c.norm, `norm(${c.name})`);
    assert.equal(sha256Norm(c.input), c.sha256, `sha256Norm(${c.name})`);
  }
  // The discriminating cases the Rust side must not get wrong.
  const byName = Object.fromEntries(golden.cases.map((c) => [c.name, c.norm]));
  assert.equal(byName['devanagari-vowel-sign'], 'ह न द', 'Mn/Mc marks are separators (General_Category, not Alphabetic)');
  assert.equal(byName['turkish-dotted-I'], 'i stanbul', 'U+0307 from lowercasing İ is a separator');
  assert.equal(byName['greek-final-sigma'], '\u03bf\u03b4\u03bf\u03c2', 'final sigma is context-sensitive');
  assert.equal(byName['fullwidth-digits'], '123');
  assert.equal(byName['apostrophe-cant'], 'can t');
});

test('norm fails closed on a non-string', () => {
  assert.throws(() => norm(undefined), TypeError);
  assert.throws(() => sha256Norm(42), TypeError);
});

// ---------------------------------------------------------------------------
// paired — identical to the Rust PairedSequentialTest
// ---------------------------------------------------------------------------

test('paired: standard threshold is 1/f32(0.05), as in the 2026-09-21 campaign receipt', () => {
  const t = PairedSequentialTest.standard();
  assert.equal(t.threshold, 19.99999970197678);
  const campaign = readJson(join(BENCH_DIR, 'results', 'optimize-tickets-2026-09-21.json'));
  assert.equal(t.threshold, campaign.arms[0].best_stat.threshold);
});

test('paired: reproduces the Rust wealth path of every golden case exactly', () => {
  const golden = readJson(join(HERE, 'paired-golden.json'));
  assert.ok(golden.cases.length >= 8);
  for (const c of golden.cases) {
    const t = new PairedSequentialTest(c.alpha, c.lambda);
    c.pairs.forEach(([b, ch], i) => {
      t.update(b, ch);
      const [w, mw, rej] = c.wealth_path[i];
      assert.equal(t.wealth, w, `${c.name} step ${i} wealth`);
      assert.equal(t.maxWealth, mw, `${c.name} step ${i} max_wealth`);
      assert.equal(t.rejected, rej, `${c.name} step ${i} rejected`);
    });
    const s = t.statistic();
    const r = c.statistic;
    assert.equal(s.alpha, Math.fround(r.alpha), `${c.name} alpha`);
    assert.equal(s.lambda, Math.fround(r.lambda), `${c.name} lambda`);
    for (const k of ['wealth', 'max_wealth', 'threshold', 'n_champion_wins', 'n_baseline_wins', 'rejected']) {
      assert.equal(s[k], r[k], `${c.name} ${k}`);
    }
    assert.equal(s.n_discordant_at_rejection, r.n_discordant_at_rejection, `${c.name} n_discordant_at_rejection`);
  }
});

test('paired: ADR-007 power arithmetic — 14 straight wins reject, 13 do not; 18 wins + 3 losses reject', () => {
  const run = (pairs) => PairedSequentialTest.standard().updateAll(pairs);
  assert.equal(run(Array(14).fill([false, true])), true);
  assert.equal(run(Array(13).fill([false, true])), false);
  assert.equal(run([...Array(3).fill([true, false]), ...Array(18).fill([false, true])]), true);
  assert.equal(run(Array(50).fill([true, true])), false, 'concordant pairs carry no information');
});

// ---------------------------------------------------------------------------
// novel-composition slice
// ---------------------------------------------------------------------------

test('novel slice: 31 template / 119 novel, frozen file matches the fixture and its pin', () => {
  const tickets = loadTickets();
  const live = computeNovelSlice(tickets);
  assert.equal(live.pool_rows, 136);
  assert.equal(live.template_ids.length, 31);
  assert.equal(live.novel_ids.length, 119);
  assert.equal(live.share_any, 141);
  const frozen = loadDerivedFixture(NOVEL_SLICE_FILE);
  assert.deepEqual(frozen, JSON.parse(JSON.stringify(novelSliceDocument(tickets))));
  const slices = loadSlices(tickets);
  assert.equal(slices.full.length, 150);
  assert.equal(new Set([...slices.novel, ...slices.template]).size, 150);
  assert.deepEqual(sentences('Hi. Two words? This has three tokens: and so does this one'), [
    'this has three tokens', 'and so does this one',
  ]);
});

test('derived fixture with a wrong pin is refused', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-derived-'));
  try {
    mkdirSync(join(dir, 'fixtures'));
    copyFileSync(join(BENCH_DIR, NOVEL_SLICE_FILE), join(dir, NOVEL_SLICE_FILE));
    writeFileSync(join(dir, 'fixtures', 'HASHES.json'), JSON.stringify({ files: {}, derived: { [NOVEL_SLICE_FILE]: '0'.repeat(64) } }));
    assert.throws(() => loadDerivedFixture(NOVEL_SLICE_FILE, { benchDir: dir, fixtureDir: join(dir, 'fixtures') }), /hash mismatch/);
    writeFileSync(join(dir, 'fixtures', 'HASHES.json'), JSON.stringify({ files: {} }));
    assert.throws(() => loadDerivedFixture(NOVEL_SLICE_FILE, { benchDir: dir, fixtureDir: join(dir, 'fixtures') }), /no derived pin/);
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
  assert.ok(FIXTURE_DIR);
});

// ---------------------------------------------------------------------------
// vs-jev tiers
// ---------------------------------------------------------------------------

const testIds = () => loadTickets().bySplit.test.map((it) => it.id);

/** A Jev baseline whose rows are correct everywhere except `wrongIds`. */
function syntheticJev(ids, wrong = { baseline: [], champion: [] }) {
  const rows = (ref) => ids.map((id) => {
    const ok = !wrong[ref].includes(id);
    return { id, ok: true, correct: { department: ok, urgent: ok, frustration: ok } };
  });
  return { test_rows: { baseline: rows('baseline'), champion: rows('champion') } };
}
const localRecords = (ids, wrongIds = []) => ids.map((id) => {
  const ok = !wrongIds.includes(id);
  return { id, correct: { department: ok, urgent: ok, frustration: ok } };
});

test('vs-jev: all-win → superior; all-tie → non-inferior; 3 losses + 18 wins → superior; 13 wins → non-inferior', () => {
  const ids = testIds();
  const sorted = [...ids].sort();
  const slices = { full: ids };
  // all-win: Jev wrong on 20, local right everywhere
  let r = compareToJev({ records: localRecords(ids), jevBaseline: syntheticJev(ids, { baseline: sorted.slice(0, 20), champion: sorted.slice(0, 20) }), testIds: ids, slices });
  assert.equal(r.results.full.department.baseline.tier, 'superior');
  assert.equal(r.claims.full.primary_holds, true);
  assert.equal(r.claims.full.secondary_holds, true);
  // all-tie
  r = compareToJev({ records: localRecords(ids), jevBaseline: syntheticJev(ids), testIds: ids, slices });
  assert.equal(r.results.full.urgent.baseline.tier, 'non-inferior');
  assert.equal(r.results.full.urgent.baseline.forward.wealth, 1);
  assert.equal(r.claims.full.primary_holds, false);
  assert.equal(r.claims.full.secondary_holds, true);
  // 18 wins (Jev wrong, local right) + 3 losses (local wrong, Jev right)
  r = compareToJev({
    records: localRecords(ids, sorted.slice(100, 103)),
    jevBaseline: syntheticJev(ids, { baseline: sorted.slice(0, 18), champion: [] }),
    testIds: ids,
    slices,
  });
  const d = r.results.full.department.baseline;
  assert.equal(d.forward.n_champion_wins, 18);
  assert.equal(d.forward.n_baseline_wins, 3);
  assert.equal(d.tier, 'superior');
  // vs champion (never wrong) the same local run has 3 losses → loses
  assert.equal(r.results.full.department.champion.tier, 'loses');
  // 13 straight wins: not enough for superior, still non-inferior
  r = compareToJev({ records: localRecords(ids), jevBaseline: syntheticJev(ids, { baseline: sorted.slice(0, 13), champion: [] }), testIds: ids, slices });
  assert.equal(r.results.full.frustration.baseline.tier, 'non-inferior');
  // Jev better by 20 → loses
  r = compareToJev({ records: localRecords(ids, sorted.slice(0, 20)), jevBaseline: syntheticJev(ids), testIds: ids, slices });
  assert.equal(r.results.full.department.baseline.tier, 'loses');
  assert.equal(r.results.full.department.baseline.reverse.rejected, true);
});

test('vs-jev: in-generator-only flag when the primary claim holds on full but not on novel', () => {
  const ids = testIds();
  const tickets = loadTickets();
  const slices = loadSlices(tickets);
  // Jev wrong only on template items + enough full-slice wins to reject on full.
  const jevWrong = [...slices.template].slice(0, 20);
  const r = compareToJev({ records: localRecords(ids), jevBaseline: syntheticJev(ids, { baseline: jevWrong, champion: [] }), testIds: ids, slices });
  assert.equal(r.claims.full.primary_holds, true);
  assert.equal(r.claims.novel.primary_holds, false);
  assert.equal(r.in_generator_only, true);
});

test('vs-jev: refuses incomplete, duplicated, foreign or limited inputs', () => {
  const ids = testIds();
  const slices = { full: ids };
  const jev = syntheticJev(ids);
  assert.throws(() => compareToJev({ records: localRecords(ids.slice(1)), jevBaseline: jev, testIds: ids, slices }), /missing/);
  assert.throws(() => compareToJev({ records: [...localRecords(ids), localRecords(ids)[0]], jevBaseline: jev, testIds: ids, slices }), /duplicate/);
  assert.throws(() => compareToJev({ records: [...localRecords(ids), { id: 'zz', correct: {} }], jevBaseline: jev, testIds: ids, slices }), /not a tickets test id/);
  assert.throws(() => recordsFrom({ suite: 'tickets', limit: 60, item_records: { local: { test: [] } } }, 'receipt'), /--limit/);
  assert.throws(() => recordsFrom({ suite: 'tickets', no_test: true, limit: null }, 'receipt'), /--no-test/);
  assert.throws(() => recordsFrom({ schema: 'x' }, 'records'), /schema/);
});

test('vs-jev CLI on the frozen Jev champion rows as challenger vs the baseline (sanity)', () => {
  const dir = mkdtempSync(join(tmpdir(), 'ts-vsjev-'));
  try {
    const records = JEV.test_rows.champion.map((r) => ({ id: r.id, correct: r.correct }));
    const recPath = join(dir, 'records.json');
    writeFileSync(recPath, JSON.stringify({ schema: 'ruvector-typesafe-bench/item-records@1', suite: 'tickets', split: 'test', limit: null, records }));
    const { code, out } = vsJevMain(['--records', recPath, '--out', join(dir, 'vs.json')]);
    assert.equal(code, 0);
    const dept = out.results.full.department.baseline;
    // 135 vs 128 with 10 wins / 3 losses: max wealth 5.24 < 20 → not superior.
    assert.equal(dept.local_correct, 135);
    assert.equal(dept.reference_correct, 128);
    assert.equal(dept.forward.n_champion_wins, 10);
    assert.equal(dept.forward.n_baseline_wins, 3);
    assert.equal(dept.tier, 'non-inferior');
    // Against itself every pair is concordant.
    const self = out.results.full.department.champion;
    assert.equal(self.forward.wealth, 1);
    assert.equal(self.tier, 'non-inferior');
    assert.equal(out.slices.novel.n, 119);
    assert.equal(out.template_ids.length, 31);
    assert.ok(readJson(join(dir, 'vs.json')).claims.full);
    assert.equal(vsJevMain(['--records', recPath, '--strict']).code, 1, '--strict fails when the primary claim does not hold');
  } finally {
    rmSync(dir, { recursive: true, force: true });
  }
});
