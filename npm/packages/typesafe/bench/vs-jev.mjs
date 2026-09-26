#!/usr/bin/env node
// Paired sequential comparison of a local run against the frozen Jev rows
// (ADR-007 §1b). For each question (department / urgent / frustration) and each
// Jev reference (`test_rows.baseline`, `test_rows.champion` in
// jev-baseline-2026-09-21.json) it runs the campaign's anytime-valid paired
// test (lib/paired.mjs — the Rust PairedSequentialTest, α = 0.05, λ = 0.5,
// fixed, never tuned) twice over the 150 test items:
//   forward — local is the challenger: rejection ⇒ local is superior;
//   reverse — Jev is the challenger: rejection ⇒ Jev is superior.
// Tier per (question, reference): superior (forward rejects) > non-inferior
// (reverse does not reject AND local accuracy ≥ Jev accuracy) > loses.
//
// Pair order is pre-registered: test ids in lexical order (JS default string
// sort, UTF-16 code units — e.g. "t1000" < "t451"). Rejection latches on the
// running maximum of wealth, so order matters and is never chosen post hoc.
//
// Every tier is reported on three slices: full (150), novel (119 items not
// composed entirely of train-pool sentences) and template (the 31 that are),
// from the frozen, hash-pinned bench/fixtures/novel-slice-2026-09-26.json
// (recomputed here and required to match).
//
// Usage:
//   node bench/vs-jev.mjs --receipt bench/results/<run>.json [--out PATH] [--strict]
//   node bench/vs-jev.mjs --records <records.json from run.mjs --emit-records> [--out PATH]
// --strict exits 1 unless the primary claim (superior to Jev baseline on all
// three questions, full slice) holds.

import { createHash } from 'node:crypto';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { dirname, isAbsolute, join } from 'node:path';
import { loadTickets, loadDerivedFixture, BENCH_DIR, FIXTURE_DIR } from './lib/fixture.mjs';
import { computeNovelSlice, NOVEL_SLICE_FILE } from './lib/novel-slice.mjs';
import { PairedSequentialTest } from './lib/paired.mjs';

export const QUESTIONS = ['department', 'urgent', 'frustration'];
export const REFERENCES = ['baseline', 'champion'];
export const ALPHA = 0.05;
export const LAMBDA = 0.5;
const JEV_FILE = 'jev-baseline-2026-09-21.json';

function parseArgs(argv) {
  const a = { receipt: undefined, records: undefined, out: undefined, strict: false };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--receipt') a.receipt = argv[++i];
    else if (t === '--records') a.records = argv[++i];
    else if (t === '--out') a.out = argv[++i];
    else if (t === '--strict') a.strict = true;
    else throw new Error(`unknown flag: ${t}`);
  }
  if (!a.receipt === !a.records) throw new Error('pass exactly one of --receipt PATH or --records PATH');
  return a;
}

/** Extract the local test records from a run receipt or an --emit-records file. */
export function recordsFrom(doc, kind) {
  if (kind === 'receipt') {
    if (doc.suite !== 'tickets') throw new Error(`receipt suite is '${doc.suite}', vs-jev needs tickets`);
    if (doc.no_test) throw new Error('receipt is a --no-test run: it has no test records');
    if (doc.limit !== null && doc.limit !== undefined) throw new Error(`receipt used --limit ${doc.limit}; vs-jev needs the full test split`);
    const recs = doc.item_records?.local?.test;
    if (!Array.isArray(recs)) throw new Error('receipt has no item_records.local.test (re-run bench/run.mjs with the local arm)');
    return { records: recs, splitsHash: doc.fixtures?.splits_hash ?? null, embedderModel: doc.embedder_model ?? null };
  }
  if (doc.schema !== 'ruvector-typesafe-bench/item-records@1') throw new Error(`unexpected records schema ${doc.schema}`);
  if (doc.suite !== 'tickets' || doc.split !== 'test') throw new Error('records must be tickets/test');
  if (doc.limit !== null && doc.limit !== undefined) throw new Error(`records used --limit ${doc.limit}; vs-jev needs the full test split`);
  return { records: doc.records, splitsHash: doc.splits_hash ?? null, embedderModel: doc.embedder_model ?? null };
}

/** id → { question: correct } after checking exact coverage of the test ids. */
function correctnessById(records, testIds, source) {
  const want = new Set(testIds);
  const byId = new Map();
  for (const r of records) {
    if (!want.has(r.id)) throw new Error(`${source}: id ${r.id} is not a tickets test id`);
    if (byId.has(r.id)) throw new Error(`${source}: duplicate id ${r.id}`);
    const c = {};
    for (const q of QUESTIONS) {
      if (typeof r.correct?.[q] !== 'boolean') throw new Error(`${source}: id ${r.id} has no boolean correct.${q}`);
      c[q] = r.correct[q];
    }
    byId.set(r.id, c);
  }
  const missing = testIds.filter((id) => !byId.has(id));
  if (missing.length) throw new Error(`${source}: ${missing.length} test id(s) missing (e.g. ${missing.slice(0, 3).join(', ')})`);
  return byId;
}

function tierFor(ids, local, ref, q) {
  const forward = PairedSequentialTest.standard();
  const reverse = PairedSequentialTest.standard();
  let localRight = 0;
  let refRight = 0;
  for (const id of ids) {
    const l = local.get(id)[q];
    const r = ref.get(id)[q];
    forward.update(r, l); // (baseline_correct, champion_correct): local challenges Jev
    reverse.update(l, r); // Jev challenges local
    localRight += l ? 1 : 0;
    refRight += r ? 1 : 0;
  }
  const tier = forward.rejected ? 'superior' : !reverse.rejected && localRight >= refRight ? 'non-inferior' : 'loses';
  return {
    tier,
    n: ids.length,
    local_correct: localRight,
    reference_correct: refRight,
    local_accuracy: ids.length ? localRight / ids.length : null,
    reference_accuracy: ids.length ? refRight / ids.length : null,
    forward: forward.statistic(),
    reverse: reverse.statistic(),
  };
}

/**
 * Pure comparison. `records`: local test item records; `jevBaseline`: parsed
 * frozen Jev JSON; `testIds`: the fixture's test ids; `slices`: { name: ids[] }.
 */
export function compareToJev({ records, jevBaseline, testIds, slices }) {
  const local = correctnessById(records, testIds, 'local records');
  const refs = {};
  for (const ref of REFERENCES) {
    const rows = jevBaseline.test_rows?.[ref];
    if (!Array.isArray(rows)) throw new Error(`jev baseline has no test_rows.${ref}`);
    refs[ref] = correctnessById(rows.filter((r) => r.ok), testIds, `jev ${ref}`);
  }
  const results = {};
  for (const [slice, sliceIds] of Object.entries(slices)) {
    const ids = [...sliceIds].sort(); // pre-registered lexical order
    results[slice] = {};
    for (const q of QUESTIONS) {
      results[slice][q] = {};
      for (const ref of REFERENCES) results[slice][q][ref] = tierFor(ids, local, refs[ref], q);
    }
  }
  const verdicts = (slice) => ({
    superior_to_baseline: Object.fromEntries(QUESTIONS.map((q) => [q, results[slice][q].baseline.tier === 'superior'])),
    non_inferior_to_champion: Object.fromEntries(QUESTIONS.map((q) => [q, results[slice][q].champion.tier !== 'loses'])),
  });
  const claims = {};
  for (const slice of Object.keys(slices)) {
    const v = verdicts(slice);
    claims[slice] = {
      ...v,
      primary_holds: QUESTIONS.every((q) => v.superior_to_baseline[q]),
      secondary_holds: QUESTIONS.every((q) => v.non_inferior_to_champion[q]),
    };
  }
  const inGeneratorOnly = claims.full && claims.novel ? claims.full.primary_holds && !claims.novel.primary_holds : null;
  return { results, claims, in_generator_only: inGeneratorOnly };
}

/** The frozen slice, re-derived from the fixture and required to match. */
export function loadSlices(tickets) {
  const frozen = loadDerivedFixture(NOVEL_SLICE_FILE, { benchDir: BENCH_DIR, fixtureDir: FIXTURE_DIR });
  const live = computeNovelSlice(tickets);
  const same = (a, b) => a.length === b.length && a.every((x, i) => x === b[i]);
  if (!same(frozen.novel_ids, live.novel_ids) || !same(frozen.template_ids, live.template_ids)) {
    throw new Error(`${NOVEL_SLICE_FILE} disagrees with the slice recomputed from the fixture — refusing`);
  }
  return { full: tickets.bySplit.test.map((it) => it.id), novel: frozen.novel_ids, template: frozen.template_ids };
}

const sha256 = (buf) => createHash('sha256').update(buf).digest('hex');
const pct = (x) => (x === null ? 'n/a' : `${(x * 100).toFixed(1)}%`);

function printReport(out) {
  for (const [slice, byQ] of Object.entries(out.results)) {
    const n = byQ.department.baseline.n;
    console.log(`\n## vs Jev — ${slice} slice (n=${n}, lexical id order, α=${ALPHA}, λ=${LAMBDA})`);
    console.log('| question | reference | local | jev | tier | fwd maxW (W/L) | rev maxW (W/L) |');
    console.log('|---|---|---|---|---|---|---|');
    for (const q of QUESTIONS) {
      for (const ref of REFERENCES) {
        const r = byQ[q][ref];
        const f = r.forward;
        const b = r.reverse;
        console.log(`| ${q} | ${ref} | ${pct(r.local_accuracy)} | ${pct(r.reference_accuracy)} | **${r.tier}** | ` +
          `${f.max_wealth.toFixed(2)} (${f.n_champion_wins}/${f.n_baseline_wins}) | ${b.max_wealth.toFixed(2)} (${b.n_champion_wins}/${b.n_baseline_wins}) |`);
      }
    }
    const c = out.claims[slice];
    console.log(`primary (superior to Jev baseline, all three): ${c.primary_holds ? 'HOLDS' : 'not met'} · ` +
      `secondary (non-inferior to Jev champion, all three): ${c.secondary_holds ? 'HOLDS' : 'not met'}`);
  }
  if (out.in_generator_only) console.log('\n**in-generator only**: the primary claim holds on the full split but not on the novel slice.');
}

export function main(argv) {
  const args = parseArgs(argv);
  const tickets = loadTickets(); // verifies every frozen fixture hash first
  const jevBytes = readFileSync(join(BENCH_DIR, JEV_FILE));
  const jevBaseline = JSON.parse(jevBytes.toString('utf8'));
  const kind = args.receipt ? 'receipt' : 'records';
  const srcPath = args.receipt ?? args.records;
  const srcBytes = readFileSync(srcPath);
  const src = recordsFrom(JSON.parse(srcBytes.toString('utf8')), kind);
  if (src.splitsHash && src.splitsHash !== tickets.splitsHash) {
    throw new Error(`input splits_hash ${src.splitsHash} ≠ fixture ${tickets.splitsHash}`);
  }
  const slices = loadSlices(tickets);
  const cmp = compareToJev({ records: src.records, jevBaseline, testIds: slices.full, slices });
  const out = {
    schema: 'ruvector-typesafe-bench/vs-jev@1',
    generated_at: new Date().toISOString(),
    protocol: {
      test: 'PairedSequentialTest (crates/ruvector-typesafe-core loop_gate/sequential.rs), JS port bench/lib/paired.mjs',
      alpha: ALPHA,
      lambda: LAMBDA,
      threshold: PairedSequentialTest.standard().threshold,
      order: 'test ids sorted lexically (JS default string sort)',
      tiers: 'superior: forward rejects; non-inferior: reverse does not reject and local acc >= reference acc; else loses',
    },
    inputs: {
      source: kind,
      source_sha256: sha256(srcBytes),
      jev_baseline_sha256: sha256(jevBytes),
      novel_slice_file: NOVEL_SLICE_FILE,
      splits_hash: tickets.splitsHash,
      embedder_model: src.embedderModel,
    },
    slices: Object.fromEntries(Object.entries(slices).map(([k, v]) => [k, { n: v.length }])),
    template_ids: slices.template,
    ...cmp,
  };
  printReport(out);
  if (args.out) {
    const p = isAbsolute(args.out) ? args.out : join(process.cwd(), args.out);
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, JSON.stringify(out, null, 2) + '\n');
    console.log(`\nvs-jev → ${p}`);
  }
  return { code: args.strict && !out.claims.full.primary_holds ? 1 : 0, out };
}

if (import.meta.url === `file://${process.argv[1]}`) {
  try {
    process.exit(main(process.argv.slice(2)).code);
  } catch (e) {
    console.error(`vs-jev failed: ${e && e.message ? e.message : e}`);
    process.exit(2);
  }
}
