#!/usr/bin/env node
// typesafe benchmark harness (ADR-006). Scores one or more suites with the jev
// (frozen replay) and/or local (binding) arms, writes one receipt per run, and
// optionally evaluates the release gates.
//
// Usage:
//   node bench/run.mjs [--suite tickets|banking77|clinc150|hwu64|all]
//                      [--arm jev|local|both] [--embedder hash|onnx]
//                      [--shots 8] [--zero-shot] [--limit N]
//                      [--out results/<name>-<date>.json]
//                      [--gate] [--report-only] [--baseline-receipt PATH]
//                      [--model NAME] [--model-dir DIR] [--manifest PATH]
//                      [--train-hashes PATH] [--no-test] [--emit-records PATH]
//
// ONNX arms resolve to ONE sha256-verified manifest entry (lib/model-dir.mjs);
// --model-dir may point at a staged candidate or a bare unpublished dir. A model
// that ships train-text-hashes.txt (or --train-hashes) is leakage-checked
// against the suite's held-out rows before anything is scored (ADR-007 §2
// Assertion B); a hit refuses the run (exit 3). --no-test scores validation +
// transfer only and never replays or scores the test split (selection runs).
// Local-arm receipts carry per-item records (item_records) for paired tests;
// --emit-records also writes the test records to their own file (bench/vs-jev.mjs).
//
// The harness never crashes on an unavailable engine: the local arm reports
// "engine unavailable" and, under --report-only, the run still exits 0.

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, isAbsolute } from 'node:path';
import { loadTickets, assertDisjoint, excludeHeldOutText, majorityLabelFromTrain, verifyFixtureHashes, BENCH_DIR, FIXTURE_DIR } from './lib/fixture.mjs';
import { assertVocabDisjoint } from './lib/vocab-guard.mjs';
import { replayJev, runLocal, trainFewShot, trainTicketQuestions } from './lib/arms.mjs';
import { scoreRecords, buildReceipt, readStats, toItemRecords, recordsDocument } from './lib/receipt.mjs';
import { resolveLocalContext } from './lib/model-dir.mjs';
import { leakageGate } from './lib/leakage.mjs';
import { evaluateGates, gatesTable, loadGates } from './lib/gates.mjs';
import { loadDataset } from './datasets/index.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(HERE, '..');
const RESULTS_DIR = join(HERE, 'results');
const SUITES = ['tickets', 'banking77', 'clinc150', 'hwu64'];

export function parseArgs(argv) {
  const a = {
    suite: 'tickets',
    arm: 'both',
    embedder: 'hash',
    shots: 8,
    regime: 'few-shot',
    limit: undefined,
    out: undefined,
    gate: false,
    reportOnly: false,
    baselineReceipt: undefined,
  };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    const next = () => argv[++i];
    if (t === '--suite') a.suite = next();
    else if (t === '--arm') a.arm = next();
    else if (t === '--embedder') a.embedder = next();
    else if (t === '--model-dir') a.modelDir = next();
    else if (t === '--model') a.model = next();
    else if (t === '--manifest') a.manifest = next();
    else if (t === '--shots') a.shots = parseInt(next(), 10);
    else if (t === '--zero-shot') a.regime = 'zero-shot';
    else if (t === '--few-shot') a.regime = 'few-shot';
    else if (t === '--limit') a.limit = parseInt(next(), 10);
    else if (t === '--out') a.out = next();
    else if (t === '--gate') a.gate = true;
    else if (t === '--report-only') a.reportOnly = true;
    else if (t === '--baseline-receipt') a.baselineReceipt = next();
    else if (t === '--engine-options') a.engineOptions = JSON.parse(next());
    else if (t === '--no-test') a.noTest = true;
    else if (t === '--train-hashes') a.trainHashes = next();
    else if (t === '--emit-records') a.emitRecords = next();
    else throw new Error(`unknown flag: ${t}`);
  }
  return a;
}

/** Resolve the ruvector binding (contract: { Engine, version(), backend }). */
function resolveBinding(injected) {
  if (injected) return { binding: injected, source: 'injected' };
  const require = createRequire(import.meta.url);
  for (const p of [join(PKG_ROOT, 'index.js'), join(PKG_ROOT, 'dist', 'index.js')]) {
    if (existsSync(p)) {
      try {
        return { binding: require(p), source: p };
      } catch (e) {
        return { binding: null, error: `require(${p}) failed: ${e && e.message}` };
      }
    }
  }
  return { binding: null, error: 'no index.js / dist/index.js — core not built yet' };
}

/** Construct an Engine; returns { engine } or { error }. For `--embedder onnx`
 * the embedder is the single sha256-verified entry from resolveOnnxModel
 * (ctx.onnx), passed inline so the binding cannot fall back to another entry. */
function makeEngine(binding, args, ctx = {}) {
  if (!binding || typeof binding.Engine !== 'function') {
    return { error: 'binding has no Engine constructor' };
  }
  let embedder = args.embedder;
  if (args.embedder === 'onnx') {
    if (!ctx.onnx || ctx.onnx.unavailable) return { error: `onnx model unavailable: ${ctx.onnx?.unavailable ?? 'not resolved'}` };
    embedder = ctx.onnx.spec;
  }
  try {
    const opts = { embedder };
    // `--engine-options '{...}'` applies champion / tuned EngineOptions (ADR-004),
    // e.g. the confirming run after an optimize campaign.
    if (args.engineOptions) opts.engine = args.engineOptions;
    return { engine: new binding.Engine(JSON.stringify(opts)) };
  } catch (e) {
    return { error: `new Engine failed: ${e && e.message}` };
  }
}

/** Score the local arm across the splits we need for metrics + gates. */
function runLocalSplits(engine, bySplit, questions, departments, splitsToScore, majorityLabels) {
  const out = {};
  const records = {};
  let unavailable = null;
  for (const split of splitsToScore) {
    const items = bySplit[split] ?? [];
    if (!items.length) continue;
    const res = runLocal(engine, items, questions);
    if (!res.available) {
      unavailable = res.error;
      break;
    }
    out[split] = scoreRecords(res.records, { departments, wallMs: res.wallMs, majorityLabels });
    records[split] = toItemRecords(res.records, { choiceKey: 'department' });
  }
  return { blocks: out, records, unavailable };
}

async function runTickets(args, deps, ctx = {}) {
  const fixtureDir = deps.fixtureDir ?? FIXTURE_DIR;
  const benchDir = deps.benchDir ?? BENCH_DIR;
  verifyFixtureHashes({ benchDir, fixtureDir });
  const tickets = loadTickets({ benchDir, fixtureDir });
  assertDisjoint(tickets.bySplit);
  const trainPool = excludeHeldOutText(tickets.bySplit);
  const majorityLabels = {
    urgent: majorityLabelFromTrain(trainPool.trainItems, 'urgent'),
    frustration: majorityLabelFromTrain(trainPool.trainItems, 'frustration'),
  };
  const vocab = assertVocabDisjoint(tickets.questions, { fixtureDir });

  const jevBaseline = JSON.parse(readFileSync(join(benchDir, 'jev-baseline-2026-09-21.json'), 'utf8'));
  const departments = tickets.departments;
  const metrics = {};
  // --no-test: the jev arm only replays the test split, so it is skipped.
  const wantJev = !args.noTest && (args.arm === 'jev' || args.arm === 'both');
  const wantLocal = args.arm === 'local' || args.arm === 'both';
  // Assertion B before any engine is built or any split is scored.
  const leakage = wantLocal
    ? leakageGate(ctx.trainHashes, {
      test: tickets.bySplit.test,
      transfer: tickets.bySplit.transfer,
      calibration: tickets.bySplit.calibration,
    })
    : null;

  // jev arm — replay frozen baseline on the test split (gen-0, apples-to-apples)
  let jevNote = null;
  if (wantJev) {
    const jev = replayJev(jevBaseline, tickets.bySplit.test, { arm: 'baseline', limit: args.limit });
    if (jev.available) {
      metrics.jev = { test: scoreRecords(jev.records, { departments }) };
      jevNote = jev.note;
      // champion is informational only
      const champ = replayJev(jevBaseline, tickets.bySplit.test, { arm: 'champion', limit: args.limit });
      if (champ.available) metrics.jev_champion = { test: scoreRecords(champ.records, { departments }) };
    }
  }

  // local arm — the binding
  let binding = null,
    training = null,
    stats = null,
    localUnavailable = null,
    itemRecords = null;
  if (wantLocal) {
    const resolved = resolveBinding(deps.binding);
    const { engine, error } = resolved.binding ? makeEngine(resolved.binding, args, ctx) : { error: resolved.error };
    if (error) {
      localUnavailable = error;
      binding = { unavailable: true, error };
    } else {
      binding = {
        version: safeCall(resolved.binding.version),
        backend: resolved.binding.backend ?? null,
        source: resolved.source,
      };
      if (args.regime === 'few-shot') {
        training = trainTicketQuestions(engine, trainPool.trainItems, tickets.questions, { shots: args.shots });
        training.excludedHeldOutTexts = trainPool.excluded;
      }
      // test needs limit-subsetting to match jev's id set; other splits full.
      const testItems = typeof args.limit === 'number' ? tickets.bySplit.test.slice(0, args.limit) : tickets.bySplit.test;
      const bySplit = { ...tickets.bySplit, test: testItems };
      const splits = args.noTest ? ['validation', 'transfer'] : ['test', 'validation', 'transfer'];
      const local = runLocalSplits(engine, bySplit, tickets.questions, departments, splits, majorityLabels);
      localUnavailable = local.unavailable;
      if (localUnavailable) binding.unavailable = true, (binding.error = localUnavailable);
      else {
        metrics.local = local.blocks;
        itemRecords = { local: local.records };
      }
      stats = readStats(engine);
    }
  }

  return {
    suite: 'tickets',
    departments,
    counts: tickets.counts,
    splitsHash: tickets.splitsHash,
    questions: tickets.questions,
    metrics,
    binding,
    training,
    stats,
    vocab: { ...vocab, note: jevNote },
    engineAvailable: !localUnavailable && wantLocal ? true : wantLocal ? false : null,
    localUnavailable,
    leakage,
    itemRecords,
  };
}

/** Build the flat metric bag the gate evaluator consumes. */
function gateMetricBag(run, args, baselineReceipt) {
  const bag = {};
  const jev = run.metrics.jev?.test;
  const local = run.metrics.local?.test;
  if (jev) bag.jev_choice_accuracy_test = jev.choice_accuracy;
  if (local) {
    bag.choice_accuracy_test = local.choice_accuracy;
    bag.ece_test = local.ece.ece;
    bag.urgent_accuracy_test = local.urgent_accuracy;
    bag.urgent_train_majority_accuracy_test = local.urgent_train_majority_accuracy;
    bag.frustration_accuracy_test = local.frustration_accuracy;
    bag.frustration_train_majority_accuracy_test = local.frustration_train_majority_accuracy;
    bag.latency_p95_test = local.latency_ms.p95;
    if (typeof local.oos_auroc === 'number') bag.oos_auroc = local.oos_auroc;
  }
  if (local && run.metrics.local?.transfer && baselineReceipt) {
    const baseTransfer = baselineReceipt?.metrics?.local?.transfer?.choice_accuracy;
    if (typeof baseTransfer === 'number') {
      bag.transfer_regression_pp = (baseTransfer - run.metrics.local.transfer.choice_accuracy) * 100;
    }
  }
  return bag;
}

export async function main(argv, deps = {}) {
  const args = parseArgs(argv);
  const suites = args.suite === 'all' ? SUITES : [args.suite];
  if (args.suite !== 'all' && !SUITES.includes(args.suite)) {
    throw new Error(`unknown suite '${args.suite}' (choose ${SUITES.join('|')}|all)`);
  }
  if (args.noTest && args.gate) throw new Error('--no-test cannot be combined with --gate (every gate is scored on test)');
  if (args.noTest && args.emitRecords) throw new Error('--emit-records writes test records; it cannot be combined with --no-test');
  if (args.emitRecords && suites.length !== 1) throw new Error('--emit-records needs a single --suite');
  if (args.noTest && args.arm === 'jev') throw new Error('--no-test with --arm jev scores nothing (the jev arm is test-only)');
  if (args.noTest && suites.some((x) => x !== 'tickets')) throw new Error('--no-test applies to --suite tickets only (public suites have no validation split)');
  const ctx = resolveLocalContext(args);
  const fixtureHashes = verifyFixtureHashes({
    benchDir: deps.benchDir ?? BENCH_DIR,
    fixtureDir: deps.fixtureDir ?? FIXTURE_DIR,
  }).hashes;

  const baselineReceipt = args.baselineReceipt ? JSON.parse(readFileSync(args.baselineReceipt, 'utf8')) : null;
  const results = [];
  let anyFail = false;

  for (const suite of suites) {
    let run;
    if (suite === 'tickets') {
      run = await runTickets(args, deps, ctx);
    } else {
      run = await runDataset(suite, args, deps, ctx);
    }
    if (run.skipped) {
      results.push({ suite, skipped: run.skipped });
      console.log(`\n## ${suite}\nskipped: ${run.skipped}`);
      continue;
    }

    // gates
    let gateResult = null;
    if (args.gate) {
      const bag = gateMetricBag(run, args, baselineReceipt);
      // Latency gate target follows the REAL backend for an onnx run: the
      // binding reports 'native' or 'wasm'. A hash-embedder run targets neither
      // (both latency gates SKIP) — the hash embedder is a placeholder.
      let embedderTarget = 'hash';
      if (args.embedder === 'onnx') {
        embedderTarget = run.binding && run.binding.backend === 'wasm' ? 'wasm' : 'native';
      }
      const ctx = {
        embedderKind: args.embedder,
        embedderTarget,
        hasOos: run.hasOos ?? false,
        hasSecondary: suite === 'tickets',
        hasBaselineReceipt: !!baselineReceipt,
        engineAvailable: run.engineAvailable,
      };
      gateResult = evaluateGates(bag, ctx, loadGates());
      if (gateResult.anyFail) anyFail = true;
    }

    const receipt = buildReceipt({
      suite,
      arms: armList(args.arm),
      embedder: args.embedder,
      regime: args.regime,
      shots: args.shots,
      limit: args.limit,
      splitsHash: run.splitsHash,
      fixtureHashes,
      counts: run.counts,
      questions: run.questions,
      metrics: run.metrics,
      gates: gateResult ? { report_only: args.reportOnly, ...gateResult } : null,
      binding: run.binding,
      training: run.training,
      vocabGuard: run.vocab,
      stats: run.stats,
      extra: buildExtra(run, args),
      embedderModel: ctx.onnx?.record ?? (ctx.onnx?.unavailable ? { unavailable: ctx.onnx.unavailable } : undefined),
      leakage: run.leakage ?? undefined,
      itemRecords: run.itemRecords ?? undefined,
      noTest: !!args.noTest,
    });
    if (args.emitRecords) {
      const outPath = isAbsolute(args.emitRecords) ? args.emitRecords : join(process.cwd(), args.emitRecords);
      mkdirSync(dirname(outPath), { recursive: true });
      writeFileSync(outPath, JSON.stringify(recordsDocument({ suite, args, run, embedderModel: ctx.onnx?.record }), null, 2) + '\n');
      console.log(`records → ${outPath}`);
    }
    results.push({ suite, receipt, gateResult });
    printSuite(suite, run, receipt, gateResult, args);
  }

  // write receipt(s)
  const outPath = resolveOut(args, suites);
  mkdirSync(dirname(outPath), { recursive: true });
  const payload = results.length === 1 && results[0].receipt ? results[0].receipt : { runs: results };
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + '\n');
  console.log(`\nreceipt → ${outPath}`);

  const failExit = args.gate && anyFail && !args.reportOnly;
  return { code: failExit ? 1 : 0, receipt: payload, results };
}

async function runDataset(suite, args, deps, ctx = {}) {
  if (args.noTest) throw new Error(`--no-test: suite '${suite}' has only train/test splits — nothing to score without test`);
  const ds = await (deps.loadDataset ?? loadDataset)(suite, { limit: args.limit, cacheDir: deps.cacheDir });
  if (ds.skipped) return { skipped: ds.skipped };
  // Dataset items carry a flat string `label`; normalise to the tickets item
  // shape ({ label: { intent } }) so the shared arm/scoring code addresses the
  // `intent` choice question uniformly. The OOS flag is carried through.
  const shape = (it) => ({
    id: it.id, text: it.text, label: { intent: it.label },
    oos: typeof it.oos === 'boolean' ? it.oos : undefined,
  });
  const trainItems = (ds.trainItems ?? []).map(shape);
  const testItems = (ds.testItems ?? []).map(shape);
  // Assertion B before any engine is built (public suites: the test split).
  const leakage = leakageGate(ctx.trainHashes, { test: testItems });
  // Non-tickets suites have no frozen Jev baseline: local arm only.
  const resolved = resolveBinding(deps.binding);
  const { engine, error } = resolved.binding ? makeEngine(resolved.binding, args, ctx) : { error: resolved.error };
  const metrics = {};
  let binding, localUnavailable, training = null, stats = null, itemRecords = null;
  if (error) {
    localUnavailable = error;
    binding = { unavailable: true, error };
  } else {
    binding = { version: safeCall(resolved.binding.version), backend: resolved.binding.backend ?? null };
    if (args.regime === 'few-shot' && trainItems.length) {
      training = trainFewShot(engine, trainItems, { shots: args.shots, question: 'intent', labelKey: 'intent' });
    }
    const res = runLocal(engine, testItems, ds.questions, { labelKey: 'intent' });
    if (!res.available) localUnavailable = res.error;
    else {
      metrics.local = { test: scoreRecords(res.records, { departments: ds.labels, wallMs: res.wallMs }) };
      itemRecords = { local: { test: toItemRecords(res.records, { choiceKey: 'intent' }) } };
    }
    stats = readStats(engine);
  }
  return {
    suite,
    departments: ds.labels,
    counts: ds.counts,
    splitsHash: ds.splitsHash ?? null,
    questions: ds.questions,
    metrics,
    binding,
    training,
    stats,
    vocab: null,
    hasOos: !!ds.hasOos,
    engineAvailable: !localUnavailable,
    localUnavailable,
    leakage,
    itemRecords,
  };
}

// ---------------------------------------------------------------------------
// presentation
// ---------------------------------------------------------------------------

function printSuite(suite, run, receipt, gateResult, args) {
  console.log(`\n## ${suite} — embedder=${args.embedder} regime=${args.regime}${args.limit ? ` limit=${args.limit}` : ''}`);
  const rows = [];
  const shown = args.noTest ? 'validation' : 'test';
  if (args.noTest) console.log('_--no-test: validation + transfer only; the test split was not scored._');
  if (run.leakage) console.log(`_leakage (Assertion B): ${run.leakage.train_rows} train hashes vs held-out rows — intersection ${run.leakage.intersection}_`);
  for (const arm of ['jev', 'local']) {
    const b = run.metrics[arm]?.[shown];
    if (!b) {
      if ((arm === 'local' && run.localUnavailable)) rows.push([arm, 'engine unavailable', '', '', '', '']);
      continue;
    }
    rows.push([
      arm,
      pct(b.choice_accuracy),
      b.macro_f1.toFixed(3),
      b.ece.ece.toFixed(4),
      b.mean_confidence.toFixed(3),
      `${b.latency_ms.p50.toFixed(1)}/${b.latency_ms.p95.toFixed(1)}`,
    ]);
  }
  console.log('| arm | acc | macroF1 | ECE | meanConf | p50/p95 ms |');
  console.log('|---|---|---|---|---|---|');
  for (const r of rows) console.log(`| ${r.join(' | ')} |`);
  if (run.localUnavailable) console.log(`\n_local arm: engine unavailable — ${run.localUnavailable}_`);
  if (run.metrics.jev) console.log(`_jev arm: replayed frozen baseline (gen-0); no network. Jev exposes no noul probability, so urgent AUROC is n/a for jev._`);
  if (gateResult) {
    console.log(`\n### Gates${args.reportOnly ? ' (report-only)' : ''}`);
    console.log(gatesTable(gateResult));
    console.log(`\n${gateResult.pass ? 'PASS' : 'FAIL'} — ${gateResult.skipped} skipped`);
  }
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const armList = (arm) => (arm === 'both' ? ['jev', 'local'] : [arm]);

/** Receipt `extra`: records the tuned EngineOptions so a confirming run's
 * numbers are reproducible (else it is indistinguishable from the default). */
function buildExtra(run, args) {
  const extra = {};
  if (run.localUnavailable) extra.local_unavailable = run.localUnavailable;
  if (args.engineOptions) extra.engine_options = args.engineOptions;
  return Object.keys(extra).length ? extra : undefined;
}
const safeCall = (fn) => {
  try {
    return typeof fn === 'function' ? fn() : null;
  } catch {
    return null;
  }
};

function resolveOut(args, suites) {
  if (args.out) return isAbsolute(args.out) ? args.out : join(PKG_ROOT, args.out);
  const date = new Date().toISOString().slice(0, 10);
  const name = suites.length === 1 ? suites[0] : 'all';
  return join(RESULTS_DIR, `${name}-${date}.json`);
}

// CLI entry
if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2))
    .then((r) => process.exit(r.code))
    .catch((e) => {
      console.error(`bench failed: ${e && e.code === 'LEAKAGE' ? e.message : e && e.stack ? e.stack : e}`);
      process.exit(e && e.code === 'LEAKAGE' ? 3 : 2);
    });
}
