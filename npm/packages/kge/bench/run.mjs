#!/usr/bin/env node
// @ruvector/kge benchmark harness (ADR-003/006). Imports triples into the Model
// binding, trains, evaluates filtered MRR/Hits with RANDOM tie-break, optionally
// measures ANN recall, the RANDOM-tie-break assertion, and the symmetry-decoy
// adversarial drop, writes one receipt, and evaluates the release gates.
//
// Usage:
//   node bench/run.mjs [--suite synthetic|fb15k237|wn18rr|codexm|yago310|all]
//                      [--scorer hole|rotate] [--dims 256] [--epochs N] [--limit N]
//                      [--ann] [--adversarial] [--tie-check]
//                      [--gate] [--report-only] [--out results/<name>-<date>.json]
//                      [--baseline-receipt PATH]
//
// The harness never crashes on an unavailable engine: each arm reports "engine
// unavailable" and, under --report-only, the run still exits 0. Every binding
// call goes through bench/lib/arms.mjs. A dev override KGE_BENCH_BINDING=<path>
// points the harness at an alternate binding module (e.g. the fake, for a real
// synthetic run without the native build).

import { readFileSync, writeFileSync, existsSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, isAbsolute } from 'node:path';
import { loadSynthetic, allTagged, assertDisjoint, verifyFixtureHashes, BENCH_DIR } from './lib/fixture.mjs';
import { loadDataset } from './datasets/index.mjs';
import { computeHashes } from '../scripts/hash-fixtures.mjs';
import {
  resolveBinding, constructModel, addTriples, train, evalSplit,
  predictTopK, scoreTriple, buildIndex, readStats, safeVersion,
} from './lib/arms.mjs';
import { detectSymmetricRelations, detectInversePairs, pickTargets, generateSymmetryDecoys } from './lib/adversarial.mjs';
import { recallAtK, latency, mean, ratePerSecond } from './lib/metrics.mjs';
import { buildReceipt, sha256hex } from './lib/receipt.mjs';
import { evaluateGates, gatesTable, loadGates } from './lib/gates.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const PKG_ROOT = join(HERE, '..');
const RESULTS_DIR = join(HERE, 'results');
const SUITES = ['synthetic', 'fb15k237', 'wn18rr', 'codexm', 'yago310'];
const ANN_QUERIES = 500;
const ADV_TARGETS = 20;

export function parseArgs(argv) {
  const a = {
    suite: 'synthetic', scorer: 'hole', dims: 256, epochs: 50, seed: 42,
    limit: undefined, ann: false, adversarial: false, tieCheck: false,
    gate: false, reportOnly: false, out: undefined, baselineReceipt: undefined,
  };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    const next = () => argv[++i];
    if (t === '--suite') a.suite = next();
    else if (t === '--scorer') a.scorer = next();
    else if (t === '--dims') a.dims = parseInt(next(), 10);
    else if (t === '--epochs') a.epochs = parseInt(next(), 10);
    else if (t === '--seed') a.seed = parseInt(next(), 10);
    else if (t === '--limit') a.limit = parseInt(next(), 10);
    else if (t === '--ann') a.ann = true;
    else if (t === '--adversarial') a.adversarial = true;
    else if (t === '--tie-check') a.tieCheck = true;
    else if (t === '--gate') a.gate = true;
    else if (t === '--report-only') a.reportOnly = true;
    else if (t === '--out') a.out = next();
    else if (t === '--baseline-receipt') a.baselineReceipt = next();
    else throw new Error(`unknown flag: ${t}`);
  }
  return a;
}

/** Load the suite's triples in the harness's canonical shape. */
async function loadSuite(suite, args, deps) {
  if (suite === 'synthetic') {
    verifyFixtureHashes();
    const kg = loadSynthetic();
    assertDisjoint(kg.splits);
    return {
      suite, splits: kg.splits, entities: kg.entities, nEntities: kg.nEntities,
      counts: kg.counts, splitsHash: kg.splitsHash, hardNegatives: null,
      datasetHashes: computeHashes(BENCH_DIR),
    };
  }
  const ds = await loadDataset(suite, { limit: args.limit, cacheDir: deps.cacheDir });
  if (ds.skipped) return { skipped: ds.skipped };
  const nEntities = ds.counts.entities;
  return {
    suite, splits: ds.splits, entities: null, nEntities,
    counts: ds.counts, splitsHash: ds.splitsHash, hardNegatives: ds.hardNegatives ?? null,
    datasetHashes: ds.fileHashes, licence: ds.licence, baselineMrr: ds.baselineMrr ?? null,
  };
}

/** ANN recall@10 of index vs exhaustive, on up to ANN_QUERIES test queries. */
function runAnn(model, testTriples) {
  const built = buildIndex(model);
  if (!built.available) return { skipped: `buildIndex unavailable: ${built.error}` };
  const pairs = [];
  const latencies = [];
  for (const t of testTriples.slice(0, ANN_QUERIES)) {
    const withIdx = predictTopK(model, { s: t.s, r: t.r, k: 10, useIndex: true });
    const noIdx = predictTopK(model, { s: t.s, r: t.r, k: 10, useIndex: false });
    if (!withIdx.available || !noIdx.available) return { skipped: `predict unavailable: ${withIdx.error ?? noIdx.error}` };
    pairs.push({ ann: withIdx.candidates.map((c) => c.o), exact: noIdx.candidates.map((c) => c.o) });
    latencies.push(withIdx.latencyMs);
  }
  const recall = mean(pairs.map((p) => recallAtK(p.ann, p.exact, 10)));
  return { recall_at_10: recall, queries: pairs.length, latency_ms: latency(latencies) };
}

/** Symmetry-decoy adversarial run: confidence drop on targeted facts. */
function runAdversarial(model, allTriples, testTriples, entities, args, config) {
  const sym = detectSymmetricRelations(allTriples);
  const targets = pickTargets(testTriples, sym, ADV_TARGETS);
  if (targets.length === 0) return { skipped: 'no targets under a symmetric relation' };
  const before = [];
  for (const t of targets) {
    const s = scoreTriple(model, t);
    if (!s.available) return { skipped: `scoreTriple unavailable: ${s.error}` };
    before.push(s.score);
  }
  const decoys = generateSymmetryDecoys(targets, entities, allTriples, { perTarget: 3 });
  const add = addTriples(model, decoys.map((d) => ({ ...d, split: 'train' })));
  if (!add.available) return { skipped: `addTriples(decoys) unavailable: ${add.error}` };
  train(model, config); // incremental retrain (no-op for stubs)
  const after = targets.map((t) => scoreTriple(model, t).score);
  // keep only targets scored on BOTH sides of the attack (a fact ranked beyond
  // the k cap returns null and cannot be compared)
  const paired = before.map((b, i) => [b, after[i]]).filter(([b, a]) => b != null && a != null);
  if (paired.length === 0) return { skipped: 'no target scored on both sides of the attack' };
  const meanBefore = mean(paired.map((p) => p[0]));
  const meanAfter = mean(paired.map((p) => p[1]));
  return {
    symmetric_relations: sym.filter((r) => r.symmetric).map((r) => ({ fraction: r.fraction, count: r.count })),
    inverse_pairs: detectInversePairs(allTriples).map((p) => ({ fraction: p.fraction, count: p.count })),
    targets: targets.length,
    targets_hash: sha256hex(targets.map((t) => `${t.s} ${t.r} ${t.o}`).sort().join('|')),
    decoys: decoys.length,
    mean_confidence_before: meanBefore,
    mean_confidence_after: meanAfter,
    drop: meanBefore - meanAfter,
  };
}

/** The RANDOM tie-break assertion via a zero-epoch (constant-scorer) model. */
function runTieCheck(binding, config, tagged, nEntities) {
  const { model, error } = constructModel(binding, { ...config, epochs: 0 });
  if (error) return { available: false, reason: `constant model unavailable: ${error}` };
  const add = addTriples(model, tagged);
  if (!add.available) return { available: false, reason: `addTriples unavailable: ${add.error}` };
  const modes = {};
  for (const mode of ['top', 'random', 'bottom']) {
    const e = evalSplit(model, 'test', mode);
    if (!e.available) return { available: false, reason: `evalJson(${mode}) unavailable: ${e.error}` };
    modes[mode] = e.mr;
  }
  return { available: true, topMr: modes.top, randomMr: modes.random, bottomMr: modes.bottom, nEntities };
}

async function runSuite(suite, args, deps) {
  const loaded = await loadSuite(suite, args, deps);
  if (loaded.skipped) return { skipped: loaded.skipped };
  const config = { scorer: args.scorer, dims: args.dims, seed: args.seed, epochs: args.epochs };
  const tagged = allTagged(loaded.splits);
  const entities = loaded.entities ?? [...new Set(tagged.flatMap((t) => [t.s, t.o]))];
  const allTriples = tagged.map((t) => ({ s: t.s, r: t.r, o: t.o }));

  const resolved = resolveBinding(PKG_ROOT, deps.binding);
  if (!resolved.binding) {
    return { ...loaded, config, binding: { unavailable: true, error: resolved.error }, unavailable: resolved.error };
  }
  const { model, error } = constructModel(resolved.binding, config);
  if (error) {
    return { ...loaded, config, binding: { unavailable: true, error }, bindingSource: resolved.source, unavailable: error };
  }
  const bindingInfo = { version: safeVersion(resolved.binding), backend: resolved.binding.backend ?? null };

  const added = addTriples(model, tagged);
  if (!added.available) return { ...loaded, config, binding: { unavailable: true, error: added.error }, bindingSource: resolved.source, unavailable: added.error };
  const training = train(model, config);
  const trainMeta = training.available
    ? {
        epochs: training.epochs,
        triplesPerSec: training.triplesPerSec ?? ratePerSecond(loaded.counts.train, training.wallMs),
        loss: training.loss,
        n3Penalty: training.n3Penalty,
        batches: training.batches,
      }
    : { unavailable: training.error };

  // filtered MRR/Hits (RANDOM tie-break) on valid + test (+ transfer)
  const metrics = {};
  let unavailable = null;
  for (const split of ['valid', 'test', 'transfer']) {
    if (!(loaded.splits[split] && loaded.splits[split].length)) continue;
    const e = evalSplit(model, split, 'random');
    if (!e.available) { unavailable = e.error; break; }
    metrics[split] = { mrr: e.mrr, mr: e.mr, hits: e.hits, perSide: e.perSide, splitSource: e.splitSource, filtered: e.filtered };
  }
  if (unavailable) {
    return { ...loaded, config, binding: { unavailable: true, error: unavailable }, bindingSource: resolved.source, unavailable };
  }

  const ann = args.ann ? runAnn(model, loaded.splits.test) : null;
  const adversarial = args.adversarial ? runAdversarial(model, allTriples, loaded.splits.test, entities, args, config) : null;
  const tieCheck = args.tieCheck ? runTieCheck(resolved.binding, config, tagged, loaded.nEntities) : null;
  const predictLatency = ann && ann.latency_ms ? ann.latency_ms : null;

  // hard negatives (codexm): informational mean score
  let hardNegatives = null;
  if (loaded.hardNegatives && loaded.hardNegatives.test?.length) {
    const scores = [];
    for (const t of loaded.hardNegatives.test.slice(0, 500)) {
      const s = scoreTriple(model, t);
      if (s.available && s.score != null) scores.push(s.score);
    }
    hardNegatives = { n: scores.length, mean_score: mean(scores) };
  }

  return {
    ...loaded, config, metrics, ann, adversarial, tieCheck, hardNegatives,
    predictLatency, training: trainMeta, binding: bindingInfo, bindingSource: resolved.source,
    stats: readStats(model), engineAvailable: true,
  };
}

/** Flat metric bag for the gate evaluator. */
function gateBag(run, baselineReceipt) {
  const bag = {};
  if (run.metrics?.test) bag.mrr_test = run.metrics.test.mrr;
  if (run.ann && typeof run.ann.recall_at_10 === 'number') bag.ann_recall_at_10 = run.ann.recall_at_10;
  if (run.adversarial && typeof run.adversarial.drop === 'number') bag.adversarial_confidence_drop = run.adversarial.drop;
  const baseT = baselineReceipt?.metrics?.transfer?.mrr;
  if (typeof baseT === 'number' && run.metrics?.transfer) {
    bag.transfer_regression_pt = (baseT - run.metrics.transfer.mrr) * 100;
  }
  return bag;
}

export async function main(argv, deps = {}) {
  const args = parseArgs(argv);
  const suites = args.suite === 'all' ? SUITES : [args.suite];
  if (args.suite !== 'all' && !SUITES.includes(args.suite)) {
    throw new Error(`unknown suite '${args.suite}' (choose ${SUITES.join('|')}|all)`);
  }
  const baselineReceipt = args.baselineReceipt ? JSON.parse(readFileSync(args.baselineReceipt, 'utf8')) : null;
  const results = [];
  let anyFail = false;

  for (const suite of suites) {
    const run = await runSuite(suite, args, deps);
    if (run.skipped) {
      results.push({ suite, skipped: run.skipped });
      console.log(`\n## ${suite}\nskipped: ${run.skipped}`);
      continue;
    }
    const engineAvailable = run.engineAvailable === true;
    // Honest tie-check row when the engine bailed at eval before the tie-check ran.
    if (args.tieCheck && run.unavailable && !run.tieCheck) {
      run.tieCheck = { available: false, reason: 'engine unavailable before tie-check' };
    }

    let gateResult = null;
    if (args.gate) {
      const ctx = {
        suite, hasAnn: !!(run.ann && !run.ann.skipped), hasAdversarial: !!(run.adversarial && !run.adversarial.skipped),
        hasBaselineReceipt: !!baselineReceipt, engineAvailable, tieCheck: run.tieCheck,
        latencyP95: run.predictLatency?.p95, backend: run.binding?.backend,
      };
      gateResult = evaluateGates(gateBag(run, baselineReceipt), ctx, loadGates());
      if (gateResult.anyFail) anyFail = true;
    }

    const receipt = buildReceipt({
      suite, scorer: args.scorer, config: run.config, tieBreak: 'random',
      datasetHashes: run.datasetHashes, splitsHash: run.splitsHash, counts: run.counts,
      metrics: run.metrics ?? {}, ann: run.ann, adversarial: run.adversarial,
      hardNegatives: run.hardNegatives, tieCheck: run.tieCheck, latency: run.predictLatency,
      training: run.training, gates: gateResult ? { report_only: args.reportOnly, ...gateResult } : null,
      binding: run.binding, bindingSource: run.bindingSource, stats: run.stats,
      extra: run.unavailable ? { engine_unavailable: run.unavailable, licence: run.licence } : (run.licence ? { licence: run.licence } : undefined),
    });
    results.push({ suite, receipt, gateResult });
    printSuite(suite, run, gateResult, args);
  }

  const outPath = resolveOut(args, suites);
  mkdirSync(dirname(outPath), { recursive: true });
  const payload = results.length === 1 && results[0].receipt ? results[0].receipt : { runs: results };
  writeFileSync(outPath, JSON.stringify(payload, null, 2) + '\n');
  console.log(`\nreceipt → ${outPath}`);

  const failExit = args.gate && anyFail && !args.reportOnly;
  return { code: failExit ? 1 : 0, receipt: payload, results };
}

function printSuite(suite, run, gateResult, args) {
  console.log(`\n## ${suite} — scorer=${args.scorer} dims=${args.dims}${args.limit ? ` limit=${args.limit}` : ''} (binding: ${run.bindingSource})`);
  if (run.unavailable) {
    console.log(`\n_engine unavailable — ${run.unavailable}_`);
  } else {
    console.log('| split | MRR | MR | H@1 | H@3 | H@10 |');
    console.log('|---|---|---|---|---|---|');
    for (const s of ['valid', 'test', 'transfer']) {
      const m = run.metrics?.[s];
      if (!m) continue;
      console.log(`| ${s} | ${f(m.mrr)} | ${m.mr?.toFixed(1)} | ${f(m.hits?.[1])} | ${f(m.hits?.[3])} | ${f(m.hits?.[10])} |`);
    }
    if (run.ann) console.log(run.ann.skipped ? `\n_ANN: ${run.ann.skipped}_` : `\nANN recall@10 = ${f(run.ann.recall_at_10)} over ${run.ann.queries} queries`);
    if (run.adversarial) console.log(run.adversarial.skipped ? `\n_adversarial: ${run.adversarial.skipped}_` : `adversarial: mean conf ${f(run.adversarial.mean_confidence_before)} → ${f(run.adversarial.mean_confidence_after)} (drop ${f(run.adversarial.drop)}) over ${run.adversarial.targets} targets, ${run.adversarial.decoys} decoys`);
    if (run.tieCheck) console.log(run.tieCheck.available ? `tie-check (|E|=${run.tieCheck.nEntities}): TOP MR=${run.tieCheck.topMr?.toFixed(1)} RANDOM MR=${run.tieCheck.randomMr?.toFixed(1)} BOTTOM MR=${run.tieCheck.bottomMr?.toFixed(1)}` : `\n_tie-check: ${run.tieCheck.reason}_`);
  }
  if (gateResult) {
    console.log(`\n### Gates${args.reportOnly ? ' (report-only)' : ''}`);
    console.log(gatesTable(gateResult));
    console.log(`\n${gateResult.pass ? 'PASS' : 'FAIL'} — ${gateResult.skipped} skipped`);
  }
}

const f = (x) => (typeof x === 'number' ? x.toFixed(3) : '—');

function resolveOut(args, suites) {
  if (args.out) return isAbsolute(args.out) ? args.out : join(PKG_ROOT, args.out);
  const date = new Date().toISOString().slice(0, 10);
  const name = suites.length === 1 ? suites[0] : 'all';
  return join(RESULTS_DIR, `${name}-${date}.json`);
}

if (import.meta.url === `file://${process.argv[1]}`) {
  main(process.argv.slice(2))
    .then((r) => process.exit(r.code))
    .catch((e) => {
      console.error(`bench failed: ${e && e.stack ? e.stack : e}`);
      process.exit(2);
    });
}
