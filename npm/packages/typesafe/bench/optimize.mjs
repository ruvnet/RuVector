#!/usr/bin/env node
// Optimize-campaign harness (ADR-004). Runs a gated `EngineOptions` grid search
// per embedding model arm over the tickets fixture's FROZEN splits, writes the
// hash-chained receipts + a campaign receipt into bench/results/, and prints the
// champion's test-split numbers next to Jev's frozen baseline.
//
// The splits come from the fixture (loadTickets), NOT re-derived — so the
// champion's "test" is exactly the fixture's test ids the Jev baseline was
// measured against. The model arm (bge / bge-int8 / MiniLM / MiniLM-int8) is
// chosen here in JS by constructing one engine per model; each engine's core
// campaign is over EngineOptions for its fixed embedder.
//
// Usage:
//   node bench/optimize.mjs [--models a,b,c] [--budget 64] [--limit N]
//                           [--out results/<name>.json]

import { createRequire } from 'node:module';
import { readFileSync, writeFileSync, mkdirSync } from 'node:fs';
import { fileURLToPath } from 'node:url';
import { dirname, join, isAbsolute } from 'node:path';
import { loadTickets, buildQuestions, BENCH_DIR, FIXTURE_DIR } from './lib/fixture.mjs';
import { scoreRecords } from './lib/receipt.mjs';
import { replayJev } from './lib/arms.mjs';

const HERE = dirname(fileURLToPath(import.meta.url));
const RESULTS_DIR = join(HERE, 'results');
const DEFAULT_MODELS = [
  'bge-small-en-v1.5',
  'bge-small-en-v1.5-int8',
  'all-MiniLM-L6-v2',
  'all-MiniLM-L6-v2-int8',
];

function parseArgs(argv) {
  const a = { models: DEFAULT_MODELS, budget: 64, limit: undefined, out: undefined, question: 'department' };
  for (let i = 0; i < argv.length; i++) {
    const t = argv[i];
    if (t === '--models') a.models = argv[++i].split(',').map((s) => s.trim()).filter(Boolean);
    else if (t === '--budget') a.budget = parseInt(argv[++i], 10);
    else if (t === '--limit') a.limit = parseInt(argv[++i], 10);
    else if (t === '--out') a.out = argv[++i];
    else if (t === '--question') a.question = argv[++i];
    else throw new Error(`unknown flag: ${t}`);
  }
  return a;
}

function resolveBinding() {
  const require = createRequire(import.meta.url);
  const pkgRoot = join(HERE, '..');
  for (const p of [join(pkgRoot, 'index.js'), join(pkgRoot, 'dist', 'index.js')]) {
    try {
      return require(p);
    } catch {
      /* try next */
    }
  }
  throw new Error('no @ruvector/typesafe binding (index.js) — build the native addon first');
}

/** Every fixture item as a campaign row for the `department` choice question. */
function campaignRows(tickets, question, limit) {
  const rows = [];
  for (const split of ['train', 'calibration', 'validation', 'transfer', 'test']) {
    const items = tickets.bySplit[split] ?? [];
    for (const it of items) {
      const label = it.label?.[question] ?? it.label?.department;
      if (typeof label !== 'string') continue;
      rows.push({ text: it.text, label, split, tier: 'A' });
    }
  }
  if (typeof limit === 'number') {
    // Keep proportional coverage: cap only the train split for speed.
    return rows.filter((r) => r.split !== 'train').concat(rows.filter((r) => r.split === 'train').slice(0, limit));
  }
  return rows;
}

function engineFor(binding, model) {
  const spec = {
    embedder: { kind: 'onnx', modelDir: 'models', manifest: 'models/manifest.json', model },
  };
  return new binding.Engine(JSON.stringify(spec));
}

const pct = (x) => `${(x * 100).toFixed(1)}%`;
const f4 = (x) => (typeof x === 'number' ? x.toFixed(4) : 'n/a');

function main(argv) {
  const args = parseArgs(argv);
  const binding = resolveBinding();
  const tickets = loadTickets({ benchDir: BENCH_DIR, fixtureDir: FIXTURE_DIR });
  const questions = buildQuestions(tickets.questions);
  const questionDef = questions[args.question];
  if (!questionDef) throw new Error(`no question '${args.question}' in the fixture`);
  const rows = campaignRows(tickets, args.question, args.limit);

  // Jev frozen baseline on the test split, for the side-by-side comparison.
  const jevBaseline = JSON.parse(readFileSync(join(BENCH_DIR, 'jev-baseline-2026-09-21.json'), 'utf8'));
  const jev = replayJev(jevBaseline, tickets.bySplit.test, { arm: 'baseline' });
  const jevBlock = jev.available ? scoreRecords(jev.records, { departments: tickets.departments }) : null;

  const arms = [];
  const allReceipts = [];
  for (const model of args.models) {
    let report;
    const t0 = performance.now();
    try {
      const engine = engineFor(binding, model);
      const spec = {
        question: args.question,
        question_def: questionDef,
        rows,
        budget_per_day: args.budget,
        day_key: 'optimize',
      };
      const raw = engine.optimizeJson(JSON.stringify(spec));
      report = JSON.parse(raw);
    } catch (e) {
      console.error(`arm ${model}: ${e && e.message ? e.message : e}`);
      arms.push({ model, error: String(e && e.message ? e.message : e) });
      continue;
    }
    const wallMs = performance.now() - t0;
    if (report.error) {
      arms.push({ model, error: report.error.message ?? JSON.stringify(report.error) });
      continue;
    }
    for (const r of report.receipts.receipts) allReceipts.push({ arm: model, ...r });
    // Best proposal's sequential-test statistic (max wealth reached).
    const best = report.arms.reduce(
      (acc, a) => (a.statistic.max_wealth > (acc?.statistic.max_wealth ?? -1) ? a : acc),
      null,
    );
    const promotedVia = [
      ...new Set(report.arms.filter((x) => x.promoted).map((x) => x.promoted_by)),
    ];
    arms.push({
      model,
      promotions: report.promotions,
      promoted_via: promotedVia,
      budget_consumed: report.budget_consumed,
      baseline: {
        val: report.baseline_val.baseline_accuracy,
        transfer: report.baseline_transfer.baseline_accuracy,
        test: report.baseline_test.baseline_accuracy,
        test_ece: report.baseline_test.ece,
      },
      champion: {
        val: report.champion_val.champion_accuracy,
        transfer: report.champion_transfer.champion_accuracy,
        test: report.champion_test.champion_accuracy,
        test_ece: report.champion_test.ece,
        options: report.champion_options,
      },
      best_stat: best ? best.statistic : null,
      val_n: report.champion_val.n,
      wallMs,
    });
  }

  // Choose the overall champion by VALIDATION accuracy (never by test — test is
  // scored only for reporting, not for selection).
  const ranked = arms.filter((a) => !a.error).sort((x, y) => y.champion.val - x.champion.val);
  const champion = ranked[0] ?? null;

  // ---- print ----
  console.log(`\n## optimize campaign — tickets/${args.question} — ${arms.length} model arms`);
  console.log('| model | base val | champ val | base test | champ test | champ ECE | promo | maxWealth | budget |');
  console.log('|---|---|---|---|---|---|---|---|---|');
  for (const a of arms) {
    if (a.error) {
      console.log(`| ${a.model} | error: ${a.error} | | | | | | | |`);
      continue;
    }
    console.log(
      `| ${a.model} | ${pct(a.baseline.val)} | ${pct(a.champion.val)} | ${pct(a.baseline.test)} | ` +
        `${pct(a.champion.test)} | ${f4(a.champion.test_ece)} | ` +
        `${a.promotions}${a.promoted_via && a.promoted_via.length ? ' (' + a.promoted_via.join(',') + ')' : ''} | ` +
        `${a.best_stat ? a.best_stat.max_wealth.toFixed(2) : 'n/a'}/${a.best_stat ? a.best_stat.threshold.toFixed(0) : '?'} | ${a.budget_consumed} |`,
    );
  }
  if (champion) {
    console.log(`\n### champion arm: ${champion.model}`);
    console.log('| arm | test acc | test ECE |');
    console.log('|---|---|---|');
    console.log(`| jev (frozen replay) | ${jevBlock ? pct(jevBlock.choice_accuracy) : 'n/a'} | ${jevBlock ? f4(jevBlock.ece.ece) : 'n/a'} |`);
    console.log(`| local champion | ${pct(champion.champion.test)} | ${f4(champion.champion.test_ece)} |`);
    console.log(`\nchampion options: ${JSON.stringify(champion.champion.options)}`);
    const s = champion.best_stat;
    if (s) {
      console.log(
        `sequential test (best proposal): wealth max ${s.max_wealth.toFixed(2)} / threshold ${s.threshold.toFixed(0)}, ` +
          `discordant champion/baseline ${s.n_champion_wins}/${s.n_baseline_wins}, rejected=${s.rejected}`,
      );
    }
    console.log(`validation N=${champion.val_n}; budget consumed ${champion.budget_consumed}`);
  }

  // ---- write receipts + campaign receipt ----
  const date = new Date().toISOString().slice(0, 10);
  mkdirSync(RESULTS_DIR, { recursive: true });
  const receiptsPath = join(RESULTS_DIR, `optimize-receipts-${date}.jsonl`);
  writeFileSync(receiptsPath, allReceipts.map((r) => JSON.stringify(r)).join('\n') + (allReceipts.length ? '\n' : ''));

  const campaign = {
    schema: 'ruvector-typesafe-bench/optimize-campaign@1',
    generated_at: new Date().toISOString(),
    suite: 'tickets',
    question: args.question,
    splits_hash: tickets.splitsHash,
    counts: tickets.counts,
    row_count: rows.length,
    budget: args.budget,
    jev_test: jevBlock ? { choice_accuracy: jevBlock.choice_accuracy, ece: jevBlock.ece.ece } : null,
    arms,
    champion: champion ? { model: champion.model, options: champion.champion.options, test: champion.champion } : null,
    receipts_file: `optimize-receipts-${date}.jsonl`,
  };
  const outPath = args.out
    ? (isAbsolute(args.out) ? args.out : join(HERE, '..', args.out))
    : join(RESULTS_DIR, `optimize-tickets-${date}.json`);
  writeFileSync(outPath, JSON.stringify(campaign, null, 2) + '\n');
  console.log(`\nreceipts → ${receiptsPath}`);
  console.log(`campaign receipt → ${outPath}`);
  return 0;
}

try {
  process.exit(main(process.argv.slice(2)));
} catch (e) {
  console.error(`optimize bench failed: ${e && e.stack ? e.stack : e}`);
  process.exit(2);
}
