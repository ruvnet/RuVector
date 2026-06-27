// MRAgent harness optimizer — Darwin Mode for graph-memory reconstruction.
//
// Principle (Meta-Harness / @metaharness/darwin): "freeze the model, evolve the
// harness." FROZEN MODEL = the RuVector Cue-Tag-Content memory substrate
// (agent/memory.mjs). EVOLVED HARNESS = the reconstruction genome in
// agent/harness.mjs (cue-k, efSearch, RRF alpha, traversal depth, fan-out, prune
// threshold, content limit, GNN rerank, prompt strategy).
//
// We use Darwin's `mapLimit` (bounded-concurrency evaluation) and `paretoFront`
// (multi-objective selection) when @metaharness/darwin is installed, and fall
// back to an equivalent in-process loop when it is not (ADR-150 invariant 3:
// graceful degradation — MODULE_NOT_FOUND must never crash the example).
//
// Run: npm run optimize

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryStore, baselineGenome, mutate, evaluate, splitByClass, kFoldByClass, runReasoningLoop } from "./agent/harness.mjs";
import { consolidate } from "./agent/consolidate.mjs";
import { detectEndpoint, llmProposeGenomes } from "./agent/llmMutator.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));

// ── ADR-150 graceful degradation: optional Darwin primitives ────────────────
async function loadDarwin() {
  try {
    const d = await import("@metaharness/darwin");
    console.log("[darwin] @metaharness/darwin loaded — using mapLimit + paretoFront");
    return { mapLimit: d.mapLimit, paretoFront: d.paretoFront, available: true };
  } catch (e) {
    if (e.code !== "ERR_MODULE_NOT_FOUND" && e.code !== "MODULE_NOT_FOUND") throw e;
    console.warn("[darwin] @metaharness/darwin not installed — using built-in evolution loop");
    return { mapLimit: localMapLimit, paretoFront: localParetoFront, available: false };
  }
}

// Minimal local stand-ins (identical contracts to the Darwin exports).
async function localMapLimit(items, _limit, fn) {
  const out = [];
  for (let i = 0; i < items.length; i++) out.push(await fn(items[i], i));
  return out;
}
function localParetoFront(items, objFn) {
  const objs = items.map(objFn);
  return items.filter((_, i) =>
    !items.some((_, j) => j !== i && dominates(objs[j], objs[i])));
}
function dominates(a, b) {
  let strictly = false;
  for (let k = 0; k < a.length; k++) {
    if (a[k] < b[k]) return false;
    if (a[k] > b[k]) strictly = true;
  }
  return strictly;
}

// ── Scoring — the Darwin fitness (see harness/scorePolicy.ts for the canonical
//    version used by `metaharness evolve`). Helpfulness (accuracy) AND calibration
//    (risk-adjusted utility — abstain instead of hallucinate) both dominate;
//    reconstruction cost (latency, hops, context) is penalised vs the baseline. ──
const BASE = { latency: 4.0, hops: 2.0, context: 6.0 };
function scalar(m) {
  const latTerm = Math.max(0, 1 - m.avgLatencyMs / BASE.latency);
  const hopTerm = Math.max(0, 1 - m.avgHops / BASE.hops);
  const ctxTerm = Math.max(0, 1 - m.avgContext / BASE.context);
  return 0.40 * m.accuracy + 0.30 * m.riskScore + 0.12 * latTerm + 0.10 * ctxTerm + 0.08 * hopTerm;
}
// Pareto maximises every component (negate minimised objectives).
function objectives(m) {
  return [m.accuracy, m.riskScore, -m.avgLatencyMs, -m.avgHops, -m.avgContext];
}

// ── Run ─────────────────────────────────────────────────────────────────────
const { mapLimit, paretoFront, available } = await loadDarwin();

// ── GPU LLM write-layer (opt-in): a local code model proposes genome leaps from
//    failure traces, the directed-search layer the random GA lacks (ADR-260).
//    Disabled with MRAGENT_LLM=off; otherwise auto-detects a local endpoint. ──
const llm = process.env.MRAGENT_LLM === "off" ? null : await detectEndpoint();
if (llm) console.log(`[llm] GPU write-layer: ${llm.model} @ ${llm.url}`);
else console.log("[llm] no local LLM endpoint — GA-only (set MRAGENT_LLM_URL to enable)");
let llmProposed = 0, llmEnteredElite = 0;

const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
// ONE memory holds all nodes (full cross-task cue competition); we evolve on the
// TRAIN queries only and report held-out TEST to prove the genome generalizes.
const store = new MemoryStore(tasks);
const { train, test } = splitByClass(tasks, 0.6);
const folds = kFoldByClass(train, 3); // cross-validation folds over the train pool

// Cross-validated fitness: mean fold score MINUS the fold range. The penalty
// rejects genomes that win on one fold but collapse on another (e.g. a knife-edge
// abstainThreshold), which is exactly the overfit a single split hides.
function cvScore(genome) {
  const fs2 = folds.map((f) => scalar(evaluate(genome, store, f)));
  const mean = fs2.reduce((a, b) => a + b, 0) / fs2.length;
  const range = Math.max(...fs2) - Math.min(...fs2);
  return mean - 0.5 * range;
}

// Compact failure trace for the LLM write-layer: tasks the genome gets wrong /
// hallucinates, with its confidence (so the model can reason about thresholds).
function failureTraces(genome, tasks, limit = 6) {
  const out = [];
  for (const t of tasks) {
    if (out.length >= limit) break;
    const isAns = t.answerable !== false;
    const r = runReasoningLoop(store.queryText(t.id), store, genome, t);
    if (isAns && !r.correct) {
      out.push(`${t.id}[${t.class ?? "?"}]: ${r.abstained ? "abstained-on-answerable" : "wrong"} conf=${r.confidence.toFixed(2)}`);
    } else if (!isAns && !r.abstained) {
      out.push(`${t.id}[${t.class ?? "?"}]: hallucinated-on-unanswerable conf=${r.confidence.toFixed(2)}`);
    }
  }
  return out.join("\n") || "none";
}
const llmGenomes = []; // every coerced LLM proposal, for end-of-run attribution

const POP = 16, GENERATIONS = 12, ELITE = 5, CONCURRENCY = 4;
const baseline = baselineGenome();
const baseMetrics = evaluate(baseline, store, train);

let population = [baseline, ...Array.from({ length: POP - 1 }, () => mutate(baseline))];
let best = { genome: baseline, metrics: baseMetrics, score: cvScore(baseline) };
const archive = [];
const history = [];

console.log("== MRAgent · Darwin harness optimizer (v2 — beyond MRAgent) ==");
console.log(`frozen model: RuVector Cue-Tag-Content graph (${tasks.length} tasks) | train ${train.length} / test ${test.length} (held out)`);
console.log(`baseline (train): acc ${(baseMetrics.accuracy * 100).toFixed(1)}% risk ${baseMetrics.riskScore.toFixed(3)} halluc ${baseMetrics.hallucinationRate.toFixed(2)}\n`);

for (let gen = 0; gen < GENERATIONS; gen++) {
  const scored = await mapLimit(population, CONCURRENCY, async (genome) => {
    const metrics = evaluate(genome, store, train);
    return { genome, metrics, score: cvScore(genome) };
  });
  archive.push(...scored);

  const front = paretoFront(scored, (e) => objectives(e.metrics));
  const winner = scored.reduce((a, b) => (b.score > a.score ? b : a));
  if (winner.score > best.score) best = winner;

  history.push({
    gen,
    best: { accuracy: winner.metrics.accuracy, avgLatencyMs: winner.metrics.avgLatencyMs, score: winner.score },
    frontSize: front.length,
  });
  console.log(
    `gen ${gen}: acc ${(winner.metrics.accuracy * 100).toFixed(1)}% risk ${winner.metrics.riskScore.toFixed(3)} ` +
    `halluc ${winner.metrics.hallucinationRate.toFixed(2)} lat ${winner.metrics.avgLatencyMs.toFixed(2)}ms hops ${winner.metrics.avgHops.toFixed(2)} ` +
    `score ${winner.score.toFixed(4)} · pareto ${front.length}`
  );

  // Next generation: elites + mutated children + a couple of random restarts to
  // keep diversity (the built-in loop has no LLM write-layer to propose leaps).
  const elites = [...scored].sort((a, b) => b.score - a.score).slice(0, ELITE).map((e) => e.genome);
  const next = [...elites];

  // GPU LLM write-layer: every 3rd generation, ask the local code model for
  // directed genome leaps from the current winner's failure traces. Proposals
  // are bounds-clamped in llmMutator, so they can only ever be safe genomes.
  if (llm && gen % 3 === 0) {
    const traces = failureTraces(winner.genome, train);
    const props = await llmProposeGenomes({ url: llm.url, model: llm.model, baseline, current: winner.genome, failures: traces, n: 2 });
    for (const g of props) {
      llmGenomes.push(g);
      llmProposed++;
      if (next.length < POP) next.push(g);
    }
    if (props.length) console.log(`  [llm] gen ${gen}: +${props.length} GPU-proposed genome(s) injected`);
  }

  const RESTARTS = 2;
  for (let r = 0; r < RESTARTS && next.length < POP; r++) {
    let g = baseline;
    for (let m = 0; m < 6; m++) g = mutate(g); // heavy random walk
    next.push(g);
  }
  while (next.length < POP) next.push(mutate(elites[Math.floor(Math.random() * elites.length)]));
  population = next;
}

// Fold GPU-proposed genomes into the archive so they compete in polish +
// acceptance on equal footing with GA candidates.
let llmBest = -Infinity;
for (const g of llmGenomes) {
  const e = { genome: g, metrics: evaluate(g, store, train), score: cvScore(g) };
  archive.push(e);
  if (e.score > llmBest) llmBest = e.score;
  if (e.score > best.score) { best = e; llmEnteredElite++; }
}
if (llm) {
  console.log(`\n[llm] GPU write-layer: ${llmProposed} genome(s) proposed, best cv-score ${llmBest > -Infinity ? llmBest.toFixed(4) : "n/a"}${llmEnteredElite ? `, became GA-best ${llmEnteredElite}×` : ""}`);
}

// ── Memetic polish — deterministic coordinate descent over each gene ─────────
// The GA explores broadly but the LLM-free fallback struggles with NARROW optima
// (e.g. the abstainThreshold band that catches hallucinations without abstaining
// on correct answers). A final hill-climb over a per-gene candidate grid finds
// them reliably and makes the shipped result reproducible. (The real Darwin
// write-layer proposes such leaps directly from failure traces — ADR-260.)
const GRID = {
  cueK: [1, 2, 3, 4, 6, 8],
  efSearch: [16, 24, 32, 48, 64, 96, 128],
  hybridAlpha: [0, 0.2, 0.35, 0.5, 0.65, 0.8, 1],
  fusion: ["rrf", "linear", "dbsf"],
  traversalDepth: [1, 2, 3, 4],
  tagFanout: [1, 2, 3, 4, 6, 8],
  pruneThreshold: [0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.4],
  maxContent: [1, 2, 3, 4, 6, 8, 12],
  haltConfidence: [0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9],
  rerank: ["gnn", "none"],
  promptStrategy: ["terse", "evidence-first", "prune-explicit"],
  abstainThreshold: [0, 0.1, 0.2, 0.3, 0.34, 0.36, 0.38, 0.4, 0.45, 0.5],
};
function localPolish(genome) {
  let cur = { ...genome };
  let curScore = cvScore(cur); // cross-validated over train folds — never see test
  for (let pass = 0; pass < 3; pass++) {
    let improved = false;
    for (const [gene, candidates] of Object.entries(GRID)) {
      for (const v of candidates) {
        if (cur[gene] === v) continue;
        const cand = { ...cur, [gene]: v };
        const s = cvScore(cand);
        if (s > curScore + 1e-9) { cur = cand; curScore = s; improved = true; }
      }
    }
    if (!improved) break;
  }
  return { genome: cur, score: curScore };
}
// Multi-start polish: greedy coordinate descent is start-dependent, so refine from
// several diverse seeds (GA winner + baseline + top archive elites) and keep the
// global best. This makes the calibrated optimum reproducible across runs.
const seeds = [best.genome, baseline, ...[...archive].sort((a, b) => b.score - a.score).slice(0, 4).map((e) => e.genome)];
for (const seed of seeds) {
  const polished = localPolish(seed);
  if (polished.score > best.score) best = { genome: polished.genome, metrics: evaluate(polished.genome, store, train), score: polished.score };
}
console.log(`\n[polish] multi-start coordinate-descent (train) → score ${best.score.toFixed(4)} (acc ${(best.metrics.accuracy * 100).toFixed(1)}% risk ${best.metrics.riskScore.toFixed(3)} halluc ${best.metrics.hallucinationRate.toFixed(2)})`);

// ── Acceptance gate over the whole archive ──────────────────────────────────
const gate = (m) => {
  const accGain = m.accuracy - baseMetrics.accuracy;
  const riskGain = m.riskScore - baseMetrics.riskScore;
  const noRegress = m.accuracy >= baseMetrics.accuracy - 1e-9 && m.riskScore >= baseMetrics.riskScore - 1e-9;
  return { accGain, riskGain, noRegress, passed: noRegress && (accGain >= 0.04 || riskGain >= 0.04) };
};
const passers = [best, ...archive]
  .map((e) => ({ e, g: gate(e.metrics) }))
  .filter((x) => x.g.passed)
  .sort((a, b) => (b.e.score - a.e.score));
const accepted = passers[0]?.e ?? best;
const acc = gate(accepted.metrics);

console.log("\n-- acceptance gate (over archive) --");
console.log(`candidates evaluated: ${archive.length} | gate-passing: ${passers.length}`);
console.log(`accepted: acc ${(accepted.metrics.accuracy * 100).toFixed(1)}% (${acc.accGain >= 0 ? "+" : ""}${(acc.accGain * 100).toFixed(1)}pt) · risk ${accepted.metrics.riskScore.toFixed(3)} (${acc.riskGain >= 0 ? "+" : ""}${acc.riskGain.toFixed(3)}) · halluc ${accepted.metrics.hallucinationRate.toFixed(2)}`);
console.log(passers.length ? "PASS — Pareto-superior harness found (freeze model, evolve harness)" : "no gate-passing variant this run");

// ── Generalization: held-out TEST split (never seen during evolution) ────────
// Generalization criterion = does evolving on TRAIN improve UNSEEN test? (not an
// absolute accuracy bar — the synthetic toy embedding has per-instance noise, and
// a single global hybridAlpha cannot perfectly serve both dense- and sparse-keyed
// queries; the question that matters is whether optimization transfers.)
const baseTest = evaluate(baseline, store, test);
const evoTest = evaluate(accepted.genome, store, test);
const generalizes = evoTest.accuracy >= baseTest.accuracy + 0.10 && evoTest.hallucinationRate <= baseTest.hallucinationRate + 1e-9;
console.log("\n-- generalization (held-out test split, never seen in evolution) --");
console.log(`baseline test: acc ${(baseTest.accuracy * 100).toFixed(1)}% risk ${baseTest.riskScore.toFixed(3)} halluc ${baseTest.hallucinationRate.toFixed(2)}`);
console.log(`evolved  test: acc ${(evoTest.accuracy * 100).toFixed(1)}% risk ${evoTest.riskScore.toFixed(3)} halluc ${evoTest.hallucinationRate.toFixed(2)}`);
console.log(`gain: +${((evoTest.accuracy - baseTest.accuracy) * 100).toFixed(1)}pt acc, +${(evoTest.riskScore - baseTest.riskScore).toFixed(3)} risk on unseen tasks`);
console.log(generalizes ? "GENERALIZES — evolution transfers to unseen tasks (not overfit)" : "WARNING — evolved genome does not transfer");

// ── Replay/consolidation pass on the accepted genome (self-reorganizing memory) ─
const memAfter = new MemoryStore(tasks);
const evoMetricsPre = evaluate(accepted.genome, memAfter, tasks);
const consolidation = consolidate(memAfter, tasks, accepted.genome);
const evoMetricsPost = evaluate(accepted.genome, memAfter, tasks);
console.log(`\n-- consolidation (replay) on accepted genome --`);
console.log(`shortcuts laid: ${consolidation.consolidated} | avgHops ${evoMetricsPre.avgHops.toFixed(3)} -> ${evoMetricsPost.avgHops.toFixed(3)} (${(((evoMetricsPre.avgHops - evoMetricsPost.avgHops) / evoMetricsPre.avgHops) * 100).toFixed(1)}% fewer) at acc ${(evoMetricsPost.accuracy * 100).toFixed(1)}%`);

const report = {
  tool: "metaharness/darwin",
  philosophy: "freeze the model, evolve the harness",
  frozenModel: "RuVector Cue-Tag-Content graph memory (agent/memory.mjs)",
  darwinAvailable: available,
  primitivesUsed: ["mapLimit", "paretoFront"],
  gpuWriteLayer: llm
    ? { endpoint: llm.url, model: llm.model, proposed: llmProposed, bestCvScore: llmBest > -Infinity ? llmBest : null, becameBest: llmEnteredElite }
    : { enabled: false },
  split: { train: train.length, test: test.length },
  baseline: { trainMetrics: baseMetrics, testMetrics: baseTest },
  evolved: { genome: accepted.genome, trainMetrics: accepted.metrics, testMetrics: evoTest, score: accepted.score },
  generalizes,
  consolidation: { shortcuts: consolidation.consolidated, avgHopsBefore: evoMetricsPre.avgHops, avgHopsAfter: evoMetricsPost.avgHops, metricsAfter: evoMetricsPost },
  acceptance: acc,
  history,
};
fs.writeFileSync(path.join(__dirname, "optimize.report.json"), JSON.stringify(report, null, 2));
console.log(`\nreport -> ${path.join(__dirname, "optimize.report.json")}`);
