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
import { MemoryStore, baselineGenome, mutate, evaluate } from "./agent/harness.mjs";
import { consolidate } from "./agent/consolidate.mjs";

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
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
const store = new MemoryStore(tasks);

const POP = 16, GENERATIONS = 12, ELITE = 5, CONCURRENCY = 4;
const baseline = baselineGenome();
const baseMetrics = evaluate(baseline, store, tasks);

let population = [baseline, ...Array.from({ length: POP - 1 }, () => mutate(baseline))];
let best = { genome: baseline, metrics: baseMetrics, score: scalar(baseMetrics) };
const archive = [];
const history = [];

console.log("== MRAgent · Darwin harness optimizer (v2 — beyond MRAgent) ==");
console.log(`frozen model: RuVector Cue-Tag-Content graph (${tasks.length} tasks) | evolving 12-gene reconstruction genome`);
console.log(`baseline: acc ${(baseMetrics.accuracy * 100).toFixed(1)}% risk ${baseMetrics.riskScore.toFixed(3)} halluc ${baseMetrics.hallucinationRate.toFixed(2)} lat ${baseMetrics.avgLatencyMs.toFixed(2)}ms hops ${baseMetrics.avgHops.toFixed(2)}\n`);

for (let gen = 0; gen < GENERATIONS; gen++) {
  const scored = await mapLimit(population, CONCURRENCY, async (genome) => {
    const metrics = evaluate(genome, store, tasks);
    return { genome, metrics, score: scalar(metrics) };
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
  const RESTARTS = 2;
  for (let r = 0; r < RESTARTS && next.length < POP; r++) {
    let g = baseline;
    for (let m = 0; m < 6; m++) g = mutate(g); // heavy random walk
    next.push(g);
  }
  while (next.length < POP) next.push(mutate(elites[Math.floor(Math.random() * elites.length)]));
  population = next;
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
  let curScore = scalar(evaluate(cur, store, tasks));
  for (let pass = 0; pass < 3; pass++) {
    let improved = false;
    for (const [gene, candidates] of Object.entries(GRID)) {
      for (const v of candidates) {
        if (cur[gene] === v) continue;
        const cand = { ...cur, [gene]: v };
        const s = scalar(evaluate(cand, store, tasks));
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
  if (polished.score > best.score) best = { genome: polished.genome, metrics: evaluate(polished.genome, store, tasks), score: polished.score };
}
console.log(`\n[polish] multi-start coordinate-descent → score ${best.score.toFixed(4)} (acc ${(best.metrics.accuracy * 100).toFixed(1)}% risk ${best.metrics.riskScore.toFixed(3)} halluc ${best.metrics.hallucinationRate.toFixed(2)})`);

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
  baseline: { genome: baseline, metrics: baseMetrics },
  evolved: { genome: accepted.genome, metrics: accepted.metrics, score: accepted.score },
  consolidation: { shortcuts: consolidation.consolidated, avgHopsBefore: evoMetricsPre.avgHops, avgHopsAfter: evoMetricsPost.avgHops, metricsAfter: evoMetricsPost },
  acceptance: acc,
  history,
};
fs.writeFileSync(path.join(__dirname, "optimize.report.json"), JSON.stringify(report, null, 2));
console.log(`\nreport -> ${path.join(__dirname, "optimize.report.json")}`);
