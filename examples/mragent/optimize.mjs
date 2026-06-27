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
//    version used by `metaharness evolve`). Accuracy dominates; reconstruction
//    cost (latency, hops, context) is penalised against the baseline. ─────────
const BASE = { latency: 4.0, hops: 2.0, context: 6.0 };
function scalar(m) {
  const latTerm = Math.max(0, 1 - m.avgLatencyMs / BASE.latency);
  const hopTerm = Math.max(0, 1 - m.avgHops / BASE.hops);
  const ctxTerm = Math.max(0, 1 - m.avgContext / BASE.context);
  return 0.7 * m.accuracy + 0.15 * latTerm + 0.1 * ctxTerm + 0.05 * hopTerm;
}
// Pareto maximises every component (negate minimised objectives).
function objectives(m) {
  return [m.accuracy, -m.avgLatencyMs, -m.avgHops, -m.avgContext];
}

// ── Run ─────────────────────────────────────────────────────────────────────
const { mapLimit, paretoFront, available } = await loadDarwin();
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
const store = new MemoryStore(tasks);

const POP = 12, GENERATIONS = 8, ELITE = 4, CONCURRENCY = 4;
const baseline = baselineGenome();
const baseMetrics = evaluate(baseline, store, tasks);

let population = [baseline, ...Array.from({ length: POP - 1 }, () => mutate(baseline))];
let best = { genome: baseline, metrics: baseMetrics, score: scalar(baseMetrics) };
const archive = [];
const history = [];

console.log("== MRAgent · Darwin harness optimizer ==");
console.log(`frozen model: RuVector Cue-Tag-Content graph (${tasks.length} tasks) | evolving reconstruction genome`);
console.log(`baseline: acc ${(baseMetrics.accuracy * 100).toFixed(1)}% lat ${baseMetrics.avgLatencyMs.toFixed(2)}ms hops ${baseMetrics.avgHops.toFixed(2)} ctx ${baseMetrics.avgContext.toFixed(1)}\n`);

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
    `gen ${gen}: acc ${(winner.metrics.accuracy * 100).toFixed(1)}% lat ${winner.metrics.avgLatencyMs.toFixed(2)}ms ` +
    `hops ${winner.metrics.avgHops.toFixed(2)} ctx ${winner.metrics.avgContext.toFixed(1)} ` +
    `score ${winner.score.toFixed(4)} · pareto ${front.length}`
  );

  // Next generation: elites + mutated children of elites.
  const elites = [...scored].sort((a, b) => b.score - a.score).slice(0, ELITE).map((e) => e.genome);
  const next = [...elites];
  while (next.length < POP) next.push(mutate(elites[Math.floor(Math.random() * elites.length)]));
  population = next;
}

// ── Acceptance gate over the whole archive ──────────────────────────────────
const gate = (m) => {
  const accGain = m.accuracy - baseMetrics.accuracy;
  const latGain = (baseMetrics.avgLatencyMs - m.avgLatencyMs) / Math.max(baseMetrics.avgLatencyMs, 1e-6);
  const noRegress = m.accuracy >= baseMetrics.accuracy - 1e-9;
  return { accGain, latGain, noRegress, passed: noRegress && (accGain >= 0.05 || latGain >= 0.2) };
};
const passers = archive
  .map((e) => ({ e, g: gate(e.metrics) }))
  .filter((x) => x.g.passed)
  .sort((a, b) => (b.e.score - a.e.score));
const accepted = passers[0]?.e ?? best;
const acc = gate(accepted.metrics);

console.log("\n-- acceptance gate (over archive) --");
console.log(`candidates evaluated: ${archive.length} | gate-passing: ${passers.length}`);
console.log(`accepted: acc ${(accepted.metrics.accuracy * 100).toFixed(1)}% (${acc.accGain >= 0 ? "+" : ""}${(acc.accGain * 100).toFixed(1)}pt) · latency ${(acc.latGain * 100).toFixed(1)}% faster · no-regress ${acc.noRegress}`);
console.log(passers.length ? "PASS — Pareto-superior harness found (freeze model, evolve harness)" : "no gate-passing variant this run");

const report = {
  tool: "metaharness/darwin",
  philosophy: "freeze the model, evolve the harness",
  frozenModel: "RuVector Cue-Tag-Content graph memory (agent/memory.mjs)",
  darwinAvailable: available,
  primitivesUsed: ["mapLimit", "paretoFront"],
  baseline: { genome: baseline, metrics: baseMetrics },
  evolved: { genome: accepted.genome, metrics: accepted.metrics, score: accepted.score },
  acceptance: acc,
  history,
};
fs.writeFileSync(path.join(__dirname, "optimize.report.json"), JSON.stringify(report, null, 2));
console.log(`\nreport -> ${path.join(__dirname, "optimize.report.json")}`);
