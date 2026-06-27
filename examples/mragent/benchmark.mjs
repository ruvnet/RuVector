// MRAgent benchmark: baseline vs Darwin-evolved reconstruction harness over the
// frozen RuVector Cue-Tag-Content corpus. Writes benchmark.report.json and prints
// a per-metric comparison. Picks up the evolved genome from optimize.report.json
// if present; otherwise compares against a hand-set reference genome.
//
// Run: npm run benchmark

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryStore, baselineGenome, evaluate } from "./agent/harness.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
const store = new MemoryStore(tasks);

const baseline = baselineGenome();

// Evolved genome: from a prior `npm run optimize`, else a sensible reference.
let evolved = { ...baseline, traversalDepth: 3, tagFanout: 3, pruneThreshold: 0.1, efSearch: 96, maxContent: 8, promptStrategy: "evidence-first" };
const reportPath = path.join(__dirname, "optimize.report.json");
if (fs.existsSync(reportPath)) {
  try {
    const rep = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    if (rep?.evolved?.genome) evolved = rep.evolved.genome;
  } catch { /* keep reference */ }
}

const base = evaluate(baseline, store, tasks);
const evo = evaluate(evolved, store, tasks);

const pct = (a, b) => (b !== 0 ? ((a - b) / Math.abs(b)) * 100 : 0);
const dAcc = (evo.accuracy - base.accuracy) * 100;            // percentage points
const dLat = pct(base.avgLatencyMs, evo.avgLatencyMs);        // % faster
const dCtx = pct(base.avgContext, evo.avgContext);            // % smaller context

console.log("== MRAgent benchmark ==");
console.log(`corpus: ${tasks.length} Cue-Tag-Content tasks (frozen RuVector memory)\n`);
console.log("config    accuracy   latency(ms)   hops   context");
for (const [name, m] of [["baseline", base], ["evolved", evo]]) {
  console.log(
    `${name.padEnd(9)} ${(m.accuracy * 100).toFixed(1).padStart(5)}%   ${m.avgLatencyMs.toFixed(2).padStart(7)}    ` +
    `${m.avgHops.toFixed(2)}   ${m.avgContext.toFixed(1)}`
  );
}
console.log(`\nevolved vs baseline: accuracy ${dAcc >= 0 ? "+" : ""}${dAcc.toFixed(1)}pt · latency ${dLat.toFixed(1)}% faster · context ${dCtx.toFixed(1)}% smaller`);

const report = {
  frozenModel: "RuVector Cue-Tag-Content graph (frozen)",
  corpusSize: tasks.length,
  baseline: { genome: baseline, metrics: base },
  evolved: { genome: evolved, metrics: evo },
  deltas: { accuracyPoints: dAcc, latencyPctFaster: dLat, contextPctSmaller: dCtx },
};
fs.writeFileSync(path.join(__dirname, "benchmark.report.json"), JSON.stringify(report, null, 2));
console.log(`\nreport -> ${path.join(__dirname, "benchmark.report.json")}`);
