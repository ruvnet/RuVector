// MRAgent benchmark (v2): baseline vs Darwin-evolved harness over the frozen
// Cue-Tag-Content corpus, plus the consolidation (replay) pass. Reports the three
// beyond-SOTA dimensions: helpfulness (accuracy), calibration (risk + halluc), and
// reconstruction cost (latency/hops/context). Picks up the evolved genome from
// optimize.report.json if present.
//
// Run: npm run benchmark

import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryStore, baselineGenome, evaluate } from "./agent/harness.mjs";
import { consolidate } from "./agent/consolidate.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;

const baseline = baselineGenome();
// Evolved genome: from a prior `npm run optimize`, else a calibrated reference.
let evolved = { ...baseline, fusion: "linear", traversalDepth: 3, abstainThreshold: 0.4, haltConfidence: 0.5, maxContent: 6, tagFanout: 3 };
const reportPath = path.join(__dirname, "optimize.report.json");
if (fs.existsSync(reportPath)) {
  try {
    const rep = JSON.parse(fs.readFileSync(reportPath, "utf8"));
    if (rep?.evolved?.genome) evolved = rep.evolved.genome;
  } catch { /* keep reference */ }
}

const base = evaluate(baseline, new MemoryStore(tasks), tasks);
const evoStore = new MemoryStore(tasks);
const evo = evaluate(evolved, evoStore, tasks);

// Consolidation pass (self-reorganizing memory) on the evolved harness.
const evoPre = evaluate(evolved, evoStore, tasks);
const cons = consolidate(evoStore, tasks, evolved);
const evoPost = evaluate(evolved, evoStore, tasks);

console.log("== MRAgent benchmark (v2 — beyond MRAgent) ==");
console.log(`corpus: ${tasks.length} Cue-Tag-Content tasks (semantic/lexical/hybrid/bridge/distractor/unanswerable)\n`);
console.log("config            accuracy  risk   halluc  latency  hops  context");
const row = (name, m) =>
  console.log(`${name.padEnd(17)} ${(m.accuracy * 100).toFixed(1).padStart(5)}%  ${m.riskScore.toFixed(3)}  ${m.hallucinationRate.toFixed(2)}   ${m.avgLatencyMs.toFixed(2).padStart(5)}   ${m.avgHops.toFixed(2)}  ${m.avgContext.toFixed(1)}`);
row("baseline", base);
row("evolved", evo);
row("evolved+replay", evoPost);

const dAcc = (evo.accuracy - base.accuracy) * 100;
const dRisk = evo.riskScore - base.riskScore;
const dHops = ((evoPre.avgHops - evoPost.avgHops) / Math.max(evoPre.avgHops, 1e-9)) * 100;
console.log(`\nevolved vs baseline: accuracy ${dAcc >= 0 ? "+" : ""}${dAcc.toFixed(1)}pt · risk ${dRisk >= 0 ? "+" : ""}${dRisk.toFixed(3)} · hallucination ${base.hallucinationRate.toFixed(2)} → ${evo.hallucinationRate.toFixed(2)}`);
console.log(`consolidation: ${cons.consolidated} shortcuts → ${dHops.toFixed(1)}% fewer hops at ${(evoPost.accuracy * 100).toFixed(1)}% accuracy`);

const report = {
  frozenModel: "RuVector Cue-Tag-Content graph (frozen)",
  corpusSize: tasks.length,
  baseline: { genome: baseline, metrics: base },
  evolved: { genome: evolved, metrics: evo },
  consolidated: { shortcuts: cons.consolidated, metrics: evoPost },
  deltas: { accuracyPoints: dAcc, riskDelta: dRisk, hopsReductionPct: dHops },
};
fs.writeFileSync(path.join(__dirname, "benchmark.report.json"), JSON.stringify(report, null, 2));
console.log(`\nreport -> ${path.join(__dirname, "benchmark.report.json")}`);
