/**
 * Darwin Mode scorePolicy for the MRAgent graph-memory harness (ADR-269).
 *
 * `metaharness evolve` calls scoreVariant() after each mutation of the harness
 * source (agent/harness.mjs). It evaluates the CURRENT genome over the frozen
 * Cue-Tag-Content corpus and returns a fitness in [0, 1]:
 *
 *   score = 0.70 × accuracy
 *         + 0.15 × (1 − avgLatencyMs / BASE_LATENCY).clamp(0,1)
 *         + 0.10 × (1 − avgContext   / BASE_CONTEXT).clamp(0,1)
 *         + 0.05 × (1 − avgHops      / BASE_HOPS).clamp(0,1)
 *
 * Accuracy dominates (a faster harness that answers wrong is worthless); the
 * remaining weight rewards cheaper reconstruction (lower latency, smaller
 * context, fewer hops) — the MRAgent "prune irrelevant paths" objective.
 *
 * This mirrors crates/ruvector-sota-bench/harness/scorePolicy.ts so the same
 * Darwin tooling drives both the ANN benchmark and the MRAgent harness.
 */

import * as fs from "node:fs";
import * as path from "node:path";
import { fileURLToPath } from "node:url";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const ROOT = path.resolve(__dirname, "..");

// Baselines (the shipped baselineGenome() over this corpus).
const BASE_LATENCY = 4.0;
const BASE_CONTEXT = 6.0;
const BASE_HOPS = 2.0;

interface Metrics {
  accuracy: number;
  avgLatencyMs: number;
  avgHops: number;
  avgContext: number;
  n: number;
}

function fitness(m: Metrics): number {
  const lat = Math.max(0, Math.min(1, 1 - m.avgLatencyMs / BASE_LATENCY));
  const ctx = Math.max(0, Math.min(1, 1 - m.avgContext / BASE_CONTEXT));
  const hop = Math.max(0, Math.min(1, 1 - m.avgHops / BASE_HOPS));
  return 0.7 * m.accuracy + 0.15 * lat + 0.1 * ctx + 0.05 * hop;
}

/**
 * Score the current working-tree harness variant. Darwin mutates the genome
 * defaults inside agent/harness.mjs; we import it fresh and evaluate.
 * Returns a fitness in [0, 1]. Any failure scores 0 (a broken mutation is unfit).
 */
export async function scoreVariant(): Promise<number> {
  try {
    const harness = await import(path.join(ROOT, "agent", "harness.mjs"));
    const corpus = JSON.parse(
      fs.readFileSync(path.join(ROOT, "data", "eval-set.json"), "utf8"),
    );
    const store = new harness.MemoryStore(corpus.tasks);
    const metrics: Metrics = harness.evaluate(
      harness.baselineGenome(),
      store,
      corpus.tasks,
    );
    return Math.max(0, Math.min(1, fitness(metrics)));
  } catch (e) {
    console.error("[scorePolicy] variant failed to evaluate:", (e as Error).message);
    return 0;
  }
}

export { fitness };
export default scoreVariant;
