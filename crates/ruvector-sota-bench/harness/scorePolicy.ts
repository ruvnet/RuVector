/**
 * Darwin score surface retained for MetaHarness compatibility.
 *
 * The benchmark is launched without a shell and its output is aggregated across
 * all configurations. A missing/invalid benchmark fails closed instead of
 * silently substituting a trace heuristic.
 */
import type { RunTrace } from "@metaharness/darwin";
import { resolve } from "node:path";
import { runBenchmark } from "./src/benchmark.js";
import { aggregateReports } from "./src/metrics.js";

export async function scoreVariant(_traces: RunTrace[]): Promise<number> {
  const repoRoot = resolve(import.meta.dirname, "../../../..");
  const report = await runBenchmark({
    repoRoot,
    policy: { ef_search: process.env.RUVECTOR_EF_SEARCH ?? "100" },
  });
  return aggregateReports([report]).primary;
}
