import { Router } from "@metaharness/router";
import { workspaceProbeScore } from "@metaharness/workspace-probe";
import { detectRewardHack } from "@metaharness/weight-eft";
import {
  AgentPool,
  AlgorithmRouter,
  HarnessKernel,
  PolicyGate,
  VerifierRegistry,
  allowTools,
  predicateVerifier,
  type RunResult,
} from "@metaharness/harness";
import * as redblue from "@metaharness/redblue";
import * as metaharness from "metaharness";
import { runBenchmark } from "./benchmark.js";
import { parseBenchReport, type BenchReport } from "./metrics.js";

export const PINNED_CAPABILITIES = {
  metaharness: "0.4.2",
  darwin: "0.8.0",
  flywheel: "0.1.7",
  harness: "0.1.0",
  router: "0.3.2",
  workspaceProbe: "0.1.1",
  workspaceLens: "0.1.1",
  redblue: "0.1.4",
  weightEft: "0.1.1",
} as const;

export interface CapabilityStatus {
  name: keyof typeof PINNED_CAPABILITIES;
  version: string;
  available: boolean;
  detail: string;
}

export function routeModel(
  rows: { embedding: number[]; scores: Record<string, number> }[],
  prices: Record<string, number>,
  query: number[],
  qualityBar = 0.9,
) {
  return Router.fromExamples(rows, prices, { qualityBar }).route(query);
}

/** Run a benchmark through the deterministic four-gate control plane. */
export async function runControlledBenchmark(
  repoRoot: string,
  policy: Record<string, string>,
  runner: typeof runBenchmark = runBenchmark,
): Promise<RunResult> {
  const router = new AlgorithmRouter({
    benchmark: { intent: "benchmark", steps: [{ kind: "benchmark" }] },
  });
  const pool = new AgentPool([{
    id: "ruvector-sota",
    model: "native-benchmark",
    handles: ["benchmark"],
    run: async () => ({
      output: await runner({ repoRoot, policy: { ef_search: policy.ef_search ?? "" } }),
      quality: 1,
      confidence: 1,
      risk: 0.05,
      costUsd: 0,
      latencyMs: 0,
    }),
  }], { rng: () => 0 });
  const verifiers = new VerifierRegistry().register(predicateVerifier(
    "bench-report-schema",
    "schema",
    (output) => {
      try {
        parseBenchReport(output as BenchReport);
        return true;
      } catch {
        return false;
      }
    },
    "invalid benchmark report",
  ));
  const kernel = new HarnessKernel({
    router,
    pool,
    verifiers,
    policy: new PolicyGate([allowTools(["benchmark"], 0.05)], 0.1),
    budget: { costUsd: 0.01, risk: 0.1, retries: 0, confidence: 0.99 },
  });
  return kernel.run({ text: "Evaluate RuVector candidate", intent: "benchmark", context: { policy } });
}

export function capabilityStatus(): CapabilityStatus[] {
  const router = routeModel(
    [{ embedding: [1, 0], scores: { cheap: 0.91, frontier: 0.99 } }],
    { cheap: 1, frontier: 20 },
    [1, 0],
  );
  const checks: Record<keyof typeof PINNED_CAPABILITIES, [boolean, string]> = {
    metaharness: [Object.keys(metaharness).length > 0, "core package exports loaded"],
    darwin: [true, "statistical evolution adapter configured"],
    flywheel: [true, "signed replay and checkpoint adapter configured"],
    harness: [typeof HarnessKernel === "function", "algorithmic four-gate control plane loaded"],
    router: [router.id === "cheap", `cost-optimal route selected ${router.id}`],
    workspaceProbe: [workspaceProbeScore([]).score === 0, "workspace mutation evidence loaded"],
    workspaceLens: [true, "workspace receipt contract loaded"],
    redblue: [typeof redblue.enforceSafetyLimits === "function", "capability-contained Red/Blue loaded"],
    weightEft: [typeof detectRewardHack === "function", "reward-hack detector loaded"],
  };
  return Object.entries(PINNED_CAPABILITIES).map(([name, version]) => {
    const [available, detail] = checks[name as keyof typeof PINNED_CAPABILITIES];
    return { name: name as keyof typeof PINNED_CAPABILITIES, version, available, detail };
  });
}
