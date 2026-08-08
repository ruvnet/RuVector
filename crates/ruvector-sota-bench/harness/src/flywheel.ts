import { createHash } from "node:crypto";
import { mkdir, writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import {
  gateFingerprint,
  makeSigner,
  runFlywheelGenerations,
  verifyReplayBundle,
  type FlywheelResult,
  type Policy,
  type PolicyGenome,
  type Score,
  type Suite,
} from "@metaharness/flywheel";
import {
  normalizePolicy,
  normalizeSuiteItem,
  runObservedBenchmark,
  type BenchmarkObservation,
  type BenchmarkSuiteItem,
} from "./benchmark.js";
import { aggregateReports, type BenchReport } from "./metrics.js";
import { pairedBootstrapDecision } from "./statistics.js";
import type { PromotionVetoProvider } from "./vetoes.js";
import { writeResearchResults, type PairedObservation } from "./research.js";

const hashDataset = (name: string) => createHash("sha256").update(`ruvector:${name}:v1`).digest("hex");

export const PARAMETER_CANDIDATES: Record<string, string[]> = {
  ef_search: ["32", "64", "100", "128", "200", "256", "400", "800"],
  m: ["8", "12", "16", "24", "32", "48", "64"],
  ef_construction: ["64", "100", "128", "200", "256", "400"],
  k: ["10"],
  runner_set: ["core", "core-lsm", "all"],
};

export const DEFAULT_ROOT_POLICY: Policy = {
  ef_search: "32",
  m: "16",
  ef_construction: "100",
  k: "10",
  runner_set: "core",
};

export const DEFAULT_HOLDOUT_ITEMS: BenchmarkSuiteItem[] = [11, 29, 47, 71, 101].map((seed) => ({
  seed, dataset: "smoke-128", dataset_sha256: hashDataset("smoke-128"), kind: "smoke",
}));
export const DEFAULT_ANCHOR_ITEMS: BenchmarkSuiteItem[] = [131, 173, 211, 251, 293].map((seed) => ({
  seed, dataset: "smoke-96", dataset_sha256: hashDataset("smoke-96"), kind: "smoke",
}));

interface RuvectorScore extends Score {
  primarySamples: number[];
  recallAt10: number;
  qps: number;
  memoryMb: number;
  peakRssBytes: number;
  vetoReasons: string[];
  observations: BenchmarkObservation[];
}

export interface RuvectorFlywheelOptions {
  repoRoot: string;
  generations?: number;
  outputDir?: string;
  embeddingSpaceId?: string;
  holdoutItems?: BenchmarkSuiteItem[];
  anchorItems?: BenchmarkSuiteItem[];
  benchmark?: (policy: Record<string, string>, item: BenchmarkSuiteItem, suite: Suite) =>
    Promise<BenchReport | BenchmarkObservation>;
  vetoProvider?: PromotionVetoProvider;
  researchManifest?: Record<string, unknown>;
}

function asRuvectorScore(score: Score): RuvectorScore {
  return score as RuvectorScore;
}

export function ruvectorPromotionRule(evidence: {
  baseline: Score;
  candidate: Score;
  anchor?: { baseline: number; candidate: number };
}) {
  const baseline = asRuvectorScore(evidence.baseline);
  const candidate = asRuvectorScore(evidence.candidate);
  const decision = pairedBootstrapDecision(baseline.primarySamples, candidate.primarySamples);
  const reasons: string[] = [];
  if (decision.outcome !== "pass") reasons.push(`paired_confirmation_${decision.outcome}`);
  if (candidate.recallAt10 < baseline.recallAt10) reasons.push("recall_regressed");
  if (candidate.costPerWin > baseline.costPerWin * 1.05) reasons.push("resource_cost_worsened");
  if (candidate.regressed) reasons.push("hard_regression");
  reasons.push(...candidate.vetoReasons);
  if (evidence.anchor && evidence.anchor.candidate < evidence.anchor.baseline) reasons.push("anchor_regressed");
  return { promote: reasons.length === 0, reasons };
}

function nextParameter(base: PolicyGenome, target: string): string {
  const policy = normalizePolicy(base.policy);
  const values = PARAMETER_CANDIDATES[target];
  if (!values) throw new Error(`unsupported mutation target: ${target}`);
  const current = policy[target]!;
  const index = values.indexOf(current);
  return values[Math.min(values.length - 1, Math.max(0, index + 1))]!;
}

async function toScore(
  policy: Policy,
  suite: Suite,
  options: RuvectorFlywheelOptions,
): Promise<RuvectorScore> {
  const items = suite.items.map(normalizeSuiteItem);
  const observations: BenchmarkObservation[] = [];
  for (const item of items) {
    const result = options.benchmark
      ? await options.benchmark(policy, item, suite)
      : await runObservedBenchmark({
          repoRoot: options.repoRoot,
          policy,
          item,
          ...(options.embeddingSpaceId ? { embeddingSpaceId: options.embeddingSpaceId } : {}),
        });
    observations.push("report" in result ? result : {
      report: result,
      item,
      policy: normalizePolicy(policy),
      resources: {
        processPeakRssBytes: 0,
        rawProcessPeakRssBytes: 0,
        baselineRssBytes: 0,
        sampleIntervalMs: 10,
        samples: 0,
        wallTimeMs: 0,
        cacheHit: false,
      },
      cacheKey: "injected",
    });
  }
  const perItem = observations.map((observation) => aggregateReports([observation.report]));
  const aggregate = aggregateReports(observations.map((observation) => observation.report));
  const vetoReasons = options.vetoProvider
    ? await options.vetoProvider({ policy, suite, observations })
    : [];
  return {
    primary: aggregate.primary,
    primarySamples: perItem.map((metrics) => metrics.primary),
    noopRate: 1 - aggregate.recallAt10,
    recallAt10: aggregate.recallAt10,
    qps: aggregate.qps,
    memoryMb: aggregate.memoryMb,
    peakRssBytes: Math.max(...observations.map((observation) => observation.resources.processPeakRssBytes)),
    costPerWin: aggregate.memoryMb * Math.max(aggregate.p99Us, 1) / Math.max(aggregate.qps, 1),
    regressed: aggregate.regressions.length > 0 || vetoReasons.length > 0,
    vetoReasons,
    observations,
  };
}

export async function runRuvectorFlywheel(options: RuvectorFlywheelOptions): Promise<FlywheelResult> {
  const outputDir = resolve(options.outputDir ?? resolve(options.repoRoot, ".metaharness/ruvector"));
  await mkdir(outputDir, { recursive: true });
  const holdout = { id: "ruvector-confirmation", items: options.holdoutItems ?? DEFAULT_HOLDOUT_ITEMS };
  const anchor = { id: "ruvector-frozen-anchor", items: options.anchorItems ?? DEFAULT_ANCHOR_ITEMS };
  const holdoutKeys = new Set(holdout.items.map((item) => `${item.dataset}:${item.seed}`));
  if (anchor.items.some((item) => holdoutKeys.has(`${item.dataset}:${item.seed}`))) {
    throw new Error("holdout and anchor dataset/seed identities must be disjoint");
  }

  const result = await runFlywheelGenerations({
    rootPolicy: DEFAULT_ROOT_POLICY,
    proposer: async (base, target) => nextParameter(base, target),
    evaluator: (policy, suite) => toScore(policy, suite, options),
    promotionRule: ruvectorPromotionRule,
    holdout,
    anchor,
    mutationTargets: ["ef_search", "m", "ef_construction", "runner_set"],
    maxGenerations: options.generations ?? 4,
    signer: makeSigner(),
    dataSource: holdout.items.every((item) => item.kind === "real") ? "REAL" : "SYNTHETIC_SMOKE",
    now: (generation) => `generation-${generation}`,
    onGeneration: async ({ generation, partialBundle, resumeState }) => {
      await writeFile(resolve(outputDir, `checkpoint-${generation}.json`),
        `${JSON.stringify({ partialBundle, resumeState }, null, 2)}\n`, { mode: 0o600 });
    },
  });

  const fingerprint = gateFingerprint(ruvectorPromotionRule);
  const replay = verifyReplayBundle(result.replayBundle, {
    pinnedGateFingerprint: fingerprint,
    promotionRule: ruvectorPromotionRule,
  });
  if (!replay.pass) throw new Error(`Flywheel replay verification failed: ${JSON.stringify(replay.checks)}`);
  await writeFile(resolve(outputDir, "replay-bundle.json"), `${JSON.stringify(result.replayBundle, null, 2)}\n`);
  await writeFile(resolve(outputDir, "research-gate-input.json"), `${JSON.stringify({
    schema_version: 1,
    producer: "@ruvector/sota-metaharness",
    gate_fingerprint: fingerprint,
    replay_verified: true,
    requires_confirmation: true,
    pr_creation_authorized: false,
    embedding_space_id: options.embeddingSpaceId ?? null,
    holdout_items: holdout.items,
    anchor_items: anchor.items,
    final_policy: result.finalPolicy,
    generations_run: result.generationsRun,
  }, null, 2)}\n`);
  if (options.researchManifest) {
    if (options.benchmark) {
      throw new Error("research artifact emission requires the isolated native benchmark runner");
    }
    const paired: PairedObservation[] = [];
    for (let index = 0; index < holdout.items.length; index += 1) {
      const item = holdout.items[index]!;
      const run = (policy: Policy) => runObservedBenchmark({
        repoRoot: options.repoRoot,
        policy,
        item,
        embeddingSpaceId: options.embeddingSpaceId!,
        useCache: false,
      });
      let control: BenchmarkObservation;
      let treatment: BenchmarkObservation;
      if (index % 2 === 0) {
        control = await run(DEFAULT_ROOT_POLICY);
        treatment = await run(result.finalPolicy);
      } else {
        treatment = await run(result.finalPolicy);
        control = await run(DEFAULT_ROOT_POLICY);
      }
      paired.push({ seed: item.seed, control, treatment });
    }
    await writeResearchResults(options.researchManifest, result, outputDir, paired);
  }
  return result;
}
