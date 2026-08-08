import { resolve } from "node:path";
import { bench, evolve, type EvolutionResult } from "@metaharness/darwin";
import {
  gepaOptimize,
  type GepaOptimizeResult,
  type Genome,
} from "@metaharness/darwin/gepa";
import {
  DEFAULT_HOLDOUT_ITEMS,
  DEFAULT_ROOT_POLICY,
  PARAMETER_CANDIDATES,
} from "./flywheel.js";
import {
  normalizePolicy,
  runObservedBenchmark,
  type BenchmarkSuiteItem,
} from "./benchmark.js";
import { aggregateReports, type BenchReport } from "./metrics.js";

export interface RuvectorDarwinOptions {
  repoRoot: string;
  benchSuite: string;
  generations?: number;
  children?: number;
  seed?: number;
}

/**
 * Darwin exploration is always tied to a hash-verified suite and uses the
 * multiple-testing/risk controls expected by ADR-282.
 */
export async function runRuvectorDarwin(options: RuvectorDarwinOptions): Promise<EvolutionResult> {
  const repoRoot = resolve(options.repoRoot);
  const suite = await bench.loadSuite(resolve(options.benchSuite));
  return evolve({
    repoRoot,
    workRoot: resolve(repoRoot, ".metaharness/darwin"),
    generations: options.generations ?? 3,
    childrenPerGeneration: options.children ?? 4,
    tasks: suite.tasks.map((task) => task.id),
    promotionDelta: 0.005,
    concurrency: 2,
    taskTimeoutMs: 300_000,
    costBudgetSeconds: 1_800,
    seed: options.seed ?? 0,
    sandboxMode: "real",
    selection: "pareto",
    crossover: true,
    epistasis: true,
    benchSuite: suite,
    benchSamples: 2_000,
    benchMinDelta: 0.005,
    riskBudgetTotal: 8,
    costCeilingFactor: 1.05,
    fdrQ: 0.05,
    curriculum: true,
    curriculumThreshold: 0.9,
  });
}

export interface RuvectorGepaOptions {
  repoRoot: string;
  items?: BenchmarkSuiteItem[];
  maxCandidates?: number;
  benchmark?: (policy: Record<string, string>, item: BenchmarkSuiteItem) => Promise<BenchReport>;
}

function reflectParameter(prompt: string): string {
  const target = Object.keys(PARAMETER_CANDIDATES)
    .find((name) => prompt.includes(`--- component: ${name} ---`));
  if (!target) throw new Error("Darwin GEPA reflection prompt has no RuVector mutation target");
  const values = PARAMETER_CANDIDATES[target]!;
  const match = new RegExp(`--- component: ${target} ---\\n${target}:value=([^\\n]+)`).exec(prompt);
  const current = match?.[1] ?? values[0]!;
  const next = values[Math.min(values.length - 1, Math.max(0, values.indexOf(current) + 1))]!;
  return `${target}:value=${next}`;
}

function policyFromGenome(genome: Genome): Record<string, string> {
  return Object.fromEntries(Object.entries(genome.components).map(([name, encoded]) => {
    const prefix = `${name}:value=`;
    if (!encoded.startsWith(prefix)) throw new Error(`invalid Darwin component encoding for ${name}`);
    return [name, encoded.slice(prefix.length)];
  }));
}

/**
 * Direct Darwin/GEPA search over ANN parameters. Unlike generic harness
 * evolution, this injected evaluator scores each genome on pinned RuVector
 * dataset/seed instances and preserves the full Pareto frontier.
 */
export async function runRuvectorGepa(options: RuvectorGepaOptions): Promise<GepaOptimizeResult> {
  const items = options.items ?? DEFAULT_HOLDOUT_ITEMS;
  const seed: Genome = {
    version: 1,
    meta: { id: "ruvector-ann-root", source: "ruvector-metaharness" },
    components: Object.fromEntries(
      Object.entries(DEFAULT_ROOT_POLICY).map(([name, value]) => [name, `${name}:value=${value}`]),
    ),
  };
  return gepaOptimize({
    seed,
    mutable: ["ef_search", "m", "ef_construction", "runner_set"],
    maxCandidates: options.maxCandidates ?? 12,
    maxMetricCalls: (options.maxCandidates ?? 12) * items.length,
    evaluate: async (genome) => {
      const policy = normalizePolicy(policyFromGenome(genome));
      const scores: Record<string, number> = {};
      const feedbacks: Record<string, string> = {};
      for (const item of items) {
        const report = options.benchmark
          ? await options.benchmark(policy, item)
          : (await runObservedBenchmark({ repoRoot: options.repoRoot, policy, item })).report;
        const metrics = aggregateReports([report]);
        const id = `${item.dataset}:${item.seed}`;
        scores[id] = metrics.primary;
        feedbacks[id] = metrics.regressions.length
          ? `mutation target: ef_search; ${metrics.regressions.join(",")}`
          : `mutation target: m; recall=${metrics.recallAt10}; qps=${metrics.qps}`;
      }
      return { scores, feedbacks, cost: 0, metricCalls: items.length };
    },
    reflect: async (prompt) => ({ raw: reflectParameter(prompt), cost: 0 }),
    rng: () => 0.5,
  });
}
