#!/usr/bin/env node
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";
import {
  ai4aiEvidenceClass,
  assertAi4aiMutation,
  assertAi4aiTaskManifest,
  commandAi4aiExecutor,
  runAi4aiLineage,
  verifyAi4aiLineage,
} from "./ai4aiBench.js";
import type { BenchmarkSuiteItem } from "./benchmark.js";
import { capabilityStatus } from "./capabilities.js";
import { runRuvectorDarwin, runRuvectorGepa } from "./darwin.js";
import { runRuvectorFlywheel } from "./flywheel.js";

function value(flag: string): string | undefined {
  const index = process.argv.indexOf(flag);
  return index < 0 ? undefined : process.argv[index + 1];
}

function integer(flag: string, fallback: number): number {
  const raw = value(flag);
  if (raw === undefined) return fallback;
  const parsed = Number(raw);
  if (!Number.isSafeInteger(parsed) || parsed < 1) throw new Error(`${flag} must be a positive integer`);
  return parsed;
}

async function main(): Promise<void> {
  const command = process.argv[2] ?? "doctor";
  const repoRoot = resolve(value("--repo") ?? resolve(import.meta.dirname, "../../../../.."));
  if (command === "doctor") {
    const status = capabilityStatus();
    process.stdout.write(`${JSON.stringify({ ok: status.every((item) => item.available), status }, null, 2)}\n`);
    if (status.some((item) => !item.available)) process.exitCode = 1;
    return;
  }
  if (command === "flywheel") {
    const outputDir = value("--output");
    const manifestPath = value("--manifest");
    const researchManifest = manifestPath
      ? JSON.parse(await readFile(resolve(manifestPath), "utf8")) as Record<string, unknown>
      : undefined;
    const dataset = researchManifest
      ? (researchManifest.datasets as Array<{ name: string; source: string; sha256: string }>)[0]
      : undefined;
    const makeItems = (seeds: number[]): BenchmarkSuiteItem[] => seeds.map((seed) => ({
      seed,
      dataset: dataset!.name,
      dataset_sha256: dataset!.sha256,
      kind: "real",
      dataset_file: resolve(dataset!.source),
    }));
    const embeddingSpaceId = researchManifest
      ? (researchManifest.embedding_space as { embedding_space_id: string }).embedding_space_id
      : undefined;
    const result = await runRuvectorFlywheel({
      repoRoot,
      generations: integer("--generations", 4),
      ...(outputDir ? { outputDir } : {}),
      ...(researchManifest ? {
        researchManifest,
        embeddingSpaceId: embeddingSpaceId!,
        holdoutItems: makeItems(researchManifest.confirmation_seeds as number[]),
        anchorItems: makeItems(researchManifest.exploration_seeds as number[]),
      } : {}),
    });
    process.stdout.write(`${JSON.stringify({
      generations: result.generationsRun,
      finalPolicy: result.finalPolicy,
      promotions: result.promotions.length,
      replayBundle: resolve(outputDir ?? resolve(repoRoot, ".metaharness/ruvector"), "replay-bundle.json"),
    }, null, 2)}\n`);
    return;
  }
  if (command === "darwin") {
    const benchSuite = value("--bench");
    if (!benchSuite) throw new Error("darwin requires --bench <hash-pinned-suite.json>");
    const result = await runRuvectorDarwin({
      repoRoot,
      benchSuite,
      generations: integer("--generations", 3),
      children: integer("--children", 4),
      seed: integer("--seed", 1),
    });
    process.stdout.write(`${JSON.stringify({
      generations: result.generations,
      winner: result.winner?.variant.id ?? null,
      lineage: result.winnerLineage,
    }, null, 2)}\n`);
    return;
  }
  if (command === "darwin-ann") {
    const result = await runRuvectorGepa({
      repoRoot,
      maxCandidates: integer("--candidates", 12),
    });
    process.stdout.write(`${JSON.stringify({
      best: result.best,
      bestMean: result.bestMean,
      frontier: result.frontier,
      budget: result.budget,
    }, null, 2)}\n`);
    return;
  }
  if (command === "darwin-ai4ai") {
    // AI4AI-Bench (arXiv:2608.20318) lineage run against an injected external
    // evaluator entrypoint. The real evaluator needs Docker + datacenter GPU
    // hardware; --executor points at whatever entrypoint the caller pins
    // (a fixture locally, the real evaluator on capable hardware).
    const manifestPath = value("--manifest");
    const mutationsPath = value("--mutations");
    const executorPath = value("--executor");
    if (!manifestPath || !mutationsPath || !executorPath) {
      throw new Error("darwin-ai4ai requires --manifest, --mutations, and --executor");
    }
    const manifest = JSON.parse(await readFile(resolve(manifestPath), "utf8")) as unknown;
    assertAi4aiTaskManifest(manifest);
    const mutations = JSON.parse(await readFile(resolve(mutationsPath), "utf8")) as unknown;
    if (!Array.isArray(mutations)) throw new Error("--mutations must be a JSON array");
    for (const mutation of mutations) assertAi4aiMutation(mutation);
    // JS entrypoints (fixtures, node-based wrappers) are run via this node;
    // anything else must be directly executable.
    const resolvedExecutor = resolve(executorPath);
    // --wrapper-indirection declares that the pinned entrypoint WRAPS the
    // evaluator, waiving the evaluator-identity binding. It must be asked for
    // explicitly and is recorded on every lineage record.
    const wrapperIndirection = process.argv.includes("--wrapper-indirection");
    const lineage = await runAi4aiLineage(
      manifest,
      mutations,
      commandAi4aiExecutor({
        executorPath: resolvedExecutor,
        ...(wrapperIndirection ? { wrapperIndirection: true } : {}),
        ...(/\.(mjs|cjs|js)$/.test(resolvedExecutor)
          ? { commandPrefixArgs: [process.execPath] }
          : {}),
      }),
    );
    // --require-evaluator-bound demands byte-derived evidence for every record.
    const vetoReasons = verifyAi4aiLineage(manifest, lineage, {
      requireEvaluatorBound: process.argv.includes("--require-evaluator-bound"),
    });
    process.stdout.write(`${JSON.stringify({
      task: manifest.taskId,
      runNonce: lineage.runNonce,
      evidenceClass: ai4aiEvidenceClass(lineage),
      records: lineage.records,
      vetoReasons,
      bestScore: Math.max(...lineage.records.map((record) => record.evaluation.score)),
    }, null, 2)}\n`);
    // A vetoed run must not read as success to automation keying on exit status.
    if (vetoReasons.length > 0) process.exitCode = 1;
    return;
  }
  throw new Error("usage: ruvector-metaharness <doctor|flywheel|darwin|darwin-ann|darwin-ai4ai> [options]");
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
