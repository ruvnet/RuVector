#!/usr/bin/env node
import { resolve } from "node:path";
import { readFile } from "node:fs/promises";
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
  throw new Error("usage: ruvector-metaharness <doctor|flywheel|darwin|darwin-ann> [options]");
}

main().catch((error: unknown) => {
  process.stderr.write(`${error instanceof Error ? error.message : String(error)}\n`);
  process.exitCode = 1;
});
