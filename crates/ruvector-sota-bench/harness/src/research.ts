import { createHash } from "node:crypto";
import { writeFile } from "node:fs/promises";
import { resolve } from "node:path";
import type { FlywheelResult } from "@metaharness/flywheel";
import { aggregateReports } from "./metrics.js";
import type { BenchmarkObservation } from "./benchmark.js";

export interface PairedObservation {
  seed: number;
  control: BenchmarkObservation;
  treatment: BenchmarkObservation;
}

// Keys sort by code unit (RFC 8785), never localeCompare: locale collation is
// locale- and ICU-build-dependent, and this encoding feeds sha256 digests and
// embeddingSpaceId, which must be reproducible across machines (#903).
function canonical(value: unknown): string {
  if (Array.isArray(value)) return `[${value.map(canonical).join(",")}]`;
  if (value && typeof value === "object") {
    return `{${Object.entries(value as Record<string, unknown>)
      .sort(([a], [b]) => (a < b ? -1 : a > b ? 1 : 0))
      .map(([key, child]) => `${JSON.stringify(key)}:${canonical(child)}`).join(",")}}`;
  }
  return JSON.stringify(value);
}

function sha256(value: unknown): string {
  return createHash("sha256").update(canonical(value)).digest("hex");
}

function embeddingSpaceId(identity: Record<string, unknown>): string {
  return createHash("sha256")
    .update("ruvector.embedding-space.v1\0")
    .update(canonical(identity))
    .digest("hex");
}

function numberParam(observation: BenchmarkObservation, name: string): number {
  return Math.max(0, ...observation.report.scores.map((score) => Number(score.params?.[name] ?? 0)));
}

function arm(observation: BenchmarkObservation, primaryMetric: string, primaryResource: string) {
  const metrics = aggregateReports([observation.report]);
  const reportedBytes = Math.round(metrics.memoryMb * 1024 * 1024);
  const payload = numberParam(observation, "encoded_payload_bytes");
  const graph = numberParam(observation, "index_graph_bytes");
  const memory = {
    encoded_payload_bytes: payload,
    index_graph_bytes: graph,
    codebooks_quantizer_bytes: 0,
    container_bookkeeping_bytes: 0,
    allocator_overhead_bytes: Math.max(0, reportedBytes - payload - graph),
    temporary_build_bytes: 0,
    temporary_query_bytes: 0,
    process_peak_rss_bytes: Math.round(observation.resources.processPeakRssBytes),
  };
  const resourceValues: Record<string, number> = {
    resident_memory_bytes: observation.resources.processPeakRssBytes,
    on_disk_bytes: 0,
    query_latency_ns: metrics.p99Us * 1_000,
    search_effort: Number(observation.policy.ef_search),
    build_time_ns: Math.max(0, ...observation.report.scores.map((score) => score.build_secs ?? 0)) * 1e9,
    cpu_seconds: observation.resources.wallTimeMs / 1_000,
  };
  return {
    metrics: {
      [primaryMetric]: metrics.primary,
      recall_at_10: metrics.recallAt10,
      qps: metrics.qps,
      p99_us: metrics.p99Us,
    },
    resources: { [primaryResource]: resourceValues[primaryResource] ?? 0 },
    memory,
  };
}

function assertResearchManifest(manifest: Record<string, unknown>): void {
  if (manifest.schema_version !== 1 || manifest.phase !== "confirmation") {
    throw new Error("research manifest must be schema v1 in confirmation phase");
  }
  const datasets = manifest.datasets as Array<{ kind?: string }> | undefined;
  if (!datasets?.length || datasets.some((dataset) => dataset.kind !== "real")) {
    throw new Error("promotable research manifests require real datasets");
  }
  const embedding = manifest.embedding_space as {
    identity?: Record<string, unknown>;
    embedding_space_id?: string;
  } | undefined;
  if (!embedding?.identity || embedding.embedding_space_id !== embeddingSpaceId(embedding.identity)) {
    throw new Error("embedding_space_id does not match the canonical identity");
  }
}

export async function writeResearchResults(
  manifest: Record<string, unknown>,
  result: FlywheelResult,
  outputDir: string,
  paired: PairedObservation[],
): Promise<void> {
  assertResearchManifest(manifest);
  const promoted = result.replayBundle.chain.find((commit) => commit.verdict === "PROMOTED");
  if (!promoted?.baselineScore || !promoted.candidateScore) {
    throw new Error("no promoted paired evidence is available for confirmation results");
  }
  const seeds = manifest.confirmation_seeds as number[];
  const controls = new Map(paired.map((run) => run.control)
    .map((observation) => [observation.item.seed, observation]));
  const treatments = new Map(paired.map((run) => run.treatment)
    .map((observation) => [observation.item.seed, observation]));
  if (seeds.some((seed) => !controls.has(seed) || !treatments.has(seed))) {
    throw new Error("Flywheel evidence does not cover every preregistered confirmation seed");
  }
  if (seeds.some((seed) =>
    controls.get(seed)!.item.kind !== "real" || treatments.get(seed)!.item.kind !== "real")) {
    throw new Error("research results cannot be emitted from synthetic or smoke observations");
  }
  const rule = manifest.decision_rule as { primary_metric: string };
  const budget = manifest.budget as { primary_resource: string };
  const results = {
    schema_version: 1,
    manifest_revision: manifest.revision,
    manifest_sha256: sha256(manifest),
    candidate_commit: manifest.commit,
    embedding_space_id: (manifest.embedding_space as { embedding_space_id: string }).embedding_space_id,
    phase: "confirmation",
    runs: seeds.map((seed) => ({
      seed,
      control: arm(controls.get(seed)!, rule.primary_metric, budget.primary_resource),
      treatment: arm(treatments.get(seed)!, rule.primary_metric, budget.primary_resource),
      selection: {
        configured_count: 1,
        selected_count: 1,
        deterministic_tie_breaker: "flywheel-primary-then-target-order",
      },
    })),
  };
  await writeFile(resolve(outputDir, "research-manifest.json"), `${JSON.stringify(manifest, null, 2)}\n`);
  await writeFile(resolve(outputDir, "raw-results.json"), `${JSON.stringify(results, null, 2)}\n`);
}
