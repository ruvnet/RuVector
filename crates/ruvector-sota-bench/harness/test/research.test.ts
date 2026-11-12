import assert from "node:assert/strict";
import { spawnSync } from "node:child_process";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import type { FlywheelResult, Score } from "@metaharness/flywheel";
import type { BenchmarkObservation } from "../src/benchmark.js";
import { writeResearchResults } from "../src/research.js";

const repoRoot = resolve(import.meta.dirname, "../../../../..");
const identityFixturePath = resolve(repoRoot, "schemas/fixtures/embedding-space-identity-v1.json");

function observation(seed: number, qps: number): BenchmarkObservation {
  return {
    item: {
      seed,
      dataset: "sift-128-euclidean",
      dataset_sha256: "d".repeat(64),
      dataset_file: "/datasets/sift.hdf5",
      kind: "real",
    },
    policy: {
      ef_search: "100",
      m: "32",
      ef_construction: "200",
      k: "10",
      runner_set: "core",
    },
    report: {
      scores: [{
        index: "core-hnsw",
        dataset: "sift-128-euclidean",
        recall: { recall_at_10: 0.98 },
        qps,
        memory_mb: 100,
        build_secs: 1,
        latency: { p99_us: 1_000 },
        params: { encoded_payload_bytes: "1000", index_graph_bytes: "500" },
      }],
    },
    resources: {
      processPeakRssBytes: 100_000,
      rawProcessPeakRssBytes: 120_000,
      baselineRssBytes: 20_000,
      sampleIntervalMs: 10,
      samples: 10,
      wallTimeMs: 1_000,
      cacheHit: false,
    },
    cacheKey: `${seed}:${qps}`,
  };
}

test("generated confirmation artifacts pass the trusted ADR-282 evaluator", async () => {
  const fixture = JSON.parse(await readFile(identityFixturePath, "utf8")) as {
    base: { identity: Record<string, unknown>; embedding_space_id: string };
  };
  const seeds = [11, 29, 47, 71, 101];
  const memoryComponents = [
    "encoded_payload_bytes", "index_graph_bytes", "codebooks_quantizer_bytes",
    "container_bookkeeping_bytes", "allocator_overhead_bytes",
    "temporary_build_bytes", "temporary_query_bytes", "process_peak_rss_bytes",
  ];
  const manifest: Record<string, unknown> = {
    schema_version: 1,
    commit: "a".repeat(40),
    revision: 1,
    phase: "confirmation",
    claim: "the candidate improves Darwin score by at least 0.005",
    independent_variable: "ANN policy",
    decision_rule: {
      primary_metric: "darwin_score",
      minimum_meaningful_effect: 0.005,
      expected_direction: "greater",
      alpha: 0.05,
      comparison: "paired",
      sample_size_or_power_rationale: "Five paired deterministic seeds are the initial confirmation floor.",
      outcome_rules: {
        pass: "ci_lower_at_or_above_minimum_meaningful_effect",
        fail: "ci_upper_at_or_below_zero",
        inconclusive: "otherwise",
      },
    },
    datasets: [{
      name: "sift-128-euclidean",
      kind: "real",
      source: "/datasets/sift.hdf5",
      sha256: "d".repeat(64),
      sampling: "seeded query rotation",
      held_out: true,
    }],
    embedding_space: {
      ...fixture.base,
      query_api: "embed_query",
      passage_api: "embed_passage",
    },
    exploration_seeds: [131, 173, 211, 251, 293],
    confirmation_seeds: seeds,
    topology: {
      claimed: "core-hnsw",
      implemented: "core-hnsw",
      configuration: { m: 32 },
    },
    budget: {
      primary_resource: "resident_memory_bytes",
      tolerance: 0.05,
      indivisible_allocation_unit: 0,
      secondary_resources: ["query_latency_ns"],
    },
    memory_accounting: {
      components: memoryComponents,
      peak_rss_protocol: {
        version: "ruvector-metaharness-v1",
        isolation: "one child process per arm",
        baseline_subtracted: true,
        warmup_operations: 0,
        sampling_interval_ms: 10,
        allocator: "system",
        cgroup_limit_bytes: 1_000_000_000,
        page_size_bytes: 4096,
        process_count: 1,
        thread_count: 1,
        child_process_policy: "no descendants",
        filesystem_cache_included: true,
        cache_state: "warm",
        kernel: "fixture",
        alternating_order: true,
      },
    },
    environment: { fixture: true },
    commands: ["sota-all"],
    evaluator_version: "research-gate-v1",
    artifact_retention_class: "candidate",
    selection: {
      candidate_family: "single Flywheel winner",
      selection_rule: "frozen signed promotion gate",
      confirmation_error_control: "single-candidate",
    },
  };
  const score = (qps: number): Score & { observations: BenchmarkObservation[] } => ({
    primary: qps,
    noopRate: 0,
    costPerWin: 1,
    regressed: false,
    observations: seeds.map((seed) => observation(seed, qps)),
  });
  const result = {
    replayBundle: {
      gate_fingerprint: "f".repeat(64),
      chain: [{
        verdict: "PROMOTED",
        baselineScore: score(500),
        candidateScore: score(1_000),
      }],
    },
  } as unknown as FlywheelResult;
  const outputDir = await mkdtemp(join(tmpdir(), "ruvector-research-test-"));
  await writeResearchResults(manifest, result, outputDir, seeds.map((seed) => ({
    seed,
    control: observation(seed, 500),
    treatment: observation(seed, 1_000),
  })));
  const evaluation = resolve(outputDir, "evaluation.json");
  const run = spawnSync("python3", [
    resolve(repoRoot, "scripts/research-gate/research_gate.py"),
    "evaluate",
    resolve(outputDir, "research-manifest.json"),
    resolve(outputDir, "raw-results.json"),
    "--output",
    evaluation,
  ], { encoding: "utf8" });
  assert.equal(run.status, 0, run.stderr);
  const summary = JSON.parse(await readFile(evaluation, "utf8")) as { outcome: string };
  assert.equal(summary.outcome, "pass");
});
