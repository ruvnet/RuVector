import assert from "node:assert/strict";
import test from "node:test";
import { aggregateReports, darwinScore, parseBenchReport } from "../src/metrics.js";
import { normalizePolicy } from "../src/benchmark.js";
import { capabilityStatus, routeModel, runControlledBenchmark } from "../src/capabilities.js";

const score = {
  index: "hnsw",
  dataset: "fixture",
  recall: { recall_at_10: 0.98 },
  qps: 1_000,
  memory_mb: 100,
  latency: { p99_us: 1_000 },
};

test("metrics are validated and aggregated without best-run cherry-picking", () => {
  const report = parseBenchReport({ scores: [score, { ...score, qps: 500 }] });
  const metrics = aggregateReports([report]);
  assert.equal(metrics.qps, 750);
  assert.equal(metrics.memoryMb, 100);
  assert.ok(metrics.primary <= darwinScore(score));
});

test("policy input is allowlisted and bounded", () => {
  assert.deepEqual(normalizePolicy({ ef_search: "0100" }), {
    ef_search: "100",
    m: "32",
    ef_construction: "200",
    k: "10",
    runner_set: "core",
  });
  assert.throws(() => normalizePolicy({ ef_search: "100; touch /tmp/pwn" }));
  assert.throws(() => normalizePolicy({ ef_search: "100", other: "1" }));
});

test("router chooses the cheapest model above the quality bar", () => {
  const selected = routeModel(
    [{ embedding: [1, 0], scores: { cheap: 0.92, frontier: 0.99 } }],
    { cheap: 1, frontier: 20 },
    [1, 0],
    0.9,
  );
  assert.equal(selected.id, "cheap");
});

test("all pinned capability probes load", () => {
  assert.ok(capabilityStatus().every((capability) => capability.available));
});

test("algorithmic control plane verifies the benchmark and its receipt chain", async () => {
  const result = await runControlledBenchmark(process.cwd(), { ef_search: "100" }, async () => ({
    scores: [score],
  }));
  assert.equal(result.success, true);
  assert.equal(result.receiptsValid, true);
  assert.equal(result.steps[0]?.gate.ok, true);
});
