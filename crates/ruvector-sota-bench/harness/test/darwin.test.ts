import assert from "node:assert/strict";
import test from "node:test";
import { runRuvectorGepa } from "../src/darwin.js";

test("Darwin GEPA evaluates RuVector genomes on each pinned suite item", async () => {
  const calls: string[] = [];
  const result = await runRuvectorGepa({
    repoRoot: process.cwd(),
    maxCandidates: 3,
    items: [11, 29].map((seed) => ({
      seed,
      dataset: "smoke-128",
      dataset_sha256: "a".repeat(64),
      kind: "smoke" as const,
    })),
    benchmark: async (policy, item) => {
      calls.push(`${policy.ef_search}:${item.seed}`);
      return {
        scores: [{
          index: "core-hnsw",
          dataset: item.dataset,
          recall: { recall_at_10: 0.96 },
          qps: 1000 + Number(policy.ef_search),
          memory_mb: 40,
          latency: { p99_us: 500 },
        }],
      };
    },
  });
  assert.equal(result.pool.length, 1);
  assert.equal(calls.length, 6);
  assert.ok(result.frontier.length > 0);
  assert.equal(result.budget.metricCalls, 6);
});
