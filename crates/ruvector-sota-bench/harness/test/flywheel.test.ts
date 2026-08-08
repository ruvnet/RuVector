import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import { runRuvectorFlywheel } from "../src/flywheel.js";

test("flywheel emits a replay-verified bundle and cannot authorize a PR", async () => {
  const outputDir = await mkdtemp(join(tmpdir(), "ruvector-flywheel-test-"));
  const seenSuites = new Set<string>();
  const seenItems = new Set<string>();
  const result = await runRuvectorFlywheel({
    repoRoot: process.cwd(),
    outputDir,
    generations: 2,
    benchmark: async (policy, item, suite) => {
      seenSuites.add(suite.id);
      seenItems.add(`${item.dataset}:${item.seed}`);
      const ef = Number(policy.ef_search);
      return {
        scores: [{
          index: "fixture",
          dataset: "fixture",
          recall: { recall_at_10: 0.96 + Math.min(ef, 200) / 10_000 },
          qps: 2_000,
          memory_mb: 40,
          latency: { p99_us: 500 },
        }],
      };
    },
  });
  assert.equal(result.generationsRun, 2);
  assert.deepEqual([...seenSuites].sort(), ["ruvector-confirmation", "ruvector-frozen-anchor"]);
  assert.equal(seenItems.size, 10);
  const gateInput = JSON.parse(await readFile(join(outputDir, "research-gate-input.json"), "utf8")) as {
    replay_verified: boolean;
    pr_creation_authorized: boolean;
  };
  assert.equal(gateInput.replay_verified, true);
  assert.equal(gateInput.pr_creation_authorized, false);
});

test("overlapping holdout and anchor identities fail closed", async () => {
  const item = {
    seed: 7,
    dataset: "smoke-128",
    dataset_sha256: "a".repeat(64),
    kind: "smoke" as const,
  };
  await assert.rejects(() => runRuvectorFlywheel({
    repoRoot: process.cwd(),
    generations: 1,
    holdoutItems: [item],
    anchorItems: [item],
    benchmark: async () => ({ scores: [] }),
  }), /must be disjoint/);
});
