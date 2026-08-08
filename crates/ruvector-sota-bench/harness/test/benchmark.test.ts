import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { runBenchmarkBatch, runObservedBenchmark } from "../src/benchmark.js";

const item = {
  seed: 11,
  dataset: "smoke-128",
  dataset_sha256: "a".repeat(64),
  kind: "smoke" as const,
};
const fixture = resolve(import.meta.dirname, "../../test/fixtures/fake-benchmark.mjs");

test("native sweep batches ef_search values and disk cache reuses exact evidence", async () => {
  const cacheDir = await mkdtemp(join(tmpdir(), "ruvector-cache-test-"));
  const common = {
    repoRoot: resolve(import.meta.dirname, "../../../../.."),
    binary: process.execPath,
    commandPrefixArgs: [fixture],
    cacheDir,
    item,
  };
  const policies = ["32", "64", "100"].map((ef_search) => ({ ef_search }));
  const batch = await runBenchmarkBatch({ ...common, policies });
  assert.deepEqual([...batch.keys()], ["32", "64", "100"]);
  assert.ok([...batch.values()].every((report) => report.scores.length === 1));

  const first = await runObservedBenchmark({ ...common, policy: policies[0]! });
  const cached = await runObservedBenchmark({ ...common, policy: policies[0]! });
  assert.equal(first.resources.cacheHit, false);
  assert.equal(cached.resources.cacheHit, true);
  assert.equal(cached.cacheKey, first.cacheKey);
});
