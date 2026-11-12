import assert from "node:assert/strict";
import { mkdtemp } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join, resolve } from "node:path";
import test from "node:test";
import { canonical, runBenchmarkBatch, runObservedBenchmark } from "../src/benchmark.js";

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

test("cache-key canonicalization is locale-independent", () => {
  // Regression test for #903: canonical() used to sort keys with
  // localeCompare, so the benchmark cache key depended on the machine's
  // locale and ICU build. localeCompare orders {z_metric, ä_metric, a_metric}
  // as a,z,ä under sv_SE but a,ä,z under en_US — different bytes, different
  // cache key. Code-unit order (RFC 8785) must hold even when the ambient
  // collator actively disagrees with it.
  const expected = canonical({ z_metric: 1, "ä_metric": 2, a_metric: 3 });
  const original = Intl.Collator;
  try {
    // Force any locale-sensitive comparison to disagree with code-unit order.
    (Intl as { Collator: unknown }).Collator = class {
      compare = (a: string, b: string) => (a > b ? -1 : a < b ? 1 : 0);
    };
    assert.equal(canonical({ a_metric: 3, "ä_metric": 2, z_metric: 1 }), expected);
  } finally {
    (Intl as { Collator: unknown }).Collator = original;
  }
  assert.equal(
    expected,
    '{"a_metric":3,"z_metric":1,"ä_metric":2}',
    "keys must serialize in code-unit order",
  );
});
