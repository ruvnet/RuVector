import assert from "node:assert/strict";
import { mkdtemp, readFile } from "node:fs/promises";
import { tmpdir } from "node:os";
import { join } from "node:path";
import test from "node:test";
import type { Score } from "@metaharness/flywheel";
import { runRuvectorFlywheel, ruvectorPromotionRule } from "../src/flywheel.js";

/** Build a gate-visible score. Defaults keep every gate other than
 *  `hard_regression` silent, so a reason list is attributable to one check. */
function gateScore(overrides: {
  primarySamples?: number[];
  recallAt10?: number;
  costPerWin?: number;
  regressions?: string[];
  vetoReasons?: string[];
}): Score {
  const primarySamples = overrides.primarySamples ?? [0.9, 0.9, 0.9, 0.9, 0.9];
  const recallAt10 = overrides.recallAt10 ?? 0.9;
  const regressions = overrides.regressions ?? [];
  const vetoReasons = overrides.vetoReasons ?? [];
  return {
    primary: primarySamples[0]!,
    primarySamples,
    noopRate: 1 - recallAt10,
    recallAt10,
    qps: 2_000,
    memoryMb: 40,
    peakRssBytes: 0,
    costPerWin: overrides.costPerWin ?? 1,
    regressed: regressions.length > 0 || vetoReasons.length > 0,
    regressions,
    vetoReasons,
    observations: [],
  } as unknown as Score;
}

const hardRegressions = (reasons: string[]) =>
  reasons.filter((reason) => reason === "hard_regression" || reason.startsWith("hard_regression:"));

test("hard_regression stays silent when the candidate inherits the baseline's regressions", () => {
  const regressions = ["recall_below_0.95"];
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions }),
    candidate: gateScore({ regressions }),
  });
  assert.deepEqual(hardRegressions(reasons), []);
});

test("hard_regression fires and names a regression the candidate adds to the baseline's", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions: ["recall_below_0.95"] }),
    candidate: gateScore({ regressions: ["recall_below_0.95", "qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression:qps_below_baseline_floor"]);
});

test("hard_regression fires when the candidate trades one regression for a different one", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({ regressions: ["recall_below_0.95"] }),
    candidate: gateScore({ regressions: ["qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression:qps_below_baseline_floor"]);
});

test("hard_regression fires when a clean baseline is replaced by a regressing candidate", () => {
  const { reasons } = ruvectorPromotionRule({
    baseline: gateScore({}),
    candidate: gateScore({ regressions: ["recall_below_0.95", "qps_below_baseline_floor"] }),
  });
  assert.deepEqual(hardRegressions(reasons), [
    "hard_regression:qps_below_baseline_floor",
    "hard_regression:recall_below_0.95",
  ]);
});

test("hard_regression stays silent when neither side regresses", () => {
  const { reasons } = ruvectorPromotionRule({ baseline: gateScore({}), candidate: gateScore({}) });
  assert.deepEqual(hardRegressions(reasons), []);
});

test("a score with no itemised regressions still fails closed on the boolean", () => {
  const legacy = gateScore({});
  delete (legacy as unknown as Record<string, unknown>).regressions;
  (legacy as unknown as Record<string, unknown>).regressed = true;
  const { reasons } = ruvectorPromotionRule({ baseline: gateScore({}), candidate: legacy });
  assert.deepEqual(hardRegressions(reasons), ["hard_regression"]);
});

test("vetoes remain absolute and are not diffed against the baseline", () => {
  const vetoReasons = ["redblue_live_credential_detected"];
  const { promote, reasons } = ruvectorPromotionRule({
    baseline: gateScore({ vetoReasons }),
    candidate: gateScore({ vetoReasons }),
  });
  assert.equal(promote, false);
  assert.ok(reasons.includes("redblue_live_credential_detected"));
});

// The real baseline/candidate pair from the 30-generation acceptance run in issue
// #920: `adapt m`, strictly better on primary, recall, qps, p99 and cost, rejected
// with `hard_regression` as its sole reason because both sides sat below the
// absolute 0.95 recall floor.
test("the issue #920 rejected candidate is no longer blocked by hard_regression", () => {
  const baseline = gateScore({
    primarySamples: [
      0.8781716212500001, 0.8597841212500001, 0.5462757812500001,
      0.8648141012499999, 0.6410117651373053,
    ],
    recallAt10: 0.888,
    costPerWin: 22.6502708788278,
    regressions: ["recall_below_0.95"],
  });
  const candidate = gateScore({
    primarySamples: [
      0.9473974412499999, 0.9617474412500001, 0.9590824612500001,
      0.9598274412500001, 0.9593016212500001,
    ],
    recallAt10: 0.9359999999999999,
    costPerWin: 0.693375479272705,
    regressions: ["recall_below_0.95"],
  });
  const { reasons } = ruvectorPromotionRule({ baseline, candidate });
  assert.deepEqual(hardRegressions(reasons), []);
});

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

interface ScoreShape {
  primary: number;
  primarySamples: number[];
  noopRate: number;
  recallAt10: number;
  qps: number;
  memoryMb: number;
  p99Us: number;
  peakRssBytes: number;
  costPerWin: number;
  regressed: boolean;
  vetoReasons: string[];
  observations: unknown[];
}

function score(primary: number, costPerWin: number, samples: number[]): ScoreShape {
  return {
    primary,
    primarySamples: samples,
    noopRate: 0.04,
    recallAt10: 0.96,
    qps: 2_000,
    memoryMb: 40,
    p99Us: 500,
    peakRssBytes: 0,
    costPerWin,
    regressed: false,
    vetoReasons: [],
    observations: [],
  };
}

const reasonsFor = (baseline: ScoreShape, candidate: ScoreShape): string[] =>
  ruvectorPromotionRule({
    baseline: baseline as never,
    candidate: candidate as never,
  }).reasons;

test("a candidate dominated by the incumbent is refused on the Pareto gate", () => {
  // Equal accuracy bought with strictly worse cost and latency: the baseline
  // dominates it on the objective vector, so the frontier gate must fire.
  const baseline = score(0.9, 10, [0.9, 0.9, 0.9, 0.9, 0.9]);
  const dominated = score(0.9, 20, [0.9, 0.9, 0.9, 0.9, 0.9]);
  assert.ok(reasonsFor(baseline, dominated).includes("pareto_dominated_by_baseline"));
});

test("a candidate better on every objective clears the Pareto gate", () => {
  const baseline = score(0.90, 20, [0.90, 0.90, 0.90, 0.90, 0.90]);
  const better = score(0.95, 10, [0.95, 0.95, 0.95, 0.95, 0.95]);
  const reasons = reasonsFor(baseline, better);
  assert.ok(!reasons.some((reason) => reason.startsWith("pareto_dominated")));
  assert.deepEqual(reasons, []);
});

test("a trade is not domination: better accuracy at worse cost stays on the frontier", () => {
  // Neither point dominates, so the Pareto gate stays silent and the existing
  // cost rule is what decides. The frontier must not become a second, hidden
  // cost gate.
  const baseline = score(0.90, 10, [0.90, 0.90, 0.90, 0.90, 0.90]);
  const traded = score(0.95, 40, [0.95, 0.95, 0.95, 0.95, 0.95]);
  const reasons = reasonsFor(baseline, traded);
  assert.ok(!reasons.some((reason) => reason.startsWith("pareto_dominated")));
  assert.ok(reasons.includes("resource_cost_worsened"));
});

test("unusable objectives block the candidate instead of killing the run", () => {
  // The pure pareto API throws, but @metaharness/flywheel calls this rule with
  // no try/catch from both its generation loop and verifyReplayBundle, so an
  // escaping throw would abort the run and destroy the replay `checks` output.
  // Unusable evidence must REFUSE the candidate, not take down the harness.
  //
  // Every case below is worse on primary (axis 1) and carries the bad value on
  // costPerWin (axis 2) or omits a field entirely — the shapes the comparison
  // loop would short-circuit past.
  const baseline = score(0.9, 10, [0.9, 0.9, 0.9, 0.9, 0.9]);
  const passing = [0.95, 0.95, 0.95, 0.95, 0.95];
  const broken: ScoreShape[] = [
    score(Number.NaN, 10, passing),
    score(0.1, Number.NaN, passing),
    score(0.1, Number.POSITIVE_INFINITY, passing),
    score(0.1, Number.NEGATIVE_INFINITY, passing),
  ];
  for (const candidate of broken) {
    const reasons = reasonsFor(baseline, candidate);
    assert.ok(reasons.includes("non_finite_objective"), JSON.stringify(reasons));
    assert.equal(
      ruvectorPromotionRule({ baseline: baseline as never, candidate: candidate as never }).promote,
      false,
    );
  }
  // A sealed score from before this vector existed: costPerWin absent.
  const legacy = score(0.1, 0, passing) as Partial<ScoreShape>;
  delete legacy.costPerWin;
  const legacyReasons = reasonsFor(baseline, legacy as ScoreShape);
  assert.ok(legacyReasons.includes("non_finite_objective"), JSON.stringify(legacyReasons));
});

test("a candidate cannot supply its own frontier to switch the gate off", () => {
  // An earlier revision read `candidate.frontier`, so an empty array admitted
  // everything silently. The frontier is now the incumbent and nothing on the
  // judged object can influence it.
  const baseline = score(0.9, 10, [0.9, 0.9, 0.9, 0.9, 0.9]);
  const dominated = { ...score(0.5, 1e9, [0.5, 0.5, 0.5, 0.5, 0.5]), frontier: [] };
  assert.ok(reasonsFor(baseline, dominated as ScoreShape).includes("pareto_dominated_by_baseline"));
});
