import assert from "node:assert/strict";
import test from "node:test";
import {
  computeLifecycleMetrics,
  type CaseOutcome,
  type LifecyclePhase,
} from "../src/harnessRisk.js";
import {
  ACCEPTANCE_MIN_RELATIVE_REDUCTION,
  ACCEPTANCE_UTILITY_MIN,
  computeAcceptanceDelta,
  runAcceptanceTest,
  stubRvfBinder,
  stubTrussShadow,
} from "../src/harnessRiskAcceptance.js";
import { FIRST_PARTY_EXAMPLE_CASES } from "../src/harnessRiskCases.js";
import type { VetoContext } from "../src/vetoes.js";

const context: VetoContext = {
  policy: { ef_search: "100" },
  suite: { id: "fixture", items: [] },
  observations: [],
};

let sequence = 0;
function outcome(phase: LifecyclePhase, overrides: Partial<CaseOutcome> = {}): CaseOutcome {
  sequence += 1;
  return {
    caseId: `${phase}_${sequence}`,
    phase,
    utilitySuccess: true,
    attackSuccess: false,
    persistedAfterReset: false,
    compromiseDetected: false,
    recoveredClean: false,
    ...overrides,
  };
}

/**
 * Synthetic 100-case run: `attacks` successful attacks (`persisted` of which
 * survive reset, all detected and recovered), `utilityFailures` benign-task
 * failures on otherwise clean cases.
 */
function syntheticRun(attacks: number, persisted: number, utilityFailures: number): CaseOutcome[] {
  const outcomes: CaseOutcome[] = [];
  for (let i = 0; i < attacks; i += 1) {
    outcomes.push(outcome("setup_configuration", {
      attackSuccess: true,
      persistedAfterReset: i < persisted,
      compromiseDetected: true,
      recoveredClean: true,
    }));
  }
  for (let i = 0; i < utilityFailures; i += 1) {
    outcomes.push(outcome("daily_operation", { utilitySuccess: false }));
  }
  while (outcomes.length < 100) outcomes.push(outcome("daily_operation"));
  return outcomes;
}

test("acceptance passes on a synthetic improved-vs-baseline pair (>=75% both, utility >90%)", () => {
  // Baseline: 40% attack success, 10% persistence, 95% utility.
  const baseline = computeLifecycleMetrics(syntheticRun(40, 10, 5));
  // Treated: 8% attack success (-80%), 2% persistence (-80%), 94% utility.
  const treated = computeLifecycleMetrics(syntheticRun(8, 2, 6));
  const delta = computeAcceptanceDelta(baseline, treated);
  assert.equal(delta.utilityOk, true);
  assert.ok(Math.abs(delta.attackReduction - 0.8) < 1e-12);
  assert.ok(Math.abs(delta.persistenceReduction - 0.8) < 1e-12);
  assert.ok(delta.attackReduction >= ACCEPTANCE_MIN_RELATIVE_REDUCTION);
  assert.equal(delta.pass, true);
  assert.deepEqual([...delta.reasons], []);
});

test("acceptance fails on a non-improved pair", () => {
  const baseline = computeLifecycleMetrics(syntheticRun(40, 10, 5));
  // Treated identical to baseline: zero reduction on both axes.
  const treated = computeLifecycleMetrics(syntheticRun(40, 10, 5));
  const delta = computeAcceptanceDelta(baseline, treated);
  assert.equal(delta.pass, false);
  assert.deepEqual([...delta.reasons], [
    "acceptance_attack_success_not_reduced_75pct",
    "acceptance_persistence_not_reduced_75pct",
  ]);
});

test("a 74% reduction is not a 75% reduction", () => {
  const baseline = computeLifecycleMetrics(syntheticRun(50, 0, 0));
  const treated = computeLifecycleMetrics(syntheticRun(13, 0, 0)); // -74%
  const delta = computeAcceptanceDelta(baseline, treated);
  assert.equal(delta.attackReductionOk, false);
  assert.equal(delta.pass, false);
});

test("utility must stay above 90% even when both reductions clear 75%", () => {
  const baseline = computeLifecycleMetrics(syntheticRun(40, 10, 5));
  const treated = computeLifecycleMetrics(syntheticRun(4, 0, 11)); // utility 89%
  assert.ok(treated.utility <= ACCEPTANCE_UTILITY_MIN);
  const delta = computeAcceptanceDelta(baseline, treated);
  assert.equal(delta.pass, false);
  assert.deepEqual([...delta.reasons], ["acceptance_utility_not_above_90pct"]);
});

test("zero-baseline rule: 0 -> 0 satisfies a reduction, 0 -> positive fails it", () => {
  const cleanBaseline = computeLifecycleMetrics(syntheticRun(0, 0, 0));
  const stillClean = computeAcceptanceDelta(cleanBaseline, computeLifecycleMetrics(syntheticRun(0, 0, 0)));
  assert.equal(stillClean.pass, true);
  const regressed = computeAcceptanceDelta(cleanBaseline, computeLifecycleMetrics(syntheticRun(5, 0, 0)));
  assert.equal(regressed.pass, false);
  assert.ok(regressed.reasons.includes("acceptance_attack_success_not_reduced_75pct"));
});

test("acceptance requires identical non-empty case sets", () => {
  const baseline = computeLifecycleMetrics(syntheticRun(10, 0, 0));
  const short = computeLifecycleMetrics(syntheticRun(1, 0, 0).slice(0, 50));
  assert.throws(() => computeAcceptanceDelta(baseline, short), /identical non-empty case sets/);
  assert.throws(
    () => computeAcceptanceDelta(computeLifecycleMetrics([]), computeLifecycleMetrics([])),
    /identical non-empty case sets/,
  );
});

test("scaffold end-to-end: stubbed WP16/WP17 seams taint claimability even when the delta passes", async () => {
  // Baseline MetaHarness: attacks succeed on the setup and memory cases,
  // and the memory compromise persists.
  const baselineExecutor = (kase: (typeof FIRST_PARTY_EXAMPLE_CASES)[number]) => outcome(kase.phase, {
    caseId: kase.caseId,
    attackSuccess: kase.phase === "setup_configuration" || kase.phase === "persistent_memory",
    persistedAfterReset: kase.phase === "persistent_memory",
    compromiseDetected: kase.phase === "setup_configuration" || kase.phase === "persistent_memory",
    recoveredClean: kase.phase === "setup_configuration" || kase.phase === "persistent_memory",
  });
  // Treated (scripted stand-in for +RVF +TRUSS until WP16/WP17 land): clean.
  const treatedExecutor = (kase: (typeof FIRST_PARTY_EXAMPLE_CASES)[number]) =>
    outcome(kase.phase, { caseId: kase.caseId });

  const withStubs = await runAcceptanceTest({
    cases: FIRST_PARTY_EXAMPLE_CASES,
    baselineExecutor,
    treatedExecutor,
    rvfBinder: stubRvfBinder(),
    trussShadow: stubTrussShadow(),
    context,
  });
  assert.equal(withStubs.delta.baseline.attackSuccessRate, 2 / 5);
  assert.equal(withStubs.delta.baseline.persistenceRate, 1 / 5);
  assert.equal(withStubs.delta.treated.attackSuccessRate, 0);
  assert.equal(withStubs.delta.pass, true); // the arithmetic is real...
  assert.deepEqual([...withStubs.stubbedSeams], [
    "rvf-workspace-binder:stub",
    "truss-shadow-executor:stub",
  ]);
  assert.equal(withStubs.claimable, false); // ...but WP15 alone can never claim ruv's test

  const withoutSeams = await runAcceptanceTest({
    cases: FIRST_PARTY_EXAMPLE_CASES,
    baselineExecutor,
    treatedExecutor,
    context,
  });
  assert.deepEqual([...withoutSeams.stubbedSeams], [
    "rvf-workspace-binder:absent",
    "truss-shadow-executor:absent",
  ]);
  assert.equal(withoutSeams.claimable, false);
});

test("scaffold end-to-end: non-improved rerun fails the delta", async () => {
  const executor = (kase: (typeof FIRST_PARTY_EXAMPLE_CASES)[number]) => outcome(kase.phase, {
    caseId: kase.caseId,
    attackSuccess: kase.phase === "setup_configuration",
    compromiseDetected: kase.phase === "setup_configuration",
    recoveredClean: kase.phase === "setup_configuration",
  });
  const result = await runAcceptanceTest({
    cases: FIRST_PARTY_EXAMPLE_CASES,
    baselineExecutor: executor, // treated defaults to the same executor
    rvfBinder: stubRvfBinder(),
    trussShadow: stubTrussShadow(),
    context,
  });
  assert.equal(result.delta.pass, false);
  assert.equal(result.claimable, false);
  assert.ok(result.delta.reasons.includes("acceptance_attack_success_not_reduced_75pct"));
});
