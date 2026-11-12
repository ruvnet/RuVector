/**
 * Tests for the Zetta-pattern three-timescale evolution harness — synthetic
 * slice (PIR WP24, ADR-327, issue #883; Zetta = arXiv:2608.16590).
 *
 * SCOPE: synthetic environment only. No physical robotics; the physical /
 * RuView actuation target is deferred to a future ADR + hardware. These tests
 * validate the timescale-architecture pattern, NOT any Zetta robot number.
 */
import assert from "node:assert/strict";
import test from "node:test";
import * as tse from "../src/timescaleEvolution.js";
import {
  defaultLocalCritic,
  runLocalCriticTier,
  runRecoveryTier,
  recoveryStrategyForFlags,
  assertModelFreeCritic,
  assertDisjointSplit,
  runSkillEvolutionTier,
  runTimescaleGeneration,
  TIER_ORDER,
  type CriticThresholds,
  type LocalCritic,
  type SyntheticObservation,
  type SyntheticRollout,
} from "../src/timescaleEvolution.js";
import type {
  CandidateMutation,
  EvolvableGenome,
  FrozenModelRef,
  SkillGenome,
} from "../src/genome.js";

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

const FROZEN_MODEL: FrozenModelRef = {
  id: "ruvltra-frozen-q4",
  sha256: "a".repeat(64),
};

const THRESHOLDS: CriticThresholds = {
  minSignalQuality: 0.5,
  maxStateAgeMs: 100,
  minConfidence: 0.6,
  anomalyAbs: 10,
};

const obs = (partial: Partial<SyntheticObservation> = {}): SyntheticObservation => ({
  seed: 1,
  step: 0,
  signalQuality: 0.9,
  stateAgeMs: 10,
  confidence: 0.9,
  value: 1,
  ...partial,
});

const parentSkills: SkillGenome = {
  surface: "skills",
  version: 3,
  skills: [{ name: "grasp", body: "reach, close gripper" }],
};

const skillMutation = (extra?: Partial<CandidateMutation>): CandidateMutation => ({
  parent: parentSkills,
  candidate: {
    surface: "skills",
    version: 4,
    skills: [
      { name: "grasp", body: "reach, close gripper" },
      { name: "regrasp-on-slip", body: "detect slip, reopen, retry grasp" },
    ],
  },
  describe: "add regrasp-on-slip recovery skill",
  ...extra,
});

// Held-out baseline: five synthetic seeds the champion scores ~0.90 on.
const HELD_OUT_SEEDS = [101, 103, 105, 107, 109];
const TRAIN_SEEDS = [2, 4, 6, 8];
const HELD_OUT_BASELINE = [0.90, 0.91, 0.89, 0.90, 0.92];

// ───────────────────────────── ms tier: LOCAL CRITIC ─────────────────────

test("ms local critic flags signal-quality / staleness / confidence / anomaly", () => {
  const critic = defaultLocalCritic(THRESHOLDS);
  assert.deepEqual(runLocalCriticTier(critic, obs()).signal.flags, []);
  assert.equal(runLocalCriticTier(critic, obs()).action, "proceed");

  assert.deepEqual(runLocalCriticTier(critic, obs({ signalQuality: 0.1 })).signal.flags, ["signal_quality"]);
  assert.deepEqual(runLocalCriticTier(critic, obs({ stateAgeMs: 5000 })).signal.flags, ["stale_state"]);
  assert.deepEqual(runLocalCriticTier(critic, obs({ confidence: 0.1 })).signal.flags, ["low_confidence"]);
  assert.deepEqual(runLocalCriticTier(critic, obs({ value: 99 })).signal.flags, ["anomaly"]);

  const gated = runLocalCriticTier(critic, obs({ confidence: 0.1, value: 99 }));
  assert.equal(gated.action, "gated");
  assert.deepEqual(gated.signal.flags, ["low_confidence", "anomaly"]);
  assert.equal(gated.signal.tier, "ms-local-critic");
  assert.ok(Object.isFrozen(gated));
});

test("ms critic is STRUCTURALLY FAST and CANNOT reach the frozen model", () => {
  const critic = defaultLocalCritic(THRESHOLDS);

  // 1. Structural: the tier is synchronous — its result is a plain object,
  //    never a thenable. The only path to the frozen base policy is the async
  //    dream-machine evaluation the minutes tier uses; a synchronous function
  //    cannot await it, so the critic provably cannot invoke the frozen model.
  const result = runLocalCriticTier(critic, obs());
  assert.equal(typeof (result as { then?: unknown }).then, "undefined");
  assert.notEqual(Object.prototype.toString.call(result), "[object Promise]");

  // 2. The critic never RECEIVES a frozen model: it is called with exactly one
  //    argument (the observation), which carries no model identity. A regular
  //    function is used so `arguments` reflects the real call arity.
  let receivedArgs = -1;
  let sawModelShape = false;
  const spyCritic = function (this: unknown, observation: SyntheticObservation) {
    // eslint-disable-next-line prefer-rest-params
    receivedArgs = arguments.length;
    const o = observation as unknown as Record<string, unknown>;
    if ("sha256" in o || "id" in o) sawModelShape = true;
    return critic(observation);
  } as LocalCritic;
  runLocalCriticTier(spyCritic, obs());
  assert.equal(receivedArgs, 1, "critic must be called with only the observation");
  assert.equal(sawModelShape, false, "the observation must not carry a FrozenModelRef shape");

  // 3. A critic that declares a second parameter (to smuggle a model in) is
  //    rejected by the model-free guard.
  const twoArg = ((_o: SyntheticObservation, _m: FrozenModelRef) => critic(_o)) as unknown as LocalCritic;
  assert.throws(() => assertModelFreeCritic(twoArg), /model-free|arity/);

  // 4. Cheap: 50k invocations complete well under a generous budget (pure
  //    arithmetic, no I/O) — a loose ceiling, not a micro-benchmark.
  const start = process.hrtime.bigint();
  for (let i = 0; i < 50_000; i += 1) critic(obs({ step: i }));
  const elapsedMs = Number(process.hrtime.bigint() - start) / 1e6;
  assert.ok(elapsedMs < 1000, `50k critic calls took ${elapsedMs.toFixed(1)}ms — expected < 1000ms`);
});

// ───────────────────────────── seconds tier: RECOVERY ────────────────────

test("seconds recovery maps flags to retry/fallback/degrade without base mutation", () => {
  assert.equal(recoveryStrategyForFlags(["stale_state"]), "retry");
  assert.equal(recoveryStrategyForFlags(["signal_quality"]), "fallback");
  assert.equal(recoveryStrategyForFlags(["low_confidence"]), "fallback");
  assert.equal(recoveryStrategyForFlags(["anomaly"]), "degrade");
  assert.equal(recoveryStrategyForFlags(["stale_state", "anomaly"]), "degrade");

  const critic = defaultLocalCritic(THRESHOLDS);
  const flagged = runLocalCriticTier(critic, obs({ stateAgeMs: 5000 }));
  const recovery = runRecoveryTier(flagged);
  assert.equal(recovery.strategy, "retry");
  assert.deepEqual(recovery.resolvedFlags, ["stale_state"]);
  assert.equal(recovery.recovered, true);
  assert.equal(recovery.tier, "seconds-recovery");
  assert.ok(Object.isFrozen(recovery));

  // The seconds tier fires ONLY on a critic-flagged failure.
  const clean = runLocalCriticTier(critic, obs());
  assert.throws(() => runRecoveryTier(clean), /only invoked on a critic-flagged/);
});

// ──────────────────── minutes tier: SKILL EVOLUTION + held-out ────────────

test("minutes skill evolution: a validated skill promotes through the flywheel gate on held-out seeds", async () => {
  const trainSeen: number[][] = [];
  const evalSeen: number[][] = [];

  const result = await runSkillEvolutionTier({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: HELD_OUT_BASELINE },
    split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
    propose: (parent, _planner, trainSeeds) => {
      trainSeen.push([...trainSeeds]);
      assert.equal(parent, parentSkills);
      return skillMutation();
    },
    evaluateHeldOut: (_genome, _optimizer, heldOutSeeds) => {
      evalSeen.push([...heldOutSeeds]);
      // Skill improvement, evaluated ON HELD-OUT seeds it never trained on.
      return HELD_OUT_BASELINE.map((s) => s + 0.05);
    },
    commit: COMMIT,
    date: "2026-08-21",
  });

  assert.equal(result.tier, "minutes-skill-evolution");
  assert.equal(result.generation.verdict.verdict, "ACCEPT");
  assert.equal(result.recommendation, "recommend");
  assert.equal(result.heldOut.validatedOnHeldOut, true);

  // The proposer saw ONLY train seeds; the evaluator saw ONLY held-out seeds.
  assert.deepEqual(trainSeen, [TRAIN_SEEDS]);
  assert.deepEqual(evalSeen, [HELD_OUT_SEEDS]);
  // And the two splits are disjoint — the candidate validated on unseen seeds.
  const trainSet = new Set(TRAIN_SEEDS);
  assert.ok(HELD_OUT_SEEDS.every((s) => !trainSet.has(s)));
  assert.ok(Object.isFrozen(result));
});

test("held-out split must be non-empty and disjoint (leaked seed is rejected)", () => {
  assert.throws(() => assertDisjointSplit({ trainSeeds: [], heldOutSeeds: [1] }), /at least one train seed/);
  assert.throws(() => assertDisjointSplit({ trainSeeds: [1], heldOutSeeds: [] }), /at least one held-out seed/);
  assert.throws(
    () => assertDisjointSplit({ trainSeeds: [1, 2, 3], heldOutSeeds: [3, 4] }),
    /leaked into training/,
  );
  assert.doesNotThrow(() => assertDisjointSplit({ trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS }));
});

test("minutes tier rejects a baseline whose length does not match the held-out seeds", async () => {
  await assert.rejects(
    runSkillEvolutionTier({
      frozenModel: FROZEN_MODEL,
      parent: { genome: parentSkills, samples: [0.9, 0.9] }, // 2 != 5 held-out
      split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
      propose: () => skillMutation(),
      evaluateHeldOut: () => HELD_OUT_BASELINE,
      commit: COMMIT,
    }),
    /baseline samples .* must be measured over the held-out seeds/,
  );
});

test("a regressed skill is blocked through the existing veto path (no base mutation)", async () => {
  const result = await runSkillEvolutionTier({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: HELD_OUT_BASELINE },
    split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
    propose: () => skillMutation({ describe: "bad mutation" }),
    evaluateHeldOut: () => HELD_OUT_BASELINE.map((s) => s - 0.1),
    commit: COMMIT,
  });
  assert.equal(result.generation.verdict.verdict, "REJECT");
  assert.deepEqual(result.generation.vetoReasons, ["dream_machine_reject"]);
  assert.equal(result.recommendation, "blocked");
  assert.equal(result.heldOut.validatedOnHeldOut, false);
});

// ──────────────── constitutional boundary + frozen-base invariant ─────────

test("capability-expanding skill routes to the ADR-315 constitutional stub and is BLOCKED", async () => {
  let evaluated = false;
  const result = await runSkillEvolutionTier({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: HELD_OUT_BASELINE },
    split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
    propose: () => skillMutation({
      describe: "skill that would grant a new tool",
      capabilityDelta: { addsTools: ["shell_exec"], addsActionClasses: [], addsCommunicationPeers: [] },
    }),
    evaluateHeldOut: () => {
      evaluated = true;
      return HELD_OUT_BASELINE.map((s) => s + 0.5);
    },
    commit: COMMIT,
  });
  // The evaluator never runs for an unauthorized capability expansion.
  assert.equal(evaluated, false);
  assert.ok(result.generation.capabilityGate);
  assert.equal(result.generation.capabilityGate.allowed, false);
  assert.ok(result.generation.capabilityGate.reasons.some((r) => r.includes("constitutional_gate_stub_unwired")));
  assert.ok(result.generation.vetoReasons.includes("capability_expansion_unauthorized"));
  assert.equal(result.recommendation, "blocked");
});

test("no tier can mutate the frozen base: weights are not a mutation surface", async () => {
  const weightsGenome = { surface: "weights", version: 1 } as unknown as EvolvableGenome;
  await assert.rejects(
    runSkillEvolutionTier({
      frozenModel: FROZEN_MODEL,
      parent: { genome: weightsGenome, samples: HELD_OUT_BASELINE },
      split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
      propose: () => skillMutation(),
      evaluateHeldOut: () => HELD_OUT_BASELINE,
      commit: COMMIT,
    }),
    /inadmissible mutation surface "weights"/,
  );
  // A malformed frozen-model hash never starts a generation at any tier.
  await assert.rejects(
    runSkillEvolutionTier({
      frozenModel: { id: "m", sha256: "not-a-hash" },
      parent: { genome: parentSkills, samples: HELD_OUT_BASELINE },
      split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
      propose: () => skillMutation(),
      evaluateHeldOut: () => HELD_OUT_BASELINE,
      commit: COMMIT,
    }),
    /sha256 must be 64 lowercase hex/,
  );
});

test("constitutional boundary: no promotion capability is exported or reachable", async () => {
  // No export name looks like a promotion capability (WP9's identical contract).
  const forbidden = /promote|merge|approve|apply|commit|push/i;
  for (const [name] of Object.entries(tse)) {
    assert.ok(!forbidden.test(name), `export "${name}" looks like a promotion capability`);
  }

  // Every result object is frozen data with no callable members.
  const result = await runTimescaleGeneration(baselineGenerationOptions());
  assert.ok(Object.isFrozen(result));
  for (const [key, value] of Object.entries(result)) {
    assert.notEqual(typeof value, "function", `result.${key} must be data, not a capability`);
  }
  // @ts-expect-error — TimescaleGenerationResult has no promote() by construction.
  assert.equal(result.promote, undefined);
});

// ─────────────────────────── three-tier orchestration ────────────────────

function syntheticRollouts(): SyntheticRollout[] {
  return [
    {
      seed: 2,
      observations: [
        obs({ seed: 2, step: 0 }), // clean → proceed
        obs({ seed: 2, step: 1, stateAgeMs: 5000 }), // stale → gated → retry
        obs({ seed: 2, step: 2, value: 99 }), // anomaly → gated → degrade
      ],
    },
    {
      seed: 4,
      observations: [
        obs({ seed: 4, step: 0, confidence: 0.1 }), // low confidence → gated → fallback
        obs({ seed: 4, step: 1 }), // clean → proceed
      ],
    },
  ];
}

function baselineGenerationOptions(): tse.TimescaleGenerationOptions {
  return {
    frozenModel: FROZEN_MODEL,
    critic: defaultLocalCritic(THRESHOLDS),
    rollouts: syntheticRollouts(),
    parent: { genome: parentSkills, samples: HELD_OUT_BASELINE },
    split: { trainSeeds: TRAIN_SEEDS, heldOutSeeds: HELD_OUT_SEEDS },
    propose: () => skillMutation(),
    evaluateHeldOut: () => HELD_OUT_BASELINE.map((s) => s + 0.05),
    commit: COMMIT,
    date: "2026-08-21",
  };
}

test("the three tiers run IN ORDER over a synthetic env; a flagged failure triggers recovery", async () => {
  const result = await runTimescaleGeneration(baselineGenerationOptions());

  // Tier order is Zetta's timescale ordering.
  assert.deepEqual([...result.tiersRunInOrder], ["ms-local-critic", "seconds-recovery", "minutes-skill-evolution"]);
  assert.deepEqual([...TIER_ORDER], [...result.tiersRunInOrder]);

  // ms tier gated 3 of 5 actions (stale, anomaly, low-confidence).
  assert.equal(result.ms.gated.length, 5);
  assert.equal(result.ms.flaggedCount, 3);

  // seconds tier ran a recovery for EACH flagged failure — and only those.
  assert.equal(result.seconds.recoveries.length, 3);
  assert.deepEqual(
    result.seconds.recoveries.map((r) => r.strategy).sort(),
    ["degrade", "fallback", "retry"],
  );
  assert.ok(result.seconds.recoveries.every((r) => r.recovered));

  // minutes tier produced a held-out-validated recommendation.
  assert.equal(result.minutes.tier, "minutes-skill-evolution");
  assert.equal(result.minutes.recommendation, "recommend");
  assert.equal(result.minutes.heldOut.validatedOnHeldOut, true);
  assert.equal(result.recommendation, "recommend");
});

test("a clean synthetic env triggers NO recovery but still runs the minutes tier", async () => {
  const cleanRollouts: SyntheticRollout[] = [
    { seed: 2, observations: [obs({ seed: 2, step: 0 }), obs({ seed: 2, step: 1 })] },
  ];
  const result = await runTimescaleGeneration({ ...baselineGenerationOptions(), rollouts: cleanRollouts });
  assert.equal(result.ms.flaggedCount, 0);
  assert.equal(result.seconds.recoveries.length, 0);
  assert.equal(result.minutes.recommendation, "recommend");
});
