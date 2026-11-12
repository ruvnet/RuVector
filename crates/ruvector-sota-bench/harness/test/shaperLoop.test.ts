import assert from "node:assert/strict";
import test from "node:test";
import * as loop from "../src/shaperLoop.js";
import * as genome from "../src/genome.js";
import { runShaperGeneration, gepaHarnessProposer } from "../src/shaperLoop.js";
import {
  constitutionalGateStub,
  expandsCapabilities,
  type CandidateMutation,
  type EvolvableGenome,
  type FrozenModelRef,
  type SkillGenome,
} from "../src/genome.js";

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

const FROZEN_MODEL: FrozenModelRef = {
  id: "ruvltra-frozen-q4",
  sha256: "a".repeat(64),
};

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

const BASELINE = [0.90, 0.91, 0.89, 0.90, 0.92];

test("one full generation: mutate → evaluate → verdict → recommendation", async () => {
  const seenModels: FrozenModelRef[] = [];
  const result = await runShaperGeneration({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: BASELINE },
    propose: (parent, planner) => {
      seenModels.push(planner);
      assert.equal(parent, parentSkills);
      return skillMutation();
    },
    evaluate: (candidate, optimizer) => {
      seenModels.push(optimizer);
      assert.equal(candidate.surface, "skills");
      // Skill improvement WITHOUT weight changes: better samples, same model.
      return BASELINE.map((s) => s + 0.05);
    },
    commit: COMMIT,
    date: "2026-08-20",
  });

  assert.equal(result.verdict.verdict, "ACCEPT");
  assert.equal(result.verdict.evaluated, "yes");
  assert.deepEqual(result.vetoReasons, []);
  assert.equal(result.recommendation, "recommend");
  // Loop-start model hash is recorded and witness-bound via the report stamp.
  assert.equal(result.modelSha256, FROZEN_MODEL.sha256);
  assert.match(result.verdict.witness, /^[0-9a-f]{64}$/);
  assert.ok(result.verdict.ledger.includes("add regrasp-on-slip recovery skill"));
  // SHAPER structural claim (configuration test): the same frozen instance
  // served as planner and optimizer.
  assert.equal(result.plannerModelId, result.optimizerModelId);
  assert.equal(seenModels.length, 2);
  assert.equal(seenModels[0], seenModels[1]);
  assert.equal(seenModels[0], FROZEN_MODEL);
});

test("a regressed candidate is blocked through the existing veto path", async () => {
  const result = await runShaperGeneration({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: BASELINE },
    propose: () => skillMutation({ describe: "bad mutation" }),
    evaluate: () => BASELINE.map((s) => s - 0.1),
    commit: COMMIT,
  });
  assert.equal(result.verdict.verdict, "REJECT");
  assert.deepEqual(result.vetoReasons, ["dream_machine_reject"]);
  assert.equal(result.recommendation, "blocked");
});

test("capability-expanding candidate is routed to the constitutional stub and BLOCKED by default", async () => {
  let evaluated = false;
  const result = await runShaperGeneration({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: BASELINE },
    propose: () => skillMutation({
      describe: "skill that would grant a new tool",
      capabilityDelta: {
        addsTools: ["shell_exec"],
        addsActionClasses: [],
        addsCommunicationPeers: [],
      },
    }),
    evaluate: () => {
      evaluated = true;
      return BASELINE.map((s) => s + 0.5);
    },
    commit: COMMIT,
  });
  // ADR-315: independently blocking, distinct from behavioral promotion —
  // the evaluator never even runs for an unauthorized capability expansion.
  assert.equal(evaluated, false);
  assert.ok(result.capabilityGate);
  assert.equal(result.capabilityGate.allowed, false);
  assert.ok(result.capabilityGate.reasons.some((r) => r.includes("constitutional_gate_stub_unwired")));
  assert.equal(result.verdict.evaluated, "blocked");
  assert.equal(result.verdict.verdict, "INCONCLUSIVE");
  assert.ok(result.vetoReasons.includes("capability_expansion_unauthorized"));
  assert.equal(result.recommendation, "blocked");
});

test("the capability boundary itself: only added tools/actions/peers cross it", () => {
  assert.equal(expandsCapabilities(undefined), false);
  assert.equal(expandsCapabilities({ addsTools: [], addsActionClasses: [], addsCommunicationPeers: [] }), false);
  assert.equal(expandsCapabilities({ addsTools: [], addsActionClasses: ["move-base"], addsCommunicationPeers: [] }), true);
  // The stub blocks even a direct call — nothing is authorized until WP11.
  const decision = constitutionalGateStub(skillMutation({
    capabilityDelta: { addsTools: [], addsActionClasses: [], addsCommunicationPeers: ["peer-b"] },
  }));
  assert.equal(decision.allowed, false);
  assert.ok(Object.isFrozen(decision));
});

test("weights are not a mutation surface: an inadmissible surface throws", async () => {
  const weightsGenome = { surface: "weights", version: 1 } as unknown as EvolvableGenome;
  await assert.rejects(
    runShaperGeneration({
      frozenModel: FROZEN_MODEL,
      parent: { genome: weightsGenome, samples: BASELINE },
      propose: () => skillMutation(),
      evaluate: () => BASELINE,
      commit: COMMIT,
    }),
    /inadmissible mutation surface "weights"/,
  );
  // And a malformed frozen-model hash never starts a loop.
  await assert.rejects(
    runShaperGeneration({
      frozenModel: { id: "m", sha256: "not-a-hash" },
      parent: { genome: parentSkills, samples: BASELINE },
      propose: () => skillMutation(),
      evaluate: () => BASELINE,
      commit: COMMIT,
    }),
    /sha256 must be 64 lowercase hex/,
  );
});

test("constitutional boundary: no promotion capability is exported or reachable", async () => {
  // 1. Neither module exports a promote/merge/approve/apply function —
  //    identical contract to WP2's dreamMachine.ts boundary test.
  const forbidden = /promote|merge|approve|apply|commit|push/i;
  for (const [name] of [...Object.entries(loop), ...Object.entries(genome)]) {
    assert.ok(
      !forbidden.test(name),
      `export "${name}" looks like a promotion capability — the loop must only emit recommendations`,
    );
  }

  // 2. The generation result is frozen data with no callable members.
  const result = await runShaperGeneration({
    frozenModel: FROZEN_MODEL,
    parent: { genome: parentSkills, samples: BASELINE },
    propose: () => skillMutation(),
    evaluate: () => BASELINE.map((s) => s + 0.05),
    commit: COMMIT,
  });
  assert.ok(Object.isFrozen(result));
  for (const [key, value] of Object.entries(result)) {
    assert.notEqual(typeof value, "function", `result.${key} must be data, not a capability`);
  }
  // @ts-expect-error — ShaperGenerationResult has no promote() by construction.
  assert.equal(result.promote, undefined);
});

test("harness-surface proposer reuses Darwin's existing GEPA search", async () => {
  const propose = gepaHarnessProposer({
    repoRoot: process.cwd(),
    maxCandidates: 3,
    items: [11, 29].map((seed) => ({
      seed,
      dataset: "smoke-128",
      dataset_sha256: "a".repeat(64),
      kind: "smoke" as const,
    })),
    benchmark: async (policy, item) => ({
      scores: [{
        index: "core-hnsw",
        dataset: item.dataset,
        recall: { recall_at_10: 0.96 },
        qps: 1000 + Number(policy.ef_search),
        memory_mb: 40,
        latency: { p99_us: 500 },
      }],
    }),
  });
  const parentHarness: EvolvableGenome = {
    surface: "harness",
    version: 7,
    parameters: { ef_search: "32", m: "16" },
  };
  const mutation = await propose(parentHarness, FROZEN_MODEL);
  if (mutation.candidate.surface !== "harness") assert.fail("expected a harness-surface candidate");
  assert.equal(mutation.candidate.version, 8);
  assert.ok(mutation.candidate.parameters.ef_search);
  // GEPA mutates ANN/runner parameters only — never a capability delta.
  assert.equal(mutation.capabilityDelta, undefined);
  // A skills-surface parent is a contract violation for this proposer.
  await assert.rejects(
    Promise.resolve(propose(parentSkills, FROZEN_MODEL)),
    /mutates the harness surface/,
  );
});
