import assert from "node:assert/strict";
import test from "node:test";
import * as skillForge from "../src/skillForge.js";
import {
  SKILLFORGE_CITATION,
  assertCoveredTarget,
  buildSkillProposal,
  corroboratesRepair,
  distillSkill,
  entityKey,
  isEntityStale,
  runSkillForge,
  synthesizeIssue,
  toCandidateMutation,
  toRuvectorRecord,
  type CoveredTarget,
  type IssueSolver,
  type RepairTrajectory,
  type SyntheticIssue,
} from "../src/skillForge.js";
import { expandsCapabilities, type SkillGenome } from "../src/genome.js";

const DATE = "2026-08-20";

// A test-covered function in this very harness — a realistic grounding target.
const COVERED_FN: CoveredTarget = {
  kind: "function",
  path: "crates/ruvector-sota-bench/harness/src/genome.ts",
  symbol: "assertEvolvableGenome",
  adr: "ADR-313",
  contentSha256: "a".repeat(64),
  coveringTests: ["test/shaperLoop.test.ts::weights are not a mutation surface"],
};

const PARENT_SKILLS: SkillGenome = {
  surface: "skills",
  version: 5,
  skills: [{ name: "grasp", body: "reach, close gripper" }],
};

/** A solver that always succeeds, restoring the broken tests. */
const successSolver = (issue: SyntheticIssue): RepairTrajectory => ({
  issueId: issue.id,
  steps: [
    { action: "locate", detail: `open ${issue.target.path}` },
    { action: "repair", detail: "restore the pinned behavior" },
  ],
  succeeded: true,
  resolvedTests: [...issue.breaksTests],
});

/** A solver that fails to repair. */
const failSolver = (issue: SyntheticIssue): RepairTrajectory => ({
  issueId: issue.id,
  steps: [{ action: "attempt", detail: "no repair found" }],
  succeeded: false,
  resolvedTests: [],
});

test("SELECT + SYNTHESIZE: a synthetic issue is generated from a covered target", () => {
  const issue = synthesizeIssue(COVERED_FN, DATE);
  // Derived from the covered target, deterministically.
  assert.equal(issue.target, COVERED_FN);
  assert.equal(issue.id, synthesizeIssue(COVERED_FN, DATE).id);
  assert.match(issue.id, /^synth-[0-9a-f]{16}$/);
  // Re-implementing test-covered functionality breaks exactly its covering tests.
  assert.deepEqual(issue.breaksTests, COVERED_FN.coveringTests);
  assert.ok(issue.title.includes("assertEvolvableGenome"));
  // No performance magnitude leaks into the synthesized description.
  assert.doesNotMatch(issue.description, /\d+(\.\d+)?\s*(%|percent|points?|pp|x\b)/i);
  // A target with no covering test cannot be selected.
  assert.throws(
    () => assertCoveredTarget({ ...COVERED_FN, coveringTests: [] }),
    /at least one covering test/,
  );
});

test("DISTILL: a successful trajectory yields an entity-grounded skill with correct entity references", () => {
  const issue = synthesizeIssue(COVERED_FN, DATE);
  const trajectory = successSolver(issue);
  const grounded = distillSkill(issue, trajectory, DATE);
  assert.ok(grounded, "a successful repair must distil a skill");

  // ENTITY GROUNDING PROOF — the skill records WHICH entity it applies to,
  // with that entity's exact content hash (integrity anchor), not a flat body.
  assert.equal(grounded.entities.length, 1);
  const ref = grounded.entities[0];
  assert.ok(ref);
  assert.equal(ref.kind, "function");
  assert.equal(ref.path, COVERED_FN.path);
  assert.equal(ref.symbol, "assertEvolvableGenome");
  assert.equal(ref.contentSha256, COVERED_FN.contentSha256);
  assert.equal(entityKey(ref), "fn:crates/ruvector-sota-bench/harness/src/genome.ts#assertEvolvableGenome");

  // The retrieval key is identity-only; the content hash is separate, so a
  // drifted target is detectable rather than silently re-applied.
  assert.equal(isEntityStale(ref, "a".repeat(64)), false);
  assert.equal(isEntityStale(ref, "b".repeat(64)), true);

  // Provenance carries issue → target → trajectory, and the fixed citation.
  assert.equal(grounded.provenance.syntheticIssueId, issue.id);
  assert.equal(grounded.provenance.coveredTarget.path, COVERED_FN.path);
  assert.equal(grounded.provenance.trajectory, trajectory);
  assert.equal(grounded.provenance.citation, SKILLFORGE_CITATION);

  // Shaped for the WP9 genome skill set (SkillDefinition).
  assert.equal(typeof grounded.skill.name, "string");
  assert.ok(grounded.skill.body.includes(SKILLFORGE_CITATION));

  // Shaped to attach into RuVector memory, grounded by identity key.
  const record = toRuvectorRecord(grounded);
  assert.deepEqual(record.entityKeys, [entityKey(ref)]);
  assert.equal(record.skillName, grounded.skill.name);
});

test("DISTILL: a FAILED trajectory yields NO skill", () => {
  const issue = synthesizeIssue(COVERED_FN, DATE);
  const grounded = distillSkill(issue, failSolver(issue), DATE);
  assert.equal(grounded, null);

  // And end to end: the failed target is recorded as skipped, never distilled.
  return runSkillForge({ targets: [COVERED_FN], parent: PARENT_SKILLS, solve: failSolver, date: DATE })
    .then((result) => {
      assert.equal(result.proposals.length, 0);
      assert.equal(result.skipped.length, 1);
      assert.equal(result.skipped[0]!.reason, "repair_failed");
    });
});

test("CORROBORATION: self-declared success alone earns no skill — resolvedTests must intersect breaksTests", async () => {
  const issue = synthesizeIssue(COVERED_FN, DATE);
  assert.ok(issue.breaksTests.length > 0);

  // succeeded:true but NO resolved test → hollow claim → NO skill.
  const empty: RepairTrajectory = { issueId: issue.id, steps: [], succeeded: true, resolvedTests: [] };
  assert.equal(corroboratesRepair(issue, empty), false);
  assert.equal(distillSkill(issue, empty, DATE), null);

  // succeeded:true but resolvedTests UNRELATED to what the issue broke → NO skill.
  const unrelated: RepairTrajectory = {
    issueId: issue.id,
    steps: [],
    succeeded: true,
    resolvedTests: ["test/somewhere-else.test.ts::unrelated"],
  };
  assert.equal(corroboratesRepair(issue, unrelated), false);
  assert.equal(distillSkill(issue, unrelated, DATE), null);

  // succeeded:true AND resolvedTests cover the broken tests → skill distilled.
  const covering: RepairTrajectory = {
    issueId: issue.id,
    steps: [{ action: "repair", detail: "restore" }],
    succeeded: true,
    resolvedTests: [...issue.breaksTests],
  };
  assert.equal(corroboratesRepair(issue, covering), true);
  assert.ok(distillSkill(issue, covering, DATE));

  // End to end: an uncorroborated "success" is recorded as skipped, not proposed.
  const hollowSolver = (i: SyntheticIssue): RepairTrajectory =>
    ({ issueId: i.id, steps: [], succeeded: true, resolvedTests: [] });
  const result = await runSkillForge({ targets: [COVERED_FN], parent: PARENT_SKILLS, solve: hollowSolver, date: DATE });
  assert.equal(result.proposals.length, 0);
  assert.equal(result.skipped.length, 1);
  assert.equal(result.skipped[0]!.reason, "repair_uncorroborated");
});

test("entityKey is injective: separator-bearing components cannot collide", () => {
  const a = entityKey({ kind: "function", path: "auth", symbol: "ts#login", contentSha256: "a".repeat(64) });
  const b = entityKey({ kind: "function", path: "auth#ts", symbol: "login", contentSha256: "a".repeat(64) });
  assert.notEqual(a, b);
  // A `:`-bearing component likewise cannot forge the fn:/mod:/adr: prefix boundary.
  const c = entityKey({ kind: "module", path: "a:b", contentSha256: "a".repeat(64) });
  const d = entityKey({ kind: "module", path: "a", contentSha256: "a".repeat(64) });
  assert.notEqual(c, d);
  // Plain paths (no separators) are unchanged — retrieval keys stay readable.
  assert.equal(
    entityKey({ kind: "function", path: "src/genome.ts", symbol: "f", contentSha256: "a".repeat(64) }),
    "fn:src/genome.ts#f",
  );
});

test("CONSTITUTIONAL BOUNDARY: a distilled skill is a PROPOSAL, no promotion reachable", async () => {
  // 1. No export looks like a promotion capability (same contract as WP2/WP9).
  const forbidden = /promote|merge|approve|apply|commit|push/i;
  for (const [name] of Object.entries(skillForge)) {
    assert.ok(!forbidden.test(name), `export "${name}" looks like a promotion capability`);
  }

  const result = await runSkillForge({
    targets: [COVERED_FN],
    parent: PARENT_SKILLS,
    solve: successSolver,
    date: DATE,
  });

  // 2. The result and its proposals are frozen data with no callable members.
  assert.ok(Object.isFrozen(result));
  const [proposal] = result.proposals;
  assert.ok(proposal);
  assert.equal(proposal.kind, "skill-proposal");
  assert.ok(Object.isFrozen(proposal));
  for (const [key, value] of Object.entries(proposal)) {
    assert.notEqual(typeof value, "function", `proposal.${key} must be data, not a capability`);
  }

  // 3. It feeds WP9's genome as a CANDIDATE mutation (skills surface, +1 skill,
  //    version bumped) — an untrusted proposal, not a merge.
  const mutation = proposal.candidateMutation;
  assert.equal(mutation.candidate.surface, "skills");
  assert.equal(mutation.candidate.version, PARENT_SKILLS.version + 1);
  if (mutation.candidate.surface !== "skills") assert.fail("expected skills surface");
  assert.equal(mutation.candidate.skills.length, PARENT_SKILLS.skills.length + 1);
  // A non-capability-expanding proposal is not routed to the gate at all.
  assert.equal(proposal.blockedByGate, false);
  assert.equal(proposal.capabilityGate, undefined);
});

test("CAPABILITY EXPANSION: a skill that would grant a new tool routes to the ADR-315 stub and is BLOCKED", async () => {
  // A repair whose strategy would add a tool declares a capabilityDelta.
  const toolGrantingSolver = (issue: SyntheticIssue): RepairTrajectory => ({
    issueId: issue.id,
    steps: [{ action: "repair", detail: "needs a new tool to fix" }],
    succeeded: true,
    resolvedTests: [...issue.breaksTests],
    capabilityDelta: { addsTools: ["shell_exec"], addsActionClasses: [], addsCommunicationPeers: [] },
  });

  const issue = synthesizeIssue(COVERED_FN, DATE);
  const grounded = distillSkill(issue, toolGrantingSolver(issue), DATE);
  assert.ok(grounded);
  assert.ok(expandsCapabilities(grounded.capabilityDelta));

  const proposal = buildSkillProposal(PARENT_SKILLS, grounded);
  // Routed to the constitutional gate stub — blocked by default (no WP11 wiring).
  assert.equal(proposal.blockedByGate, true);
  assert.ok(proposal.capabilityGate);
  assert.equal(proposal.capabilityGate.allowed, false);
  assert.ok(proposal.capabilityGate.reasons.some((r) => r.includes("constitutional_gate_stub_unwired")));

  // The capabilityDelta rides through onto the WP9 candidate mutation, so
  // shaperLoop's own gate re-adjudicates it — this source bypasses no gate.
  assert.ok(expandsCapabilities(proposal.candidateMutation.capabilityDelta));

  // End to end, the capability-expanding proposal is still produced (a
  // candidate) but carries its blocked gate.
  const result = await runSkillForge({
    targets: [COVERED_FN],
    parent: PARENT_SKILLS,
    solve: toolGrantingSolver,
    date: DATE,
  });
  assert.equal(result.proposals.length, 1);
  assert.equal(result.proposals[0]!.blockedByGate, true);
});

test("EVIDENCE DISCIPLINE: the citation carries NO performance magnitude", () => {
  // Exact ADR-321 citation, and not a single number attached to it.
  assert.equal(SKILLFORGE_CITATION, "SkillForge (arXiv:2608.18933)");
  const grounded = distillSkill(
    synthesizeIssue(COVERED_FN, DATE),
    successSolver(synthesizeIssue(COVERED_FN, DATE)),
    DATE,
  );
  assert.ok(grounded);
  // No "beats baselines by N%" style magnitude anywhere in the distilled body.
  assert.doesNotMatch(grounded.skill.body, /\d+(\.\d+)?\s*(%|percent|points?|pp)\b/i);
  assert.doesNotMatch(grounded.skill.body, /\bbeats?\b|\boutperform/i);
});

test("toCandidateMutation appends to the parent genome without mutating it", () => {
  const grounded = distillSkill(
    synthesizeIssue(COVERED_FN, DATE),
    successSolver(synthesizeIssue(COVERED_FN, DATE)),
    DATE,
  );
  assert.ok(grounded);
  const before = PARENT_SKILLS.skills.length;
  const mutation = toCandidateMutation(PARENT_SKILLS, grounded);
  assert.equal(PARENT_SKILLS.skills.length, before, "parent genome must not be mutated");
  assert.equal(mutation.parent, PARENT_SKILLS);
  assert.ok(mutation.describe.includes("skillforge candidate"));
});
