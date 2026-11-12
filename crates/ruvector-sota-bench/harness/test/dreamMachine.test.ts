import assert from "node:assert/strict";
import test from "node:test";
import * as adapter from "../src/dreamMachine.js";
import {
  evaluateCandidate,
  verdictFromDecision,
  vetoesFromVerdict,
  dreamMachineVetoProvider,
  DREAM_MACHINE_VERSION,
} from "../src/dreamMachine.js";
import type { PairedDecision } from "../src/statistics.js";

const decision = (outcome: PairedDecision["outcome"]): PairedDecision => ({
  meanDelta: outcome === "pass" ? 12.5 : 0,
  lower95: outcome === "pass" ? 1 : -1,
  upper95: outcome === "fail" ? -0.5 : 2,
  outcome,
  samples: 2000,
});

const COMMIT = "0123456789abcdef0123456789abcdef01234567";

test("verdict mapping: statistical outcome → engine verdict vocabulary", () => {
  assert.equal(verdictFromDecision(decision("pass")), "ACCEPT");
  assert.equal(verdictFromDecision(decision("fail")), "REJECT");
  assert.equal(verdictFromDecision(decision("inconclusive")), "INCONCLUSIVE");
  // A blocked/never-run evaluator can never be rounded up to ACCEPT.
  assert.equal(verdictFromDecision(decision("pass"), "blocked"), "INCONCLUSIVE");
  assert.equal(verdictFromDecision(decision("pass"), "no"), "INCONCLUSIVE");
});

test("verdict → veto reasons: ACCEPT adds no credit, everything else blocks", async () => {
  const accept = await evaluateCandidate({
    finding: "smoke", report: "# r", commit: COMMIT, decision: decision("pass"),
  });
  const reject = await evaluateCandidate({
    finding: "smoke", report: "# r", commit: COMMIT, decision: decision("fail"),
  });
  const inconclusive = await evaluateCandidate({
    finding: "smoke", report: "# r", commit: COMMIT, decision: decision("inconclusive"),
  });
  assert.deepEqual(vetoesFromVerdict(accept), []);
  assert.deepEqual(vetoesFromVerdict(reject), ["dream_machine_reject"]);
  assert.deepEqual(vetoesFromVerdict(inconclusive), ["dream_machine_inconclusive"]);
});

test("smoke: the dream-machine engine loads and evaluates a trivial candidate", async () => {
  const verdict = await evaluateCandidate({
    finding: "trivial candidate",
    report: "# evaluation report\nrecall@10 improved.\n",
    commit: COMMIT,
    decision: decision("pass"),
    date: "2026-08-19",
  });
  assert.equal(verdict.verdict, "ACCEPT");
  assert.equal(verdict.evaluated, "yes");
  assert.equal(verdict.engineVersion, DREAM_MACHINE_VERSION);
  // Engine-produced witness stamp: sha256(sha256(report) + commit).
  assert.match(verdict.witness, /^[0-9a-f]{64}$/);
  // Engine-produced, engine-verified ledger row.
  assert.ok(verdict.ledger.includes("| Date |"));
  assert.ok(verdict.ledger.includes("trivial candidate"));
  assert.ok(verdict.ledger.includes("ACCEPT"));
  assert.ok(verdict.ledger.includes(verdict.witness.slice(0, 8)));
});

test("adapter plugs into the existing vetoes.ts provider contract", async () => {
  const provider = dreamMachineVetoProvider(() => ({
    finding: "provider smoke",
    report: "# r",
    commit: COMMIT,
    decision: decision("fail"),
  }));
  const reasons = await provider({
    policy: { ef_search: "100" },
    suite: { id: "fixture", items: [] },
    observations: [],
  });
  assert.deepEqual(reasons, ["dream_machine_reject"]);
});

test("constitutional boundary: no promotion capability is exported or reachable", async () => {
  // 1. The module exports no promote/merge/approve/apply function.
  const forbidden = /promote|merge|approve|apply|commit|push/i;
  for (const [name, value] of Object.entries(adapter)) {
    assert.ok(
      !forbidden.test(name),
      `export "${name}" looks like a promotion capability — the adapter must only emit verdicts/vetoes`,
    );
    void value;
  }

  // 2. The verdict object is frozen data with no callable members.
  const verdict = await evaluateCandidate({
    finding: "boundary", report: "# r", commit: COMMIT, decision: decision("pass"),
  });
  assert.ok(Object.isFrozen(verdict));
  for (const [key, value] of Object.entries(verdict)) {
    assert.notEqual(typeof value, "function", `verdict.${key} must be data, not a capability`);
  }
  // @ts-expect-error — DreamMachineVerdict has no promote() by construction.
  assert.equal(verdict.promote, undefined);
});
