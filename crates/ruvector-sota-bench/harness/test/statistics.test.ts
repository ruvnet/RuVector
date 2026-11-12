import assert from "node:assert/strict";
import test from "node:test";
import { pairedBootstrapDecision } from "../src/statistics.js";

test("paired bootstrap distinguishes pass, fail, and inconclusive", () => {
  assert.equal(
    pairedBootstrapDecision([0.7, 0.71, 0.69, 0.72, 0.7], [0.72, 0.73, 0.71, 0.74, 0.72]).outcome,
    "pass",
  );
  assert.equal(
    pairedBootstrapDecision([0.7, 0.71, 0.69, 0.72, 0.7], [0.69, 0.7, 0.68, 0.71, 0.69]).outcome,
    "fail",
  );
  assert.equal(pairedBootstrapDecision([0.7], [0.8]).outcome, "inconclusive");
});
