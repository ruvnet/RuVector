// Tests for the GPU LLM write-layer's safety boundary: whatever a model returns,
// coerceGenome must produce a genome whose every gene is within declared bounds.
import { test } from "node:test";
import assert from "node:assert/strict";
import { coerceGenome } from "../agent/llmMutator.mjs";
import { baselineGenome } from "../agent/harness.mjs";

test("coerceGenome clamps out-of-range / wrong-type LLM output to safe genome", () => {
  const base = baselineGenome();
  const hostile = {
    cueK: 9999,            // way over max
    efSearch: -50,         // under min
    hybridAlpha: 7.5,      // over 1
    traversalDepth: 0,     // under min
    tagFanout: "lots",     // wrong type → ignored, keeps baseline
    pruneThreshold: 2,     // over max
    maxContent: 1000,
    haltConfidence: -1,    // under min (0.2)
    abstainThreshold: 5,
    fusion: "telepathy",   // invalid enum → baseline
    rerank: "none",        // valid enum → applied
    promptStrategy: "evil",// invalid enum → baseline
    injected: "ignore me", // unknown key → dropped
  };
  const g = coerceGenome(hostile, base);

  assert.ok(g.cueK >= 1 && g.cueK <= 12);
  assert.ok(g.efSearch >= 16 && g.efSearch <= 256);
  assert.ok(g.hybridAlpha >= 0 && g.hybridAlpha <= 1);
  assert.ok(g.traversalDepth >= 1 && g.traversalDepth <= 4);
  assert.equal(g.tagFanout, base.tagFanout); // wrong type ignored
  assert.ok(g.pruneThreshold >= 0 && g.pruneThreshold <= 0.6);
  assert.ok(g.maxContent >= 1 && g.maxContent <= 20);
  assert.ok(g.haltConfidence >= 0.2 && g.haltConfidence <= 0.9);
  assert.ok(g.abstainThreshold >= 0 && g.abstainThreshold <= 0.6);
  assert.equal(g.fusion, base.fusion);      // invalid enum → baseline
  assert.equal(g.rerank, "none");           // valid enum → applied
  assert.equal(g.promptStrategy, base.promptStrategy);
  assert.ok(!("injected" in g));            // unknown key dropped
  assert.ok(Number.isInteger(g.cueK) && Number.isInteger(g.efSearch));
});

test("coerceGenome on junk returns the baseline untouched", () => {
  const base = baselineGenome();
  assert.deepEqual(coerceGenome(null, base), base);
  assert.deepEqual(coerceGenome("not an object", base), base);
  assert.deepEqual(coerceGenome(42, base), base);
});
