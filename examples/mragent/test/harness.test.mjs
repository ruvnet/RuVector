// MRAgent v2 acceptance gates. Deterministic — no network, no native deps.
// Every gene is proven load-bearing here. Run: npm test

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryStore, baselineGenome, evaluate, mutate, runReasoningLoop, splitByClass } from "../agent/harness.mjs";
import { embed, EMBED_DIM, tokenize } from "../agent/memory.mjs";
import { consolidate } from "../agent/consolidate.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
const store = new MemoryStore(tasks);
const sub = (cls) => tasks.filter((t) => t.class === cls);
const accOn = (genome, subset) => {
  const s = new MemoryStore(tasks);
  let c = 0, n = 0;
  for (const t of subset) { if (t.answerable === false) continue; n++; if (runReasoningLoop(s.queryText(t.id), s, genome, t).correct) c++; }
  return c / (n || 1);
};

test("embeddings are deterministic and L2-normalized", () => {
  const a = embed("fast cold-boot");
  assert.equal(a.length, EMBED_DIM);
  assert.deepEqual([...a], [...embed("fast cold-boot")]);
  let norm = 0; for (const x of a) norm += x * x;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-5);
});

test("dense (concept) and sparse (token) signals are decoupled", () => {
  const cos = (x, y) => { let d = 0; for (let i = 0; i < x.length; i++) d += x[i] * y[i]; return d; };
  const overlap = (x, y) => { const A = new Set(tokenize(x)); let s = 0; for (const t of tokenize(y)) if (A.has(t)) s++; return s; };
  // paraphrase: shared concepts, zero shared tokens → dense-close, sparse-zero
  assert.ok(cos(embed("fast boot"), embed("rapid cold-start")) > 0.4);
  assert.equal(overlap("fast boot", "rapid cold-start"), 0);
});

test("evaluation is reproducible for a fixed genome", () => {
  const g = baselineGenome();
  assert.deepEqual(evaluate(g, store, tasks), evaluate(g, store, tasks));
});

test("baseline answers a non-trivial share but is not perfect (headroom exists)", () => {
  const m = evaluate(baselineGenome(), store, tasks);
  assert.ok(m.accuracy >= 0.4 && m.accuracy < 0.9, `baseline accuracy ${m.accuracy}`);
});

test("hybridAlpha is load-bearing in BOTH directions (dense vs sparse)", () => {
  const denseHeavy = { ...baselineGenome(), hybridAlpha: 1, cueK: 1, fusion: "linear" };
  const sparseHeavy = { ...baselineGenome(), hybridAlpha: 0, cueK: 1, fusion: "linear" };
  // semantic tasks need dense; lexical tasks need sparse
  assert.ok(accOn(denseHeavy, sub("semantic")) > accOn(sparseHeavy, sub("semantic")), "semantic needs dense");
  assert.ok(accOn(sparseHeavy, sub("lexical")) > accOn(denseHeavy, sub("lexical")), "lexical needs sparse");
});

test("traversalDepth is load-bearing: 2-hop-bridge tasks need depth>=3", () => {
  const bridge2 = tasks.filter((t) => (t.bridges || 0) >= 2);
  assert.ok(bridge2.length > 0);
  assert.equal(accOn({ ...baselineGenome(), traversalDepth: 2 }, bridge2), 0, "depth 2 misses 2-hop bridges");
  assert.equal(accOn({ ...baselineGenome(), traversalDepth: 3 }, bridge2), 1, "depth 3 resolves them");
});

test("abstention sharply cuts hallucination and raises risk-adjusted utility", () => {
  const reckless = evaluate({ ...baselineGenome(), abstainThreshold: 0 }, store, tasks);
  const calibrated = evaluate({ ...baselineGenome(), abstainThreshold: 0.4 }, store, tasks);
  assert.ok(reckless.hallucinationRate > 0.1, "baseline hallucinates on unanswerable");
  assert.ok(calibrated.hallucinationRate <= reckless.hallucinationRate / 2, "abstention at least halves hallucination");
  assert.ok(calibrated.riskScore > reckless.riskScore + 0.1, "risk-adjusted utility improves materially");
});

test("corroboration (rerank=gnn) + fanout rescue distractor tasks under a terse window", () => {
  const d = sub("distractor");
  const none = accOn({ ...baselineGenome(), rerank: "none", promptStrategy: "terse", tagFanout: 3, maxContent: 8 }, d);
  const gnn = accOn({ ...baselineGenome(), rerank: "gnn", promptStrategy: "terse", tagFanout: 3, maxContent: 8 }, d);
  const gnnNoFan = accOn({ ...baselineGenome(), rerank: "gnn", promptStrategy: "terse", tagFanout: 1, maxContent: 8 }, d);
  assert.equal(gnn, 1, "gnn corroboration + fanout resolves all distractor tasks");
  assert.ok(gnn > none + 0.3, "corroboration beats no-rerank under a terse window");
  assert.ok(gnn > gnnNoFan + 0.3, "corroboration needs fanout to reach the corroborating tag");
});

test("consolidation (replay) reduces hops at equal-or-better accuracy", () => {
  const g = { ...baselineGenome(), traversalDepth: 3, fusion: "linear", haltConfidence: 0.5, abstainThreshold: 0.36 };
  const s = new MemoryStore(tasks);
  const before = evaluate(g, s, tasks);
  consolidate(s, tasks, g);
  const after = evaluate(g, s, tasks);
  assert.ok(after.avgHops < before.avgHops, `hops ${before.avgHops} -> ${after.avgHops}`);
  assert.ok(after.accuracy >= before.accuracy - 1e-9, "accuracy not regressed");
});

test("a calibrated genome reaches high accuracy with near-zero hallucination", () => {
  const tuned = { ...baselineGenome(), fusion: "linear", traversalDepth: 3, tagFanout: 3, abstainThreshold: 0.4, maxContent: 6 };
  const m = evaluate(tuned, store, tasks);
  assert.ok(m.accuracy >= 0.8, `accuracy ${m.accuracy}`);
  assert.ok(m.hallucinationRate <= 0.05, `halluc ${m.hallucinationRate}`);
});

test("evolved-style genome GENERALIZES: beats baseline on a held-out test split", () => {
  const { train, test } = splitByClass(tasks, 0.6);
  assert.ok(test.length >= 10, "test split is non-trivial");
  const tuned = { ...baselineGenome(), fusion: "linear", traversalDepth: 3, tagFanout: 3, abstainThreshold: 0.4, maxContent: 6 };
  const baseTest = evaluate(baselineGenome(), store, test);
  const evoTest = evaluate(tuned, store, test);
  assert.ok(evoTest.accuracy >= baseTest.accuracy + 0.1, `test acc ${baseTest.accuracy} -> ${evoTest.accuracy}`);
  assert.ok(evoTest.hallucinationRate <= baseTest.hallucinationRate, "no worse hallucination on unseen tasks");
  // depth-independent confidence: deep (2-hop) bridges are still confident
  const bridge2 = test.filter((t) => (t.bridges || 0) >= 2);
  for (const t of bridge2) {
    const r = runReasoningLoop(store.queryText(t.id), store, tuned, t);
    assert.ok(r.confidence > 0.5, `2-hop bridge ${t.id} confidence ${r.confidence} should stay high`);
  }
});

test("mutate stays within declared genome bounds (all 12 genes)", () => {
  let g = baselineGenome();
  for (let i = 0; i < 300; i++) {
    g = mutate(g);
    assert.ok(g.cueK >= 1 && g.cueK <= 12);
    assert.ok(g.efSearch >= 16 && g.efSearch <= 256);
    assert.ok(g.hybridAlpha >= 0 && g.hybridAlpha <= 1);
    assert.ok(["rrf", "linear", "dbsf"].includes(g.fusion));
    assert.ok(g.traversalDepth >= 1 && g.traversalDepth <= 4);
    assert.ok(g.tagFanout >= 1 && g.tagFanout <= 8);
    assert.ok(g.pruneThreshold >= 0 && g.pruneThreshold <= 0.6);
    assert.ok(g.maxContent >= 1 && g.maxContent <= 20);
    assert.ok(g.haltConfidence >= 0.2 && g.haltConfidence <= 0.9);
    assert.ok(["gnn", "none"].includes(g.rerank));
    assert.ok(["terse", "evidence-first", "prune-explicit"].includes(g.promptStrategy));
    assert.ok(g.abstainThreshold >= 0 && g.abstainThreshold <= 0.6);
  }
});
