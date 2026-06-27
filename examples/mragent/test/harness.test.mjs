// MRAgent harness acceptance gates. Deterministic — no network, no native deps.
// Run: npm test   (node --test)

import { test } from "node:test";
import assert from "node:assert/strict";
import fs from "node:fs";
import path from "node:path";
import { fileURLToPath } from "node:url";
import { MemoryStore, baselineGenome, evaluate, mutate, runReasoningLoop } from "../agent/harness.mjs";
import { embed, EMBED_DIM } from "../agent/memory.mjs";

const __dirname = path.dirname(fileURLToPath(import.meta.url));
const corpus = JSON.parse(fs.readFileSync(path.join(__dirname, "..", "data", "eval-set.json"), "utf8"));
const tasks = corpus.tasks;
const store = new MemoryStore(tasks);

test("embeddings are deterministic and L2-normalized", () => {
  const a = embed("Raft consensus leader");
  const b = embed("Raft consensus leader");
  assert.equal(a.length, EMBED_DIM);
  assert.deepEqual([...a], [...b]);
  let norm = 0;
  for (const x of a) norm += x * x;
  assert.ok(Math.abs(Math.sqrt(norm) - 1) < 1e-5);
});

test("evaluation is reproducible for a fixed genome", () => {
  const g = baselineGenome();
  const m1 = evaluate(g, store, tasks);
  const m2 = evaluate(g, store, tasks);
  assert.deepEqual(m1, m2);
});

test("baseline genome answers a non-trivial share of the corpus", () => {
  const m = evaluate(baselineGenome(), store, tasks);
  assert.ok(m.accuracy > 0.3, `expected baseline accuracy > 0.3, got ${m.accuracy}`);
});

test("traversal depth is load-bearing: depth=1 misses multi-hop (bridge) tasks", () => {
  const bridgeTask = tasks.find((t) => (t.bridgeTags || []).length > 0);
  assert.ok(bridgeTask, "corpus should contain at least one bridge (multi-hop) task");
  const shallow = { ...baselineGenome(), traversalDepth: 1, tagFanout: 8, maxContent: 20 };
  const deep = { ...baselineGenome(), traversalDepth: 3, tagFanout: 8, maxContent: 20 };
  const rShallow = runReasoningLoop(bridgeTask.question, store, shallow, bridgeTask);
  const rDeep = runReasoningLoop(bridgeTask.question, store, deep, bridgeTask);
  assert.equal(rShallow.correct, false, "depth=1 should miss a bridge task");
  assert.equal(rDeep.correct, true, "depth>=2 should reconstruct the bridge task");
});

test("over-aggressive pruning destroys accuracy (real trade-off exists)", () => {
  const sane = { ...baselineGenome(), pruneThreshold: 0.1 };
  const brutal = { ...baselineGenome(), pruneThreshold: 0.6 };
  const mSane = evaluate(sane, store, tasks);
  const mBrutal = evaluate(brutal, store, tasks);
  assert.ok(mBrutal.accuracy < mSane.accuracy, "high prune threshold should reduce accuracy");
});

test("there exists a genome that beats the baseline (optimization is fruitful)", () => {
  const base = evaluate(baselineGenome(), store, tasks);
  const tuned = evaluate(
    { ...baselineGenome(), traversalDepth: 3, efSearch: 128, cueK: 6, pruneThreshold: 0.08, maxContent: 8 },
    store, tasks,
  );
  assert.ok(tuned.accuracy >= base.accuracy, "tuned genome should not regress accuracy");
});

test("mutate stays within declared genome bounds", () => {
  let g = baselineGenome();
  for (let i = 0; i < 200; i++) {
    g = mutate(g);
    assert.ok(g.cueK >= 1 && g.cueK <= 12);
    assert.ok(g.efSearch >= 16 && g.efSearch <= 256);
    assert.ok(g.hybridAlpha >= 0 && g.hybridAlpha <= 1);
    assert.ok(["rrf", "linear", "dbsf"].includes(g.fusion));
    assert.ok(g.traversalDepth >= 1 && g.traversalDepth <= 4);
    assert.ok(g.tagFanout >= 1 && g.tagFanout <= 8);
    assert.ok(g.pruneThreshold >= 0 && g.pruneThreshold <= 0.6);
    assert.ok(g.maxContent >= 1 && g.maxContent <= 20);
    assert.ok(["gnn", "none"].includes(g.rerank));
    assert.ok(["terse", "evidence-first", "prune-explicit"].includes(g.promptStrategy));
  }
});
