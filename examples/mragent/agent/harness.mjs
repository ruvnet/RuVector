// MRAgent EVOLVED HARNESS (v2 — beyond the paper) — the surface Darwin mutates.
//
// MRAgent's contribution: memory is a Cue-Tag-Content graph, reconstructed (not
// retrieved) by searching cues, traversing cue→tag→content, and pruning paths.
// This v2 adds three mechanisms the paper does not have, each a tunable gene:
//
//   • ADAPTIVE DEPTH  (haltConfidence) — stop traversing once evidence is decisive,
//                       so easy queries cost fewer hops (ACT-style adaptive compute).
//   • ABSTENTION      (abstainThreshold) — answer "I don't know" when reconstructed
//                       evidence is too weak, instead of confidently hallucinating.
//   • CORROBORATION   (rerank=gnn) — boost content reached by MULTIPLE paths, so a
//                       single high-similarity distractor cannot win.
//
// The memory substrate (agent/memory.mjs) stays frozen. Darwin edits only the
// DARWIN_MUTABLE_BLOCK regions.

import { MemoryStore } from "./memory.mjs";

// ─── DARWIN_MUTABLE_BLOCK: reconstruction genome ────────────────────────────
export function baselineGenome() {
  return {
    // Stage 1 — hybrid cue search (RuVector hybridSearch).
    cueK: 4,             // initial cue vectors fetched           [1..12]
    efSearch: 64,        // HNSW search depth / candidate pool     [16..256]
    hybridAlpha: 0.5,    // RRF weight: 0=sparse … 1=dense         [0..1]
    fusion: "rrf",       // rrf | linear | dbsf

    // Stage 2 — active reconstruction (Cypher LINKED_TO*1..N traversal).
    traversalDepth: 2,   // cue→tag→content hops                   [1..4]
    tagFanout: 3,        // tags expanded per frontier node        [1..8]
    pruneThreshold: 0.1, // drop paths below this evidence score   [0..0.6]
    maxContent: 8,       // content nodes handed to synthesis(LIMIT)[1..20]
    haltConfidence: 0.9, // adaptive-depth: stop when top≥this     [0.2..0.9]

    // Stage 3 — synthesis (LLM prompt strategy + safety).
    rerank: "gnn",       // gnn | none  (corroboration-aware rerank)
    promptStrategy: "evidence-first", // terse | evidence-first | prune-explicit
    abstainThreshold: 0.0, // answer "I don't know" if top score < this [0..0.6]
  };
}
// ─── END DARWIN_MUTABLE_BLOCK ───────────────────────────────────────────────

const STRATEGY_WINDOW = { terse: 2, "evidence-first": Infinity, "prune-explicit": 5 };

// Corroboration-aware rerank: content reached by multiple distinct paths is
// boosted, so a single high-similarity ranking-distractor cannot outrank a
// well-corroborated answer. (rerank="none" leaves raw similarity order.)
function gnnRerank(reconstructed) {
  return [...reconstructed]
    .map((c) => ({ ...c, score: c.score * (1 + 0.7 * ((c.paths ?? 1) - 1)) }))
    .sort((a, b) => b.score - a.score);
}

/**
 * Synthesis judge — deterministic stand-in for the LLM. Decides: abstain, answer
 * correctly, or answer wrongly, given the reconstructed context + confidence.
 */
function synthesize(reconstructed, task, genome, confidence) {
  // ABSTENTION: weak evidence → refuse rather than hallucinate.
  if (confidence < genome.abstainThreshold) return { abstained: true, correct: false, answer: "I don't know." };

  const window = STRATEGY_WINDOW[genome.promptStrategy] ?? Infinity;
  const visible = reconstructed.slice(0, window === Infinity ? reconstructed.length : window);
  const hitIdx = visible.findIndex((c) => c.taskId === task.id);

  if (hitIdx === -1) {
    // Nothing correct in the window. If the top is a confident distractor, the LLM
    // would emit it → a (wrong) answer; otherwise it produces an empty/no answer.
    const wrong = visible.length > 0;
    return { abstained: false, correct: false, answer: wrong ? "(distractor)" : "(no answer)" };
  }

  if (genome.promptStrategy === "prune-explicit") {
    const distractorsAbove = visible.slice(0, hitIdx).filter((c) => c.taskId !== task.id).length;
    if (distractorsAbove >= 2) return { abstained: false, correct: false, answer: "Pruned: ambiguous." };
  }
  return { abstained: false, correct: true, answer: task.expected_fact };
}

/** MRAgent reasoning loop for one task → deterministic result + telemetry. */
export function runReasoningLoop(queryText, store, genome, task) {
  const cueIds = store.hybridSearch(queryText, genome);
  let { content, stats } = store.reconstruct(queryText, cueIds, genome);
  if (genome.rerank === "gnn") content = gnnRerank(content);

  // Abstention confidence = chosen content's RAW relevance (depth-independent),
  // not its decayed ranking score — robust across traversal depths.
  const confidence = content.length ? (content[0].sim ?? content[0].score) : 0;
  const out = task ? synthesize(content, task, genome, confidence) : { abstained: false, correct: false };

  const latencyMs =
    0.02 * genome.efSearch +
    0.05 * stats.nodesVisited +
    0.30 * Math.min(content.length, genome.maxContent) +
    (genome.rerank === "gnn" ? 0.4 : 0);

  return { ...out, confidence, latencyMs, hops: stats.hops, halted: stats.halted, nodesVisited: stats.nodesVisited, contextSize: content.length };
}

/**
 * Evaluate a genome over the corpus. Reports raw accuracy AND a risk-adjusted
 * utility that rewards correct answers, tolerates honest abstention, and PUNISHES
 * confident hallucination — the calibration objective a 25-year-out memory system
 * is graded on, not raw accuracy alone.
 *
 *   answerable:   correct → +1 | abstain → 0 | wrong → −1
 *   unanswerable: abstain → +1 | any answer → −1
 */
export function evaluate(genome, store, tasks) {
  let correct = 0, answerable = 0, hallucinations = 0, util = 0;
  let latency = 0, hops = 0, ctx = 0;
  for (const task of tasks) {
    const isAnswerable = task.answerable !== false;
    const r = runReasoningLoop(store.queryText(task.id), store, genome, task);
    if (isAnswerable) {
      answerable++;
      if (r.correct) { correct++; util += 1; }
      else if (r.abstained) { util += 0; }
      else { util -= 1; }
    } else {
      if (r.abstained) { util += 1; }
      else { util -= 1; hallucinations++; }
    }
    latency += r.latencyMs; hops += r.hops; ctx += r.contextSize;
  }
  const n = tasks.length || 1;
  return {
    accuracy: correct / (answerable || 1),       // helpfulness on answerable tasks
    riskScore: (util / n + 1) / 2,               // risk-adjusted utility in [0,1]
    hallucinationRate: hallucinations / n,
    avgLatencyMs: latency / n,
    avgHops: hops / n,
    avgContext: ctx / n,
    n,
  };
}

/**
 * Deterministic, class-stratified train/test split. Within each class the first
 * `trainFrac` (rounded, ≥1 each side when the class has ≥2) go to train, the rest
 * to test. Used to prove the evolved genome GENERALIZES (we evolve on train, then
 * report held-out test) rather than overfitting the eval set.
 */
export function splitByClass(tasks, trainFrac = 0.6) {
  const byClass = new Map();
  for (const t of tasks) {
    const c = t.class ?? "default";
    if (!byClass.has(c)) byClass.set(c, []);
    byClass.get(c).push(t);
  }
  const train = [], test = [];
  for (const group of byClass.values()) {
    let nTrain = Math.round(group.length * trainFrac);
    if (group.length >= 2) nTrain = Math.min(group.length - 1, Math.max(1, nTrain));
    group.forEach((t, i) => (i < nTrain ? train : test).push(t));
  }
  return { train, test };
}

/**
 * Deterministic, class-stratified k-fold partition. Each fold draws ~1/k of every
 * class (round-robin), so folds are balanced. Used for cross-validated genome
 * selection: scoring on mean-minus-variance across folds rejects genomes tuned to
 * one split (e.g. a knife-edge abstainThreshold), which is what prevents overfit.
 */
export function kFoldByClass(tasks, k = 3) {
  const byClass = new Map();
  for (const t of tasks) {
    const c = t.class ?? "default";
    if (!byClass.has(c)) byClass.set(c, []);
    byClass.get(c).push(t);
  }
  const folds = Array.from({ length: k }, () => []);
  for (const group of byClass.values()) group.forEach((t, i) => folds[i % k].push(t));
  return folds.filter((f) => f.length > 0);
}

// ─── DARWIN_MUTABLE_BLOCK: mutation operators ───────────────────────────────
const FUSIONS = ["rrf", "linear", "dbsf"];
const RERANKS = ["gnn", "none"];
const STRATEGIES = ["terse", "evidence-first", "prune-explicit"];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clampI = (v, lo, hi) => clamp(Math.round(v), lo, hi);
const pick = (a) => a[Math.floor(Math.random() * a.length)];

export function mutate(genome) {
  const g = { ...genome };
  if (Math.random() < 0.4) g.cueK = clampI(g.cueK + (Math.random() * 4 - 2), 1, 12);
  if (Math.random() < 0.4) g.efSearch = clampI(g.efSearch * (0.7 + Math.random() * 0.8), 16, 256);
  if (Math.random() < 0.5) g.hybridAlpha = clamp(g.hybridAlpha + (Math.random() * 0.4 - 0.2), 0, 1);
  if (Math.random() < 0.3) g.fusion = pick(FUSIONS);
  if (Math.random() < 0.4) g.traversalDepth = clampI(g.traversalDepth + (Math.random() < 0.5 ? 1 : -1), 1, 4);
  if (Math.random() < 0.4) g.tagFanout = clampI(g.tagFanout + (Math.random() * 4 - 2), 1, 8);
  if (Math.random() < 0.4) g.pruneThreshold = clamp(g.pruneThreshold + (Math.random() * 0.2 - 0.1), 0, 0.6);
  if (Math.random() < 0.4) g.maxContent = clampI(g.maxContent + (Math.random() * 6 - 3), 1, 20);
  if (Math.random() < 0.4) g.haltConfidence = clamp(g.haltConfidence + (Math.random() * 0.3 - 0.15), 0.2, 0.9);
  if (Math.random() < 0.3) g.rerank = pick(RERANKS);
  if (Math.random() < 0.3) g.promptStrategy = pick(STRATEGIES);
  if (Math.random() < 0.4) g.abstainThreshold = clamp(g.abstainThreshold + (Math.random() * 0.2 - 0.1), 0, 0.6);
  return g;
}
// ─── END DARWIN_MUTABLE_BLOCK ───────────────────────────────────────────────

export { MemoryStore };
