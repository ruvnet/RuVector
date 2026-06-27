// MRAgent EVOLVED HARNESS — this is the code surface Darwin Mode mutates.
//
// Paper: "Memory is Reconstructed, Not Retrieved: Graph Memory for LLM Agents"
// (MRAgent). Memory is a Cue-Tag-Content associative graph; answering a question
// is an *active reconstruction* — search for cues, traverse cue→tag→content,
// prune irrelevant paths, synthesize. The reconstruction dynamics live in the
// GENOME below. The memory substrate (agent/memory.mjs) stays frozen.
//
// Darwin edits the DARWIN_MUTABLE_BLOCK regions to maximize fitness (accuracy
// minus reconstruction cost). Everything outside those blocks is structural.

import { MemoryStore } from "./memory.mjs";

// ─── DARWIN_MUTABLE_BLOCK: reconstruction genome ────────────────────────────
// These are the knobs Darwin evolves. Each maps to a real RuVector retrieval /
// Cypher-traversal parameter, so an evolved genome transfers to production.
export function baselineGenome() {
  return {
    // Stage 1 — hybrid cue search (RuVector hybridSearch).
    cueK: 5,             // initial cue vectors fetched           [1..12]
    efSearch: 64,        // HNSW search depth / candidate pool     [16..256]
    hybridAlpha: 0.5,    // RRF weight: 0=sparse … 1=dense         [0..1]
    fusion: "rrf",       // rrf | linear | dbsf

    // Stage 2 — active reconstruction (Cypher LINKED_TO*1..N traversal).
    traversalDepth: 2,   // cue→tag→content hops                   [1..4]
    tagFanout: 4,        // max tags expanded per frontier node    [1..8]
    pruneThreshold: 0.15,// drop paths below this evidence score   [0..0.6]
    maxContent: 10,      // content nodes handed to synthesis(LIMIT)[1..20]

    // Stage 3 — synthesis (LLM prompt strategy for pruning/grounding).
    rerank: "gnn",       // gnn | none  (self-learning GNN rerank)
    promptStrategy: "evidence-first", // terse | evidence-first | prune-explicit
  };
}
// ─── END DARWIN_MUTABLE_BLOCK ───────────────────────────────────────────────

// Effective synthesis window per prompt strategy. A terse prompt only reads the
// top of the reconstructed context; evidence-first reads the full LIMIT;
// prune-explicit reads a middle window but is penalised if distractor content
// outranks the answer (it instructs the LLM to prune, so a noisy top hurts).
const STRATEGY_WINDOW = { terse: 3, "evidence-first": Infinity, "prune-explicit": 6 };

/**
 * Deterministic synthesis judge — stands in for the LLM call. Returns whether
 * the reconstructed context lets the model surface the expected fact, given the
 * prompt strategy's effective window. Deterministic so the eval is reproducible.
 */
function synthesize(reconstructed, task, genome) {
  const window = STRATEGY_WINDOW[genome.promptStrategy] ?? Infinity;
  const visible = reconstructed.slice(0, window === Infinity ? reconstructed.length : window);
  const hitIdx = visible.findIndex((c) => c.taskId === task.id);
  if (hitIdx === -1) return { correct: false, answer: "I don't have that in memory." };

  // prune-explicit: if 2+ distractor contents rank above the answer, the model
  // is told to prune and may discard the (low-ranked) correct path.
  if (genome.promptStrategy === "prune-explicit") {
    const distractorsAbove = visible.slice(0, hitIdx).filter((c) => c.taskId !== task.id).length;
    if (distractorsAbove >= 2) return { correct: false, answer: "Pruned: ambiguous evidence." };
  }
  return { correct: true, answer: task.content };
}

// Optional GNN rerank: nudge content that is corroborated by multiple high-score
// paths upward (proximity-weighted). Frozen weights — this is a harness toggle,
// not model training.
function gnnRerank(reconstructed) {
  const boost = new Map();
  for (const c of reconstructed) boost.set(c.taskId, (boost.get(c.taskId) ?? 0) + c.score);
  return [...reconstructed]
    .map((c) => ({ ...c, score: 0.7 * c.score + 0.3 * (boost.get(c.taskId) ?? 0) }))
    .sort((a, b) => b.score - a.score);
}

/**
 * The MRAgent reasoning loop for ONE question. Pure function of (question, store,
 * genome) → deterministic result with latency/hop telemetry for scoring.
 */
export function runReasoningLoop(question, store, genome, task) {
  // 1. Hybrid search for entry cues.
  const cueIds = store.hybridSearch(question, genome);

  // 2. Active reconstruction: traverse + prune the Cue-Tag-Content graph.
  let { content, stats } = store.reconstruct(question, cueIds, genome);

  // 3. Optional GNN rerank before synthesis.
  if (genome.rerank === "gnn") content = gnnRerank(content);

  // 4. Synthesis.
  const out = task ? synthesize(content, task, genome) : { correct: false, answer: "" };

  // Deterministic latency proxy (µs-scale weights mirror RuVector cost drivers):
  //   efSearch dominates stage-1, nodesVisited dominates traversal, maxContent
  //   dominates the synthesis context cost.
  const latencyMs =
    0.02 * genome.efSearch +
    0.05 * stats.nodesVisited +
    0.30 * Math.min(content.length, genome.maxContent) +
    (genome.rerank === "gnn" ? 0.4 : 0);

  return { ...out, latencyMs, hops: stats.hops, nodesVisited: stats.nodesVisited, contextSize: content.length, cueIds };
}

/**
 * Evaluate a genome over the whole eval set → aggregate metrics. This is what
 * the Darwin scorePolicy and the benchmark consume.
 */
export function evaluate(genome, store, tasks) {
  let correct = 0, latency = 0, hops = 0, ctx = 0;
  for (const task of tasks) {
    const r = runReasoningLoop(task.question, store, genome, task);
    if (r.correct) correct++;
    latency += r.latencyMs;
    hops += r.hops;
    ctx += r.contextSize;
  }
  const n = tasks.length || 1;
  return {
    accuracy: correct / n,
    avgLatencyMs: latency / n,
    avgHops: hops / n,
    avgContext: ctx / n,
    n,
  };
}

// ─── DARWIN_MUTABLE_BLOCK: mutation operators ───────────────────────────────
// Random mutation used as the deterministic fallback when no LLM write layer is
// available. Each op respects the genome's declared ranges.
const FUSIONS = ["rrf", "linear", "dbsf"];
const RERANKS = ["gnn", "none"];
const STRATEGIES = ["terse", "evidence-first", "prune-explicit"];
const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
const clampI = (v, lo, hi) => clamp(Math.round(v), lo, hi);
const pick = (a) => a[Math.floor(Math.random() * a.length)];

export function mutate(genome) {
  const g = { ...genome };
  if (Math.random() < 0.5) g.cueK = clampI(g.cueK + (Math.random() * 4 - 2), 1, 12);
  if (Math.random() < 0.5) g.efSearch = clampI(g.efSearch * (0.7 + Math.random() * 0.8), 16, 256);
  if (Math.random() < 0.5) g.hybridAlpha = clamp(g.hybridAlpha + (Math.random() * 0.4 - 0.2), 0, 1);
  if (Math.random() < 0.3) g.fusion = pick(FUSIONS);
  if (Math.random() < 0.5) g.traversalDepth = clampI(g.traversalDepth + (Math.random() < 0.5 ? 1 : -1), 1, 4);
  if (Math.random() < 0.4) g.tagFanout = clampI(g.tagFanout + (Math.random() * 4 - 2), 1, 8);
  if (Math.random() < 0.5) g.pruneThreshold = clamp(g.pruneThreshold + (Math.random() * 0.2 - 0.1), 0, 0.6);
  if (Math.random() < 0.5) g.maxContent = clampI(g.maxContent + (Math.random() * 6 - 3), 1, 20);
  if (Math.random() < 0.3) g.rerank = pick(RERANKS);
  if (Math.random() < 0.3) g.promptStrategy = pick(STRATEGIES);
  return g;
}
// ─── END DARWIN_MUTABLE_BLOCK ───────────────────────────────────────────────

export { MemoryStore };
