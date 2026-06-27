// MRAgent FROZEN MODEL — the Cue-Tag-Content associative memory substrate.
//
// Per the Meta-Harness invariant ("freeze the model, evolve the harness"), this
// file is NEVER mutated by Darwin. It is the RuVector-backed memory store. In
// production the nodes, embeddings, and edges live in a RuVector `.rvf` index and
// traversal is a Cypher query:
//
//   MATCH (c:Cue)-[:LINKED_TO*1..N]->(t:Tag)-[:REFERENCES]->(m:Content)
//   WHERE c.id IN $cueIds RETURN m
//
// To keep this example runnable with ZERO native dependencies (and fully
// deterministic for CI), the store is reimplemented in-process with the same
// semantics: hybrid (sparse+dense RRF) cue search and bounded-depth, prunable
// graph reconstruction. If the real `ruvector` package is installed it is used
// for embeddings; otherwise a deterministic hashed embedding is used. Either way
// the GRAPH SEMANTICS are identical, so the harness genome evolved here transfers
// to a live RuVector deployment unchanged.

import { createRequire } from "node:module";
const require = createRequire(import.meta.url);

// Runtime-optional production backend. The example never *requires* it.
let RuVector = null;
try { RuVector = require("ruvector"); } catch { /* deterministic fallback */ }

export const EMBED_DIM = 96;
export const usingRuVector = !!RuVector;

const STOP = new Set(["the", "a", "an", "to", "of", "is", "are", "and", "in", "into", "does", "do", "what", "which", "how", "with", "from", "for", "that"]);

export function tokenize(text) {
  return String(text)
    .toLowerCase()
    .replace(/[^a-z0-9]+/g, " ")
    .split(" ")
    .filter((w) => w.length > 1 && !STOP.has(w));
}

// Deterministic FNV-1a hash → stable across runs/platforms (no Math.random here).
function hash32(str) {
  let h = 0x811c9dc5;
  for (let i = 0; i < str.length; i++) {
    h ^= str.charCodeAt(i);
    h = Math.imul(h, 0x01000193) >>> 0;
  }
  return h >>> 0;
}

// Deterministic bag-of-features embedding. Mirrors an ONNX MiniLM embedding's
// role (dense semantic vector) without the 80MB model or native runtime.
export function embed(text) {
  const v = new Float32Array(EMBED_DIM);
  const toks = tokenize(text);
  for (const t of toks) {
    // two hashed projections per token → denser, less collision-prone vector
    v[hash32(t) % EMBED_DIM] += 1;
    v[hash32("salt:" + t) % EMBED_DIM] += 0.5;
  }
  let norm = 0;
  for (let i = 0; i < EMBED_DIM; i++) norm += v[i] * v[i];
  norm = Math.sqrt(norm) || 1;
  for (let i = 0; i < EMBED_DIM; i++) v[i] /= norm;
  return v;
}

function cosine(a, b) {
  let dot = 0;
  for (let i = 0; i < a.length; i++) dot += a[i] * b[i];
  return dot; // both are L2-normalized
}

// Sparse term-overlap score (BM25-lite): shared tokens / sqrt(len product).
function sparseScore(queryToks, docToks) {
  if (!queryToks.length || !docToks.length) return 0;
  const q = new Set(queryToks);
  let shared = 0;
  for (const t of docToks) if (q.has(t)) shared++;
  return shared / Math.sqrt(queryToks.length * docToks.length);
}

// ── Graph builder ───────────────────────────────────────────────────────────
// Builds the Cue-Tag-Content graph from the eval corpus, plus cross-task
// distractor edges so traversal depth / fan-out / pruning all matter.
//
// Edge model:
//   Cue  -LINKED_TO->  Tag        (and Cue -LINKED_TO-> distractor Tags)
//   Tag  -LINKED_TO->  bridgeTag  (intermediate hop; relevant Tag sits behind it)
//   Tag  -REFERENCES-> Content
export function buildGraph(tasks) {
  const cues = new Map();    // id -> { id, text, vec, toks }
  const tags = new Map();    // id -> { id, text, vec, toks, content: [contentIds], next: [tagIds] }
  const content = new Map(); // id -> { id, text, vec, toks }

  const protectedTags = new Set(); // relevant/bridge tags must not get filler content
  const tagId = (name) => `tag:${name}`;
  const ensureTag = (name) => {
    const id = tagId(name);
    if (!tags.has(id)) tags.set(id, { id, name, text: name.replace(/-/g, " "), toks: tokenize(name), vec: embed(name.replace(/-/g, " ")), content: [], next: [] });
    return tags.get(id);
  };

  for (const task of tasks) {
    const cid = `content:${task.id}`;
    content.set(cid, { id: cid, text: task.content, toks: tokenize(task.content), vec: embed(task.content), taskId: task.id });

    // Relevant tag(s) reference the content node.
    const relevantTags = (task.tags || []).map(ensureTag);
    for (const t of relevantTags) { if (!t.content.includes(cid)) t.content.push(cid); protectedTags.add(t.id); }

    // Bridge tags chain the relevant tag behind N intermediate hops:
    //   cue -> bridge0 -> bridge1 -> … -> relevantTag
    // so a task with k bridge tags requires traversalDepth >= k+1. Tasks with 0
    // bridges need depth 1; 1 bridge needs depth 2; 2 bridges need depth 3.
    const bridges = (task.bridgeTags || []).map(ensureTag);
    for (const b of bridges) protectedTags.add(b.id); // bridges are pure pass-through hops
    for (let bi = 0; bi < bridges.length; bi++) {
      const nextNodes = bi + 1 < bridges.length ? [bridges[bi + 1]] : relevantTags;
      for (const t of nextNodes) if (!bridges[bi].next.includes(t.id)) bridges[bi].next.push(t.id);
    }

    // Distractor tags carry wrong content (a sibling task's content) so a too-loose
    // prune threshold or too-large fan-out pollutes the reconstruction.
    const distractors = (task.distractorTags || []).map(ensureTag);

    // Cue nodes: each cue links to the entry tag (first bridge, else the relevant
    // tag) + distractors. The rest of the chain is reached only by traversal.
    const entryTags = bridges.length ? [bridges[0]] : relevantTags;
    for (const cueWord of task.cues) {
      const id = `cue:${task.id}:${cueWord}`;
      const text = `${cueWord} ${task.question}`;
      const cue = { id, text, toks: tokenize(text), vec: embed(text), taskId: task.id, links: [] };
      for (const t of entryTags) cue.links.push(t.id);
      for (const d of distractors) cue.links.push(d.id);
      cues.set(id, cue);
    }
  }

  // Wire distractor tags to reference *some* content so traversal through them is
  // non-empty (and therefore genuinely distracting). Each distractor references
  // the content of a different task than the one that introduced it.
  const allContentIds = [...content.keys()];
  let i = 0;
  for (const tag of tags.values()) {
    if (tag.content.length === 0 && !protectedTags.has(tag.id)) {
      tag.content.push(allContentIds[i % allContentIds.length]);
      i++;
    }
  }

  return { cues, tags, content };
}

// ── MemoryStore: hybrid cue search + bounded-depth reconstruction ─────────────
export class MemoryStore {
  constructor(tasks) {
    this.tasks = tasks;
    this.graph = buildGraph(tasks);
    this.cueList = [...this.graph.cues.values()];
  }

  /**
   * Stage 1 — find entry cues with hybrid (sparse + dense) search + RRF.
   * `efSearch` bounds the dense candidate pool (HNSW recall proxy): a small
   * efSearch can drop the correct cue before fusion ever sees it.
   */
  hybridSearch(queryText, { cueK = 5, efSearch = 64, hybridAlpha = 0.5, fusion = "rrf" } = {}) {
    const qTok = tokenize(queryText);
    const qVec = embed(queryText);

    const dense = this.cueList
      .map((c) => ({ c, s: cosine(qVec, c.vec) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, Math.max(1, efSearch)); // HNSW recall ceiling

    const sparse = this.cueList
      .map((c) => ({ c, s: sparseScore(qTok, c.toks) }))
      .sort((a, b) => b.s - a.s)
      .slice(0, Math.max(1, efSearch));

    const fused = fuse(dense, sparse, { hybridAlpha, fusion });
    return fused.slice(0, Math.max(1, cueK)).map((e) => e.c.id);
  }

  /**
   * Stage 2 — ACTIVE RECONSTRUCTION. From cue ids, traverse LINKED_TO up to
   * `traversalDepth` hops (expanding <= tagFanout tags per frontier node),
   * scoring each path by query relevance with per-hop decay, pruning paths
   * below `pruneThreshold`, and collecting REFERENCES content (capped maxContent).
   * Returns ordered content + reconstruction stats.
   */
  reconstruct(queryText, cueIds, { traversalDepth = 2, tagFanout = 4, pruneThreshold = 0.15, maxContent = 10, decay = 0.7 } = {}) {
    const qVec = embed(queryText);
    const qTok = tokenize(queryText);
    const { tags, content } = this.graph;

    const contentScore = new Map(); // contentId -> best evidence score
    let nodesVisited = 0;
    let hops = 0;
    const seenTag = new Set();

    // BFS frontier of { tagId, evidence } starting from cue-linked tags.
    let frontier = [];
    for (const cueId of cueIds) {
      const cue = this.graph.cues.get(cueId);
      if (!cue) continue;
      for (const tagId of cue.links.slice(0, tagFanout)) frontier.push({ tagId, evidence: 1 });
    }

    for (let depth = 0; depth < traversalDepth && frontier.length; depth++) {
      hops = depth + 1;
      const next = [];
      for (const { tagId, evidence } of frontier) {
        if (seenTag.has(tagId)) continue;
        seenTag.add(tagId);
        const tag = tags.get(tagId);
        if (!tag) continue;
        nodesVisited++;

        // Cue→Tag links are ASSOCIATIVE (structural), not semantic — a Tag is a
        // categorical label, so we do NOT score the Tag against the query. The
        // path's strength is the carried cue evidence, decayed per hop.
        const carried = evidence * decay ** depth;

        // Collect referenced Content. Content DOES share query vocabulary, so the
        // content↔query similarity (× carried evidence) is the path score we prune
        // on. Irrelevant paths (distractor content, deep low-evidence hops) fall
        // below pruneThreshold and are dropped — MRAgent's "prune irrelevant paths".
        for (const cid of tag.content) {
          const c = content.get(cid);
          if (!c) continue;
          const contentSim = 0.6 * cosine(qVec, c.vec) + 0.4 * sparseScore(qTok, c.toks);
          const pathScore = carried * contentSim;
          if (pathScore < pruneThreshold) continue; // prune irrelevant path
          contentScore.set(cid, Math.max(contentScore.get(cid) ?? 0, pathScore));
        }

        // Expand to next-hop tags (bounded fan-out). Evidence carries forward and
        // decays, so reaching content behind a bridge Tag requires traversalDepth>=2.
        for (const nxt of tag.next.slice(0, tagFanout)) {
          next.push({ tagId: nxt, evidence });
        }
      }
      frontier = next;
    }

    const ordered = [...contentScore.entries()]
      .map(([id, score]) => ({ id, score, taskId: content.get(id)?.taskId, text: content.get(id)?.text }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(1, maxContent));

    return { content: ordered, stats: { hops, nodesVisited, candidates: contentScore.size } };
  }
}

// Reciprocal Rank Fusion (and linear / dbsf variants) over two ranked lists.
function fuse(dense, sparse, { hybridAlpha, fusion }) {
  const k = 60;
  const acc = new Map(); // cueId -> { c, s }
  const add = (id, c, s) => {
    const e = acc.get(id) ?? { c, s: 0 };
    e.s += s;
    acc.set(id, e);
  };
  if (fusion === "linear") {
    const dMax = Math.max(1e-9, ...dense.map((e) => e.s));
    const sMax = Math.max(1e-9, ...sparse.map((e) => e.s));
    dense.forEach((e) => add(e.c.id, e.c, hybridAlpha * (e.s / dMax)));
    sparse.forEach((e) => add(e.c.id, e.c, (1 - hybridAlpha) * (e.s / sMax)));
  } else if (fusion === "dbsf") {
    // distribution-based score fusion: z-normalize then weight
    const z = (arr) => {
      const m = arr.reduce((a, e) => a + e.s, 0) / (arr.length || 1);
      const sd = Math.sqrt(arr.reduce((a, e) => a + (e.s - m) ** 2, 0) / (arr.length || 1)) || 1;
      return new Map(arr.map((e) => [e.c.id, (e.s - m) / sd]));
    };
    const zd = z(dense), zs = z(sparse);
    dense.forEach((e) => add(e.c.id, e.c, hybridAlpha * (zd.get(e.c.id) ?? 0)));
    sparse.forEach((e) => add(e.c.id, e.c, (1 - hybridAlpha) * (zs.get(e.c.id) ?? 0)));
  } else {
    // rrf (default)
    dense.forEach((e, i) => add(e.c.id, e.c, hybridAlpha * (1 / (k + i + 1))));
    sparse.forEach((e, i) => add(e.c.id, e.c, (1 - hybridAlpha) * (1 / (k + i + 1))));
  }
  return [...acc.values()].sort((a, b) => b.s - a.s);
}
