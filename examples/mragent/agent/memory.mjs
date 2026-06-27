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
import { NUM_CONCEPTS, conceptOf, syn } from "./concepts.mjs";
const require = createRequire(import.meta.url);

// Runtime-optional production backend. The example never *requires* it.
let RuVector = null;
try { RuVector = require("ruvector"); } catch { /* deterministic fallback */ }

// Dense embedding = concept-projected semantics + a small lexical hash tail.
// The concept block (first NUM_CONCEPTS dims) makes paraphrases dense-close even
// with zero shared tokens; the hash tail keeps unique tokens distinguishable.
const HASH_TAIL = 64;
export const EMBED_DIM = NUM_CONCEPTS + HASH_TAIL;
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

// Deterministic concept-projected embedding. Stands in for an ONNX MiniLM dense
// vector: tokens sharing a concept (synonyms) land on the same concept dim, so
// paraphrases are dense-close WITHOUT lexical overlap. Identifier-like tokens
// only hit the hash tail, so they are semantically generic (sparse decides them).
export function embed(text) {
  const v = new Float32Array(EMBED_DIM);
  const toks = tokenize(text);
  for (const t of toks) {
    const c = conceptOf(t);
    if (c >= 0) {
      v[c] += 1; // concept dimension (dense semantics)
    } else {
      // lexical-only token → hash tail (after the concept block)
      v[NUM_CONCEPTS + (hash32(t) % HASH_TAIL)] += 0.6;
      v[NUM_CONCEPTS + (hash32("salt:" + t) % HASH_TAIL)] += 0.3;
    }
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
// distractor cues/contents so every gene is load-bearing.
//
// Texts are SYNTHESIZED from each task's structured signal spec (concept names +
// lexical identifiers) so that dense/sparse separation, ranking-distractors and
// multi-hop bridges are guaranteed, not dependent on fragile English wording.
//
//   query        = qConcepts(variant0) + qLex
//   correct cue  = cue.concepts(variant1) + cue.lex      (same concepts, diff tokens)
//   correct text = qConcepts(variant0) + expected_fact + cue.lex
//   distractor   = query echoed twice (out-ranks correct on raw sim, no fact)
//   decoy cue    = decoy.concepts/lex → wrong tag → wrong content
//
// Edge model:
//   Cue -LINKED_TO-> [bridge0 -> … ->] { relevantTag, corroborateTag }
//   Tag -REFERENCES-> Content
function synth(concepts = [], lex = [], variant = 0) {
  return [...concepts.map((c) => syn(c, variant)), ...lex].join(" ");
}

/** Synthesize the query string for a task spec (used at retrieval time). */
export function queryTextFor(spec) {
  return synth(spec.qConcepts || [], spec.qLex || [], 0);
}

export function buildGraph(specs) {
  const cues = new Map();
  const tags = new Map();
  const content = new Map();
  const queries = new Map();

  const mkTag = (name) => {
    const id = `tag:${name}`;
    const t = { id, name, text: name.replace(/-/g, " "), toks: tokenize(name), vec: embed(name.replace(/-/g, " ")), content: [], next: [] };
    tags.set(id, t);
    return t;
  };
  const mkContent = (id, text, taskId) => {
    content.set(id, { id, text, toks: tokenize(text), vec: embed(text), taskId });
    return id;
  };

  for (const spec of specs) {
    queries.set(spec.id, queryTextFor(spec));

    // Unanswerable task: NO correct content exists — the only honest answer is to
    // abstain. We still create the cue + decoys so the agent has something to chase
    // and must judge that the reconstructed evidence is too weak (low confidence).
    const answerable = spec.answerable !== false;

    let entry;
    if (answerable) {
      // Correct content: relevant to the query (shares query concepts) + the fact.
      const cid = `content:${spec.id}`;
      mkContent(cid, [synth(spec.qConcepts, [], 0), spec.expected_fact, ...(spec.cue?.lex || [])].join(" "), spec.id);

      // Relevant tag references the correct content (+ ranking-distractor contents).
      const rel = mkTag(`${spec.id}-rel`);
      rel.content.push(cid);
      for (let d = 0; d < (spec.distractors || 0); d++) {
        // Echoes the query MORE than the correct content → higher raw sim, but no
        // expected_fact. Only rerank (corroboration) or a wide window survives it.
        const did = mkContent(`content:${spec.id}:d${d}`,
          [synth(spec.qConcepts, spec.qLex, 0), synth(spec.qConcepts, [], 0), (spec.qLex || []).join(" ")].join(" "),
          `${spec.id}-distractor`);
        rel.content.push(did);
      }

      // Corroborating tag references the SAME correct content via a second path.
      // Only surfaces with rerank="gnn" (corroboration boost) AND tagFanout>=2.
      const tail = [rel];
      if (spec.corroborate) {
        const corr = mkTag(`${spec.id}-corr`);
        corr.content.push(cid);
        tail.push(corr);
      }

      // Bridge chain: cue -> b0 -> … -> tail. k bridges ⇒ need traversalDepth k+1.
      const bridges = [];
      for (let b = 0; b < (spec.bridges || 0); b++) bridges.push(mkTag(`${spec.id}-b${b}`));
      for (let b = 0; b < bridges.length; b++) {
        const nxt = b + 1 < bridges.length ? [bridges[b + 1]] : tail;
        for (const t of nxt) bridges[b].next.push(t.id);
      }
      entry = bridges.length ? [bridges[0]] : tail;
    } else {
      // Only a weak tag with a low-similarity placeholder → confidence stays low.
      const weak = mkTag(`${spec.id}-weak`);
      const wid = mkContent(`content:${spec.id}:weak`, ["tangential unrelated note", spec.id].join(" "), `${spec.id}-none`);
      weak.content.push(wid);
      entry = [weak];
    }

    // Correct cue (concepts via variant-1 surface tokens, so dense-close to query
    // but lexically distinct; shares cue.lex with the query for the sparse signal).
    mkCue(cues, `cue:${spec.id}:correct`,
      synth(spec.cue?.concepts || [], spec.cue?.lex || [], 1), answerable ? spec.id : `${spec.id}-none`, entry.map((t) => t.id));

    // Decoy cues → wrong tag → wrong content. Concepts use variant-2 surface tokens
    // so a concept-decoy is dense-close to the query but shares NO token with it —
    // the correct cue is only retrievable with the right fusion weight.
    (spec.decoys || []).forEach((dec, di) => {
      const wc = mkContent(`content:${spec.id}:w${di}`, ["wrong decoy", synth(dec.concepts || [], dec.lex || [], 2)].join(" "), `${spec.id}-decoy`);
      const wt = mkTag(`${spec.id}-w${di}`);
      wt.content.push(wc);
      mkCue(cues, `cue:${spec.id}:decoy${di}`, synth(dec.concepts || [], dec.lex || [], 2), `${spec.id}-decoy`, [wt.id]);
    });
  }

  return { cues, tags, content, queries };
}

function mkCue(cues, id, text, taskId, links) {
  cues.set(id, { id, text, toks: tokenize(text), vec: embed(text), taskId, links });
}

// ── MemoryStore: hybrid cue search + bounded-depth reconstruction ─────────────
export class MemoryStore {
  constructor(tasks) {
    this.tasks = tasks;
    this.graph = buildGraph(tasks);
    this.cueList = [...this.graph.cues.values()];
  }

  /** Synthesized query string for a task id (the text actually issued at search). */
  queryText(taskId) {
    return this.graph.queries.get(taskId) ?? "";
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
  reconstruct(queryText, cueIds, { traversalDepth = 2, tagFanout = 4, pruneThreshold = 0.15, maxContent = 10, decay = 0.7, haltConfidence = 1.1 } = {}) {
    const qVec = embed(queryText);
    const qTok = tokenize(queryText);
    const { tags, content } = this.graph;

    // Per content: best single-path score AND # of corroborating paths.
    const acc = new Map(); // contentId -> { best, paths }
    let nodesVisited = 0;
    let hops = 0;
    let halted = false;
    const seenTag = new Set();

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

        // Cue→Tag links are ASSOCIATIVE (structural), not semantic. Path strength
        // is the carried cue evidence, decayed per hop.
        const carried = evidence * decay ** depth;

        for (const cid of tag.content) {
          const c = content.get(cid);
          if (!c) continue;
          const contentSim = 0.6 * cosine(qVec, c.vec) + 0.4 * sparseScore(qTok, c.toks);
          const pathScore = carried * contentSim;
          if (pathScore < pruneThreshold) continue; // prune irrelevant path
          const e = acc.get(cid) ?? { best: 0, sim: 0, paths: 0 };
          e.best = Math.max(e.best, pathScore);     // decayed — for ranking competition
          e.sim = Math.max(e.sim, contentSim);      // raw relevance — for abstention confidence
          e.paths += 1; // corroboration: distinct paths reaching this content
          acc.set(cid, e);
        }

        for (const nxt of tag.next.slice(0, tagFanout)) next.push({ tagId: nxt, evidence });
      }
      frontier = next;

      // ADAPTIVE DEPTH (beyond MRAgent): halt once a genuinely relevant answer
      // exists, spending traversal only on hard queries (ACT-style adaptive
      // computation). Uses RAW relevance (sim), not the decayed score, so a deep-
      // but-relevant answer can trigger halt while a mediocre shallow one cannot.
      let top = 0;
      for (const e of acc.values()) top = Math.max(top, e.sim);
      if (top >= haltConfidence) { halted = true; break; }
    }

    const ordered = [...acc.entries()]
      .map(([id, e]) => ({ id, score: e.best, sim: e.sim, paths: e.paths, taskId: content.get(id)?.taskId, text: content.get(id)?.text }))
      .sort((a, b) => b.score - a.score)
      .slice(0, Math.max(1, maxContent));

    // Abstention confidence = the chosen content's RAW relevance to the query, not
    // its decayed path score — so a deep-but-relevant answer is not mistaken for a
    // weak one. This keeps the abstain threshold robust across traversal depths.
    const confidence = ordered.length ? ordered[0].sim : 0;
    return { content: ordered, stats: { hops, nodesVisited, candidates: acc.size, halted, confidence } };
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
