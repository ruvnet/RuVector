// Concept layer — gives the FROZEN model a genuine *semantic* dimension that is
// decoupled from raw lexical overlap.
//
// Why this matters: with a plain hash-of-tokens embedding, dense cosine and
// sparse term-overlap are almost perfectly correlated, so `hybridAlpha` and
// `fusion` have nothing to tune (ADR-269 measured Δfit≈0 for both). Real
// embeddings differ: paraphrases ("rapid cold-start" ~ "fast boot") are dense-
// close with ZERO token overlap, while rare identifiers ("rvf-7", "cve-2") are
// lexically decisive but semantically generic.
//
// We model that split deterministically: tokens that belong to a synonym group
// project onto a shared CONCEPT dimension (dense semantics), and identifier-like
// tokens stay in a lexical tail. Result:
//   • semantic queries  → answerable by DENSE only  (no shared tokens)
//   • lexical  queries  → answerable by SPARSE only (concept-generic)
//   • hybrid   queries  → need RRF over both
// which is exactly the regime where hybridAlpha + fusion are load-bearing.

// Synonym groups → concept ids. Tokens in the same group are dense-equivalent.
const CONCEPT_GROUPS = [
  ["fast", "rapid", "quick", "speed", "swift", "low-latency", "sub-millisecond", "instant"],
  ["boot", "cold-start", "startup", "initialize", "cold-boot", "launch", "spin-up"],
  ["compress", "compression", "quantize", "quantization", "shrink", "squeeze", "pack"],
  ["store", "storage", "persist", "write", "save", "backend", "durable"],
  ["search", "retrieve", "retrieval", "query", "lookup", "find", "recall"],
  ["graph", "topology", "network", "node", "nodes", "edge", "edges", "associative"],
  ["consensus", "agreement", "leader", "elect", "authoritative", "quorum"],
  ["secure", "security", "tamper", "tamper-evident", "witness", "proof", "cryptographic", "immutable"],
  ["merge", "fuse", "fusion", "combine", "aggregate", "blend"],
  ["prune", "filter", "drop", "discard", "remove", "trim"],
  ["accuracy", "recall", "precision", "fidelity", "correct", "quality"],
  ["memory", "remember", "reconstruct", "reconstruction", "cue", "tag", "content"],
  ["validate", "validation", "reject", "fail-fast", "guard", "check"],
  ["concurrency", "lock-free", "parallel", "branching", "copy-on-write", "throughput"],
  ["embedding", "vector", "dense", "representation", "latent"],
];

export const NUM_CONCEPTS = CONCEPT_GROUPS.length;

// Canonical concept name = first token of each group. Corpus specs reference
// concepts by these names; buildGraph synthesizes DIFFERENT surface tokens from
// the same group for query vs cue, so they share a concept but not a token.
export const CONCEPT_NAMES = CONCEPT_GROUPS.map((g) => g[0]);

const NAME_TO_INDEX = new Map(CONCEPT_NAMES.map((n, i) => [n, i]));

/** k-th distinct surface token of a concept (by name), wrapping the group. */
export function syn(conceptName, k = 0) {
  const ci = NAME_TO_INDEX.get(conceptName);
  if (ci === undefined) return conceptName; // treat unknown as a literal token
  const group = CONCEPT_GROUPS[ci];
  return group[k % group.length];
}

const TOKEN_TO_CONCEPT = new Map();
CONCEPT_GROUPS.forEach((group, ci) => {
  for (const tok of group) TOKEN_TO_CONCEPT.set(tok, ci);
});

/** Concept id for a token, or -1 if it is lexical-only (identifier-like). */
export function conceptOf(token) {
  if (TOKEN_TO_CONCEPT.has(token)) return TOKEN_TO_CONCEPT.get(token);
  return -1;
}

// A token is "identifier-like" (purely lexical) if it carries a digit or hyphen
// with a digit, or is a known id prefix. These never get a concept, so only
// sparse search can pin them down.
export function isIdentifier(token) {
  return /\d/.test(token) || /^(rvf|cve|adr|t\d|id)/.test(token);
}
