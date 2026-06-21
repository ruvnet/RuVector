# ADR-210: Default-On Semantic Embeddings — all-MiniLM-L6-v2 as the Intelligence Engine's Primary Embedder

- **Status**: accepted (with hardening edits, review of 2026-06-12)
- **Date**: 2026-06-12
- **Deciders**: ruv
- **Tags**: embeddings, onnx, intelligence-engine, sona, rabitq, hnsw, performance

## Context

The `ruvector` npm package bundles `all-MiniLM-L6-v2` (a sentence-transformers
checkpoint executed by a pure-WASM ONNX runtime, zero native dependencies) and,
in `onnx-optimized.ts`, registers `bge-small-en-v1.5` and `e5-small-v2` with
fp16/int8 variants plus a zero-dependency parallel worker pool (ADR-194
lineage). Despite all this machinery, the model is effectively **off**:

1. **`IntelligenceEngine` defaults to `enableOnnx: false`** —
   `intelligence-engine.ts:183` gates on `config.enableOnnx &&
   isOnnxAvailable()`. By default, `hooks route`, memory search, pattern
   matching, and trajectory embeddings run on a 256-dim character-hash
   embedder with no semantic signal: "fix failing test" and "repair broken
   spec" share no overlap. The #517 route-learning fix (state keys derived
   from task text, Q-values per state) compounds this — semantically adjacent
   tasks cannot share learned routing because their hash embeddings are
   unrelated.
2. **SONA's TypeScript coordinator hashes characters into 64 dims**
   (`npm/packages/ruvllm/src/sona.ts createEmbedding`). The micro-LoRA
   instant-learning loop wired for #553 and `ReasoningBank.findSimilar` both
   adapt over hash buckets rather than meaning. The CI drift-gate fingerprints
   (`scripts/sona-drift/reference.rvf`) consequently measure adaptation over
   hash-space, not semantic-space.
3. **History of silent quality loss**: #523 documented an entire BEIR
   benchmark run silently executing on hash fallback (nDCG@10 0.262, rank
   11/11 vs published baselines). The contract fixes (honest
   `isOnnxInitialized()`/`isReady()`) shipped in `ruvector@0.2.29`, making a
   default-on policy safe to implement without reintroducing silent fallback.
4. **New search machinery is misaligned with the embedder's training**:
   MiniLM is trained for cosine similarity; the RaBitQ two-stage path
   (rvf-runtime 0.3.0) is L2-only in v1, and the HNSW/RaBitQ recall gates
   (ef_search=256 floor, 640-candidate RaBitQ floor) were tuned on uniform
   random vectors — the worst case for ANN, unrepresentative of real text
   embeddings with low intrinsic dimensionality.
5. **Known quality ceiling**: #524 measured BGE-base at +0.08 nDCG@10 over
   MiniLM on BEIR NFCorpus (rank 10/11 → 2/11). `bge-small-en-v1.5` is
   registered, same 384 dims (index-compatible drop-in), but there is no
   default-selection logic and no query/passage prefix support, which
   E5/BGE-class models require for full quality.

## Decision

Defaulting to ONNX MiniLM is not a model upgrade. It is a **contract
upgrade**: the intelligence layer stops pretending hash buckets are meaning
and makes semantic behavior the normal path, while keeping fallback
observable, deterministic, and auditable.

Make semantic embeddings the default brain of the intelligence layer, in four
coordinated changes plus one cross-cutting invariant:

### D0. Embedding-provenance invariant (cross-cutting, mandatory)

Every persisted vector store (hooks intelligence stores, HNSW memories,
`.rvf`/`.db` files created through the embedding path) MUST record:

```
{ embedderKind, modelId, dimension, normalize, prefixPolicy }
```

- Inserts whose provenance does not match the store's recorded provenance are
  **refused** (clear error naming both sides), not coerced. No mixed stores.
- Legacy stores without provenance metadata are treated as
  `{ embedderKind: "hash", dimension: <recorded or inferred>, normalize:
  false, prefixPolicy: "none" }` and open **read-only** for vector writes
  until re-embedded.
- This invariant is the defense against the decision's real failure mode —
  partial migration (see Risks).

### D1. `enableOnnx` defaults to true with graceful, *loud* fallback
- `IntelligenceEngine` constructor flips to `enableOnnx: true`.
- Initialization is lazy and non-blocking (existing `initOnnx()` pattern);
  until ready, or when the model cannot load (offline, restricted CI), the
  engine uses the hash embedder and **reports it**: `stats().embedderKind =
  'onnx-minilm' | 'hash-fallback'`, a one-line stderr notice on first
  fallback, and the existing honest `isOnnxInitialized()` gate. No silent
  quality loss (the #523 failure mode) — fallback is visible in stats, logs,
  and the quality envelope.
- Embedding dimension migration: ONNX path is 384-dim, hash path 256-dim.
  Persisted stores created at 256 dims continue to load (dimension recorded
  in the store/sidecar); mixing is refused with a clear re-embed message.
  `hooks` intelligence stores (`.ruvector/intelligence.json` + HNSW memories)
  record the embedder kind and dimension; a `hooks reembed` maintenance
  command upgrades hash-era memories on demand.

### D2. Normalize at embed time; align RaBitQ/HNSW with cosine geometry
- The embedder's `normalize` option becomes default-true (unit vectors).
  The precise claim: for unit-norm vectors, `||a − b||² = 2 − 2·cos(a, b)`,
  so L2 distance is a strictly decreasing function of cosine similarity and
  the two rankings are identical — **but only when both vectors are unit
  norm**. The D0 provenance invariant (`normalize: true` recorded per store,
  mixed inserts refused) is what makes this equivalence safe to rely on; a
  single un-normalized vector in the store silently breaks the ranking
  equivalence. With it, the L2-only RaBitQ v1 estimator and the HNSW distance
  kernels compose with MiniLM today — no third correction scalar, no
  IP/cosine codec work.
- Re-tune the documented ANN floors on text embeddings: add a benchmark
  embedding a fixed public text corpus (deterministic, committed fixture)
  with MiniLM, measuring recall@10 vs exact for HNSW (ef sweep) and RaBitQ
  (oversample sweep). Publish measured floors for the text-embedding regime
  alongside the existing uniform-random worst-case floors. Gates stay
  conservative; documentation stops overstating required ef/oversample for
  real workloads.

### D3. Bulk paths route through the int8 parallel pool
- `insert <db> <file>` (CLI), memory import, and any batch-embedding path use
  the bundled `ParallelEmbedder` worker pool with the registered int8 variant
  by default (fp32 single-session remains for single-query latency).
  Rationale: int8 ≈ 4× smaller download/resident size, ~2× CPU throughput,
  ~1-point quality cost — the right trade for ingest; queries keep fp32.
- Pool startup remains lazy; model bytes are shared with the main session
  (existing `bundledPool` capture).

### D4. Model registry grows defaults + prefix conventions (prepares #524)
- Loader metadata gains `queryPrefix`/`passagePrefix` fields; the embedder
  applies them automatically (`embedQuery` vs `embedPassage` entry points;
  plain `embed()` = passage). Per-model facts, from the model cards:
  - `all-MiniLM-L6-v2`: 384-dim, general semantic search, **no prefixes**
    (https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2).
  - `e5-small-v2`: **requires** `query: ` / `passage: ` prefixes — the model
    card states quality degrades without them
    (https://huggingface.co/intfloat/e5-small-v2).
  - `bge-small-en-v1.5`: query instruction **recommended for short-query →
    long-passage retrieval**; passages need no instruction
    (https://huggingface.co/BAAI/bge-small-en-v1.5).
  The registry encodes exactly these policies (`prefixPolicy: none |
  required | query-recommended`), and `prefixPolicy` is part of the D0
  provenance record so stores embedded with and without prefixes can never
  silently mix.
- Default model stays `all-MiniLM-L6-v2` in this ADR (bundle-size neutral).
  Switching the default to BGE-small (same 384 dims) is deferred to #524's
  decision once bundling strategy is settled; this ADR makes that switch a
  one-line registry change with prefixes already handled.

### D5. Rollout flags (operator escape hatches)

Environment variables override config, for staged rollout and incident
response without code changes:

- `RUVECTOR_EMBEDDER=auto|minilm|hash` — `auto` (default): MiniLM when
  loadable, loud hash fallback otherwise; `minilm`: hard-require the model
  (fail rather than fall back); `hash`: force the legacy embedder
  (bug-for-bug escape hatch).
- `RUVECTOR_ONNX=0|1` — kill switch for the entire ONNX runtime path
  (`0` ≡ `RUVECTOR_EMBEDDER=hash`); `1` ≡ `minilm`.
- `RUVECTOR_REEMBED=refuse|warn|auto` — what happens when opening a store
  whose provenance mismatches the active embedder: `refuse` (default; the D0
  invariant), `warn` (open read-only with a warning), `auto` (re-embed
  in place — requires source text to be present; refuses otherwise).

### Acceptance gates (test-enforced before the default flips)

1. `stats().embedderKind === "onnx-minilm"` when the model loads.
2. Fallback emits exactly **one** warning per process, not one per call.
3. A 256-dim legacy store opens **read-only** for vector writes.
4. Mixed-provenance insert (256-dim hash store + 384-dim MiniLM vector, or
   prefix-policy mismatch) fails with a clear error.
5. Normalized embedding L2 norm ∈ [0.999, 1.001] for every emitted vector.
6. `embedQuery()` applies the registered prefix for E5/BGE.
7. MiniLM applies **no** prefix on either entry point.
8. The RaBitQ/HNSW recall benchmark runs on the real text fixture (D2), not
   only uniform-random vectors.

### Explicit non-goals
- No change to the Rust crates' embedding story (ruvllm neural embeddings are
  ADR-074 territory).
- No Python/`@xenova/transformers` dependency — the xenova → sharp → libvips
  chain stays out (per #524's analysis).
- SONA TS coordinator migration to MiniLM (item 2 in Context) is staged
  separately because it invalidates the CI drift-gate reference fingerprints:
  it requires a coordinated `rvf-fingerprint.mjs --update` with pinned model
  bytes for determinism. Tracked as a follow-up, not part of this ADR's
  initial landing.

## Consequences

### Positive
- Every learned-routing, memory-recall, and pattern-match decision upgrades
  from token overlap to semantics, compounding the #517 fix (semantically
  near tasks share learned routing patterns).
- The silent-fallback failure class stays closed: fallback is loud,
  inspectable, and quality-attributed (embedderKind in stats/envelope).
- MiniLM + RaBitQ/HNSW compose correctly via unit-norm vectors; users get
  honest, regime-appropriate recall guidance instead of worst-case-only
  numbers.
- Ingest throughput: parallel int8 embedding makes embed-at-ingest the cheap
  default instead of an opt-in cost.
- Prefix support removes a correctness trap before BGE/E5 adoption (#524).

### Negative
- **The primary risk is not latency — it is partial migration**: old hash
  memories, new MiniLM memories, and SONA hash fingerprints coexisting
  without clear attribution would corrupt every similarity comparison
  silently. The D0 provenance invariant (mandatory metadata, refused mixed
  inserts, read-only legacy stores) is the mitigation, and acceptance gates
  3–4 enforce it; if D0 ships incompletely, this ADR's default flip must not
  ship at all.
- First-use latency and ~23 MB model download (or bundle weight, per #524's
  eventual bundling decision) become the default experience; offline/CI
  environments exercise the fallback path routinely — mitigated by lazy init,
  the disk cache, and the loud-fallback contract.
- 384-dim vectors cost 1.5× the memory/compute of the 256-dim hash space for
  hooks memories; HNSW/RaBitQ offset this at search time, and int8 offsets it
  at ingest.
- Dimension migration adds a maintenance surface (`hooks reembed`, mixed-dim
  refusal) that must be tested.
- One more divergence between the three SONA implementations until the staged
  coordinator migration lands (the drift gate documents rather than blocks
  this, since fingerprints are per-implementation).

### Neutral
- Default model identity is unchanged (MiniLM); quality-ceiling work moves to
  #524 with the prefix groundwork laid.
- The hash embedder remains in-tree permanently as the deterministic,
  dependency-free fallback and the no-model CI path.

## Links

- Model cards: [all-MiniLM-L6-v2](https://huggingface.co/sentence-transformers/all-MiniLM-L6-v2)
  · [e5-small-v2](https://huggingface.co/intfloat/e5-small-v2)
  · [bge-small-en-v1.5](https://huggingface.co/BAAI/bge-small-en-v1.5)
- Issue #517 — route learning (state keys over task text; semantic synergy)
- Issue #523 — ONNX contract fixes that make default-on safe (shipped 0.2.29)
- Issue #524 — BGE bundling (+0.08 nDCG@10); prefix conventions prepared here
- Issue #553 / `scripts/sona-drift/` — SONA coordinator hash embedder; staged
  follow-up with drift-reference regeneration
- ADR-074 — ruvllm neural embeddings (Rust-side, related but separate)
- ADR-194 lineage — bundled parallel worker pool used by D3
- `npm/packages/ruvector/src/core/onnx-embedder.ts`, `onnx-optimized.ts`,
  `intelligence-engine.ts` — implementation surfaces
