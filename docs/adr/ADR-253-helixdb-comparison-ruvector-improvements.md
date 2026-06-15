---
adr: 253
title: "HelixDB vs RuVector — Comparative Analysis and Improvement Opportunities"
status: proposed
date: 2026-06-15
authors: [ruv, claude-flow]
related: [ADR-001, ADR-193, ADR-194, ADR-195, ADR-210]
tags: [ruvector, graph, vector, helixdb, query-language, schema, hnsw, storage, benchmarks, rag, dx, competitive-analysis]
---

# ADR-253 — HelixDB vs RuVector: Comparative Analysis and Improvement Opportunities

## Status

**Accepted — partially implemented.** This is an analysis-and-direction ADR. It
compares [HelixDB](https://github.com/HelixDB/helix-db) against RuVector's graph +
vector stack (`crates/ruvector-core`, `crates/ruvector-graph`,
`crates/ruvector-server`, and the `ruvector` npm package) and proposes a
prioritized set of improvements that adopt HelixDB's strongest ideas where they
fit RuVector's thesis.

**P1, P2, P3, and P4 are now implemented natively** in `ruvector-graph`
on branch `claude/helixdb-ruvector-comparison-qmsx42`:

- `crates/ruvector-graph/src/schema.rs` (P1) — opt-in `GraphSchema` (node/edge/vector
  type declarations, indexed properties, `from`/`to` edge constraints,
  vector-type→label/property binding), load-time validation (`validate_self`,
  `validate_node`, `validate_edge`, `validate_vector_dims`), higher-is-better
  distance metrics, and `reciprocal_rank_fusion` (P4 core).
- `crates/ruvector-graph/src/typed_graph.rs` (P2) — `TypedGraph` wrapper that
  validates mutations before they touch storage and a fused, typed
  `search_then_traverse` operator (HelixQL's `SearchV<T>(q,k)::In/Out<E>`), with
  optimized bounded-heap top-k (O(n log k)), a rayon parallel scan, and an opt-in
  HNSW push-down path.
- `crates/ruvector-graph/src/embed.rs` (P3) — `Embedder` trait + `HashEmbedder`,
  with inline `embed()` at insert/query on `TypedGraph`.
- `crates/ruvector-graph/src/bm25.rs` (P4) — Okapi-BM25 index feeding the
  tri-modal `hybrid_search_text` (vector + keyword + graph, RRF-fused).
- `crates/ruvector-graph/src/codegen.rs` (P6) — schema-driven typed client
  codegen: deterministic TypeScript interfaces, Python `TypedDict`s, and Rust
  structs plus a vector-type manifest, carrying labels/property types/edge
  `from`/`to`/vector dims across the language boundary.

Pure-Rust (rayon already a crate dependency), WASM-safe schema/embed/BM25/codegen
layers; **42 new unit tests, 168/168 lib tests green, clippy clean**.

A criterion benchmark (`crates/ruvector-graph/benches/typed_graph_bench.rs`,
P5 seed) measures the operator and validation. After optimizing the hot path —
**zero-copy borrow scoring** (`GraphDB::with_node`/`node_ids_by_label`, no node or
embedding clones), a **single-pass fused cosine** (q·c and c·c in one read of the
candidate), **query-norm hoisting**, a **bounded top-k heap** (O(n log k)), and a
**rayon parallel scan** over `DashMap` for ≥4096 candidates — measured speedups
over the first-cut implementation (128-dim, k=10):

| candidates | before | after | speedup |
|---|---|---|---|
| 1,000 (serial) | 539 µs | 432 µs | 1.25× (fused cosine) |
| 10,000 (parallel) | 7.20 ms | 3.08 ms | 2.34× |
| 50,000 (parallel) | 74.3 ms | 28.5 ms | 2.61× |

`validate_node` is 128 ns; `reciprocal_rank_fusion` over 2×1000 lists is 318 µs.

**HNSW push-down (landed).** `search_then_traverse` now takes an opt-in ANN path:
`TypedGraph::build_vector_index(vector_type)` builds a per-vector-type
`HybridIndex` (HNSW under the `hnsw_rs` feature, exact `FlatIndex` otherwise),
kept current incrementally on `create_node`. The query then does an ~O(log n)
approximate search, **over-fetches** (`max(4k, k+32)`), and **rescores the
candidates exactly** with the schema metric — so the ANN path returns *identical
higher-is-better score semantics* to the brute-force path while skipping the
full-label scan. Brute force remains the default (exact, no build step, no extra
memory). Measured at 50k nodes / 128-dim / k=10:

| backend | latency | vs parallel scan | vs first cut |
|---|---|---|---|
| brute-force parallel scan | 27.6 ms | 1× | 2.7× |
| HNSW push-down | 1.05 ms | **26×** | **~70×** |

**Inline `embed()` (P3, landed).** `crates/ruvector-graph/src/embed.rs` adds a
pluggable `Embedder` trait and a dependency-free, deterministic `HashEmbedder`
(feature-hashing; lexical-overlap only — an *explicit* opt-in, never a silent
fallback per ADR-194). `TypedGraph::with_embedder` attaches a model;
`create_node_from_text` vectorizes a text field into the bound property at insert
(HelixQL `AddN<T>({ embedding: Embed(text) })`) with dimension validation, and
`search_text` embeds the query inline (`SearchV<T>(Embed(text), k)`). A real
model (MiniLM/ONNX per ADR-210) plugs in via the trait.

**Tri-modal hybrid query (P4, landed).** `crates/ruvector-graph/src/bm25.rs` is a
self-contained Okapi-BM25 inverted index; `TypedGraph::hybrid_search_text` fuses
**ANN vector similarity + BM25 keyword relevance + graph traversal** in one typed
call, combining the vector and keyword rankings with `reciprocal_rank_fusion`
(`rrf_k`, conventionally 60) before traversing the fused top-k. This tri-modal
single-call fusion goes beyond HelixDB's current ANN+BM25 hybrid by folding the
graph expansion into the same operation.

Measured: `HashEmbedder` embed is 717 ns (256-dim); the full tri-modal hybrid
(inline embed + HNSW + BM25 + RRF + traversal) over 10k docs is **1.63 ms** end
to end.

**Schema-driven codegen (P6, landed).** `crates/ruvector-graph/src/codegen.rs`
emits typed client stubs from a `GraphSchema` — `generate_typescript`,
`generate_python`, `generate_rust` — with deterministic (sorted) output suitable
for check-in. Node labels become interfaces/`TypedDict`s/structs with correctly
typed + optional/required properties, edges carry `from`/`to` label constraints,
and a vector-type manifest exports dimensions + metric. This closes HelixDB's
`schema → typed SDK` DX advantage.

The remaining items (a single-statement hybrid surface in Cypher itself, P5
head-to-head benchmarks vs Neo4j/pgvector/Qdrant/HelixDB, P7 object-storage spike)
remain authorized and will land under follow-up ADRs.

## Context

HelixDB is an open-source, Rust-built OLTP **graph-vector database** (YC X25,
~5.2k stars, Apache-2.0 on `main` as of mid-2026) positioned explicitly for RAG,
knowledge graphs, and AI-agent memory. It overlaps RuVector's territory closely
enough — Rust core, embedded-first, graph + vector in one engine, MCP-native,
RAG-focused — that it is the most direct architectural peer to RuVector's graph
stack. The user asked us to compare the two and extract concrete improvements.

The comparison is worth doing precisely because the two projects made **different
bets on the same problem**. RuVector bet on breadth and self-learning (an agentic
OS: SONA self-optimization, GNN-learned ranking, 50+ attention mechanisms, local
LLMs, WASM/`.rvf` single-file deploy, a PostgreSQL extension, DiskANN
billion-scale). HelixDB bet on a narrow, sharp wedge: a **compiled, type-safe,
schema-first query language** (HelixQL) over a tight LMDB core, with graph and
vector treated as a single first-class data model. Where HelixDB is sharper, we
should learn from it.

### What HelixDB is (technical profile)

- **Storage.** Open-source v1–v2 line is **LMDB**, accessed via **heed** (the
  Meilisearch team's Rust LMDB wrapper), with *separate* LMDB databases for nodes,
  edges, and vectors. Full ACID via LMDB MVCC, single-writer/multi-reader, very
  low read latency (zero-copy reads off the mmap). The newer **v3 / HelixDB Cloud**
  line replaces LMDB with an **LSM engine backed by object storage** (compute
  decoupled from storage, tiered in-memory + SSD cache, serializable snapshot
  isolation, single writer + auto-scaling readers) to escape LMDB's sequential-write
  and capacity limits. The embedded fast engine and the scalable Cloud engine are
  effectively *different products*.
- **Query language — HelixQL.** Strongly-typed, **compile-time-checked**,
  schema-first DSL that blends Gremlin/Cypher/Rust. Schema and queries live in
  `.hx` files; the toolchain type-checks them against the schema and **compiles
  them to Rust that becomes API endpoints** — no runtime parse overhead, type
  errors caught before deploy. A second path (Rust/TS DSL → JSON AST → `POST
  /v1/query`) allows dynamic queries with no build step.
- **Graph + vector unification — the core thesis.** Vectors live in their own
  LMDB database but are **attached to graph nodes by typed edges**, so a vector
  hit can be traversed back into the graph. Search and traversal compose in one
  typed query:

  ```
  QUERY search_similar_professors(query_vector: [F64], k: I64) =>
      vecs <- SearchV<ResearchAreaEmbedding>(query_vector, k)
      professors <- vecs::In<HasResearchAreaEmbedding>
      RETURN professors
  ```

  Built-in `Embed(text)` performs **inline vectorization at insert/query time**.
  Recent versions fuse **ANN + BM25 + multi-hop traversal in one query** with
  **RRF** (Reciprocal Rank Fusion).
- **Schema.** `N::` node, `E::` edge (typed `From`/`To`), `V::` vector type,
  `INDEX` for secondary-indexed properties. Schema is mandatory and is the unit
  of compile-time validation.
- **DX.** `helix` CLI: `init` (scaffold), `start dev` (port 6969), `query`,
  `chef` (MCP bootstrap), plus Cloud commands. SDKs for Rust, TypeScript, Go,
  Python (`helix-py`); Rust/TS DSLs emit identical JSON ASTs. Built-in **MCP**
  server for agent traversal.
- **Benchmarks (substantiated).** `graph-vector-bench` (Nov 2025, v2.1.0 vs
  Neo4j 2025.09 vs Postgres 16.10, AWS c6g.2xlarge, ~4M edges, P50/P95/P99):
  **PointGet 16x/12x**, **OneHop 5.9x/13x**, **OneHopFilter 4.2x/20x** vs
  Neo4j/Postgres respectively.
- **Caveats.** **No published vector benchmarks** (vs Qdrant/pgvector/Pinecone) —
  the "matches Pinecone/Qdrant" and "1000x faster than Neo4j" claims are marketing;
  only the 4–20x graph numbers are defensible. HNSW tunables (M/ef/metric) are not
  publicly documented. Young, fast-moving (v1→v3 in ~a year), small team, license
  has shifted across sources (AGPL-3.0 historically → Apache-2.0 on current main).

### What RuVector is today (relevant subset)

- **Storage.** `redb` (a pure-Rust embedded B-tree, MVCC, single-writer) for both
  vector metadata (`crates/ruvector-core/src/storage.rs`) and graph
  (`crates/ruvector-graph/src/storage.rs`), with memory-mapped vectors, a shared
  connection pool, and a memory-only backend for WASM. Plus the single-file `.rvf`
  container format (25 segment types) and DiskANN/Vamana SSD-backed billion-scale ANN.
- **Query.** A full **Cypher** engine in `crates/ruvector-graph/src/cypher`
  (lexer → nom parser → AST → semantic analysis → optimizer → logical/physical
  plan → executor) — but it is **interpreted at runtime**, schemaless, and not
  type-checked ahead of time. Hybrid graph+vector queries exist
  (`crates/ruvector-graph/src/hybrid`, `HybridQuery { graph_pattern: String,
  vector_constraint, ... }`) but the graph pattern is a runtime string and the
  vector binding is by property name, not a typed, compiler-verified relationship.
- **Hybrid retrieval.** Sparse + dense with RRF fusion, Graph RAG with community
  detection, ColBERT multi-vector, Matryoshka — strong, but exposed as separate
  APIs rather than one declarative query operator.
- **Indexes.** HNSW, DiskANN/Vamana, RaBitQ + RAIRS/IVF quantization
  (ADR-193), tunable and well-documented.
- **Surfaces.** REST (Axum, `crates/ruvector-server`), npm package with a bundled
  ONNX embedder (ADR-194/195), WASM/browser, PostgreSQL extension
  (`crates/ruvector-postgres`), MCP server (`crates/mcp-gate`).
- **Differentiators HelixDB lacks.** Self-learning (SONA, GNN-learned ranking),
  N-ary **hyperedges**, 50+ attention mechanisms, local LLM inference, `.rvf`
  single-file/WASM deploy, PostgreSQL embedding, Cypher familiarity.

### Head-to-head summary

| Dimension | HelixDB | RuVector | Edge |
|---|---|---|---|
| Embedded storage engine | LMDB via heed (mmap, zero-copy reads) | redb (pure-Rust B-tree, MVCC) | ~Even; LMDB more battle-tested, redb pure-Rust/WASM-friendlier |
| Decoupled/cloud storage | v3 LSM + object storage (compute/storage split) | DiskANN SSD + `.rvf`; no object-storage tier | **HelixDB** for cloud-scale elasticity |
| Query language | HelixQL — typed, **compiled to Rust**, schema-first | Cypher — familiar, **runtime-interpreted**, schemaless | **HelixDB** for safety/perf; **RuVector** for familiarity |
| Graph+vector model | Vectors as graph citizens via typed edges; one typed query | Hybrid via runtime string pattern + property binding | **HelixDB** for first-class integration |
| Inline embedding | `Embed()` in-query | Bundled ONNX embedder, not in-query | **HelixDB** for ergonomics |
| Hybrid (ANN+BM25+graph) | One query, RRF | Pieces exist (sparse+dense RRF, Graph RAG), separate APIs | **HelixDB** for unification; parity reachable |
| Hyperedges (N-ary) | Pairwise typed edges only | Native hyperedges | **RuVector** |
| Self-learning / adaptivity | None | SONA + GNN-learned ranking | **RuVector** |
| Codegen / typed SDKs | schema → Rust endpoints + multi-lang SDKs | Hand-written REST/npm bindings | **HelixDB** for DX |
| Benchmarks | Reproducible graph harness, P50/95/99 | Scattered; no public Neo4j/pgvector head-to-head | **HelixDB** for rigor |
| Breadth (LLM, WASM, PG, attention) | Narrow | Very broad | **RuVector** |
| Maturity / churn | Young, high churn, small team | Broad but also fast-moving | ~Even |

## Decision

Adopt the following improvements, in priority order. Each is a HelixDB-inspired
sharpening of an area where RuVector is currently weaker, scoped so it **augments**
RuVector's existing Cypher/hybrid stack rather than replacing it. None of these
require abandoning RuVector's breadth or self-learning thesis.

### P1 — Optional schema layer with compile-time / load-time validation

RuVector's graph is schemaless and its Cypher is interpreted with no
ahead-of-time type checking. HelixDB's biggest, most defensible win is catching
type errors before they hit production.

- Add an **optional** schema definition (`N::`/`E::`/`V::`-equivalent: node labels,
  typed edges with `From`/`To` constraints, vector types with dimension + metric,
  indexed properties) expressed in a `ruvector.schema` file and/or a builder API.
- Validate registered Cypher/hybrid queries **against the schema at load time**
  (and offline via a CLI lint command), surfacing label/property/edge-direction/
  dimension-mismatch errors before execution — extending the existing
  `cypher/semantic.rs` analyzer from best-effort to schema-aware.
- Schema remains **opt-in**: schemaless mode (today's behavior) stays the default
  so we keep Cypher's low-friction onboarding and don't break existing users.

### P2 — First-class typed graph↔vector binding and a unified search-then-traverse operator

Make "vectors are graph citizens" a first-class, type-checked relationship rather
than a runtime string + property name.

- Allow a vector index to be **bound to a node label/property by schema**, so the
  planner knows the linkage statically.
- Add a Cypher extension operator equivalent to HelixQL's `SearchV<...>(q, k)::In<...>`
  — `CALL vector.search(label, $q, k) YIELD node ... MATCH (node)-[:REL]->(...)` —
  that fuses ANN search and traversal in **one plan** through the existing
  `executor` pipeline, with the vector step pushed down before the traversal join.
- Keep hyperedges supported on the traversal side (a RuVector advantage HelixDB
  lacks).

### P3 — In-query inline embedding (`embed()` function)

HelixDB's `Embed(text)` removes a round-trip and a class of dimension-mismatch
bugs. RuVector already ships a bundled ONNX embedder (ADR-194/195) and a default
MiniLM path (ADR-210) — we have the pieces.

- Expose an `embed($text)` scalar usable in Cypher/hybrid `CREATE`/`SET`/`WHERE`
  and in search predicates, routed through the existing embedder abstraction with
  the model pinned by schema (so the stored vector's dimension/metric is validated
  against the bound vector type).
- Available in core, server, and the npm/WASM surface; honor the existing
  hash-fallback guardrails from ADR-194 (never silently fall back).

### P4 — Unified hybrid query: ANN + BM25 + graph traversal with RRF in one statement

RuVector already has sparse+dense RRF and Graph RAG, but as separate APIs. Match
HelixDB's single-query hybrid ergonomics.

- Add a declarative hybrid operator that runs **BM25 keyword + ANN vector +
  optional graph-traversal expansion** and fuses with **RRF**, exposed through one
  query/endpoint, reusing the existing fusion and Graph-RAG implementations.

### P5 — Reproducible graph+vector benchmark harness with published percentiles

HelixDB's credibility comes substantially from `graph-vector-bench` (fixed
hardware, fixed dataset, P50/P95/P99). RuVector's numbers are scattered. Notably,
HelixDB has **no public vector benchmark** — a head-to-head harness is an
opportunity for RuVector to lead exactly where HelixDB is silent.

- Build a harness under `benchmarks/` (or extend `crates/ruvector-bench`) covering
  PointGet / OneHop / OneHopFilter (HelixDB-comparable) **plus** vector recall@k
  and hybrid retrieval, against Neo4j, pgvector, Qdrant, **and HelixDB itself**,
  reporting P50/P95/P99 + throughput on pinned hardware with a reproducible script.

### P6 — Schema-driven codegen for typed client SDKs (DX)

HelixDB's `schema → Rust endpoints + typed SDKs` is a real DX advantage. RuVector's
bindings are hand-written.

- From the P1 schema, generate **typed TypeScript/Python/Rust client stubs** (and
  optionally pre-registered query endpoints) so callers get compile-time-checked
  node/edge/vector types. Reuse the schema as the single source of truth.

### P7 — Investigate a decoupled object-storage tier (research spike, not a commitment)

HelixDB v3's compute/storage split (LSM over object storage) targets elastic
cloud scale — a gap relative to RuVector's embedded/`.rvf`/DiskANN model.

- Run a **research spike** evaluating an object-storage-backed tier (e.g. an
  LSM/segment layer over S3-compatible storage) layered beneath the existing
  storage abstraction, **without** disturbing the embedded redb/`.rvf`/WASM path.
  Decide go/no-go in a follow-up ADR; do not commit to a rewrite here.

### Explicitly NOT adopted

- **Replacing Cypher with a bespoke compiled DSL.** Cypher familiarity and the
  existing engine are assets; we add schema/typing/codegen *around* it rather than
  forcing a HelixQL-style lock-in language and AOT-to-Rust compilation. (We may
  later add a typed query builder, but not a new surface language as a hard
  dependency.)
- **Swapping redb for LMDB/heed.** redb is pure-Rust and WASM/`.rvf`-friendly,
  which matters for RuVector's browser/single-file thesis; LMDB's mmap model is
  hostile to WASM. P7 covers the genuinely missing capability (decoupled cloud
  storage) without giving up the embedded engine.
- **Marketing-grade "1000x" claims.** We commit to publishing only reproducible,
  percentile-backed numbers (P5).

## Consequences

**Positive**
- Closes RuVector's three real gaps vs HelixDB: ahead-of-time type safety,
  first-class typed graph↔vector binding, and single-query hybrid ergonomics.
- P5 lets RuVector compete on exactly the axis (vector benchmarks) where HelixDB
  has published nothing, and on the graph axis on equal, reproducible footing.
- All changes are additive and opt-in; schemaless Cypher, `.rvf`, WASM, the PG
  extension, hyperedges, and the self-learning stack are untouched.

**Negative / risks**
- A schema layer plus codegen is non-trivial surface area and must not regress the
  zero-config onboarding that schemaless Cypher provides — hence opt-in.
- In-query `embed()` couples query execution to embedder availability; the ADR-194
  no-silent-fallback discipline must be enforced.
- P7 (object storage) is the highest-effort, lowest-certainty item and is
  deliberately gated as a spike.
- Some "facts" about HelixDB are unverified (exact HNSW params, the precise license
  of any version we'd benchmark against, embedded-vs-Cloud engine). Before P5
  publishes head-to-head numbers, confirm: (1) HNSW M/ef/metric by reading source,
  (2) the license of the pinned HelixDB version, (3) which HelixDB engine (LMDB
  embedded vs v3 LSM/Cloud) is under test.

**Neutral**
- Each P-item ships under its own follow-up ADR with its own design and tests; this
  ADR only authorizes the direction and priority.

## References

- HelixDB: [github.com/HelixDB/helix-db](https://github.com/HelixDB/helix-db) ·
  [docs.helix-db.com](https://docs.helix-db.com/database/introduction) ·
  [graph benchmarks v1](https://docs.helix-db.com/benchmarks/v1) ·
  [graph-vector-bench](https://github.com/helixdb/graph-vector-bench) ·
  [GraphRAG/HelixQL blog](https://www.helix-db.com/blog/building-a-graphrag-system-for-professor-recommendations-with-helixdb) ·
  [DeepWiki](https://deepwiki.com/HelixDB/helix-db) ·
  [YC launch](https://www.ycombinator.com/launches/Naz-helixdb-the-database-for-rag-ai)
- RuVector internals: `crates/ruvector-core/src/storage.rs`,
  `crates/ruvector-graph/src/cypher/`, `crates/ruvector-graph/src/hybrid/`,
  `crates/ruvector-graph/src/executor/`, `crates/ruvector-server/`,
  `crates/ruvector-postgres/`, `crates/ruvector-diskann/`, `crates/ruvector-rabitq/`
- Related ADRs: ADR-001 (core architecture), ADR-193 (RAIRS/IVF), ADR-194/195
  (ONNX embedder API + unification), ADR-210 (default-on MiniLM embeddings)
</content>
</invoke>
