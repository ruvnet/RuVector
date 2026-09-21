# ADR-001: @ruvector/kge — architecture

## Status
Proposed

## Date
2026-09-21

## Context

`ruvector-graph` stores a property/hypergraph with Cypher, ACID transactions,
and an `EdgeTypeIndex` (`crates/ruvector-graph/src/index.rs:180`,
`graph.rs:17`), but it has **no learned relation vectors**: its Graph-RAG path
is a plain adjacency-list `KnowledgeGraph` with BFS retrieval (ADR-128 §7). No
crate in the repo scores a triple `(head, relation, tail)` for plausibility, so
link prediction, relation similarity, and 2-hop composition queries are not
answerable today. This is genuinely new territory — but every heavy supporting
piece already exists: `ruvector-router-core`'s HNSW for candidate retrieval
(with `DistanceMetric::DotProduct`, `types.rs:14`), `ruvector-gnn`'s EWC and
optimizers for training/continual updates, and `rvf-manifest`'s lineage field
for provenance.

The design brief is the HolE (Holographic Embeddings) family. HolE (Nickel,
Rosasco, Poggio, AAAI 2016) scores `r · (e_s ⋆ e_o)` — a relation vector dotted
with the **circular correlation** of the head and tail entity vectors — and
Hayashi & Shimbo (ACL 2017, arXiv:1702.05563) proved HolE is algebraically
equivalent to ComplEx (Trouillon 2016). That equivalence is the load-bearing
fact: it lets us ship the real-valued circular-correlation form (half the
parameters of ComplEx's doubled real/imaginary table) while inheriting
ComplEx's accuracy *and* its inner-product structure, which is what makes ANN
candidate retrieval possible.

## Decision

A **new top-level crate `ruvector-kge`**, not folded into `ruvector-graph` or
`ruvector-gnn`, with a Rust core exposed two ways, mirroring the router:

```
  TypeScript API   @ruvector/kge: predict / similar_relation / compose · train · eval · bench
  (index.ts)       ─────────────┬───────────────────────────┬─────────────────
                     native (napi-rs)                   wasm (wasm-bindgen)
  Rust core        ruvector-kge-ffi                    ruvector-kge-wasm
                     ────────────┴─── ruvector-kge (one crate, both targets) ───┘
                       scorer trait · entity/relation tables · train · eval · N3
                       ANN retrieval → ruvector-router-core HNSW (DotProduct)
                       continual update → ruvector-gnn::ElasticWeightConsolidation
```

1. **`ruvector-kge` owns everything that scores.** Two tables: an entity
   embedding matrix `E ∈ ℝ^{|entities|×d}` (`d` even) and a relation matrix
   `R ∈ ℝ^{|relations|×d}`. A `Scorer` trait (ADR-002) abstracts the scoring
   function; the v1 default is real-valued circular correlation (literal HolE,
   ComplEx-equivalent), with RotatE as an opt-in second scorer for graphs where
   relation **composition** matters — the one pattern HolE/ComplEx cannot
   represent (Sun et al. 2019, arXiv:1902.10197).

2. **`ruvector-graph` calls `ruvector-kge`, never the reverse.** The Cypher
   engine is the natural home for the query surface, so `ruvector-graph` takes
   `ruvector-kge` as a dependency and extends its existing vector-Cypher layer
   (`crates/ruvector-graph/src/hybrid/cypher_extensions.rs`, which already ships
   `VectorCypherParser`/`VectorCypherExecutor`/`SimilarityPredicate`) with three
   operators: `PREDICT (a)-[:r]->(?e)` (score-rank candidate tails),
   `SIMILAR_RELATION(r1, r2)` (cosine of relation vectors), and `COMPOSE(r1,
   r2)` (circular correlation of two relation vectors as a synthetic 2-hop
   relation). This reuses an in-repo precedent for query-language vector
   operators rather than inventing one.

3. **The HNSW index over entities makes trilinear scoring an inner product.**
   Circular correlation `[e_s ⋆ e_o]_k = Σ_i (e_s)_i (e_o)_{(i+k) mod d}` has the
   DFT identity `F(e_s ⋆ e_o) = conj(F(e_s)) ⊙ F(e_o)`. Writing `ŝ=F(e_s)`,
   `ô=F(e_o)`, `r̂=F(r)`, Parseval gives `score = r·(e_s ⋆ e_o) =
   (1/d)·Re⟨ r̂ ⊙ ŝ, ô ⟩`. Because `Re⟨x,y⟩ = Re(x)·Re(y) + Im(x)·Im(y)`, the
   score is a **plain real dot product** between the query vector
   `q = [Re(r̂⊙ŝ); Im(r̂⊙ŝ)]` and the tail's Fourier feature
   `t = [Re(ô); Im(ô)]`. Head queries `(?, r, o)` use `q = [Re(conj(r̂)⊙ô); …]`
   symmetrically. So candidate tails for `(h, r, ?)` are retrieved by
   `DistanceMetric::DotProduct` HNSW over a **Fourier-domain copy** of the entity
   table, then exact-reranked by the true score. The canonical trained table
   stays `d` reals per entity (the "half the parameters" claim); the ANN index
   holds `~d+2` reals per entity (Hermitian symmetry of a real FFT). The
   ANN-compatibility claim is *derived* from the equivalence, not cited for HolE
   directly — ADR-006 gates it on measured recall@k versus exhaustive scoring.

4. **"Holographic" here means one concrete thing:** HolE's real-valued circular
   correlation over dense float embeddings (Nickel 2016's
   holographic-reduced-representation / associative-memory binding). It is **not**
   the binary XOR-bind / majority-vote module in
   `crates/ruvector-nervous-system/src/hdc/ops.rs`, which is a different
   mathematical object (10,000-bit hypervectors) and is not a HolE substitute.

5. **npm shape copies `@ruvector/router` exactly** — `@ruvector/kge` with five
   platform packages under `optionalDependencies` (`kge-linux-x64-gnu`,
   `kge-linux-arm64-gnu`, `kge-darwin-x64`, `kge-darwin-arm64`,
   `kge-win32-x64-msvc`) plus a `-wasm` fallback. The `optional-deps-resolvable-
   on-npm` job (`.github/workflows/regression-guard.yml:351`) enumerates every
   `package.json` under `npm/packages` and fails if a listed version is
   unpublished, so **the first PR ships zero platform `optionalDependencies`**;
   they are added in a later bump PR once the platform packages exist on npm,
   the sequence the router's 0.1.31 revert documented (typesafe ADR-002 §5).

6. **ADR numbering is package-local.** `scripts/adr-index.mjs` scans only
   `docs/adr` (`ADR_DIR`, adr-index.mjs:46) and the duplicate-number gate runs
   as `adr-index.mjs --check` in `.github/workflows/ci.yml:53`. These records
   under `npm/packages/kge/docs/adr/` are outside that scan, so ADR-001..006
   here do not collide with the root ADR sequence.

### Package layout

```
npm/packages/kge/
  package.json           @ruvector/kge, optionalDependencies → 5 platform packages (added later)
  src/index.ts           predict / similar_relation / compose, train/eval, CLI
  wasm/                  wasm-bindgen output (bundler + nodejs)
  docs/adr/              this record set
  bench/                 datasets, harness, receipts (ADR-006)
crates/ruvector-kge/      Rust: scorer trait, tables, training, eval, N3, ANN glue
crates/ruvector-kge-ffi/  napi-rs bindings (mirrors ruvector-router-ffi)
crates/ruvector-kge-wasm/ wasm-bindgen bindings (mirrors ruvector-router-wasm)
```

## Consequences

- Link prediction, relation similarity, and 2-hop composition become Cypher
  operators over a trained table, at CPU cost and no egress.
- Reusing `ruvector-router-core`'s HNSW inherits its single-layer, single-entry
  graph (typesafe ADR-001) — adequate for the 14k–123k-entity benchmark graphs;
  a multi-layer index is a v2 concern only if a real graph outgrows it.
- Two scorers behind one trait means the RotatE path must stay tested to the
  same evaluation protocol as HolE, not treated as a second-class extra.
- The ANN pipeline is an approximation of exhaustive scoring; its recall is a
  benchmark gate (ADR-006), never an assumed property.

## Alternatives considered

- **Fold KGE into `ruvector-gnn` as another training mode.** Rejected for v1:
  conflates node-classification GNN metrics with filtered-MRR/Hits@k, and
  complicates the crate's 500-line-file discipline. Revisit only if an
  NBFNet-style inductive successor needs message-passing *and* KGE scoring fused.
- **HDC/binary scorer as the default** (reusing `hdc/ops.rs` XOR-bind). Rejected
  for v1: no literature ties binary HDC binding to filtered-MRR-competitive link
  prediction; a speculative v2 "ultra-compressed" variant behind the same trait.
- **Skip embedding-table KGE, wait for ULTRA/NBFNet** (Galkin et al., ICLR 2024,
  arXiv:2310.04562). Rejected for v1: different infrastructure (GNN
  message-passing, pretrain-then-transfer), does not produce the compositional
  Cypher surface. Tracked as the v2 inductive tier.

## Evidence

- HolE: Nickel, Rosasco, Poggio, AAAI 2016. HolE≡ComplEx: Hayashi & Shimbo, ACL
  2017 (arXiv:1702.05563), corroborated arXiv:1707.01475. RotatE composition:
  Sun et al., ICLR 2019 (arXiv:1902.10197). Inductive tier: ULTRA
  (arXiv:2310.04562), NBFNet (NeurIPS 2021).
- Repo reuse: `crates/ruvector-graph/src/{graph.rs:17,index.rs:180}` and
  `hybrid/cypher_extensions.rs`; `crates/ruvector-router-core/src/{types.rs:14,
  distance.rs:18}`; `crates/ruvector-gnn/src/ewc.rs`;
  `crates/rvf/rvf-manifest/src/level0.rs`;
  `crates/ruvector-nervous-system/src/hdc/ops.rs`.
- Packaging/guards: `@ruvector/router` `optionalDependencies`;
  `.github/workflows/regression-guard.yml:351`; `scripts/adr-index.mjs:46`,
  `.github/workflows/ci.yml:53`. Research brief: `scratchpad/hole-research-report.md`.
