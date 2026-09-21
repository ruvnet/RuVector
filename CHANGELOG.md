# Changelog

All notable changes to RuVector will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.0.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [@ruvector/router 0.1.31] - 2026-09-21
### Fixed
- Ship the issue #430 HNSW fixes (result-heap eviction order, insert beam width,
  distance-based pruning, `k > efSearch`) that landed on `main` in May but never
  reached npm: published 0.1.30 scored recall@10 = 3.6% on 5,000 real MiniLM
  embeddings, returning neighbours from the last-inserted region regardless of
  the query.
### Changed
- Neighbour selection now uses the HNSW paper's diversity heuristic (Algorithm 4,
  with kept-pruned fill) for both new-node edges and overflow pruning, instead of
  "the m closest". On clustered data the closest-m rule left almost no
  inter-cluster edges, so single-entry-point search stalled inside a cluster
  (recall@10 0.89 at defaults with clusters inserted contiguously). Measured on
  5,000 real embeddings: recall@10 1.000 in every configuration tried (from
  0.96–0.99 at defaults and 0.88–0.92 at m=16), with lower query latency
  (p95 0.58 ms vs 0.85 ms at defaults); inserts are ~25% slower.
- New regression test pins recall@10 ≥ 0.90 against brute force on clustered
  data in both contiguous and shuffled insertion order, and fails if results
  are drawn from the insertion tail — the failure mode uniform-random-vector
  tests cannot see.

## [ruvector npm 0.3.2] - 2026-09-21
### Fixed
- `DbOptions` documented `hnsw: { m, efConstruction, efSearch }` and `path`,
  but the `VectorDB` wrapper only read `hnswConfig` and `storagePath`, so a
  caller following the published types had their HNSW settings silently
  dropped and got the defaults. Both spellings are honoured; a partial `hnsw`
  merges over the defaults; the canonical name wins if both are passed. The
  resolution is a pure exported `resolveVectorDbOptions()` with a workflow test.
- `DbOptions` now documents the canonical native names (`dimensions`,
  `hnswConfig`, `storagePath`) with the old spellings marked as aliases; the
  never-implemented `autoPersist` field is gone.
- `ruvector route info` printed `v0.1.28` for every installed router version
  (a hardcoded fallback); it now reads the installed package's version.

## [ruvector npm 0.2.37] - 2026-07-27

### Fixed

- Restore the release CI's native backend installation so functional,
  learning, performance, and recall gates exercise HNSW instead of the
  in-memory fallback.
- Preserve the Rust core's default HNSW behavior when JavaScript callers omit
  `hnswConfig`; the wrapper previously selected the exact FlatIndex path.
- Normalize the package binary metadata and its integration assertion.
- Remove the npm workspace's critical `tar` advisory with a patched override.
- Correct the Shor factoring milestone table to distinguish a failed
  35-factor experiment from the separate Schnorr/QAOA hybrid result.

## [ruvector npm 0.2.36] - 2026-07-26

### Fixed

- Start the stdio MCP transport when launched through `ruvector mcp start`
  ([#715](https://github.com/ruvnet/ruvector/issues/715)).
- Keep ONNX model-loader diagnostics on stderr so they cannot corrupt MCP
  JSON-RPC frames ([#710](https://github.com/ruvnet/ruvector/issues/710)).

### Security

- Update `anyhow` to 1.0.104 and `memmap2` to 0.9.11, removing the
  corresponding RustSec soundness exceptions.
- Update the MCP SDK to 1.29.x, clearing its shared-transport data-leak and
  ReDoS advisories, and remove the unused `js-beautify` dependency.

## [hailo-backend] - 2026-05-03

Branch-only entry; not yet released to a versioned tag. 38 iters /
45 commits arc covering iter 133-171 to land NPU-accelerated
embedding inference on Pi 5 + AI HAT+ end-to-end.

### Added

- **NPU acceleration via Hailo-8** (ADR-176, iters 158-163). New
  modules `hef_pipeline.rs`, `host_embeddings.rs`, `hef_embedder.rs`
  in `ruvector-hailo`. `HailoEmbedder` auto-detects `model.hef` +
  safetensors trio and routes through the NPU; falls back to
  candle-on-CPU when HEF is absent.
- **First-known all-MiniLM-L6-v2 HEF for Hailo-8** — published as
  GitHub Release `hailo-encoder-v0.1.0-iter156b` (15.7 MB, sha256
  `cdbc89...`). Compiled iter 156b after working around four
  distinct Hailo Dataflow Compiler v3.33 SDK bugs from user-space
  (KeyError, AccelerasValueError, ElementwiseAddDirectOp Keras
  serialize via `acceleras` Layer monkey-patch, RGB-align via
  single-input encoder form).
- **`deploy/download-encoder-hef.sh`** — sha256-pinned downloader
  for the HEF artifact, matches the iter-134
  `download-cpu-fallback-model.sh` pattern.
- **`deploy/compile-encoder-hef.py`** — Python SDK driver for the
  HEF compile pipeline, replaces the iter-131 CLI invocations that
  hit the `hailo` CLI's `-y` auto-accept-recommendation bug.
- **`deploy/export-minilm-encoder-onnx.py`** — torch.onnx.export
  helper that strips the BERT embedding lookup so the encoder
  block compiles cleanly on Hailo-8.
- **iter-167 ranking-aware startup self-test** — worker now
  embeds three reference phrases at boot and checks
  `sim(close) > sim(far)`; refuses to serve if the encoder is
  producing nonsense vectors.
- **Pi 4 / Pi 5-without-HAT deploy** as a first-class target
  (ADR-177). cpu-fallback is fully hardware-agnostic across
  aarch64.
- **iter-147 cpu-fallback embedder pool** —
  `RUVECTOR_CPU_FALLBACK_POOL_SIZE=N` runs N parallel BertModel
  instances behind try-lock dispatch. Measured 1.75× throughput
  on x86 release, 4× on Pi 5 expected.
- **iter-143 fingerprint integrity** — `compute_fingerprint` now
  hashes the safetensors+tokenizer+config trio so cpu-fallback
  workers get integrity-checked by the cluster.
- **iter-141 cross-build with `--with-worker`** — produces an
  aarch64 cpu-fallback worker binary in one command on x86.

### Performance

Measured on cognitum-v0 (Pi 5 + AI HAT+) at concurrency=4 via
cluster-bench:

| Path | Throughput | p50 latency | Δ vs cpu-fallback |
|---|---:|---:|---:|
| cpu-fallback (Pi 5) | 7.0 / sec | 572 ms | — |
| NPU HEF (iter 163) | **67.3 / sec** | **57 ms** | **9.6×** |
| cache hit (in-process) | **15.86 M / sec** | <1 µs | **226,000×** |

Saturation test (iter 170): 60s burst at C=100 — Pi never OOMs,
worker RSS stable at 91 MB, tonic-level backpressure correctly
drops excess requests with `ResourceExhausted`.

### Documentation

- **ADR-167** updated: NPU is now production-default
- **ADR-173** updated: ruvllm-hailo upstream embedding seam is
  NPU-accelerated transparently
- **ADR-175** Option A status flipped to "production default"
- **ADR-176** new EPIC tracking the full integration arc
- **ADR-177** new — Pi 4 / no-HAT deploy
- Cluster README now has an Operator QUICKSTART covering all
  three deploy paths

### Internal

- iter-153 `acceleras` Keras-registration monkey-patch — the
  breakthrough that unblocked the entire HEF compile pipeline.
  `compile-encoder-hef.py` walks every module under
  `hailo_model_optimization.acceleras` at import time and applies
  `keras.saving.register_keras_serializable()` to every
  `keras.layers.Layer` subclass it finds.

## [2.0.5] - 2026-02-26

### Fixed
- **ruvector-gnn**: Replace `assert!()` with `Result` in `MultiHeadAttention::new()` and `RuvectorLayer::new()` — prevents fatal `abort()` in NAPI-RS/WASM bindings ([#216](https://github.com/ruvnet/ruvector/issues/216))
- **ruvector-gnn**: Fix pre-existing `mmap.rs` test compilation error (`grad_offset` returns `Option<usize>`)
- **install.sh**: Remove stale hardcoded version pins (`@0.1.2`, `@0.1.23`), always fetch latest
- **install.sh**: Fix operator precedence bug in CLI install guard (`--npm-only` now correctly skips CLI)
- Docs: Fix stale capability counts in root README
- Docs: Update guides to match current API surface and versions

### Added
- OpenFang Agent OS RVF example — 24 RVF capabilities demonstrated
- OpenFang project research document
- Missing capabilities added to advanced features guide

### Security
- **SEC-001**: Harden mmap pointer arithmetic with checked bounds
- **SEC-002**: Cryptographic hash binding for proof attestations (prevents spoofing)

### Changed
- Workspace version bumped from 2.0.4 to 2.0.5
- `@ruvector/gnn` bumped from 0.1.24 to 0.1.25 (all 7 platform packages)
- All WASM/NAPI wrappers (`ruvector-gnn-wasm`, `ruvector-gnn-node`, `ruvector-attention-unified-wasm`) now propagate layer construction errors as catchable JS exceptions instead of process crashes

### Published
- `ruvector-core@2.0.5` → crates.io
- `ruvector-gnn@2.0.5` → crates.io
- `@ruvector/gnn@0.1.25` → npm (linux-x64-gnu, linux-x64-musl, linux-arm64-gnu, linux-arm64-musl, darwin-x64, darwin-arm64, win32-x64-msvc)

## [2.0.4] - 2026-02-25

### Added
- **ADR-043: External Intelligence Providers** for SONA learning — pluggable external AI intelligence integration
- **Intelligence module** in `@ruvector/ruvllm@2.5.0`
- **Security Hardened RVF v3.0** — 30 verified capabilities, AIDefence + TEE hardened container (ADR-042)
- **Proof-gated graph transformer** with 8 verified modules ([#212](https://github.com/ruvnet/ruvector/pull/212))
- **Formal verification** with lean-agentic dependent types ([#206](https://github.com/ruvnet/ruvector/pull/206))
- **WASM cognitive stack** — canonical min-cut, spectral coherence, container orchestration, cold-tier GNN training ([#201](https://github.com/ruvnet/ruvector/pull/201))
- **rvDNA health biomarker analysis engine**:
  - 20-SNP panel with streaming simulation
  - LPA cardiovascular SNPs from SOTA meta-analysis
  - CUSUM changepoint detection, gene-biomarker correlations
  - SNP weights calibrated from clinical meta-analyses
  - npm `@ruvector/rvdna` package with risk scoring and benchmarks
- SPARQL parser backtrack fix and executor memory leak fix in `ruvector-postgres@2.0.4`

### Security
- **Harden intelligence providers** — type-safe enums, input validation, file size limits
- **Fix path traversal** in MCP server `vector_db_backup` (CWE-22) ([#211](https://github.com/ruvnet/ruvector/pull/211))
- **Harden MCP servers** against command injection, CORS bypass, and prototype pollution ([#213](https://github.com/ruvnet/ruvector/pull/213))

### Fixed
- Migrate attention/dag/tiny-dancer to workspace versioning
- Fix all dependency version specs for crates.io publishing
- Include prebuilt binaries in `@ruvector/gnn` platform packages ([#195](https://github.com/ruvnet/ruvector/issues/195))
- CI: Node.js upgraded to 20 in GNN build workflow
- CI: Auto-publish on push to main for GNN packages
- RVF `NodeBackend` string ID ↔ numeric label mapping

## [0.3.0] - 2026-02-21

Major release introducing the RuVector Format (RVF) cognitive container, AGI runtime substrate, and a significant expansion of the platform from vector database to cognitive computing framework.

### Added

#### RuVector Format (RVF) — Universal Cognitive Container
- Complete RVF SDK with cognitive container specification ([#166](https://github.com/ruvnet/ruvector/pull/166))
- New crates: `rvf-types`, `rvf-crypto`, `rvf-runtime`, `rvf-node`, `rvf-wasm`, `rvf-solver`, `rvf-solver-wasm`, `rvf-cli`
- WASM segment (`WASM_SEG 0x10`) for self-bootstrapping RVF files
- Ed25519 asymmetric signing (RFC 8032) behind feature gate
- Witness auto-append, CLI verification, prebuilt fallbacks
- Integration into `npx ruvector` and `rvlite` (ADR-032)
- Platform-specific scripts for Linux, Windows, Node, browser, Docker
- Real Linux 6.8.12 kernel embedded in RVF for live-boot proof

#### AGI Cognitive Container (ADR-036)
- `authority_config` and `domain_profile` TLV support
- Authority guard, coherence monitor, benchmarks
- Multi-dimensional IQ with cost/robustness/AGI contract
- 5-level superintelligence pathway engine
- KnowledgeCompiler Strategy Zero, StrategyRouter bandit, ablation protocol
- Three-class memory, loop gating, RVF artifacts, rollback witnesses
- Thompson Sampling two-signal model, speculative dual-path, constraint propagation

#### QR Cognitive Seed (ADR-034)
- Pure-Rust QR code encoder for RVF seed bytes
- In-browser RVF seed decoder PWA
- Swift App Clip skeleton for iOS mobile FFI

#### Progressive Indexing Hardening (ADR-033)
- `QualityEnvelope`, triple budget caps, selective scan, fuzz benchmark
- `ResultQuality` extended to API boundary
- Malicious manifest test and brute-force cap

#### Sublinear-Time Sparse Solver
- Complete `ruvector-solver` crate with zero-overhead SpMV
- Fused Neumann iteration kernel
- WASM solver: self-learning AGI engine compiled to WASM
- Min-cut gating experiment modules

#### Additional Systems
- **RvBot**: Self-contained RVF bot with real Linux 6.6 kernel and initramfs boot
- **rvDNA Genomics**: Complete SOTA genomic analysis pipeline, native 23andMe genotyping v0.2.0
- **Domain Expansion**: Cross-domain AGI transfer learning engine with WASM bindings and meta-learning
- **OSPipe**: RuVector-enhanced personal AI memory for Screenpipe ([#163](https://github.com/ruvnet/ruvector/pull/163))
- **Quantum Simulation**: `ruqu-core`, `ruqu-algorithms`, `ruqu-wasm`, Bell test CHSH inequality
- **Causal Atlas** (ADR-040): Dashboard, solver, and desktop app
- **ruvector-postgres v0.3.0**: 43 new SQL functions (ADR-044)

### Fixed
- HNSW index bugs, agent/SPARQL crashes ([#152](https://github.com/ruvnet/ruvector/issues/152), [#164](https://github.com/ruvnet/ruvector/issues/164), [#167](https://github.com/ruvnet/ruvector/issues/167), [#171](https://github.com/ruvnet/ruvector/issues/171))
- LRU security fix ([#148](https://github.com/ruvnet/ruvector/issues/148))
- FPGA-transformer `BackendSpec.as_ref` and HNSW array indexing
- Platform-specific errno on macOS/BSD ([#174](https://github.com/ruvnet/ruvector/issues/174))
- WASM path resolution in CJS→ESM interop
- Docker Rust version bumped to 1.85 for edition2024

### Changed
- `rvf-types`, `rvf-crypto`, `rvf-runtime` bumped to 0.2.0
- npm: `ruvector@0.1.99`, `rvlite@0.2.4`, `rvf@0.1.3`

## [0.2.6] - 2025-12-09

### Added
- **`ruvector-postgres` PostgreSQL extension** with SIMD optimizations and 53 SQL function definitions
- **PostgreSQL 18 support** with backward compatibility
- **`@ruvector/postgres-cli`** with native installation support
- **W3C SPARQL 1.1 query language** support in PostgreSQL extension
- **GNN v2** comprehensive implementation with cognitive substrate
- **iOS-optimized WASM recommendation engine**
- **9 cognitive substrate crates** published as EXO-AI 2025
- **Neuromorphic HNSW v2.3** with SNN (Spiking Neural Network) integration
- **Ultra-low-latency meta-simulation engine** example
- **8 specialized Docker images** with publishing infrastructure
- **RuVector Studio** — complete web UI application
- `ruvector-attention` functions exported from PostgreSQL extension

### Fixed
- Docker build and extension SQL for PG17
- SPARQL build compilation — achieved 100% clean build
- Docker Hub README and image references

### Changed
- npm packages reorganized from `/src` to `/npm/packages`

### Breaking Changes
- npm import paths changed due to `/src` → `/npm/packages` reorganization

## [0.1.32] - 2026-01-17

### Added
- **SONA Neural Architecture** npm package (`sona@0.1.5`)
- **RuvLLM** npm package with intelligence module
- **Graph Node** bindings (`@ruvector/graph-node@0.1.26`)
- npm package expansion and version consolidation

## [0.1.19] - 2025-12-01

### Fixed
- **GNN Node.js bindings**: Use `Float32Array` for NAPI bindings to fix type conversion errors

## [0.1.16] - 2025-11-27

### Added
- **Persistent GNN layer caching** — 250-500x performance improvement
- **Self-learning GNN strategy** for accuracy improvement
- GNN NAPI-RS bindings for all platforms

## [0.1.0] - 2025-11-25

Initial release of RuVector — a high-performance vector database written in Rust.

### Added

#### Core Vector Database
- HNSW (Hierarchical Navigable Small World) graph indexing
- SIMD-optimized distance metrics (Euclidean, Cosine, Dot Product, Manhattan)
- Memory-mapped vector access via memmap2
- Parallel index construction using rayon
- Zero-copy serialization with rkyv
- Scalar quantization (int8) for 4x memory compression

#### AgenticDB Compatibility Layer
- Full 5-table schema: `vectors_table`, `reflexion_episodes`, `skills_library`, `causal_edges`, `learning_sessions`
- Reflexion Memory API with semantic search over self-critique episodes
- Skill Library with auto-consolidation and usage tracking
- Causal Memory Graph with confidence scoring and hypergraph support
- 9 RL algorithms (Q-Learning, SARSA, DQN, PPO, Actor-Critic, Policy Gradient, Decision Transformer, MCTS, Model-Based)

#### Advanced Search
- Product Quantization (PQ) with 8-16x memory compression at 90-95% recall
- Filtered search (pre/post-filtering with complex expressions)
- Hybrid search (vector similarity + BM25 keyword scoring)
- MMR (Maximal Marginal Relevance) diversity-aware ranking
- Conformal prediction with distribution-free confidence intervals

#### Multi-Platform Deployment
- **Node.js** (NAPI-RS): Async API, TypeScript types, zero-copy Float32Array
- **WASM**: Browser-compatible, Web Workers, IndexedDB persistence
- **CLI**: JSON/CSV/NPY support, shell completions, benchmarking
- **Cross-platform builds**: Linux (x64/arm64), macOS (x64/arm64), Windows (x64), WASM

#### Performance
- 10-100x faster than Python/TypeScript implementations
- Sub-millisecond latency (p50 < 0.8ms for 1M vectors)
- 95%+ recall with HNSW (ef_search=100)
- 4-32x memory compression with quantization
- 200-300x distance calculation speedup with SIMD
- Near-linear scaling to CPU core count

### Dependencies
- **Core**: redb, memmap2, hnsw_rs, simsimd, rayon, crossbeam
- **Serialization**: rkyv, bincode, serde, serde_json
- **Node.js**: napi, napi-derive
- **WASM**: wasm-bindgen, wasm-bindgen-futures, js-sys, web-sys
- **Math**: ndarray, rand, rand_distr
- **CLI**: clap, indicatif, console

---

For questions or issues, visit: https://github.com/ruvnet/ruvector/issues
