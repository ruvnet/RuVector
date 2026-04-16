# ADR-151: EML Selected-Dimension HNSW — Acceptance Scope and Integration Constraints

## Status

Proposed (v2 merged into `feat/eml-hnsw-optimizations-v2` as PR #356; v3 SOTA push in progress — see §Open Tiers)

## Date

2026-04-16

## Context

Two community PRs propose applying ideas from Odrzywołek 2026 ("All elementary functions from a single operator", arXiv:2603.21852v2) to RuVector's HNSW index:

- **PR #352 (shaal)** — `QuantizationConfig::Log`, `HnswIndex::new_unified()`, branch-free SIMD distance kernel, optional per-call zero-padding. Four-stage proof chain on real ANN datasets (SIFT1M +14.0% QPS, GloVe-100d −10.4% QPS, honest self-disproof of the padding hypothesis). Opt-in, defaults unchanged.
- **PR #353 (aepod)** — `ruvector-eml-hnsw` crate with six "learned optimizations": cosine decomposition (dim selection), progressive dimensionality, adaptive ef, search path prediction, rebuild prediction, PQ correction. Claim: "10-30× distance, 2-5× search". Stage-3 real-data validation deferred.

Empirical validation on `ruvultra` (AMD Ryzen 9 9950X / 32T / 123 GB) uncovered that PR #353's shipped crate has **no downstream consumer** — nothing in `ruvector-core` or `ruvector-graph` depends on `ruvector-eml-hnsw`, and the `eml` feature on the vendored `hnsw_rs` fork is never enabled. The crate compiles, its unit tests pass, but the contribution produces zero runtime effect on any RuVector HNSW path. PR #353's own Stage 1 disproves the per-call EML `fast_distance` wrapper (2.1× slower than baseline), and the author's follow-up comments pivot to "use plain cosine on selected dims only" — a pattern that is described but never shipped as callable code.

We validated the contribution end-to-end, including downloading real SIFT1M and running a 6-agent optimization swarm to characterize which parts actually help, under what conditions. This ADR records the scope in which the EML contribution is accepted, the constraints under which it must be shipped, and the experiments that remain open.

## Decision

1. **Accept** the selected-dimension cosine approach as a **candidate pre-filter stage** of a retrieval pipeline, paired with exact re-rank. Ship it as a thin opt-in wrapper (`EmlHnsw`) — defaults to `HnswIndex` remain unchanged.
2. **Reject** the per-call EML tree distance (`fast_distance`) on evidence: 2.35× slower than scalar baseline on ruvultra.
3. **Reject** `AdaptiveEfModel` in its current form: 290 ns/query overhead vs claimed ~3 ns is too large to amortize against typical ef-search budgets.
4. **Couple** the rerank stage with the SIMD kernel from PR #352 (`UnifiedDistanceParams::cosine()` → SimSIMD) so rerank throughput scales with `fetch_k`.
5. **Require** `fetch_k ≥ 500` at `selected_k ∈ [32, 48]` for any workload that claims recall@10 ≥ 0.85 on standard ANN benchmarks.
6. **Promote** the retention-objective selector (Tier 1C) to the default selector for any new `EmlHnsw` built after the v2 merge — it beats Pearson by +10.5 pp recall@10 on SIFT1M at statistical significance.
7. **Defer** to v3 SOTA push (currently running): PQ-native HNSW with codes-in-graph, rayon-parallel rerank, 1M-scale validation, `HnswIndex::new_with_selected_dims()` first-class integration, beam-search selector, corrector normalization fix.

### Accepted Architecture (v2, shipped in PR #356)

```
┌───────────────────────────────────────────────────────────────────┐
│  Application                                                       │
│    EmlHnsw::search_with_rerank(query, k=10, fetch_k=500, ef=64)   │
└───────────────┬───────────────────────────────────────────────────┘
                │
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 1 — reduced-dim HNSW (fast, approximate)                   │
│    * vectors projected to selected_dims (|D|=32..48)              │
│    * cosine over projection via hnsw_rs::Hnsw                     │
│    * returns fetch_k=500 candidates                               │
│    * measured: 130-200 µs p50 at SIFT1M 50k                       │
└───────────────┬───────────────────────────────────────────────────┘
                │ 500 candidate ids
                ▼
┌───────────────────────────────────────────────────────────────────┐
│  Stage 2 — exact re-rank (SIMD)                                   │
│    * full-dim cosine via simsimd::SpatialSimilarity (PR #352)     │
│    * reorder; truncate to k                                       │
│    * measured: ~140 µs p50 at fetch_k=500 on ruvultra (SIMD)      │
└───────────────┬───────────────────────────────────────────────────┘
                │ top-k by exact distance
                ▼
             results
```

### Runtime API (shipped in v2)

```rust
// Offline: teach the selector which dims discriminate on your data.
// Use retention objective (+10.5 pp vs Pearson) for new indexes.
let mut selector = EmlDistanceModel::new(128, 32);
selector.train_for_retention(&learn_corpus, &learn_queries, /*target_k*/ 10, /*pool*/ 100);

let idx = EmlHnsw::new(selector, EmlMetric::Cosine, max_elements, m, ef_construction)?;
idx.add_batch(&corpus);

// Online: approximate + exact rerank. fetch_k ≥ 500 required for recall@10 ≥ 0.85.
let hits = idx.search_with_rerank(&query, 10, /* fetch_k */ 500, /* ef_search */ 64);
```

Defaults in the constructor remain unchanged. `HnswIndex::new()` is untouched.

## Measured Evidence

All numbers on ruvultra (AMD Ryzen 9 9950X, 32T, 123 GB, Linux 6.17) with `cargo bench` / `cargo test --release`, 100 samples per micro-benchmark, 200-query SIFT1M configurations at 50k base unless stated.

### PR #353 claims vs reality

| Claim (PR #353 body) | Measured on ruvultra |
|---|---|
| "93 unit tests — all passing" | 60 unit + 3 doctests = 63 actual |
| `fast_distance` 3.0× faster at k=32 | **2.35× SLOWER** (70.5 µs vs 29.96 µs, 500 pairs) |
| Raw 16-d L2 proxy 9.3× faster | 10.4× faster (2.89 µs vs 29.96 µs) ✓ |
| Adaptive ef ~3 ns/query | 290 ns/query ✗ |
| Rebuild prediction 2.8 ns | 3.54 ns ✓ (within budget) |
| ρ ≈ 0.85–0.95 on SIFT | recall@10 reduced = **0.194**, +rerank(50) = **0.438** |

### Tier 1A — fetch_k × selected_k sweep at selected_k=32 (commit `a5806096`)

| fetch_k | recall@10 reduced | recall@10 rerank | reduced p50 (µs) | rerank p50 (µs) |
|---|---|---|---|---|
| 10   | 0.193 | 0.193 | 129 | 54 |
| 50   | 0.193 | 0.439 | 122 | 65 |
| 200  | 0.193 | 0.725 | 133 | 312 |
| **500**  | 0.193 | **0.857** | 137 | 779 |
| 1000 | 0.193 | 0.931 | 128 | 1412 |

| selected_k at fetch_k=1000 | recall@10 reduced | recall@10 rerank | reduced p50 (µs) |
|---|---|---|---|
| 8  | 0.020 | 0.391 | 56 |
| 16 | 0.074 | 0.731 | 131 |
| 32 | 0.196 | 0.933 | 162 |
| **48** | 0.306 | **0.974** | 191 |
| 64 | 0.436 | 0.986 | 244 |

Reduced recall is constant at 0.193 across fetch_k — selector is the bottleneck.

### Tier 1B — SimSIMD rerank kernel (commit `3ed71248`)

| dim | scalar | SimSIMD | speedup |
|---|---|---|---|
| 128 | 59.1 ns | 10.5 ns | **5.65×** |
| 384 | 177.2 ns | 28.5 ns | **6.22×** |

Recall preserved across the swap (Δ = 0.002, f32-vs-f64 accumulation noise).

### Tier 1C — retention-objective selector (commit `a453e4ea`)

| selector | recall@10 (selected_k=32, fetch_k=200) | selector train wall-clock |
|---|---|---|
| Pearson (PR #353) | 0.7120 | 1.06 s |
| retention (greedy forward) | **0.8170** | 39.7 s |

+10.5 pp is >3σ of the n=200 binomial SE (≈0.03). Training 37× slower, offline/one-shot. Reproduced on v2 merged branch: +11.0 pp (0.7140 vs 0.8240).

### Tier 2 — Sliced Wasserstein rerank (commit `9a79f948`) — **FALSIFIED**

| rerank kernel | recall@10 vs cos-GT | recall@10 vs euclidean-GT | p50 µs | p95 µs |
|---|---|---|---|---|
| cosine baseline | 0.7185 | 0.7876 | 404 | 543 |
| SW L=16 | 0.2810 | 0.3769 | 6 627 | 7 141 |
| SW L=50 | 0.3250 | 0.4398 | 20 558 | 22 038 |
| SW L=100 | 0.3380 | 0.4459 | 44 930 | 49 057 |

SW is 50.9× slower AND 38.1 pp worse than cosine on SIFT1M. Structurally wrong: SIFT is quantized gradient histograms where bin identity carries signal; SW sorts projected coordinates per slice and destroys that information.

### Tier 3A — ProgressiveEmlHnsw `[8, 32, 128]` cascade (commit `f81a43dc`)

| | baseline `EmlHnsw` k=32 | Progressive [8, 32, 128] |
|---|---|---|
| build | 12.5 s | 73.7 s (5.9×) |
| recall@10 | 0.196 | **0.984** |
| p50 search | 317 µs | 961 µs (3.0×) |
| p95 search | 425 µs | 1213 µs |

Pareto-dominates single-index `EmlHnsw k=48, fetch_k=1000` (0.974 at 1950 µs) → 2× latency at matched recall. Build cost is the caveat.

### Tier 3B — PqEmlHnsw 8×256 (commit `6a42d16d`)

| index | recall@10 | rerank@10 | p50 red (µs) | p50 rr (µs) | bytes/vec |
|---|---|---|---|---|---|
| EmlHnsw (float reduced-dim) | 0.1905 | 0.7235 | 432 | 342 | 512 |
| PqEmlHnsw (PQ codes) | 0.4125 | **0.9515** | 583 | 569 | **8** |

64× memory reduction in training-side storage. **Runtime caveat (surfaced in v2 integration test):** the current `PqEmlHnsw` keeps reconstructed floats in the underlying HNSW graph — 64× is a training-side property only. SOTA v3 (in flight) fixes this.

`PqDistanceCorrector` **increased MSE** (1.4e9 → 6.4e10) on training — feature normalization against global `max_pq_dist` saturates on SIFT's O(10⁵) distance scale. Kept advisory-only; final exact cosine rerank shields recall. SOTA v3 proposes per-vector normalization as the fix.

## Consequences

### Accepted contribution surface (v2, PR #356)

- `crates/ruvector-eml-hnsw/src/cosine_decomp.rs` — `EmlDistanceModel` with Pearson + retention trainers
- `crates/ruvector-eml-hnsw/src/selected_distance.rs` — cosine/L2 selected-dim kernels + `cosine_distance_simd`
- `crates/ruvector-eml-hnsw/src/hnsw_integration.rs` — `EmlHnsw`, `search_with_rerank`
- `crates/ruvector-eml-hnsw/src/progressive_hnsw.rs` — `ProgressiveEmlHnsw` multi-level cascade
- `crates/ruvector-eml-hnsw/src/pq.rs` + `pq_hnsw.rs` — PQ codebook + `PqEmlHnsw`
- `crates/ruvector-eml-hnsw/tests/` — recall_integration, sift1m_real, retention_vs_pearson, progressive_sift1m, sift1m_pq
- `crates/ruvector-eml-hnsw/benches/rerank_kernel.rs` — scalar vs SIMD micro-bench

### Rejected surface

- `EmlDistanceModel::fast_distance` (EML tree per call) — slower than baseline; reference-only, not on any public API path.
- `AdaptiveEfModel` — 290 ns/query disqualifies a per-query decision path until a <20 ns feature extractor is demonstrated.
- Sliced Wasserstein rerank — documented closed negative result.

### Default behavior

- `HnswIndex::new(...)` and all existing RuVector retrieval paths unchanged.
- `DbOptions::default()` produces the same behavior as before PR #353 or PR #352.
- `EmlHnsw` / `ProgressiveEmlHnsw` / `PqEmlHnsw` are explicitly constructed by callers opting into the approximate-then-exact pipeline.

### Coupling to PR #352

Accepted `EmlHnsw::search_with_rerank` requires a SIMD cosine kernel for the rerank stage. This ADR documents a **dependency** on PR #352's unified kernel landing; Tier 1B's direct SimSIMD integration in `selected_distance.rs::cosine_distance_simd` is the standalone fallback (already shipped in v2).

## v2 Branch Artifacts (shipped)

| Branch | Commit | Outcome |
|---|---|---|
| `fix/eml-hnsw-integration` | `aaea60af` | Stage-0: `EmlHnsw` wrapper + tests |
| `tier1a-fetchk-sweep` | `a5806096` | Fetch_k × selected_k grid. fetch_k=500 crosses 0.85 rerank. |
| `tier1b-simsimd-rerank` | `3ed71248` | SimSIMD cosine rerank: 5.65× @ 128d, 6.22× @ 384d. Recall Δ 0.002. |
| `tier1c-retention-selector` | `a453e4ea` | Retention: 0.817 vs Pearson: 0.712 at `selected_k=32, fetch_k=200`. +0.105 > 3σ. |
| `tier2-sliced-wasserstein` | `9a79f948` | SW @ L=100: 50.9× slower, 38pp worse. Falsified. |
| `tier3a-progressive-hnsw` | `f81a43dc` | 0.984 recall@10 at 961 µs p50 (2× latency at matched recall vs Tier-1A single-index). Build 5.9×. |
| `tier3b-pq-corrector` | `6a42d16d` | 64× memory (training-side), rerank recall 0.9515. Corrector MSE flaw documented. |
| `feat/eml-hnsw-optimizations-v2` | `db1c58b0` | Integrated. 92 tests pass. PR #356. Co-authored: @aepod, @shaal. |

### Summary of the 6-tier v2 outcome

- **4/5 follow-up tiers passed** their pre-declared acceptance bar (1B, 1C, 3A, 3B).
- **1/5 cleanly falsified** (Tier 2).
- **Biggest single finding:** retention-objective selector (+10.5 pp recall).
- **Biggest engineering lever:** SimSIMD rerank kernel (~6× kernel speedup).
- **Production unlock:** PQ pairing — 64× training-side memory reduction at recall ≥ 0.95 after rerank.
- **Documented design flaw:** `PqDistanceCorrector` normalization; SOTA v3 addresses it.

## SOTA v3 Push (in progress — 4-agent swarm)

| Tier | Agent | Target |
|---|---|---|
| SOTA-A | ml-developer | PQ-native HNSW (codes in graph, asymmetric PQ distance) + OPQ rotation. Realize 64× memory claim at runtime; +20-30% recall from OPQ. |
| SOTA-B | coder | rayon-parallel rerank (all 3 indexes) + 1M full-SIFT1M benchmark + plain `hnsw_rs` baseline for honest SOTA-gap measurement. |
| SOTA-C | ml-developer | Beam-search retention selector (width=4) + `PqDistanceCorrector` per-vector normalization fix. |
| SOTA-D | coder | First-class `HnswIndex::new_with_selected_dims()` in `ruvector-core`. Enum-backend pattern; no inverted dependency. |

Acceptance criteria declared pre-run; results fold into §Measured Evidence on completion.

## Open Questions (post-swarm, pre-SOTA)

1. **CLOSED — Tier 1C.** Retention-objective selector beats Pearson by +10.5 pp on SIFT1M (a453e4ea). The ceiling *was* the training objective, not SIFT's correlation structure.
2. **CLOSED — Tier 2.** Sliced Wasserstein is 50× slower and 38pp worse than cosine rerank on SIFT. Falsified for gradient-histogram datasets.
3. **OPEN — v3.** At what corpus size does PQ (Tier 3B) beat float storage+rerank on the memory-recall Pareto? 50k is too small. SOTA-B targets 1M.
4. **OPEN — v3.** Does the retention-objective selector scale to higher-dim transformer embeddings (CLIP-512, BGE-1024)? Tier 1C result is SIFT-128-d specific.
5. **OPEN — v3.** Can Tier 3A's progressive cascade be reformulated to avoid the 5.9× build-time penalty via native per-layer distance (fork hnsw_rs)?
6. **IN PROGRESS — SOTA-C.** Can `PqDistanceCorrector` be rescued with per-vector exact-distance normalization, or is the architecture fundamentally unsuited to SIFT's distance scale?
7. **IN PROGRESS — SOTA-D.** Landing `HnswIndex::new_with_selected_dims()` as first-class core API without creating a circular crate dependency.

## Alternatives Considered

### A) Merge PR #353 as-is

Rejected. No consumer, unsupported headline claim, `fast_distance` empirically broken. Would ship 7,182 orphan lines plus an unused fork feature.

### B) Reject PR #353 outright

Rejected. The Pearson-selected-dims-plus-exact-rerank pattern **is** measurably useful (Tier 1A: 0.974 recall@10 at selected_k=48, fetch_k=1000), and the author's pivot ("EML is the teacher, not the runtime") is correct. The gap was *integration wiring + honest measurement*, both of which v2 supplies.

### C) Accept only PR #352, defer all of PR #353

Partial overlap with the accepted decision — PR #352 is orthogonal and compelling. But PR #352 does not provide a retrieval-level pre-filter; it optimizes the inner kernel. Pre-filter + exact SIMD rerank is strictly additive over PR #352 alone.

### D) Build an orthogonal pre-filter (PCA/DiskANN-style) instead

Deferred. PCA directly optimizes variance preservation (what our Pearson selector approximates); a measured A/B vs learned-selected-dims at matched `selected_k` is tracked as a Tier-4 follow-up once the retention-objective question settles. `EmlHnsw` API is selector-agnostic, so swapping in a PCA selector is a one-file change.

## References

- **PR #352** — https://github.com/ruvnet/RuVector/pull/352 (shaal, unified SIMD kernel)
- **Issue #351** — https://github.com/ruvnet/RuVector/issues/351 (shaal, proposal + proof methodology)
- **PR #353** — https://github.com/ruvnet/RuVector/pull/353 (aepod, original crate)
- **PR #356** — https://github.com/ruvnet/RuVector/pull/356 (v2, this ADR's primary artifact)
- **Odrzywołek 2026** — "All elementary functions from a single operator", arXiv:2603.21852v2

### Datasets used

- `sift_base.fvecs` (1M × 128d, Texmex SIFT1M)
- `sift_query.fvecs` (10k × 128d)
- `sift_learn.fvecs` (100k × 128d — selector training, held out from base/query)
- `sift_groundtruth.ivecs` (10k × top-100 euclidean, for baseline comparison)

---

End of ADR-151. Supersedes nothing. Superseded by: none yet. Next ADR: 152.
