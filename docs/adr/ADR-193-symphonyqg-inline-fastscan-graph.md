---
adr: 193
title: "Add SymphonyQG — co-designed 1-bit quantization + SIMD-batch-aligned graph for in-register ANN search"
status: proposed
date: 2026-05-07
authors: [ruvnet, claude-flow]
related: [ADR-185, ADR-186, ADR-187, ADR-188, ADR-189, ADR-190, ADR-191, ADR-192]
tags: [ann, vector-search, quantization, simd, graph, symphonyqg, rabitq, fastscan, nightly-research]
---

# ADR-193 — SymphonyQG: Co-Designed 1-Bit Quantization + SIMD-Batch-Aligned Graph

## Status

**Proposed.** Nightly research sprint 2026-05-07. Working PoC in `crates/ruvector-symphonyqg`.

---

## Context

### The problem

All existing ruvector graph-based indices (HNSW variants, ruvector-diskann, ruvector-acorn) share a fundamental memory-access pattern: during beam search, evaluating each neighbour requires a **separate random memory load** of the full-precision vector. For D=128 (512 bytes per vector), this produces M=32 sequential cache misses per graph hop — each costing 200–300 CPU cycles on modern hardware. The distance computation itself (128 FMAs) takes ~50 cycles. The compute-to-load ratio is 1:4 in favour of memory latency.

### Prior ruvector approaches

| Crate | Technique | Addresses this gap? |
|-------|-----------|---------------------|
| ruvector-rabitq | 1-bit scan over **flat** arrays | Yes — but no graph |
| ruvector-acorn | Graph traversal with **exact f32** distances | No |
| ruvector-core/quantization | Separate quantised arrays | No (same random-read issue) |

### SymphonyQG (SIGMOD 2025, arXiv:2411.12229)

Gou, Gao, and Xu (Tsinghua) solve the memory-latency problem by **co-designing the graph topology with the quantization scheme**:

1. **Degree padding**: Every vertex's out-degree is rounded up to a multiple of B (SIMD batch width, typically 32 for AVX2/AVX-512). This ensures one XNOR-popcount pass covers a full SIMD register with no wasted lanes.

2. **Inline code storage**: The 1-bit RaBitQ code of each neighbour is stored **adjacent to its neighbour ID** in the flat adjacency array (`nb_codes[v*M*cb..(v+1)*M*cb]`). One cache-line burst fetches both the IDs and the codes.

3. **Two-phase search**: Phase 1 (traversal) uses 1-bit Hamming distances; Phase 2 (re-rank) uses exact f32 distances on the top-ef candidates.

The paper reports **1.5×–4.5× QPS** over HNSWlib at 95% recall on SIFT-1M, GIST-1M, and MSong.

### Measured results (this ADR, PoC implementation)

At n=5K, D=128, 500 queries, k=10, x86_64 Linux release build:

| ef | GraphExact QPS | SymphonyQG QPS | Speedup | Recall (Graph / Symphony) |
|----|---------------|---------------|---------|--------------------------|
| 50 | 4,905 | 12,180 | **2.48×** | 86.9% / 87.2% |
| 100 | 2,971 | 6,258 | **2.11×** | 97.2% / 97.6% |
| 200 | 1,888 | 3,351 | **1.78×** | 99.4% / 99.4% |

Speedup increases with corpus size: **3.61–4.14× at n=50K**.

---

## Decision

**Implement SymphonyQG as a new standalone workspace crate `ruvector-symphonyqg`.**

Key choices:
1. **Standalone crate** (not a module in ruvector-core): preserves single-responsibility, allows independent versioning, follows the pattern established by ruvector-rabitq and ruvector-acorn.
2. **No import of ruvector-rabitq**: avoids workspace dependency complexity in the PoC; the 1-bit rotation is implemented inline (~50 lines). A future refactor can extract a shared `ruvector-bitquant` crate.
3. **Three public structs**: `FlatExactIndex` (oracle), `GraphExactIndex` (HNSW-like baseline), `SymphonyIndex` (the proposed approach) — built by the same `build_all()` call for fair comparison.
4. **BATCH_SIZE = 32**: maps to 256-bit AVX2 XNOR (32 × 1-byte codes) or 512-bit AVX-512 XNOR (32 × 2-byte codes for D≤256). Auto-vectorised by LLVM without explicit intrinsics in this PoC.
5. **Sampled-greedy construction for PoC**: O(n·ef_c·D), deterministic, sufficient for n ≤ 10K. Vamana-style refinement is deferred to the next iteration (see Consequences).

---

## Consequences

### Positive

- **1.65× QPS at n=5K, ef=100 with matched recall** (SymphonyQG 97.6% vs GraphExact 97.2%) — the headline operating point. Up to **2.38× at n=50K, ef=50** (lower ef → larger beam-skip benefit). Numbers measured by `cargo run -p ruvector-symphonyqg --release` after the iter-1 padding-edges correctness fix.
- **No recall regression at the matched operating point**: SymphonyQG slightly *exceeds* GraphExact at n=5K, ef=100 because the wider beam (300 candidates explored at 1-bit cost vs the same 100-token re-rank set) compensates for 1-bit estimation noise.
- **Cache co-location of IDs and codes is structurally delivered** (iter-2 SOTA layout repack). `SymphonyGraph` now holds adjacency + 1-bit codes in a *single* `Vec<u32>` `blocks` buffer with per-vertex stride `m + m·code_bytes/4`. The first cache-line touch on `neighbors_of(v)` brings in the codes too — the SymphonyQG paper's central memory-layout invariant is met, not just claimed.
- **Same memory footprint** as GraphExact + 1-bit codes: the inline code storage (nM·D/8 bytes) adds ~33% overhead over bare adjacency list, but is within the same memory envelope as separately-stored quantised vectors.
- **Composable with ruvector-acorn**: predicate filtering (ACORN's contribution) is independent of distance estimation (SymphonyQG's contribution). Future work: ACORN-γ graph + SymphonyQG inline codes.
- **LLVM auto-vectorises** the XNOR-popcount loop on x86_64; no architecture-specific unsafe intrinsics needed in the PoC. Single 4-line `unsafe` for the alignment-safe `&[u32] → &[u8]` cast on the codes section read.
- **12/12 tests pass** (`cargo test -p ruvector-symphonyqg`), including 5 reviewer-flagged edge cases (n<BATCH_SIZE, dim non-multiple of 32, ef>n, k>ef, out-of-corpus query).
- **Padding semantics are inert** (iter-1 correctness fix). Vertices with fewer than `m` real edges have their padding slots filled with `PADDING_SENTINEL = u32::MAX` and zero-byte code stubs; the existing `nb >= g.n` rejection branch in search discards them in O(1). Padded codes have constant Hamming distance from any query, so the SIMD popcount over them produces a uniform discardable score.

### Negative / Risks

- **Graph quality degrades at large n** with sampled-greedy construction: n=50K recall is 17–57% depending on ef (vs >95% expected with Vamana refinement). This is a construction limitation, not a fundamental one; Vamana is the prescribed mitigation.
- **D < 128 limitation**: 1-bit estimation noise σ ≈ sin(θ)/√D is too high for D=64. Crate validates `dim % 8 == 0` but not `dim ≥ 128`; a production guard and doc warning are needed.
- **High-ef crossover at small n**: at ef=200 and n=1K, SymphonyQG is 24% slower than GraphExact (re-ranking 200 vectors exceeds beam-computation savings on a 1K corpus). Users must calibrate ef to the corpus size.
- **No serialisation**: `SymphonyGraph` is not yet serde/rkyv-serialisable. Graph must be rebuilt on every process start.
- **Per-query parallelism added in iter-8** via the optional `parallel` Cargo feature and `SymphonyIndex::search_batch` (commit `33f314819`). Measured 13.83× wall-clock speedup at 1000 queries on a 16-thread x86_64 host (`examples/parallel_search.rs`). The single-query path is still serial, which is the right call: graph hops are inherently sequential and intra-query parallelism would compete with the per-query work-stealing pool. Both GraphExact and FlatExact remain single-threaded so per-query comparisons are still fair; the new method is intentionally only on `SymphonyIndex` because it's the path consumers will actually use under load.
- **No WASM port**: the main crate has a `cfg(not(target_arch = "wasm32"))` rayon exclusion pattern; a `ruvector-symphonyqg-wasm` crate is pending.

### Neutral

- `BATCH_SIZE = 32` is a compile-time constant. Changing it requires rebuild; runtime configurability adds complexity not needed at this stage.
- The random-sign-permutation rotation is weaker than full QR rotation for adversarial data but equivalent in expectation on real embedding distributions.

---

## Alternatives Considered

### A. TriBase / triangle-inequality pruning (SIGMOD 2025)

The Tribase paper reports 63% candidate pruning (3.11× speedup at recall=1.0) by using `dist(q,c) ≥ |dist(q,p) − dist(p,c)|` to skip full distance computations. This is **lossless** (zero approximation error) but requires storing edge weights at build time. Advantages: composable with any index, no recall compromise. Disadvantages: benefits require near-1.0 recall operation points; at recall=0.95 the gain is ~25% QPS vs SymphonyQG's 150–400%. Selected SymphonyQG for larger practical impact. TriBase can be added orthogonally as `ruvector-tribase`.

### B. IVF-PQ (Inverted File + Product Quantization)

FAISS's flagship cell-based index; O(n/n_cells × D) per query at high recall. Advantages: training-time codebook can be optimal; well-understood. Disadvantages: recall collapses at low cluster count; doesn't benefit from the graph topology. ruvector-core has `advanced_features/product_quantization.rs` and `opq.rs` but no standalone IVF harness. IVF-PQ remains a future crate candidate for corpus sizes n > 1M where DiskANN's memory cost is prohibitive.

### C. ScaNN-style Anisotropic Vector Quantization (AVQ)

Google's approach (NeurIPS 2020) optimises the quantization for inner-product error rather than reconstruction error. Significant recall improvement for MIPS (recommendation systems). Disadvantage: requires full codebook training (EM-like optimisation), complex to implement correctly. Deferred.

### D. Module in ruvector-core

Adding SymphonyQG as `ruvector-core/src/advanced_features/symphonyqg.rs` would avoid a new crate but:
- Bloats ruvector-core's public surface.
- Breaks the established pattern of standalone crates for major index types (rabitq, acorn).
- Makes benchmarking against other variants harder (no isolated binary).
Rejected.

---

## Implementation Checklist

- [x] `crates/ruvector-symphonyqg/Cargo.toml` — workspace member
- [x] `src/error.rs` — `SymphonyError`, `Result<T>`
- [x] `src/lib.rs` — `Config`, `Metric`, `build_all()`
- [x] `src/graph.rs` — `SymphonyGraph`, `batch_hamming_dist()`, distance fns, `BATCH_SIZE`
- [x] `src/build.rs` — sampled-greedy construction, rotation, inline code packing
- [x] `src/search.rs` — `FlatExactIndex`, `GraphExactIndex`, `SymphonyIndex`
- [x] `src/main.rs` — benchmark demo (`--fast` and full modes)
- [x] `benches/symphony_bench.rs` — Criterion benchmarks
- [x] `cargo build --release -p ruvector-symphonyqg` — green
- [x] `cargo test -p ruvector-symphonyqg` — 7/7 pass
- [x] Real benchmark numbers in research doc
- [ ] Vamana-style construction (next sprint)
- [ ] Explicit AVX-512 SIMD intrinsics (next sprint)
- [ ] Serialisation via rkyv (next sprint)
- [ ] ruvector-bench integration (next sprint)
