# ADR-272: Scalar Quantization with Two-Stage Re-ranking in HNSW Graph

**Status**: Proposed  
**Date**: 2026-07-11  
**Authors**: Nightly Research Agent  
**Crate**: `crates/ruvector-sq-hnsw`  
**Branch**: `research/nightly/2026-07-11-sq-rerank-hnsw`

---

## Context

RuVector's existing quantization work covers product quantization (ADR via `ruvector-pq-search`, nightly 2026-06-20).  Product quantization groups dimensions into sub-spaces and trains a codebook per sub-space.  It offers excellent compression but requires Lloyd's k-means training and is computationally expensive for small or dynamically-updating corpora.

**Scalar quantization (SQ)** is simpler: each float32 dimension is independently scaled to an integer in [0, 255] (SQ8) or [0, 15] (SQ4).  Training is a single pass to find per-dimension min/max.  SQ is used in production by Qdrant (v1.9), LanceDB, and FAISS's `IndexScalarQuantizer`.

The need: a memory-efficient, zero-dependency, pure-Rust SQ implementation composable with RuVector's graph traversal primitives, with well-understood recall characteristics at 128-dim scale.

---

## Decision

Add `crates/ruvector-sq-hnsw` implementing five variants of SQ-based ANN:

1. **FlatExact** — exact brute force (ground truth).
2. **FlatSq8** — SQ8 brute-force scan + full-precision re-rank (ef = k×10).
3. **GraphSq8 (NSW)** — single-layer NSW graph with SQ8 distances + re-rank.
4. **GraphSq4 (NSW)** — single-layer NSW graph with SQ4 distances + re-rank.
5. **SqHnsw2** — two-layer HNSW with SQ8 distances + re-rank (the primary contribution).

All variants share the `NnSearch` trait.  No external crate dependencies.  A single `ScalarQuantizer` trained once on the corpus is passed to all variants.

---

## Consequences

### Positive
- **Memory**: SQ8 reduces vector storage 4×; SQ4 reduces 8× compared with f32.
- **Recall**: SqHnsw2 achieves 0.937 recall@10 at 10K × 128-dim; FlatSq8 achieves 1.000 (re-rank fully recovers).
- **Zero dependencies**: crate is self-contained, compiles on any `rustc stable` target including WASM.
- **Trait composability**: `NnSearch` plugs into any retrieval pipeline without coupling to the quantization scheme.
- **NSW ceiling documented**: benchmarks show NSW flattens at ~0.80 recall at 128 dims; HNSW2 breaks this ceiling.

### Negative / Tradeoffs
- **Graph memory overhead**: HNSW2 stores both SQ codes AND original f32 vectors (needed for re-ranking), so total memory is ~2.3× vs full-precision flat index.  For pure compression, drop originals and accept approximate-only distances.
- **Build time**: HNSW2 at n=10K builds in ~20s (single-threaded).  Parallelism with rayon is a future work item.
- **Static quantizer**: if the embedding distribution shifts (e.g., model upgrade), the quantizer must be retrained and the index rebuilt.

---

## Alternatives Considered

| Alternative | Reason not selected |
|-------------|-------------------|
| Extend `ruvector-pq-search` with SQ | PQ and SQ have different API shapes; a new crate keeps concerns separate |
| Use `half` crate for f16 | f16 reduces storage 2× but doesn't enable integer-ops path; SQ8 is more portable |
| Implement full multi-layer HNSW | 2-layer HNSW achieves 0.937 recall which meets threshold; full HNSW is future production work |
| Quantize only during graph traversal, store f32 codes | Higher memory; SQ8 codes are needed for SIMD-friendly distance ops |

---

## Implementation Plan

### Phase 1 (this PR — PoC)
- [x] `ScalarQuantizer` with SQ8 and SQ4
- [x] `FlatExact`, `FlatSq8`
- [x] `GraphSq8`, `GraphSq4` (NSW)
- [x] `SqHnsw2` (2-layer HNSW)
- [x] Integration tests (6 passing)
- [x] Benchmark binary with real numbers

### Phase 2 (production hardening)
- [ ] SELECT-NEIGHBORS-HEURISTIC edge pruning (HNSW paper §4.2)
- [ ] rayon parallel construction
- [ ] serde / bincode serialization for persist/load
- [ ] `no_std` support (swap `Vec` internals for fixed-size arrays)
- [ ] Feature flag: `simd` for `avx2` / `neon` integer distance

### Phase 3 (ecosystem integration)
- [ ] Wire `SqHnsw2` as a backend option in `ruvector-core`
- [ ] Expose as MCP tool surface (`sq_hnsw_insert`, `sq_hnsw_search`)
- [ ] ruFlo step for agent memory compaction with SQ migration

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-sq-hnsw --example benchmark`, x86_64 Linux, 2026-07-11.

| Variant    | Mean(μs) | p50(μs) | p95(μs) | QPS  | Mem(MB) | Recall@10 |
|-----------|---------|--------|--------|------|--------|---------|
| FlatExact  | 1773    | 1769   | 1835   | 564  | 4.88   | 1.000   |
| FlatSq8    | 2520    | 2424   | 3012   | 397  | 6.10   | 1.000   |
| NSW-SQ8    | 5127    | 5104   | 5369   | 195  | 8.55   | 0.798   |
| NSW-SQ4    | 6272    | 6245   | 6682   | 159  | 7.94   | 0.802   |
| HNSW2-SQ8  | 3660    | 3629   | 4009   | 273  | 11.14  | 0.937   |

n = 10,000, dims = 128, k = 10, queries = 100.

---

## Failure Modes

| Mode | Detection | Recovery |
|------|----------|---------|
| Recall below threshold | Canary recall monitoring | Increase ef_search or M |
| Quantizer staleness after distribution shift | Recall drops on new queries | Retrain quantizer, rebuild index |
| Graph node degree cap exceeded | Connectivity degrades | Increase M cap or use SELECT-NEIGHBORS |
| Memory pressure | OOM on edge device | Switch to SQ4; drop originals (approximate-only) |
| Build time too long | CI timeout | Reduce ef_build, add parallelism |

---

## Security Considerations

- SQ codes are **not cryptographically secure** — they expose approximate value ranges.
- Pair with `ruvector-proof-gate` when inserting into shared agent memory that requires integrity guarantees.
- Adversarial vector insertion can degrade graph quality; monitor mean recall against a canary set.
- Do not use raw SQ codes as authentication tokens.

---

## Migration Path

**From full-precision flat index:**
1. Train `ScalarQuantizer` on existing corpus.
2. Rebuild index with `SqHnsw2`, passing originals for re-ranking.
3. Serve from the new index; validate recall on held-out test set.
4. Optionally drop f32 originals once recall is confirmed acceptable.

**From PQ-ADC (`ruvector-pq-search`):**
- Both implement different quantization schemes; keep both as backends selected by corpus size / dimension.
- PQ preferred for d ≥ 256 (better compression ratio).
- SQ preferred for d ≤ 128, fast retraining, or WASM targets.

---

## Open Questions

1. **Should SQ and graph be two separate crates** (`ruvector-sq` + `ruvector-hnsw`) rather than combined? Composability would improve; complexity of cross-crate dependency increases.
2. **How to handle incremental quantizer updates** without full index rebuild?
3. **Is SELECT-NEIGHBORS-HEURISTIC needed for correctness** at n > 100K, or is edge cap sufficient?
4. **WASM size**: with originals stored, WASM binary grows. Should the WASM variant drop originals and accept approximate-only distances?
5. **Coherence scoring**: can SQ distances be incorporated into the coherence gating (ADR-XXX) pipeline, or does coherence require full-precision vectors?
