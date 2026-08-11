# ADR-273: Vector Similarity Join as a First-Class RuVector Operation

**Status:** Proposed  
**Date:** 2026-07-30  
**Crate:** `ruvector-sim-join`  
**Branch:** `research/nightly/2026-07-30-sim-join`

---

## Context

RuVector provides k-NN search: given a single query vector, return the k most similar vectors in an index. This covers most retrieval use cases but misses an important class of operations:

**Similarity join**: given two sets A and B of vectors, return all pairs (a, b) ∈ A × B such that `cosine_similarity(a, b) ≥ θ`.

This is structurally different from k-NN:
- No single query vector; the entire set B is the query
- Output size is variable (0 to |A|×|B| pairs) and not bounded by k
- Recall is measured against the complete set of true pairs, not just k

The operation appears in three RuVector use cases that currently require application-layer workarounds:

1. **Knowledge graph edge induction**: build a semantic edge between entity a and entity b whenever their embeddings are cosine-similar above θ
2. **Agent memory cross-reference**: find all pairs of memories that are semantically related (for linking or deduplication)
3. **RAG corpus deduplication**: before indexing, remove near-duplicate chunks using `self_join(chunks, 0.95)`

Without a first-class similarity join, users must run n k-NN queries (one per element of B), which is suboptimal and does not expose the LSH/IVF tradeoff.

---

## Decision

Add `crates/ruvector-sim-join` as a standalone research crate implementing the `SimJoin` trait with three strategies.

The crate is a PoC with the following properties:
- Zero external dependencies
- Pure safe Rust
- Trait-based (`SimJoin`) for composability
- Three measurable variants: `BruteJoin`, `LshJoin`, `IvfJoin`
- Real benchmark binary with measured latency, recall, and throughput

The `SimJoin` trait and `Pair` type are proposed API candidates for eventual integration into `ruvector-core` or as a standalone `ruvector-sim-join` production crate.

---

## Consequences

**Positive:**

- Enables knowledge graph construction directly from raw embeddings without repeated k-NN queries
- Provides a principled API for agent memory cross-reference and deduplication
- Establishes the regime-dependent LSH/IVF tradeoff empirically (IVF is better for low-threshold, high-density joins)
- Zero-dependency Rust crate compiles to WASM for edge deployment

**Negative / Risks:**

- Approximate variants (LshJoin, IvfJoin) have tuning parameters that require calibration for production use
- At n=5000, the serial IvfJoin takes 1.3s — acceptable for background tasks but not low-latency search
- Self-join memory scales as O(|pairs| × 24 bytes) — at n=5000 with 2M pairs: 48MB of pair storage

**Neutral:**

- The `BruteJoin` baseline is O(n²d), which is already the only option users have today via repeated k-NN; this ADR adds approximate variants on top

---

## Alternatives Considered

### A: Expose similarity join as a server-side query type

Add a `JOIN` endpoint to `ruvector-server` that takes two named collections and a threshold. This is the right production direction but requires server changes and blocking cross-collection queries. The standalone crate is a prerequisite.

### B: Use k-NN queries with n queries (one per B element)

Current workaround. Complexity: O(n × HNSW-search) ≈ O(n × log(n) × d). For n=2000: ~2000 × 10 × 128 ≈ 2.56M comparisons vs brute-force 512M. Sounds better, but: (a) HNSW search finds only k nearest, not all above threshold; (b) each element of B must be indexed, not just queried; (c) the merge step is complex. The IvfJoin approach is comparable in complexity and easier to reason about for the join problem.

### C: GPU-accelerated matrix multiplication (à la FAISS)

For n=10,000+ at batch scale, GPU matmul is the fastest approach (using cuBLAS `Sgemm` to compute the full A×B similarity matrix in one shot). This is the right direction for very large n but requires GPU availability and a CUDA/Metal dependency. The CPU-only `IvfJoin` is the right starting point.

---

## Implementation Plan

**Phase 1 (this PR):**
- `SimJoin` trait definition
- `BruteJoin`, `LshJoin`, `IvfJoin` implementations
- Deterministic benchmark binary
- Measured acceptance tests

**Phase 2 (future):**
- Parallel `IvfJoin` using `rayon` (4× speedup on 4-core)
- k-means++ initialisation for higher recall
- Auto-tuning calibration phase
- `ruvector-graph::add_edges_from_join()` integration

**Phase 3 (production):**
- MCP tool surface: `vector_similarity_join` tool
- ruFlo scheduler integration for periodic memory cross-reference
- WASM feature flag for edge deployment
- Privacy-preserving sketch-based join for multi-tenant

---

## Benchmark Evidence

All numbers from `cargo run --release -p ruvector-sim-join --bin benchmark` on Linux x86_64, 4 CPUs, Rust 2.3.0, release build.

**n=2000, d=128, threshold=0.29 (data-driven), 407,021 GT pairs:**

| Variant | Mean (µs) | Recall | Speedup |
|---------|-----------|--------|---------|
| BruteJoin | 255,902 | 1.000 | 1.00× |
| LshJoin (4b×10t) | 486,740 | 0.887 | 0.53× |
| **IvfJoin (K=16, p=3)** | **169,755** | **0.998** | **1.51×** |

**Acceptance criteria:**
- BruteJoin recall = 1.000: ✓ PASS
- LshJoin recall ≥ 0.70: ✓ PASS (0.887)
- IvfJoin recall ≥ 0.70: ✓ PASS (0.998)
- At least one approx variant faster than BruteJoin at n=2000: ✓ PASS (IvfJoin 1.51×)

---

## Failure Modes

1. **LSH bucket explosion at low thresholds**: when true-pair density > 5%, LSH is slower than brute force. Mitigation: switch to IvfJoin below θ ≈ 0.40 (document in integration guide).

2. **IVF centroid divergence**: k-means may not converge with random init on pathological data. Mitigation: 10-iteration Lloyd's is sufficient for most embeddings; add k-means++ in Phase 2.

3. **Memory pressure at large n**: storing all pairs as `Vec<Pair>` at n=5000 (2M pairs, 48MB) is acceptable for background tasks; for streaming use cases, emit pairs via a callback or channel instead.

---

## Security Considerations

- Cross-tenant similarity join would reveal inter-tenant semantic relationships: enforce namespace isolation before any join operation (compatible with ruvector-capgated's access model)
- Adversarial embeddings designed to match legitimate entity embeddings could inject spurious graph edges: combine with proof-gated write semantics (ruvector-proof-gate) to require attestation for edge induction
- Pair output may expose ranking information about private documents: apply differential privacy noise to similarities if exposing join results via MCP or API

---

## Migration Path

This is a new crate with no breaking changes to existing APIs. Integration with `ruvector-graph` and `ruvector-agent-memory` is additive. When Phase 2 lands, users currently running n k-NN queries as a join workaround can migrate to `IvfJoin::join()` with a one-line change.

---

## Open Questions

1. Should `SimJoin` go into `ruvector-core` as a first-class trait, or stay in `ruvector-sim-join` as a standalone crate?
2. At what density does IvfJoin stop being faster than BruteJoin? Current evidence suggests it remains faster at density up to ~40% (407,021 / 4,000,000 ≈ 10%). Needs testing at higher densities.
3. Should we add a `threshold_calibrate(a_sample, b_sample) -> f32` function that estimates the right threshold for a dataset automatically?
4. Is LSH join worth supporting at all, or should IvfJoin be the only approximate strategy? Current evidence suggests LSH should remain for high-threshold (sparse) joins only.
