# ADR-269: Matryoshka Resolution Index (ruvector-mrl)

**Date:** 2026-06-26  
**Status:** Accepted  
**Deciders:** ruvnet, claude-flow  
**Branch:** `research/nightly/2026-06-26-matryoshka-mrl-index`

---

## 1. Context

MRL (Matryoshka Representation Learning) embeddings—shipped by OpenAI (`text-embedding-3`), Cohere (`embed-v3`), and Nomic (`embed-v1.5`)—guarantee that the first `D'` dimensions form a meaningful approximation of the full-`D` vector. This property enables a two-stage ANN strategy: screen candidates cheaply in the prefix space, then rerank with full-dimensional cosine.

No existing ruvector crate exploits this property. The ecosystem needs a proof-of-concept that:
1. Implements two concrete two-stage index variants (linear and graph)
2. Validates that the MRL property is required (not just helpful) for good recall
3. Provides real benchmarks on synthetic MRL-structured data as an acceptance reference

---

## 2. Decision

Introduce `ruvector-mrl`: a new Rust library crate with a `mrl-bench` binary that implements and benchmarks two MRL-aware index variants:

- **MrlLinear** — brute-force O(N·D_FAST) prefix scan + exact full-dim rerank
- **MrlGraph** — greedy kNN graph on D_FAST prefix (beam search) + exact full-dim rerank

Both implement the shared `MrlSearch` trait: `insert(id, &[f32])` and `search(&[f32], k) → Vec<SearchResult>`.

---

## 3. Rationale

### Why two-stage (prefix screen + full-dim rerank)?

The MRL guarantee makes prefix cosine a reliable proxy for full-dim cosine. Two-stage search exploits this: compute the cheap proxy for all N vectors, then pay the full cost only for the top `k × oversample` candidates. Cost reduction is proportional to `D_FAST / D_FULL`.

### Why both MrlLinear and MrlGraph?

MrlLinear isolates the dimension-reduction contribution (no graph structure). MrlGraph adds graph navigation for sub-linear candidate finding. Comparing the two separates the two speedup sources:
- MrlLinear speedup ≈ D_FULL / D_FAST (from dimension reduction alone)
- MrlGraph speedup ≈ (D_FULL / D_FAST) × (N / (ef × M)) (dimension reduction × graph navigation)

### Why two-phase graph build?

Greedy sequential insertion (connect node i to its M nearest among {0..i-1}) leaves node 0 with zero outgoing edges, crippling beam search from that entry point. The two-phase design: stage 1 stores all vectors, stage 2 computes the full O(N²·D_FAST) symmetric kNN—guarantees every node has well-connected edges before the first search.

### Why two experiments?

The fundamental insight this research validates is: **MRL speedup is training-dependent**. Random Gaussian vectors have no Matryoshka structure; the prefix is no more informative than a random projection. A single experiment on MRL-structured data would not document this limitation. The two-experiment design:
- Experiment A (random): recall@10 ≈ 0.28 at 1.9–3.6× speedup → documents the failure mode
- Experiment B (MRL-sim): recall@10 ≈ 0.94–1.00 at 2.0–3.5× speedup → documents the opportunity

---

## 4. Consequences

### Positive

- Establishes the MRL two-stage pattern as a first-class citizen in the ruvector crate family
- Provides reproducible acceptance thresholds (recall@10 ≥ 0.80 linear, ≥ 0.70 graph; speedup ≥ 1.5× linear, ≥ 3.0× graph on MRL-structured data)
- The `MrlSearch` trait is stable enough for downstream crates to implement and extend
- Clean extension point for future work: SIMD dot products, incremental `build_edges`, multi-layer HNSW graph, integration with real MRL APIs

### Negative / Trade-offs

- O(N²·D_FAST) graph build does not support streaming insertion (batch-only)
- No SIMD acceleration in this iteration—full throughput potential not realised
- Acceptance thresholds are tuned to synthetic MRL-sim data; real embedding recall may vary
- The benchmark binary is a single-process latency measurement, not a concurrent throughput benchmark

---

## 5. Alternatives Considered

| Alternative | Rejected Because |
|-------------|-----------------|
| HNSW multi-layer on D_FAST | Higher implementation complexity; single-layer graph sufficient to validate the concept |
| Product quantisation in D_FAST | PQ adds compression orthogonal to prefix truncation; out of scope for this PR |
| Integration with real OpenAI API | Requires network access and API key; would make benchmarks non-reproducible |
| Extend ruvector-hnsw | Adding MRL as a mode to an existing graph changes its public API; new crate is cleaner |

---

## 6. Implementation

```
crates/ruvector-mrl/
├── Cargo.toml
└── src/
    ├── lib.rs    — MrlSearch trait, dot(), normalize(), recall_at_k()
    ├── flat.rs   — FlatIndex (ground-truth baseline)
    ├── graph.rs  — GreedyGraph (insert, build_edges, beam_fast, rerank)
    ├── mrl.rs    — MrlLinear, MrlGraph
    └── main.rs   — mrl-bench binary (Experiment A + B + acceptance test)
```

All files under 500 lines. 7 unit tests. Workspace dependency on `rand = "0.8"` (already present).

---

## 7. Measured Results (2026-06-26, release build)

### Experiment A — Random Gaussian

| Variant | Recall@10 | Speedup |
|---------|-----------|---------|
| FlatFull | 1.000 | 1.0× |
| MrlLinear | 0.284 | 1.9× |
| MrlGraph | 0.211 | 3.6× |

### Experiment B — MRL-Simulated (α=0.25)

| Variant | Recall@10 | Speedup |
|---------|-----------|---------|
| FlatFull | 1.000 | 1.0× |
| MrlLinear | 1.000 | 2.0× |
| MrlGraph | 0.943 | 3.5× |

Acceptance test result: **PASS** (all 4 criteria met on Experiment B).

---

## 8. Security Considerations

- No network access; benchmark uses seeded synthetic data only
- No user input at runtime beyond command-line invocation (no args expected)
- No file I/O beyond stdout; no path sanitisation required
- No secrets, credentials, or external dependencies beyond `rand`

---

## 9. Related ADRs

| ADR | Topic | Relationship |
|-----|-------|--------------|
| ADR-083 | HNSW single-layer graph | Same graph primitive; MRL adds dimension split |
| ADR-187 | Filtered ANN (ACORN) | Orthogonal filtering vs. dimension reduction |
| ADR-268 | Capability-gated ANN | Capability gating is orthogonal; can wrap MrlSearch |
| ADR-265 | Matryoshka coarse-fine HNSW | Prior nightly; this ADR isolates pure prefix truncation |

---

## 10. Open Questions

1. What alpha produces the best recall–speedup curve on real MRL embeddings?
2. Can incremental `build_edges` be added without breaking the trait API?
3. Does adding SIMD dot products to `graph.rs` close the gap between MrlLinear and FlatFull latency enough to justify the unsafe code?
4. Should `MrlSearch` be promoted to the ruvector-core crate for cross-crate use?

---

## 11. Decision Record

| Field | Value |
|-------|-------|
| Status | Accepted |
| Date | 2026-06-26 |
| Implemented in | `crates/ruvector-mrl` v0.1.0 |
| Benchmark command | `CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse cargo run --release -p ruvector-mrl --bin mrl-bench` |
| Test command | `CARGO_REGISTRIES_CRATES_IO_PROTOCOL=sparse cargo test -p ruvector-mrl` |
| Tests passing | 7 / 7 |
| Acceptance test | PASS |
