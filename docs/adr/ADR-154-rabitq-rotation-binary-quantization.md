# ADR-154: RaBitQ — Rotation-Based 1-Bit Quantization for ANNS

## Status

Proposed

## Date

2026-04-23

## Authors

ruv.io · RuVector Nightly Research (automated nightly agent)

## Relates To

- ADR-001 — Tiered quantization strategy (BinaryQuantized in ruvector-core)
- ADR-006 — Unified Memory Service (AgentDB)
- ADR-027 — HNSW parameterised query fix
- Research: `docs/research/nightly/2026-04-23-rabitq/README.md`

---

## Context

ruvector-core already exposes four quantization tiers (ADR-001):

| Tier | Method | Compression | Recall |
|------|--------|-------------|--------|
| Scalar (u8) | threshold-quantize | 4× | ~95% |
| Int4 | nibble-pack | 8× | ~90% |
| Product (PQ) | k-means codebook | 8–16× | ~85% |
| Binary | sign(x_i) | 32× | ~20–60% |

The existing `BinaryQuantized` implementation uses **naive sign quantization**:
it sets bit_i = 1 if x_i ≥ 0 and then measures **Hamming distance** between
raw bit-patterns. This has two known deficiencies:

1. **No rotation**: correlated dimensions produce highly correlated bits,
   making the Hamming code a poor distance proxy for L2-structured data.
2. **Wrong distance model**: the linear Hamming distance does not correspond
   to the angular distance, so the ranking of candidates is unreliable.

RaBitQ (Gao & Long, SIGMOD 2024, arXiv:2405.12497) addresses both:

1. Applies a **random orthogonal rotation** P (Haar-uniform) before binarisation,
   making quantisation error isotropic across all dimensions. Error is O(1/√D).
2. Uses the **angular correction estimator**:
   ```
   est_sq_dist(q, x) = ‖q‖² + ‖x‖² − 2‖q‖·‖x‖·cos(π·(1 − B/D))
   ```
   where B = XNOR-popcount(B(q̂), B(x̂)), derived from
   E[B/D] = 1 − arccos(⟨q̂, x̂⟩)/π.

The VLDB 2025 extension (arXiv:2409.12353) adds asymmetric query encoding
(query in f32, database in 1-bit) and higher-order correction; this ADR
covers the symmetric baseline, which is the highest-value starting point.

### Measured gap between BinaryQuantized and RaBitQ

On n=5K Gaussian-cluster data (100 clusters, D=128, σ=0.6, k=10):

| Method | Recall@10 | QPS | Memory |
|--------|-----------|-----|--------|
| FlatF32 (exact) | 100.0% | 2,087 | 2.4 MB |
| BinaryQuantized (naive sign) | ~15–20%* | ~3,500 | 0.2 MB |
| **RaBitQ 1-bit (rotation + angular est.)** | **40.8%** | **4,396** | **0.2 MB** |
| RaBitQ+ rerank×5 | **98.9%** | **4,271** | 2.6 MB |
| RaBitQ+ rerank×10 | 100.0% | 4,069 | 2.6 MB |

*Estimated from literature; exact comparison requires wiring BinaryQuantized into the same search loop.

RaBitQ+ with 5× reranking achieves:
- **98.9% recall** vs FlatF32's 100%
- **2.05× throughput improvement** over exact flat search
- **17.5× memory compression** for the binary codes alone

---

## Decision

Introduce a standalone crate `crates/ruvector-rabitq` that implements:

1. **`RandomRotation`** — Haar-uniform random orthogonal D×D matrix via
   Gram–Schmidt orthonormalization, stored once and shared across all vectors.

2. **`BinaryCode`** — packed u64 bit-array with XNOR-popcount kernel and
   the angular correction distance estimator.

3. **Three swappable backends behind the `AnnIndex` trait**:
   - `FlatF32Index` — exact f32 brute-force (baseline)
   - `RabitqIndex` — 1-bit angular scan only
   - `RabitqPlusIndex` — 1-bit scan + configurable exact f32 reranking

The crate is intentionally standalone (no dependency on ruvector-core) so it
can be integrated into HNSW, DiskANN, or the graph index as a compression tier
without coupling to the quantization.rs refactor.

### Integration path (future)

```
ruvector-core quantization.rs
  → add RaBitQQuantized implementing QuantizedVector trait
  → wire into ruvector-hnsw as the "Binary" tier backing

ruvector-diskann
  → use BinaryCode for the in-memory candidate list during beam search
  → full vectors remain on SSD; binary codes in DRAM for filtering
```

### What is NOT in scope

- IVF partitioning (would lift recall at large n; separate ADR)
- Asymmetric query encoding (VLDB 2025 extension; separate ADR)
- WASM / Node.js bindings (follow-on once API stabilises)

---

## Consequences

### Positive

- **2.05× throughput** over exact flat search at 98.9% recall@10 (n=5K, D=128)
- **17.5× memory compression** for the binary code store (16 bytes/vec at D=128)
- **Theoretical error bound** unlike naive sign quantisation: recall degrades
  gracefully as O(1/√D) as dimensionality grows
- **Drop-in trait**: callers switch from `FlatF32Index` to `RabitqPlusIndex`
  by changing one constructor call
- Enables DRAM-resident billion-scale indexes: 1B × D=128 → ~16 GB binary
  vs ~512 GB f32

### Negative / Risks

- **Rotation cost**: building the D×D matrix is O(D³) (Gram–Schmidt); for D=1536
  (OpenAI embeddings) this is 3.6B operations — acceptable once per index load
  but must be cached
- **Rotation apply cost**: O(D²) per vector at build time; for n=50M at D=1536
  this is ~113T ops — must be parallelised with Rayon in production
- **Flat-scan recall degrades with large n**: at n=50K and rerank×10, recall@10
  is 56%; IVF partitioning is required to maintain recall at scale (ADR-155 TBD)
- **Clustered data assumption**: recall is substantially lower on uniform-random
  data (which does not occur in practice for trained embedding models)

### Neutral

- The `rand_distr::StandardNormal` dependency is already in the workspace
- Serialisation via `serde` allows index snapshots with zero extra work

---

## Alternatives Considered

| Alternative | Reason not chosen |
|-------------|-------------------|
| ACORN (SIGMOD 2024): predicate-agnostic filtered HNSW | Requires invasive graph-build-time changes; 400–600 LOC touching hnsw_rs internals |
| Fresh-DiskANN: streaming updates | Covered by existing delta-index / delta-graph crates |
| MRL (Matryoshka): adaptive truncation | Already implemented in ruvector-core (matryoshka.rs) |
| HNSW-SQ: scalar quantisation in graph traversal | Less novel; narrower impact than binary compression |
| IVF-Flat: inverted file index | Correct next step after RaBitQ; separate ADR planned |

---

## References

- Gao & Long, "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error
  Bound for Approximate Nearest Neighbor Search", SIGMOD 2024. arXiv:2405.12497
- Gao & Long, "RaBitQ+: Revisiting and Improving RaBitQ…", VLDB 2025. arXiv:2409.12353
- Indyk & Motwani, "Approximate Nearest Neighbors: Towards Removing the Curse of
  Dimensionality", STOC 1998 (LSH foundation)
- Johnson et al., "Billion-scale similarity search with GPUs" (FAISS), arXiv:1702.08734
- Qdrant v1.9.0 release notes: binary quantisation with oversampling rescoring (2024)
- RuVector crate: `crates/ruvector-rabitq/` (this PR)
