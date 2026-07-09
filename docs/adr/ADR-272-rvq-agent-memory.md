# ADR-272: Residual Vector Quantization for Compact Agent Memory

**Date**: 2026-07-09  
**Status**: Proposed  
**Deciders**: RuVector nightly research agent  
**Tags**: compression, vector-quantization, agent-memory, rvq

---

## Context

RuVector's agent memory layer (RVM, pi-brain) stores embeddings as raw float32 vectors. At production scale (1M+ agent-memory turns with 1536-dim embeddings), raw storage requires ~6 GB per million vectors. This is prohibitive for edge deployments, Hailo inference clusters, and long-running agents with multi-year episodic memory.

Three compression strategies are well-studied in the literature:

1. **Scalar Quantization (SQ)**: map each float32 dimension to uint8 via per-dim min-max scaling. 4× compression, near-lossless (MSQE ≈ 0.0003 at D=32).
2. **Product Quantization (PQ)**: split D dims into M independent sub-spaces, each quantized to K centroids. 32× compression at M=4, but assumes IID dimensions.
3. **Residual Vector Quantization (RVQ)**: L sequential stages, each quantizing the full-D residual from the prior stage. Same 32× compression at L=4, but captures cross-dimension correlation.

LLM embeddings are fundamentally **not** IID. They live near semantic manifolds with cluster structure determined by topic, modality, and language. This makes PQ's independence assumption incorrect and motivates investigating RVQ.

---

## Decision

Introduce `crates/ruvector-rvq` as a measured Rust proof-of-concept for RVQ as the compression primitive in RuVector's agent memory pipeline.

The crate provides:
- A `VectorQuantizer` trait with `train / encode / decode / bytes_per_vector / codebook_bytes / name`
- Three implementations: `ScalarQuantizer`, `ProductQuantizer`, `ResidualQuantizer`
- A two-suite benchmark binary (`benchmark`) that validates the core hypothesis: **RVQ MSQE < PQ MSQE on clustered semantic data at equal byte budget**
- Test coverage with deterministic acceptance thresholds

The acceptance criterion is: `rq_msqe < pq_msqe` on clustered data.

---

## Consequences

### Positive

- **5.2× lower MSQE** than PQ at same 4 bytes/vector on clustered semantic data (measured)
- **32× storage compression** vs raw float32 (same as PQ, vs. 4× for SQ)
- **Codebook fits in L1 cache**: 4 stages × 32 centroids × 32 dims × 4 bytes = 16 KB
- **Decode is O(L) additions**: ~0.09 μs/vector, negligible vs. LLM inference
- **No external dependencies**: pure Rust, only `rand` and `rand_distr` from workspace

### Negative

- **Encode is slower than PQ**: ~3.9 μs/vector vs ~1.0 μs/vector (4× slower; acceptable since encoding happens at write-time, not search-time)
- **Codebook is 4× larger than PQ**: 16 KB vs 4 KB (still cache-resident at L1/L2)
- **Training is slower than PQ**: ~167 ms vs ~55 ms for 5K vectors at K=32 (offline, done once)
- **Recall@10 comparable to PQ** at these settings (0.506 vs 0.499); improving recall requires larger K or more stages

### Neutral

- On isotropic Gaussian data (Suite 1), RVQ and PQ perform comparably (0.557 vs 0.530 MSQE). RVQ does not regress on non-clustered data.
- SQ remains the quality baseline at 32 bytes/vector (MSQE ≈ 0.0003); the 32× compression of RVQ costs ~1500× in MSQE but keeps Recall@10 ≈ 0.50.

---

## Benchmark Evidence

All results: Intel Xeon @ 2.80 GHz, x86_64 Linux, `cargo run --release`, 5K train / 2K test / 200 queries, D=32.

### Suite 2: Clustered Semantic Data (critical test, 100 clusters, σ=3.0)

| Variant | Bytes/Vec | MSQE | Recall@10 |
|---------|-----------|------|-----------|
| ScalarQ-8bit | 32 | 0.000324 | 0.949 |
| ProductQ | 4 | 2.568973 | 0.499 |
| **ResidualQ-4** | **4** | **0.497257** | **0.506** |

RVQ improvement over PQ: **5.2× lower MSQE**. Acceptance test: **PASS ✓**

### Suite 1: Isotropic Gaussian (baseline, PQ-optimal regime)

| Variant | Bytes/Vec | MSQE | Recall@10 |
|---------|-----------|------|-----------|
| ScalarQ-8bit | 32 | 0.000171 | 0.984 |
| ProductQ | 4 | 0.529766 | 0.150 |
| ResidualQ-4 | 4 | 0.556656 | 0.162 |

On IID data RVQ is within 5% of PQ (no regression).

---

## Alternatives Considered

### Alternative 1: Stay with PQ Only

PQ is already used in FAISS and is well-understood. However, the 5.2× quality gap on semantic data is too large to accept when this quality directly affects agent reasoning quality (retrieved context fidelity).

### Alternative 2: Additive Quantization (AQ)

AQ jointly optimizes all codebooks via beam search over code assignments. Higher quality than RVQ at same byte budget, but exponentially higher encode cost. Rejected for online encoding; viable for offline archival compression.

### Alternative 3: Neural VQ-VAE / Codec

End-to-end learned quantization achieves highest quality, but requires a neural encoder/decoder (inference cost) and domain-specific training. Viable for a dedicated "memory distillation" pipeline; too heavy for general-purpose inline compression.

### Alternative 4: RaBitQ (1-bit per dim)

Extreme compression (D/8 bytes/vector) with random rotation and binary quantization. Achieves good ANN recall but very high MSQE; not suitable when agents need to reconstruct approximate vectors (e.g., for context summarization or diff computation).

---

## Implementation Notes

- `kmeans` uses `StdRng` (not `SmallRng`, which requires a feature flag in rand 0.8)
- The `VectorQuantizer::name()` returns `&'static str` to allow `BenchResult.name: &'static str`
- `generate_clustered_vectors` uses a fixed internal seed (`0xDEAD_BEEF_CAFE_1234`) for cluster centers so that train/test/query splits share the same semantic manifold structure
- The benchmark uses `N_CLUSTERS=100 > K=32` to force the clustered-data scenario where PQ's product code wastes capacity on never-occurring sub-space combinations

---

## Future Work

1. **Integrate `ruvector-rvq` into RVM's write path** as an optional compression codec (ADR candidate)
2. **Add SIMD-accelerated nearest-centroid search** — the inner product loop is the encode bottleneck
3. **Implement online codebook update** via EWC++ for adaptive drift handling
4. **Benchmark on real LLM embedding datasets** (MS-MARCO, BEIR) at D=768/1536
5. **Add quantization-aware encode path** that returns distance-to-nearest-centroid for uncertainty-weighted retrieval
6. **Explore hierarchical RVQ** (coarse IVF + fine RVQ) for billion-scale memory
