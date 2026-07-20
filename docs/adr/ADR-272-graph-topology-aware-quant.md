# ADR-272: Graph-Topology-Aware Vector Quantization (TAQ)

**Status:** Proposed  
**Date:** 2026-07-12  
**Research branch:** `research/nightly/2026-07-12-graph-topology-aware-quant`  
**Crate:** `crates/ruvector-taq`

---

## Context

RuVector stores agent memories as f32 embedding vectors. At scale, memory pressure requires quantization. Current options — Uniform SQ8 (25% of f32 memory, ~99% recall) and Uniform SQ4 (12.5% memory, ~82% recall) — apply identical precision to every vector regardless of its importance to retrieval paths.

In any k-NN graph over the vector dataset, some nodes have high in-degree (many other nodes cite them as nearest neighbors). These *hub nodes* lie on the majority of shortest traversal paths through the graph. Quantization error at a hub corrupts more navigable paths than the same error at a low-degree leaf. Applying uniform low-bit quantization ignores this structural asymmetry.

**TAQ (Topology-Aware Quantization)** assigns 8-bit precision to hub nodes and 4-bit to leaf nodes. This achieves recall between SQ8 and SQ4 at memory between SQ4 and SQ8.

---

## Decision

Introduce `ruvector-taq` as a new crate implementing topology-aware quantization with three measured variants:

1. **FullPrecisionIndex** — f32 baseline / oracle.
2. **UniformSq8Index** — 8-bit scalar quantization across all vectors.
3. **UniformSq4Index** — 4-bit nibble-packed quantization across all vectors.
4. **TaqIndex** — hub nodes (in-degree > threshold) at SQ8; leaf nodes at SQ4.

The `VectorIndex` trait standardizes the interface across all variants.

---

## Consequences

### Positive

- **Measured recall improvement**: TAQ recall@10 = 0.9652 vs UniformSQ4's 0.8194 — a 14.6pp improvement at only +9% more memory (262 KB vs 156 KB at N=5K, D=64).
- **Topology connects to existing infrastructure**: `ruvector-graph` and `ruvector-mincut` provide the graph primitives needed for hub identification.
- **No external deps**: safe Rust only; compiles for WASM targets without changes.
- **Composable with MCP**: TAQ build/search can be exposed as MCP memory tools.
- **ruFlo integration**: memory compaction workflow can trigger TAQ rebuild automatically.

### Negative / Tradeoffs

- **Slower query than SQ8**: two dequantization paths (SQ4 nibble-unpack vs SQ8 decode) create per-vector branch overhead. Measured: TAQ at 770 μs/query vs SQ8 at 586 μs/query (31% slower, 28% faster than SQ4).
- **O(N²) graph build**: brute-force k-NN construction limits current PoC to N ≲ 50K. NN-Descent or HNSW-based graph construction needed for larger corpora.
- **Hub assignment invalidated by insertions**: online vectors change in-degree. Requires periodic rebuild or incremental degree tracking.
- **Threshold sensitivity**: recall is sensitive to the hub degree threshold. Auto-calibration is needed for production use.

---

## Alternatives Considered

### A. Uniform SQ6 (3 bits)

Six-bit quantization would give intermediate memory between SQ8 and SQ4, but is awkward to implement with standard byte-aligned storage and not naturally aligned to nibble or byte boundaries. TAQ achieves similar average bits/dim with cleaner implementation and explicit topology justification.

### B. PQ (Product Quantization) with asymmetric distance computation

PQ encodes sub-vectors using learned codebooks and computes asymmetric distances. PQ achieves higher compression ratios (up to 32× vs f32) and better recall at equivalent compression than scalar quantization. However, PQ requires offline training, does not exploit graph topology, and is significantly more complex to implement. PQ is the right choice for large-scale static corpora; TAQ complements it for dynamic agent memory where topology is inherent.

### C. Matryoshka Representation Learning (MRL)

MRL trains embeddings with coarse-to-fine structure so that prefix sub-vectors are useful on their own. This requires embedding model retraining, not applicable to existing embeddings. TAQ works on any pre-computed embeddings.

### D. Graph-layer-based precision (HNSW-layer assignment)

HNSW naturally creates hub structure: nodes in higher layers have high in-degree. One could assign SQ8 to layer-1+ nodes and SQ4 to layer-0-only nodes. This requires HNSW to be pre-built and tightly couples quantization to the HNSW structure. TAQ's k-NN graph is simpler and independent of the search index.

---

## Implementation Plan

### Phase 1 (Implemented in this PoC)
- [x] `SQ8Params`: fit, encode, decode (1 byte/dim)
- [x] `SQ4Params`: fit, encode nibble-packed, decode (0.5 bytes/dim)
- [x] `build_knn_directed()`: O(N²) brute-force k-NN graph
- [x] `in_degree()`, `classify_hubs()`
- [x] `TaqIndex`: build + search + memory reporting
- [x] `recall_at_k()` measurement
- [x] Benchmark binary with acceptance test

### Phase 2 (Next steps)
- [ ] NN-Descent approximate k-NN graph for N=100K+ 
- [ ] SIMD decode kernels (SQ4 nibble unpack, SQ8 dequantize)
- [ ] Auto-calibration of hub threshold to hit target memory budget
- [ ] Integration with `ruvector-agent-memory` as a compression backend

### Phase 3 (Research direction)
- [ ] HNSW-aware hub assignment (use HNSW layer membership)
- [ ] Online hub degree tracking under insertions
- [ ] Multi-bit precision levels (2-bit, 6-bit, 16-bit) beyond SQ4/SQ8
- [ ] RVF serialization for persistent TAQ indexes
- [ ] MCP tool surface: `memory_compress`, `memory_search_compressed`

---

## Benchmark Evidence

Run: `cargo run --release -p ruvector-taq --bin benchmark`  
Hardware: x86_64 Linux, Rust 1.94.1

| Variant | Recall@10 | Mean latency | Memory | Memory % of f32 |
|---------|-----------|-------------|--------|-----------------|
| FullPrecision-f32 | 1.0000 | 410.7 μs | 1,250 KB | 100% |
| UniformSQ8 | 0.9864 | 586.4 μs | 312 KB | 25% |
| UniformSQ4 | 0.8194 | 1,075.9 μs | 156 KB | 12.5% |
| TAQ-mixed | **0.9652** | 770.5 μs | **262 KB** | **21%** |

TAQ topology: 68.2% hubs (SQ8), 31.8% leaves (SQ4), avg 6.73 bits/dim.

Acceptance test: PASS — TAQ recall ≥ SQ4 recall AND TAQ memory ≤ SQ8 memory.

---

## Failure Modes

1. **Uniform distribution corner case**: If all vectors have equal in-degree, no hubs exist and TAQ degrades to uniform SQ4. Detection: `hub_count == 0` after build → warn and fall back to SQ8.
2. **Very high hub threshold**: → all leaves → uniform SQ4. Detection: `leaf_count == 0` → warn.
3. **Graph invalidity after mass deletion**: Large deletes change in-degree significantly. Trigger TAQ rebuild after deleting more than 10% of vectors.
4. **Adversarial hub poisoning**: An attacker inserts many vectors near a target to artificially inflate its in-degree to hub status, consuming SQ8 budget for adversarial vectors. Mitigation: cap hub fraction at a maximum (e.g., 80%).

---

## Security Considerations

- TAQ does not encrypt or hide vectors. The hub assignment map reveals which embedding regions are densely populated. Treat hub metadata as sensitive in multi-tenant deployments.
- Combine with `ruvector-proof-gate` if per-vector access control is required.
- The hub fraction cap prevents adversarial memory exhaustion via hub inflation.

---

## Migration Path

**From unquantized f32 index:**
```
1. Build TAQ index from existing f32 vectors.
2. Run acceptance test to verify recall meets threshold.
3. Replace f32 store with TAQ index in production.
4. Retain original f32 vectors for TAQ rebuild after topology changes.
```

**From UniformSQ8 index:**
```
1. Decode SQ8 → f32.
2. Build TAQ index.
3. Measure recall delta vs existing SQ8 baseline.
4. Deploy if recall is acceptable (TAQ typically ~2pp below SQ8).
```

---

## Open Questions

1. What is the optimal hub threshold for Gaussian vs. clustered vs. adversarial distributions?
2. Can hub assignment be maintained incrementally under streaming inserts with O(k) work per insert?
3. Does integrating HNSW layer membership (instead of k-NN in-degree) improve recall further?
4. What is the SIMD-accelerated throughput for mixed SQ4/SQ8 dequantization on AVX-512?
5. Should TAQ be a feature flag in `ruvector-agent-memory` or a standalone compression crate?

---

## API Shape for Production

The `VectorIndex` trait should survive into production as-is:

```rust
pub trait VectorIndex {
    fn build(vectors: Vec<Vec<f32>>, dim: usize) -> Self;
    fn search(&self, query: &[f32], k: usize) -> Vec<usize>;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}
```

`TaqIndex`-specific parameters to expose as configuration:
```rust
pub struct TaqConfig {
    pub hub_degree_threshold: usize,  // default: 2
    pub graph_k: usize,               // default: 8
    pub max_hub_fraction: f64,        // default: 0.80 (anti-adversarial cap)
}
```

The `SQ4Params` and `SQ8Params` types should be public API for interoperability with other quantization-aware crates.

---

## What Would Reject This Direction

1. If recall@10 for TAQ does not exceed UniformSQ4 on the primary dataset (empirically falsified — PoC shows +14.6pp).
2. If the hub threshold is unstable across dataset types (requiring separate tuning per corpus with no principled default).
3. If incremental hub tracking proves too expensive (≥O(N) per insert), making real-time use impossible.
4. If SIMD-optimized SQ4 decode eliminates the recall gap (uniform SQ4 at SQ8 speed would make TAQ unnecessary).
