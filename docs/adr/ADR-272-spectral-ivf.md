# ADR-272: Spectral IVF — Graph-Laplacian Partitioned ANN

**Status**: Proposed  
**Date**: 2026-07-20  
**Deciders**: RuVector nightly research  
**Supersedes**: —  
**Related**: ADR-193 (RAIRS), ADR-268 (SPANN, capability-gated ANN)

---

## Context

RuVector's IVF-based indexes (`ruvector-rairs`, `ruvector-spann`) all use k-means as their partitioning primitive. k-means assigns each vector to its nearest Euclidean centroid. The partition boundaries are Voronoi cells, which are isotropic in L2 space but do not respect the **semantic neighbourhood topology** of the embedding space.

The consequence: vectors that are mutual nearest neighbours in the embedding space may fall into different Voronoi cells if they happen to be equidistant from two centroids. At low `nprobe`, these vectors go unvisited and recall suffers.

SPANN (ADR-268) and RAIRS (ADR-193) both address this post-hoc by duplicating boundary vectors into adjacent cells. Spectral IVF addresses it at construction time: the partition boundaries themselves are drawn along **minimum graph cuts**, so neighbourhood-connected vectors are grouped together by definition.

The Fiedler vector — the second eigenvector of the graph Laplacian L = D − W — is the continuous relaxation of the minimum balanced graph cut. Partitioning by its sign produces two groups that minimise inter-group edge weight, where edge weight encodes semantic similarity (cosine similarity between vectors).

---

## Decision

Add `ruvector-spectral-ivf` as a new standalone crate providing three IVF variants under a shared `AnnIndex` trait:

1. **KMeansIvf** — baseline Lloyd's k-means IVF (pure Rust, no external deps)
2. **SpectralIvf** — recursive Fiedler bisection with cosine-similarity kNN edges
3. **CoherenceSpectralIvf** — Fiedler bisection with cosine²-weighted edges (coherence emphasis)

The crate is pure Rust, zero unsafe, WASM-compatible, and has no external dependencies.

---

## Consequences

### Positive

- Recall@10 improves from 0.801 (k-means) to 1.000 (SpectralIvf) and 0.990 (CoherenceSpectralIvf) on the benchmark corpus (n=800, dim=64, 8 partitions, nprobe=4)
- Zero memory overhead vs. k-means IVF: same partition layout, same representative storage
- Natural alignment with RuVector's mincut infrastructure (`ruvector-mincut`, `ruvector-coherence`)
- Partition labels are deterministic: same seed → same partitions → reproducible
- WASM-compatible: query path makes no OS calls
- Sets foundation for coherence-domain-aligned storage (RVM, Cognitum Seed)

### Negative

- Build time: 90ms for n=800 (vs. <1ms for k-means). Build is O(n² × k × iters); scales poorly to n > 100k without approximate kNN construction.
- Query latency: ~31µs vs. ~18µs for k-means at same nprobe (1.7× slower due to cosine distance for probing instead of L2 squared)
- The synthetic benchmark is favourable: on real high-dimensional data (768-dim, uniform manifold), recall improvement may be smaller

---

## Alternatives Considered

### 1. SPANN spilling (ADR-268)

Duplicates boundary vectors into adjacent cells. Addresses the boundary problem at the cost of storage overhead (1.5–2× memory). Spectral IVF avoids wrong assignments at construction time without storage overhead.

**Why not**: Already implemented. Spectral IVF is complementary (different failure mode).

### 2. RAIRS dual assignment (ADR-193)

Assigns each vector to primary + secondary cell. Recall improvement similar to SPANN. Also storage overhead.

**Why not**: Already implemented. Same reasoning as SPANN.

### 3. Spectral clustering (full)

Run k-means on the top-m Fiedler eigenvectors (Ng et al., 2001). Better partition quality for non-bisectable topologies. Requires computing m eigenvectors, not just one.

**Why not tonight**: O(n × m) eigenvectors needed; m × power_iteration cost. Planned as v2.

### 4. METIS-style multilevel partitioning

Iteratively coarsen the graph, find partition at coarsest level, then refine. Industry-standard for graph partitioning (Karypis & Kumar, 1998).

**Why not**: Complex implementation (400+ lines). Fiedler bisection gives >99% recall on PoC corpus; marginal benefit doesn't justify complexity for a PoC.

### 5. Learned partitioning

Neural model trained to predict partition assignment from vector content. Potentially superior partition quality on real data. Requires training data and inference infrastructure.

**Why not**: Violates "no external service dependency" constraint; adds training complexity. Out of scope for nightly PoC.

---

## Implementation Plan

### Phase 1 (This PoC — done)
- [x] `AnnIndex` trait with `build / search / memory_bytes / name`
- [x] `KMeansIvf`: Lloyd's k-means, L2 centroids, L2 probing
- [x] `SpectralIvf`: cosine-similarity kNN, Fiedler bisect, mean representatives, cosine probing
- [x] `CoherenceSpectralIvf`: cosine²-weighted edges, same bisect
- [x] 15 unit tests, all passing
- [x] Benchmark binary with real measured numbers
- [x] Acceptance test: recall@10 ≥ 0.60 for all three variants

### Phase 2 (Production hardening)
- [ ] Approximate kNN construction via HNSW (remove O(n²) bottleneck)
- [ ] Parallelise graph build with `rayon`
- [ ] Evaluate on real datasets: SIFT-128, glove-100, text-embedding-3-small
- [ ] Recall vs. nprobe curves at n=100k
- [ ] WASM serialisation (index → bytes → deserialise at edge)
- [ ] Integration with `ruvector-server` as an IVF backend option

### Phase 3 (Future research)
- [ ] Streaming Fiedler updates via Lanczos restarts
- [ ] Coherence domain mapping to RVM
- [ ] MCP memory tool wrapping per-partition namespaces
- [ ] ruFlo trigger for partition quality monitoring and auto-rebuild

---

## Benchmark Evidence

All measurements from `cargo run --release -p ruvector-spectral-ivf --bin benchmark`.

**Environment**: Linux x86_64, Rust 1.94.1, kernel 6.18.5

| Variant | Build(ms) | Mean(µs) | p50(µs) | p95(µs) | QPS | Recall@10 | Mem(KB) |
|---------|-----------|---------|---------|---------|-----|-----------|---------|
| KMeansIvf | 0 | 18.3 | 17.4 | 23.1 | 54,704 | 0.801 | 208.2 |
| SpectralIvf | 90 | 31.3 | 30.2 | 38.2 | 31,933 | 1.000 | 208.2 |
| CoherenceSpectralIvf | 92 | 30.9 | 30.0 | 37.9 | 32,400 | 0.990 | 208.2 |

Dataset: n=800, dim=64, 8 partitions, nprobe=4, 200 queries.

**Acceptance**: All variants recall@10 ≥ 0.60. ✓

---

## Failure Modes

1. **O(n²) build**: Unacceptable at n > 100k. Mitigation: approximate kNN via HNSW.
2. **Fiedler non-convergence**: Degenerate graphs with no clear bisection. Mitigation: fall back to k-means when λ₂ < ε.
3. **Empty partitions**: Degenerate bisection places all vectors on one side. Mitigation: enforce minimum partition size; split at sorted index if Fiedler degenerates.
4. **High-dimensional recall regression**: On 1536-dim LLM embeddings, cosine similarities concentrate; kNN graph becomes nearly regular; Fiedler bisection loses information. Mitigation: normalise before constructing the graph; evaluate on real data before promoting to production.

---

## Security Considerations

- No external network calls in any code path
- No `unsafe` code (enforced by `#![forbid(unsafe_code)]`)
- Deterministic: same corpus + seed → same partition labels (auditable)
- Partition labels could be combined with proof-gated writes (ADR-227) to enforce per-domain write access: a vector can only be inserted into the partition its Fiedler label assigns it to

---

## Migration Path

This crate is additive. Existing users of `ruvector-rairs` and `ruvector-spann` are unaffected. To migrate:

```rust
// Before (k-means IVF)
let mut idx = ruvector_spann::SinglePartition::new(8);
idx.build(&corpus);

// After (spectral IVF — same trait surface)
let mut idx = ruvector_spectral_ivf::SpectralIvf::new(8, 10);
idx.build(&corpus);
```

The `AnnIndex` trait is compatible with the `PartitionIndex` trait in `ruvector-spann` modulo `memory_bytes` vs `total_assignments`. A trivial adapter resolves this.

---

## Open Questions

1. Does CoherenceSpectralIvf outperform SpectralIvf on real embedding datasets (not just synthetic)?
2. At what n does the O(n²) build cost become unacceptable in practice (estimated: n > 50k)?
3. Should `nprobe` selection use cosine distance to representatives (current) or Fiedler-space distance (more principled)?
4. Is there a streaming variant of power iteration that can update the Fiedler vector in O(k) time after a single insertion?
5. Should the `AnnIndex` trait be promoted to a shared workspace trait usable by `ruvector-rairs`, `ruvector-spann`, and `ruvector-spectral-ivf` alike?
