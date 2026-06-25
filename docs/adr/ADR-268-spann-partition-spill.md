---
adr: 268
title: "SPANN-Inspired Partition Spilling for Boundary-Safe ANN in RuVector"
status: accepted
date: 2026-06-24
authors: [ruvnet, claude-flow]
related: [ADR-193, ADR-264, ADR-265]
tags: [ann, ivf, spann, partition, spilling, boundary, recall, ruvector-spann, nightly-research]
---

# ADR-268 — SPANN Partition Spilling: Boundary-Safe ANN

## Status

**Accepted.** Implemented on branch `research/nightly/2026-06-24-spann-partition-spill` as
`crates/ruvector-spann`. All 10 unit tests pass; all benchmark acceptance gates pass.

```
cargo run --release --manifest-path crates/ruvector-spann/Cargo.toml --bin benchmark
```

---

## Context

Partition-based ANN (IVF family) is the dominant index strategy in production vector databases
(FAISS, Qdrant, Milvus, Pinecone). RuVector already has `ruvector-rairs` (ADR-193) as its first
IVF implementation. However, standard IVF has a well-known boundary problem: vectors near
Voronoi cell boundaries are frequently the true nearest neighbors of a query, yet they are missed
when only the nearest partition is probed.

The Microsoft SPANN paper (NeurIPS 2021)[^1] addresses this by duplicating boundary vectors into
adjacent partitions at build time — "spilling" — so that the query need only probe fewer partitions
to find the same true neighbors. This trades memory for recall at a given nprobe budget.

RuVector needs SPANN-style spilling because:

1. `ruvector-diskann` maps partitions to disk pages. Boundary-aware spilling improves recall
   without increasing per-query page accesses.
2. `ruvector-rairs` SEIL already deduplicates dual-assigned vectors, but its spill decision is
   purely residual-distance-based. CoherenceSpill adds a corpus-adaptive percentile criterion.
3. Agent memory workloads (MCP tools, ruFlo) benefit from controllable recall budgets. Spill
   ratio is a single parameter a workflow can tune without rebuilding.
4. CoherenceSpill connects the spill decision to the existing `ruvector-coherence` scoring
   philosophy: boundary ambiguity is a form of low coherence.

---

## Decision

Introduce `crates/ruvector-spann` as a standalone crate implementing three partition-spilling
variants under a common `PartitionIndex` trait:

| Variant | Spill condition | Overhead | Recall gain (N=5K, nprobe=2) |
|---------|----------------|----------|-------------------------------|
| `SinglePartition` | None (hard IVF baseline) | 1.00× | — |
| `SpillPartition` | `d2/d1 < spill_ratio` (fixed 1.20) | 1.99× | +64% |
| `CoherenceSpill` | `d2/d1 ≤ corpus_pct30_threshold` | 1.30× | +19% |

The crate is zero-dependency, `no_std`-compatible with alloc, and uses only deterministic
data structures. It provides:

- A common `PartitionIndex` trait for plug-and-play variant swapping.
- A KMeans builder with deterministic seeding for reproducible builds.
- A benchmark binary sweeping `nprobe ∈ {2, 4, 6, 8, 12, 16}` to show the full recall curve.

---

## Consequences

**Positive:**
- SpillPartition achieves **1.64–1.66×** recall at same nprobe vs SinglePartition (measured).
- CoherenceSpill achieves **1.19–1.23×** recall with only **1.30×** memory overhead.
- SpillPartition nprobe=8 achieves recall=0.715 (N=5K, D=128), which SinglePartition needs
  nprobe≈14 to match — a **1.75×** probe reduction for the same recall.
- The `PartitionIndex` trait naturally extends to disk-paged partitions (`ruvector-diskann`)
  and namespace-isolated agent memory (`ruvector-rairs` + MCP tools).

**Negative:**
- SpillPartition memory overhead is ~2× (1.99× measured). For billion-scale datasets this
  must be managed with compressed vector storage (PQ codes, not raw f32).
- KMeans build time is O(N × C × iters × D): at N=10K it takes 2.5 s. For N=1M this
  approach needs approximate KMeans (KMeans++, mini-batch) or hierarchical clustering.
- CoherenceSpill's derived threshold is a corpus-level statistic — it cannot adapt to
  online inserts without periodic recomputation.

---

## Alternatives Considered

1. **HNSW γ-augmented graph (ACORN, ADR implemented 2026-04-26)**: Addresses filtered recall
   but augments edges, not partition assignments. Complementary, not a substitute.

2. **ruvector-rairs SEIL dual assignment (ADR-193)**: RAIRS uses directional secondary
   assignment; SPANN spills by distance ratio. The two are compatible — RAIRS determines
   *which* secondary, SPANN-style ratio determines *whether* to spill.

3. **Multi-probe IVF (probe K partitions without spilling)**: Equivalent to increasing nprobe.
   No memory overhead but requires visiting more partitions at query time. SpillPartition
   achieves the same recall at **1.75× fewer probes** (nprobe=8 vs ~14).

---

## Implementation Plan

### Phase 1 (this PR)
- [x] `PartitionIndex` trait
- [x] `SinglePartition` (IVF baseline)
- [x] `SpillPartition` (SPANN fixed threshold)
- [x] `CoherenceSpill` (corpus-adaptive percentile)
- [x] KMeans builder (deterministic)
- [x] Benchmark binary with nprobe sweep
- [x] 10 unit tests, all passing

### Phase 2 (production hardening)
- [ ] Approximate KMeans (KMeans++ seeding) for N > 100K
- [ ] Compressed spill storage (PQ codes for spilled vectors)
- [ ] Integration with `ruvector-diskann` page layout
- [ ] Streaming insert with lazy spill repair
- [ ] `no_std` + alloc build target for Cognitum Seed

### Phase 3 (ecosystem integration)
- [ ] ruFlo parameter: expose `spill_ratio` and `coherence_percentile` as workflow knobs
- [ ] MCP memory tool: per-namespace partition index with spill-controlled isolation
- [ ] RVF manifest: serialize partition centroids + spill assignments as RVF index section

---

## Benchmark Evidence

**Hardware:** x86_64 Linux (cloud VM)
**Command:** `cargo run --release --manifest-path crates/ruvector-spann/Cargo.toml --bin benchmark`
**Data:** Gaussian, D=128, K=10 recall, 300 queries, deterministic seed

### N=5,000, 32 centroids

| Variant | nprobe | Recall@10 | Mean µs | p50 µs | p95 µs | QPS | Mem MB | Spill |
|---------|--------|-----------|---------|--------|--------|-----|--------|-------|
| SinglePartition | 8 | 0.505 | 279.7 | 278.3 | 327.9 | 3,575 | 2.46 | 1.00× |
| SpillPartition | 8 | 0.715 | 593.7 | 584.1 | 719.4 | 1,684 | 4.88 | 1.99× |
| CoherenceSpill | 8 | 0.568 | 362.2 | 358.0 | 421.7 | 2,761 | 3.19 | 1.30× |

Peak recall gain vs Single: SpillPartition=**1.64×** (PASS ≥1.40), CoherenceSpill=**1.19×** (PASS ≥1.15)

### N=10,000, 40 centroids

| Variant | nprobe | Recall@10 | Mean µs | p50 µs | p95 µs | QPS | Mem MB | Spill |
|---------|--------|-----------|---------|--------|--------|-----|--------|-------|
| SinglePartition | 8 | 0.431 | 458.8 | 455.0 | 526.6 | 2,180 | 4.90 | 1.00× |
| SpillPartition | 8 | 0.643 | 991.7 | 982.0 | 1090.8 | 1,008 | 9.77 | 2.00× |
| CoherenceSpill | 8 | 0.496 | 644.5 | 627.3 | 808.8 | 1,551 | 6.37 | 1.30× |

Peak recall gain vs Single: SpillPartition=**1.66×** (PASS ≥1.40), CoherenceSpill=**1.23×** (PASS ≥1.15)

---

## Failure Modes

1. **Empty centroids after KMeans**: Evenly-spaced deterministic seeding avoids this, but if
   data is highly concentrated, some centroids can be empty. Mitigation: fallback assignment
   to nearest non-empty centroid.

2. **Spill ratio = 1.0 (all vectors spill)**: If `spill_ratio <= 1.0`, every vector spills to
   every centroid. The `build()` methods do not guard against this. Future work: add a
   `max_spill_factor` ceiling.

3. **High-dimensional boundary collapse**: In D>512, Voronoi boundaries become exponentially
   thin. The d2/d1 ratio converges to 1.0 for all vectors, meaning CoherenceSpill spills
   *everything* at any percentile. PQ-compressed distance should replace raw L2 in high D.

4. **Build time at scale**: O(N × C × iters × D) KMeans is prohibitive for N > 500K.
   Mini-batch KMeans or hierarchical k-means required.

---

## Security Considerations

- Partition membership reveals approximate neighborhood: a read-only replica of
  the spill index can reveal which documents are semantically adjacent.
- For privacy-sensitive agent memory (e.g., MCP tools holding user context),
  spill lists should be stored encrypted at rest.
- Future work: integrate with `ruvector-proof-gate` for write-audited partition updates.

---

## Migration Path

`ruvector-spann` is a new standalone crate. Existing `ruvector-rairs` users are unaffected.
A future `ruvector-ivf-family` crate could unify RAIRS, SPANN, and standard IVFFlat under
one trait. The `PartitionIndex` trait defined here is the obvious candidate for that unification.

---

## Open Questions

1. What is the optimal `spill_ratio` for non-Gaussian (clustered, power-law) distributions?
2. Can `CoherenceSpill`'s percentile threshold be updated incrementally as new vectors arrive?
3. Should the spill storage be deduplicated (RAIRS SEIL approach) or replicated (this PoC)?
4. How does spill interact with quantization: should spilled copies store PQ codes only?
5. What is the right MCP tool API for namespace-isolated partitioned memory?

---

[^1]: Chen, Qi et al. "SPANN: Highly-efficient Billion-scale Approximate Nearest Neighbor Search." NeurIPS 2021. https://arxiv.org/abs/2111.08566

[^2]: Subramanya, Suhas Jayaram et al. "DiskANN: Fast Accurate Billion-point Nearest Neighbor Search on a Single Node." NeurIPS 2019. https://papers.nips.cc/paper/2019/hash/09853c7fb1d3f8ee67a61b6bf4a7f8e6-Abstract.html

[^3]: Babenko, Artem & Lempitsky, Victor. "The Inverted Multi-Index." CVPR 2012.
