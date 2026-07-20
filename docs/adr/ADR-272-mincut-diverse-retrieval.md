# ADR-272: Partition-Aware Diverse ANN Retrieval (ruvector-diverse-retrieval)

**Status**: Proposed  
**Date**: 2026-07-16  
**Author**: Nightly Research Agent  
**Supersedes**: —  
**Related**: ADR-227 (Proof-Gated RAG), ADR-268 (Capability-Gated ANN), ADR-270 (RVM Coherence Domains)

---

## Context

Standard top-K ANN retrieval maximises relevance but provides no diversity
guarantee.  In clustered corpora—which dominate agent-memory, RAG, and
enterprise-search workloads—the k nearest vectors frequently all originate
from the same semantic neighbourhood, delivering redundant context to downstream
LLM calls.

Maximal Marginal Relevance (MMR, Carbonell & Goldstein 1998) partially addresses
this with a greedy λ-weighted score, but it treats all inter-candidate distances
as equally informative.  It does not model the graph structure of the candidate
pool, so it cannot guarantee that selected results come from distinct semantic
regions.

RuVector lacks a first-class **diversity-aware retrieval primitive**.  This ADR
proposes one.

---

## Decision

Add `crates/ruvector-diverse-retrieval` to the workspace.  The crate exposes
three implementations of a `DiverseRetriever` trait:

| Variant | Description |
|---------|-------------|
| `TopKRetriever` | Baseline: brute-force nearest-neighbour |
| `MmrRetriever` | Standard MMR with configurable λ |
| `PartitionMmrRetriever` | **MMR + graph-partition same-cluster penalty** |

### PartitionMMR Algorithm

1. Retrieve `POOL_FACTOR × k` candidates by L2 distance.
2. Estimate connectivity threshold `T = 0.55 × mean_pairwise_L2(full_pool)`.
3. Build a binary connectivity graph: connect candidates i, j if `L2(i,j) < T`.
4. Extract connected components via Union-Find (path compression + union-by-rank).
5. Greedy selection with modified score:

```
score(c) = −λ·dist(c,q) + (1−λ)·min_dist_to_selected
           − partition_penalty · same_partition_count(c, selected)
```

When `partition_penalty = 0` this reduces to standard MMR.
When `lambda = 1` this reduces to TopK ordering.

### Threshold Estimation: Full Pool Requirement

The threshold **must** be estimated from all C pool candidates, not a nearest-N
subset.  The nearest candidates are all from the same sub-cluster; sampling only
them yields a threshold too small to connect within-sub pairs (all singletons →
penalty never fires → PartitionMMR ≡ MMR).  Using the full pool produces a
bimodal distance distribution whose mean lies between intra- and inter-cluster
modes.

---

## Benchmark Results

Measured on x86_64 Linux, `cargo run --release`:

**Dataset**: 10 super-clusters × 6 sub-clusters × 50 vectors = 3,000 total,
64 dims, super_spread ±8.0, sub_spread σ=1.2, noise σ=0.25.

| Variant | Mean µs | QPS | MeanDiv | MeanRel |
|---------|---------|-----|---------|---------|
| TopK (baseline) | 271.3 | 3,686 | 2.574 | 2.234 |
| MMR (λ=0.5) | 443.8 | 2,253 | 5.696 | 4.017 |
| PartitionMMR | 725.8 | 1,378 | **8.377** | 6.207 |

**Acceptance tests (all PASS)**:
1. PartitionMMR diversity ≥ TopK × 1.15 → ratio **3.254** ✓
2. PartitionMMR diversity > MMR diversity → **8.377 > 5.696** ✓
3. MMR diversity > TopK diversity → **5.696 > 2.574** ✓
4. PartitionMMR rel ≤ MMR × 2.0 → ratio **1.545** ✓

---

## Consequences

### Positive

- **Measurable diversity improvement**: PartitionMMR delivers 3.25× more diverse
  results than TopK and 47% more than plain MMR on structured data.
- **Zero dependency on ML runtime**: O(C²D) deterministic computation, WASM-safe,
  no-std-compatible with `alloc`.
- **Composable**: Operates downstream of any candidate source (brute-force,
  HNSW, capability-gated ANN from ADR-268).
- **Tunable**: `partition_penalty` and `lambda` expose the diversity-relevance
  trade-off as first-class parameters.
- **Proof-gate compatible**: Can be integrated with ADR-227 so that diverse
  retrieval is a verifiable claim in proof-gated RAG pipelines.

### Negative / Trade-offs

- **Latency overhead**: PartitionMMR is 2.7× slower than TopK (725 µs vs 271 µs
  at C=60, D=64).  Acceptable for interactive use; potentially too slow for
  sub-millisecond requirements.
- **Relevance degradation**: Spreading across 6 partitions increases mean L2 to
  query by 1.55× vs MMR and 2.78× vs TopK.  This is the diversity-relevance
  trade-off and is expected; operators must tune `partition_penalty`.
- **Threshold sensitivity**: The `0.55 × mean_d` heuristic works well on
  Gaussian clusters.  Skewed or manifold-structured datasets may need adaptive
  thresholds.
- **Brute-force pool selection**: Candidate selection is O(ND).  Production use
  requires replacing this with HNSW beam search.

### Neutral

- The crate is standalone (no ruvector-core dep) and can be merged into
  `ruvector-coherence-hnsw` in a future PR without breaking changes.
- Unit tests cover all three retrievers (26 tests, all passing).

---

## Alternatives Considered

### A: Pure MMR (no partition layer)

Already implemented as `MmrRetriever`.  Benchmark shows it achieves 2.21×
TopK diversity — a meaningful improvement but less than PartitionMMR's 3.25×.

### B: Determinantal Point Processes

Theoretically superior diversity criterion but O(k³) per query — impractical
above k ≈ 50.  PartitionMMR achieves comparable practical diversity at O(C²D).

### C: Pre-computed cluster labels

Assign persistent cluster IDs to vectors at index time (e.g., k-means).
The partition penalty then uses stored labels with zero per-query graph cost.
Rejected for the PoC stage because it requires index pre-processing and
doesn't adapt to the query-local candidate distribution.  Recommended as a
production optimisation once PartitionMMR is validated on real workloads.

---

## Implementation Plan

1. **PoC** (this ADR): `crates/ruvector-diverse-retrieval` with brute-force
   pool selection and synthetic-data benchmark.  **Complete.**
2. **HNSW integration**: Replace brute-force pool selection with HNSW beam
   search from `ruvector-coherence-hnsw`.
3. **MCP tool**: Expose as `vector/search_diverse` with `diversity_mode` param.
4. **ruFlo integration**: Add `recall_diverse` step to ruFlo workflow library.
5. **WASM wrapper**: `ruvector-diverse-retrieval-wasm` for Cognitum Seed / browser.

---

## References

- Carbonell & Goldstein (1998). The use of MMR, diversity-based reranking for
  reordering documents and producing summaries. *SIGIR 1998*.
- Nightly research README: `docs/research/nightly/2026-07-16-mincut-diverse-retrieval/README.md`
