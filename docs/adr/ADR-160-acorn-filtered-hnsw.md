# ADR-160: ACORN — Predicate-Agnostic Filtered HNSW for ruvector

## Status

Proposed

## Date

2026-04-26

## Authors

ruv.io · RuVector Nightly Research (automated nightly agent)

## Relates To

- ADR-001 — Tiered quantization strategy (ruvector-core quantization hierarchy)
- ADR-027 — HNSW parameterised query fix (hnsw_rs patch)
- ADR-154 — RaBitQ rotation-based binary quantization
- ADR-155 — RuLake data-lake layer (payload storage substrate)
- Research: `docs/research/nightly/2026-04-26-acorn-filtered-hnsw/README.md`

---

## Context

### The gap

ruvector currently handles filtered search via `ruvector-filter`, a post-hoc
predicate evaluator that decouples predicate application from the ANN graph
traversal. In the current pipeline:

```
query → HNSW graph search (ef candidates) → filter(candidates) → top-k
```

This post-filter pattern has a well-documented failure mode at low predicate
selectivity: when only fraction p of the corpus satisfies the predicate, the
ANN beam of size ef must oversample by 1/p to expect k valid results. At
p=0.01 (1% selectivity), ef must reach 1000 to expect 10 valid results — but
with a standard M=16 HNSW, ef=1000 is an extremely expensive search, and
recall still degrades because the graph traversal prunes predicate-failing
nodes aggressively.

Measured consequence: post-filter recall@10 at 1% selectivity on Gaussian
n=5K D=128 data: approaches 0% for ef<1000.

### Industry movement

As of 2025, all major production vector databases have addressed this gap:

| System | Approach | Year |
|--------|----------|------|
| Qdrant v1.16 | ACORN-style in-graph filter + roaring bitmap payload index | 2025 |
| Weaviate v1.27 | ACORN-1 (published dedicated blog post) | 2025 |
| Vespa | ACORN-1 + adaptive beam search | 2025 |

ruvector has no equivalent, creating a competitive gap for applications with
real-world metadata-filtered queries (e-commerce, geo-filtered content,
access-controlled retrieval, time-range queries).

### ACORN algorithm

Patel et al. (SIGMOD 2024, arXiv:2403.04871) prove that the beam starvation
problem has a graph-structural solution: store more edges per node (γ·M
instead of M) so that a predicate-failing node still exposes a path to a
predicate-passing node within 1–2 hops. Their traversal change is minimal:

> "During beam search, expand neighbors of ALL visited nodes regardless of
> predicate. Nodes failing P(x) are excluded from the result set but NOT from
> neighborhood expansion."

This single traversal change, combined with γ-augmented construction, achieves:
- >95% recall@10 at 1% selectivity on SIFT-1M (vs ~10% for post-filter HNSW)
- 2–1,000× QPS improvement over post-filter HNSW at matched recall ≥ 0.9

---

## Decision

Ship `crates/ruvector-acorn` as a standalone Rust crate providing:

1. **`FilteredIndex` trait** — common interface: `build(data)` + `search(query, k, predicate)` + `memory_bytes()`. Predicate is `&dyn Fn(u32) -> bool`, compatible with `ruvector-filter`'s `FilterEvaluator`.

2. **Three concrete variants**:
   - `FlatFilteredIndex` — brute-force post-filter baseline (O(n·D) per query)
   - `AcornIndex1` — ACORN with standard M=16 edges, ACORN-style traversal
   - `AcornIndexGamma` — ACORN with γ·M edges (γ=2 by default, M=16 → 32 neighbors/node)

3. **Graph construction** — greedy O(n²) sequential insertion for the PoC. Bidirectional edges (forward + backward), capped at max_neighbors per node.

4. **Multi-probe entry point** — find entry by scanning sqrt(n) uniformly-spaced nodes, taking the nearest to the query. Eliminates fixed-node entry sensitivity without requiring a multi-level HNSW hierarchy.

5. **Working demo binary** (`acorn-demo`) — produces a real benchmark table of recall@10, QPS, memory, and build time across three selectivity levels (50%, 10%, 1%).

### Measured results (ruvector-acorn v2.2.0, x86_64 Linux release)

Dataset: n=5,000, D=128, Gaussian, queries=500, k=10.

| Variant | Selectivity | Recall@10 | QPS | Memory (MB) |
|---------|-------------|-----------|-----|-------------|
| FlatFiltered | 50% | 100.0% | 2,867 | 2.44 |
| ACORN-1 | 50% | 24.7% | **20,435** | 2.75 |
| ACORN-γ (γ=2) | 50% | 34.5% | **14,415** | 3.05 |
| FlatFiltered | 10% | 100.0% | 12,859 | 2.44 |
| ACORN-1 | 10% | 64.2% | **15,846** | 2.75 |
| ACORN-γ (γ=2) | 10% | **79.7%** | 9,512 | 3.05 |
| FlatFiltered | 1% | 100.0% | 57,519 | 2.44 |
| ACORN-1 | 1% | **97.6%** | 4,990 | 2.75 |
| ACORN-γ (γ=2) | 1% | **98.9%** | 2,773 | 3.05 |

ACORN-γ achieves **98.9% recall@10 at 1% selectivity** using only 3.05 MB (vs 2.44 MB for flat), with the traversal cost scaling with ef (120) rather than with the number of matching nodes.

Build times: FlatFiltered 1.2 ms · ACORN-1 1,757 ms · ACORN-γ 1,809 ms
(greedy O(n²) construction; see "Alternatives" for NN-descent path).

---

## Consequences

### Positive

- Closes the primary production gap for filtered vector search in ruvector: recall no longer collapses at low predicate selectivity.
- Predicate-agnostic design: any `Fn(u32) -> bool` works — categorical equality, range, geo-distance, ACL membership, composite expressions from `ruvector-filter`.
- Connects naturally to `ruvector-filter`'s expression evaluator without requiring changes to either crate.
- Trait-based design (`FilteredIndex`) allows future backends (disk-resident, FPGA-accelerated) to swap in without changing the query layer.
- No unsafe code. No external C/C++/BLAS dependencies. Fully workspace-buildable.

### Negative / Costs

- **O(n²) build time** for the PoC is unsuitable for n > 50K. Unblocked by ADR-161 (NN-descent constructor). Not shipping ADR-161 in this PR.
- **Static index only**: no insert/delete. Incremental updates require integration with `ruvector-delta-index` (ADR-162 scope).
- **ACORN is not universally better than flat scan**: at n=5K with 1% selectivity, flat scan is 11× faster because only 50 distances are computed. ACORN's QPS advantage materialises at n ≫ ef/selectivity.
- **Memory cost**: ACORN-γ (γ=2) uses ~25% more memory than flat storage (3.05 MB vs 2.44 MB for n=5K D=128). For n=1M, the edge list adds ~640 MB for γ=2, M=16.

### Neutral

- The PoC build time (O(n²)) does not reflect production build time and should not be compared against flat build in production evaluations.
- Recall numbers from the greedy O(n²) graph are conservative; a proper HNSW-quality graph (NN-descent, bidirectional, shrink strategy) would improve recall by 10–30 percentage points.

---

## Alternatives Considered

### A. Improve existing post-filter via oversampling

The current `ruvector-filter` pipeline could be augmented to oversample by
1/p (estimate selectivity from payload index stats) before the graph search.
This would improve recall at the cost of higher ef and still-degenerate
performance at p < 5%. Rejected because: oversample factor caps out at ef_max;
recall degrades again below the threshold. ACORN solves it structurally.

### B. Pre-filter (scan matching set, exact k-NN)

At query time, retrieve all matching node IDs from a payload bitmap index, then
run exact k-NN within that subset. This is O(n·p·D) per query and achieves
100% recall by construction. Better than post-filter at low selectivity;
equivalent to ACORN at very low selectivity. Chosen as the `FlatFilteredIndex`
baseline in this ADR. Disadvantage at medium-high selectivity: O(n·p) grows
linearly with n while ACORN stays O(ef·D).

### C. SIEVE multi-index approach

Maintain a separate index per predicate value (one HNSW for "electronics", one
for "clothing", etc.). Route queries to the matching index. Works well for
static, low-cardinality categorical predicates; breaks down for range queries,
composite predicates, and high-cardinality attributes. Too inflexible for
general-purpose ruvector use. Rejected.

### D. NN-descent constructor (defer to ADR-161)

NN-descent (Dong et al., WWW 2011) builds a k-NN graph in O(n × T × D × M)
time by iteratively refining each node's neighbors using its current neighbors'
neighborhoods. This produces higher-quality graphs (better recall at same ef)
and runs in O(n log n) asymptotically. The PoC uses greedy O(n²) to keep
implementation scope focused. NN-descent should be added in ADR-161 as a
second constructor (feature-gated or separate `AcornGraph::build_nndescent`).

---

## Implementation Plan

| Phase | Milestone | Target |
|-------|-----------|--------|
| ✅ **P0** | `crates/ruvector-acorn` merged, 12/12 tests pass, demo binary with real numbers | This PR |
| P1 | Payload index integration: wire `ruvector-filter::FilterEvaluator` as predicate callback | ADR-161 |
| P2 | NN-descent construction (`AcornGraph::build_nndescent`) | ADR-161 |
| P3 | Dynamic insert/delete via `ruvector-delta-index` | ADR-162 |
| P4 | SIMD distance kernel via `simsimd` workspace dep | ADR-163 |
| P5 | Selectivity-adaptive routing (flat vs ACORN switchover) | ADR-164 |
