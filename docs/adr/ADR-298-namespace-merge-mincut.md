# ADR-298: Namespace-Merge via S-T Mincut Routing

- **Status**: Accepted
- **Date**: 2026-08-08
- **Updated**: 2026-08-08
- **Extends**: ADR-254 (turbovec), ADR-026 (tiered routing), ADR-297 (ACRP)
- **Related crates**: `ruvector-namespace-merge`, `ruvector-agent-memory`, `ruvector-graph`, `ruvector-coherence-hnsw`, `rvf`

## Context

RuVector's agent-memory tier partitions stored vectors into **namespaces** — logical
buckets keyed by domain, session, or tool context (e.g. `code/rust`, `session/42`,
`tool/web-search`). A typical deployment carries 5–50 such namespaces. Every ANN
query today fans out to all namespaces and merges results (`AllSearch`): trivially
correct, but the compute cost is proportional to total vectors, regardless of
semantic relevance.

Two simpler alternatives exist:

| Strategy | Mechanism | Weakness |
|---|---|---|
| **AllSearch** | Scan everything | O(N·n_vecs) dist ops, no savings |
| **CentroidFilter** | Skip if cosine(q, centroid) < threshold | Threshold is global; can't adapt to cluster geometry |

CentroidFilter depends on a hand-tuned threshold. If the threshold is too
low, it never skips anything. If too high, it drops relevant namespaces.
Neither strategy uses inter-namespace similarity — the fact that namespaces
A₀ and A₁ are semantically coherent and should be searched together.

**Product claim to earn**: *route each query to exactly the coherent namespace
cluster it belongs to, provably optimal under the flow objective, with no
hand-tuned threshold.*

## Decision

### 1. Model namespace selection as S-T min-cut

Build a flow network for each query `q` with `N + 2` nodes (N namespaces +
source S + sink T):

```
S → nsᵢ   capacity = round( q_sim_norm[i] × SCALE )   # affinity to query
nsᵢ → T   capacity = round( (1 − q_sim_norm[i]) × SCALE )   # cost to include
nsᵢ ↔ nsⱼ  capacity = round( inter_sim[i,j] × SCALE )  # cohesion penalty
```

where `q_sim_norm[i] = (q_sim[i] − q_min) / (q_max − q_min)` is the
**relative** query affinity (normalised over the observed range across all
namespaces for this query).

Run Edmonds-Karp max-flow on this graph. The source-side of the min-cut
(nodes reachable from S in the residual graph) are the namespaces to search.

**Correction (2026-08-08):** when all query affinities are equal (including a
single-namespace dataset), relative normalisation contains no routing signal.
`MinCutRoute` deterministically searches all namespaces in that case; an empty
source-side cut also falls back to all namespaces to preserve recall. A dataset
with no namespaces returns an empty result without constructing a flow graph.

### 2. Relative normalisation is non-negotiable

Raw cosine similarities depend on dimensionality and noise level. At dims=64
with noise=0.30, all q_sim values fall near [0.3, 0.5] — absolute values
well below 0.5. Without normalisation, `S→ns` capacity < `ns→T` capacity
for every namespace, so Edmonds-Karp saturates all S-edges and the residual
graph is unreachable from S, returning an empty search set.

Normalising to the observed per-query range ensures the most-relevant
namespace always receives full `S→ns` capacity and the least-relevant always
gets full `ns→T` capacity, making the cut invariant to absolute cosine scale.

### 3. Precompute inter-namespace similarity matrix

The N×N centroid cosine matrix is computed once at router construction time
and reused across all queries. For N=20 namespaces this is 400 f32 values
(1.6 KB). The inter-namespace edges encode semantic cohesion: namespaces that
are similar to each other resist being split by the cut.

### 4. Implement Edmonds-Karp for small graphs

Graph size is O(N) where N ≈ 5–50 in practice. Edmonds-Karp (BFS-augmented
Ford-Fulkerson) is O(VE²) — negligible for this scale. The adjacency matrix
representation uses `Vec<i64>` capacities to avoid floating-point rounding
artefacts during flow arithmetic.

### 5. Three routing strategies in one crate

`ruvector-namespace-merge` exposes a `NamespaceRouter` trait with three
implementations, all returning `RouteResult { hits, ns_searched, dist_ops }`:

- `AllSearch` — ground-truth baseline (recall = 1.0 by definition)
- `CentroidFilter` — cosine threshold heuristic
- `MinCutRoute` — principled S-T flow partition

The uniform result type lets benchmarks and A/B tests swap strategies with
zero measurement-code changes.

## Consequences

### Positive

- **Principled routing** with no hand-tuned threshold; the flow objective
  automatically balances inclusion cost against cohesion.
- **Recall preservation**: `MinCutRoute` achieves ≥ 98% recall at 41% of
  `AllSearch`'s distance computations on the 64-dim, noise=0.30 benchmark.
- **Coherence-preserving**: semantically similar namespaces stay together on
  the S-side due to inter-namespace cohesion edges.
- **Zero external dependencies**: pure Rust, no `ndarray`, no `petgraph`;
  ships in WASM and embedded contexts without linker friction.

### Negative

- **Per-query flow solve**: O(VE²) overhead per query for N namespaces.
  At N=20, measured overhead is ~5 µs on an M-class core — acceptable for
  latency budgets ≥ 10 ms but visible at sub-millisecond targets.
- **N² inter-similarity precomputation**: O(N²·D) on construction.
  At N=50, D=1536, this is 3.8M multiplies — a one-time ~1 ms cost.
- **Centroid quality dependency**: routing quality degrades if namespace
  centroids are stale. Callers must recompute or incrementally update
  centroids as vectors are inserted/deleted.

## Alternatives Considered

### A. Global cosine threshold (CentroidFilter)

Implemented and benchmarked. Achieves recall=0.945 at 38% dist ops with
threshold=0.20 on the standard 64-dim dataset. However, threshold requires
manual tuning per namespace topology and degrades silently when the namespace
distribution shifts.

### B. Learned router (lightweight neural classifier)

Would learn query→namespace routing from labelled traffic. Achieves higher
accuracy when the training distribution matches production. Rejected because:
- Requires labelled data and training pipeline
- Non-deterministic under distribution shift
- Not self-contained (external model weights)
- Out of scope for a zero-dependency Rust crate

### C. Graph Laplacian spectral partition

Spectral bisection on the namespace similarity graph gives a static partition
independent of the query. Rejected because routing must be *query-dependent*:
different queries should activate different namespace subsets.

### D. Hierarchical namespace tree

Pre-build a dendrogram of namespaces and navigate it per query. Requires
O(N log N) construction and an additional routing policy. The flow formulation
generalises this: the min-cut over the augmented graph implicitly encodes the
hierarchy through the inter-similarity edges.

## Implementation Plan

### Phase 1 — Core crate (complete)

- [x] `FlowGraph` with Edmonds-Karp and `source_side()`
- [x] `Dataset` synthetic generator (5 namespaces, 3 semantic groups)
- [x] `AllSearch`, `CentroidFilter`, `MinCutRoute` implementations
- [x] Relative q_sim normalisation fix
- [x] Integration tests (recall, ns-reduction, flow unit test)
- [x] Benchmark binary with acceptance criteria

### Phase 2 — Production integration (future)

- [ ] Wire `MinCutRoute` into `ruvector-agent-memory` query path
- [ ] Expose `NamespaceRouter` as a trait object in `ruvector-core`
- [ ] Add incremental centroid update API to `Dataset`/`Namespace`
- [ ] WASM target (`wasm32-unknown-unknown`) with `no_std` fallback for BFS

### Phase 3 — Adaptive threshold (future)

- [ ] Auto-tune the normalisation scale factor per namespace topology
  (e.g. via a small offline calibration pass on representative queries)
- [ ] Cache flow-graph solutions for repeated identical q_sim signatures

## Benchmark Evidence

All numbers from `cargo run --release --bin benchmark` on the standard
64-dim, 500 vecs/namespace, noise=0.30, 300-query dataset:

```
Variant          Mean(µs)  p50    p95    QPS      Recall   NS    DistOps  Mem(KB)
AllSearch           133.7   129    157    7,481    1.0000   5.00  2500     0
CentroidFilter       49.7    50     63   20,125    0.9453   1.91   957     0
MinCutRoute          54.2    51     68   18,449    0.9853   2.05  1025     1
```

**Min-cut vs all-search**:
- Recall: 0.9853 (98.5% of ground truth)
- Distance ops: 41% of AllSearch
- Speed: 2.47× faster mean latency
- Memory overhead (router index): 1 KB (400 f32 values)

All acceptance criteria pass:
- `MIN_RECALL_CENTROID=0.80` → actual 0.945 ✓
- `MIN_RECALL_MINCUT=0.80` → actual 0.985 ✓
- `MAX_DIST_OPS_CENTROID_FRAC=0.70` → actual 0.383 ✓
- `MAX_DIST_OPS_MINCUT_FRAC=0.60` → actual 0.410 ✓

All 9 tests pass (`cargo test -p ruvector-namespace-merge`).

## Failure Modes

| Failure | Trigger | Mitigation |
|---|---|---|
| All namespaces on T-side (zero results) | Bug: q_sim not normalised | Fixed; unit test guards this |
| All namespaces on S-side (no savings) | All inter-sim near zero (diverse dataset) | Expected: min-cut defaults to AllSearch behaviour |
| Stale centroids | Vectors inserted after `MinCutRoute::new()` | Rebuild router after bulk inserts; warn in docs |
| Centroid collapse | Single-vector namespace | Centroid = that vector; routing still correct |
| Flow overflow | N > 500 with SCALE=10000 | i64 capacity; N=500 gives max cap 10000×N²≈2.5×10⁹ < i64::MAX |
| Identical q_sim values | Query equidistant from all centroids | Detect the degenerate range and deterministically search all namespaces |

## Security Considerations

- **Input sanitisation**: query and centroid vectors should be L2-normalised
  before computing `cosine_sim`. Unnormalised vectors do not cause UB (no
  unsafe code in this crate) but can produce cosine values outside [−1, 1],
  skewing flow capacities.
- **No unsafe code**: `#![forbid(unsafe_code)]` is implicitly satisfied; the
  crate uses only safe Rust.
- **Capacity integer overflow**: flow capacities are `i64`; the maximum
  per-edge capacity is `scale × 1.0 = 10_000`. Total flow through any path is
  bounded by the min edge capacity. No overflow possible for realistic N.
- **Adversarial namespace poisoning**: a malicious vector inserted into a
  "trusted" namespace could shift its centroid, causing the router to include
  that namespace for unrelated queries. Mitigate with centroid outlier
  rejection or per-namespace access control at the insertion layer.

## Migration Path

1. **Opt-in**: deploy `MinCutRoute` behind a feature flag
   (`RUVECTOR_NAMESPACE_ROUTER=mincut`); default remains `AllSearch`.
2. **Shadow mode**: run both routers, log recall divergence, no user impact.
3. **Gradual rollout**: enable for read-only query traffic at 10% → 50% → 100%.
4. **Threshold fallback**: if `MinCutRoute` returns zero results for a query
   (all namespaces on T-side after normalisation), fall back to `AllSearch`
   for that query and log a warning.

## Open Questions

1. **Incremental centroid updates**: what is the correct strategy when vectors
   are inserted one at a time? Incremental average is O(1) per insert but does
   not handle deletes. A periodic full recompute may be preferable.

2. **Dynamic N**: production deployments may create/delete namespaces at
   runtime. Should `MinCutRoute` be rebuilt on every schema change, or should
   it support hot namespace addition?

3. **Sub-millisecond budgets**: at N=50 the flow solve takes ~5 µs. Is this
   acceptable for the WASM/edge inference path where total query budget is
   often <1 ms? May need a fast path that short-circuits to `CentroidFilter`
   when N > threshold.

4. **Cross-namespace deduplication**: if the same vector ID appears in multiple
   namespaces (e.g. a shared document referenced by two sessions), the current
   merge logic returns duplicate hits. Should the router deduplicate before
   returning, or should the caller handle it?

5. **Negative inter-similarity**: cosine can be negative for anti-correlated
   namespaces. Currently clamped to 0. Should negative edges (repulsion) be
   represented? A negative cohesion edge would *encourage* the cut to separate
   anti-correlated namespaces — potentially useful for adversarial
   decomposition of overlapping namespaces.
