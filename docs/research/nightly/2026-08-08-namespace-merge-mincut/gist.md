# S-T Mincut Namespace Routing for Multi-Namespace Vector Search

**Repository**: [ruvnet/ruvector](https://github.com/ruvnet/ruvector)  
**Crate**: `ruvector-namespace-merge`  
**Date**: 2026-08-08  
**Topic**: Principled namespace routing via max-flow/min-cut for agent memory vector search

---

## The Problem

Agent memory systems partition stored vectors into **namespaces** — logical buckets by
domain, session, or tool (e.g. `code/rust`, `session/42`, `tool/web-search`). Today,
every query scans all namespaces and merges results. That's O(N·n_vecs) distance
computations regardless of query relevance.

Simpler fixes fall short:

- **Cosine threshold**: skip namespaces with `cosine(q, centroid) < 0.35`. Requires
  hand-tuning. Doesn't use inter-namespace relationships.
- **Top-k namespaces**: take the k most-similar centroids. Hard-coded k ignores
  cluster geometry — sometimes 1 namespace is right, sometimes 3.

What we want: route each query to the **coherent semantic cluster** of namespaces it
belongs to, without any hand-tuned parameter.

---

## The Solution: Flow Graph over Namespaces

Model namespace selection as an S-T min-cut problem.

Build a flow network with `N + 2` nodes (N namespaces, source S, sink T):

```
S → nsᵢ    capacity = round( q_sim_norm[i] × 10000 )
nsᵢ → T    capacity = round( (1 − q_sim_norm[i]) × 10000 )
nsᵢ ↔ nsⱼ  capacity = round( inter_sim[i,j] × 10000 )
```

where `q_sim_norm[i]` is the query's cosine similarity to namespace i's centroid,
**normalised to [0,1] over the observed range for this query**.

Run Edmonds-Karp max-flow. The source-side of the min-cut (nodes reachable from S in
the residual graph) = namespaces to search.

**The min-cut minimises**:
- `S → nsᵢ` cut = cost of *not* searching a relevant namespace
- `nsᵢ → T` cut = cost of *including* an irrelevant namespace  
- `nsᵢ ↔ nsⱼ` cut = cost of separating similar namespaces

This naturally keeps coherent clusters together.

---

## Critical Implementation Detail: Relative Normalisation

Raw cosine similarities are sensitive to dimensionality and noise. At dims=64 with
30% noise, all q_sim values fall in [0.3, 0.5] — every value is below 0.5, so
`S→ns` capacity < `ns→T` capacity for all namespaces. Edmonds-Karp saturates all
S-edges; no namespace is reachable from S in the residual graph; the router returns
zero results.

The fix: normalise **per-query** to the observed range:

```rust
let q_min = q_sim.iter().cloned().fold(f32::INFINITY, f32::min);
let q_max = q_sim.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
let range = (q_max - q_min).max(1e-6);  // clamp: never divide by zero

for i in 0..n {
    let qs = ((q_sim[i] - q_min) / range).clamp(0.0, 1.0);
    g.add_edge(s, i, (qs * scale as f32).round() as i64);
    g.add_edge(i, t, ((1.0 - qs) * scale as f32).round() as i64);
}
```

After this fix: the most-relevant namespace always gets full `S→ns` capacity;
the least-relevant always gets full `ns→T` capacity. The cut adapts automatically.

---

## Benchmark Results (64-dim, noise=0.30, 300 queries)

```
Variant          Mean(µs)  p95(µs)   QPS      Recall   NS searched  Dist ops
AllSearch           133.7     157     7,481    1.0000      5.00        2500
CentroidFilter       49.7      63    20,125    0.9453      1.91         957
MinCutRoute          54.2      68    18,449    0.9853      2.05        1025
```

MinCutRoute achieves **98.5% recall** while performing only **41% of AllSearch's
distance computations** — a 2.47× speedup in mean latency with near-perfect recall.

---

## Rust Implementation (zero dependencies)

```rust
// Flow graph — integer adjacency matrix
pub struct FlowGraph { n: usize, cap: Vec<i64> }

impl FlowGraph {
    pub fn new(n: usize) -> Self {
        FlowGraph { n, cap: vec![0; n * n] }
    }
    pub fn add_edge(&mut self, u: usize, v: usize, c: i64) {
        self.cap[u * self.n + v] += c;
    }
    pub fn add_undirected(&mut self, u: usize, v: usize, c: i64) {
        self.cap[u * self.n + v] += c;
        self.cap[v * self.n + u] += c;
    }
    pub fn max_flow(&mut self, s: usize, t: usize) -> i64 { /* Edmonds-Karp */ }
    pub fn source_side(&self, s: usize) -> Vec<bool> { /* BFS on residual */ }
}

// Router
pub struct MinCutRoute {
    inter_sim: Vec<f32>,  // N×N centroid cosine matrix (precomputed)
    n_ns: usize,
    scale: i64,           // 10_000
}

impl MinCutRoute {
    pub fn new(dataset: &Dataset) -> Self { /* O(N²D) precompute */ }
    fn route(&self, q_sim: &[f32]) -> Vec<bool> { /* build graph + solve */ }
}
```

---

## Test Coverage

```
test all_search_recall_one                    ... ok  (recall = 1.0000)
test centroid_filter_high_recall              ... ok  (recall = 0.945 ≥ 0.75)
test mincut_route_searches_fewer_ns_than_all  ... ok  (recall = 0.985 ≥ 0.70, ns = 2.05 < 5.0)
test flow_unit_two_cluster_query              ... ok  (A0, A1 on S-side; C on T-side)
test flow::tests::test_simple_max_flow        ... ok  (flow = 5)
test flow::tests::test_source_side            ... ok  (only s reachable after saturation)
```

---

## Why This Matters for Agent Memory

In RuVector's agent-memory tier, each namespace corresponds to a memory domain:

- `code/rust` — code snippets and API documentation
- `session/42` — conversation history
- `tool/web-search` — retrieved web content
- `persona/technical` — role-specific knowledge

A query from a Rust coding task should search `code/rust` and `persona/technical`,
not `session/42` or `tool/web-search`. MinCutRoute discovers this partitioning
automatically from centroid geometry — no configuration required.

The O(VE²) flow solve is ~5 µs for N=20 namespaces. The precomputed N×N centroid
matrix is 1.6 KB for N=20. Both are negligible against the ANN search cost.

---

## References

- Edmonds, J. & Karp, R.M. (1972). "Theoretical improvements in algorithmic
  efficiency for network flow problems." *JACM* 19(2), 248–264.
- Ford, L.R. & Fulkerson, D.R. (1956). "Maximal flow through a network."
  *Canadian Journal of Mathematics* 8, 399–404.
- Graph cuts for image segmentation: Boykov & Jolly (ICCV 2001) — the
  original inspiration for applying min-cut to partitioning with coherence.

---

*Part of the RuVector nightly research series. See
`docs/research/nightly/2026-08-08-namespace-merge-mincut/README.md` for the full
research document and `docs/adr/ADR-298-namespace-merge-mincut.md` for the
architecture decision record.*
