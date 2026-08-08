//! Three namespace routing strategies implementing [`NamespaceRouter`].
//!
//! All three return the same type ([`RouteResult`]) so benchmarks can swap
//! strategies without changing measurement code.

use crate::{cosine_sim, dataset::Dataset, flow::FlowGraph, sq_l2, Hit};

// ─── shared result type ───────────────────────────────────────────────────────

/// Result of a routed search: the top-k hits plus diagnostic counters.
#[derive(Debug, Clone)]
pub struct RouteResult {
    pub hits: Vec<Hit>,
    /// Number of namespaces actually searched.
    pub ns_searched: usize,
    /// Total distance computations performed.
    pub dist_ops: usize,
}

// ─── trait ───────────────────────────────────────────────────────────────────

pub trait NamespaceRouter: Send + Sync {
    fn search(&self, dataset: &Dataset, query: &[f32], k: usize) -> RouteResult;
    fn name(&self) -> &str;
    /// Heap memory used by the router (excluding the dataset itself).
    fn memory_bytes(&self) -> usize;
}

// ─── 1. AllSearch — baseline ─────────────────────────────────────────────────

/// Flat scan over every namespace unconditionally.
/// Ground truth: always achieves recall 1.0 by definition.
pub struct AllSearch;

impl NamespaceRouter for AllSearch {
    fn name(&self) -> &str {
        "AllSearch"
    }

    fn search(&self, dataset: &Dataset, query: &[f32], k: usize) -> RouteResult {
        let mut all: Vec<Hit> = Vec::new();
        let mut dist_ops = 0usize;
        for (ns_idx, ns) in dataset.namespaces.iter().enumerate() {
            for i in 0..ns.n {
                let v = ns.vector(i);
                let d = sq_l2(query, v);
                dist_ops += 1;
                let global_id = ns_idx * dataset.per_ns + i;
                all.push(Hit {
                    id: global_id,
                    dist: d,
                });
            }
        }
        all.sort_unstable();
        all.truncate(k);
        RouteResult {
            hits: all,
            ns_searched: dataset.namespaces.len(),
            dist_ops,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

// ─── 2. CentroidFilter — threshold heuristic ─────────────────────────────────

/// Skip namespaces whose centroid cosine similarity to the query is below
/// `threshold`. The threshold is set at build time.
pub struct CentroidFilter {
    pub threshold: f32,
}

impl CentroidFilter {
    pub fn new(threshold: f32) -> Self {
        CentroidFilter { threshold }
    }
}

impl NamespaceRouter for CentroidFilter {
    fn name(&self) -> &str {
        "CentroidFilter"
    }

    fn search(&self, dataset: &Dataset, query: &[f32], k: usize) -> RouteResult {
        let mut all: Vec<Hit> = Vec::new();
        let mut ns_searched = 0usize;
        let mut dist_ops = 0usize;
        for (ns_idx, ns) in dataset.namespaces.iter().enumerate() {
            let sim = cosine_sim(query, &ns.centroid);
            if sim < self.threshold {
                continue;
            }
            ns_searched += 1;
            for i in 0..ns.n {
                let v = ns.vector(i);
                let d = sq_l2(query, v);
                dist_ops += 1;
                let global_id = ns_idx * dataset.per_ns + i;
                all.push(Hit {
                    id: global_id,
                    dist: d,
                });
            }
        }
        all.sort_unstable();
        all.truncate(k);
        RouteResult {
            hits: all,
            ns_searched,
            dist_ops,
        }
    }

    fn memory_bytes(&self) -> usize {
        0
    }
}

// ─── 3. MinCutRoute — S-T flow partition ─────────────────────────────────────

/// Principled namespace routing via S-T min-cut on a namespace similarity graph.
///
/// **Flow network construction** (for a given query `q`):
///
/// Nodes:  `S` (source), `T` (sink), one node per namespace.
/// Total:  `N + 2` nodes, where `N` = number of namespaces.
///
/// Edges:
/// - `S → ns_i`  capacity = `round(q_sim[i] * SCALE)`
///   (query affinity: how much the query "wants" this namespace on the S-side)
/// - `ns_i → T`  capacity = `round((1 - q_sim[i]) * SCALE)`
///   (separation cost: how expensive it is to put ns_i on the S-side)
/// - `ns_i ↔ ns_j` capacity = `round(inter_sim[i][j] * SCALE)` (undirected)
///   (cohesion: semantically similar namespaces "resist" being split)
///
/// After running Edmonds-Karp max-flow, the source-side reachable set
/// (BFS on residual graph) gives the namespaces to search.
///
/// The min cut minimises the total capacity of severed edges, which trades off:
/// - cutting `S → ns_i` = paying the cost of *not* searching a relevant namespace
/// - cutting `ns_i → T` = paying the cost of *including* an irrelevant namespace
/// - cutting `ns_i ↔ ns_j` = paying the cost of separating similar namespaces
///
/// This naturally produces coherence-preserving routing: groups of semantically
/// similar namespaces tend to end up on the same side of the cut.
pub struct MinCutRoute {
    /// Precomputed inter-namespace cosine similarities (N × N matrix).
    inter_sim: Vec<f32>,
    pub n_ns: usize,
    /// Capacity scale factor (converts cosine [0,1] to integer capacity).
    scale: i64,
}

impl MinCutRoute {
    pub fn new(dataset: &Dataset) -> Self {
        let n = dataset.namespaces.len();
        let mut inter = vec![0f32; n * n];
        for i in 0..n {
            for j in 0..n {
                inter[i * n + j] = cosine_sim(
                    &dataset.namespaces[i].centroid,
                    &dataset.namespaces[j].centroid,
                );
            }
        }
        MinCutRoute {
            inter_sim: inter,
            n_ns: n,
            scale: 10_000,
        }
    }

    /// Compute query-to-centroid cosine similarities.
    fn query_sims(&self, dataset: &Dataset, query: &[f32]) -> Vec<f32> {
        dataset
            .namespaces
            .iter()
            .map(|ns| cosine_sim(query, &ns.centroid))
            .collect()
    }

    /// Build flow graph and run max-flow. Return source-side membership.
    ///
    /// Capacities are normalised so the most relevant namespace always has
    /// S→ns capacity = `scale`, making the cut invariant to the absolute
    /// magnitude of cosine similarities (which depends on noise and dimension).
    fn route(&self, q_sim: &[f32]) -> Vec<bool> {
        let n = self.n_ns;
        // Nodes: 0..n = namespaces, n = source (S), n+1 = sink (T)
        let s = n;
        let t = n + 1;
        let mut g = FlowGraph::new(n + 2);

        // Normalise q_sim into [0, 1] relative to its observed range so the
        // most-relevant namespace always receives full S→ns capacity.
        let q_min = q_sim.iter().cloned().fold(f32::INFINITY, f32::min);
        let q_max = q_sim.iter().cloned().fold(f32::NEG_INFINITY, f32::max);
        let range = (q_max - q_min).max(1e-6);

        for i in 0..n {
            let qs = ((q_sim[i] - q_min) / range).clamp(0.0, 1.0);
            let s_cap = (qs * self.scale as f32).round() as i64;
            let t_cap = ((1.0 - qs) * self.scale as f32).round() as i64;
            g.add_edge(s, i, s_cap);
            g.add_edge(i, t, t_cap);
        }

        for i in 0..n {
            for j in (i + 1)..n {
                let sim = self.inter_sim[i * n + j].max(0.0).min(1.0);
                let cap = (sim * self.scale as f32).round() as i64;
                g.add_undirected(i, j, cap);
            }
        }

        g.max_flow(s, t);
        let side = g.source_side(s);
        // Only return namespace nodes (indices 0..n)
        side[..n].to_vec()
    }
}

impl NamespaceRouter for MinCutRoute {
    fn name(&self) -> &str {
        "MinCutRoute"
    }

    fn search(&self, dataset: &Dataset, query: &[f32], k: usize) -> RouteResult {
        let q_sim = self.query_sims(dataset, query);
        let on_source_side = self.route(&q_sim);

        let mut all: Vec<Hit> = Vec::new();
        let mut ns_searched = 0usize;
        let mut dist_ops = 0usize;
        for (ns_idx, ns) in dataset.namespaces.iter().enumerate() {
            if !on_source_side[ns_idx] {
                continue;
            }
            ns_searched += 1;
            for i in 0..ns.n {
                let v = ns.vector(i);
                let d = sq_l2(query, v);
                dist_ops += 1;
                let global_id = ns_idx * dataset.per_ns + i;
                all.push(Hit {
                    id: global_id,
                    dist: d,
                });
            }
        }
        all.sort_unstable();
        all.truncate(k);
        RouteResult {
            hits: all,
            ns_searched,
            dist_ops,
        }
    }

    fn memory_bytes(&self) -> usize {
        self.inter_sim.len() * 4
    }
}
