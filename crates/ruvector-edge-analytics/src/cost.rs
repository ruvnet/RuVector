//! A-priori cost model for min-cut queries, so an over-budget query is
//! refused with `budget_exceeded` (413) *before* the solver allocates or
//! spins — the solver itself has no step counter to interrupt.
//!
//! `ruvector-mincut`'s exact solver (`algorithm/exact.rs`) first runs an
//! O(n + m) certificate: it returns immediately when the graph is
//! disconnected, when the lightest bridge weighs `<= 2 * lightest edge`, or
//! when the minimum weighted degree is `<= min(lightest bridge, 2 *
//! lightest edge)`. Otherwise it runs sparse Stoer-Wagner,
//! O(n * (n + m) * log n). [`precheck`] evaluates the same predicate with the
//! same summation order (edges sorted by `(u, v)`), so the estimate knows
//! which branch the solver will take; `tests/cost_drift.rs` pins that.
//!
//! Work units are calibrated (ignored `tests/calibrate.rs`, x86_64 release)
//! so one unit is at most about one nanosecond of native CPU: measured
//! 0.7-1.0 ns/unit on the linear path at the time of writing, 0.36-0.95 on
//! Stoer-Wagner (the n (n + 2m) log n bound is loose on dense graphs, which
//! contract fast). wasm32 is slower
//! by a constant the budget absorbs **[U]**: re-measure on workerd.
//! Memory estimates are upper bounds on allocator peaks, asserted by
//! `tests/memory.rs` (measured ~370 B/edge on large sparse graphs, mostly
//! `DynamicGraph`'s concurrent maps), with >= 1.15x headroom.

use crate::graph::{EdgeRecord, TenantGraph};
use serde::{Deserialize, Serialize};

/// Which solver branch the estimate predicts.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum SolverPath {
    /// Fewer than two vertices: no cut exists.
    Trivial,
    /// The O(n + m) certificate decides the cut.
    Certified,
    /// Full sparse Stoer-Wagner.
    StoerWagner,
}

/// A cost estimate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub struct CostEstimate {
    /// Predicted solver branch.
    pub path: SolverPath,
    /// Work units (~ns native).
    pub work: u64,
    /// Upper bound on peak heap bytes for rebuild + query.
    pub memory_bytes: u64,
}

/// Result of the certificate predicate, mirroring the solver's.
#[derive(Debug, Clone, Copy, PartialEq)]
pub struct Precheck {
    /// Distinct vertices.
    pub vertices: u64,
    /// Whether the certificate decides the cut.
    pub certified: bool,
    /// The cut value the certificate implies, when `certified`.
    pub certified_value: Option<f64>,
}

/// Linear work per vertex/edge (rebuild into `DynamicGraph` + certificate).
pub const WORK_LINEAR_PER_ITEM: u64 = 500;
/// Stoer-Wagner work per `(n * (n + 2m) * log2 n)` step.
pub const WORK_SW_PER_STEP: u64 = 1;

/// Heap bytes per edge for rebuild + certified query (upper bound).
pub const MEM_PER_EDGE: u64 = 420;
/// Heap bytes per vertex for rebuild + certified query (upper bound).
pub const MEM_PER_VERTEX: u64 = 450;
/// Extra heap bytes per edge for Stoer-Wagner's contraction maps.
pub const MEM_SW_PER_EDGE: u64 = 160;
/// Extra heap bytes per vertex for Stoer-Wagner's heap and groups.
pub const MEM_SW_PER_VERTEX: u64 = 96;

fn log2_ceil(n: u64) -> u64 {
    u64::from(64 - n.max(2).saturating_sub(1).leading_zeros())
}

/// Evaluate the solver's certificate predicate on a canonical graph.
pub fn precheck(g: &TenantGraph) -> Precheck {
    let edges = g.edges();
    let mut ids: Vec<u64> = Vec::with_capacity(edges.len() * 2);
    for e in edges {
        ids.push(e.u);
        ids.push(e.v);
    }
    ids.sort_unstable();
    ids.dedup();
    let n = ids.len();
    if n < 2 {
        return Precheck {
            vertices: n as u64,
            certified: true,
            certified_value: None,
        };
    }
    let idx = |x: u64| ids.binary_search(&x).unwrap_or(0);
    let dense: Vec<(usize, usize, f64)> = edges
        .iter()
        .map(|e: &EdgeRecord| (idx(e.u), idx(e.v), e.w))
        .collect();
    drop(ids);
    let (certified, value) = certificate(n, &dense);
    Precheck {
        vertices: n as u64,
        certified,
        certified_value: value,
    }
}

/// Same arithmetic as `ruvector-mincut` `exact::certified_cut`: edges in
/// canonical order, degrees summed in that order, iterative DFS low-link.
fn certificate(n: usize, edges: &[(usize, usize, f64)]) -> (bool, Option<f64>) {
    // CSR adjacency (smaller than Vec<Vec<_>>), neighbor order = edge order.
    let mut start = vec![0usize; n + 1];
    for &(u, v, _) in edges {
        start[u + 1] += 1;
        start[v + 1] += 1;
    }
    for i in 0..n {
        start[i + 1] += start[i];
    }
    let mut fill = start.clone();
    let mut adj = vec![(0usize, 0f64); edges.len() * 2];
    let mut degrees = vec![0.0f64; n];
    let mut lightest = f64::INFINITY;
    for &(u, v, w) in edges {
        adj[fill[u]] = (v, w);
        fill[u] += 1;
        adj[fill[v]] = (u, w);
        fill[v] += 1;
        degrees[u] += w;
        degrees[v] += w;
        lightest = lightest.min(w);
    }
    drop(fill);
    let mut enter = vec![usize::MAX; n];
    let mut low = vec![0usize; n];
    let mut parent = vec![usize::MAX; n];
    let mut next = start[..n].to_vec();
    let mut stack = vec![0usize];
    enter[0] = 0;
    let mut visited = 1usize;
    let mut bridge_weight = f64::INFINITY;
    let mut has_bridge = false;
    while let Some(&v) = stack.last() {
        if next[v] < start[v + 1] {
            let (u, _) = adj[next[v]];
            next[v] += 1;
            if u == parent[v] {
                continue;
            }
            if enter[u] == usize::MAX {
                parent[u] = v;
                enter[u] = visited;
                low[u] = visited;
                visited += 1;
                stack.push(u);
            } else {
                low[v] = low[v].min(enter[u]);
            }
        } else {
            stack.pop();
            let p = parent[v];
            if p != usize::MAX {
                low[p] = low[p].min(low[v]);
                if low[v] > enter[p] {
                    let w = adj[next[p] - 1].1;
                    if !has_bridge || w < bridge_weight {
                        bridge_weight = w;
                        has_bridge = true;
                    }
                }
            }
        }
    }
    if visited != n {
        return (true, Some(0.0));
    }
    if has_bridge && bridge_weight <= 2.0 * lightest {
        return (true, Some(bridge_weight));
    }
    let bound = bridge_weight.min(2.0 * lightest);
    let min_deg = degrees
        .iter()
        .copied()
        .fold(f64::INFINITY, |a, d| if d < a { d } else { a });
    if min_deg <= bound {
        return (true, Some(min_deg));
    }
    (false, None)
}

fn linear(n: u64, m: u64) -> (u64, u64) {
    let work = WORK_LINEAR_PER_ITEM.saturating_mul(n.saturating_add(m));
    let mem = MEM_PER_EDGE
        .saturating_mul(m)
        .saturating_add(MEM_PER_VERTEX.saturating_mul(n));
    (work, mem)
}

/// Estimate an exact query from a precheck and the edge count.
pub fn estimate_exact(pre: &Precheck, edges: u64) -> CostEstimate {
    let n = pre.vertices;
    let (work, mem) = linear(n, edges);
    if n < 2 {
        return CostEstimate {
            path: SolverPath::Trivial,
            work,
            memory_bytes: mem,
        };
    }
    if pre.certified {
        return CostEstimate {
            path: SolverPath::Certified,
            work,
            memory_bytes: mem,
        };
    }
    let steps = n
        .saturating_mul(n.saturating_add(edges.saturating_mul(2)))
        .saturating_mul(log2_ceil(n));
    CostEstimate {
        path: SolverPath::StoerWagner,
        work: work.saturating_add(steps.saturating_mul(WORK_SW_PER_STEP)),
        memory_bytes: mem
            .saturating_add(MEM_SW_PER_EDGE.saturating_mul(edges))
            .saturating_add(MEM_SW_PER_VERTEX.saturating_mul(n)),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::graph::GraphLimits;

    fn g(edges: &[(u64, u64, f64)]) -> TenantGraph {
        TenantGraph::from_edges([0; 16], 1, edges, &GraphLimits::INLINE).unwrap()
    }

    #[test]
    fn certificate_branches() {
        // Disconnected.
        let p = precheck(&g(&[(1, 2, 3.0), (3, 4, 3.0)]));
        assert_eq!((p.certified, p.certified_value), (true, Some(0.0)));
        // Pendant bridge of weight 1.
        let p = precheck(&g(&[(1, 2, 1.0), (2, 3, 1.0), (3, 1, 1.0), (3, 4, 1.0)]));
        assert_eq!((p.certified, p.certified_value), (true, Some(1.0)));
        // K4 with unit weights: no bridge, min degree 3 > 2 -> Stoer-Wagner.
        let k4: Vec<_> = (0..4u64)
            .flat_map(|a| (a + 1..4).map(move |b| (a, b, 1.0)))
            .collect();
        let p = precheck(&g(&k4));
        assert!(!p.certified);
        assert_eq!(estimate_exact(&p, 6).path, SolverPath::StoerWagner);
        // Single vertex pair is certified; no edges is trivial.
        assert_eq!(
            estimate_exact(&precheck(&g(&[])), 0).path,
            SolverPath::Trivial
        );
    }

    #[test]
    fn log2_ceil_values() {
        assert_eq!(log2_ceil(0), 1);
        assert_eq!(log2_ceil(2), 1);
        assert_eq!(log2_ceil(3), 2);
        assert_eq!(log2_ceil(1024), 10);
        assert_eq!(log2_ceil(1025), 11);
    }
}
