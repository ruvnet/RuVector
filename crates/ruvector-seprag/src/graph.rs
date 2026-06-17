//! Undirected, positively-weighted graph + the brute-force k-NN oracle.
//!
//! The oracle (Dijkstra over the raw graph) is the ground truth that every
//! SepRAG result is validated against in the M0 correctness gate.

use std::cmp::Ordering;
use std::collections::BinaryHeap;

pub type NodeId = u32;

/// Simple adjacency-list graph. Edges are undirected; duplicate edges keep the
/// minimum weight; self-loops are ignored. Weights must be strictly positive
/// (a precondition for additive shortest-path semantics — see ADR-198).
#[derive(Clone, Debug, Default)]
pub struct Graph {
    pub n: usize,
    pub adj: Vec<Vec<(NodeId, f64)>>,
}

impl Graph {
    #[must_use]
    pub fn new(n: usize) -> Self {
        Graph { n, adj: vec![Vec::new(); n] }
    }

    /// Insert/relax an undirected edge. O(deg) due to the dedup scan — fine at
    /// M0 scale; replaced by CSR ingestion at M1.
    pub fn add_edge(&mut self, u: NodeId, v: NodeId, w: f64) {
        if u == v {
            return;
        }
        debug_assert!(w > 0.0, "edge weights must be strictly positive");
        Self::relax_dir(&mut self.adj[u as usize], v, w);
        Self::relax_dir(&mut self.adj[v as usize], u, w);
    }

    fn relax_dir(row: &mut Vec<(NodeId, f64)>, to: NodeId, w: f64) {
        if let Some(e) = row.iter_mut().find(|(x, _)| *x == to) {
            if w < e.1 {
                e.1 = w;
            }
        } else {
            row.push((to, w));
        }
    }

    /// Iterate canonical undirected edges `(u, v, w)` with `u < v`.
    pub fn edges(&self) -> impl Iterator<Item = (NodeId, NodeId, f64)> + '_ {
        self.adj.iter().enumerate().flat_map(|(u, row)| {
            let u = u as NodeId;
            row.iter()
                .filter(move |(v, _)| *v > u)
                .map(move |&(v, w)| (u, v, w))
        })
    }

    /// Single-source shortest paths from `src` (Dijkstra). `dist[v] = +inf` for
    /// unreachable vertices.
    #[must_use]
    pub fn dijkstra(&self, src: NodeId) -> Vec<f64> {
        let mut dist = vec![f64::INFINITY; self.n];
        dist[src as usize] = 0.0;
        let mut heap = BinaryHeap::new();
        heap.push(HeapItem { dist: 0.0, node: src });
        while let Some(HeapItem { dist: d, node }) = heap.pop() {
            if d > dist[node as usize] {
                continue;
            }
            for &(v, w) in &self.adj[node as usize] {
                let nd = d + w;
                if nd < dist[v as usize] {
                    dist[v as usize] = nd;
                    heap.push(HeapItem { dist: nd, node: v });
                }
            }
        }
        dist
    }

    /// Brute-force k nearest POIs from `src` by graph distance. Deterministic
    /// tie-break: ascending `(distance, node id)`. This is the oracle.
    #[must_use]
    pub fn knn_oracle(&self, src: NodeId, pois: &[NodeId], k: usize) -> Vec<(NodeId, f64)> {
        let dist = self.dijkstra(src);
        let mut cand: Vec<(NodeId, f64)> = pois
            .iter()
            .map(|&p| (p, dist[p as usize]))
            .filter(|(_, d)| d.is_finite())
            .collect();
        cand.sort_by(|a, b| cmp_dist_id(*a, *b));
        cand.truncate(k);
        cand
    }
}

/// Canonical ordering for `(node, dist)` results: distance asc, then id asc.
pub fn cmp_dist_id(a: (NodeId, f64), b: (NodeId, f64)) -> Ordering {
    a.1.partial_cmp(&b.1)
        .unwrap_or(Ordering::Equal)
        .then(a.0.cmp(&b.0))
}

struct HeapItem {
    dist: f64,
    node: NodeId,
}
impl PartialEq for HeapItem {
    fn eq(&self, o: &Self) -> bool {
        self.dist == o.dist
    }
}
impl Eq for HeapItem {}
impl PartialOrd for HeapItem {
    fn partial_cmp(&self, o: &Self) -> Option<Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for HeapItem {
    // Reversed: BinaryHeap is a max-heap, we want min-distance first.
    fn cmp(&self, o: &Self) -> Ordering {
        o.dist
            .partial_cmp(&self.dist)
            .unwrap_or(Ordering::Equal)
            .then(o.node.cmp(&self.node))
    }
}
