//! Deterministic synthetic weighted-graph generator and canonical encodings.
//!
//! The generator models a sparse agent-memory association graph: nodes are
//! memory items, edges are semantic/temporal association weights. It is
//! deliberately simple (no claim of modeling any real memory workload) —
//! its only job is to produce a connected, reproducible, sparse graph of a
//! chosen size so the import-latency hypothesis can be measured
//! independently of any particular graph topology.
//!
//! Construction, for a fixed seed and target average degree `avg_degree`:
//!
//! 1. A backbone: node `i` (for `i in 1..n`) gets `backbone_links` edges to
//!    distinct, uniformly chosen earlier nodes `j < i`. This guarantees the
//!    graph is connected (every node has a path back to node 0), so the
//!    global minimum cut is never trivially zero.
//! 2. Extra random edges between arbitrary distinct node pairs, deduplicated
//!    against the backbone and each other, until the average degree target
//!    is reached.
//! 3. Every edge gets a weight drawn uniformly from `[0.1, 1.0)`, modeling a
//!    semantic/temporal association strength.
//!
//! All randomness comes from `rand::rngs::StdRng::seed_from_u64(seed)`, so
//! the same `(n, seed, avg_degree)` always produces byte-identical output.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use sha2::{Digest, Sha256};
use std::collections::HashSet;

/// Domain tag for the canonical graph encoding fed into [`CutCertificate`]
/// digests. Mirrors the `MSG_TAG`-style domain separation used by
/// `ruvector-agent-memory::witness_signing` and
/// `ruvector-retrieval-receipt::signing`.
const GRAPH_TAG: &[u8] = b"ruvector-partition-import:graph:v1:";
/// Domain tag for the canonical partition encoding.
const PARTITION_TAG: &[u8] = b"ruvector-partition-import:partition:v1:";

/// A simple undirected weighted graph: `n` nodes numbered `0..n`, plus a
/// deduplicated edge list `(u, v, weight)` with `u < v`.
#[derive(Clone, Debug)]
pub struct WeightedGraph {
    pub n: usize,
    pub edges: Vec<(u32, u32, f64)>,
}

impl WeightedGraph {
    /// Adjacency list view: `adj[v]` is `(neighbor, weight)` pairs for `v`.
    pub fn adjacency(&self) -> Vec<Vec<(u32, f64)>> {
        let mut adj = vec![Vec::new(); self.n];
        for &(u, v, w) in &self.edges {
            adj[u as usize].push((v, w));
            adj[v as usize].push((u, w));
        }
        adj
    }

    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Canonical byte encoding used as the pre-image of the graph digest
    /// inside a [`crate::certificate::CutCertificate`]. Edges are already
    /// stored with `u < v` and in generation order; we additionally sort
    /// them here so the encoding is stable regardless of how the caller
    /// built the edge list, which matters once this leaves the generator
    /// (e.g. after deserializing a transferred payload).
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut edges = self.edges.clone();
        edges.sort_unstable_by_key(|&(u, v, _)| (u, v));

        let mut bytes = Vec::with_capacity(GRAPH_TAG.len() + 8 + edges.len() * 16);
        bytes.extend_from_slice(GRAPH_TAG);
        bytes.extend_from_slice(&(self.n as u64).to_be_bytes());
        for (u, v, w) in edges {
            bytes.extend_from_slice(&u.to_be_bytes());
            bytes.extend_from_slice(&v.to_be_bytes());
            bytes.extend_from_slice(&w.to_bits().to_be_bytes());
        }
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    /// Transfer payload size in bytes: a reasonable fixed-width wire
    /// encoding (not the canonical hashing encoding, which omits nothing
    /// but is not meant to be minimal) — `4 + 4 + 8` bytes per edge plus an
    /// 8-byte node count header. Used only for the payload-size metric in
    /// the benchmark; it does not need to be the project's actual wire
    /// format to make a fair baseline-vs-candidate comparison, since both
    /// variants transmit `G` under the same encoding.
    pub fn wire_size_bytes(&self) -> usize {
        8 + self.edges.len() * (4 + 4 + 8)
    }
}

/// A cut partition over a graph's `n` vertices: `side[v]` is `false` for
/// side 0 and `true` for side 1.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Partition {
    pub side: Vec<bool>,
}

impl Partition {
    pub fn new(side: Vec<bool>) -> Self {
        Self { side }
    }

    pub fn len(&self) -> usize {
        self.side.len()
    }

    pub fn is_empty(&self) -> bool {
        self.side.is_empty()
    }

    /// Canonical byte encoding used as the pre-image of the partition
    /// digest inside a [`crate::certificate::CutCertificate`].
    pub fn canonical_bytes(&self) -> Vec<u8> {
        let mut bytes = Vec::with_capacity(PARTITION_TAG.len() + 8 + self.side.len());
        bytes.extend_from_slice(PARTITION_TAG);
        bytes.extend_from_slice(&(self.side.len() as u64).to_be_bytes());
        for &s in &self.side {
            bytes.push(u8::from(s));
        }
        bytes
    }

    pub fn digest(&self) -> [u8; 32] {
        Sha256::digest(self.canonical_bytes()).into()
    }

    pub fn wire_size_bytes(&self) -> usize {
        8 + self.side.len()
    }

    /// The weight of edges crossing this partition, in the given graph.
    pub fn cut_value(&self, graph: &WeightedGraph) -> f64 {
        graph
            .edges
            .iter()
            .filter(|&&(u, v, _)| self.side[u as usize] != self.side[v as usize])
            .map(|&(_, _, w)| w)
            .sum()
    }
}

/// Generate a deterministic, connected, sparse weighted graph.
///
/// `avg_degree` is the target average node degree (edges counted once per
/// endpoint, so a node with 3 edges has degree 3). `backbone_links` (fixed
/// at 2 here) guarantees connectivity independent of the random extra
/// edges; `avg_degree` should be set well above `2 * backbone_links` for
/// the backbone to be a minor contributor to the final degree
/// distribution.
pub fn generate_graph(n: usize, seed: u64, avg_degree: usize) -> WeightedGraph {
    assert!(n >= 2, "graph must have at least 2 nodes");
    let mut rng = StdRng::seed_from_u64(seed);
    let backbone_links = 2usize.min(n - 1);

    let mut seen: HashSet<(u32, u32)> = HashSet::new();
    let mut edges: Vec<(u32, u32, f64)> = Vec::new();

    let push_edge = |u: u32,
                     v: u32,
                     rng: &mut StdRng,
                     seen: &mut HashSet<(u32, u32)>,
                     edges: &mut Vec<(u32, u32, f64)>| {
        let (a, b) = if u < v { (u, v) } else { (v, u) };
        if a == b {
            return;
        }
        if seen.insert((a, b)) {
            let w = rng.gen_range(0.1..1.0);
            edges.push((a, b, w));
        }
    };

    // 1. Backbone: guarantees connectivity.
    for i in 1..n {
        let links = backbone_links.min(i);
        let mut chosen: HashSet<usize> = HashSet::new();
        while chosen.len() < links {
            let j = rng.gen_range(0..i);
            chosen.insert(j);
        }
        let mut chosen: Vec<usize> = chosen.into_iter().collect();
        chosen.sort_unstable();
        for j in chosen {
            push_edge(i as u32, j as u32, &mut rng, &mut seen, &mut edges);
        }
    }

    // 2. Extra random edges to reach the target average degree.
    let target_edges = (n * avg_degree) / 2;
    let mut attempts = 0usize;
    let max_attempts = target_edges.saturating_mul(20).max(1000);
    while edges.len() < target_edges && attempts < max_attempts {
        attempts += 1;
        let u = rng.gen_range(0..n) as u32;
        let v = rng.gen_range(0..n) as u32;
        push_edge(u, v, &mut rng, &mut seen, &mut edges);
    }

    WeightedGraph { n, edges }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generation_is_deterministic() {
        let g1 = generate_graph(200, 42, 8);
        let g2 = generate_graph(200, 42, 8);
        assert_eq!(g1.digest(), g2.digest());
    }

    #[test]
    fn different_seeds_differ() {
        let g1 = generate_graph(200, 1, 8);
        let g2 = generate_graph(200, 2, 8);
        assert_ne!(g1.digest(), g2.digest());
    }

    #[test]
    fn graph_is_connected() {
        let g = generate_graph(500, 7, 8);
        let adj = g.adjacency();
        let mut visited = vec![false; g.n];
        let mut stack = vec![0usize];
        visited[0] = true;
        let mut count = 1;
        while let Some(u) = stack.pop() {
            for &(v, _) in &adj[u] {
                if !visited[v as usize] {
                    visited[v as usize] = true;
                    count += 1;
                    stack.push(v as usize);
                }
            }
        }
        assert_eq!(count, g.n, "graph generator must produce a connected graph");
    }

    #[test]
    fn partition_tamper_changes_digest() {
        let mut p = Partition::new(vec![false, true, false, true]);
        let d1 = p.digest();
        p.side[0] = true;
        let d2 = p.digest();
        assert_ne!(d1, d2);
    }

    #[test]
    fn graph_tamper_changes_digest() {
        let mut g = generate_graph(50, 3, 8);
        let d1 = g.digest();
        g.edges[0].2 += 1.0;
        let d2 = g.digest();
        assert_ne!(d1, d2);
    }
}
