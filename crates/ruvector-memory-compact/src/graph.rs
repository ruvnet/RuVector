//! k-NN coherence graph construction over a MemoryStore.
//!
//! Builds a sparse similarity graph: each node is a memory entry; each edge
//! (i, j) carries the cosine similarity between entry i and entry j. Only the
//! top-k neighbours per node are stored to keep the graph tractable.

use crate::{cosine_sim, MemoryEntry};

/// A weighted edge in the coherence graph.
#[derive(Debug, Clone)]
pub struct Edge {
    pub a: usize,
    pub b: usize,
    pub weight: f32,
}

/// Sparse k-NN coherence graph.
pub struct CoherenceGraph {
    pub n: usize,
    /// Adjacency list: for each node, its (neighbour_index, similarity) pairs.
    pub adj: Vec<Vec<(usize, f32)>>,
    pub edges: Vec<Edge>,
}

impl CoherenceGraph {
    /// Build from a slice of memory entries with `k` neighbours per node.
    pub fn build(entries: &[MemoryEntry], k: usize) -> Self {
        let n = entries.len();
        let mut adj: Vec<Vec<(usize, f32)>> = vec![Vec::new(); n];
        let mut edges: Vec<Edge> = Vec::new();

        for i in 0..n {
            // Compute similarity to all other nodes.
            let mut sims: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (cosine_sim(&entries[i].embedding, &entries[j].embedding), j))
                .collect();
            // Keep top-k by similarity.
            sims.sort_unstable_by(|a, b| b.0.partial_cmp(&a.0).unwrap());
            sims.truncate(k);

            for (sim, j) in sims {
                adj[i].push((j, sim));
                if i < j {
                    edges.push(Edge {
                        a: i,
                        b: j,
                        weight: sim,
                    });
                }
            }
        }
        Self { n, adj, edges }
    }

    /// Return all edge weights above `threshold` as (a, b) index pairs.
    pub fn edges_above(&self, threshold: f32) -> Vec<(usize, usize)> {
        self.edges
            .iter()
            .filter(|e| e.weight >= threshold)
            .map(|e| (e.a, e.b))
            .collect()
    }

    /// Intra-cluster coherence: average similarity among all pairs in `cluster`.
    pub fn cluster_coherence(&self, cluster: &[usize]) -> f32 {
        if cluster.len() < 2 {
            return 1.0;
        }
        let mut sum = 0.0_f32;
        let mut count = 0usize;
        for (ii, &a) in cluster.iter().enumerate() {
            for &b in &cluster[ii + 1..] {
                // Look up in adjacency list first (fast path).
                if let Some(&(_, w)) = self.adj[a].iter().find(|(n, _)| *n == b) {
                    sum += w;
                } else if let Some(&(_, w)) = self.adj[b].iter().find(|(n, _)| *n == a) {
                    sum += w;
                }
                // If not in k-NN graph, skip (similarity is low by assumption).
                count += 1;
            }
        }
        if count == 0 {
            1.0
        } else {
            sum / count as f32
        }
    }

    /// Coherence score for a cluster: 1 - std_dev(pairwise similarities).
    /// High score means all members are uniformly similar (tight cluster).
    pub fn cluster_coherence_score(&self, cluster: &[usize]) -> f32 {
        if cluster.len() < 2 {
            return 1.0;
        }
        let mut sims: Vec<f32> = Vec::new();
        for (ii, &a) in cluster.iter().enumerate() {
            for &b in &cluster[ii + 1..] {
                if let Some(&(_, w)) = self.adj[a].iter().find(|(n, _)| *n == b) {
                    sims.push(w);
                } else if let Some(&(_, w)) = self.adj[b].iter().find(|(n, _)| *n == a) {
                    sims.push(w);
                }
            }
        }
        if sims.is_empty() {
            return 0.0;
        }
        let mean = sims.iter().sum::<f32>() / sims.len() as f32;
        let variance = sims.iter().map(|&s| (s - mean).powi(2)).sum::<f32>() / sims.len() as f32;
        (1.0 - variance.sqrt()).max(0.0)
    }
}

/// Union-Find for connected-component clustering.
pub struct UnionFind {
    parent: Vec<usize>,
    rank: Vec<usize>,
}

impl UnionFind {
    pub fn new(n: usize) -> Self {
        Self {
            parent: (0..n).collect(),
            rank: vec![0; n],
        }
    }

    pub fn find(&mut self, x: usize) -> usize {
        if self.parent[x] != x {
            self.parent[x] = self.find(self.parent[x]);
        }
        self.parent[x]
    }

    pub fn union(&mut self, x: usize, y: usize) {
        let rx = self.find(x);
        let ry = self.find(y);
        if rx == ry {
            return;
        }
        match self.rank[rx].cmp(&self.rank[ry]) {
            std::cmp::Ordering::Less => self.parent[rx] = ry,
            std::cmp::Ordering::Greater => self.parent[ry] = rx,
            std::cmp::Ordering::Equal => {
                self.parent[ry] = rx;
                self.rank[rx] += 1;
            }
        }
    }

    /// Collect components as groups of node indices.
    pub fn components(&mut self, n: usize) -> Vec<Vec<usize>> {
        let mut map: std::collections::HashMap<usize, Vec<usize>> =
            std::collections::HashMap::new();
        for i in 0..n {
            let root = self.find(i);
            map.entry(root).or_default().push(i);
        }
        map.into_values().collect()
    }
}
