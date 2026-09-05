//! Variant 2 — GraphHop: k-NN brute scan + 1-hop neighbourhood expansion.
//!
//! At query time, find the top `ef_init` candidates via brute L2 scan,
//! then expand by adding each candidate's graph neighbours from the
//! pre-built k-NN adjacency list, re-score everything, and return top-k.
//!
//! This variant approximates the kind of graph traversal used in HNSW's
//! lower layers or DiskANN's one-hop fetches, but on an explicit flat graph.

use crate::{CommunitySearch, Hit, VectorMeta, l2_sq};
use std::collections::HashSet;

/// Pre-computed k-NN adjacency list entry.
#[derive(Clone)]
struct Neighbour {
    id: usize,
}

pub struct GraphHop {
    vectors: Vec<Vec<f32>>,
    metas: Vec<VectorMeta>,
    /// For each node, its `knn_k` nearest neighbours.
    adj: Vec<Vec<Neighbour>>,
    /// Number of neighbours per node built into the graph.
    knn_k: usize,
    /// Number of initial candidates before expansion.
    ef_init: usize,
}

impl GraphHop {
    pub fn new(knn_k: usize, ef_init: usize) -> Self {
        Self {
            vectors: Vec::new(),
            metas: Vec::new(),
            adj: Vec::new(),
            knn_k,
            ef_init,
        }
    }

    fn build_adj(&mut self) {
        let n = self.vectors.len();
        self.adj = vec![Vec::new(); n];
        for i in 0..n {
            // Brute-force nearest neighbours for each node.
            let mut dists: Vec<(usize, f32)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| (j, l2_sq(&self.vectors[i], &self.vectors[j])))
                .collect();
            dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
            self.adj[i] = dists
                .into_iter()
                .take(self.knn_k)
                .map(|(id, _)| Neighbour { id })
                .collect();
        }
    }
}

impl CommunitySearch for GraphHop {
    fn insert(&mut self, vector: &[f32], community: usize) {
        let id = self.vectors.len();
        self.metas.push(VectorMeta { id, community });
        self.vectors.push(vector.to_vec());
    }

    fn build(&mut self) {
        self.build_adj();
    }

    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let n = self.vectors.len();
        let ef = self.ef_init.min(n);

        // Stage 1: brute-force top-ef_init.
        let mut initial: Vec<(usize, f32)> = (0..n)
            .map(|i| (i, l2_sq(query, &self.vectors[i])))
            .collect();
        initial.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        initial.truncate(ef);

        // Stage 2: collect 1-hop neighbours of initial candidates.
        let mut candidates: HashSet<usize> = initial.iter().map(|(id, _)| *id).collect();
        for (id, _) in &initial {
            for nb in &self.adj[*id] {
                candidates.insert(nb.id);
            }
        }

        // Stage 3: re-score and return top-k.
        let mut scored: Vec<Hit> = candidates
            .into_iter()
            .map(|id| Hit { id, distance: l2_sq(query, &self.vectors[id]) })
            .collect();
        scored.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        scored.truncate(k);
        scored
    }

    fn memory_bytes(&self) -> usize {
        let vec_bytes: usize = self.vectors.iter().map(|v| v.len() * 4).sum();
        let adj_bytes: usize = self.adj.iter().map(|nb| nb.len() * 8).sum();
        let meta_bytes = self.metas.len() * std::mem::size_of::<VectorMeta>();
        vec_bytes + adj_bytes + meta_bytes
    }

    fn name(&self) -> &'static str {
        "GraphHop"
    }
}
