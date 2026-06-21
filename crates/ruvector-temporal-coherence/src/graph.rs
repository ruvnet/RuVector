//! Lightweight coherence graph for agent memory gating.
//!
//! Builds an undirected adjacency structure where memories are nodes and
//! edges connect memories whose cosine similarity exceeds `threshold`.
//! A memory's coherence gate value is its normalised in-degree, scaled to [0, 1].
//!
//! Graph construction is O(n²) — appropriate for PoC sizes (up to ~10K nodes).
//! For production, an approximate k-NN graph via HNSW would replace the scan.

use crate::{cosine_sim, MemoryStore};

pub struct CoherenceGraph {
    /// degree[i] = number of neighbors above threshold
    degree: Vec<u32>,
    /// max degree for normalisation
    max_degree: u32,
    /// number of memories when built
    n: usize,
}

impl CoherenceGraph {
    /// Build the coherence graph from a fully-populated MemoryStore.
    pub fn build(store: &MemoryStore, threshold: f32) -> Self {
        let n = store.len();
        let mut degree = vec![0u32; n];

        let records: Vec<_> = store.records().collect();
        for i in 0..n {
            for j in (i + 1)..n {
                let sim = cosine_sim(&records[i].vec, &records[j].vec);
                if sim >= threshold {
                    degree[i] += 1;
                    degree[j] += 1;
                }
            }
        }

        let max_degree = *degree.iter().max().unwrap_or(&1);
        Self {
            degree,
            max_degree: max_degree.max(1),
            n,
        }
    }

    /// Coherence gate value in [0, 1] for memory `id`.
    /// Returns 0 for out-of-range ids (graceful degradation).
    pub fn gate(&self, id: u64) -> f32 {
        let idx = id as usize;
        if idx >= self.n {
            return 0.0;
        }
        self.degree[idx] as f32 / self.max_degree as f32
    }

    /// Number of nodes in the graph.
    pub fn node_count(&self) -> usize {
        self.n
    }

    /// Sum of all edge degrees / 2 = number of edges.
    pub fn edge_count(&self) -> usize {
        self.degree.iter().map(|&d| d as usize).sum::<usize>() / 2
    }

    /// Mean coherence gate value across all nodes.
    pub fn mean_gate(&self) -> f32 {
        if self.n == 0 {
            return 0.0;
        }
        self.degree
            .iter()
            .map(|&d| d as f32 / self.max_degree as f32)
            .sum::<f32>()
            / self.n as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{MemoryMetadata, MemoryStore};

    fn store_with(vecs: Vec<Vec<f32>>) -> MemoryStore {
        let dims = vecs[0].len();
        let mut s = MemoryStore::new(dims);
        for v in vecs {
            s.insert(
                v,
                MemoryMetadata {
                    timestamp: 0,
                    source: "t".into(),
                    tags: vec![],
                },
            );
        }
        s
    }

    #[test]
    fn identical_vectors_are_connected() {
        let v = vec![1.0f32, 0.0, 0.0];
        let store = store_with(vec![v.clone(), v.clone(), v.clone()]);
        let g = CoherenceGraph::build(&store, 0.99);
        // Each node connects to the other 2 → degree=2 for all → gate=1.0
        assert!((g.gate(0) - 1.0).abs() < 1e-5);
        assert_eq!(g.edge_count(), 3);
    }

    #[test]
    fn orthogonal_vectors_no_edges() {
        let store = store_with(vec![
            vec![1.0f32, 0.0, 0.0],
            vec![0.0f32, 1.0, 0.0],
            vec![0.0f32, 0.0, 1.0],
        ]);
        let g = CoherenceGraph::build(&store, 0.5);
        assert_eq!(g.edge_count(), 0);
        // gate should be 0 for all (degree=0, but max_degree clamped to 1)
        assert_eq!(g.gate(0), 0.0);
    }

    #[test]
    fn mean_gate_empty() {
        let store = MemoryStore::new(4);
        let g = CoherenceGraph::build(&store, 0.5);
        assert_eq!(g.mean_gate(), 0.0);
    }
}
