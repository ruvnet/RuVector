//! Two-level cluster tree (leaf vectors → cluster centroids).
//!
//! Level 0: raw agent-memory vectors (leaves).
//! Level 1: cluster centroids (one per k-means cluster).
//!
//! The tree enables both IVF-style and coherence-weighted retrieval.

use crate::cluster::{build_inverted_lists, KMeansResult};

/// Immutable two-level cluster tree built from a corpus of vectors.
pub struct ClusterTree {
    /// Original leaf vectors (length = n).
    pub leaves: Vec<Vec<f32>>,
    /// Level-1 centroids (length = k).
    pub centroids: Vec<Vec<f32>>,
    /// Per-cluster cohesion ∈ [-1, 1] (length = k).
    pub cohesion: Vec<f32>,
    /// Inverted lists: cluster_id → [leaf_ids] (length = k).
    pub inverted: Vec<Vec<usize>>,
    /// Number of clusters.
    pub k: usize,
}

impl ClusterTree {
    /// Build a tree from a corpus and a completed k-means result.
    pub fn new(leaves: Vec<Vec<f32>>, km: KMeansResult) -> Self {
        let k = km.centroids.len();
        let inverted = build_inverted_lists(&km.assignments, k);
        Self {
            leaves,
            centroids: km.centroids,
            cohesion: km.cohesion,
            inverted,
            k,
        }
    }

    /// Approximate memory footprint in bytes.
    ///
    /// Leaf storage + centroid storage + inverted-list index overhead.
    pub fn mem_bytes(&self) -> usize {
        let n = self.leaves.len();
        let dim = if n > 0 { self.leaves[0].len() } else { 0 };
        let leaf_bytes = n * dim * 4;
        let centroid_bytes = self.k * dim * 4;
        let inv_bytes = self.inverted.iter().map(|l| l.len() * 8).sum::<usize>();
        leaf_bytes + centroid_bytes + inv_bytes
    }

    /// Number of leaf vectors.
    pub fn len(&self) -> usize {
        self.leaves.len()
    }

    pub fn is_empty(&self) -> bool {
        self.leaves.is_empty()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::{cluster::kmeans, dataset::generate_vectors};

    fn build_tree(n: usize, dim: usize, k: usize) -> ClusterTree {
        let vecs = generate_vectors(n, dim, 99);
        let km = kmeans(&vecs, k, 15);
        ClusterTree::new(vecs, km)
    }

    #[test]
    fn tree_covers_all_leaves() {
        let tree = build_tree(500, 32, 10);
        let covered: usize = tree.inverted.iter().map(|l| l.len()).sum();
        assert_eq!(covered, 500, "all leaves must appear in inverted lists");
    }

    #[test]
    fn centroid_count_matches_k() {
        let tree = build_tree(500, 32, 10);
        assert_eq!(tree.centroids.len(), 10);
        assert_eq!(tree.cohesion.len(), 10);
    }

    #[test]
    fn mem_bytes_positive() {
        let tree = build_tree(200, 16, 4);
        assert!(tree.mem_bytes() > 0);
    }

    #[test]
    fn every_inverted_id_is_valid() {
        let tree = build_tree(300, 32, 8);
        let n = tree.leaves.len();
        for list in &tree.inverted {
            for &id in list {
                assert!(id < n, "leaf id {id} out of range [0, {n})");
            }
        }
    }
}
