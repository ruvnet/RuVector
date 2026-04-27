use crate::dist::l2_sq;
use crate::error::FingerError;

#[cfg(not(target_arch = "wasm32"))]
use rayon::prelude::*;

/// Trait for a graph that FINGER can run on.
///
/// Implementors provide read-only access to node vectors and neighbor lists.
/// `Sync` is required so that rayon can parallelize basis construction across nodes.
pub trait GraphWalk: Sync {
    fn n_nodes(&self) -> usize;
    fn dim(&self) -> usize;
    /// Neighbor node IDs for `node_id`.
    fn neighbors(&self, node_id: usize) -> &[u32];
    /// Embedding vector for `node_id`.
    fn vector(&self, node_id: usize) -> &[f32];
    /// Entry node for beam search.
    fn entry_point(&self) -> u32;
}

/// Flat greedy k-NN graph built by brute-force O(N² × D) neighbor search.
///
/// Designed for correctness and transparency in benchmarks — not for production
/// indexing at N > 50K. Parallel build via rayon.
pub struct FlatGraph {
    data: Vec<f32>,
    neighbors: Vec<Vec<u32>>,
    dim: usize,
    n: usize,
    entry: u32,
}

impl FlatGraph {
    /// Build a flat k-NN graph with `m` neighbors per node.
    pub fn build(data: &[Vec<f32>], m: usize) -> Result<Self, FingerError> {
        let n = data.len();
        if n == 0 {
            return Err(FingerError::EmptyDataset);
        }
        let dim = data[0].len();
        for (i, row) in data.iter().enumerate() {
            if row.len() != dim {
                return Err(FingerError::DimMismatch { expected: dim, got: row.len() });
            }
            let _ = i;
        }

        // Flatten into contiguous buffer for cache-friendly distance scanning.
        let flat: Vec<f32> = data.iter().flat_map(|v| v.iter().copied()).collect();

        let neighbors = Self::build_neighbors(&flat, n, dim, m);

        Ok(FlatGraph { data: flat, neighbors, dim, n, entry: 0 })
    }

    #[cfg(not(target_arch = "wasm32"))]
    fn build_neighbors(flat: &[f32], n: usize, dim: usize, m: usize) -> Vec<Vec<u32>> {
        let row = |i: usize| -> &[f32] { &flat[i * dim..(i + 1) * dim] };
        let m_actual = m.min(n.saturating_sub(1));
        (0..n)
            .into_par_iter()
            .map(|i| {
                let ri = row(i);
                let mut dists: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (l2_sq(ri, row(j)), j as u32))
                    .collect();
                dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(m_actual).map(|(_, j)| *j).collect()
            })
            .collect()
    }

    #[cfg(target_arch = "wasm32")]
    fn build_neighbors(flat: &[f32], n: usize, dim: usize, m: usize) -> Vec<Vec<u32>> {
        let row = |i: usize| -> &[f32] { &flat[i * dim..(i + 1) * dim] };
        let m_actual = m.min(n.saturating_sub(1));
        (0..n)
            .map(|i| {
                let ri = row(i);
                let mut dists: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| (l2_sq(ri, row(j)), j as u32))
                    .collect();
                dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                dists.iter().take(m_actual).map(|(_, j)| *j).collect()
            })
            .collect()
    }

    /// Compute recall@k: fraction of ground-truth top-k found in `retrieved`.
    pub fn brute_force_knn(&self, query: &[f32], k: usize) -> Vec<u32> {
        let mut dists: Vec<(f32, u32)> = (0..self.n)
            .map(|i| (l2_sq(query, self.vector(i)), i as u32))
            .collect();
        dists.sort_unstable_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
        dists.iter().take(k).map(|(_, id)| *id).collect()
    }
}

impl GraphWalk for FlatGraph {
    fn n_nodes(&self) -> usize {
        self.n
    }

    fn dim(&self) -> usize {
        self.dim
    }

    fn neighbors(&self, node_id: usize) -> &[u32] {
        &self.neighbors[node_id]
    }

    fn vector(&self, node_id: usize) -> &[f32] {
        let start = node_id * self.dim;
        &self.data[start..start + self.dim]
    }

    fn entry_point(&self) -> u32 {
        self.entry
    }
}

/// Compute recall@k: fraction of ground_truth IDs found in retrieved.
pub fn recall_at_k(retrieved: &[u32], ground_truth: &[u32], k: usize) -> f64 {
    let gt: std::collections::HashSet<u32> = ground_truth.iter().take(k).copied().collect();
    let hits = retrieved.iter().take(k).filter(|id| gt.contains(id)).count();
    hits as f64 / gt.len().min(k) as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let mut v = vec![0.0f32; dim];
                v[i % dim] = (i + 1) as f32;
                v
            })
            .collect()
    }

    #[test]
    fn test_build_basic() {
        let data = make_data(10, 4);
        let g = FlatGraph::build(&data, 3).unwrap();
        assert_eq!(g.n_nodes(), 10);
        assert_eq!(g.dim(), 4);
        for i in 0..10 {
            assert!(g.neighbors(i).len() <= 3);
        }
    }

    #[test]
    fn test_vector_retrieval() {
        let data = vec![vec![1.0f32, 2.0], vec![3.0, 4.0]];
        let g = FlatGraph::build(&data, 1).unwrap();
        assert_eq!(g.vector(0), &[1.0f32, 2.0]);
        assert_eq!(g.vector(1), &[3.0f32, 4.0]);
    }

    #[test]
    fn test_brute_force_knn_ordering() {
        let data = vec![
            vec![0.0f32, 0.0],
            vec![1.0, 0.0],
            vec![10.0, 0.0],
        ];
        let g = FlatGraph::build(&data, 2).unwrap();
        let query = vec![0.5f32, 0.0];
        let knn = g.brute_force_knn(&query, 2);
        // Closest to (0.5,0): node 0 at dist 0.25, node 1 at dist 0.25, node 2 far
        assert!(knn.contains(&0) || knn.contains(&1));
    }

    #[test]
    fn test_recall_perfect() {
        let retrieved = vec![0u32, 1, 2];
        let gt = vec![0u32, 1, 2];
        assert!((recall_at_k(&retrieved, &gt, 3) - 1.0).abs() < 1e-9);
    }

    #[test]
    fn test_recall_zero() {
        let retrieved = vec![3u32, 4, 5];
        let gt = vec![0u32, 1, 2];
        assert!((recall_at_k(&retrieved, &gt, 3)).abs() < 1e-9);
    }
}
