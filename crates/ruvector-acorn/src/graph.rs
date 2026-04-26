use std::collections::BinaryHeap;

use crate::dist::l2_sq;
use crate::error::AcornError;

/// Ordered f32 wrapper: total ordering via `total_cmp`.
#[derive(Clone, Copy, PartialEq)]
pub struct OrdF32(pub f32);
impl Eq for OrdF32 {}
impl PartialOrd for OrdF32 {
    fn partial_cmp(&self, o: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(o))
    }
}
impl Ord for OrdF32 {
    fn cmp(&self, o: &Self) -> std::cmp::Ordering {
        self.0.total_cmp(&o.0)
    }
}

/// Greedy k-NN graph used by all ACORN variants.
///
/// Build strategy: for each node `i`, scan all previous nodes `j < i` and
/// keep the `max_neighbors` nearest. Bidirectional edges are added (each
/// node also gets at most `max_neighbors` back-edges). This gives an
/// O(n² × D) build — appropriate for the PoC scale (≤ 20 K vectors).
pub struct AcornGraph {
    /// `neighbors[i]` = sorted-by-distance list of neighbor node IDs.
    pub neighbors: Vec<Vec<u32>>,
    /// Raw vectors (owned — avoids separate lifetime parameter).
    pub data: Vec<Vec<f32>>,
    pub dim: usize,
    /// Edge budget per node (M for ACORN-1, γ·M for ACORN-γ).
    pub max_neighbors: usize,
}

impl AcornGraph {
    pub fn build(
        data: Vec<Vec<f32>>,
        max_neighbors: usize,
    ) -> Result<Self, AcornError> {
        if data.is_empty() {
            return Err(AcornError::EmptyDataset);
        }
        let dim = data[0].len();
        let n = data.len();
        let mut neighbors: Vec<Vec<u32>> = vec![Vec::new(); n];

        for i in 1..n {
            let edge_limit = max_neighbors.min(i);
            // Max-heap of (distance, id) — we keep the `edge_limit` nearest.
            let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();

            for j in 0..i {
                let d = l2_sq(&data[i], &data[j]);
                if heap.len() < edge_limit {
                    heap.push((OrdF32(d), j as u32));
                } else if let Some(&(OrdF32(worst), _)) = heap.peek() {
                    if d < worst {
                        heap.pop();
                        heap.push((OrdF32(d), j as u32));
                    }
                }
            }

            for (_, j) in heap.iter() {
                neighbors[i].push(*j);
                // Bidirectional: add i as neighbor of j if j has room.
                if neighbors[*j as usize].len() < max_neighbors {
                    neighbors[*j as usize].push(i as u32);
                }
            }
        }

        Ok(Self { neighbors, data, dim, max_neighbors })
    }

    pub fn len(&self) -> usize {
        self.data.len()
    }

    /// Estimated heap memory in bytes: edge lists + raw f32 vectors.
    pub fn memory_bytes(&self) -> usize {
        let edges: usize = self.neighbors.iter().map(|v| v.len()).sum();
        let vecs = self.data.len() * self.dim * 4;
        edges * 4 + vecs
    }
}

/// Find the `k` nearest neighbors of `query` among `data` by brute force.
/// Returns indices sorted nearest-first. Used by the post-filter baseline.
pub fn flat_k_nearest(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<u32> {
    let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::new();
    for (i, v) in data.iter().enumerate() {
        let d = l2_sq(v, query);
        if heap.len() < k {
            heap.push((OrdF32(d), i as u32));
        } else if let Some(&(OrdF32(w), _)) = heap.peek() {
            if d < w {
                heap.pop();
                heap.push((OrdF32(d), i as u32));
            }
        }
    }
    let mut out: Vec<(OrdF32, u32)> = heap.into_sorted_vec();
    out.sort_by(|a, b| a.0.cmp(&b.0));
    out.into_iter().map(|(_, id)| id).collect()
}

/// Compute exact top-k result set for recall measurement.
pub fn exact_filtered_knn(
    data: &[Vec<f32>],
    query: &[f32],
    k: usize,
    predicate: impl Fn(u32) -> bool,
) -> Vec<u32> {
    let mut scored: Vec<(OrdF32, u32)> = data
        .iter()
        .enumerate()
        .filter(|(i, _)| predicate(*i as u32))
        .map(|(i, v)| (OrdF32(l2_sq(v, query)), i as u32))
        .collect();
    scored.sort_by(|a, b| a.0.cmp(&b.0));
    scored.truncate(k);
    scored.into_iter().map(|(_, id)| id).collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_data(n: usize, d: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| (0..d).map(|j| (i * d + j) as f32 * 0.01).collect())
            .collect()
    }

    #[test]
    fn build_small_graph() {
        let data = make_data(20, 8);
        let g = AcornGraph::build(data, 4).unwrap();
        assert_eq!(g.len(), 20);
        // Every node except node 0 has at least 1 neighbor.
        for i in 1..20usize {
            assert!(!g.neighbors[i].is_empty(), "node {i} has no neighbors");
        }
    }

    #[test]
    fn flat_knn_returns_self() {
        let data: Vec<Vec<f32>> = vec![
            vec![0.0, 0.0],
            vec![1.0, 0.0],
            vec![0.0, 1.0],
            vec![10.0, 10.0],
        ];
        let query = vec![0.01_f32, 0.01];
        let nn = flat_k_nearest(&data, &query, 1);
        assert_eq!(nn[0], 0); // node 0 is [0,0] — closest
    }
}
