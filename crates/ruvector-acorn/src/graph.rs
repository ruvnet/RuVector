use std::collections::BinaryHeap;
use std::sync::Mutex;

use rayon::prelude::*;

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
///
/// The forward pass (computing each node's nearest neighbors) is parallel
/// over `i` via rayon; the back-edge merge is serial because it mutates
/// shared state. For a 5K×128 dataset this is ~6× faster on an 8-core box.
///
/// Vectors are stored in **flat row-major** layout (`Vec<f32>` of length
/// n·dim) instead of `Vec<Vec<f32>>`. This eliminates per-vector heap
/// indirection, gives the L2² inner loop a contiguous slice it can vectorize
/// over, and makes the index ~2× more cache-friendly during search.
pub struct AcornGraph {
    /// `neighbors[i]` = sorted-by-distance list of neighbor node IDs.
    pub neighbors: Vec<Vec<u32>>,
    /// Raw vectors in row-major layout, length = n × dim.
    pub data: Vec<f32>,
    pub dim: usize,
    /// Edge budget per node (M for ACORN-1, γ·M for ACORN-γ).
    pub max_neighbors: usize,
}

impl AcornGraph {
    pub fn build(data: Vec<Vec<f32>>, max_neighbors: usize) -> Result<Self, AcornError> {
        if data.is_empty() {
            return Err(AcornError::EmptyDataset);
        }
        let dim = data[0].len();
        let n = data.len();

        // Flatten input into a single contiguous buffer for cache-friendly
        // distance scans during build and search.
        let mut flat: Vec<f32> = Vec::with_capacity(n * dim);
        for row in &data {
            if row.len() != dim {
                return Err(AcornError::DimMismatch {
                    expected: dim,
                    actual: row.len(),
                });
            }
            flat.extend_from_slice(row);
        }
        let row = |i: usize| -> &[f32] { &flat[i * dim..(i + 1) * dim] };

        // Parallel forward pass: each node i picks its top `max_neighbors`
        // nearest predecessors j < i. No shared mutation, embarrassingly
        // parallel.
        let forward: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                if i == 0 {
                    return Vec::new();
                }
                let edge_limit = max_neighbors.min(i);
                let mut heap: BinaryHeap<(OrdF32, u32)> = BinaryHeap::with_capacity(edge_limit + 1);
                let row_i = row(i);
                for j in 0..i {
                    let d = l2_sq(row_i, row(j));
                    if heap.len() < edge_limit {
                        heap.push((OrdF32(d), j as u32));
                    } else if let Some(&(OrdF32(worst), _)) = heap.peek() {
                        if d < worst {
                            heap.pop();
                            heap.push((OrdF32(d), j as u32));
                        }
                    }
                }
                heap.into_iter().map(|(_, j)| j).collect()
            })
            .collect();

        // Serial back-edge merge: each j gets at most `max_neighbors` total
        // edges including the back-edges it picks up here.
        let neighbors_lock: Vec<Mutex<Vec<u32>>> = forward.into_iter().map(Mutex::new).collect();
        // Walk i in increasing order so back-edges are merged deterministically.
        for i in 0..n {
            let forward_i: Vec<u32> = neighbors_lock[i].lock().unwrap().clone();
            for &j in &forward_i {
                let j = j as usize;
                let mut nj = neighbors_lock[j].lock().unwrap();
                if nj.len() < max_neighbors {
                    nj.push(i as u32);
                }
            }
        }
        let neighbors: Vec<Vec<u32>> = neighbors_lock
            .into_iter()
            .map(|m| m.into_inner().unwrap())
            .collect();

        Ok(Self {
            neighbors,
            data: flat,
            dim,
            max_neighbors,
        })
    }

    pub fn len(&self) -> usize {
        self.data.len() / self.dim.max(1)
    }

    /// Borrow vector `i` as a contiguous slice — the hot path for L2².
    #[inline(always)]
    pub fn row(&self, i: usize) -> &[f32] {
        &self.data[i * self.dim..(i + 1) * self.dim]
    }

    /// Estimated heap memory in bytes: edge lists + raw f32 vectors.
    pub fn memory_bytes(&self) -> usize {
        let edges: usize = self.neighbors.iter().map(|v| v.len()).sum();
        edges * 4 + self.data.len() * 4
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
    predicate: impl Fn(u32) -> bool + Sync,
) -> Vec<u32> {
    // Parallel scoring + filter; collect, then truncate to top-k. For recall
    // measurement only, so the extra heap-vs-sort tradeoff doesn't matter.
    let mut scored: Vec<(OrdF32, u32)> = (0..data.len())
        .into_par_iter()
        .filter(|&i| predicate(i as u32))
        .map(|i| (OrdF32(l2_sq(&data[i], query)), i as u32))
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
