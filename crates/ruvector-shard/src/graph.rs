use crate::error::{ShardError, ShardResult};

/// Configuration for building the k-NN proximity graph.
pub struct GraphConfig {
    /// Number of neighbors per node in the graph.
    pub k_neighbors: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        Self { k_neighbors: 16 }
    }
}

/// A k-nearest-neighbor proximity graph over float32 vectors.
///
/// Built via exact brute-force search — correct but O(n² × d). Suitable for
/// proof-of-concept with n ≤ 8 192 on modern hardware.
pub struct KnnGraph {
    pub n: usize,
    pub dim: usize,
    /// Row-major storage: `vectors[i * dim .. (i+1) * dim]` is node i.
    pub vectors: Vec<f32>,
    /// `neighbors[i]` holds the k nearest neighbor IDs for node i, sorted by
    /// ascending distance (closest first).
    pub neighbors: Vec<Vec<u32>>,
}

impl KnnGraph {
    /// Build an exact k-NN graph from the given row-major vector slice.
    pub fn build(vectors: Vec<f32>, dim: usize, config: &GraphConfig) -> ShardResult<KnnGraph> {
        if dim == 0 || vectors.len() % dim != 0 {
            return Err(ShardError::InvalidDimension);
        }
        let n = vectors.len() / dim;
        if n == 0 {
            return Err(ShardError::EmptyGraph);
        }
        let k = config.k_neighbors.min(n.saturating_sub(1));

        let neighbors: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let vi = &vectors[i * dim..(i + 1) * dim];
                let mut dists: Vec<(f32, u32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let vj = &vectors[j * dim..(j + 1) * dim];
                        (cosine_distance(vi, vj), j as u32)
                    })
                    .collect();
                // Partial sort: only need the k closest.
                dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap_or(std::cmp::Ordering::Equal));
                dists.truncate(k);
                dists.into_iter().map(|(_, id)| id).collect()
            })
            .collect();

        Ok(KnnGraph {
            n,
            dim,
            vectors,
            neighbors,
        })
    }

    /// Slice the raw vector for node `idx`.
    #[inline]
    pub fn get_vector(&self, idx: usize) -> &[f32] {
        &self.vectors[idx * self.dim..(idx + 1) * self.dim]
    }

    /// Count how many times each node appears as a neighbor (incoming degree).
    /// High incoming degree ≈ hub node ≈ upper HNSW layer.
    pub fn incoming_degree(&self) -> Vec<u32> {
        let mut degree = vec![0u32; self.n];
        for nlist in &self.neighbors {
            for &nb in nlist {
                degree[nb as usize] = degree[nb as usize].saturating_add(1);
            }
        }
        degree
    }

    /// Heap-allocated bytes consumed by this graph (approximate).
    pub fn memory_bytes(&self) -> usize {
        let vec_bytes = self.n * self.dim * std::mem::size_of::<f32>();
        let nb_bytes: usize = self.neighbors.iter().map(|nl| nl.len() * 4).sum();
        vec_bytes + nb_bytes
    }
}

/// Cosine distance in [0, 2].  Returns 1.0 for zero vectors.
pub fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0f32;
    let mut na = 0.0f32;
    let mut nb = 0.0f32;
    // 4× unrolled for ILP
    let chunks = a.len() / 4;
    for i in 0..chunks {
        let base = i * 4;
        dot += a[base] * b[base]
            + a[base + 1] * b[base + 1]
            + a[base + 2] * b[base + 2]
            + a[base + 3] * b[base + 3];
        na += a[base] * a[base]
            + a[base + 1] * a[base + 1]
            + a[base + 2] * a[base + 2]
            + a[base + 3] * a[base + 3];
        nb += b[base] * b[base]
            + b[base + 1] * b[base + 1]
            + b[base + 2] * b[base + 2]
            + b[base + 3] * b[base + 3];
    }
    for i in (chunks * 4)..a.len() {
        dot += a[i] * b[i];
        na += a[i] * a[i];
        nb += b[i] * b[i];
    }
    let denom = (na * nb).sqrt();
    if denom < 1e-12 {
        1.0
    } else {
        (1.0 - dot / denom).clamp(0.0, 2.0)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_distance_identical() {
        let v = vec![1.0f32, 0.0, 0.0];
        assert!(
            cosine_distance(&v, &v) < 1e-6,
            "identical vectors → distance ≈ 0"
        );
    }

    #[test]
    fn cosine_distance_orthogonal() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let d = cosine_distance(&a, &b);
        assert!(
            (d - 1.0).abs() < 1e-5,
            "orthogonal vectors → distance ≈ 1, got {d}"
        );
    }

    #[test]
    fn build_small_graph() {
        let vectors: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 0.0];
        let graph = KnnGraph::build(vectors, 3, &GraphConfig { k_neighbors: 2 }).unwrap();
        assert_eq!(graph.n, 4);
        for nlist in &graph.neighbors {
            assert_eq!(nlist.len(), 2);
        }
    }

    #[test]
    fn incoming_degree_nonzero() {
        let vectors: Vec<f32> = (0..16).flat_map(|i| vec![i as f32, 0.0]).collect();
        let graph = KnnGraph::build(vectors, 2, &GraphConfig { k_neighbors: 2 }).unwrap();
        let deg = graph.incoming_degree();
        let total: u32 = deg.iter().sum();
        // Each node has 2 outgoing edges; total incoming = 2 * n
        assert_eq!(total, 2 * graph.n as u32);
    }
}
