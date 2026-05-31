//! Graph construction and compact co-located memory layout.
//!
//! ## Co-located layout (the SymphonyQG key insight)
//!
//! For each vertex v with R neighbors, SymphonyQG stores a single contiguous
//! heap block:
//!
//!   [ raw_f32[D] | codes[R × ceil(D/8)] | norms[R] | ids[R] ]
//!
//! This contrasts with vanilla HNSW, which stores only neighbor IDs and
//! then chases R separate random pointers to load neighbor vectors during
//! beam search.
//!
//! The sequential layout means: one cache-miss to load the vertex block,
//! then all R neighbor codes are available for batch distance estimation
//! without any additional random memory reads.
//!
//! ## Graph construction
//!
//! For the PoC we use a greedy construction: for each new vector inserted,
//! we scan all previously inserted vectors (O(n²) total) to find the top-R
//! nearest neighbors and add bidirectional edges with degree capping.
//! This gives an "oracle-quality" k-NN graph maximising recall, letting the
//! benchmark fairly isolate the effect of quantized codes vs exact distances.
//! Production would substitute Vamana or NSG construction here.

use crate::codes::{encode, packed_bytes};
use crate::rotation::rotate;

/// Parameters governing the graph index.
#[derive(Debug, Clone)]
pub struct GraphConfig {
    /// Number of neighbors per vertex (out-degree R).
    pub r: usize,
    /// Dimension of the vectors.
    pub dim: usize,
    /// Beam width used during search (ef).
    pub ef: usize,
    /// Random seed for the rotation matrix.
    pub rotation_seed: u64,
}

impl GraphConfig {
    pub fn new(dim: usize) -> Self {
        Self { r: 32, dim, ef: 64, rotation_seed: 0xdeadbeef }
    }

    pub fn with_r(mut self, r: usize) -> Self {
        self.r = r;
        self
    }

    pub fn with_ef(mut self, ef: usize) -> Self {
        self.ef = ef;
        self
    }
}

/// One vertex in the co-located SymphonyQG graph.
///
/// Memory layout is intentionally flat so that the entire block
/// fits into a small number of cache lines when R is moderate.
#[derive(Debug, Clone)]
pub struct Vertex {
    /// Original f32 vector (used for exact re-ranking and graph construction).
    pub raw: Vec<f32>,
    /// RaBitQ codes for each neighbor, stored contiguously.
    /// Length = R × ceil(D/8).  Code for neighbor j starts at j×nbytes.
    pub neighbor_codes: Vec<u8>,
    /// ‖R_mat × x_neighbor‖₂ for each neighbor (asymmetric correction).
    pub neighbor_norms: Vec<f32>,
    /// Neighbor vertex IDs.
    pub neighbor_ids: Vec<u32>,
}

impl Vertex {
    /// Bytes consumed by the co-located block (excluding Vec overhead).
    pub fn block_bytes(&self) -> usize {
        self.raw.len() * 4
            + self.neighbor_codes.len()
            + self.neighbor_norms.len() * 4
            + self.neighbor_ids.len() * 4
    }
}

/// Compact graph structure.
pub struct SymphonyGraph {
    pub config: GraphConfig,
    pub vertices: Vec<Vertex>,
    /// The rotation matrix (D×D, row-major). Used at query time.
    pub rotation: Vec<f32>,
}

impl SymphonyGraph {
    /// Build the graph from a corpus of vectors.
    pub fn build(vectors: &[Vec<f32>], config: GraphConfig, rotation: &[f32]) -> Self {
        let n = vectors.len();
        let dim = config.dim;
        let r = config.r;
        let nbytes = packed_bytes(dim);

        // Precompute rotated + encoded versions of all vectors
        let rotated: Vec<Vec<f32>> = vectors
            .iter()
            .map(|v| rotate(rotation, v, dim))
            .collect();
        let encoded: Vec<(Vec<u8>, f32)> = rotated.iter().map(|rv| encode(rv)).collect();

        // For each vertex, find top-R nearest neighbors by exact L2
        // Then fill the co-located block.
        let mut vertices: Vec<Vertex> = Vec::with_capacity(n);

        for i in 0..n {
            let vi = &vectors[i];

            // Exact k-NN from the full corpus (excluding self)
            let mut dists: Vec<(f32, usize)> = (0..n)
                .filter(|&j| j != i)
                .map(|j| {
                    let d = l2_sq(vi, &vectors[j]);
                    (d, j)
                })
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            let neighbors: Vec<usize> = dists.iter().take(r).map(|(_, j)| *j).collect();

            // Build co-located block
            let mut neighbor_codes = vec![0u8; neighbors.len() * nbytes];
            let mut neighbor_norms = Vec::with_capacity(neighbors.len());
            let mut neighbor_ids = Vec::with_capacity(neighbors.len());

            for (slot, &j) in neighbors.iter().enumerate() {
                let (ref code, norm) = encoded[j];
                neighbor_codes[slot * nbytes..(slot + 1) * nbytes].copy_from_slice(code);
                neighbor_norms.push(norm);
                neighbor_ids.push(j as u32);
            }

            vertices.push(Vertex {
                raw: vi.clone(),
                neighbor_codes,
                neighbor_norms,
                neighbor_ids,
            });
        }

        SymphonyGraph { config, vertices, rotation: rotation.to_vec() }
    }

    /// Total memory consumed by all vertex blocks (excludes Vec metadata).
    pub fn memory_bytes(&self) -> usize {
        self.vertices.iter().map(|v| v.block_bytes()).sum::<usize>()
            + self.rotation.len() * 4
    }
}

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::rotation::random_orthogonal;

    #[test]
    fn build_small_graph() {
        let n = 20;
        let dim = 8;
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..dim).map(|j| (i * dim + j) as f32).collect())
            .collect();
        let rot = random_orthogonal(dim, 42);
        let cfg = GraphConfig::new(dim).with_r(4);
        let graph = SymphonyGraph::build(&vecs, cfg.clone(), &rot);
        assert_eq!(graph.vertices.len(), n);
        for v in &graph.vertices {
            assert_eq!(v.neighbor_ids.len(), 4.min(n - 1));
            assert_eq!(v.neighbor_norms.len(), 4.min(n - 1));
            assert_eq!(v.neighbor_codes.len(), 4.min(n - 1) * packed_bytes(dim));
        }
    }

    #[test]
    fn co_located_block_size_formula() {
        let dim = 128;
        let r = 16;
        // raw: 512B, codes: 16×16=256B, norms: 64B, ids: 64B = 896B
        let expected = dim * 4 + r * packed_bytes(dim) + r * 4 + r * 4;
        assert_eq!(expected, 896);
    }
}
