//! Flat navigable-small-world draft graph.
//!
//! Structurally the same as `ruvector-coherence-hnsw`'s `FlatGraph` (exact
//! local k-NN edges + random long-jump shortcuts), but each node stores both
//! its f32 vector (for the exact verify stage) and a u8 scalar-quantized copy
//! (for cheap draft-time distance ranking during traversal) — the same
//! quantizer `ruvector-speculative-ann` uses for its linear draft.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;

use crate::{sq_l2_f32, ScalarQuantizer};

#[derive(Clone, Debug)]
pub struct GraphConfig {
    pub m: usize,
    pub m_longjump: usize,
    pub dims: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            m: 16,
            m_longjump: 4,
            dims: 128,
        }
    }
}

impl GraphConfig {
    #[cfg(test)]
    pub fn local_only(m: usize, dims: usize) -> Self {
        GraphConfig {
            m,
            m_longjump: 0,
            dims,
        }
    }
}

pub struct DraftGraph {
    f32_vectors: Vec<f32>,
    u8_vectors: Vec<u8>,
    pub neighbors: Vec<Vec<u32>>,
    pub config: GraphConfig,
    pub n: usize,
}

impl DraftGraph {
    /// Build local k-NN edges (exact f32 distance, so index quality is not
    /// penalized by quantization) plus random long-jump shortcuts, then
    /// store a u8-quantized copy of every vector for draft-time ranking.
    pub fn build(f32_vectors: Vec<f32>, quantizer: &ScalarQuantizer, config: GraphConfig) -> Self {
        let n = f32_vectors.len() / config.dims;
        assert_eq!(
            f32_vectors.len(),
            n * config.dims,
            "vector store length mismatch"
        );
        let m = config.m.min(n.saturating_sub(1));
        let dims = config.dims;

        let mut neighbors: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
            .map(|i| {
                let vi = &f32_vectors[i * dims..(i + 1) * dims];
                let mut dists: Vec<(u32, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let vj = &f32_vectors[j * dims..(j + 1) * dims];
                        (j as u32, sq_l2_f32(vi, vj))
                    })
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                dists.truncate(m);
                dists.into_iter().map(|(idx, _)| idx).collect()
            })
            .collect();

        let m_lj = config.m_longjump.min(n.saturating_sub(1));
        if m_lj > 0 {
            let mut rng = StdRng::seed_from_u64(0xBAD_C0FFEEu64);
            let dist = Uniform::new(0usize, n);
            #[allow(clippy::needless_range_loop)]
            for i in 0..n {
                let mut added = 0usize;
                let mut attempts = 0usize;
                while added < m_lj && attempts < m_lj * 8 {
                    attempts += 1;
                    let j = dist.sample(&mut rng);
                    if j != i && !neighbors[i].contains(&(j as u32)) {
                        neighbors[i].push(j as u32);
                        added += 1;
                    }
                }
            }
        }

        let u8_vectors: Vec<u8> = (0..n)
            .flat_map(|i| quantizer.quantize(&f32_vectors[i * dims..(i + 1) * dims]))
            .collect();

        DraftGraph {
            f32_vectors,
            u8_vectors,
            neighbors,
            config: GraphConfig {
                m,
                m_longjump: m_lj,
                dims,
            },
            n,
        }
    }

    #[inline]
    pub fn row_f32(&self, i: usize) -> &[f32] {
        let d = self.config.dims;
        &self.f32_vectors[i * d..(i + 1) * d]
    }

    #[inline]
    pub fn row_u8(&self, i: usize) -> &[u8] {
        let d = self.config.dims;
        &self.u8_vectors[i * d..(i + 1) * d]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }

    /// Heap bytes for both vector stores plus adjacency lists.
    pub fn memory_bytes(&self) -> usize {
        let f32_bytes = self.f32_vectors.len() * std::mem::size_of::<f32>();
        let u8_bytes = self.u8_vectors.len();
        let adj_bytes: usize = self
            .neighbors
            .iter()
            .map(|v| v.len() * std::mem::size_of::<u32>())
            .sum();
        f32_bytes + u8_bytes + adj_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_quantizer(vecs: &[Vec<f32>]) -> ScalarQuantizer {
        ScalarQuantizer::train(vecs)
    }

    #[test]
    fn build_connects_all_nodes() {
        let vecs: Vec<f32> = (0..20)
            .map(|i| (i as f32) / 20.0)
            .cycle()
            .take(80)
            .collect();
        let rows: Vec<Vec<f32>> = vecs.chunks(4).map(|c| c.to_vec()).collect();
        let q = make_quantizer(&rows);
        let g = DraftGraph::build(
            vecs,
            &q,
            GraphConfig {
                m: 3,
                m_longjump: 1,
                dims: 4,
            },
        );
        assert_eq!(g.n, 20);
        for (i, nbrs) in g.neighbors.iter().enumerate() {
            assert!(!nbrs.is_empty(), "node {i} has no neighbors");
            for &nb in nbrs {
                assert!((nb as usize) < g.n);
                assert_ne!(nb as usize, i);
            }
        }
    }

    #[test]
    fn row_u8_and_f32_match_quantizer() {
        let rows = vec![vec![0.0, 0.0], vec![10.0, 10.0]];
        let flat: Vec<f32> = rows.iter().flatten().copied().collect();
        let q = make_quantizer(&rows);
        let g = DraftGraph::build(flat, &q, GraphConfig::local_only(1, 2));
        assert_eq!(g.row_u8(1), q.quantize(&rows[1]).as_slice());
    }
}
