//! Flat navigable small-world proximity graph (zero dependencies).
//!
//! Each node connects to M exact nearest neighbours (local edges) plus
//! M_longjump random nodes (long-jump edges). The random long-jumps give
//! the navigable small-world property needed for greedy beam search.
//!
//! Build is O(N² × D) — correct and deterministic at PoC scale (N ≤ 5000).

use crate::dataset::Lcg;

/// Graph construction parameters.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Local k-NN neighbor count (exact brute-force).
    pub m: usize,
    /// Random long-jump neighbor count (navigability shortcuts).
    pub m_longjump: usize,
    /// Vector dimensionality.
    pub dims: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig { m: 16, m_longjump: 6, dims: 128 }
    }
}

/// Flat navigable small-world graph.
pub struct FlatGraph {
    /// Row-major vector store: node i at `vectors[i*dims..(i+1)*dims]`.
    pub vectors: Vec<f32>,
    /// Adjacency: `neighbors[i]` = neighbor node ids.
    pub neighbors: Vec<Vec<u32>>,
    pub config: GraphConfig,
    pub n: usize,
}

impl FlatGraph {
    /// Build the graph from a flat vector buffer.
    pub fn build(vectors: Vec<f32>, config: GraphConfig) -> Self {
        let n = vectors.len() / config.dims;
        assert_eq!(vectors.len(), n * config.dims, "vector length / dims mismatch");
        let m = config.m.min(n.saturating_sub(1));
        let dims = config.dims;

        // ── Local k-NN edges (sequential brute-force) ─────────────────────
        let mut neighbors: Vec<Vec<u32>> = (0..n)
            .map(|i| {
                let vi = &vectors[i * dims..(i + 1) * dims];
                let mut dists: Vec<(u32, f32)> = (0..n)
                    .filter(|&j| j != i)
                    .map(|j| {
                        let vj = &vectors[j * dims..(j + 1) * dims];
                        (j as u32, l2_sq(vi, vj))
                    })
                    .collect();
                dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
                dists.truncate(m);
                dists.into_iter().map(|(idx, _)| idx).collect()
            })
            .collect();

        // ── Long-jump (random) edges ───────────────────────────────────────
        let lj = config.m_longjump.min(n.saturating_sub(1));
        if lj > 0 {
            let mut rng = Lcg::new(0xBEEF_CAFE_0001);
            for i in 0..n {
                let mut added = 0;
                let mut attempts = 0usize;
                while added < lj && attempts < lj * 20 {
                    let j = rng.next_usize(n);
                    attempts += 1;
                    if j != i && !neighbors[i].contains(&(j as u32)) {
                        neighbors[i].push(j as u32);
                        added += 1;
                    }
                }
            }
        }

        FlatGraph { vectors, neighbors, config, n }
    }

    /// Vector slice for node `i`.
    #[inline]
    pub fn vector(&self, i: usize) -> &[f32] {
        let d = self.config.dims;
        &self.vectors[i * d..(i + 1) * d]
    }
}

/// Squared L2 distance (no sqrt — monotone for ranking).
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
