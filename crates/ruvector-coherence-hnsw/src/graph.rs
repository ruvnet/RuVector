//! Flat k-NN proximity graph — the HNSW layer-0 equivalent.
//!
//! ## Construction
//!
//! Two connection types are combined per node:
//!
//! * **Local edges** (count = `m`): each node's M exact nearest neighbors.
//!   Built by brute-force O(N² · D) for correctness at small N.
//!
//! * **Long-jump edges** (count = `m_longjump`): random globally-sampled
//!   connections. These act as the "navigable small world" shortcuts that
//!   HNSW achieves via its upper layers. Without long-jump edges, a fixed
//!   distant entry point gets trapped in its local cluster and recall collapses.
//!
//! With long-jump edges the graph becomes a **navigable small world**: any node
//! is reachable from any entry in O(log N) hops. The coherence gate then has
//! genuine opportunity to distinguish on-path hops (long-jumps and local edges
//! in the query's cluster) from off-path hops (local edges in the entry's
//! cluster), pruning the latter without losing recall.

use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Uniform};
use rayon::prelude::*;

/// Configuration for the proximity graph.
#[derive(Clone, Debug)]
pub struct GraphConfig {
    /// Local neighbors per node (exact k-NN).
    pub m: usize,
    /// Long-jump (random) neighbors per node — navigable small world shortcuts.
    pub m_longjump: usize,
    /// Number of dimensions per vector.
    pub dims: usize,
}

impl Default for GraphConfig {
    fn default() -> Self {
        GraphConfig {
            m: 16,
            m_longjump: 4,
            dims: 64,
        }
    }
}

impl GraphConfig {
    /// Convenience constructor for tests / benchmarks without long-jumps.
    pub fn local_only(m: usize, dims: usize) -> Self {
        GraphConfig {
            m,
            m_longjump: 0,
            dims,
        }
    }
}

/// Flat navigable small-world graph.
///
/// Each node stores `m` local + `m_longjump` random neighbors,
/// giving a total adjacency of up to `m + m_longjump` per node.
pub struct FlatGraph {
    /// Flat row-major vector store: node i lives at [i*dims .. (i+1)*dims].
    vectors: Vec<f32>,
    /// Adjacency: `neighbors[i]` = deduplicated list of node i's neighbors.
    pub neighbors: Vec<Vec<u32>>,
    pub config: GraphConfig,
    pub n: usize,
}

impl FlatGraph {
    /// Build: exact brute-force local k-NN + random long-jump edges.
    pub fn build(vectors: Vec<f32>, config: GraphConfig) -> Self {
        let n = vectors.len() / config.dims;
        assert_eq!(
            vectors.len(),
            n * config.dims,
            "vector store length mismatch"
        );
        let m = config.m.min(n.saturating_sub(1));
        let dims = config.dims;

        // ── Local k-NN (exact, parallel) ──────────────────────────────────
        let mut neighbors: Vec<Vec<u32>> = (0..n)
            .into_par_iter()
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

        // ── Long-jump edges (random, single-threaded for determinism) ──────
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

        FlatGraph {
            vectors,
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
    pub fn row(&self, i: usize) -> &[f32] {
        let d = self.config.dims;
        &self.vectors[i * d..(i + 1) * d]
    }

    pub fn len(&self) -> usize {
        self.n
    }

    pub fn is_empty(&self) -> bool {
        self.n == 0
    }
}

/// Squared L2 distance — no sqrt, consistent with HNSW conventions.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn build_connects_all_nodes() {
        let vecs: Vec<f32> = (0..20)
            .map(|i| (i as f32) / 20.0)
            .cycle()
            .take(20 * 4)
            .collect();
        let g = FlatGraph::build(
            vecs,
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
                assert!((nb as usize) < g.n, "neighbor {nb} out of bounds");
                assert_ne!(nb as usize, i, "self-loop at {i}");
            }
        }
    }

    #[test]
    fn long_jump_adds_global_connectivity() {
        // Build two back-to-back clusters: 0..5 and 5..10.
        // With only local k-NN (m=2), the two clusters are disconnected.
        // With long-jump edges, cross-cluster edges appear.
        let cluster_a: Vec<f32> = vec![
            1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.95, 0.05, 0.0, 0.0, 0.85,
            0.15, 0.0, 0.0,
        ];
        let cluster_b: Vec<f32> = vec![
            0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.9, 0.1, 0.0, 0.0, 0.8, 0.2, 0.0, 0.0, 0.95, 0.05, 0.0,
            0.0, 0.85, 0.15,
        ];
        let mut vecs = cluster_a;
        vecs.extend(cluster_b);

        // Without long-jumps — clusters should be isolated.
        let g_local = FlatGraph::build(vecs.clone(), GraphConfig::local_only(2, 4));
        let has_cross_local = g_local.neighbors[0].iter().any(|&nb| nb as usize >= 5);
        assert!(
            !has_cross_local,
            "expected no cross-cluster edges with local-only"
        );

        // With long-jumps — expect at least one cross-cluster edge.
        let g_lj = FlatGraph::build(
            vecs,
            GraphConfig {
                m: 2,
                m_longjump: 3,
                dims: 4,
            },
        );
        let has_cross_lj = (0..5).any(|i| g_lj.neighbors[i].iter().any(|&nb| nb as usize >= 5));
        assert!(has_cross_lj, "expected cross-cluster edges via long-jump");
    }
}
