//! Synthetic planted-partition graphs for testing, benchmarking, and demos.
//!
//! Produces a graph with `num_communities` ground-truth communities: dense,
//! heavy intra-community edges and sparse, light inter-community edges, with
//! each community's embeddings drawn around a distinct centroid and sharing a
//! class label. This is the canonical stress test for a structure-preserving
//! condenser — a good condenser should recover the planted communities.

use crate::features::NodeFeatures;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use ruvector_mincut::DynamicGraph;

/// Parameters for a planted-partition (stochastic-block-model-style) graph.
#[derive(Debug, Clone)]
pub struct PlantedPartition {
    /// Number of ground-truth communities.
    pub num_communities: usize,
    /// Vertices per community.
    pub community_size: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Probability of an edge between two vertices in the same community.
    pub p_intra: f64,
    /// Probability of an edge between two vertices in different communities.
    pub p_inter: f64,
    /// Weight assigned to intra-community edges.
    pub w_intra: f64,
    /// Weight assigned to inter-community edges.
    pub w_inter: f64,
    /// RNG seed for reproducibility.
    pub seed: u64,
}

impl Default for PlantedPartition {
    fn default() -> Self {
        Self {
            num_communities: 8,
            community_size: 32,
            dim: 16,
            p_intra: 0.4,
            p_inter: 0.002,
            w_intra: 1.0,
            w_inter: 0.1,
            seed: 0xC0FFEE,
        }
    }
}

impl PlantedPartition {
    /// Total vertex count.
    pub fn total_vertices(&self) -> usize {
        self.num_communities * self.community_size
    }

    /// Generate the graph and matching [`NodeFeatures`].
    ///
    /// Vertices are numbered `0..total_vertices`; community `c` owns the
    /// contiguous block `[c*size, (c+1)*size)`. Every vertex receives an
    /// embedding (so condensation never hits a missing feature) clustered
    /// around its community centroid, plus that community's class label.
    pub fn generate(&self) -> (DynamicGraph, NodeFeatures) {
        let mut rng = StdRng::seed_from_u64(self.seed);
        let n = self.total_vertices();
        let graph = DynamicGraph::with_capacity(n, n * 4);
        let mut features = NodeFeatures::new(self.dim, self.num_communities);

        // Community centroids spaced far apart so feature space mirrors topology.
        let centroids: Vec<Vec<f32>> = (0..self.num_communities)
            .map(|c| {
                let mut v = vec![0f32; self.dim];
                v[c % self.dim] = 10.0 * (c / self.dim + 1) as f32;
                v
            })
            .collect();

        for (c, centroid) in centroids.iter().enumerate() {
            for i in 0..self.community_size {
                let vid = (c * self.community_size + i) as u64;
                let mut emb = centroid.clone();
                for x in &mut emb {
                    *x += rng.gen_range(-1.0..1.0);
                }
                // set() only fails on dimension mismatch, which cannot happen here.
                let _ = features.set(vid, emb, c);
                graph.add_vertex(vid);
            }
        }

        // Edges. insert_edge dedups and rejects self-loops, so collisions are
        // simply skipped.
        for a in 0..n {
            for b in (a + 1)..n {
                let same = a / self.community_size == b / self.community_size;
                let (p, w) = if same {
                    (self.p_intra, self.w_intra)
                } else {
                    (self.p_inter, self.w_inter)
                };
                if rng.gen_bool(p) {
                    let _ = graph.insert_edge(a as u64, b as u64, w);
                }
            }
        }

        (graph, features)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condense::condense;

    #[test]
    fn generates_requested_size() {
        let pp = PlantedPartition {
            num_communities: 4,
            community_size: 10,
            ..Default::default()
        };
        let (g, f) = pp.generate();
        assert_eq!(g.num_vertices(), 40);
        assert_eq!(f.len(), 40);
        assert_eq!(f.num_classes(), 4);
    }

    #[test]
    fn condenser_recovers_planted_structure() {
        // Strong planted structure should condense to roughly the planted count
        // and keep most weight intra-region.
        let pp = PlantedPartition {
            num_communities: 4,
            community_size: 24,
            dim: 8,
            p_intra: 0.6,
            p_inter: 0.001,
            seed: 7,
            ..Default::default()
        };
        let (g, f) = pp.generate();
        let c = condense(&g, &f).unwrap();
        assert_eq!(c.source_nodes, 96);
        // Recursive min-cut can over-split; expect at least the planted count
        // and a strong reduction.
        assert!(c.node_count() >= 4);
        assert!(c.node_reduction_ratio() > 2.0);
        let m = crate::metrics::evaluate(&g, &c);
        assert!(
            m.intra_weight_ratio > 0.8,
            "intra ratio {}",
            m.intra_weight_ratio
        );
    }
}
