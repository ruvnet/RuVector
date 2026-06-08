//! The condensation engine: partition a feature graph into structural regions
//! and collapse each region into a representative super-node.
//!
//! Unlike gradient-/distribution-matching condensation (GCond, GCDM, SFGC),
//! which *synthesise* a small graph by optimising a learning objective, this is
//! a **structure-preserving** condenser: regions come from the dynamic min-cut
//! community structure, so the condensed topology mirrors the real cut
//! structure of the source graph. Boundary edges survive as weighted
//! super-edges; cuts are preserved by construction rather than by training.

use crate::diffcut::{DiffCutCondenser, DiffCutConfig};
use crate::error::{CondenseError, Result};
use crate::features::NodeFeatures;
use crate::node::{CondensedEdge, CondensedGraph, CondensedNode};
use crate::regions::{
    centroid_and_medoid, class_distribution, ensure_coverage, l2_normalize, weak_boundary_regions,
};
use ruvector_mincut::{CommunityDetector, DynamicGraph, GraphPartitioner, VertexId};
use std::collections::HashMap;
use std::sync::Arc;

/// How the source graph is partitioned into regions before collapsing.
///
/// Note: region detection only decides *membership*. Super-edges are always
/// rebuilt from the original graph's edges, so structure preservation does not
/// depend on the method chosen here.
#[derive(Debug, Clone, PartialEq)]
pub enum CondenseMethod {
    /// **Default.** Cut every edge lighter than `relative_threshold * mean
    /// edge weight`, then take the connected components of what remains. This
    /// is a one-shot approximation to removing the light min-cut boundaries:
    /// robust, deterministic, and effective whenever intra-community edges are
    /// heavier than inter ones. With near-uniform weights it degrades gracefully
    /// to [`CondenseMethod::ConnectedComponents`].
    WeakBoundary {
        /// Fraction of the mean edge weight below which an edge is treated as a
        /// boundary and removed. `0.5` is a sensible default.
        relative_threshold: f64,
    },
    /// Recursive min-cut community detection via
    /// [`ruvector_mincut::CommunityDetector`]. Structure-aware for graphs with
    /// clear bottlenecks, but recursive *global* min cut tends to peel off
    /// single low-degree vertices otherwise (many tiny regions); prefer
    /// [`CondenseMethod::WeakBoundary`]. `min_region_size` bounds recursion.
    MinCutCommunity {
        /// Recursion stops splitting regions at or below this size.
        min_region_size: usize,
    },
    /// Recursive bisection into up to `num_regions` regions via
    /// [`ruvector_mincut::GraphPartitioner`]. Effective on clustered graphs;
    /// reduction is graph-dependent (the bisection can peel single vertices,
    /// which become singleton regions). Prefer [`CondenseMethod::WeakBoundary`].
    Partition {
        /// Target number of regions.
        num_regions: usize,
    },
    /// **Differentiable min-cut** (relaxed normalized cut, MinCutPool-style):
    /// learns a soft `N×K` assignment by gradient descent on a cut +
    /// orthogonality loss, then hardens it (argmax) into regions. The only
    /// method whose regions are *trained* to preserve the cut — see
    /// [`crate::diffcut`]. `K` upper-bounds the super-node count.
    DiffMinCut(DiffCutConfig),
    /// Cheap baseline: one region per connected component.
    ConnectedComponents,
}

impl Default for CondenseMethod {
    fn default() -> Self {
        CondenseMethod::WeakBoundary {
            relative_threshold: 0.5,
        }
    }
}

/// Configuration for [`GraphCondenser`].
///
/// `Default` yields [`CondenseMethod::WeakBoundary`] with a `0.5` threshold and
/// no centroid normalisation.
#[derive(Debug, Clone, Default)]
pub struct CondenseConfig {
    /// Region partitioning strategy.
    pub method: CondenseMethod,
    /// L2-normalise centroids after averaging (useful for cosine-space
    /// embeddings such as HNSW vectors).
    pub normalize_centroids: bool,
}

/// Stateless condenser. Construct once with a [`CondenseConfig`] and reuse
/// across graphs.
#[derive(Debug, Clone, Default)]
pub struct GraphCondenser {
    config: CondenseConfig,
}

impl GraphCondenser {
    /// Create a condenser with the given configuration.
    pub fn new(config: CondenseConfig) -> Self {
        Self { config }
    }

    /// Borrow the active configuration.
    pub fn config(&self) -> &CondenseConfig {
        &self.config
    }

    /// Condense `graph` using the per-vertex `features`.
    ///
    /// Every vertex in `graph` must have an embedding in `features` (a vertex
    /// with no incident edges is still condensed, as a singleton region).
    ///
    /// # Errors
    /// - [`CondenseError::EmptyGraph`] if the graph has no vertices.
    /// - [`CondenseError::MissingFeature`] if a vertex lacks an embedding.
    /// - [`CondenseError::InvalidConfig`] for a degenerate configuration.
    pub fn condense(
        &self,
        graph: &DynamicGraph,
        features: &NodeFeatures,
    ) -> Result<CondensedGraph> {
        let vertices = graph.vertices();
        if vertices.is_empty() {
            return Err(CondenseError::EmptyGraph);
        }
        let dim = features.dim();
        let num_classes = features.num_classes();

        // 1. Partition into structural regions, then guarantee full coverage
        //    and a deterministic ordering (region id == position).
        let mut regions = self.partition_regions(graph)?;
        ensure_coverage(&mut regions, &vertices);
        for r in &mut regions {
            r.sort_unstable();
        }
        regions.retain(|r| !r.is_empty());
        regions.sort_by(|a, b| a[0].cmp(&b[0]));

        // 2. Vertex -> region index.
        let mut region_of: HashMap<VertexId, usize> = HashMap::with_capacity(vertices.len());
        for (ri, members) in regions.iter().enumerate() {
            for &v in members {
                region_of.insert(v, ri);
            }
        }

        // 3. Single edge pass: internal vs boundary weight (for coherence) and
        //    super-edge accumulation.
        let n = regions.len();
        let mut internal_w = vec![0f64; n];
        let mut boundary_w = vec![0f64; n];
        let mut super_edges: HashMap<(usize, usize), (f64, u32)> = HashMap::new();

        for e in graph.edges() {
            // region_of is total over graph vertices after ensure_coverage.
            let rs = region_of[&e.source];
            let rt = region_of[&e.target];
            if rs == rt {
                internal_w[rs] += e.weight;
            } else {
                boundary_w[rs] += e.weight;
                boundary_w[rt] += e.weight;
                let key = if rs < rt { (rs, rt) } else { (rt, rs) };
                let slot = super_edges.entry(key).or_insert((0.0, 0));
                slot.0 += e.weight;
                slot.1 += 1;
            }
        }

        // 4. Build super-nodes.
        let mut nodes = Vec::with_capacity(n);
        for (ri, members) in regions.iter().enumerate() {
            let (mut centroid, representative) = centroid_and_medoid(members, features, dim)?;
            if self.config.normalize_centroids {
                l2_normalize(&mut centroid);
            }
            let class_distribution = class_distribution(members, features, num_classes);
            let iw = internal_w[ri];
            let bw = boundary_w[ri];
            let coherence = if iw + bw <= 0.0 {
                1.0
            } else {
                (iw / (iw + bw)) as f32
            };
            nodes.push(CondensedNode {
                id: ri as u64,
                centroid,
                weight: members.len() as u32,
                class_distribution,
                coherence,
                representative,
                members: members.clone(),
            });
        }

        // 5. Build super-edges (region index == id), canonical & sorted.
        let mut edges: Vec<CondensedEdge> = super_edges
            .into_iter()
            .map(|((s, t), (w, c))| CondensedEdge {
                source: s as u64,
                target: t as u64,
                weight: w,
                crossings: c,
            })
            .collect();
        edges.sort_by_key(|e| (e.source, e.target));

        Ok(CondensedGraph {
            nodes,
            edges,
            source_nodes: vertices.len(),
            source_edges: graph.num_edges(),
            dim,
            num_classes,
        })
    }

    fn partition_regions(&self, graph: &DynamicGraph) -> Result<Vec<Vec<VertexId>>> {
        match &self.config.method {
            CondenseMethod::ConnectedComponents => Ok(graph.connected_components()),
            CondenseMethod::WeakBoundary { relative_threshold } => {
                Ok(weak_boundary_regions(graph, *relative_threshold))
            }
            CondenseMethod::MinCutCommunity { min_region_size } => {
                let arc = Arc::new(graph.clone());
                let mut detector = CommunityDetector::new(arc);
                Ok(detector.detect(*min_region_size).to_vec())
            }
            CondenseMethod::Partition { num_regions } => {
                if *num_regions == 0 {
                    return Err(CondenseError::InvalidConfig(
                        "num_regions must be > 0".to_string(),
                    ));
                }
                let arc = Arc::new(graph.clone());
                let partitioner = GraphPartitioner::new(arc, *num_regions);
                Ok(partitioner.partition())
            }
            CondenseMethod::DiffMinCut(cfg) => {
                let result = DiffCutCondenser::new(cfg.clone()).train(graph)?;
                Ok(result.hard_regions())
            }
        }
    }
}

/// Convenience wrapper: condense with default ([`CondenseMethod::WeakBoundary`])
/// settings.
pub fn condense(graph: &DynamicGraph, features: &NodeFeatures) -> Result<CondensedGraph> {
    GraphCondenser::default().condense(graph, features)
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_triangles() -> (DynamicGraph, NodeFeatures) {
        let g = DynamicGraph::new();
        for &(u, v, w) in &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.05),
        ] {
            g.insert_edge(u, v, w).unwrap();
        }
        let mut f = NodeFeatures::new(2, 2);
        // Cluster A near (0,0) labelled 0, cluster B near (10,10) labelled 1.
        for v in 0..3u64 {
            f.set(v, vec![v as f32 * 0.01, 0.0], 0).unwrap();
        }
        for v in 3..6u64 {
            f.set(v, vec![10.0 + v as f32 * 0.01, 10.0], 1).unwrap();
        }
        (g, f)
    }

    #[test]
    fn empty_graph_errors() {
        let g = DynamicGraph::new();
        let f = NodeFeatures::new(2, 0);
        assert!(matches!(
            condense(&g, &f).unwrap_err(),
            CondenseError::EmptyGraph
        ));
    }

    #[test]
    fn missing_feature_errors() {
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        let mut f = NodeFeatures::new(2, 0);
        f.set_embedding(0, vec![0.0, 0.0]).unwrap();
        // vertex 1 has no feature
        assert!(matches!(
            condense(&g, &f).unwrap_err(),
            CondenseError::MissingFeature(1)
        ));
    }

    #[test]
    fn condenses_two_communities() {
        let (g, f) = two_triangles();
        let c = condense(&g, &f).unwrap();
        // Should collapse 6 nodes into 2 communities.
        assert_eq!(c.source_nodes, 6);
        assert_eq!(c.node_count(), 2);
        assert_eq!(c.total_weight(), 6);
        // Exactly one super-edge across the bridge.
        assert_eq!(c.edge_count(), 1);
        let e = c.edges[0];
        assert_eq!((e.source, e.target), (0, 1));
        assert_eq!(e.crossings, 1);
        assert!((e.weight - 0.05).abs() < 1e-9);
        // Region ids are deterministic & sorted; first region holds {0,1,2}.
        assert_eq!(c.nodes[0].members, vec![0, 1, 2]);
        assert_eq!(c.nodes[1].members, vec![3, 4, 5]);
        // Pure, well-formed class distributions.
        assert_eq!(c.nodes[0].dominant_class(), Some(0));
        assert_eq!(c.nodes[1].dominant_class(), Some(1));
        assert!(c.nodes[0].purity() > 0.99);
        // High internal cohesion (3 internal edges vs 0.05 boundary).
        assert!(c.nodes[0].coherence > 0.9);
    }

    #[test]
    fn mincut_community_recovers_clear_bottleneck() {
        // Dense triangles (weight 5) joined by a single light bridge (weight 1).
        // This is the regime where recursive min-cut community detection works:
        // a sharp bottleneck and no low-degree vertices to peel.
        let g = DynamicGraph::new();
        for &(u, v, w) in &[
            (0, 1, 5.0),
            (1, 2, 5.0),
            (2, 0, 5.0),
            (3, 4, 5.0),
            (4, 5, 5.0),
            (5, 3, 5.0),
            (2, 3, 1.0),
        ] {
            g.insert_edge(u, v, w).unwrap();
        }
        let mut f = NodeFeatures::new(1, 0);
        for v in 0..6u64 {
            f.set_embedding(v, vec![v as f32]).unwrap();
        }
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::MinCutCommunity { min_region_size: 2 },
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        // The min-cut engine reduces and fully covers the graph. It may split
        // more finely than the planted 2 communities (recursive global min cut
        // is aggressive); we assert reduction + coverage, not exact recovery —
        // exact community recovery is the default WeakBoundary method's job.
        assert_eq!(c.total_weight(), 6); // full coverage
        assert!(c.node_count() >= 2 && c.node_count() < 6); // runs + reduces
    }

    #[test]
    fn diff_mincut_condenses_via_trained_assignment() {
        use crate::diffcut::DiffCutConfig;
        let (g, f) = two_triangles();
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::DiffMinCut(DiffCutConfig {
                num_clusters: 2,
                ..Default::default()
            }),
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        assert_eq!(c.node_count(), 2);
        assert_eq!(c.total_weight(), 6);
        assert_eq!(c.nodes[0].members, vec![0, 1, 2]);
        assert_eq!(c.nodes[1].members, vec![3, 4, 5]);
        assert_eq!(c.edge_count(), 1); // the bridge -> one super-edge
    }

    #[test]
    fn weak_boundary_falls_back_to_components_without_contrast() {
        // Uniform weights -> no edge is below 0.5*mean -> nothing cut ->
        // regions equal connected components (here, one).
        let g = DynamicGraph::new();
        for &(u, v) in &[(0, 1), (1, 2), (2, 0)] {
            g.insert_edge(u, v, 1.0).unwrap();
        }
        let mut f = NodeFeatures::new(1, 0);
        for v in 0..3u64 {
            f.set_embedding(v, vec![v as f32]).unwrap();
        }
        let c = condense(&g, &f).unwrap(); // default WeakBoundary
        assert_eq!(c.node_count(), 1);
        assert_eq!(c.total_weight(), 3);
    }

    #[test]
    fn partition_runs_and_covers() {
        // GraphPartitioner is best-effort; we assert it runs and covers every
        // vertex exactly once (reduction is graph-dependent, not guaranteed).
        let g = DynamicGraph::new();
        for i in 0..15u64 {
            g.insert_edge(i, i + 1, 1.0).unwrap();
        }
        let mut f = NodeFeatures::new(1, 0);
        for v in 0..16u64 {
            f.set_embedding(v, vec![v as f32]).unwrap();
        }
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::Partition { num_regions: 4 },
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        assert_eq!(c.total_weight(), 16); // full, non-overlapping coverage
        assert!(c.node_count() <= 16 && c.node_count() >= 1);
    }

    #[test]
    fn partition_zero_regions_errors() {
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        let mut f = NodeFeatures::new(1, 0);
        f.set_embedding(0, vec![0.0]).unwrap();
        f.set_embedding(1, vec![1.0]).unwrap();
        let err = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::Partition { num_regions: 0 },
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap_err();
        assert!(matches!(err, CondenseError::InvalidConfig(_)));
    }

    #[test]
    fn coverage_includes_isolated_vertex() {
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        g.add_vertex(99); // isolated
        let mut f = NodeFeatures::new(1, 0);
        for v in [0u64, 1, 99] {
            f.set_embedding(v, vec![v as f32]).unwrap();
        }
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::ConnectedComponents,
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        // {0,1} component + {99} singleton = 2 regions covering all vertices.
        assert_eq!(c.total_weight(), 3);
        assert!(c.nodes.iter().any(|n| n.members == vec![99]));
    }

    #[test]
    fn centroid_is_member_mean_and_medoid_valid() {
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        g.insert_edge(1, 2, 1.0).unwrap();
        g.insert_edge(2, 0, 1.0).unwrap();
        let mut f = NodeFeatures::new(1, 0);
        f.set_embedding(0, vec![0.0]).unwrap();
        f.set_embedding(1, vec![2.0]).unwrap();
        f.set_embedding(2, vec![4.0]).unwrap();
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::ConnectedComponents,
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        assert_eq!(c.node_count(), 1);
        assert!((c.nodes[0].centroid[0] - 2.0).abs() < 1e-6);
        // Medoid is the member nearest the mean (2.0) -> vertex 1.
        assert_eq!(c.nodes[0].representative, 1);
    }

    #[test]
    fn normalize_centroids_unit_length() {
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        let mut f = NodeFeatures::new(2, 0);
        f.set_embedding(0, vec![3.0, 0.0]).unwrap();
        f.set_embedding(1, vec![3.0, 0.0]).unwrap();
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::ConnectedComponents,
            normalize_centroids: true,
        })
        .condense(&g, &f)
        .unwrap();
        let norm: f32 = c.nodes[0]
            .centroid
            .iter()
            .map(|x| x * x)
            .sum::<f32>()
            .sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
    }
}
