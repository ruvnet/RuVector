//! Condensed graph data model: super-nodes (regions) and super-edges.

use ruvector_mincut::{DynamicGraph, VertexId};
use serde::{Deserialize, Serialize};

/// A single super-node in a condensed graph: one structural region of the
/// original graph collapsed to a representative summary.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CondensedNode {
    /// Stable region id (assigned deterministically by the condenser).
    pub id: u64,
    /// Mean embedding of the region's members.
    pub centroid: Vec<f32>,
    /// Number of original vertices collapsed into this region.
    pub weight: u32,
    /// Normalised class histogram over `num_classes` (empty when unsupervised).
    pub class_distribution: Vec<f32>,
    /// Internal cohesion in `[0, 1]`: fraction of incident edge weight that
    /// stays inside the region (1.0 = fully self-contained).
    pub coherence: f32,
    /// Member closest to the centroid (the region's medoid).
    pub representative: VertexId,
    /// The original vertices that belong to this region (sorted ascending).
    pub members: Vec<VertexId>,
}

impl CondensedNode {
    /// The dominant class of this region, if a class distribution is present.
    pub fn dominant_class(&self) -> Option<usize> {
        if self.class_distribution.is_empty() {
            return None;
        }
        let mut best = 0usize;
        let mut best_p = self.class_distribution[0];
        for (i, &p) in self.class_distribution.iter().enumerate().skip(1) {
            if p > best_p {
                best_p = p;
                best = i;
            }
        }
        Some(best)
    }

    /// Purity of the dominant class (its share of the region), or `1.0` when
    /// unsupervised (empty distribution).
    pub fn purity(&self) -> f32 {
        if self.class_distribution.is_empty() {
            return 1.0;
        }
        self.class_distribution
            .iter()
            .copied()
            .fold(0.0_f32, f32::max)
    }
}

/// A weighted super-edge between two regions, aggregating every original edge
/// that crosses the corresponding region boundary.
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub struct CondensedEdge {
    /// Source region id (always `< target` for canonical undirected storage).
    pub source: u64,
    /// Target region id.
    pub target: u64,
    /// Sum of crossing original edge weights.
    pub weight: f64,
    /// Number of original edges merged into this super-edge.
    pub crossings: u32,
}

/// The result of condensing a graph: a small set of super-nodes and the
/// weighted super-edges connecting them, plus provenance for computing
/// reduction ratios.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CondensedGraph {
    /// Super-nodes, ordered by ascending `id`.
    pub nodes: Vec<CondensedNode>,
    /// Super-edges (canonical, deduplicated).
    pub edges: Vec<CondensedEdge>,
    /// Original vertex count (provenance).
    pub source_nodes: usize,
    /// Original edge count (provenance).
    pub source_edges: usize,
    /// Embedding dimension.
    pub dim: usize,
    /// Class count (`0` if unsupervised).
    pub num_classes: usize,
}

impl CondensedGraph {
    /// Number of super-nodes.
    pub fn node_count(&self) -> usize {
        self.nodes.len()
    }

    /// Number of super-edges.
    pub fn edge_count(&self) -> usize {
        self.edges.len()
    }

    /// Node reduction factor (`source_nodes / condensed_nodes`).
    pub fn node_reduction_ratio(&self) -> f64 {
        if self.nodes.is_empty() {
            return 0.0;
        }
        self.source_nodes as f64 / self.nodes.len() as f64
    }

    /// Edge reduction factor (`source_edges / condensed_edges`).
    pub fn edge_reduction_ratio(&self) -> f64 {
        if self.edges.is_empty() {
            return if self.source_edges == 0 {
                1.0
            } else {
                self.source_edges as f64
            };
        }
        self.source_edges as f64 / self.edges.len() as f64
    }

    /// Look up a super-node by region id (binary search; nodes are id-sorted).
    pub fn get_node(&self, id: u64) -> Option<&CondensedNode> {
        self.nodes
            .binary_search_by_key(&id, |n| n.id)
            .ok()
            .map(|i| &self.nodes[i])
    }

    /// Total weight (member count) across all super-nodes — equals
    /// `source_nodes` for a complete partition.
    pub fn total_weight(&self) -> u64 {
        self.nodes.iter().map(|n| n.weight as u64).sum()
    }

    /// Rebuild the condensed graph as a [`DynamicGraph`] (region id → vertex
    /// id). Enables hierarchical / iterated condensation and feeding the
    /// condensed structure back into the min-cut engine.
    pub fn to_dynamic_graph(&self) -> DynamicGraph {
        let g = DynamicGraph::with_capacity(self.nodes.len(), self.edges.len());
        for n in &self.nodes {
            g.add_vertex(n.id);
        }
        for e in &self.edges {
            // Super-edges are canonical and unique, so insert cannot collide.
            let _ = g.insert_edge(e.source, e.target, e.weight);
        }
        g
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn node(id: u64, dist: Vec<f32>) -> CondensedNode {
        CondensedNode {
            id,
            centroid: vec![0.0],
            weight: 1,
            class_distribution: dist,
            coherence: 1.0,
            representative: id,
            members: vec![id],
        }
    }

    #[test]
    fn dominant_class_picks_argmax() {
        let n = node(0, vec![0.1, 0.7, 0.2]);
        assert_eq!(n.dominant_class(), Some(1));
        let unsup = node(1, vec![]);
        assert_eq!(unsup.dominant_class(), None);
    }

    #[test]
    fn reduction_ratios() {
        let g = CondensedGraph {
            nodes: vec![node(0, vec![]), node(1, vec![])],
            edges: vec![CondensedEdge {
                source: 0,
                target: 1,
                weight: 1.0,
                crossings: 3,
            }],
            source_nodes: 100,
            source_edges: 400,
            dim: 1,
            num_classes: 0,
        };
        assert_eq!(g.node_reduction_ratio(), 50.0);
        assert_eq!(g.edge_reduction_ratio(), 400.0);
        assert_eq!(g.total_weight(), 2);
    }

    #[test]
    fn get_node_binary_search() {
        let g = CondensedGraph {
            nodes: vec![node(0, vec![]), node(5, vec![]), node(9, vec![])],
            edges: vec![],
            source_nodes: 3,
            source_edges: 0,
            dim: 1,
            num_classes: 0,
        };
        assert_eq!(g.get_node(5).map(|n| n.id), Some(5));
        assert!(g.get_node(7).is_none());
    }

    #[test]
    fn round_trips_to_dynamic_graph() {
        let g = CondensedGraph {
            nodes: vec![node(0, vec![]), node(1, vec![]), node(2, vec![])],
            edges: vec![
                CondensedEdge {
                    source: 0,
                    target: 1,
                    weight: 2.0,
                    crossings: 1,
                },
                CondensedEdge {
                    source: 1,
                    target: 2,
                    weight: 3.0,
                    crossings: 1,
                },
            ],
            source_nodes: 3,
            source_edges: 2,
            dim: 1,
            num_classes: 0,
        };
        let dg = g.to_dynamic_graph();
        assert_eq!(dg.num_vertices(), 3);
        assert_eq!(dg.num_edges(), 2);
        assert_eq!(dg.edge_weight(0, 1), Some(2.0));
    }
}
