//! # ruvector-graph-condense
//!
//! Structure-preserving **graph condensation** built on RuVector's dynamic
//! min-cut engine ([`ruvector_mincut`]).
//!
//! ## What this is (and isn't)
//!
//! The graph-condensation literature (GCond, SFGC, GEOM, SGDD, …) defines
//! *condensation* as **synthesising a small fake graph** by optimising a
//! learning objective (gradient/distribution/trajectory matching) so that a GNN
//! trained on the synthetic graph matches one trained on the original. That is
//! powerful but expensive (bi-level optimisation), supervised, and — by design
//! — **destroys the mapping back to real nodes**.
//!
//! This crate takes the complementary, **training-free** route that the 2024–
//! 2026 surveys flag as under-explored:
//!
//! - **Min-cut community structure as the condensation prior.** Regions come
//!   from recursive dynamic min-cut ([`ruvector_mincut::CommunityDetector`]),
//!   not k-means. No published method (as of 2026) uses graph-cut community
//!   detection as the core condensation mechanism — the closest analogs are
//!   CGC (generic clustering, 2025) and GCTD (tensor decomposition, 2025).
//! - **A differentiable min-cut *loss*** ([`diffcut`], [`CondenseMethod::DiffMinCut`]).
//!   A relaxed normalized-cut + orthogonality objective (MinCutPool-style) whose
//!   region structure is *trained* by gradient descent to preserve the cut.
//!   The surveys flag an explicit differentiable min-cut term in the
//!   condensation loss as unpublished; only spectral terms (SGDD's LED, GDEM's
//!   eigenbasis) exist. Gradients are analytic (no autodiff dependency) and
//!   gradient-checked.
//! - **Cuts preserved by construction.** Every original edge that crosses a
//!   region boundary survives as a weighted super-edge, so the condensed graph
//!   reproduces the source's cut structure instead of having to learn it. The
//!   [`metrics::cut_inflation`] proxy quantifies exactly this.
//! - **Provenance retained.** Each [`CondensedNode`] keeps its `members`, so
//!   the original↔condensed mapping is intact (useful for audit / explainability
//!   — the thing learned condensation throws away).
//!
//! In the field's taxonomy this is closer to **structure-preserving coarsening
//! with synthetic representatives** than to GCond-style condensation: it trades
//! peak downstream accuracy for being fast, label-optional, deterministic,
//! streaming-friendly, and interpretable.
//!
//! ## Pipeline
//!
//! ```text
//! DynamicGraph + NodeFeatures
//!        │  recursive dynamic min-cut
//!        ▼
//!    Regions (communities)
//!        │  per region: centroid · weight · class histogram · coherence · medoid
//!        ▼
//!    CondensedGraph  (super-nodes + boundary-weighted super-edges)
//! ```
//!
//! ## Quick start
//!
//! ```
//! use ruvector_graph_condense::{condense, NodeFeatures};
//! use ruvector_mincut::DynamicGraph;
//!
//! // Two triangles joined by a weak bridge.
//! let g = DynamicGraph::new();
//! for &(u, v, w) in &[(0,1,1.0),(1,2,1.0),(2,0,1.0),
//!                     (3,4,1.0),(4,5,1.0),(5,3,1.0),
//!                     (2,3,0.05)] {
//!     g.insert_edge(u, v, w).unwrap();
//! }
//! let mut f = NodeFeatures::new(1, 0);
//! for v in 0..6u64 { f.set_embedding(v, vec![v as f32]).unwrap(); }
//!
//! let condensed = condense(&g, &f).unwrap();
//! assert_eq!(condensed.node_count(), 2);          // recovered both communities
//! assert_eq!(condensed.edge_count(), 1);          // the bridge -> one super-edge
//! assert!(condensed.node_reduction_ratio() == 3.0);
//! ```

#![forbid(unsafe_code)]

pub mod condense;
mod cutloss;
pub mod diffcut;
pub mod error;
pub mod features;
pub mod gnn_eval;
pub mod metrics;
pub mod node;
mod regions;
pub mod stream;
pub mod synthetic;

pub use condense::{condense, CondenseConfig, CondenseMethod, GraphCondenser};
pub use diffcut::{
    min_cut_loss, DiffCutCondenser, DiffCutConfig, DiffCutResult, InitStrategy, MinCutLoss,
    Optimizer,
};
pub use error::{CondenseError, Result};
pub use features::NodeFeatures;
pub use metrics::{cut_inflation, evaluate, evaluate_full, CondensationMetrics};
pub use node::{CondensedEdge, CondensedGraph, CondensedNode};
pub use stream::StreamingCondenser;
pub use synthetic::PlantedPartition;

/// Crate version string.
pub const VERSION: &str = env!("CARGO_PKG_VERSION");

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_mincut::DynamicGraph;

    #[test]
    fn end_to_end_condense_and_evaluate() {
        let pp = PlantedPartition {
            num_communities: 5,
            community_size: 16,
            dim: 8,
            p_intra: 0.5,
            p_inter: 0.001,
            seed: 42,
            ..Default::default()
        };
        let (g, f) = pp.generate();
        let condensed = condense(&g, &f).unwrap();
        let m = evaluate(&g, &condensed);

        assert_eq!(m.source_nodes, 80);
        assert!(m.condensed_nodes >= 5);
        assert!(m.node_reduction_ratio > 1.0);
        assert!(m.intra_weight_ratio > 0.8);
        assert!(m.label_purity > 0.8);
    }

    #[test]
    fn public_api_is_reachable() {
        let _ = VERSION;
        let g = DynamicGraph::new();
        g.insert_edge(0, 1, 1.0).unwrap();
        let mut f = NodeFeatures::new(1, 0);
        f.set_embedding(0, vec![0.0]).unwrap();
        f.set_embedding(1, vec![1.0]).unwrap();
        let condenser = GraphCondenser::new(CondenseConfig::default());
        let c = condenser.condense(&g, &f).unwrap();
        assert_eq!(c.total_weight(), 2);
    }
}
