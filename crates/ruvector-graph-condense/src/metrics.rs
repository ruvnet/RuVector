//! Quality metrics for a condensation result.
//!
//! Accuracy-retention (retrain-a-GNN) evaluation is out of scope for this crate
//! — the 2024–2026 literature explicitly calls for cheap *proxy* metrics that
//! avoid retraining many GNNs. These are structural proxies computable directly
//! from the source and condensed graphs.

use crate::node::CondensedGraph;
use ruvector_mincut::{DynamicGraph, MinCutBuilder};

/// A bundle of cheap, retrain-free quality proxies.
#[derive(Debug, Clone, PartialEq)]
pub struct CondensationMetrics {
    /// Original vertex count.
    pub source_nodes: usize,
    /// Condensed super-node count.
    pub condensed_nodes: usize,
    /// `source_nodes / condensed_nodes`.
    pub node_reduction_ratio: f64,
    /// Original edge count.
    pub source_edges: usize,
    /// Condensed super-edge count.
    pub condensed_edges: usize,
    /// `source_edges / condensed_edges`.
    pub edge_reduction_ratio: f64,
    /// Fraction of total edge weight that stayed *inside* a region. Higher is
    /// better: it means the partition cut few/light edges (good community
    /// structure was found).
    pub intra_weight_ratio: f64,
    /// Mean per-region coherence in `[0, 1]`.
    pub mean_coherence: f32,
    /// Weight-averaged region purity (dominant-class share); `1.0` when
    /// unsupervised.
    pub label_purity: f32,
    /// Global-min-cut inflation: `mincut(condensed) / mincut(source)`.
    /// `Some(1.0)` means the source's global min cut survives coarsening
    /// exactly; `> 1.0` means the true cut got hidden inside a region. `None`
    /// when undefined (disconnected source, or condensed graph too small).
    pub cut_inflation: Option<f64>,
}

/// Compute the cheap structural proxies (no min-cut solve).
pub fn evaluate(graph: &DynamicGraph, condensed: &CondensedGraph) -> CondensationMetrics {
    let total_weight: f64 = graph.edges().iter().map(|e| e.weight).sum();
    let inter_weight: f64 = condensed.edges.iter().map(|e| e.weight).sum();
    let intra_weight_ratio = if total_weight > 0.0 {
        ((total_weight - inter_weight) / total_weight).clamp(0.0, 1.0)
    } else {
        1.0
    };

    let (mean_coherence, label_purity) = aggregate_node_quality(condensed);

    CondensationMetrics {
        source_nodes: condensed.source_nodes,
        condensed_nodes: condensed.node_count(),
        node_reduction_ratio: condensed.node_reduction_ratio(),
        source_edges: condensed.source_edges,
        condensed_edges: condensed.edge_count(),
        edge_reduction_ratio: condensed.edge_reduction_ratio(),
        intra_weight_ratio,
        mean_coherence,
        label_purity,
        cut_inflation: None,
    }
}

/// Like [`evaluate`], but also solves the global min cut on both graphs to fill
/// in [`CondensationMetrics::cut_inflation`]. This is **O(min-cut)** on the full
/// source graph and is therefore opt-in.
pub fn evaluate_full(graph: &DynamicGraph, condensed: &CondensedGraph) -> CondensationMetrics {
    let mut m = evaluate(graph, condensed);
    m.cut_inflation = cut_inflation(graph, condensed);
    m
}

/// Ratio of the condensed graph's global min cut to the source's. See
/// [`CondensationMetrics::cut_inflation`] for interpretation.
pub fn cut_inflation(graph: &DynamicGraph, condensed: &CondensedGraph) -> Option<f64> {
    // Need a meaningful cut on both sides.
    if graph.num_vertices() < 2 || condensed.node_count() < 2 {
        return None;
    }

    let source_cut = global_min_cut(graph.edges().iter().map(|e| (e.source, e.target, e.weight)))?;
    if source_cut <= 0.0 {
        // Disconnected source: ratio undefined.
        return None;
    }
    let condensed_cut = global_min_cut(
        condensed
            .edges
            .iter()
            .map(|e| (e.source, e.target, e.weight)),
    )?;

    Some(condensed_cut / source_cut)
}

/// Solve an exact global min cut over an edge iterator, returning `None` if the
/// result is non-finite (e.g. fewer than 2 connected vertices).
fn global_min_cut<I>(edges: I) -> Option<f64>
where
    I: IntoIterator<Item = (u64, u64, f64)>,
{
    let edge_vec: Vec<(u64, u64, f64)> = edges.into_iter().collect();
    if edge_vec.is_empty() {
        return None;
    }
    let mincut = MinCutBuilder::new()
        .exact()
        .with_edges(edge_vec)
        .build()
        .ok()?;
    let v = mincut.min_cut_value();
    if v.is_finite() {
        Some(v)
    } else {
        None
    }
}

fn aggregate_node_quality(condensed: &CondensedGraph) -> (f32, f32) {
    if condensed.nodes.is_empty() {
        return (0.0, 1.0);
    }
    let mut coherence_sum = 0.0f32;
    let mut purity_weighted = 0.0f32;
    let mut weight_total = 0.0f32;
    for n in &condensed.nodes {
        coherence_sum += n.coherence;
        let w = n.weight as f32;
        purity_weighted += n.purity() * w;
        weight_total += w;
    }
    let mean_coherence = coherence_sum / condensed.nodes.len() as f32;
    let label_purity = if weight_total > 0.0 {
        purity_weighted / weight_total
    } else {
        1.0
    };
    (mean_coherence, label_purity)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::condense::{condense, CondenseConfig, CondenseMethod, GraphCondenser};
    use crate::features::NodeFeatures;

    fn barbell() -> (DynamicGraph, NodeFeatures) {
        // Two K3 cliques joined by a single weak bridge.
        let g = DynamicGraph::new();
        for &(u, v, w) in &[
            (0, 1, 1.0),
            (1, 2, 1.0),
            (2, 0, 1.0),
            (3, 4, 1.0),
            (4, 5, 1.0),
            (5, 3, 1.0),
            (2, 3, 0.1),
        ] {
            g.insert_edge(u, v, w).unwrap();
        }
        let mut f = NodeFeatures::new(1, 2);
        for v in 0..3u64 {
            f.set(v, vec![0.0], 0).unwrap();
        }
        for v in 3..6u64 {
            f.set(v, vec![1.0], 1).unwrap();
        }
        (g, f)
    }

    #[test]
    fn reports_reduction_and_quality() {
        let (g, f) = barbell();
        let c = condense(&g, &f).unwrap();
        let m = evaluate(&g, &c);
        assert_eq!(m.source_nodes, 6);
        assert_eq!(m.condensed_nodes, 2);
        assert_eq!(m.node_reduction_ratio, 3.0);
        // Only the 0.1 bridge crosses regions; 6 unit edges stay internal.
        assert!(m.intra_weight_ratio > 0.95);
        assert!(m.mean_coherence > 0.9);
        assert!(m.label_purity > 0.99);
        assert_eq!(m.cut_inflation, None); // evaluate() doesn't solve cuts
    }

    #[test]
    fn cut_inflation_preserved_for_clean_partition() {
        let (g, f) = barbell();
        let c = condense(&g, &f).unwrap();
        // Source global min cut = 0.1 (the bridge). Condensed graph is a single
        // super-edge of weight 0.1, so its min cut is also 0.1 -> ratio 1.0.
        let infl = cut_inflation(&g, &c).expect("defined for connected barbell");
        assert!((infl - 1.0).abs() < 1e-9, "got {infl}");
    }

    #[test]
    fn evaluate_full_fills_cut() {
        let (g, f) = barbell();
        let c = GraphCondenser::new(CondenseConfig {
            method: CondenseMethod::ConnectedComponents,
            normalize_centroids: false,
        })
        .condense(&g, &f)
        .unwrap();
        // Connected barbell -> single component -> single super-node -> cut None.
        let m = evaluate_full(&g, &c);
        assert_eq!(m.condensed_nodes, 1);
        assert_eq!(m.cut_inflation, None);
    }
}
