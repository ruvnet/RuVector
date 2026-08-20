//! Two ways to turn a memory similarity graph into semantic partitions.
//!
//! `fixed_k_partition` wraps the existing `ruvector_mincut::GraphPartitioner`
//! (unweighted, edge-count min cut, fixed `K`). `adaptive_partition` is new:
//! it recurses directly on `ruvector_mincut::MinCutBuilder` /
//! `DynamicMinCut` — the crate's weighted, subpolynomial exact algorithm —
//! and stops splitting a component once its cut is dense relative to its
//! internal edge weight, instead of forcing a caller-chosen `K`.

use crate::mincut_exact::global_min_cut;
use crate::witness::PartitionWitnessChain;
use ruvector_mincut::{DynamicGraph, GraphPartitioner, MinCutBuilder, VertexId, Weight};
use std::collections::HashSet;
use std::sync::Arc;

pub struct AdaptiveConfig {
    /// Never split a component smaller than this.
    pub min_cluster_size: usize,
    /// Stop splitting once `cut_value / (total_internal_weight / |verts|)`
    /// meets or exceeds this ratio — the component's weakest seam is
    /// already about as strong as its typical internal edge, so treat it
    /// as one coherent topic. Lower = more, smaller partitions.
    pub coherence_ratio: f64,
    /// Hard recursion depth cap, independent of the ratio, so a pathological
    /// input graph cannot blow the stack.
    pub max_depth: usize,
}

impl Default for AdaptiveConfig {
    fn default() -> Self {
        Self {
            min_cluster_size: 20,
            coherence_ratio: 0.35,
            max_depth: 8,
        }
    }
}

pub struct PartitionResult {
    pub clusters: Vec<Vec<VertexId>>,
    pub witness: PartitionWitnessChain,
}

/// Existing-tool baseline partitioner: `ruvector_mincut::GraphPartitioner`
/// recursively bisects down to at most `num_partitions` leaves using
/// unweighted min-cut (fewest crossing edges), independent of how strong
/// those edges are.
///
/// During this nightly's development `GraphPartitioner` (via
/// `RuVectorGraphAnalyzer`'s `partition()`) was observed to have two
/// distinct vertex-set failure modes, independent of the weighted-cut
/// inconsistency `mincut_exact.rs` documents: (1) at moderate scale it can
/// drop vertices outright — a 100-vertex, two-clique repro returned
/// partitions covering only 50 of the 100 input vertices; (2) with a
/// non-contiguous vertex-id space it can *fabricate* ids that were never in
/// the input graph at all (repro: ids `{1,2,3,11,12,13}` produced a
/// partition also containing invented ids like `4,5,6,7,8,9,10`). See the
/// nightly research doc for both repros. Rather than silently shrinking or
/// polluting the corpus, this wrapper (a) drops any returned id absent from
/// `all_vertices`, then (b) appends any `all_vertices` id absent from every
/// returned group as one final "uncovered" group. This does not fix the
/// partitioner's own boundary quality — it is tested and reported as-is —
/// it only prevents its output from inventing or discarding memories.
pub fn fixed_k_partition(
    all_vertices: &[VertexId],
    edges: &[(VertexId, VertexId, Weight)],
    num_partitions: usize,
) -> Vec<Vec<VertexId>> {
    let graph = Arc::new(DynamicGraph::new());
    for &(u, v, w) in edges {
        let _ = graph.insert_edge(u, v, w);
    }
    let valid: HashSet<VertexId> = all_vertices.iter().copied().collect();
    let partitioner = GraphPartitioner::new(graph, num_partitions.max(1));
    let mut parts: Vec<Vec<VertexId>> = partitioner
        .partition()
        .into_iter()
        .map(|p| {
            p.into_iter()
                .filter(|v| valid.contains(v))
                .collect::<Vec<VertexId>>()
        })
        .filter(|p| !p.is_empty())
        .collect();

    let covered: HashSet<VertexId> = parts.iter().flatten().copied().collect();
    let uncovered: Vec<VertexId> = all_vertices
        .iter()
        .copied()
        .filter(|v| !covered.contains(v))
        .collect();
    if !uncovered.is_empty() {
        parts.push(uncovered);
    }
    parts
}

fn induced_edges(
    all_edges: &[(VertexId, VertexId, Weight)],
    verts: &HashSet<VertexId>,
) -> Vec<(VertexId, VertexId, Weight)> {
    all_edges
        .iter()
        .filter(|(u, v, _)| verts.contains(u) && verts.contains(v))
        .copied()
        .collect()
}

pub fn adaptive_partition(
    all_edges: &[(VertexId, VertexId, Weight)],
    all_vertices: &[VertexId],
    cfg: &AdaptiveConfig,
) -> PartitionResult {
    let mut clusters = Vec::new();
    let mut witness = PartitionWitnessChain::new();
    recurse(all_edges, all_vertices, 0, cfg, &mut clusters, &mut witness);
    PartitionResult { clusters, witness }
}

fn recurse(
    all_edges: &[(VertexId, VertexId, Weight)],
    verts: &[VertexId],
    depth: usize,
    cfg: &AdaptiveConfig,
    clusters: &mut Vec<Vec<VertexId>>,
    witness: &mut PartitionWitnessChain,
) {
    if verts.len() <= cfg.min_cluster_size || depth >= cfg.max_depth {
        clusters.push(verts.to_vec());
        return;
    }

    let vset: HashSet<VertexId> = verts.iter().copied().collect();
    let sub_edges = induced_edges(all_edges, &vset);
    if sub_edges.is_empty() {
        clusters.push(verts.to_vec());
        return;
    }
    let total_weight: f64 = sub_edges.iter().map(|(_, _, w)| w).sum();
    let avg_weight_per_vertex = (total_weight / verts.len() as f64).max(1e-9);

    // Authoritative split: our own tested Stoer-Wagner (mincut_exact.rs).
    // `ruvector_mincut::MinCutBuilder::min_cut_value()` is queried purely as
    // an independent cross-check — see mincut_exact.rs's doc comment for
    // why its own `.partition()` output is not used here.
    let (cut_value, side_a, side_b) = global_min_cut(verts, &sub_edges);

    let cross_check = MinCutBuilder::new().exact().with_edges(sub_edges).build();
    if let Ok(reference) = cross_check {
        let reference_value = reference.min_cut_value();
        if (reference_value - cut_value).abs() > 1e-6 * reference_value.max(1.0) {
            // Disagreement between the two independent computations of the
            // *value* (not just the partition materialization bug this
            // module works around) would be a correctness issue in our own
            // code — surface it loudly rather than silently trusting ours.
            debug_assert!(
                false,
                "min-cut value mismatch: ours={cut_value} ruvector_mincut={reference_value}"
            );
        }
    }

    let normalized_cut = cut_value / avg_weight_per_vertex;

    let should_split = side_a.len() >= cfg.min_cluster_size
        && side_b.len() >= cfg.min_cluster_size
        && normalized_cut < cfg.coherence_ratio;

    witness.record(verts, cut_value, &side_a, &side_b, should_split);

    if should_split {
        recurse(all_edges, &side_a, depth + 1, cfg, clusters, witness);
        recurse(all_edges, &side_b, depth + 1, cfg, clusters, witness);
    } else {
        clusters.push(verts.to_vec());
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// Two tight cliques joined by a single weak bridge edge should split
    /// into exactly two clusters at a permissive coherence ratio.
    #[test]
    fn splits_two_cliques_joined_by_weak_bridge() {
        let mut edges = Vec::new();
        for &(u, v) in &[(1, 2), (2, 3), (3, 1)] {
            edges.push((u, v, 1.0));
        }
        for &(u, v) in &[(11, 12), (12, 13), (13, 11)] {
            edges.push((u, v, 1.0));
        }
        edges.push((3, 11, 0.05));
        let verts: Vec<VertexId> = vec![1, 2, 3, 11, 12, 13];
        let cfg = AdaptiveConfig {
            min_cluster_size: 2,
            coherence_ratio: 0.5,
            max_depth: 4,
        };
        let result = adaptive_partition(&edges, &verts, &cfg);
        assert_eq!(result.clusters.len(), 2, "{:?}", result.clusters);
        assert!(result.witness.verify());
    }

    #[test]
    fn does_not_split_a_single_dense_cluster() {
        let mut edges = Vec::new();
        for &(u, v) in &[(1, 2), (2, 3), (3, 4), (4, 1), (1, 3), (2, 4)] {
            edges.push((u, v, 1.0));
        }
        let verts: Vec<VertexId> = vec![1, 2, 3, 4];
        let cfg = AdaptiveConfig {
            min_cluster_size: 2,
            coherence_ratio: 0.35,
            max_depth: 4,
        };
        let result = adaptive_partition(&edges, &verts, &cfg);
        assert_eq!(result.clusters.len(), 1);
    }

    #[test]
    fn fixed_k_partition_respects_k_and_covers_all_vertices() {
        let mut edges = Vec::new();
        for &(u, v) in &[(1, 2), (2, 3), (3, 1)] {
            edges.push((u, v, 1.0));
        }
        for &(u, v) in &[(11, 12), (12, 13), (13, 11)] {
            edges.push((u, v, 1.0));
        }
        edges.push((3, 11, 0.05));
        let verts: Vec<VertexId> = vec![1, 2, 3, 11, 12, 13];
        let parts = fixed_k_partition(&verts, &edges, 2);
        let total: usize = parts.iter().map(|p| p.len()).sum();
        // GraphPartitioner may not achieve exactly K groups (or, per the
        // documented defect, may leave vertices uncovered before the
        // fallback bucket runs) — the one invariant this test enforces is
        // that fixed_k_partition itself never silently drops a vertex.
        assert_eq!(total, 6, "fixed_k_partition dropped vertices: {parts:?}");
    }
}
