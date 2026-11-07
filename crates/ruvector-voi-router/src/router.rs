//! Per-query routing between baseline and coherence-gated search.

use ruvector_coherence_hnsw::graph::{l2_sq, FlatGraph};
use ruvector_coherence_hnsw::search::{
    BaselineSearch, CoherenceGatedSearch, SearchResult, Searcher,
};

/// Routes each query to [`BaselineSearch`] or [`CoherenceGatedSearch`] based
/// on the squared L2 distance from the search entry point to the query.
///
/// The routing signal costs nothing beyond what every beam search already
/// computes as its first step (`d0`, entry-to-query distance) — this is not
/// a second search or a shallow probe, just a threshold comparison on a
/// value both policies would compute anyway.
///
/// * `d0 <= distance_threshold` → entry is already close; route to
///   [`BaselineSearch`] (skip the per-pop coherence check, which has little
///   left to prune here and is pure overhead).
/// * `d0 >  distance_threshold` → entry is far; route to
///   [`CoherenceGatedSearch`] with `gate_threshold`, where pruning has real
///   room to reduce expansions.
pub struct VoiRoutedSearch {
    pub distance_threshold: f32,
    pub gate_threshold: f32,
}

/// A search result annotated with which policy actually handled the query.
pub struct RoutedResult {
    pub result: SearchResult,
    pub routed_to_gated: bool,
}

impl VoiRoutedSearch {
    pub fn new(distance_threshold: f32, gate_threshold: f32) -> Self {
        Self {
            distance_threshold,
            gate_threshold,
        }
    }

    /// Same routing as [`Searcher::search`], but also reports which policy
    /// handled the query — used by the benchmark to break results down by
    /// route without re-deriving `d0` from the outside.
    pub fn search_routed(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> RoutedResult {
        if graph.is_empty() {
            return RoutedResult {
                result: SearchResult {
                    neighbors: vec![],
                    pops: 0,
                    expansions: 0,
                },
                routed_to_gated: false,
            };
        }
        let entry = entry_id.min(graph.len() - 1);
        let d0 = l2_sq(query, graph.row(entry));

        if d0 <= self.distance_threshold {
            RoutedResult {
                result: BaselineSearch.search(graph, query, k, ef, entry_id),
                routed_to_gated: false,
            }
        } else {
            RoutedResult {
                result: CoherenceGatedSearch {
                    threshold: self.gate_threshold,
                }
                .search(graph, query, k, ef, entry_id),
                routed_to_gated: true,
            }
        }
    }
}

impl Searcher for VoiRoutedSearch {
    fn search(
        &self,
        graph: &FlatGraph,
        query: &[f32],
        k: usize,
        ef: usize,
        entry_id: usize,
    ) -> SearchResult {
        self.search_routed(graph, query, k, ef, entry_id).result
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvector_coherence_hnsw::graph::GraphConfig;

    fn tiny_graph() -> FlatGraph {
        // 4 points on a line in 1D, embedded in 2D as (x, 0).
        let vecs: Vec<f32> = vec![0.0, 0.0, 1.0, 0.0, 5.0, 0.0, 6.0, 0.0];
        FlatGraph::build(vecs, GraphConfig::local_only(2, 2))
    }

    #[test]
    fn routes_to_baseline_when_entry_is_close() {
        let g = tiny_graph();
        let router = VoiRoutedSearch::new(/* distance_threshold */ 10.0, 0.5);
        // entry = node 0 at (0,0); query near it, d0 = 0.25 <= 10.0.
        let r = router.search_routed(&g, &[0.5, 0.0], 2, 4, 0);
        assert!(!r.routed_to_gated);
        assert!(!r.result.neighbors.is_empty());
    }

    #[test]
    fn routes_to_gated_when_entry_is_far() {
        let g = tiny_graph();
        let router = VoiRoutedSearch::new(/* distance_threshold */ 1.0, 0.5);
        // entry = node 0 at (0,0); query far from it, d0 = 36.0 > 1.0.
        let r = router.search_routed(&g, &[6.0, 0.0], 2, 4, 0);
        assert!(r.routed_to_gated);
        assert!(!r.result.neighbors.is_empty());
    }

    #[test]
    fn threshold_boundary_is_inclusive_to_baseline() {
        let g = tiny_graph();
        let router = VoiRoutedSearch::new(1.0, 0.5);
        // d0 exactly 1.0 (query at (1,0), entry at (0,0)) must route baseline.
        let r = router.search_routed(&g, &[1.0, 0.0], 1, 4, 0);
        assert!(!r.routed_to_gated);
    }

    #[test]
    fn empty_graph_returns_empty_without_panicking() {
        let g = FlatGraph::build(vec![], GraphConfig::local_only(2, 2));
        let router = VoiRoutedSearch::new(1.0, 0.5);
        let r = router.search_routed(&g, &[0.0, 0.0], 2, 4, 0);
        assert!(r.result.neighbors.is_empty());
        assert!(!r.routed_to_gated);
    }

    #[test]
    fn search_trait_impl_matches_search_routed() {
        let g = tiny_graph();
        let router = VoiRoutedSearch::new(1.0, 0.5);
        let via_trait = Searcher::search(&router, &g, &[6.0, 0.0], 2, 4, 0);
        let via_routed = router.search_routed(&g, &[6.0, 0.0], 2, 4, 0);
        assert_eq!(via_trait.neighbors, via_routed.result.neighbors);
    }
}
