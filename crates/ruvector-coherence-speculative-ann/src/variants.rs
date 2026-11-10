//! Speculative ANN variants: graph draft (u8-ranked, optionally
//! coherence-gated) + exact f32 verify, wired behind the shared `AnnVariant`
//! trait so the benchmark can drive all three variants identically.

use std::sync::atomic::{AtomicU64, Ordering};

use crate::graph::DraftGraph;
use crate::search::DraftSearcher;
use crate::{sq_l2_f32, AnnVariant, Hit, ScalarQuantizer};

/// A fixed, distant entry point simulating a coarse upper-layer descent in
/// full HNSW: the draft stage must navigate toward the query rather than
/// starting already close to it. This is where a coherence gate has genuine
/// opportunity to prune off-path branches (see `ruvector-coherence-hnsw`).
const ENTRY_ID: usize = 0;

pub struct GraphDraftSpeculative<S: DraftSearcher> {
    graph: DraftGraph,
    quantizer: ScalarQuantizer,
    searcher: S,
    /// Draft pool size handed to verify (k' in the speculative literature).
    k_prime: usize,
    /// Beam width bound during traversal.
    ef: usize,
    total_expansions: AtomicU64,
    total_queries: AtomicU64,
}

impl<S: DraftSearcher> GraphDraftSpeculative<S> {
    pub fn new(
        graph: DraftGraph,
        quantizer: ScalarQuantizer,
        searcher: S,
        k_prime: usize,
        ef: usize,
    ) -> Self {
        GraphDraftSpeculative {
            graph,
            quantizer,
            searcher,
            k_prime,
            ef,
            total_expansions: AtomicU64::new(0),
            total_queries: AtomicU64::new(0),
        }
    }
}

impl<S: DraftSearcher> AnnVariant for GraphDraftSpeculative<S> {
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let query_u8 = self.quantizer.quantize(query);
        let draft = self.searcher.draft(
            &self.graph,
            query,
            &query_u8,
            self.k_prime.max(k),
            self.ef,
            ENTRY_ID,
        );

        self.total_expansions
            .fetch_add(draft.expansions as u64, Ordering::Relaxed);
        self.total_queries.fetch_add(1, Ordering::Relaxed);

        // Verify: exact f32 distances for the draft candidate pool only.
        let mut hits: Vec<Hit> = draft
            .candidate_ids
            .iter()
            .map(|&id| Hit {
                id: id as usize,
                dist: sq_l2_f32(query, self.graph.row_f32(id as usize)),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }

    fn name(&self) -> &str {
        self.searcher.name()
    }

    fn memory_bytes(&self) -> usize {
        self.graph.memory_bytes()
    }

    fn last_draft_cost(&self) -> Option<f64> {
        let q = self.total_queries.load(Ordering::Relaxed);
        if q == 0 {
            return None;
        }
        Some(self.total_expansions.load(Ordering::Relaxed) as f64 / q as f64)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::dataset::Dataset;
    use crate::graph::GraphConfig;
    use crate::search::{BaselineGraphDraft, CoherenceGatedGraphDraft};

    fn build(n: usize, dims: usize) -> (DraftGraph, ScalarQuantizer, Dataset) {
        let ds = Dataset::generate(n, dims, 10, 999);
        let quantizer = ScalarQuantizer::train(&ds.corpus_vectors());
        let graph = DraftGraph::build(
            ds.corpus.clone(),
            &quantizer,
            GraphConfig {
                m: 10,
                m_longjump: 4,
                dims,
            },
        );
        (graph, quantizer, ds)
    }

    #[test]
    fn baseline_graph_draft_returns_k_hits() {
        let (graph, quantizer, ds) = build(300, 16);
        let v = GraphDraftSpeculative::new(graph, quantizer, BaselineGraphDraft, 20, 40);
        let hits = v.search(ds.query(0), 10);
        assert_eq!(hits.len(), 10);
    }

    #[test]
    fn coherence_gated_tracks_lower_draft_cost() {
        let (graph, quantizer, ds) = build(300, 16);
        let graph2_vecs = ds.corpus.clone();
        let quantizer2 = ScalarQuantizer::train(&ds.corpus_vectors());
        let graph2 = DraftGraph::build(
            graph2_vecs,
            &quantizer2,
            GraphConfig {
                m: 10,
                m_longjump: 4,
                dims: 16,
            },
        );

        let base = GraphDraftSpeculative::new(graph, quantizer, BaselineGraphDraft, 20, 40);
        let gated = GraphDraftSpeculative::new(
            graph2,
            quantizer2,
            CoherenceGatedGraphDraft { threshold: 0.15 },
            20,
            40,
        );

        for i in 0..ds.n_queries {
            base.search(ds.query(i), 10);
            gated.search(ds.query(i), 10);
        }

        assert!(gated.last_draft_cost().unwrap() <= base.last_draft_cost().unwrap());
    }
}
