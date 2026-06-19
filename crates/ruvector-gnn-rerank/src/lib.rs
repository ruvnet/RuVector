//! # ruvector-gnn-rerank
//!
//! GNN-enhanced candidate reranking for approximate ANN search.
//!
//! After a first-stage approximate retriever (HNSW, DiskANN, IVF) returns a
//! candidate set, this crate applies graph neural score diffusion over the
//! candidate k-NN subgraph to recover recall lost to quantisation noise.
//!
//! ## Variant summary
//!
//! | Variant | Algorithm | Design rationale |
//! |---------|-----------|-----------------|
//! | `NoisyScoreReranker` | passthrough | baseline — sorts by approximate score |
//! | `GnnDiffusionReranker` | 1-hop score propagation | cancels i.i.d. noise by averaging cluster neighbours |
//! | `GnnMincutReranker` | coherence-gated propagation | blocks cross-cluster pollution (mincut-inspired) |
//! | `ExactL2Reranker` | exact Euclidean sort | oracle upper bound |
//!
//! All four implement [`CandidateReranker`].
//!
//! ## Research context
//!
//! Nightly research 2026-05-21.  Design rationale in `docs/adr/ADR-194-gnn-rerank.md`.
//! Companion papers: GNRR (arXiv 2406.11720), Maniscope (arXiv 2602.15860),
//! AQR-HNSW (arXiv 2602.21600).

#![forbid(unsafe_code)]

pub mod error;
pub mod graph;
pub mod reranker;

pub use error::RerankerError;
pub use graph::CandidateGraph;
pub use reranker::{
    Candidate, CandidateReranker, ExactL2Reranker, GnnDiffusionReranker, GnnMincutReranker,
    NoisyScoreReranker, RankedResult,
};

#[cfg(test)]
mod tests {
    use super::*;

    fn make_candidates(n: usize, dim: usize, seed: u64) -> Vec<Candidate> {
        use rand::{rngs::StdRng, Rng, SeedableRng};
        let mut rng = StdRng::seed_from_u64(seed);
        (0..n)
            .map(|i| Candidate {
                id: i as u32,
                vector: (0..dim).map(|_| rng.gen_range(-1.0_f32..1.0)).collect(),
                noisy_score: rng.gen_range(0.1_f32..1.0),
            })
            .collect()
    }

    fn make_query(dim: usize) -> Vec<f32> {
        vec![0.0_f32; dim]
    }

    #[test]
    fn noisy_reranker_returns_k_results() {
        let cands = make_candidates(20, 8, 1);
        let query = make_query(8);
        let r = NoisyScoreReranker.rerank(&query, &cands, 5).unwrap();
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn gnn_diffusion_returns_k_results() {
        let cands = make_candidates(20, 8, 2);
        let query = make_query(8);
        let r = GnnDiffusionReranker::default()
            .rerank(&query, &cands, 5)
            .unwrap();
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn gnn_mincut_returns_k_results() {
        let cands = make_candidates(20, 8, 3);
        let query = make_query(8);
        let r = GnnMincutReranker::default()
            .rerank(&query, &cands, 5)
            .unwrap();
        assert_eq!(r.len(), 5);
    }

    #[test]
    fn exact_l2_returns_closest_to_origin() {
        let mut cands: Vec<Candidate> = (0..10)
            .map(|i| Candidate {
                id: i as u32,
                // id=0 is origin (closest), others are progressively farther
                vector: vec![(i as f32) * 0.5; 4],
                noisy_score: 0.5,
            })
            .collect();
        // Shuffle scores so noisy ordering would fail
        cands[0].noisy_score = 0.1; // lowest noisy score but closest
        cands[9].noisy_score = 0.9; // highest noisy score but farthest

        let query = vec![0.0_f32; 4];
        let r = ExactL2Reranker.rerank(&query, &cands, 3).unwrap();
        // Should pick id=0 (L2=0), id=1 (L2=0.5×sqrt(4)=1.0), id=2 first
        assert_eq!(
            r[0].id, 0,
            "ExactL2 must pick the true nearest neighbour first"
        );
    }

    #[test]
    fn k_too_large_returns_error() {
        let cands = make_candidates(5, 4, 4);
        let query = make_query(4);
        assert!(matches!(
            NoisyScoreReranker.rerank(&query, &cands, 10),
            Err(RerankerError::KTooLarge { .. })
        ));
    }

    #[test]
    fn empty_candidates_returns_error() {
        let cands: Vec<Candidate> = vec![];
        let query = make_query(4);
        assert!(matches!(
            NoisyScoreReranker.rerank(&query, &cands, 1),
            Err(RerankerError::Empty)
        ));
    }

    #[test]
    fn candidate_graph_has_correct_degree() {
        let cands = make_candidates(20, 8, 5);
        let k_graph = 4;
        let g = CandidateGraph::build(&cands, k_graph);
        for neighbours in &g.edges {
            assert!(neighbours.len() <= k_graph);
        }
    }
}
