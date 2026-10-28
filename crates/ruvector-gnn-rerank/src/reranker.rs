//! Reranker variants for approximate ANN candidates.
//!
//! ## Why graph score diffusion improves recall
//!
//! Approximate retrievers (RaBitQ, RAIRS IVF, coarse HNSW) introduce
//! independent, zero-mean noise into distance estimates.  True top-k candidates
//! for a given query are typically drawn from the same vector cluster — so in
//! the candidate k-NN graph they are mutually connected.  Averaging noisy scores
//! across this neighbourhood cancels the noise (law of large numbers), pushing
//! true positives back toward the top of the ranking.  False positives with
//! artificially high noisy scores are isolated from the true cluster, so
//! diffusion reduces rather than amplifies their scores.
//!
//! This is the discrete analogue of graph spectral low-pass filtering:
//! diffusion preserves low-frequency (cluster-level) signals while attenuating
//! high-frequency (per-item noise) components.

use crate::{error::RerankerError, graph::CandidateGraph};

// ── public types ─────────────────────────────────────────────────────────────

/// A vector from approximate ANN retrieval.
pub struct Candidate {
    /// Corpus index of this vector.
    pub id: u32,
    /// Full-precision vector fetched from the store.
    pub vector: Vec<f32>,
    /// Approximate similarity score from the quantised / coarse first-stage index.
    /// Convention: **higher = closer to query**.
    pub noisy_score: f32,
}

/// A reranked result.
#[derive(Debug, Clone)]
pub struct RankedResult {
    pub id: u32,
    pub score: f32,
}

/// Rerank a set of approximate ANN candidates.
pub trait CandidateReranker {
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError>;
}

// ── shared helpers ────────────────────────────────────────────────────────────

fn validate(candidates: &[Candidate], k: usize) -> Result<(), RerankerError> {
    if candidates.is_empty() {
        return Err(RerankerError::Empty);
    }
    if k > candidates.len() {
        return Err(RerankerError::KTooLarge {
            k,
            n: candidates.len(),
        });
    }
    // Reject untrusted input that would silently corrupt the ranking: all
    // candidate vectors must share one dimension and contain only finite values,
    // and scores must be finite. Fail-fast vs. producing a poisoned ranking.
    let dim = candidates[0].vector.len();
    for c in candidates {
        if c.vector.len() != dim {
            return Err(RerankerError::DimMismatch {
                query: dim,
                candidate: c.vector.len(),
            });
        }
        if !c.noisy_score.is_finite() {
            return Err(RerankerError::NonFinite {
                what: "candidate score",
            });
        }
        if c.vector.iter().any(|x| !x.is_finite()) {
            return Err(RerankerError::NonFinite {
                what: "candidate vector",
            });
        }
    }
    Ok(())
}

fn sort_take(mut scored: Vec<(usize, f32)>, k: usize, cands: &[Candidate]) -> Vec<RankedResult> {
    scored.sort_unstable_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));
    scored
        .into_iter()
        .take(k)
        .map(|(i, s)| RankedResult {
            id: cands[i].id,
            score: s,
        })
        .collect()
}

fn l2_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| (x - y).powi(2))
        .sum::<f32>()
        .sqrt()
}

// ── Variant 1: NoisyScoreReranker ─────────────────────────────────────────────

/// Passthrough reranker: sort candidates by their original approximate score.
///
/// Baseline.  Represents what a quantised ANN index gives without any
/// post-retrieval reranking step.
pub struct NoisyScoreReranker;

impl CandidateReranker for NoisyScoreReranker {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError> {
        validate(candidates, k)?;
        let scored = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, c.noisy_score))
            .collect();
        Ok(sort_take(scored, k, candidates))
    }
}

// ── Variant 2: GnnDiffusionReranker ──────────────────────────────────────────

/// 1-hop GNN score diffusion reranker.
///
/// Builds a cosine k-NN graph over the candidate set (O(n²×dim)), then runs
/// `hops` rounds of score averaging:
///
/// ```text
/// s_i^{t+1} = α · s_i^t + (1-α) · mean_{j ∈ N(i)} s_j^t
/// ```
///
/// Inspired by PassageRank (2503.14802), GNRR (2406.11720), and the
/// graph spectral filtering literature.
pub struct GnnDiffusionReranker {
    /// Self-weight in each diffusion round.  Range (0, 1).  Default: 0.60.
    pub alpha: f32,
    /// Number of message-passing hops.  Default: 1.
    pub hops: usize,
    /// Neighbours per candidate in the candidate k-NN graph.  Default: 8.
    pub k_graph: usize,
}

impl Default for GnnDiffusionReranker {
    fn default() -> Self {
        Self {
            alpha: 0.60,
            hops: 1,
            k_graph: 8,
        }
    }
}

impl CandidateReranker for GnnDiffusionReranker {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError> {
        validate(candidates, k)?;
        let n = candidates.len();
        let graph = CandidateGraph::build(candidates, self.k_graph);
        let mut scores: Vec<f32> = candidates.iter().map(|c| c.noisy_score).collect();

        for _ in 0..self.hops {
            let prev = scores.clone();
            for i in 0..n {
                if graph.edges[i].is_empty() {
                    continue;
                }
                let mean_nbr: f32 = graph.edges[i].iter().map(|&(j, _)| prev[j]).sum::<f32>()
                    / graph.edges[i].len() as f32;
                scores[i] = self.alpha * prev[i] + (1.0 - self.alpha) * mean_nbr;
            }
        }

        let scored = scores.into_iter().enumerate().collect();
        Ok(sort_take(scored, k, candidates))
    }
}

// ── Variant 3: GnnMincutReranker ─────────────────────────────────────────────

/// Coherence-gated GNN reranker (mincut-inspired).
///
/// Extends `GnnDiffusionReranker` with structural edge gating: only propagates
/// score across edges where the **cosine similarity between candidates** exceeds
/// `coherence_threshold`.  This gates diffusion on vector-space structure rather
/// than on noisy scores, preventing score bleeding across semantic cluster
/// boundaries.
///
/// Rationale: score-ratio gating (min/max of noisy scores) is too conservative —
/// a true positive that received a low noisy score has incoherent edges with its
/// correctly-scored true-positive neighbours, so ratio gating blocks exactly the
/// edges that would help.  Structural gating avoids this failure mode.
///
/// Inspired by mincut coherence gating in `ruvector-attn-mincut` and
/// `ruvector-mincut`.
pub struct GnnMincutReranker {
    /// Self-weight in gated diffusion.  Default: 0.60.
    pub alpha: f32,
    /// Minimum cosine similarity between candidates to propagate.  Default: 0.50.
    pub coherence_threshold: f32,
    /// Neighbours per candidate in the candidate k-NN graph.  Default: 8.
    pub k_graph: usize,
}

impl Default for GnnMincutReranker {
    fn default() -> Self {
        Self {
            alpha: 0.60,
            coherence_threshold: 0.50,
            k_graph: 8,
        }
    }
}

impl CandidateReranker for GnnMincutReranker {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError> {
        validate(candidates, k)?;
        let n = candidates.len();
        let graph = CandidateGraph::build(candidates, self.k_graph);

        let mut scores: Vec<f32> = candidates.iter().map(|c| c.noisy_score).collect();

        let prev = scores.clone();
        for i in 0..n {
            let mut w_sum = 0.0_f32;
            let mut w_total = 0.0_f32;
            // Gate: only propagate across structurally coherent edges.
            for &(j, sim) in &graph.edges[i] {
                if sim >= self.coherence_threshold {
                    w_sum += sim * prev[j];
                    w_total += sim;
                }
            }
            if w_total > 0.0 {
                let weighted_mean = w_sum / w_total;
                scores[i] = self.alpha * prev[i] + (1.0 - self.alpha) * weighted_mean;
            }
        }

        let scored = scores.into_iter().enumerate().collect();
        Ok(sort_take(scored, k, candidates))
    }
}

// ── Variant 4: AdaptiveCoherenceReranker ─────────────────────────────────────

/// Per-node adaptive coherence-gated diffusion reranker.
///
/// ## Why a global threshold breaks under metadata filtering
///
/// `GnnMincutReranker` gates every edge against one **global absolute**
/// cosine-similarity cutoff (`coherence_threshold`). That cutoff is implicitly
/// calibrated against the similarity distribution of a full, unfiltered
/// candidate pool. When an upstream metadata filter (tenant scoping, ACLs,
/// freshness windows, …) shrinks the pool — especially when the filter is
/// independent of semantic cluster membership — the *local* similarity
/// structure around each surviving node sparsifies unevenly: some nodes keep
/// several same-cluster neighbours, others are left with only cross-cluster
/// neighbours below the global bar, or none at all. Nodes in the latter group
/// silently fall back to their unfiltered, un-reranked `noisy_score`, i.e. the
/// reranker degenerates toward `NoisyScoreReranker` exactly where reranking
/// would help most (small, noisy post-filter candidate sets).
///
/// `AdaptiveCoherenceReranker` gates **per node, relative to that node's own**
/// local neighbour-similarity ranking rather than against one fixed absolute
/// bar: each node keeps its own top `keep_fraction` of edges (floored at
/// `min_keep` whenever it has any edges at all). A node's gate therefore
/// adapts automatically to how sparse its own neighbourhood has become after
/// filtering, instead of assuming the pre-filter similarity scale still
/// applies. This trades some of `GnnMincutReranker`'s hard rejection of
/// genuinely incoherent edges (an absolute bar can veto *all* of a node's
/// edges when none are locally relevant) for a gate that degrades gracefully
/// as the candidate pool shrinks.
pub struct AdaptiveCoherenceReranker {
    /// Self-weight in gated diffusion. Default: 0.60 (matches `GnnMincutReranker`
    /// for a fair baseline comparison).
    pub alpha: f32,
    /// Fraction ρ ∈ (0, 1] of each node's own local edges to keep after
    /// per-node gating. Default: 0.50.
    pub keep_fraction: f32,
    /// Floor on the number of edges kept per node, applied only when the node
    /// has at least this many edges available. Default: 1.
    pub min_keep: usize,
    /// Neighbours per candidate in the candidate k-NN graph. Default: 8.
    pub k_graph: usize,
}

impl Default for AdaptiveCoherenceReranker {
    fn default() -> Self {
        Self {
            alpha: 0.60,
            keep_fraction: 0.50,
            min_keep: 1,
            k_graph: 8,
        }
    }
}

impl CandidateReranker for AdaptiveCoherenceReranker {
    fn rerank(
        &self,
        _query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError> {
        validate(candidates, k)?;
        let n = candidates.len();
        let graph = CandidateGraph::build(candidates, self.k_graph);

        let mut scores: Vec<f32> = candidates.iter().map(|c| c.noisy_score).collect();
        let prev = scores.clone();

        for i in 0..n {
            let deg = graph.edges[i].len();
            if deg == 0 {
                continue;
            }
            // `graph.edges[i]` is already sorted descending by similarity
            // (CandidateGraph::build), so truncating keeps each node's own
            // strongest local edges.
            let keep = ((self.keep_fraction * deg as f32).ceil() as usize)
                .max(self.min_keep)
                .min(deg);
            let mut w_sum = 0.0_f32;
            let mut w_total = 0.0_f32;
            for &(j, sim) in &graph.edges[i][..keep] {
                w_sum += sim * prev[j];
                w_total += sim;
            }
            if w_total > 0.0 {
                let weighted_mean = w_sum / w_total;
                scores[i] = self.alpha * prev[i] + (1.0 - self.alpha) * weighted_mean;
            }
        }

        let scored = scores.into_iter().enumerate().collect();
        Ok(sort_take(scored, k, candidates))
    }
}

// ── Oracle: ExactL2Reranker ───────────────────────────────────────────────────

/// Oracle reranker: sort by exact Euclidean distance to the query.
///
/// Upper bound for any reranker that sees the same candidate set.
/// Requires fetching and scoring all full-precision candidate vectors,
/// which is the expensive but optimal baseline.
pub struct ExactL2Reranker;

impl CandidateReranker for ExactL2Reranker {
    fn rerank(
        &self,
        query: &[f32],
        candidates: &[Candidate],
        k: usize,
    ) -> Result<Vec<RankedResult>, RerankerError> {
        validate(candidates, k)?;
        // Negate distance so sort_take (descending) picks the closest.
        let scored = candidates
            .iter()
            .enumerate()
            .map(|(i, c)| (i, -l2_dist(query, &c.vector)))
            .collect();
        Ok(sort_take(scored, k, candidates))
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_cand(id: u32, v: Vec<f32>, score: f32) -> Candidate {
        Candidate {
            id,
            vector: v,
            noisy_score: score,
        }
    }

    #[test]
    fn noisy_picks_highest_score() {
        let cands = vec![
            make_cand(0, vec![0.0, 0.0], 0.2),
            make_cand(1, vec![1.0, 0.0], 0.9),
            make_cand(2, vec![0.0, 1.0], 0.5),
        ];
        let r = NoisyScoreReranker.rerank(&[0.0, 0.0], &cands, 1).unwrap();
        assert_eq!(r[0].id, 1);
    }

    #[test]
    fn exact_l2_picks_closest_vector() {
        let cands = vec![
            make_cand(0, vec![10.0, 0.0], 0.9), // far, high noisy score
            make_cand(1, vec![0.1, 0.0], 0.1),  // close, low noisy score
            make_cand(2, vec![5.0, 0.0], 0.5),
        ];
        let query = vec![0.0, 0.0];
        let r = ExactL2Reranker.rerank(&query, &cands, 1).unwrap();
        assert_eq!(
            r[0].id, 1,
            "ExactL2 must prefer the geometrically closest vector"
        );
    }

    #[test]
    fn diffusion_produces_k_results() {
        let cands: Vec<Candidate> = (0..10)
            .map(|i| make_cand(i, vec![i as f32, 0.0], 0.5 + i as f32 * 0.04))
            .collect();
        let r = GnnDiffusionReranker::default()
            .rerank(&[0.0, 0.0], &cands, 3)
            .unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn mincut_reranker_produces_k_results() {
        let cands: Vec<Candidate> = (0..10)
            .map(|i| make_cand(i, vec![i as f32, 0.0], 0.5 + i as f32 * 0.04))
            .collect();
        let r = GnnMincutReranker::default()
            .rerank(&[0.0, 0.0], &cands, 3)
            .unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn adaptive_coherence_returns_k_results() {
        let cands: Vec<Candidate> = (0..10)
            .map(|i| make_cand(i, vec![i as f32, 0.0], 0.5 + i as f32 * 0.04))
            .collect();
        let r = AdaptiveCoherenceReranker::default()
            .rerank(&[0.0, 0.0], &cands, 3)
            .unwrap();
        assert_eq!(r.len(), 3);
    }

    #[test]
    fn adaptive_coherence_single_candidate_is_unchanged() {
        // A lone candidate has a zero-degree graph: the reranker must not
        // panic and must leave its score untouched (nothing to diffuse from).
        let cands = vec![make_cand(0, vec![1.0, 0.0], 0.42)];
        let r = AdaptiveCoherenceReranker::default()
            .rerank(&[0.0, 0.0], &cands, 1)
            .unwrap();
        assert_eq!(r.len(), 1);
        assert!(
            (r[0].score - 0.42).abs() < 1e-6,
            "lone candidate with zero-degree graph must keep its original score, got {}",
            r[0].score
        );
    }

    #[test]
    fn min_keep_floor_changes_diffusion_result() {
        // Node 0 has three neighbours of decreasing similarity (1 ≈8°, 2 ≈45°,
        // 3 ≈90°). keep_fraction=0.30 alone floors to keeping only the single
        // closest neighbour (ceil(0.30*3)=1); `min_keep` forces a wider
        // neighbourhood to be used instead. With alpha=0.0 the resulting score
        // for node 0 is exactly the kept-neighbour weighted mean, so a larger
        // min_keep must dilute the pure "closest neighbour" signal (score 1.0)
        // toward the weaker, near-orthogonal neighbours (score 0.0).
        let cands = vec![
            make_cand(0, vec![1.0, 0.0], 0.0),
            make_cand(1, vec![0.99, 0.14], 1.0),
            make_cand(2, vec![0.70, 0.71], 0.0),
            make_cand(3, vec![0.0, 1.0], 0.0),
        ];
        let query = vec![0.0, 0.0];

        let narrow = AdaptiveCoherenceReranker {
            alpha: 0.0,
            keep_fraction: 0.30,
            min_keep: 1,
            k_graph: 3,
        };
        let wide = AdaptiveCoherenceReranker {
            alpha: 0.0,
            keep_fraction: 0.30,
            min_keep: 3,
            k_graph: 3,
        };

        let r_narrow = narrow.rerank(&query, &cands, 4).unwrap();
        let r_wide = wide.rerank(&query, &cands, 4).unwrap();

        let score0_narrow = r_narrow.iter().find(|r| r.id == 0).unwrap().score;
        let score0_wide = r_wide.iter().find(|r| r.id == 0).unwrap().score;

        assert!(
            score0_narrow > score0_wide,
            "min_keep=1 ({score0_narrow}) should weight only the single closest \
             neighbour (score≈1.0), while min_keep=3 ({score0_wide}) dilutes with \
             weaker neighbours"
        );
    }
}
