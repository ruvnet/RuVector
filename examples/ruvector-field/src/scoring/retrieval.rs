//! Candidate scoring — spec section 8.3.
//!
//! ```text
//! candidate_score = semantic_similarity
//!                 * shell_fit
//!                 * coherence_fit
//!                 * continuity_fit
//!                 * resonance_fit
//! risk            = contradiction_risk + drift_risk + policy_risk
//! safety          = 1 / (1 + risk)
//! final_score     = candidate_score * safety * (1 + novelty_bonus * 0.2)
//! ```
//!
//! The small `novelty_bonus` term is the geometric-antipode novelty boost
//! from spec 10.1.

use crate::model::{Embedding, FieldNode, Shell};

/// Factors pulled out so callers can log individual components.
#[derive(Debug, Clone, Copy)]
pub struct CandidateFactors {
    /// Semantic similarity mapped into `[0, 1]`.
    pub semantic_similarity: f32,
    /// Closer to target shell = higher.
    pub shell_fit: f32,
    /// Node coherence.
    pub coherence_fit: f32,
    /// Node continuity.
    pub continuity_fit: f32,
    /// Resonance-fit sigmoid.
    pub resonance_fit: f32,
    /// Novelty against geometric antipodes of already-selected results.
    pub novelty_bonus: f32,
    /// Contradiction risk in `[0, 1]`.
    pub contradiction_risk: f32,
    /// Drift risk in `[0, 1]`.
    pub drift_risk: f32,
    /// Policy risk in `[0, 1]`.
    pub policy_risk: f32,
}

impl CandidateFactors {
    /// Final score with safety and novelty applied.
    pub fn final_score(&self) -> f32 {
        let candidate = self.semantic_similarity
            * self.shell_fit
            * self.coherence_fit
            * self.continuity_fit
            * self.resonance_fit;
        let risk = self.contradiction_risk + self.drift_risk + self.policy_risk;
        let safety = 1.0 / (1.0 + risk);
        (candidate * safety * (1.0 + self.novelty_bonus * 0.2)).max(0.0)
    }
}

/// Score a single candidate against a query.
///
/// `already_selected_antipodes` is the list of geometric antipodes of the
/// nodes already chosen in this pass — see spec 10.1.
///
/// # Example
///
/// ```
/// use ruvector_field::model::{AxisScores, Embedding, EmbeddingId, FieldNode, NodeId, NodeKind, Shell};
/// use ruvector_field::scoring::retrieval::score_candidate;
/// let q = Embedding::new(vec![1.0, 0.0, 0.0]);
/// let e = Embedding::new(vec![0.9, 0.1, 0.0]);
/// let node = FieldNode {
///     id: NodeId(1), kind: NodeKind::Interaction,
///     semantic_embedding: EmbeddingId(1), geometric_antipode: EmbeddingId(2),
///     semantic_antipode: None, shell: Shell::Event,
///     axes: AxisScores::new(0.8, 0.7, 0.6, 0.8),
///     coherence: 0.9, continuity: 0.8, resonance: 0.5,
///     policy_mask: 0, witness_ref: None, ts_ns: 0, temporal_bucket: 0,
///     text: String::new(), shell_entered_ts: 0, promotion_streak: 0,
///     promotion_history: vec![], selection_count: 0, contradiction_hits: 0,
///     edges_at_last_tick: 0,
/// };
/// let factors = score_candidate(&q, &e, &node, Shell::Event, 0.0, 0.0, 0.0, &[]);
/// assert!(factors.final_score() > 0.0);
/// ```
pub fn score_candidate(
    query: &Embedding,
    candidate_embedding: &Embedding,
    candidate_node: &FieldNode,
    target_shell: Shell,
    drift_risk: f32,
    policy_risk: f32,
    contradiction_risk: f32,
    already_selected_antipodes: &[&Embedding],
) -> CandidateFactors {
    let raw_sim = query.cosine(candidate_embedding).clamp(-1.0, 1.0);
    let semantic_similarity = ((raw_sim + 1.0) / 2.0).clamp(0.0, 1.0);

    // Shell fit: full credit when depths match, decays linearly with depth gap.
    let depth_gap = (candidate_node.shell.depth() as i32 - target_shell.depth() as i32).abs() as f32;
    let shell_fit = (1.0 - depth_gap * 0.15).clamp(0.1, 1.0);

    let coherence_fit = candidate_node.coherence.clamp(0.0, 1.0);
    let continuity_fit = candidate_node.continuity.clamp(0.0, 1.0);
    let resonance_fit = (0.5 + 0.5 * candidate_node.resonance).clamp(0.0, 1.0);

    // Novelty — geometric antipode term from spec 10.1. High when the
    // candidate is far from the geometric antipodes of already-selected results.
    let novelty_bonus = if already_selected_antipodes.is_empty() {
        0.0
    } else {
        let worst = already_selected_antipodes
            .iter()
            .map(|a| candidate_embedding.cosine01(a))
            .fold(0.0_f32, f32::max);
        (1.0 - worst).clamp(0.0, 1.0)
    };

    CandidateFactors {
        semantic_similarity,
        shell_fit,
        coherence_fit,
        continuity_fit,
        resonance_fit,
        novelty_bonus,
        contradiction_risk: contradiction_risk.clamp(0.0, 1.0),
        drift_risk: drift_risk.clamp(0.0, 1.0),
        policy_risk: policy_risk.clamp(0.0, 1.0),
    }
}
