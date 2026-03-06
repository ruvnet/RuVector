//! Search result ranking with quality and recency adjustments.
//!
//! The ranking engine applies lightweight quality and recency bonuses on top
//! of the hybrid score (embedding + keyword + reputation) computed by the
//! search handler.  It intentionally preserves the input ordering's relative
//! signal rather than overriding it.

use crate::types::BrainMemory;

/// Rank search results using quality, recency, and the incoming hybrid score.
pub struct RankingEngine {
    quality_weight: f32,
    recency_weight: f32,
    similarity_weight: f32,
}

impl RankingEngine {
    pub fn new(_embedding_dim: usize) -> Self {
        Self {
            quality_weight: 0.10,
            recency_weight: 0.05,
            similarity_weight: 0.85,
        }
    }

    /// Rank memories by composite score.
    ///
    /// The incoming `sim_score` is the hybrid score from the search handler
    /// (embedding + keyword + reputation).  We apply small quality and
    /// recency bonuses on top — never override.
    pub fn rank(&self, results: &mut [(f64, BrainMemory)]) {
        let now = chrono::Utc::now();

        for (sim_score, memory) in results.iter_mut() {
            let quality = memory.quality_score.mean();
            let age_hours = (now - memory.updated_at).num_hours().max(1) as f64;
            let recency = 1.0 / (1.0 + (age_hours / 24.0).ln_1p());

            let composite = self.similarity_weight as f64 * *sim_score
                + self.quality_weight as f64 * quality
                + self.recency_weight as f64 * recency;

            *sim_score = composite;
        }

        results.sort_by(|a, b| b.0.partial_cmp(&a.0).unwrap_or(std::cmp::Ordering::Equal));
    }
}

impl Default for RankingEngine {
    fn default() -> Self {
        Self::new(128)
    }
}
