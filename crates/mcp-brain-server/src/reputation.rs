//! Multi-factor reputation scoring system

use crate::types::ReputationScore;

/// Reputation manager that tracks and updates contributor reputations.
///
/// Accuracy uses a Bayesian prior Beta(1,1) to prevent early luck from
/// distorting scores. Expected accuracy = (upvotes+1)/(upvotes+downvotes+2).
/// Accuracy only influences composite after min_observations_for_accuracy (5).
pub struct ReputationManager {
    min_observations_for_penalty: u32,
    quality_threshold_for_penalty: f64,
}

impl ReputationManager {
    pub fn new() -> Self {
        Self {
            min_observations_for_penalty: 5,
            quality_threshold_for_penalty: 0.2,
        }
    }

    /// Update accuracy using Bayesian Beta(1,1) prior.
    /// upvotes/downvotes are total lifetime counts for this contributor.
    pub fn update_accuracy_bayesian(
        score: &mut ReputationScore,
        upvotes: u64,
        downvotes: u64,
        min_obs: u32,
    ) {
        let total = upvotes + downvotes;
        if total < min_obs as u64 {
            // Below minimum observations — use prior mean 0.5
            score.accuracy = 0.5;
        } else {
            // Bayesian expected accuracy: (upvotes+1)/(upvotes+downvotes+2)
            score.accuracy = (upvotes as f64 + 1.0) / (total as f64 + 2.0);
        }
        score.compute_composite();
    }

    /// Update accuracy based on vote outcome (EMA fallback for streaming updates)
    pub fn update_accuracy(score: &mut ReputationScore, was_upvoted: bool) {
        if was_upvoted {
            score.accuracy = (score.accuracy * 0.9) + 0.1;
        } else {
            score.accuracy = score.accuracy * 0.9;
        }
        score.compute_composite();
    }

    /// Update uptime (called on each contribution)
    pub fn record_activity(score: &mut ReputationScore) {
        score.uptime = (score.uptime * 0.95) + 0.05;
        score.compute_composite();
    }

    /// Check if poisoning penalty should apply
    pub fn check_poisoning_penalty(
        &self,
        score: &mut ReputationScore,
        downvote_count: u32,
        quality: f64,
    ) -> bool {
        if downvote_count >= self.min_observations_for_penalty
            && quality < self.quality_threshold_for_penalty
        {
            score.apply_poisoning_penalty();
            true
        } else {
            false
        }
    }

    /// Apply monthly decay for inactive users
    pub fn apply_inactivity_decay(score: &mut ReputationScore, months_inactive: f64) {
        if months_inactive > 0.0 {
            score.apply_decay(months_inactive);
        }
    }

    /// Get the contribution weight for a reputation level
    pub fn contribution_weight(score: &ReputationScore) -> f64 {
        // New users (composite ~0.1) are weighted 10x less
        score.composite.max(0.01)
    }
}

impl Default for ReputationManager {
    fn default() -> Self {
        Self::new()
    }
}
