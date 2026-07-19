//! Semantic temperature model for vector tier placement.
//!
//! `SemanticTemperature` combines three signals to produce a scalar "heat" value
//! that determines how important a vector is to keep in fast storage:
//!
//! * **Recency**: exponential decay since last access epoch.
//! * **Coherence**: mean L2 proximity to k sampled neighbours (tighter cluster →
//!   higher coherence → hotter).
//! * **Centrality**: log-normalised neighbour count within a fixed L2 threshold.

/// Per-vector metadata tracked by the tiered store.
#[derive(Clone, Debug, Default)]
pub struct VectorMeta {
    /// How many times this vector has appeared in search results.
    pub access_count: u64,
    /// Logical epoch of most recent access (set by caller).
    pub last_access_epoch: u64,
    /// Coherence score in [0, 1].  Computed lazily in `evaluate_tiers`.
    pub coherence_score: f32,
    /// Approximate number of vectors within `centrality_radius` L2 distance.
    pub graph_degree: u32,
}

/// Configuration for tier evaluation.
#[derive(Clone, Debug)]
pub struct TieringConfig {
    /// Weight of recency component in semantic temperature (0–1).
    pub recency_weight: f32,
    /// Weight of coherence component (0–1).
    pub coherence_weight: f32,
    /// Weight of centrality component (0–1).
    pub centrality_weight: f32,
    /// Exponential decay rate λ.  Higher → faster decay.
    pub decay_lambda: f32,
    /// L2 distance threshold for graph_degree counting.
    pub centrality_radius: f32,
    /// Number of random neighbours sampled for coherence.
    pub coherence_sample_k: usize,
    /// Maximum vectors in the hot tier.
    pub hot_capacity: usize,
    /// Maximum additional vectors in the warm tier.
    pub warm_capacity: usize,
}

impl Default for TieringConfig {
    fn default() -> Self {
        Self {
            recency_weight: 0.35,
            coherence_weight: 0.40,
            centrality_weight: 0.25,
            decay_lambda: 0.05,
            centrality_radius: 1.5,
            coherence_sample_k: 16,
            hot_capacity: 500,
            warm_capacity: 1500,
        }
    }
}

impl VectorMeta {
    /// Access-count score (baseline variant).
    #[inline]
    pub fn access_score(&self) -> f32 {
        self.access_count as f32
    }

    /// Pure coherence score (variant 2).
    #[inline]
    pub fn coherence_score(&self) -> f32 {
        self.coherence_score
    }

    /// Composite semantic temperature (variant 3).
    ///
    /// * `current_epoch` — caller's logical clock value.
    /// * `cfg` — tiering configuration.
    pub fn semantic_temperature(&self, current_epoch: u64, cfg: &TieringConfig) -> f32 {
        let age = current_epoch.saturating_sub(self.last_access_epoch) as f32;
        let recency = (-cfg.decay_lambda * age).exp();
        // log-normalise centrality; cap at 1.0 to bound the contribution
        let centrality = ((1.0 + self.graph_degree as f32).ln() / 5.0).min(1.0);
        cfg.recency_weight * recency
            + cfg.coherence_weight * self.coherence_score
            + cfg.centrality_weight * centrality
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn fresh_access_gives_high_recency() {
        let meta = VectorMeta {
            last_access_epoch: 100,
            ..Default::default()
        };
        let cfg = TieringConfig::default();
        // age == 0 → recency == 1.0
        let temp = meta.semantic_temperature(100, &cfg);
        // At least recency component must be near recency_weight
        assert!(
            temp >= cfg.recency_weight - 0.01,
            "recent vector should score high"
        );
    }

    #[test]
    fn high_coherence_scores_high() {
        let meta = VectorMeta {
            coherence_score: 1.0,
            last_access_epoch: 0,
            ..Default::default()
        };
        let cfg = TieringConfig {
            decay_lambda: 100.0,
            ..Default::default()
        }; // kill recency
        let temp = meta.semantic_temperature(1000, &cfg);
        // Only coherence remains
        assert!((temp - cfg.coherence_weight).abs() < 0.01);
    }

    #[test]
    fn high_degree_contributes_centrality() {
        // graph_degree 200 → ln(201)/5 ≈ 1.06 → capped to 1.0
        // centrality contribution = centrality_weight * 1.0 = 0.25
        let meta = VectorMeta {
            graph_degree: 200,
            last_access_epoch: 0,
            ..Default::default()
        };
        // Kill recency and coherence so only centrality contributes.
        let cfg = TieringConfig {
            decay_lambda: 100.0,
            ..Default::default()
        };
        let temp = meta.semantic_temperature(1000, &cfg);
        assert!(
            temp >= cfg.centrality_weight - 0.01,
            "expected temp ≥ {}, got {temp}",
            cfg.centrality_weight - 0.01
        );
    }
}
