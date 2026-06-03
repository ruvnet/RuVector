//! Temporal decay weighting for age-aware ANN search.
//!
//! Older vectors receive a higher effective distance so recent memories
//! are preferred in top-k retrieval without changing stored embeddings.

/// Configuration for temporal decay.
#[derive(Debug, Clone)]
pub struct DecayConfig {
    /// Decay strength in [0.0, 10.0]. 0.0 = no decay.
    pub decay_strength: f32,
    /// Half-life in seconds. After this many seconds, decay weight ≈ 0.632 * decay_strength.
    pub half_life_secs: f64,
    /// Coherence gate: prune candidates older than this (seconds) when raw distance > cutoff.
    pub max_age_gate_secs: f64,
    /// Coherence gate distance cutoff. Only used in CoherenceGated variant.
    pub coherence_cutoff: f32,
}

impl DecayConfig {
    /// No decay — equivalent to standard flat search.
    pub fn no_decay() -> Self {
        Self {
            decay_strength: 0.0,
            half_life_secs: f64::MAX,
            max_age_gate_secs: f64::MAX,
            coherence_cutoff: f32::MAX,
        }
    }

    /// Standard temporal decay with given strength and half-life.
    pub fn standard(decay_strength: f32, half_life_secs: f64) -> Self {
        Self {
            decay_strength,
            half_life_secs,
            max_age_gate_secs: f64::MAX,
            coherence_cutoff: f32::MAX,
        }
    }

    /// Temporal decay with coherence gate: prune old+distant candidates.
    pub fn with_gate(
        decay_strength: f32,
        half_life_secs: f64,
        max_age_gate_secs: f64,
        coherence_cutoff: f32,
    ) -> Self {
        Self {
            decay_strength,
            half_life_secs,
            max_age_gate_secs,
            coherence_cutoff,
        }
    }

    /// Returns the temporal weight multiplier for a given age in seconds.
    ///
    /// Weight >= 1.0; older = higher weight = larger effective distance.
    /// Formula: `1.0 + decay_strength * (1.0 - exp(-age_secs / half_life_secs))`
    pub fn weight(&self, age_secs: f64) -> f32 {
        if self.decay_strength == 0.0 {
            return 1.0;
        }
        let ratio = (age_secs / self.half_life_secs) as f32;
        1.0 + self.decay_strength * (1.0 - (-ratio).exp())
    }

    /// Returns true if a candidate should be pruned by the coherence gate.
    ///
    /// A candidate is pruned when it is both old (age > max_age_gate_secs)
    /// and distant (raw_dist > coherence_cutoff).
    pub fn should_gate(&self, age_secs: f64, raw_dist: f32) -> bool {
        age_secs > self.max_age_gate_secs && raw_dist > self.coherence_cutoff
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_decay_returns_one() {
        let cfg = DecayConfig::no_decay();
        assert_eq!(cfg.weight(0.0), 1.0);
        assert_eq!(cfg.weight(1_000_000.0), 1.0);
    }

    #[test]
    fn decay_increases_with_age() {
        let cfg = DecayConfig::standard(2.0, 3600.0);
        let w0 = cfg.weight(0.0);
        let w1h = cfg.weight(3600.0);
        let w24h = cfg.weight(86400.0);
        assert!(w0 < w1h, "weight should increase with age");
        assert!(w1h < w24h);
        // At age=0: weight should be ~1.0
        assert!((w0 - 1.0).abs() < 1e-5);
        // At half_life: weight = 1 + 2*(1 - exp(-1)) ≈ 1 + 2*0.632 ≈ 2.264
        assert!((w1h - 2.264).abs() < 0.01, "got {w1h}");
    }

    #[test]
    fn gate_prunes_old_distant_vectors() {
        let cfg = DecayConfig::with_gate(1.0, 3600.0, 7200.0, 0.5);
        // old and distant: should be gated
        assert!(cfg.should_gate(10000.0, 0.8));
        // old but close: should not be gated
        assert!(!cfg.should_gate(10000.0, 0.3));
        // recent and distant: should not be gated
        assert!(!cfg.should_gate(100.0, 0.8));
    }
}
