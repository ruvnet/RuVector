//! Temporal decay functions for memory scoring.
//!
//! All functions return a multiplier in [0, 1] to apply to cosine similarity.

/// How temporal decay is computed.
#[derive(Clone, Debug)]
pub enum DecayKind {
    /// No decay — all memories score equally regardless of age.
    None,
    /// Linear decay: score = max(0, 1 − age / half_life).
    Linear { half_life: u64 },
    /// Exponential decay: score = e^(-lambda * age).
    /// lambda = ln(2) / half_life reproduces the classic half-life model.
    Exponential { lambda: f32 },
}

/// Bundle of decay configuration and query timestamp.
#[derive(Clone, Debug)]
pub struct DecayConfig {
    pub kind: DecayKind,
    /// Current query time; memories older than this are in the past.
    pub now: u64,
}

impl DecayConfig {
    pub fn none(now: u64) -> Self {
        Self {
            kind: DecayKind::None,
            now,
        }
    }

    pub fn linear(now: u64, half_life: u64) -> Self {
        Self {
            kind: DecayKind::Linear { half_life },
            now,
        }
    }

    pub fn exponential(now: u64, half_life: u64) -> Self {
        let lambda = std::f32::consts::LN_2 / half_life as f32;
        Self {
            kind: DecayKind::Exponential { lambda },
            now,
        }
    }

    /// Returns a multiplier in [0, 1].
    pub fn factor(&self, memory_ts: u64) -> f32 {
        let age = self.now.saturating_sub(memory_ts);
        match self.kind {
            DecayKind::None => 1.0,
            DecayKind::Linear { half_life } => {
                let h = half_life.max(1) as f32;
                (1.0 - age as f32 / h).max(0.0)
            }
            DecayKind::Exponential { lambda } => (-lambda * age as f32).exp(),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn none_always_one() {
        let cfg = DecayConfig::none(1000);
        assert_eq!(cfg.factor(0), 1.0);
        assert_eq!(cfg.factor(1000), 1.0);
    }

    #[test]
    fn linear_at_half_life() {
        let cfg = DecayConfig::linear(1000, 500);
        // age = 500 → 1 - 500/500 = 0
        let f = cfg.factor(500);
        assert!(f.abs() < 1e-5, "factor={f}");
    }

    #[test]
    fn linear_at_zero_age() {
        let cfg = DecayConfig::linear(1000, 500);
        let f = cfg.factor(1000);
        assert!((f - 1.0).abs() < 1e-5, "factor={f}");
    }

    #[test]
    fn exponential_at_half_life() {
        let cfg = DecayConfig::exponential(1000, 500);
        let f = cfg.factor(500); // age = 500 = half_life → should be ~0.5
        assert!((f - 0.5).abs() < 0.01, "factor={f}");
    }

    #[test]
    fn exponential_at_zero_age() {
        let cfg = DecayConfig::exponential(1000, 500);
        let f = cfg.factor(1000);
        assert!((f - 1.0).abs() < 1e-5, "factor={f}");
    }

    #[test]
    fn decay_never_exceeds_one() {
        let cfg = DecayConfig::exponential(500, 200);
        // future memory (ts > now) — age saturates to 0 via saturating_sub
        let f = cfg.factor(600);
        assert!(f <= 1.0 + 1e-5, "factor={f}");
        assert!(f >= 0.0, "factor={f}");
    }
}
