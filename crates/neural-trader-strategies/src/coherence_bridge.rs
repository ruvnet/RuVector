//! Bridge between [`neural_trader_coherence::CoherenceGate`] and the
//! [`crate::Strategy`]/[`crate::RiskGate`] pipeline.
//!
//! The coherence gate models the neural trader's *internal* state
//! confidence (mincut floor, CUSUM drift, regime). When it reports
//! `allow_act = false`, no strategy output should reach the exchange —
//! even if the `RiskGate` would otherwise approve.
//!
//! This module provides a thin wrapper so the full actuation path is:
//!
//!     Strategy → Intent → CoherenceChecker.check → RiskGate → Order
//!
//! The wrapper is lightweight by design: operators may want to skip the
//! coherence gate in paper-only flows (which they can, by not
//! constructing a `CoherenceChecker`).

pub use neural_trader_coherence::{
    CoherenceDecision, CoherenceGate, GateConfig, GateContext, RegimeLabel, ThresholdGate,
};

use crate::intent::Intent;

/// Pre-order coherence check. Uses any `CoherenceGate` impl to decide
/// whether an intent is allowed to proceed to the `RiskGate`.
pub struct CoherenceChecker<G: CoherenceGate> {
    pub gate: G,
}

/// Outcome of a coherence check.
#[derive(Debug, Clone, PartialEq)]
pub enum CoherenceOutcome {
    /// Coherence gate approved — proceed to the risk gate.
    Pass(Intent),
    /// Gate blocked actuation. Intent returned unchanged for logging.
    Block {
        intent: Intent,
        decision: CoherenceDecision,
    },
}

impl<G: CoherenceGate> CoherenceChecker<G> {
    pub fn new(gate: G) -> Self {
        Self { gate }
    }

    /// Run the gate. On error the intent is blocked defensively (fail
    /// closed — an internal gate failure must never authorize actuation).
    pub fn check(&self, intent: Intent, ctx: &GateContext) -> CoherenceOutcome {
        match self.gate.evaluate(ctx) {
            Ok(d) if d.allow_act => CoherenceOutcome::Pass(intent),
            Ok(d) => CoherenceOutcome::Block {
                intent,
                decision: d,
            },
            Err(e) => CoherenceOutcome::Block {
                intent,
                decision: CoherenceDecision {
                    allow_retrieve: false,
                    allow_write: false,
                    allow_learn: false,
                    allow_act: false,
                    mincut_value: 0,
                    partition_hash: [0u8; 16],
                    drift_score: 0.0,
                    cusum_score: 0.0,
                    reasons: vec![format!("gate error: {e}")],
                },
            },
        }
    }
}

/// Build a plausible [`GateContext`] from a sliding window of observed
/// events. This is a pragmatic default — operators with richer
/// graph/embedding pipelines should inject their own context instead.
pub fn simple_context(
    symbol_id: u32,
    venue_id: u16,
    ts_ns: u64,
    n_levels: u64,
    recent_price_cents: &[i64],
) -> GateContext {
    // CUSUM approx: cumulative absolute price-return over the window.
    let cusum = recent_price_cents
        .windows(2)
        .map(|w| (w[1] - w[0]).unsigned_abs() as f32)
        .sum::<f32>();
    // Drift: fraction of window with price moves ≥ 2¢.
    let big_moves = recent_price_cents
        .windows(2)
        .filter(|w| (w[1] - w[0]).abs() >= 2)
        .count();
    let drift = if recent_price_cents.len() > 1 {
        big_moves as f32 / (recent_price_cents.len() - 1) as f32
    } else {
        0.0
    };
    let regime = if cusum > 20.0 {
        RegimeLabel::Volatile
    } else if cusum > 5.0 {
        RegimeLabel::Normal
    } else {
        RegimeLabel::Calm
    };
    GateContext {
        symbol_id,
        venue_id,
        ts_ns,
        mincut_value: n_levels,
        partition_hash: [0u8; 16],
        cusum_score: cusum,
        drift_score: drift,
        regime,
        boundary_stable_count: 10,
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::intent::{Action, Intent, Side};

    fn sample_intent() -> Intent {
        Intent {
            symbol_id: 1,
            side: Side::Yes,
            action: Action::Buy,
            limit_price_cents: 24,
            quantity: 10,
            edge_bps: 600,
            confidence: 0.5,
            strategy: "t",
        }
    }

    #[test]
    fn healthy_context_passes() {
        let checker = CoherenceChecker::new(ThresholdGate::new(GateConfig::default()));
        let ctx = simple_context(1, 1001, 0, 20, &[24, 25, 24]);
        match checker.check(sample_intent(), &ctx) {
            CoherenceOutcome::Pass(_) => {}
            other => panic!("expected Pass, got {other:?}"),
        }
    }

    #[test]
    fn low_mincut_blocks() {
        let checker = CoherenceChecker::new(ThresholdGate::new(GateConfig::default()));
        // n_levels=1 is well below any regime floor.
        let ctx = simple_context(1, 1001, 0, 1, &[24, 25]);
        let out = checker.check(sample_intent(), &ctx);
        match out {
            CoherenceOutcome::Block { decision, .. } => {
                assert!(!decision.allow_act);
                assert!(!decision.reasons.is_empty());
            }
            other => panic!("expected Block, got {other:?}"),
        }
    }

    #[test]
    fn simple_context_volatile_regime() {
        // Large swings → CUSUM > 20 → Volatile regime.
        let ctx = simple_context(1, 1001, 0, 10, &[10, 30, 10, 30, 10]);
        assert!(matches!(ctx.regime, RegimeLabel::Volatile));
    }
}
