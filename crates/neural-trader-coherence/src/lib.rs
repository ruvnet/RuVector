//! # neural-trader-coherence
//!
//! MinCut coherence gate, CUSUM drift detection, and proof-gated
//! mutation protocol for the RuVector Neural Trader (ADR-084).

use serde::{Deserialize, Serialize};

// ---------------------------------------------------------------------------
// Coherence decision
// ---------------------------------------------------------------------------

/// Result of the coherence gate evaluation.
///
/// Every memory write, model update, retrieval, and actuation must pass
/// through this gate before proceeding.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct CoherenceDecision {
    pub allow_retrieve: bool,
    pub allow_write: bool,
    pub allow_learn: bool,
    pub allow_act: bool,
    pub mincut_value: u64,
    pub partition_hash: [u8; 16],
    pub drift_score: f32,
    pub cusum_score: f32,
    pub reasons: Vec<String>,
}

impl CoherenceDecision {
    /// Returns `true` if all gates are open.
    pub fn all_allowed(&self) -> bool {
        self.allow_retrieve && self.allow_write && self.allow_learn && self.allow_act
    }

    /// Returns `true` if nothing is allowed.
    pub fn fully_blocked(&self) -> bool {
        !self.allow_retrieve && !self.allow_write && !self.allow_learn && !self.allow_act
    }
}

// ---------------------------------------------------------------------------
// Gate context
// ---------------------------------------------------------------------------

/// Input context for the coherence gate evaluation.
#[derive(Debug, Clone)]
pub struct GateContext {
    pub symbol_id: u32,
    pub venue_id: u16,
    pub ts_ns: u64,
    /// Current mincut value from the local induced subgraph.
    pub mincut_value: u64,
    /// Partition hash over boundary nodes.
    pub partition_hash: [u8; 16],
    /// Rolling CUSUM score for drift detection.
    pub cusum_score: f32,
    /// Embedding drift magnitude since last stable window.
    pub drift_score: f32,
    /// Current regime label.
    pub regime: RegimeLabel,
    /// Number of consecutive windows with stable boundary identity.
    /// Gate requires this >= `GateConfig::boundary_stability_windows`.
    pub boundary_stable_count: usize,
}

/// Coarse regime classification.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RegimeLabel {
    Calm,
    Normal,
    Volatile,
}

// ---------------------------------------------------------------------------
// Gate configuration
// ---------------------------------------------------------------------------

/// Configurable thresholds for the coherence gate.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct GateConfig {
    pub mincut_floor_calm: u64,
    pub mincut_floor_normal: u64,
    pub mincut_floor_volatile: u64,
    pub cusum_threshold: f32,
    pub boundary_stability_windows: usize,
    pub max_drift_score: f32,
}

impl Default for GateConfig {
    fn default() -> Self {
        Self {
            mincut_floor_calm: 12,
            mincut_floor_normal: 9,
            mincut_floor_volatile: 6,
            cusum_threshold: 4.5,
            boundary_stability_windows: 8,
            max_drift_score: 0.5,
        }
    }
}

// ---------------------------------------------------------------------------
// Coherence gate trait and default implementation
// ---------------------------------------------------------------------------

/// Evaluates coherence of the current market-graph state.
pub trait CoherenceGate {
    fn evaluate(&self, ctx: &GateContext) -> anyhow::Result<CoherenceDecision>;
}

/// Default threshold-based coherence gate.
pub struct ThresholdGate {
    pub config: GateConfig,
}

impl ThresholdGate {
    pub fn new(config: GateConfig) -> Self {
        Self { config }
    }
}

impl CoherenceGate for ThresholdGate {
    fn evaluate(&self, ctx: &GateContext) -> anyhow::Result<CoherenceDecision> {
        let floor = match ctx.regime {
            RegimeLabel::Calm => self.config.mincut_floor_calm,
            RegimeLabel::Normal => self.config.mincut_floor_normal,
            RegimeLabel::Volatile => self.config.mincut_floor_volatile,
        };

        let cut_ok = ctx.mincut_value >= floor;
        let cusum_ok = ctx.cusum_score < self.config.cusum_threshold;
        let drift_ok = ctx.drift_score < self.config.max_drift_score;
        let boundary_ok = ctx.boundary_stable_count >= self.config.boundary_stability_windows;
        // Learning requires tighter drift margin (half the max).
        let learn_drift_ok = ctx.drift_score < self.config.max_drift_score * 0.5;

        let mut reasons = Vec::new();
        if !cut_ok {
            reasons.push(format!(
                "mincut {} below floor {} for {:?}",
                ctx.mincut_value, floor, ctx.regime
            ));
        }
        if !cusum_ok {
            reasons.push(format!(
                "CUSUM {:.3} exceeds threshold {:.3}",
                ctx.cusum_score, self.config.cusum_threshold
            ));
        }
        if !drift_ok {
            reasons.push(format!(
                "drift {:.3} exceeds max {:.3}",
                ctx.drift_score, self.config.max_drift_score
            ));
        }
        if !boundary_ok {
            reasons.push(format!(
                "boundary stable for {} windows, need {}",
                ctx.boundary_stable_count, self.config.boundary_stability_windows
            ));
        }

        let base_ok = cut_ok && cusum_ok && drift_ok && boundary_ok;

        Ok(CoherenceDecision {
            allow_retrieve: cut_ok, // retrieval is most permissive
            allow_write: base_ok,
            allow_learn: base_ok && learn_drift_ok, // stricter: half drift margin
            allow_act: base_ok,
            mincut_value: ctx.mincut_value,
            partition_hash: ctx.partition_hash,
            drift_score: ctx.drift_score,
            cusum_score: ctx.cusum_score,
            reasons,
        })
    }
}

// ---------------------------------------------------------------------------
// Proof-gated mutation
// ---------------------------------------------------------------------------

/// A verified mutation token. Only issued when the coherence gate and
/// policy kernel both approve.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct VerifiedToken {
    pub token_id: [u8; 16],
    pub ts_ns: u64,
    pub coherence_hash: [u8; 16],
    pub policy_hash: [u8; 16],
    pub action_intent: String,
}

/// Witness receipt appended after every state mutation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct WitnessReceipt {
    pub ts_ns: u64,
    pub model_id: String,
    pub input_segment_hash: [u8; 16],
    pub coherence_witness_hash: [u8; 16],
    pub policy_hash: [u8; 16],
    pub action_intent: String,
    pub verified_token_id: [u8; 16],
    pub resulting_state_hash: [u8; 16],
}

/// Logs witness receipts for auditability.
pub trait WitnessLogger {
    fn append_receipt(&mut self, receipt: WitnessReceipt) -> anyhow::Result<()>;
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    fn make_ctx(mincut: u64, cusum: f32, drift: f32, regime: RegimeLabel) -> GateContext {
        GateContext {
            symbol_id: 1,
            venue_id: 1,
            ts_ns: 0,
            mincut_value: mincut,
            partition_hash: [0u8; 16],
            cusum_score: cusum,
            drift_score: drift,
            regime,
            boundary_stable_count: 10, // above default threshold of 8
        }
    }

    #[test]
    fn calm_regime_passes_when_above_floor() {
        let gate = ThresholdGate::new(GateConfig::default());
        let ctx = make_ctx(15, 1.0, 0.1, RegimeLabel::Calm);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(d.all_allowed());
        assert!(d.reasons.is_empty());
    }

    #[test]
    fn calm_regime_blocks_when_below_floor() {
        let gate = ThresholdGate::new(GateConfig::default());
        let ctx = make_ctx(5, 1.0, 0.1, RegimeLabel::Calm);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(!d.allow_act);
        assert!(!d.allow_write);
    }

    #[test]
    fn cusum_breach_blocks_learning() {
        let gate = ThresholdGate::new(GateConfig::default());
        let ctx = make_ctx(15, 5.0, 0.1, RegimeLabel::Calm);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(!d.allow_learn);
    }

    #[test]
    fn volatile_regime_has_lower_floor() {
        let gate = ThresholdGate::new(GateConfig::default());
        let ctx = make_ctx(7, 1.0, 0.1, RegimeLabel::Volatile);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(d.all_allowed());
    }

    #[test]
    fn drift_blocks_learning() {
        let gate = ThresholdGate::new(GateConfig::default());
        // drift 0.4 > max_drift*0.5 (0.25) so learning blocked,
        // but < max_drift (0.5) so act/write still allowed.
        let ctx = make_ctx(15, 1.0, 0.4, RegimeLabel::Calm);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(!d.allow_learn);
        assert!(d.allow_act); // act is still permitted
    }

    #[test]
    fn high_drift_blocks_everything() {
        let gate = ThresholdGate::new(GateConfig::default());
        let ctx = make_ctx(15, 1.0, 0.8, RegimeLabel::Calm);
        let d = gate.evaluate(&ctx).unwrap();
        assert!(!d.allow_learn);
        assert!(!d.allow_act);
        assert!(!d.allow_write);
        assert!(d.allow_retrieve); // retrieval only needs cut_ok
    }

    #[test]
    fn boundary_instability_blocks_action() {
        let gate = ThresholdGate::new(GateConfig::default());
        let mut ctx = make_ctx(15, 1.0, 0.1, RegimeLabel::Calm);
        ctx.boundary_stable_count = 3; // below threshold of 8
        let d = gate.evaluate(&ctx).unwrap();
        assert!(!d.allow_act);
        assert!(!d.allow_write);
        assert!(d.allow_retrieve); // retrieval permissive
    }
}
