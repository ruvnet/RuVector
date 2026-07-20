//! The fail-closed guard (the paper's *Ward*).
//!
//! Calyx's third trust principle is *fail closed*: a missing grounded path, an
//! unknown lens, a shape mismatch, or an uncalibrated guard returns a
//! structured refusal rather than a silent wrong answer. This module turns a
//! fused candidate plus its cross-lens agreement and grounding into an explicit
//! [`GuardDecision`].

use crate::ledger::{AnchorKind, GroundingPath};

/// Thresholds the guard enforces. Tunable by the anneal optimizer.
#[derive(Debug, Clone)]
pub struct GuardProfile {
    /// Minimum fused (RRF) score the top candidate must reach.
    pub min_fusion_score: f32,
    /// Minimum cross-lens *min*-agreement: every lens must corroborate.
    pub min_cross_lens_agreement: f32,
    /// Minimum anchor weight required for a grounded path.
    pub min_grounding_weight: f32,
    /// Whether a grounded path is mandatory (grounding is mandatory).
    pub require_grounding: bool,
}

impl Default for GuardProfile {
    fn default() -> Self {
        Self {
            min_fusion_score: 0.0,
            min_cross_lens_agreement: 0.45,
            min_grounding_weight: 0.5,
            require_grounding: true,
        }
    }
}

/// Why the guard refused — structured, never silent.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum RefusalReason {
    /// No candidate was retrieved at all.
    NoCandidate,
    /// The fused score was below the calibrated floor.
    LowFusionScore,
    /// At least one lens dissented (min-agreement below floor).
    CrossLensDisagreement,
    /// No grounded evidence path crossed the threshold.
    NoGroundedPath,
    /// The guard profile itself was not calibrated (degenerate thresholds).
    UncalibratedGuard,
}

/// The guard's verdict.
#[derive(Debug, Clone, PartialEq)]
pub enum GuardDecision {
    /// Answer with this record, carrying the evidence that justified it.
    Answer {
        record_id: u64,
        fusion_score: f32,
        cross_lens_agreement: f32,
        grounding_kind: Option<AnchorKind>,
        grounding_weight: f32,
    },
    /// Refuse, with a structured reason.
    Refuse(RefusalReason),
}

impl GuardDecision {
    pub fn is_answer(&self) -> bool {
        matches!(self, GuardDecision::Answer { .. })
    }

    pub fn answered_id(&self) -> Option<u64> {
        match self {
            GuardDecision::Answer { record_id, .. } => Some(*record_id),
            GuardDecision::Refuse(_) => None,
        }
    }
}

/// Evidence the guard weighs for the top fused candidate.
pub struct Candidate {
    pub record_id: u64,
    pub fusion_score: f32,
    pub cross_lens_agreement: f32,
    pub grounding: Option<GroundingPath>,
}

/// Apply the guard. Fails closed on every shortfall.
pub fn adjudicate(candidate: Option<Candidate>, profile: &GuardProfile) -> GuardDecision {
    // A guard with impossible thresholds is treated as uncalibrated rather than
    // silently passing everything.
    if !(0.0..=1.0).contains(&profile.min_cross_lens_agreement)
        || !(0.0..=1.0).contains(&profile.min_grounding_weight)
    {
        return GuardDecision::Refuse(RefusalReason::UncalibratedGuard);
    }

    let c = match candidate {
        Some(c) => c,
        None => return GuardDecision::Refuse(RefusalReason::NoCandidate),
    };

    if c.fusion_score < profile.min_fusion_score {
        return GuardDecision::Refuse(RefusalReason::LowFusionScore);
    }
    if c.cross_lens_agreement < profile.min_cross_lens_agreement {
        return GuardDecision::Refuse(RefusalReason::CrossLensDisagreement);
    }

    let grounding = match (&c.grounding, profile.require_grounding) {
        (None, true) => return GuardDecision::Refuse(RefusalReason::NoGroundedPath),
        (g, _) => g.clone(),
    };
    if let Some(g) = &grounding {
        if g.best_weight < profile.min_grounding_weight {
            return GuardDecision::Refuse(RefusalReason::NoGroundedPath);
        }
    }

    GuardDecision::Answer {
        record_id: c.record_id,
        fusion_score: c.fusion_score,
        cross_lens_agreement: c.cross_lens_agreement,
        grounding_kind: grounding.as_ref().map(|g| g.best_kind),
        grounding_weight: grounding.map(|g| g.best_weight).unwrap_or(0.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::ledger::AnchorKind;

    fn grounded(w: f32) -> Option<GroundingPath> {
        Some(GroundingPath {
            record_id: 1,
            best_kind: AnchorKind::PassedTest,
            best_weight: w,
        })
    }

    #[test]
    fn answers_when_all_gates_pass() {
        let d = adjudicate(
            Some(Candidate {
                record_id: 1,
                fusion_score: 0.5,
                cross_lens_agreement: 0.8,
                grounding: grounded(0.9),
            }),
            &GuardProfile::default(),
        );
        assert_eq!(d.answered_id(), Some(1));
    }

    #[test]
    fn refuses_on_disagreement_then_on_missing_grounding() {
        let p = GuardProfile::default();
        let d = adjudicate(
            Some(Candidate {
                record_id: 1,
                fusion_score: 0.5,
                cross_lens_agreement: 0.1,
                grounding: grounded(0.9),
            }),
            &p,
        );
        assert_eq!(d, GuardDecision::Refuse(RefusalReason::CrossLensDisagreement));

        let d2 = adjudicate(
            Some(Candidate {
                record_id: 1,
                fusion_score: 0.5,
                cross_lens_agreement: 0.8,
                grounding: None,
            }),
            &p,
        );
        assert_eq!(d2, GuardDecision::Refuse(RefusalReason::NoGroundedPath));
    }

    #[test]
    fn uncalibrated_guard_fails_closed() {
        let p = GuardProfile {
            min_cross_lens_agreement: 5.0,
            ..GuardProfile::default()
        };
        let d = adjudicate(None, &p);
        assert_eq!(d, GuardDecision::Refuse(RefusalReason::UncalibratedGuard));
    }
}
