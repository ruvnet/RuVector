//! Causal-attribution gate for incoming latent messages (ADR-326 §4 security
//! gate; ADR-310 controlled-replacement causal audit).
//!
//! **A received latent message must pass causal attribution before it is
//! incorporated into the thought graph — it is not trusted merely because it
//! arrived.** This is the security property WP23 must hold: attributability of
//! every incorporated message (the WP6 concern flagged on RuVector#882).
//!
//! ## Controlled-replacement causal audit
//!
//! A message claims "I follow causally from antecedent step `A`." ADR-310's
//! audit does not take that on faith. The gate applies two conditions:
//!
//! 1. **Attribution** — the message's latent contribution must actually be
//!    explained by `A`'s latent signature (cosine ≥ `accept_threshold`). A
//!    forged message whose latent is unrelated to its claimed cause fails here.
//! 2. **Specificity (the controlled replacement)** — `A` must be *specifically*
//!    load-bearing. We substitute a **control** antecedent (the mean signature of
//!    the other steps) and re-measure attribution. If the control explains the
//!    message just as well, `A` is not the genuine cause — the attribution is
//!    spurious — and the message fails.
//!
//! Only a message that is both attributable to its claimed cause *and*
//! specifically so is [`CausalVerdict::Accept`]ed. Everything else is rejected
//! and never reaches the thought graph.

use serde::{Deserialize, Serialize};

use crate::thoughtmap::{StepId, ThoughtMap};
use crate::transport::LatentMessage;
use crate::vecmath::{cosine, mean};

/// Outcome of the causal-attribution audit.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub enum CausalVerdict {
    /// Message is attributable to its claimed antecedent and that antecedent is
    /// specifically load-bearing. Safe to incorporate.
    Accept,
    /// The claimed antecedent step does not exist in the thought map.
    Reject(RejectReason),
}

/// Why a message failed the causal gate.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum RejectReason {
    /// The claimed antecedent step is not in the thought map.
    MissingAntecedent,
    /// The message's latent is not explained by its claimed cause.
    Unattributable,
    /// A control antecedent explains the message just as well — the attribution
    /// to the claimed cause is spurious.
    Spurious,
}

impl CausalVerdict {
    pub fn is_accepted(&self) -> bool {
        matches!(self, CausalVerdict::Accept)
    }
}

/// The ADR-310 gate. A production audit (WP6) can implement this trait
/// differently; the receiving agent only ever calls [`CausalVerifier::verify`].
pub trait CausalVerifier {
    fn verify(&self, map: &ThoughtMap, msg: &LatentMessage) -> CausalVerdict;
}

/// Default controlled-replacement verifier.
#[derive(Debug, Clone, Copy)]
pub struct ControlledReplacementVerifier {
    /// Minimum attribution (cosine) of message → claimed antecedent.
    pub accept_threshold: f32,
    /// Minimum margin by which the claimed antecedent must beat the control.
    pub specificity_margin: f32,
}

impl Default for ControlledReplacementVerifier {
    fn default() -> Self {
        Self {
            accept_threshold: 0.6,
            specificity_margin: 0.1,
        }
    }
}

impl ControlledReplacementVerifier {
    pub fn new(accept_threshold: f32, specificity_margin: f32) -> Self {
        Self {
            accept_threshold,
            specificity_margin,
        }
    }
}

impl CausalVerifier for ControlledReplacementVerifier {
    fn verify(&self, map: &ThoughtMap, msg: &LatentMessage) -> CausalVerdict {
        let antecedent: StepId = msg.causal.antecedent;

        // Condition 0: the claimed cause must exist. A forged reference to a
        // non-existent step is rejected outright.
        let Some(node) = map.node(antecedent) else {
            return CausalVerdict::Reject(RejectReason::MissingAntecedent);
        };

        // Condition 1: attribution — is the message explained by its claimed cause?
        let attribution = cosine(&msg.latent, &node.latent);
        if attribution < self.accept_threshold {
            return CausalVerdict::Reject(RejectReason::Unattributable);
        }

        // Condition 2: controlled replacement — substitute a control antecedent
        // (mean of the *other* steps) and re-measure. The claimed cause must be
        // specifically responsible, not generically so.
        let controls = map.control_signatures(antecedent);
        if !controls.is_empty() {
            let control = mean(&controls);
            let control_attribution = cosine(&msg.latent, &control);
            if attribution - control_attribution < self.specificity_margin {
                return CausalVerdict::Reject(RejectReason::Spurious);
            }
        }

        CausalVerdict::Accept
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::capability::PeerId;
    use crate::thoughtmap::ThoughtGraphBackend;
    use crate::transport::{integrity_tag, CausalTag, LatentMessage};

    fn msg(from: u64, latent: Vec<f32>, antecedent: StepId) -> LatentMessage {
        LatentMessage {
            from: PeerId(from),
            to: PeerId(0),
            integrity: integrity_tag(PeerId(from), PeerId(0), &latent, antecedent),
            causal: CausalTag { antecedent },
            latent,
            summary: String::new(),
        }
    }

    #[test]
    fn genuine_message_is_accepted() {
        let mut map = ThoughtMap::new();
        // A "topic A" antecedent and an unrelated "topic B" node so the control
        // population is meaningfully different.
        let a = map.add_step(PeerId(1), vec![1.0, 0.0, 0.0], "A");
        let _b = map.add_step(PeerId(2), vec![0.0, 1.0, 0.0], "B");

        // A message genuinely derived from A (close to A, far from the control).
        let m = msg(2, vec![0.98, 0.02, 0.0], a);
        let v = ControlledReplacementVerifier::default();
        assert_eq!(v.verify(&map, &m), CausalVerdict::Accept);
    }

    #[test]
    fn forged_message_unrelated_to_cause_is_rejected() {
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0, 0.0, 0.0], "A");
        let _b = map.add_step(PeerId(2), vec![0.0, 1.0, 0.0], "B");

        // Latent orthogonal to the claimed antecedent A → not attributable.
        let m = msg(9, vec![0.0, 0.0, 1.0], a);
        let v = ControlledReplacementVerifier::default();
        assert_eq!(
            v.verify(&map, &m),
            CausalVerdict::Reject(RejectReason::Unattributable)
        );
    }

    #[test]
    fn nan_latent_never_accepted_at_the_gate() {
        // Defense-in-depth: even if a non-finite latent reached the gate directly
        // (bypassing the incorporate choke point), guarded cosine returns 0.0 (not
        // NaN), so attribution < threshold and the message is Rejected — never the
        // silent NaN→Accept fall-through.
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0, 0.0, 0.0], "A");
        let _b = map.add_step(PeerId(2), vec![0.0, 1.0, 0.0], "B");
        let v = ControlledReplacementVerifier::default();

        let nan = msg(9, vec![f32::NAN, 0.0, 0.0], a);
        assert!(
            !v.verify(&map, &nan).is_accepted(),
            "NaN must not be accepted"
        );

        let inf = msg(9, vec![f32::INFINITY, 0.0, 0.0], a);
        assert!(
            !v.verify(&map, &inf).is_accepted(),
            "inf must not be accepted"
        );
    }

    #[test]
    fn missing_antecedent_is_rejected() {
        let map = ThoughtMap::new();
        let m = msg(9, vec![1.0, 0.0, 0.0], StepId(42));
        let v = ControlledReplacementVerifier::default();
        assert_eq!(
            v.verify(&map, &m),
            CausalVerdict::Reject(RejectReason::MissingAntecedent)
        );
    }

    #[test]
    fn spurious_attribution_to_a_generic_cause_is_rejected() {
        // When the message latent looks like the *average* of the map (explained
        // equally by any node), attribution to the specific claimed cause is
        // spurious and the controlled replacement catches it.
        let mut map = ThoughtMap::new();
        let a = map.add_step(PeerId(1), vec![1.0, 1.0, 1.0], "A");
        let _b = map.add_step(PeerId(2), vec![1.0, 1.0, 1.0], "B");
        let _c = map.add_step(PeerId(3), vec![1.0, 1.0, 1.0], "C");
        // Message aligned with the shared direction — every node explains it.
        let m = msg(2, vec![1.0, 1.0, 1.0], a);
        let v = ControlledReplacementVerifier::default();
        assert_eq!(
            v.verify(&map, &m),
            CausalVerdict::Reject(RejectReason::Spurious)
        );
    }
}
