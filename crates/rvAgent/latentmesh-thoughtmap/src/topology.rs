//! Dead-end-triggered topology adaptation (ADR-326 Decision §6).
//!
//! # ⚠️ UNVERIFIED DESIGN CHOICE — NOT A REPRODUCED DeAR CLAIM
//!
//! DeAR (arXiv:2608.17282) grades **B+, not A**, precisely because of the
//! behavior in this module. The abstract's confirmed text on dead-end handling
//! is **four words** — *"topology update for adaptive error correction"*. It
//! does **not** state a continue-vs-restart mechanism.
//!
//! The behavior implemented here — on a dead end, **mutate topology and CONTINUE
//! from the current reasoning state rather than restarting** — is
//! **this program's own design decision** (ADR-326 §6), *informed by but not
//! confirmed against* the paper. It MUST NOT be presented as reproduced from
//! DeAR, and this WP MUST NOT harden acceptance criteria around it as a fixed
//! behavioral contract.
//!
//! **Re-verify against the full paper (unavailable today) or the eventual code
//! release before locking this in.** If the full text contradicts the
//! continue-not-restart reading, [`DeadEndPolicy::default`] and ADR-326 §6 must
//! be revised. The behavior is kept **swappable** via [`DeadEndPolicy`] for
//! exactly this reason: flipping to [`DeadEndPolicy::Restart`] is a one-line
//! change with no other code depending on the choice.
//!
//! (Per ADR-326's binding "design-choice labeling discipline" gate: a missing
//! label on this behavior is a review-blocking defect. Hence the loud markers.)

use serde::{Deserialize, Serialize};

use crate::capability::PeerId;
use crate::thoughtmap::StepId;

/// How the reasoning fabric reacts when a reasoning path hits a dead end.
///
/// **The variants encode the unverified continue-vs-restart choice explicitly so
/// it stays a *choice*, not a buried assumption.**
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DeadEndPolicy {
    /// **This program's design choice (UNVERIFIED, ADR-326 §6):** change topology
    /// and continue from the *current* state — re-route to a different peer via
    /// updated capability vectors rather than discarding progress. Not attributed
    /// to DeAR.
    ContinueWithTopologyChange,
    /// Swappable alternative: abandon the current reasoning state and restart
    /// from the root. Provided so the continue-vs-restart decision can be flipped
    /// if the full paper / released code contradicts the working interpretation.
    Restart,
}

impl Default for DeadEndPolicy {
    /// Defaults to the program's working interpretation. **This default is the
    /// unverified design choice, not a paper claim** — see module docs.
    fn default() -> Self {
        DeadEndPolicy::ContinueWithTopologyChange
    }
}

/// A single topology mutation. **Every mutation is attributable** (ADR-326 §
/// acceptance: topology changes trace to a cause) — `cause` names why it
/// happened, and `triggered_by` names the peer/agent whose action prompted it,
/// so the quarantine layer can refuse churn proposed by a quarantined peer.
#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct TopologyChange {
    /// Where reasoning resumes from after the change.
    pub resume_from: StepId,
    /// The newly selected collaboration peer (if the change re-routed).
    pub new_peer: Option<PeerId>,
    /// Machine-readable cause — makes the change auditable end-to-end.
    pub cause: TopologyCause,
    /// The peer/agent whose action prompted the change (attribution).
    pub triggered_by: PeerId,
    /// Whether progress was preserved (continue) or discarded (restart).
    pub continued: bool,
}

/// Why a topology change happened. Auditable, witness-loggable cause.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum TopologyCause {
    /// A reasoning path hit a dead end and the [`DeadEndPolicy`] fired.
    DeadEnd,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn default_policy_is_the_unverified_continue_choice() {
        assert_eq!(
            DeadEndPolicy::default(),
            DeadEndPolicy::ContinueWithTopologyChange
        );
    }

    #[test]
    fn policy_is_swappable() {
        // The whole point of the enum: flipping the design choice is trivial.
        let p = DeadEndPolicy::Restart;
        assert_ne!(p, DeadEndPolicy::default());
    }
}
