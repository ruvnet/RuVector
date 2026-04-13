//! Proof gate — separates cheap eligibility from privileged commit.
//!
//! Routing hints that carry `requires_proof = true` must be authorized by a
//! [`ProofGate`] implementation before [`crate::scoring::RoutingHint::commit`]
//! will transition them from `issued` to `committed` and emit a
//! [`crate::witness::WitnessEvent::RoutingHintCommitted`] event.
//!
//! The demo ships [`NoopProofGate`] (allow everything) and [`ManualProofGate`]
//! (allowlist by hint id).
//!
//! # Example
//!
//! ```
//! use ruvector_field::proof::{ManualProofGate, ProofGate};
//! use ruvector_field::model::HintId;
//! use ruvector_field::scoring::RoutingHint;
//! let mut gate = ManualProofGate::new();
//! gate.allow(HintId(7));
//! let hint = RoutingHint::demo(HintId(7), true);
//! assert!(gate.authorize(&hint).is_ok());
//! ```

use crate::model::HintId;
use crate::scoring::RoutingHint;

/// Proof authorization token returned by a successful gate call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofToken {
    /// Hint id this token authorizes.
    pub hint: HintId,
    /// Monotonic sequence number within the issuing gate.
    pub sequence: u64,
}

/// Error produced when a proof gate rejects a hint.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ProofError {
    /// Hint did not require a proof but was sent through the gate anyway.
    NotRequired,
    /// Hint is not in the gate's allowlist.
    Denied(&'static str),
}

/// Proof gate trait. Implementations decide which hints may commit.
pub trait ProofGate {
    /// Authorize `hint`. Must return `Err` unless the hint is eligible.
    fn authorize(&mut self, hint: &RoutingHint) -> Result<ProofToken, ProofError>;
}

/// No-op proof gate used in the demo — allows every hint through.
///
/// # Example
///
/// ```
/// use ruvector_field::proof::{NoopProofGate, ProofGate};
/// use ruvector_field::model::HintId;
/// use ruvector_field::scoring::RoutingHint;
/// let mut gate = NoopProofGate::default();
/// assert!(gate.authorize(&RoutingHint::demo(HintId(1), true)).is_ok());
/// ```
#[derive(Debug, Default)]
pub struct NoopProofGate {
    sequence: u64,
}

impl ProofGate for NoopProofGate {
    fn authorize(&mut self, hint: &RoutingHint) -> Result<ProofToken, ProofError> {
        self.sequence += 1;
        Ok(ProofToken {
            hint: hint.id,
            sequence: self.sequence,
        })
    }
}

/// Allowlist proof gate — only hints whose id is explicitly permitted pass.
#[derive(Debug, Default)]
pub struct ManualProofGate {
    allowed: std::collections::HashSet<HintId>,
    sequence: u64,
}

impl ManualProofGate {
    /// Empty allowlist.
    pub fn new() -> Self {
        Self::default()
    }
    /// Allow `hint` to pass future authorize calls.
    pub fn allow(&mut self, hint: HintId) {
        self.allowed.insert(hint);
    }
    /// Remove `hint` from the allowlist.
    pub fn deny(&mut self, hint: HintId) {
        self.allowed.remove(&hint);
    }
}

impl ProofGate for ManualProofGate {
    fn authorize(&mut self, hint: &RoutingHint) -> Result<ProofToken, ProofError> {
        if !hint.requires_proof {
            return Err(ProofError::NotRequired);
        }
        if self.allowed.contains(&hint.id) {
            self.sequence += 1;
            Ok(ProofToken {
                hint: hint.id,
                sequence: self.sequence,
            })
        } else {
            Err(ProofError::Denied("hint not in allowlist"))
        }
    }
}
