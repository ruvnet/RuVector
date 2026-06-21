//! # RVM Security Policy
//!
//! Security policy enforcement for the RVM microhypervisor. This crate
//! provides the policy decision point that combines capability checks,
//! proof verification, and witness logging into a unified security gate.
//!
//! ## Security Model
//!
//! Every hypercall passes through a three-stage gate:
//!
//! 1. **Capability check** -- Does the caller hold the required rights?
//! 2. **Proof verification** -- Is the state transition properly attested?
//! 3. **Witness logging** -- Record the decision for future audit
//!
//! Only after all three stages pass does the hypercall proceed.
//!
//! ## Modules
//!
//! - [`gate`] -- Unified security gate (single entry point)
//! - [`validation`] -- Input validation for security-critical parameters
//! - [`attestation`] -- Attestation chain and report generation
//! - [`budget`] -- DMA and resource budget enforcement

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod attestation;
pub mod budget;
pub mod gate;
pub mod validation;

use rvm_types::{CapRights, CapToken, CapType, RvmError, RvmResult, WitnessHash};

// Re-export key types for convenience.
pub use attestation::{AttestationChain, AttestationReport, verify_attestation};
pub use budget::{DmaBudget, ResourceQuota};
pub use gate::{
    GateRequest, GateResponse, P3WitnessChain, SecurityError, SecurityGate, SecurityGateV2,
    SignedSecurityGate,
};

/// The result of a security policy decision.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum PolicyDecision {
    /// The operation is allowed.
    Allow,
    /// The operation is denied with a reason.
    Deny(RvmError),
}

/// A lightweight policy request for quick evaluate/enforce checks.
///
/// For the full unified security gate with witness logging, use
/// [`gate::GateRequest`] with [`SecurityGate::check_and_execute`].
#[derive(Debug, Clone, Copy)]
pub struct PolicyRequest<'a> {
    /// The capability token presented by the caller.
    pub token: &'a CapToken,
    /// The required capability type.
    pub required_type: CapType,
    /// The required access rights.
    pub required_rights: CapRights,
    /// Optional proof commitment (required for state-mutating operations).
    pub proof_commitment: Option<&'a WitnessHash>,
    /// Optional current epoch for staleness check.
    ///
    /// If provided, the token's epoch must match. This catches stale
    /// capabilities that survived revocation by epoch rotation.
    pub current_epoch: Option<u32>,
}

/// Evaluate a policy request against the security policy.
///
/// Returns `Allow` only if the capability check and (optional) proof
/// commitment are satisfied. The caller is responsible for witness
/// logging after the decision.
///
/// For the full gate pipeline with automatic witness logging, use
/// [`SecurityGate::check_and_execute`] instead.
#[must_use]
pub fn evaluate(request: &PolicyRequest<'_>) -> PolicyDecision {
    // Stage 0: Epoch freshness check (if requested).
    if let Some(epoch) = request.current_epoch {
        if request.token.epoch() != epoch {
            return PolicyDecision::Deny(RvmError::InsufficientCapability);
        }
    }

    // Stage 1: Capability type check.
    if request.token.cap_type() != request.required_type {
        return PolicyDecision::Deny(RvmError::CapabilityTypeMismatch);
    }

    // Stage 2: Rights check.
    if !request.token.has_rights(request.required_rights) {
        return PolicyDecision::Deny(RvmError::InsufficientCapability);
    }

    // Stage 3: Proof commitment presence check (if required).
    // Actual proof verification is handled by rvm-proof; here we
    // only ensure the commitment was provided.
    if let Some(commitment) = request.proof_commitment {
        if commitment.is_zero() {
            return PolicyDecision::Deny(RvmError::ProofInvalid);
        }
    }

    PolicyDecision::Allow
}

/// Convenience function: evaluate and return `RvmResult<()>`.
///
/// # Errors
///
/// Returns the [`RvmError`] from the policy decision if the operation is denied.
pub fn enforce(request: &PolicyRequest<'_>) -> RvmResult<()> {
    match evaluate(request) {
        PolicyDecision::Allow => Ok(()),
        PolicyDecision::Deny(e) => Err(e),
    }
}
