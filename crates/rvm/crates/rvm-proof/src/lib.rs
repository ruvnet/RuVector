//! # RVM Proof Engine
//!
//! Proof-gated state transitions for the RVM microhypervisor, as
//! specified in ADR-135. Every mutation to partition state requires a
//! valid proof that is recorded in the witness trail.
//!
//! ## Proof Tiers
//!
//! | Tier | Verification | Cost | Use Case |
//! |------|-------------|------|----------|
//! | `Hash` | SHA-256 preimage | O(1) | Routine transitions |
//! | `Witness` | Witness chain verification | O(n) | Cross-partition ops |
//! | `Zk` | Zero-knowledge proof | Expensive | Privacy-preserving |
//!
//! ## Modules
//!
//! - [`context`]: Proof context with builder pattern for P2 validation
//! - [`engine`]: Unified proof engine (P1 -> P2 -> witness pipeline)
//! - [`policy`]: P2 policy rules with constant-time evaluation
//! - [`constant_time`]: Constant-time comparison utilities
//! - [`signer`]: Witness signing traits and implementations (ADR-142)
//! - [`tee`]: TEE attestation trait definitions (ADR-142)
//! - [`tee_provider`]: Software TEE quote provider (ADR-142 Phase 3)
//! - [`tee_verifier`]: Software TEE quote verifier (ADR-142 Phase 3)
//! - [`tee_signer`]: TEE-backed witness signer pipeline (ADR-142 Phase 3)

#![no_std]
#![forbid(unsafe_code)]
#![deny(missing_docs)]
#![deny(clippy::all)]
#![warn(clippy::pedantic)]

#[cfg(feature = "alloc")]
extern crate alloc;

#[cfg(feature = "std")]
extern crate std;

pub mod constant_time;
pub mod context;
pub mod engine;
pub mod policy;
pub mod signer;
pub mod tee;
pub mod tee_provider;
pub mod tee_verifier;
pub mod tee_signer;

// Re-export signer traits and types for ergonomic access.
pub use signer::{SealSignerAdapter, SignatureError, WitnessSigner};
#[cfg(feature = "crypto-sha256")]
pub use signer::HmacSha256WitnessSigner;
#[cfg(feature = "crypto-sha256")]
pub use signer::DualHmacSigner;
#[cfg(feature = "crypto-sha256")]
pub use signer::{KeyBundle, derive_witness_key, derive_key_bundle, dev_measurement};
#[cfg(feature = "ed25519")]
pub use signer::Ed25519WitnessSigner;
#[cfg(any(test, feature = "null-signer"))]
pub use signer::NullSigner;
pub use tee::{TeePlatform, TeeQuoteProvider, TeeQuoteVerifier};
#[cfg(feature = "crypto-sha256")]
pub use tee_provider::SoftwareTeeProvider;
#[cfg(feature = "crypto-sha256")]
pub use tee_verifier::SoftwareTeeVerifier;
#[cfg(feature = "crypto-sha256")]
pub use tee_signer::TeeWitnessSigner;

use rvm_types::{CapRights, CapToken, RvmError, RvmResult, WitnessHash, fnv1a_64};

/// The tier of proof required for a state transition.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum ProofTier {
    /// SHA-256 preimage proof (cheapest).
    Hash = 0,
    /// Witness chain verification.
    Witness = 1,
    /// Zero-knowledge proof (most expensive).
    Zk = 2,
}

/// A proof payload submitted with a state-transition request.
#[derive(Debug, Clone, Copy)]
pub struct Proof {
    /// The tier of this proof.
    pub tier: ProofTier,
    /// The hash commitment this proof satisfies.
    pub commitment: WitnessHash,
    /// Raw proof bytes (truncated to a fixed maximum for `no_std`).
    data: [u8; 64],
    /// Length of valid data in the `data` buffer.
    data_len: u8,
}

impl Proof {
    /// Create a hash-tier proof from a preimage.
    ///
    /// The preimage is truncated to 64 bytes if longer.
    #[must_use]
    pub fn hash_proof(commitment: WitnessHash, preimage: &[u8]) -> Self {
        let mut data = [0u8; 64];
        let len = preimage.len().min(64);
        data[..len].copy_from_slice(&preimage[..len]);
        Self {
            tier: ProofTier::Hash,
            commitment,
            data,
            // Safe: len is clamped to 64 above, which fits in u8.
            #[allow(clippy::cast_possible_truncation)]
            data_len: len as u8,
        }
    }

    /// Return the proof data as a byte slice.
    #[must_use]
    pub fn data(&self) -> &[u8] {
        &self.data[..self.data_len as usize]
    }
}

/// Compute the FNV-1a hash of proof data and pack it into a 32-byte
/// `WitnessHash`.
///
/// The 8-byte FNV-1a digest is placed in the first 8 bytes (little-endian),
/// with the remaining 24 bytes zeroed. This matches how `Proof::hash_proof`
/// commitments are constructed.
#[must_use]
pub fn compute_data_hash(data: &[u8]) -> WitnessHash {
    let digest = fnv1a_64(data);
    let mut bytes = [0u8; 32];
    bytes[..8].copy_from_slice(&digest.to_le_bytes());
    WitnessHash::from_bytes(bytes)
}

/// Verify that a proof is valid for the given commitment.
///
/// Dispatches to tier-specific verifiers:
/// - **Hash**: Computes FNV-1a over the proof data and compares to the commitment.
/// - **Witness**: Validates that proof data contains a valid witness chain
///   with correct `prev_hash` linkage.
/// - **Zk**: Not yet implemented; requires TEE support (see ADR for TEE integration).
///
/// # Errors
///
/// Returns [`RvmError::ProofInvalid`] if the commitment does not match,
/// the proof data is empty, or tier-specific verification fails.
/// Returns [`RvmError::Unsupported`] for ZK proofs (TEE required).
pub fn verify(proof: &Proof, expected_commitment: &WitnessHash) -> RvmResult<()> {
    if proof.commitment != *expected_commitment {
        return Err(RvmError::ProofInvalid);
    }

    match proof.tier {
        ProofTier::Hash => {
            if proof.data_len == 0 {
                return Err(RvmError::ProofInvalid);
            }
            // Hash the proof data and compare to the commitment.
            let computed = compute_data_hash(&proof.data[..proof.data_len as usize]);
            if computed != proof.commitment {
                return Err(RvmError::ProofInvalid);
            }
            Ok(())
        }
        ProofTier::Witness => {
            // Witness chain verification: the proof data must contain at
            // least one 16-byte witness record pair (prev_hash: u64,
            // record_hash: u64) and each record's prev_hash must equal
            // the preceding record's record_hash.
            if proof.data_len == 0 {
                return Err(RvmError::ProofInvalid);
            }
            let data = &proof.data[..proof.data_len as usize];
            // Each link is 16 bytes: 8 bytes prev_hash + 8 bytes record_hash.
            const LINK_SIZE: usize = 16;
            if data.len() < LINK_SIZE {
                return Err(RvmError::ProofInvalid);
            }
            let link_count = data.len() / LINK_SIZE;
            if link_count == 0 {
                return Err(RvmError::ProofInvalid);
            }
            // Walk the chain: for each consecutive pair of links, verify
            // that link[i].record_hash == link[i+1].prev_hash.
            for i in 0..link_count.saturating_sub(1) {
                let offset = i * LINK_SIZE;
                let record_hash = u64::from_le_bytes([
                    data[offset + 8],
                    data[offset + 9],
                    data[offset + 10],
                    data[offset + 11],
                    data[offset + 12],
                    data[offset + 13],
                    data[offset + 14],
                    data[offset + 15],
                ]);
                let next_offset = (i + 1) * LINK_SIZE;
                let next_prev_hash = u64::from_le_bytes([
                    data[next_offset],
                    data[next_offset + 1],
                    data[next_offset + 2],
                    data[next_offset + 3],
                    data[next_offset + 4],
                    data[next_offset + 5],
                    data[next_offset + 6],
                    data[next_offset + 7],
                ]);
                if record_hash != next_prev_hash {
                    return Err(RvmError::ProofInvalid);
                }
            }
            Ok(())
        }
        ProofTier::Zk => {
            // ZK proof verification requires TEE support which is not
            // yet available. Silently accepting would be a security hole.
            Err(RvmError::Unsupported)
        }
    }
}

/// Check that a capability token authorizes proof submission, then verify.
///
/// # Errors
///
/// Returns [`RvmError::InsufficientCapability`] if the token lacks `PROVE` rights.
/// Returns [`RvmError::ProofInvalid`] if the proof verification fails.
pub fn verify_with_cap(
    proof: &Proof,
    expected_commitment: &WitnessHash,
    token: &CapToken,
) -> RvmResult<()> {
    if !token.has_rights(CapRights::PROVE) {
        return Err(RvmError::InsufficientCapability);
    }
    verify(proof, expected_commitment)
}
