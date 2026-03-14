//! Proof verification engine for RuVix Cognition Kernel.
//!
//! The ProofEngine verifies proof tokens before allowing mutations.
//! It implements the tiered proof system from ADR-087:
//!
//! - **Reflex**: Sub-microsecond hash check (cached)
//! - **Standard**: Merkle witness verification
//! - **Deep**: Full coherence verification with mincut analysis

use crate::{ProofAttestation, ProofPayload, ProofTier, ProofToken, Result};
use ruvix_types::KernelError;

#[cfg(feature = "alloc")]
extern crate alloc;
#[cfg(feature = "alloc")]
use alloc::collections::BTreeSet;

/// Configuration for the proof engine.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ProofEngineConfig {
    /// Maximum nonces to track (prevents replay).
    pub max_nonces: usize,
    /// Default validity duration for proofs (nanoseconds).
    pub default_validity_ns: u64,
    /// Enable proof caching for Reflex tier.
    pub enable_cache: bool,
    /// Cache size for Reflex proofs.
    pub cache_size: usize,
}

impl ProofEngineConfig {
    /// Creates a new configuration with default values.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            max_nonces: 10_000,
            default_validity_ns: 60_000_000_000, // 60 seconds
            enable_cache: true,
            cache_size: 1000,
        }
    }
}

impl Default for ProofEngineConfig {
    fn default() -> Self {
        Self::new()
    }
}

/// Result of proof verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofVerifyResult {
    /// Proof is valid.
    Valid {
        /// Verification time in nanoseconds.
        verify_time_ns: u64,
        /// Whether the result was cached.
        cached: bool,
    },
    /// Proof is invalid.
    Invalid {
        /// Reason for rejection.
        reason: ProofRejectReason,
    },
}

impl ProofVerifyResult {
    /// Returns true if the proof was valid.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::Valid { .. })
    }
}

/// Reason for proof rejection.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ProofRejectReason {
    /// Proof has expired.
    Expired,
    /// Nonce has been used before (replay attack).
    NonceReused,
    /// Hash mismatch.
    HashMismatch,
    /// Invalid Merkle witness.
    InvalidWitness,
    /// Coherence score too low.
    CoherenceViolation,
    /// Invalid signature.
    InvalidSignature,
}

impl ProofRejectReason {
    /// Returns the reason as a string.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Expired => "Proof expired",
            Self::NonceReused => "Nonce reused (replay attack)",
            Self::HashMismatch => "Hash mismatch",
            Self::InvalidWitness => "Invalid Merkle witness",
            Self::CoherenceViolation => "Coherence constraint violated",
            Self::InvalidSignature => "Invalid signature",
        }
    }
}

/// The proof verification engine.
///
/// Manages proof verification for all tiers and tracks used nonces
/// to prevent replay attacks.
pub struct ProofEngine {
    /// Configuration.
    config: ProofEngineConfig,

    /// Used nonces (for replay prevention).
    #[cfg(feature = "alloc")]
    used_nonces: BTreeSet<u64>,

    /// Used nonces (fixed array for no_std).
    #[cfg(not(feature = "alloc"))]
    used_nonces: [u64; 1024],
    #[cfg(not(feature = "alloc"))]
    nonce_count: usize,

    /// Current time source (nanoseconds since boot).
    current_time_ns: u64,

    /// Statistics.
    stats: ProofEngineStats,
}

/// Statistics about proof engine operations.
#[derive(Debug, Clone, Copy, Default, PartialEq, Eq)]
pub struct ProofEngineStats {
    /// Total proofs verified.
    pub proofs_verified: u64,
    /// Total proofs rejected.
    pub proofs_rejected: u64,
    /// Cache hits (Reflex tier).
    pub cache_hits: u64,
    /// Cache misses.
    pub cache_misses: u64,
    /// Replay attacks detected.
    pub replay_attacks: u64,
}

impl ProofEngine {
    /// Creates a new proof engine with the given configuration.
    #[must_use]
    pub fn new(config: ProofEngineConfig) -> Self {
        Self {
            config,
            #[cfg(feature = "alloc")]
            used_nonces: BTreeSet::new(),
            #[cfg(not(feature = "alloc"))]
            used_nonces: [0u64; 1024],
            #[cfg(not(feature = "alloc"))]
            nonce_count: 0,
            current_time_ns: 0,
            stats: ProofEngineStats::default(),
        }
    }

    /// Creates a proof engine with default configuration.
    #[must_use]
    pub fn with_defaults() -> Self {
        Self::new(ProofEngineConfig::default())
    }

    /// Updates the current time (must be called by the kernel).
    #[inline]
    pub fn set_current_time(&mut self, time_ns: u64) {
        self.current_time_ns = time_ns;
    }

    /// Returns the current configuration.
    #[inline]
    #[must_use]
    pub const fn config(&self) -> &ProofEngineConfig {
        &self.config
    }

    /// Returns the current statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &ProofEngineStats {
        &self.stats
    }

    /// Verifies a proof token.
    ///
    /// Returns `Ok(result)` with verification details, or `Err` on internal error.
    pub fn verify(&mut self, token: &ProofToken, expected_hash: &[u8; 32]) -> Result<ProofVerifyResult> {
        // Check expiry
        if token.is_expired(self.current_time_ns) {
            self.stats.proofs_rejected += 1;
            return Ok(ProofVerifyResult::Invalid {
                reason: ProofRejectReason::Expired,
            });
        }

        // Check nonce reuse
        if self.is_nonce_used(token.nonce) {
            self.stats.proofs_rejected += 1;
            self.stats.replay_attacks += 1;
            return Ok(ProofVerifyResult::Invalid {
                reason: ProofRejectReason::NonceReused,
            });
        }

        // Check mutation hash
        if &token.mutation_hash != expected_hash {
            self.stats.proofs_rejected += 1;
            return Ok(ProofVerifyResult::Invalid {
                reason: ProofRejectReason::HashMismatch,
            });
        }

        // Verify based on tier
        let verify_start = self.current_time_ns;
        let result = match token.tier {
            ProofTier::Reflex => self.verify_reflex(&token.payload),
            ProofTier::Standard => self.verify_standard(&token.payload),
            ProofTier::Deep => self.verify_deep(&token.payload),
        };

        match result {
            Ok(true) => {
                // Mark nonce as used
                self.mark_nonce_used(token.nonce);
                self.stats.proofs_verified += 1;

                let verify_time_ns = self.current_time_ns.saturating_sub(verify_start);
                Ok(ProofVerifyResult::Valid {
                    verify_time_ns,
                    cached: false, // TODO: implement caching
                })
            }
            Ok(false) => {
                self.stats.proofs_rejected += 1;
                Ok(ProofVerifyResult::Invalid {
                    reason: ProofRejectReason::InvalidWitness,
                })
            }
            Err(e) => Err(e),
        }
    }

    /// Generates a proof attestation for a verified proof.
    pub fn generate_attestation(&self, token: &ProofToken) -> ProofAttestation {
        ProofAttestation::new(
            token.mutation_hash,
            [0u8; 32], // Environment hash (would be computed from kernel state)
            self.current_time_ns,
            0x00_01_00_00, // Version 0.1.0
            1,             // Single reduction step for simple proofs
            0,             // No cache hit (would be computed)
        )
    }

    /// Creates a proof token for a mutation.
    ///
    /// This is a convenience method for testing. In production, proofs would
    /// be generated by a trusted proof generator.
    pub fn create_proof(
        &self,
        mutation_hash: [u8; 32],
        tier: ProofTier,
        nonce: u64,
    ) -> ProofToken {
        let payload = match tier {
            ProofTier::Reflex => ProofPayload::Hash { hash: mutation_hash },
            ProofTier::Standard => ProofPayload::MerkleWitness {
                root: mutation_hash,
                leaf_index: 0,
                path_len: 0,
                path: [[0u8; 32]; 32],
            },
            ProofTier::Deep => ProofPayload::CoherenceCert {
                score_before: 10000,
                score_after: 10000,
                partition_id: 0,
                signature: [0u8; 64],
            },
        };

        ProofToken::new(
            mutation_hash,
            tier,
            payload,
            self.current_time_ns + self.config.default_validity_ns,
            nonce,
        )
    }

    // =========================================================================
    // Private Methods
    // =========================================================================

    fn verify_reflex(&self, payload: &ProofPayload) -> Result<bool> {
        match payload {
            ProofPayload::Hash { hash: _ } => {
                // Reflex tier: just check that hash format is valid
                // In production, this would verify against a precomputed cache
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn verify_standard(&self, payload: &ProofPayload) -> Result<bool> {
        match payload {
            ProofPayload::MerkleWitness {
                root: _,
                leaf_index: _,
                path_len,
                path: _,
            } => {
                // Standard tier: verify Merkle witness
                // For now, accept any witness with valid structure
                if *path_len > 32 {
                    return Err(KernelError::InvalidArgument);
                }
                // In production, this would verify the Merkle path
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    fn verify_deep(&self, payload: &ProofPayload) -> Result<bool> {
        match payload {
            ProofPayload::CoherenceCert {
                score_before,
                score_after,
                partition_id: _,
                signature: _,
            } => {
                // Deep tier: verify coherence constraints
                // Coherence should not decrease by more than 10%
                let threshold = (*score_before as u32 * 90) / 100;
                if (*score_after as u32) < threshold {
                    return Ok(false);
                }
                // In production, this would verify the signature
                Ok(true)
            }
            _ => Ok(false),
        }
    }

    #[cfg(feature = "alloc")]
    fn is_nonce_used(&self, nonce: u64) -> bool {
        self.used_nonces.contains(&nonce)
    }

    #[cfg(not(feature = "alloc"))]
    fn is_nonce_used(&self, nonce: u64) -> bool {
        for i in 0..self.nonce_count {
            if self.used_nonces[i] == nonce {
                return true;
            }
        }
        false
    }

    #[cfg(feature = "alloc")]
    fn mark_nonce_used(&mut self, nonce: u64) {
        // Prune old nonces if necessary
        while self.used_nonces.len() >= self.config.max_nonces {
            if let Some(&oldest) = self.used_nonces.iter().next() {
                self.used_nonces.remove(&oldest);
            }
        }
        self.used_nonces.insert(nonce);
    }

    #[cfg(not(feature = "alloc"))]
    fn mark_nonce_used(&mut self, nonce: u64) {
        if self.nonce_count < 1024 {
            self.used_nonces[self.nonce_count] = nonce;
            self.nonce_count += 1;
        } else {
            // Rotate: remove oldest, add new
            for i in 0..1023 {
                self.used_nonces[i] = self.used_nonces[i + 1];
            }
            self.used_nonces[1023] = nonce;
        }
    }
}

impl Default for ProofEngine {
    fn default() -> Self {
        Self::with_defaults()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_proof_engine_creation() {
        let engine = ProofEngine::with_defaults();
        assert_eq!(engine.stats().proofs_verified, 0);
    }

    #[test]
    fn test_verify_valid_proof() {
        let mut engine = ProofEngine::with_defaults();
        engine.set_current_time(1_000_000_000);

        let mutation_hash = [1u8; 32];
        let proof = engine.create_proof(mutation_hash, ProofTier::Reflex, 42);

        let result = engine.verify(&proof, &mutation_hash).unwrap();
        assert!(result.is_valid());
        assert_eq!(engine.stats().proofs_verified, 1);
    }

    #[test]
    fn test_verify_expired_proof() {
        let mut engine = ProofEngine::with_defaults();
        engine.set_current_time(1_000_000_000);

        let mutation_hash = [1u8; 32];
        let proof = engine.create_proof(mutation_hash, ProofTier::Reflex, 42);

        // Fast forward time past expiry
        engine.set_current_time(100_000_000_000);

        let result = engine.verify(&proof, &mutation_hash).unwrap();
        assert!(!result.is_valid());
        assert!(matches!(
            result,
            ProofVerifyResult::Invalid {
                reason: ProofRejectReason::Expired
            }
        ));
    }

    #[test]
    fn test_verify_nonce_reuse() {
        let mut engine = ProofEngine::with_defaults();
        engine.set_current_time(1_000_000_000);

        let mutation_hash = [1u8; 32];
        let proof = engine.create_proof(mutation_hash, ProofTier::Reflex, 42);

        // First verification succeeds
        let result = engine.verify(&proof, &mutation_hash).unwrap();
        assert!(result.is_valid());

        // Second verification with same nonce fails
        let result = engine.verify(&proof, &mutation_hash).unwrap();
        assert!(!result.is_valid());
        assert!(matches!(
            result,
            ProofVerifyResult::Invalid {
                reason: ProofRejectReason::NonceReused
            }
        ));
        assert_eq!(engine.stats().replay_attacks, 1);
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let mut engine = ProofEngine::with_defaults();
        engine.set_current_time(1_000_000_000);

        let mutation_hash = [1u8; 32];
        let wrong_hash = [2u8; 32];
        let proof = engine.create_proof(mutation_hash, ProofTier::Reflex, 42);

        let result = engine.verify(&proof, &wrong_hash).unwrap();
        assert!(!result.is_valid());
        assert!(matches!(
            result,
            ProofVerifyResult::Invalid {
                reason: ProofRejectReason::HashMismatch
            }
        ));
    }

    #[test]
    fn test_all_proof_tiers() {
        let mut engine = ProofEngine::with_defaults();
        engine.set_current_time(1_000_000_000);

        let mutation_hash = [1u8; 32];

        for (tier, nonce) in [
            (ProofTier::Reflex, 1),
            (ProofTier::Standard, 2),
            (ProofTier::Deep, 3),
        ] {
            let proof = engine.create_proof(mutation_hash, tier, nonce);
            let result = engine.verify(&proof, &mutation_hash).unwrap();
            assert!(result.is_valid(), "Tier {:?} should verify", tier);
        }

        assert_eq!(engine.stats().proofs_verified, 3);
    }

    #[test]
    fn test_generate_attestation() {
        let engine = ProofEngine::with_defaults();
        let proof = ProofToken::default();
        let attestation = engine.generate_attestation(&proof);

        assert_eq!(attestation.verifier_version, 0x00_01_00_00);
    }
}
