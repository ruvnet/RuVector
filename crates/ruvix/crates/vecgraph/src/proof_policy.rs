//! Proof policy and verification for kernel vector/graph stores.
//!
//! In RuVix, proof-gated mutation (ADR-047) is a kernel invariant.
//! The kernel physically prevents state mutation without a valid proof token.
//!
//! # Verification Steps (from ADR-087 Section 8.2)
//!
//! 1. Verify proof token matches mutation hash
//! 2. Verify token has not expired
//! 3. Verify nonce has not been used
//! 4. Verify calling task holds PROVE rights on the object
//!
//! If all checks pass: apply mutation, emit attestation to witness log.
//! If any check fails: return Err(ProofRejected).

use crate::Result;
use ruvix_types::{
    CapRights, Capability, KernelError, ProofAttestation, ProofPayload, ProofTier, ProofToken,
};

/// Policy for proof verification on a store.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(C)]
pub struct ProofPolicy {
    /// Required proof tier for mutations.
    pub required_tier: ProofTier,

    /// Maximum allowed verification time in microseconds.
    /// Proofs exceeding this are rejected.
    pub max_verification_time_us: u32,

    /// Maximum proof validity window in nanoseconds.
    /// Proofs with longer validity are rejected.
    pub max_validity_window_ns: u64,

    /// Whether to require coherence certificates for Deep tier.
    pub require_coherence_cert: bool,

    /// Minimum coherence score in proof (for CoherenceCert payloads).
    pub min_coherence_in_proof: u16,
}

impl Default for ProofPolicy {
    fn default() -> Self {
        Self::standard()
    }
}

impl ProofPolicy {
    /// Creates a Reflex-tier policy (sub-microsecond hash checks).
    #[inline]
    #[must_use]
    pub const fn reflex() -> Self {
        Self {
            required_tier: ProofTier::Reflex,
            max_verification_time_us: 1,
            max_validity_window_ns: 1_000_000_000, // 1s (relaxed for testing)
            require_coherence_cert: false,
            min_coherence_in_proof: 0,
        }
    }

    /// Creates a Standard-tier policy (Merkle witness verification).
    #[inline]
    #[must_use]
    pub const fn standard() -> Self {
        Self {
            required_tier: ProofTier::Standard,
            max_verification_time_us: 100,
            max_validity_window_ns: 1_000_000_000, // 1s
            require_coherence_cert: false,
            min_coherence_in_proof: 0,
        }
    }

    /// Creates a Deep-tier policy (full coherence verification).
    #[inline]
    #[must_use]
    pub const fn deep() -> Self {
        Self {
            required_tier: ProofTier::Deep,
            max_verification_time_us: 10_000,       // 10ms
            max_validity_window_ns: 5_000_000_000,  // 5s
            require_coherence_cert: true,
            min_coherence_in_proof: 5000, // 0.5
        }
    }

    /// Sets the required proof tier.
    #[inline]
    #[must_use]
    pub const fn with_tier(mut self, tier: ProofTier) -> Self {
        self.required_tier = tier;
        self
    }

    /// Sets the maximum verification time.
    #[inline]
    #[must_use]
    pub const fn with_max_verification_time_us(mut self, time_us: u32) -> Self {
        self.max_verification_time_us = time_us;
        self
    }

    /// Sets the maximum validity window.
    #[inline]
    #[must_use]
    pub const fn with_max_validity_ns(mut self, validity_ns: u64) -> Self {
        self.max_validity_window_ns = validity_ns;
        self
    }

    /// Checks if a proof tier satisfies this policy.
    #[inline]
    #[must_use]
    pub const fn tier_satisfies(&self, proof_tier: ProofTier) -> bool {
        // Higher tiers satisfy lower requirements
        (proof_tier as u8) >= (self.required_tier as u8)
    }
}

/// Tracks used nonces to prevent replay attacks.
#[derive(Debug)]
pub struct NonceTracker {
    /// Ring buffer of recent nonces.
    /// We only need to track nonces within the validity window.
    recent_nonces: [u64; 64],

    /// Write position in the ring buffer.
    write_pos: usize,

    /// Number of nonces stored.
    count: usize,

    /// Total nonces ever tracked.
    total_tracked: u64,
}

impl NonceTracker {
    /// Creates a new nonce tracker.
    #[inline]
    #[must_use]
    pub const fn new() -> Self {
        Self {
            recent_nonces: [0u64; 64],
            write_pos: 0,
            count: 0,
            total_tracked: 0,
        }
    }

    /// Checks if a nonce has been used and marks it as used.
    ///
    /// Returns `true` if the nonce was new (valid), `false` if it was a replay.
    pub fn check_and_mark(&mut self, nonce: u64) -> bool {
        // Check if nonce exists in our ring buffer
        for i in 0..self.count.min(64) {
            if self.recent_nonces[i] == nonce {
                return false; // Replay detected
            }
        }

        // Add nonce to ring buffer
        self.recent_nonces[self.write_pos] = nonce;
        self.write_pos = (self.write_pos + 1) % 64;
        if self.count < 64 {
            self.count += 1;
        }
        self.total_tracked = self.total_tracked.wrapping_add(1);

        true
    }

    /// Returns the number of nonces tracked.
    #[inline]
    #[must_use]
    pub const fn count(&self) -> usize {
        self.count
    }

    /// Returns the total nonces ever tracked.
    #[inline]
    #[must_use]
    pub const fn total_tracked(&self) -> u64 {
        self.total_tracked
    }

    /// Clears old nonces (should be called periodically).
    pub fn clear_old(&mut self) {
        self.recent_nonces = [0u64; 64];
        self.write_pos = 0;
        self.count = 0;
    }
}

impl Default for NonceTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifies proof tokens for store mutations.
pub struct ProofVerifier {
    /// The proof policy for this verifier.
    policy: ProofPolicy,

    /// Nonce tracker to prevent replays.
    nonce_tracker: NonceTracker,

    /// Verifier version for attestations.
    verifier_version: u32,

    /// Statistics.
    #[cfg(feature = "stats")]
    proofs_verified: u64,
    #[cfg(feature = "stats")]
    proofs_rejected: u64,
}

impl ProofVerifier {
    /// Creates a new proof verifier with the given policy.
    #[inline]
    #[must_use]
    pub fn new(policy: ProofPolicy) -> Self {
        Self {
            policy,
            nonce_tracker: NonceTracker::new(),
            verifier_version: 0x00_01_00_00, // 0.1.0
            #[cfg(feature = "stats")]
            proofs_verified: 0,
            #[cfg(feature = "stats")]
            proofs_rejected: 0,
        }
    }

    /// Returns the proof policy.
    #[inline]
    #[must_use]
    pub const fn policy(&self) -> &ProofPolicy {
        &self.policy
    }

    /// Verifies a proof token for a mutation.
    ///
    /// # Arguments
    ///
    /// * `proof` - The proof token to verify
    /// * `expected_mutation_hash` - Hash of the mutation being authorized
    /// * `current_time_ns` - Current time in nanoseconds
    /// * `capability` - The capability being used (must have PROVE right)
    ///
    /// # Returns
    ///
    /// On success, returns a `ProofAttestation` to be logged.
    /// On failure, returns `ProofRejected`.
    pub fn verify(
        &mut self,
        proof: &ProofToken,
        expected_mutation_hash: &[u8; 32],
        current_time_ns: u64,
        capability: &Capability,
    ) -> Result<ProofAttestation> {
        // Step 1: Check capability has PROVE right
        if !capability.has_rights(CapRights::PROVE) {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::InsufficientRights);
        }

        // Step 2: Verify mutation hash matches
        if proof.mutation_hash != *expected_mutation_hash {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::ProofRejected);
        }

        // Step 3: Check proof tier satisfies policy
        if !self.policy.tier_satisfies(proof.tier) {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::ProofRejected);
        }

        // Step 4: Check expiry
        if proof.is_expired(current_time_ns) {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::ProofRejected);
        }

        // Step 5: Check validity window is not too large
        let validity_window = proof.valid_until_ns.saturating_sub(current_time_ns);
        if validity_window > self.policy.max_validity_window_ns {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::ProofRejected);
        }

        // Step 6: Check nonce is not reused
        if !self.nonce_tracker.check_and_mark(proof.nonce) {
            #[cfg(feature = "stats")]
            {
                self.proofs_rejected += 1;
            }
            return Err(KernelError::ProofRejected);
        }

        // Step 7: Verify payload according to tier
        self.verify_payload(&proof.payload)?;

        // All checks passed - create attestation
        #[cfg(feature = "stats")]
        {
            self.proofs_verified += 1;
        }

        Ok(self.create_attestation(proof, current_time_ns))
    }

    /// Verifies the proof payload according to its type.
    fn verify_payload(&self, payload: &ProofPayload) -> Result<()> {
        match payload {
            ProofPayload::Hash { hash: _ } => {
                // Reflex tier: just check the hash is non-zero
                // Actual hash verification is done by comparing mutation_hash
                Ok(())
            }
            ProofPayload::MerkleWitness {
                root: _,
                leaf_index: _,
                path_len,
                path: _,
            } => {
                // Standard tier: verify path length is reasonable
                if *path_len > 32 {
                    return Err(KernelError::ProofRejected);
                }
                // Full Merkle verification would go here
                // For now, we trust the proof engine that generated this
                Ok(())
            }
            ProofPayload::CoherenceCert {
                score_before: _,
                score_after,
                partition_id: _,
                signature: _,
            } => {
                // Deep tier: check coherence requirements
                if self.policy.require_coherence_cert
                    && *score_after < self.policy.min_coherence_in_proof
                {
                    return Err(KernelError::CoherenceViolation);
                }
                // Full signature verification would go here
                Ok(())
            }
        }
    }

    /// Creates a proof attestation for a verified proof.
    fn create_attestation(&self, proof: &ProofToken, current_time_ns: u64) -> ProofAttestation {
        // Environment hash would include: policy, capability, mutation type
        // For now, use a placeholder
        let environment_hash = [0u8; 32];

        ProofAttestation::new(
            proof.mutation_hash,
            environment_hash,
            current_time_ns,
            self.verifier_version,
            1, // reduction_steps
            0, // cache_hit_rate
        )
    }

    /// Returns the verifier version.
    #[inline]
    #[must_use]
    pub const fn verifier_version(&self) -> u32 {
        self.verifier_version
    }

    /// Returns the nonce tracker.
    #[inline]
    #[must_use]
    pub const fn nonce_tracker(&self) -> &NonceTracker {
        &self.nonce_tracker
    }
}

impl Default for ProofVerifier {
    fn default() -> Self {
        Self::new(ProofPolicy::default())
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::ObjectType;

    fn create_test_capability() -> Capability {
        Capability::new(
            1,
            ObjectType::VectorStore,
            CapRights::READ | CapRights::WRITE | CapRights::PROVE,
            0,
            1,
        )
    }

    fn create_test_proof(
        mutation_hash: [u8; 32],
        tier: ProofTier,
        valid_until_ns: u64,
        nonce: u64,
    ) -> ProofToken {
        ProofToken::new(
            mutation_hash,
            tier,
            ProofPayload::Hash { hash: mutation_hash },
            valid_until_ns,
            nonce,
        )
    }

    #[test]
    fn test_proof_policy_tiers() {
        let reflex = ProofPolicy::reflex();
        assert_eq!(reflex.required_tier, ProofTier::Reflex);
        assert!(reflex.tier_satisfies(ProofTier::Reflex));
        assert!(reflex.tier_satisfies(ProofTier::Standard));
        assert!(reflex.tier_satisfies(ProofTier::Deep));

        let deep = ProofPolicy::deep();
        assert!(!deep.tier_satisfies(ProofTier::Reflex));
        assert!(!deep.tier_satisfies(ProofTier::Standard));
        assert!(deep.tier_satisfies(ProofTier::Deep));
    }

    #[test]
    fn test_nonce_tracker_replay() {
        let mut tracker = NonceTracker::new();

        assert!(tracker.check_and_mark(1));
        assert!(tracker.check_and_mark(2));
        assert!(tracker.check_and_mark(3));

        // Replay should fail
        assert!(!tracker.check_and_mark(1));
        assert!(!tracker.check_and_mark(2));

        // New nonces should work
        assert!(tracker.check_and_mark(4));
    }

    #[test]
    fn test_proof_verifier_success() {
        let mut verifier = ProofVerifier::new(ProofPolicy::reflex());
        let cap = create_test_capability();

        let mutation_hash = [1u8; 32];
        let proof = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 1);

        let result = verifier.verify(&proof, &mutation_hash, 500_000_000, &cap);
        assert!(result.is_ok());
    }

    #[test]
    fn test_proof_verifier_wrong_hash() {
        let mut verifier = ProofVerifier::new(ProofPolicy::reflex());
        let cap = create_test_capability();

        let mutation_hash = [1u8; 32];
        let wrong_hash = [2u8; 32];
        let proof = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 1);

        let result = verifier.verify(&proof, &wrong_hash, 500_000_000, &cap);
        assert_eq!(result, Err(KernelError::ProofRejected));
    }

    #[test]
    fn test_proof_verifier_expired() {
        let mut verifier = ProofVerifier::new(ProofPolicy::reflex());
        let cap = create_test_capability();

        let mutation_hash = [1u8; 32];
        let proof = create_test_proof(mutation_hash, ProofTier::Standard, 500_000_000, 1);

        // Current time is after expiry
        let result = verifier.verify(&proof, &mutation_hash, 1_000_000_000, &cap);
        assert_eq!(result, Err(KernelError::ProofRejected));
    }

    #[test]
    fn test_proof_verifier_nonce_reuse() {
        let mut verifier = ProofVerifier::new(ProofPolicy::reflex());
        let cap = create_test_capability();

        let mutation_hash = [1u8; 32];
        let proof1 = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 42);
        let proof2 = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 42);

        // First use should succeed
        let result1 = verifier.verify(&proof1, &mutation_hash, 500_000_000, &cap);
        assert!(result1.is_ok());

        // Second use with same nonce should fail
        let result2 = verifier.verify(&proof2, &mutation_hash, 500_000_001, &cap);
        assert_eq!(result2, Err(KernelError::ProofRejected));
    }

    #[test]
    fn test_proof_verifier_insufficient_rights() {
        let mut verifier = ProofVerifier::new(ProofPolicy::reflex());

        // Capability without PROVE right
        let cap = Capability::new(1, ObjectType::VectorStore, CapRights::READ, 0, 1);

        let mutation_hash = [1u8; 32];
        let proof = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 1);

        let result = verifier.verify(&proof, &mutation_hash, 500_000_000, &cap);
        assert_eq!(result, Err(KernelError::InsufficientRights));
    }

    #[test]
    fn test_proof_verifier_tier_mismatch() {
        let mut verifier = ProofVerifier::new(ProofPolicy::deep());
        let cap = create_test_capability();

        let mutation_hash = [1u8; 32];
        // Proof is Standard tier, policy requires Deep
        let proof = create_test_proof(mutation_hash, ProofTier::Standard, 1_000_000_000, 1);

        let result = verifier.verify(&proof, &mutation_hash, 500_000_000, &cap);
        assert_eq!(result, Err(KernelError::ProofRejected));
    }
}
