//! Proof verification with security checks.
//!
//! The verifier ensures:
//! - Proof token matches mutation hash
//! - Token has not expired
//! - Nonce has not been consumed
//! - Caller holds PROVE rights on the object

use crate::cache::ProofCache;
use crate::error::{ProofError, ProofResult};
use ruvix_cap::{CapRights, Capability};
use ruvix_types::{ProofPayload, ProofTier, ProofToken};

/// Configuration for the proof verifier.
#[derive(Debug, Clone, Copy)]
pub struct VerifierConfig {
    /// Whether to check nonce consumption.
    pub check_nonce: bool,
    /// Whether to verify Merkle witnesses.
    pub verify_merkle: bool,
    /// Whether to verify coherence certificates.
    pub verify_coherence: bool,
    /// Minimum coherence score (0-10000 = 0%-100%).
    pub min_coherence_score: u16,
}

impl Default for VerifierConfig {
    fn default() -> Self {
        Self {
            check_nonce: true,
            verify_merkle: true,
            verify_coherence: true,
            min_coherence_score: 5000, // 50% minimum coherence
        }
    }
}

/// Result of proof verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct VerificationResult {
    /// The verified proof tier.
    pub tier: ProofTier,
    /// Whether the proof was cached.
    pub was_cached: bool,
    /// Nonce that was consumed.
    pub nonce: u64,
    /// Verification latency in nanoseconds (if measured).
    pub latency_ns: u64,
}

/// Proof verifier with capability checking.
#[derive(Debug)]
pub struct ProofVerifier {
    /// Configuration.
    config: VerifierConfig,
    /// Proof cache for nonce tracking.
    cache: ProofCache,
    /// Statistics.
    stats: VerifierStats,
}

/// Verifier statistics.
#[derive(Debug, Clone, Copy, Default)]
pub struct VerifierStats {
    /// Total verifications attempted.
    pub verifications: u64,
    /// Successful verifications.
    pub successes: u64,
    /// Failed verifications.
    pub failures: u64,
    /// Expired proofs rejected.
    pub expired_rejections: u64,
    /// Nonce reuse rejections.
    pub nonce_rejections: u64,
    /// Hash mismatch rejections.
    pub hash_rejections: u64,
    /// Capability rejections.
    pub capability_rejections: u64,
}

impl ProofVerifier {
    /// Creates a new proof verifier with default configuration.
    #[must_use]
    pub fn new() -> Self {
        Self::with_config(VerifierConfig::default())
    }

    /// Creates a new proof verifier with custom configuration.
    #[must_use]
    pub fn with_config(config: VerifierConfig) -> Self {
        Self {
            config,
            cache: ProofCache::new(),
            stats: VerifierStats::default(),
        }
    }

    /// Returns the verifier statistics.
    #[inline]
    #[must_use]
    pub const fn stats(&self) -> &VerifierStats {
        &self.stats
    }

    /// Returns a mutable reference to the proof cache.
    #[inline]
    pub fn cache_mut(&mut self) -> &mut ProofCache {
        &mut self.cache
    }

    /// Verifies a proof token.
    ///
    /// # Arguments
    ///
    /// * `token` - The proof token to verify
    /// * `expected_hash` - The expected mutation hash
    /// * `current_time_ns` - Current time in nanoseconds since epoch
    ///
    /// # Returns
    ///
    /// A verification result on success, or an error describing the failure.
    pub fn verify(
        &mut self,
        token: &ProofToken,
        expected_hash: &[u8; 32],
        current_time_ns: u64,
    ) -> ProofResult<VerificationResult> {
        self.stats.verifications += 1;

        // 1. Check mutation hash
        if token.mutation_hash != *expected_hash {
            self.stats.failures += 1;
            self.stats.hash_rejections += 1;
            return Err(ProofError::HashMismatch {
                expected_prefix: ProofCache::hash_prefix(expected_hash),
                actual_prefix: ProofCache::hash_prefix(&token.mutation_hash),
            });
        }

        // 2. Check expiry
        if token.is_expired(current_time_ns) {
            self.stats.failures += 1;
            self.stats.expired_rejections += 1;
            return Err(ProofError::Expired {
                valid_until: token.valid_until_ns,
                current_time: current_time_ns,
            });
        }

        // 3. Check nonce (if enabled)
        if self.config.check_nonce {
            if self.cache.is_nonce_consumed(&token.mutation_hash, token.nonce) {
                self.stats.failures += 1;
                self.stats.nonce_rejections += 1;
                return Err(ProofError::NonceReused { nonce: token.nonce });
            }
        }

        // 4. Verify payload based on tier
        let was_cached = self.verify_payload(token)?;

        // 5. Consume the nonce
        if self.config.check_nonce {
            // Insert into cache as consumed
            let _ = self.cache.insert(
                &token.mutation_hash,
                token.nonce,
                token.tier,
                current_time_ns,
            );
            // Mark as consumed
            let _ = self
                .cache
                .lookup_and_consume(&token.mutation_hash, token.nonce, current_time_ns);
        }

        self.stats.successes += 1;

        Ok(VerificationResult {
            tier: token.tier,
            was_cached,
            nonce: token.nonce,
            latency_ns: 0, // Would be measured in production
        })
    }

    /// Verifies a proof token with capability checking.
    ///
    /// # Arguments
    ///
    /// * `token` - The proof token to verify
    /// * `expected_hash` - The expected mutation hash
    /// * `current_time_ns` - Current time in nanoseconds since epoch
    /// * `capability` - The capability for the target object
    ///
    /// # Returns
    ///
    /// A verification result on success, or an error describing the failure.
    pub fn verify_with_capability(
        &mut self,
        token: &ProofToken,
        expected_hash: &[u8; 32],
        current_time_ns: u64,
        capability: &Capability,
    ) -> ProofResult<VerificationResult> {
        // Check PROVE rights
        if !capability.has_rights(CapRights::PROVE) {
            self.stats.failures += 1;
            self.stats.capability_rejections += 1;
            return Err(ProofError::MissingProveRights {
                object_id: capability.object_id,
            });
        }

        // Proceed with standard verification
        self.verify(token, expected_hash, current_time_ns)
    }

    /// Verifies the proof payload based on tier.
    fn verify_payload(&self, token: &ProofToken) -> ProofResult<bool> {
        match (token.tier, &token.payload) {
            (ProofTier::Reflex, ProofPayload::Hash { hash }) => {
                // Verify hash matches mutation hash
                if *hash != token.mutation_hash {
                    return Err(ProofError::PayloadMismatch {
                        tier: ProofTier::Reflex,
                    });
                }
                Ok(false)
            }

            (ProofTier::Standard, ProofPayload::MerkleWitness { path_len, .. }) => {
                if !self.config.verify_merkle {
                    return Ok(false);
                }

                // Basic validation - full Merkle verification would be done here
                if *path_len > 32 {
                    return Err(ProofError::MerkleVerificationFailed {
                        failed_at_index: *path_len,
                    });
                }
                Ok(false)
            }

            (
                ProofTier::Deep,
                ProofPayload::CoherenceCert {
                    score_before,
                    score_after,
                    ..
                },
            ) => {
                if !self.config.verify_coherence {
                    return Ok(false);
                }

                // Verify coherence score is above threshold
                if *score_after < self.config.min_coherence_score {
                    return Err(ProofError::CoherenceVerificationFailed {
                        score: *score_after,
                        threshold: self.config.min_coherence_score,
                    });
                }

                // Verify coherence doesn't degrade too much
                if *score_after < *score_before / 2 {
                    return Err(ProofError::CoherenceVerificationFailed {
                        score: *score_after,
                        threshold: *score_before / 2,
                    });
                }

                Ok(false)
            }

            // Payload doesn't match tier
            (tier, _) => Err(ProofError::PayloadMismatch { tier }),
        }
    }

    /// Resets the verifier state (clears cache, resets stats).
    pub fn reset(&mut self) {
        self.cache.clear();
        self.stats = VerifierStats::default();
    }
}

impl Default for ProofVerifier {
    fn default() -> Self {
        Self::new()
    }
}

/// Builder for constructing proof verifiers.
#[derive(Debug, Default)]
pub struct ProofVerifierBuilder {
    config: VerifierConfig,
}

impl ProofVerifierBuilder {
    /// Creates a new verifier builder.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            config: VerifierConfig {
                check_nonce: true,
                verify_merkle: true,
                verify_coherence: true,
                min_coherence_score: 5000,
            },
        }
    }

    /// Sets whether to check nonce consumption.
    #[must_use]
    pub const fn check_nonce(mut self, check: bool) -> Self {
        self.config.check_nonce = check;
        self
    }

    /// Sets whether to verify Merkle witnesses.
    #[must_use]
    pub const fn verify_merkle(mut self, verify: bool) -> Self {
        self.config.verify_merkle = verify;
        self
    }

    /// Sets whether to verify coherence certificates.
    #[must_use]
    pub const fn verify_coherence(mut self, verify: bool) -> Self {
        self.config.verify_coherence = verify;
        self
    }

    /// Sets the minimum coherence score.
    #[must_use]
    pub const fn min_coherence_score(mut self, score: u16) -> Self {
        self.config.min_coherence_score = score;
        self
    }

    /// Builds the verifier.
    #[must_use]
    pub fn build(self) -> ProofVerifier {
        ProofVerifier::with_config(self.config)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvix_types::ObjectType;

    fn make_reflex_token(hash: [u8; 32], nonce: u64, valid_until: u64) -> ProofToken {
        ProofToken::new(
            hash,
            ProofTier::Reflex,
            ProofPayload::Hash { hash },
            valid_until,
            nonce,
        )
    }

    fn make_standard_token(hash: [u8; 32], nonce: u64, valid_until: u64) -> ProofToken {
        let mut path = [[0u8; 32]; 32];
        path[0] = hash;

        ProofToken::new(
            hash,
            ProofTier::Standard,
            ProofPayload::MerkleWitness {
                root: hash,
                leaf_index: 0,
                path_len: 1,
                path,
            },
            valid_until,
            nonce,
        )
    }

    fn make_deep_token(hash: [u8; 32], nonce: u64, valid_until: u64) -> ProofToken {
        ProofToken::new(
            hash,
            ProofTier::Deep,
            ProofPayload::CoherenceCert {
                score_before: 9000,
                score_after: 8500,
                partition_id: 1,
                signature: [0u8; 64],
            },
            valid_until,
            nonce,
        )
    }

    #[test]
    fn test_verify_reflex_success() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        let result = verifier.verify(&token, &hash, 1000).unwrap();
        assert_eq!(result.tier, ProofTier::Reflex);
        assert_eq!(result.nonce, 1);
        assert_eq!(verifier.stats().successes, 1);
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let wrong_hash = [1u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        let err = verifier.verify(&token, &wrong_hash, 1000).unwrap_err();
        assert!(matches!(err, ProofError::HashMismatch { .. }));
        assert_eq!(verifier.stats().hash_rejections, 1);
    }

    #[test]
    fn test_verify_expired() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 1000);

        let err = verifier.verify(&token, &hash, 2000).unwrap_err();
        assert!(matches!(err, ProofError::Expired { .. }));
        assert_eq!(verifier.stats().expired_rejections, 1);
    }

    #[test]
    fn test_verify_nonce_reuse() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        // First verification succeeds
        verifier.verify(&token, &hash, 1000).unwrap();

        // Second verification fails (nonce reused)
        let err = verifier.verify(&token, &hash, 1000).unwrap_err();
        assert!(matches!(err, ProofError::NonceReused { nonce: 1 }));
        assert_eq!(verifier.stats().nonce_rejections, 1);
    }

    #[test]
    fn test_verify_standard_success() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_standard_token(hash, 1, 2000);

        let result = verifier.verify(&token, &hash, 1000).unwrap();
        assert_eq!(result.tier, ProofTier::Standard);
    }

    #[test]
    fn test_verify_deep_success() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_deep_token(hash, 1, 2000);

        let result = verifier.verify(&token, &hash, 1000).unwrap();
        assert_eq!(result.tier, ProofTier::Deep);
    }

    #[test]
    fn test_verify_deep_low_coherence() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];

        let token = ProofToken::new(
            hash,
            ProofTier::Deep,
            ProofPayload::CoherenceCert {
                score_before: 9000,
                score_after: 4000, // Below 50% threshold
                partition_id: 1,
                signature: [0u8; 64],
            },
            2000,
            1,
        );

        let err = verifier.verify(&token, &hash, 1000).unwrap_err();
        assert!(matches!(
            err,
            ProofError::CoherenceVerificationFailed { score: 4000, .. }
        ));
    }

    #[test]
    fn test_verify_with_capability_success() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        let cap = Capability::new(
            0x1000,
            ObjectType::VectorStore,
            CapRights::PROVE | CapRights::WRITE,
            0,
            1,
        );

        let result = verifier
            .verify_with_capability(&token, &hash, 1000, &cap)
            .unwrap();
        assert_eq!(result.tier, ProofTier::Reflex);
    }

    #[test]
    fn test_verify_with_capability_missing_rights() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        let cap = Capability::new(
            0x1000,
            ObjectType::VectorStore,
            CapRights::READ, // No PROVE right
            0,
            1,
        );

        let err = verifier
            .verify_with_capability(&token, &hash, 1000, &cap)
            .unwrap_err();
        assert!(matches!(
            err,
            ProofError::MissingProveRights { object_id: 0x1000 }
        ));
        assert_eq!(verifier.stats().capability_rejections, 1);
    }

    #[test]
    fn test_verifier_builder() {
        let verifier = ProofVerifierBuilder::new()
            .check_nonce(false)
            .verify_merkle(false)
            .verify_coherence(true)
            .min_coherence_score(7500)
            .build();

        assert!(!verifier.config.check_nonce);
        assert!(!verifier.config.verify_merkle);
        assert!(verifier.config.verify_coherence);
        assert_eq!(verifier.config.min_coherence_score, 7500);
    }

    #[test]
    fn test_verifier_reset() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];
        let token = make_reflex_token(hash, 1, 2000);

        verifier.verify(&token, &hash, 1000).unwrap();
        assert_eq!(verifier.stats().successes, 1);

        verifier.reset();
        assert_eq!(verifier.stats().successes, 0);
        assert!(verifier.cache_mut().is_empty());
    }

    #[test]
    fn test_payload_mismatch() {
        let mut verifier = ProofVerifier::new();
        let hash = [0u8; 32];

        // Create a Reflex token with wrong payload type
        let token = ProofToken::new(
            hash,
            ProofTier::Reflex,
            ProofPayload::CoherenceCert {
                score_before: 9000,
                score_after: 9000,
                partition_id: 0,
                signature: [0u8; 64],
            },
            2000,
            1,
        );

        let err = verifier.verify(&token, &hash, 1000).unwrap_err();
        assert!(matches!(
            err,
            ProofError::PayloadMismatch {
                tier: ProofTier::Reflex
            }
        ));
    }
}
