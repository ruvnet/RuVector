//! Security integration tests for ruvix-proof.
//!
//! These tests verify the security properties required by ADR-087 Section 20.4:
//! - Time-bounded validity (100ms default)
//! - Single-use nonces
//! - Capability-gated verification
//! - Cache limits (64 entries, 100ms TTL)

use ruvix_cap::{CapRights, Capability};
use ruvix_proof::{
    ProofCache, ProofCacheConfig, ProofEngine, ProofEngineConfig, ProofError, ProofTier,
    ProofVerifier, VerifierConfig,
};
use ruvix_types::ObjectType;

/// Test that proofs expire after the validity window.
#[test]
fn test_proof_expires_after_validity_window() {
    let mut engine = ProofEngine::new(ProofEngineConfig {
        validity_window_ns: 1000, // 1us for testing
        ..Default::default()
    });
    let mut verifier = ProofVerifier::new();
    let hash = [0xABu8; 32];

    // Generate at time 0
    let token = engine.generate_reflex_proof(&hash, 0).unwrap();

    // Verify at time 500 (before expiry) - should succeed
    assert!(verifier.verify(&token, &hash, 500).is_ok());

    // Create new token for expiry test
    let token2 = engine.generate_reflex_proof(&hash, 0).unwrap();

    // Verify at time 1001 (after expiry) - should fail
    let result = verifier.verify(&token2, &hash, 1001);
    assert!(matches!(result, Err(ProofError::Expired { .. })));
}

/// Test that nonces cannot be reused.
#[test]
fn test_nonce_single_use() {
    let mut engine = ProofEngine::default();
    let mut verifier = ProofVerifier::new();
    let hash = [0xCDu8; 32];

    let token = engine.generate_reflex_proof(&hash, 1000).unwrap();

    // First use succeeds
    let result1 = verifier.verify(&token, &hash, 1000);
    assert!(result1.is_ok());

    // Second use fails with NonceReused
    let result2 = verifier.verify(&token, &hash, 1000);
    assert!(matches!(result2, Err(ProofError::NonceReused { .. })));

    // Statistics should reflect this
    assert_eq!(verifier.stats().nonce_rejections, 1);
}

/// Test that PROVE capability is required for verification.
#[test]
fn test_capability_gated_verification() {
    let mut engine = ProofEngine::default();
    let mut verifier = ProofVerifier::new();
    let hash = [0xEFu8; 32];

    let token = engine.generate_reflex_proof(&hash, 1000).unwrap();

    // Capability with PROVE right - should succeed
    let cap_with_prove = Capability::new(
        0x1000,
        ObjectType::VectorStore,
        CapRights::PROVE | CapRights::WRITE,
        0,
        1,
    );
    let result1 = verifier.verify_with_capability(&token, &hash, 1000, &cap_with_prove);
    assert!(result1.is_ok());

    // New token for second test
    let token2 = engine.generate_reflex_proof(&hash, 1000).unwrap();

    // Capability without PROVE right - should fail
    let cap_without_prove = Capability::new(
        0x2000,
        ObjectType::VectorStore,
        CapRights::READ | CapRights::WRITE,
        0,
        1,
    );
    let result2 = verifier.verify_with_capability(&token2, &hash, 1000, &cap_without_prove);
    assert!(matches!(
        result2,
        Err(ProofError::MissingProveRights { object_id: 0x2000 })
    ));
}

/// Test that cache respects the 64 entry limit.
#[test]
fn test_cache_max_entries() {
    let mut cache = ProofCache::with_config(ProofCacheConfig {
        max_entries: 64,
        ttl_ns: 1_000_000_000, // 1 second
        evict_expired: false,
    });

    // Insert 100 entries
    for i in 0..100u64 {
        let mut hash = [0u8; 32];
        hash[0..8].copy_from_slice(&i.to_le_bytes());
        cache.insert(&hash, i, ProofTier::Reflex, 1000).unwrap();
    }

    // Should have at most 64 entries
    assert!(cache.len() <= 64);

    // Should have evicted some entries
    assert!(cache.stats().evictions >= 36);
}

/// Test that cache respects 100ms TTL.
#[test]
fn test_cache_ttl_100ms() {
    let mut cache = ProofCache::with_config(ProofCacheConfig {
        max_entries: 64,
        ttl_ns: 100_000_000, // 100ms
        evict_expired: true,
    });

    let hash = [0u8; 32];

    // Insert at time 0
    cache.insert(&hash, 1, ProofTier::Reflex, 0).unwrap();
    assert_eq!(cache.len(), 1);

    // Lookup at 50ms - should succeed
    let result = cache.lookup_and_consume(&hash, 1, 50_000_000);
    assert!(result.is_ok());

    // Insert another entry
    cache.insert(&hash, 2, ProofTier::Reflex, 0).unwrap();

    // Evict at 150ms - entry should be gone
    cache.evict_expired(150_000_000);
    assert_eq!(cache.len(), 0);
}

/// Test all three proof tiers route correctly.
#[test]
fn test_all_tiers_generate_correctly() {
    let mut engine = ProofEngine::default();
    let hash = [0x12u8; 32];

    // Reflex tier
    let reflex = engine.generate_reflex_proof(&hash, 1000).unwrap();
    assert_eq!(reflex.tier, ProofTier::Reflex);

    // Standard tier
    let standard = engine
        .generate_standard_proof(&hash, &[1u8; 32], 0, &[], 1000)
        .unwrap();
    assert_eq!(standard.tier, ProofTier::Standard);

    // Deep tier
    let deep = engine
        .generate_deep_proof(&hash, 9000, 8500, 1, &[0u8; 64], 1000)
        .unwrap();
    assert_eq!(deep.tier, ProofTier::Deep);

    // Check statistics
    assert_eq!(engine.stats().reflex_proofs, 1);
    assert_eq!(engine.stats().standard_proofs, 1);
    assert_eq!(engine.stats().deep_proofs, 1);
    assert_eq!(engine.stats().proofs_generated, 3);
}

/// Test that hash mismatches are detected.
#[test]
fn test_hash_mismatch_detection() {
    let mut engine = ProofEngine::default();
    let mut verifier = ProofVerifier::new();
    let hash1 = [0xAAu8; 32];
    let hash2 = [0xBBu8; 32];

    let token = engine.generate_reflex_proof(&hash1, 1000).unwrap();

    // Verify with wrong hash
    let result = verifier.verify(&token, &hash2, 1000);
    assert!(matches!(result, Err(ProofError::HashMismatch { .. })));
    assert_eq!(verifier.stats().hash_rejections, 1);
}

/// Test coherence score threshold enforcement.
#[test]
fn test_coherence_threshold_enforcement() {
    let mut engine = ProofEngine::default();
    let mut verifier = ProofVerifier::with_config(VerifierConfig {
        min_coherence_score: 6000, // 60%
        ..Default::default()
    });
    let hash = [0x55u8; 32];

    // Deep proof with score above threshold
    let token_ok = engine
        .generate_deep_proof(&hash, 8000, 7000, 1, &[0u8; 64], 1000)
        .unwrap();
    assert!(verifier.verify(&token_ok, &hash, 1000).is_ok());

    // Deep proof with score below threshold
    let token_fail = engine
        .generate_deep_proof(&hash, 8000, 5000, 1, &[0u8; 64], 1000)
        .unwrap();
    let result = verifier.verify(&token_fail, &hash, 1000);
    assert!(matches!(
        result,
        Err(ProofError::CoherenceVerificationFailed { score: 5000, .. })
    ));
}

/// Test that attestation size is exactly 82 bytes.
#[test]
fn test_attestation_size_82_bytes() {
    use ruvix_proof::{AttestationBuilder, WitnessLog};

    let attestation = AttestationBuilder::new()
        .proof_term_hash([0xAAu8; 32])
        .environment_hash([0xBBu8; 32])
        .timestamp_ns(1_000_000_000)
        .reduction_steps(100)
        .cache_hit_rate_bps(7500)
        .build();

    let bytes = WitnessLog::serialize_attestation(&attestation);
    assert_eq!(bytes.len(), 82);

    // Verify round-trip
    let restored = WitnessLog::deserialize_attestation(&bytes);
    assert_eq!(attestation.proof_term_hash, restored.proof_term_hash);
    assert_eq!(attestation.environment_hash, restored.environment_hash);
    assert_eq!(attestation.reduction_steps, restored.reduction_steps);
}

/// Test nonces are monotonically increasing.
#[test]
fn test_nonce_monotonicity() {
    let mut engine = ProofEngine::default();
    let hash = [0u8; 32];

    let mut prev_nonce = 0;
    for _ in 0..100 {
        let token = engine.generate_reflex_proof(&hash, 1000).unwrap();
        assert!(token.nonce > prev_nonce || prev_nonce == 0);
        prev_nonce = token.nonce;
    }
}

/// Test concurrent proof generation maintains unique nonces.
#[test]
fn test_unique_nonces() {
    let mut engine = ProofEngine::default();
    let hash = [0u8; 32];

    let mut nonces = std::collections::HashSet::new();

    for _ in 0..1000 {
        let token = engine.generate_reflex_proof(&hash, 1000).unwrap();
        assert!(
            nonces.insert(token.nonce),
            "Duplicate nonce detected: {}",
            token.nonce
        );
    }
}
