//! Security tests for ADR-087 SEC-001: Boot signature verification.
//!
//! These tests verify that boot signature failures result in immediate panics
//! with no fallback path, as required by SEC-001.

use ruvix_cap::{
    verify_boot_signature_or_panic, verify_signature, BootSignature, BootVerifier,
    SignatureAlgorithm, SignatureVerifyResult, TrustedKey, TrustedKeyStore,
};

// =============================================================================
// SEC-001: Boot Signature Panic Tests
// =============================================================================

fn create_valid_signature() -> BootSignature {
    let mut signature = [0u8; 64];
    signature[0] = 0x42; // Non-zero to pass basic check

    BootSignature::ed25519(
        signature,
        [1u8; 32], // public key
        [2u8; 32], // message hash
    )
}

fn create_trusted_store() -> TrustedKeyStore {
    let mut store = TrustedKeyStore::new();
    store.add_key(TrustedKey::permanent([1u8; 32], 1));
    store
}

#[test]
fn test_sec001_valid_signature_does_not_panic() {
    let signature = create_valid_signature();
    let store = create_trusted_store();

    // This should NOT panic
    verify_boot_signature_or_panic(&signature, &[], &store, 0);
}

#[test]
#[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
fn test_sec001_invalid_signature_panics() {
    let signature = BootSignature::ed25519(
        [0u8; 64],  // All zeros = invalid
        [1u8; 32],  // trusted key
        [2u8; 32],
    );
    let store = create_trusted_store();

    // This MUST panic
    verify_boot_signature_or_panic(&signature, &[], &store, 0);
}

#[test]
#[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
fn test_sec001_untrusted_key_panics() {
    let signature = BootSignature::ed25519(
        [1u8; 64],   // non-zero signature
        [99u8; 32],  // UNTRUSTED key
        [2u8; 32],
    );
    let store = create_trusted_store();

    // This MUST panic
    verify_boot_signature_or_panic(&signature, &[], &store, 0);
}

#[test]
fn test_sec001_verify_returns_correct_error_codes() {
    let store = create_trusted_store();

    // Test UntrustedKey
    let sig = BootSignature::ed25519([1u8; 64], [99u8; 32], [0u8; 32]);
    assert_eq!(
        verify_signature(&sig, &[], &store, 0),
        SignatureVerifyResult::UntrustedKey
    );

    // Test Invalid (zero signature)
    let sig = BootSignature::ed25519([0u8; 64], [1u8; 32], [0u8; 32]);
    assert_eq!(
        verify_signature(&sig, &[], &store, 0),
        SignatureVerifyResult::Invalid
    );

    // Test Valid
    let sig = create_valid_signature();
    assert_eq!(
        verify_signature(&sig, &[], &store, 0),
        SignatureVerifyResult::Valid
    );
}

#[test]
fn test_sec001_trusted_key_expiry() {
    let mut store = TrustedKeyStore::new();
    store.add_key(TrustedKey::new([1u8; 32], 1, 1000)); // expires at 1000ns

    let sig = create_valid_signature();

    // Before expiry: should be valid
    assert_eq!(
        verify_signature(&sig, &[], &store, 500),
        SignatureVerifyResult::Valid
    );

    // After expiry: should be UntrustedKey
    assert_eq!(
        verify_signature(&sig, &[], &store, 2000),
        SignatureVerifyResult::UntrustedKey
    );
}

#[test]
#[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
fn test_sec001_expired_key_panics() {
    let mut store = TrustedKeyStore::new();
    store.add_key(TrustedKey::new([1u8; 32], 1, 1000));

    let sig = create_valid_signature();

    // After expiry: should panic
    verify_boot_signature_or_panic(&sig, &[], &store, 2000);
}

#[test]
fn test_sec001_key_store_capacity() {
    let mut store = TrustedKeyStore::new();

    // Should be able to add 8 keys
    for i in 0..8 {
        let key = TrustedKey::permanent([i as u8; 32], i as u64);
        assert!(store.add_key(key), "Should add key {}", i);
    }

    // 9th key should fail
    let key = TrustedKey::permanent([99u8; 32], 99);
    assert!(!store.add_key(key), "Should reject 9th key");
}

#[test]
fn test_sec001_permanent_key_never_expires() {
    let key = TrustedKey::permanent([0u8; 32], 1);

    // Permanent keys have valid_until_ns = 0, meaning no expiry
    assert!(!key.is_expired(0));
    assert!(!key.is_expired(u64::MAX)); // Even at max time
}

#[test]
fn test_sec001_boot_verifier_valid() {
    let store = create_trusted_store();
    let verifier = BootVerifier::new(store);
    let sig = create_valid_signature();

    // Should not panic
    verifier.verify_or_panic(&sig, &[], 0);
}

#[test]
fn test_sec001_signature_algorithms() {
    // Test that supported algorithms are correct
    assert_eq!(SignatureAlgorithm::Ed25519 as u8, 0);
    assert_eq!(SignatureAlgorithm::EcdsaP256 as u8, 1);
    assert_eq!(SignatureAlgorithm::RsaPss2048 as u8, 2);
    assert_eq!(SignatureAlgorithm::MlDsa as u8, 3);
}

#[test]
fn test_sec001_unsupported_algorithm() {
    let store = create_trusted_store();

    // RSA is not yet supported
    let sig = BootSignature::new(
        SignatureAlgorithm::RsaPss2048,
        [1u8; 64],
        [1u8; 32],
        [0u8; 32],
    );

    assert_eq!(
        verify_signature(&sig, &[], &store, 0),
        SignatureVerifyResult::UnsupportedAlgorithm
    );
}

#[test]
#[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
fn test_sec001_unsupported_algorithm_panics() {
    let store = create_trusted_store();

    let sig = BootSignature::new(
        SignatureAlgorithm::MlDsa, // Not supported
        [1u8; 64],
        [1u8; 32],
        [0u8; 32],
    );

    verify_boot_signature_or_panic(&sig, &[], &store, 0);
}
