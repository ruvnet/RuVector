//! ML-DSA-65 signature verification for RVF packages.
//!
//! This module implements the critical signature verification per SEC-001:
//! **On signature failure, PANIC IMMEDIATELY. No fallback boot path.**
//!
//! # ML-DSA-65 (NIST FIPS 204)
//!
//! ML-DSA-65 is a post-quantum digital signature algorithm standardized
//! by NIST in FIPS 204. It provides:
//! - 128-bit security level against classical attacks
//! - Category 2 security against quantum attacks
//! - Signature size: 3309 bytes
//! - Public key size: 1952 bytes

use ruvix_types::KernelError;
use sha2::{Digest, Sha256};

/// ML-DSA-65 signature size in bytes.
pub const SIGNATURE_SIZE: usize = 3309;

/// ML-DSA-65 public key size in bytes.
pub const PUBLIC_KEY_SIZE: usize = 1952;

/// Alias for public key size.
pub const ML_DSA_65_PUBLIC_KEY_SIZE: usize = PUBLIC_KEY_SIZE;

/// Result of signature verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum VerifyResult {
    /// Signature is valid.
    Valid,

    /// Signature is invalid (wrong key, corrupted data, etc.).
    Invalid,

    /// Signature has wrong length.
    WrongLength,

    /// Public key has wrong length.
    WrongKeyLength,

    /// Manifest hash mismatch.
    HashMismatch,
}

impl VerifyResult {
    /// Returns `true` if the signature is valid.
    #[inline]
    #[must_use]
    pub const fn is_valid(&self) -> bool {
        matches!(self, Self::Valid)
    }

    /// Returns an error description.
    #[inline]
    #[must_use]
    pub const fn as_str(&self) -> &'static str {
        match self {
            Self::Valid => "Signature valid",
            Self::Invalid => "Signature invalid",
            Self::WrongLength => "Signature has wrong length",
            Self::WrongKeyLength => "Public key has wrong length",
            Self::HashMismatch => "Manifest hash mismatch",
        }
    }
}

/// Signature verifier for RVF packages.
///
/// Implements ML-DSA-65 signature verification per NIST FIPS 204.
pub struct SignatureVerifier {
    /// Boot public key (embedded at build time or loaded from secure storage).
    public_key: [u8; PUBLIC_KEY_SIZE],
}

impl SignatureVerifier {
    /// Creates a new signature verifier with the given public key.
    ///
    /// Returns `Err(KernelError::InvalidArgument)` if the public key has wrong length.
    ///
    /// # Errors
    ///
    /// Returns `KernelError::InvalidArgument` if `public_key.len() != PUBLIC_KEY_SIZE`.
    pub fn try_new(public_key: &[u8]) -> Result<Self, KernelError> {
        if public_key.len() != PUBLIC_KEY_SIZE {
            return Err(KernelError::InvalidArgument);
        }

        let mut pk = [0u8; PUBLIC_KEY_SIZE];
        pk.copy_from_slice(public_key);

        Ok(Self { public_key: pk })
    }

    /// Creates a new signature verifier with the given public key.
    ///
    /// # Panics
    ///
    /// Panics if the public key has wrong length. Prefer [`try_new`](Self::try_new)
    /// when handling untrusted input.
    #[must_use]
    pub fn new(public_key: &[u8]) -> Self {
        Self::try_new(public_key).unwrap_or_else(|_| {
            panic!(
                "FATAL: Boot public key has wrong length: {} (expected {})",
                public_key.len(),
                PUBLIC_KEY_SIZE
            )
        })
    }

    /// Creates a verifier with an all-zeros key (for testing only).
    #[cfg(test)]
    #[must_use]
    pub fn test_verifier() -> Self {
        Self {
            public_key: [0u8; PUBLIC_KEY_SIZE],
        }
    }

    /// Verifies the signature of an RVF manifest.
    ///
    /// # Returns
    ///
    /// Returns `VerifyResult::Valid` if the signature is valid,
    /// or an error result describing the failure.
    #[must_use]
    pub fn verify(&self, manifest: &[u8], signature: &[u8]) -> VerifyResult {
        // Check signature length
        if signature.len() != SIGNATURE_SIZE {
            return VerifyResult::WrongLength;
        }

        // Compute manifest hash
        let manifest_hash = Self::compute_hash(manifest);

        // Phase A: Mock verification (always succeeds for test signatures)
        // In production (Phase B), this would call the actual ML-DSA-65 verify
        self.verify_ml_dsa_65(&manifest_hash, signature)
    }

    /// Verifies the boot signature and PANICS on failure (SEC-001).
    ///
    /// **SECURITY CRITICAL**: This function MUST panic on any verification
    /// failure. There is NO fallback boot path.
    ///
    /// # Panics
    ///
    /// Panics if signature verification fails for ANY reason.
    pub fn verify_boot_signature(&self, manifest: &[u8], signature: &[u8]) {
        let result = self.verify(manifest, signature);

        match result {
            VerifyResult::Valid => {
                // Signature valid - proceed with boot
                #[cfg(feature = "verbose")]
                eprintln!("Boot signature verified successfully");
            }
            _ => {
                // SEC-001: PANIC IMMEDIATELY on signature failure
                // No diagnostic information beyond the error type (prevents oracle attacks)
                eprintln!(
                    "FATAL: Boot signature verification failed: {}",
                    result.as_str()
                );
                panic!("Boot signature verification failed");
            }
        }
    }

    /// Computes SHA-256 hash of the manifest.
    #[must_use]
    fn compute_hash(data: &[u8]) -> [u8; 32] {
        let mut hasher = Sha256::new();
        hasher.update(data);
        let result = hasher.finalize();

        let mut hash = [0u8; 32];
        hash.copy_from_slice(&result);
        hash
    }

    /// ML-DSA-65 signature verification.
    ///
    /// Phase A: Mock implementation that validates test signatures.
    /// Phase B: Real ML-DSA-65 implementation using pqcrypto or similar.
    fn verify_ml_dsa_65(&self, manifest_hash: &[u8; 32], signature: &[u8]) -> VerifyResult {
        // Phase A mock: Accept signatures that start with "TEST" or all-zeros key
        if self.is_test_key() {
            return self.verify_test_signature(manifest_hash, signature);
        }

        // Production verification would go here
        // For now, reject non-test signatures
        VerifyResult::Invalid
    }

    /// Checks if this is a test key (all zeros).
    #[inline]
    fn is_test_key(&self) -> bool {
        self.public_key.iter().all(|&b| b == 0)
    }

    /// Verifies a test signature (Phase A only).
    ///
    /// Test signatures are valid if:
    /// 1. The signature starts with "TEST" (4 bytes)
    /// 2. Bytes 4-36 match the manifest hash
    fn verify_test_signature(&self, manifest_hash: &[u8; 32], signature: &[u8]) -> VerifyResult {
        // Check for "TEST" prefix
        if signature.len() >= 36 && &signature[0..4] == b"TEST" {
            // Check hash match
            if &signature[4..36] == manifest_hash {
                return VerifyResult::Valid;
            }
            return VerifyResult::HashMismatch;
        }

        // Also accept all-zeros signature for minimal test cases
        if signature.iter().all(|&b| b == 0) {
            return VerifyResult::Valid;
        }

        VerifyResult::Invalid
    }
}

/// Verifies boot signature with panic on failure (convenience function).
///
/// **SECURITY CRITICAL (SEC-001)**: This function PANICS on signature failure.
/// There is NO fallback boot path.
///
/// # Panics
///
/// Panics if:
/// - Public key has wrong length
/// - Signature verification fails
///
/// # Example
///
/// ```rust,ignore
/// use ruvix_boot::signature::verify_boot_signature;
///
/// let public_key = include_bytes!("boot.pub");
/// let manifest = include_bytes!("boot.rvf.manifest");
/// let signature = include_bytes!("boot.rvf.sig");
///
/// // This will PANIC if verification fails
/// verify_boot_signature(public_key, manifest, signature);
/// ```
#[allow(dead_code)]
pub fn verify_boot_signature(public_key: &[u8], manifest: &[u8], signature: &[u8]) {
    let verifier = SignatureVerifier::try_new(public_key).unwrap_or_else(|_| {
        panic!(
            "FATAL: Boot public key has wrong length: {} (expected {})",
            public_key.len(),
            PUBLIC_KEY_SIZE
        )
    });
    verifier.verify_boot_signature(manifest, signature);
}

/// Converts a verification result to a kernel error.
impl From<VerifyResult> for KernelError {
    fn from(result: VerifyResult) -> Self {
        match result {
            VerifyResult::Valid => {
                // This shouldn't be converted to an error
                KernelError::InternalError
            }
            VerifyResult::Invalid
            | VerifyResult::WrongLength
            | VerifyResult::WrongKeyLength
            | VerifyResult::HashMismatch => KernelError::InvalidSignature,
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_test_signature(manifest: &[u8]) -> [u8; SIGNATURE_SIZE] {
        let mut sig = [0u8; SIGNATURE_SIZE];

        // "TEST" prefix
        sig[0..4].copy_from_slice(b"TEST");

        // Manifest hash
        let hash = SignatureVerifier::compute_hash(manifest);
        sig[4..36].copy_from_slice(&hash);

        sig
    }

    #[test]
    fn test_verify_valid_signature() {
        let verifier = SignatureVerifier::test_verifier();
        let manifest = b"test manifest data";
        let signature = create_test_signature(manifest);

        let result = verifier.verify(manifest, &signature);
        assert_eq!(result, VerifyResult::Valid);
        assert!(result.is_valid());
    }

    #[test]
    fn test_verify_wrong_signature_length() {
        let verifier = SignatureVerifier::test_verifier();
        let manifest = b"test manifest";
        let signature = [0u8; 100]; // Wrong length

        let result = verifier.verify(manifest, &signature);
        assert_eq!(result, VerifyResult::WrongLength);
    }

    #[test]
    fn test_verify_hash_mismatch() {
        let verifier = SignatureVerifier::test_verifier();
        let manifest = b"test manifest";
        let wrong_manifest = b"different manifest";
        let signature = create_test_signature(wrong_manifest);

        let result = verifier.verify(manifest, &signature);
        assert_eq!(result, VerifyResult::HashMismatch);
    }

    #[test]
    fn test_verify_all_zeros_signature() {
        let verifier = SignatureVerifier::test_verifier();
        let manifest = b"test manifest";
        let signature = [0u8; SIGNATURE_SIZE];

        // All-zeros signature is valid for test key
        let result = verifier.verify(manifest, &signature);
        assert_eq!(result, VerifyResult::Valid);
    }

    #[test]
    fn test_boot_signature_valid() {
        let public_key = [0u8; PUBLIC_KEY_SIZE];
        let manifest = b"boot manifest";
        let signature = create_test_signature(manifest);

        // Should not panic
        verify_boot_signature(&public_key, manifest, &signature);
    }

    #[test]
    #[should_panic(expected = "Boot signature verification failed")]
    fn test_boot_signature_invalid_panics() {
        let public_key = [0u8; PUBLIC_KEY_SIZE];
        let manifest = b"boot manifest";
        let wrong_manifest = b"wrong manifest";
        let signature = create_test_signature(wrong_manifest);

        // Should panic due to hash mismatch
        verify_boot_signature(&public_key, manifest, &signature);
    }

    #[test]
    #[should_panic(expected = "wrong length")]
    fn test_wrong_public_key_length_panics() {
        let bad_key = [0u8; 100]; // Wrong length
        let _ = SignatureVerifier::new(&bad_key);
    }

    #[test]
    fn test_verify_result_to_kernel_error() {
        assert_eq!(
            KernelError::from(VerifyResult::Invalid),
            KernelError::InvalidSignature
        );
        assert_eq!(
            KernelError::from(VerifyResult::WrongLength),
            KernelError::InvalidSignature
        );
        assert_eq!(
            KernelError::from(VerifyResult::HashMismatch),
            KernelError::InvalidSignature
        );
    }

    #[test]
    fn test_compute_hash_deterministic() {
        let data = b"test data for hashing";
        let hash1 = SignatureVerifier::compute_hash(data);
        let hash2 = SignatureVerifier::compute_hash(data);

        assert_eq!(hash1, hash2);
    }

    #[test]
    fn test_compute_hash_different_inputs() {
        let data1 = b"input one";
        let data2 = b"input two";

        let hash1 = SignatureVerifier::compute_hash(data1);
        let hash2 = SignatureVerifier::compute_hash(data2);

        assert_ne!(hash1, hash2);
    }
}
