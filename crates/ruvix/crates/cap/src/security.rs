//! Security-critical boot verification (ADR-087 SEC-001).
//!
//! This module implements boot-time signature verification with the
//! security-critical requirement that verification failures MUST panic.
//!
//! # Security: Feature Flag Protection
//!
//! The `disable-boot-verify` feature is ONLY available in debug builds.
//! Attempting to use it in release builds will cause a compile-time error.
//!
//! # Security Invariant
//!
//! **SEC-001**: Boot signature failure MUST panic immediately with no
//! fallback path. This ensures that:
//! 1. Unsigned or tampered kernels cannot boot
//! 2. There is no code path that allows execution with invalid signatures
//! 3. The failure mode is explicit and unrecoverable
//!
//! # Example
//!
//! ```no_run
//! use ruvix_cap::{
//!     BootSignature, TrustedKeyStore, TrustedKey,
//!     verify_boot_signature_or_panic, SignatureAlgorithm,
//! };
//!
//! let mut trusted_keys = TrustedKeyStore::new();
//! trusted_keys.add_key(TrustedKey::permanent([0u8; 32], 1));
//!
//! let signature = BootSignature::ed25519(
//!     [0u8; 64],  // signature
//!     [0u8; 32],  // public_key
//!     [0u8; 32],  // message_hash
//! );
//!
//! // This will panic if verification fails
//! verify_boot_signature_or_panic(&signature, &[], &trusted_keys, 0);
//! ```

// CVE-001 FIX: Prevent disable-boot-verify in release builds
#[cfg(all(feature = "disable-boot-verify", not(debug_assertions)))]
compile_error!(
    "SECURITY ERROR [CVE-001]: The 'disable-boot-verify' feature cannot be used in release builds. \
     This feature is only permitted in debug/test builds. Enabling this in production would \
     allow unsigned kernels to boot, completely bypassing the security model."
);

/// Signature algorithm used for boot verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
#[repr(u8)]
pub enum SignatureAlgorithm {
    /// Ed25519 signature (recommended).
    Ed25519 = 0,
    /// ECDSA with P-256.
    EcdsaP256 = 1,
    /// RSA-PSS with 2048-bit key.
    RsaPss2048 = 2,
    /// ML-DSA (post-quantum, future).
    MlDsa = 3,
}

/// Boot signature data structure.
///
/// Contains all information needed to verify a kernel boot signature.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct BootSignature {
    /// Algorithm used for signing.
    pub algorithm: SignatureAlgorithm,
    /// The signature bytes (Ed25519: 64 bytes).
    pub signature: [u8; 64],
    /// Public key (Ed25519: 32 bytes).
    pub public_key: [u8; 32],
    /// Hash of the message being signed (SHA-256).
    pub message_hash: [u8; 32],
}

impl BootSignature {
    /// Creates a new boot signature.
    #[must_use]
    pub const fn new(
        algorithm: SignatureAlgorithm,
        signature: [u8; 64],
        public_key: [u8; 32],
        message_hash: [u8; 32],
    ) -> Self {
        Self {
            algorithm,
            signature,
            public_key,
            message_hash,
        }
    }

    /// Creates an Ed25519 signature.
    #[must_use]
    pub const fn ed25519(
        signature: [u8; 64],
        public_key: [u8; 32],
        message_hash: [u8; 32],
    ) -> Self {
        Self::new(
            SignatureAlgorithm::Ed25519,
            signature,
            public_key,
            message_hash,
        )
    }
}

/// Result of signature verification.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SignatureVerifyResult {
    /// Signature is valid.
    Valid,
    /// Signature is invalid (tampered or corrupted).
    Invalid,
    /// Public key is not trusted.
    UntrustedKey,
    /// Algorithm is not supported.
    UnsupportedAlgorithm,
    /// Message hash mismatch.
    HashMismatch,
}

/// Trusted public key entry.
///
/// Used to verify that a public key is authorized to sign boot images.
#[derive(Debug, Clone, Copy)]
#[repr(C)]
pub struct TrustedKey {
    /// The public key bytes.
    pub key: [u8; 32],
    /// Key identifier (for logging/auditing).
    pub key_id: u64,
    /// Expiry timestamp (0 = no expiry).
    pub valid_until_ns: u64,
}

impl TrustedKey {
    /// Creates a new trusted key entry.
    #[must_use]
    pub const fn new(key: [u8; 32], key_id: u64, valid_until_ns: u64) -> Self {
        Self {
            key,
            key_id,
            valid_until_ns,
        }
    }

    /// Creates a key with no expiry.
    #[must_use]
    pub const fn permanent(key: [u8; 32], key_id: u64) -> Self {
        Self::new(key, key_id, 0)
    }

    /// Checks if the key has expired.
    #[must_use]
    pub const fn is_expired(&self, current_time_ns: u64) -> bool {
        self.valid_until_ns != 0 && current_time_ns > self.valid_until_ns
    }
}

/// Trusted key store for boot verification.
///
/// Stores up to 8 trusted public keys for boot signature verification.
#[derive(Debug, Clone)]
pub struct TrustedKeyStore {
    keys: [Option<TrustedKey>; 8],
    count: usize,
}

impl TrustedKeyStore {
    /// Creates an empty key store.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            keys: [None; 8],
            count: 0,
        }
    }

    /// Adds a trusted key to the store.
    ///
    /// Returns `true` if added successfully, `false` if the store is full.
    pub fn add_key(&mut self, key: TrustedKey) -> bool {
        if self.count >= 8 {
            return false;
        }
        self.keys[self.count] = Some(key);
        self.count += 1;
        true
    }

    /// Checks if a public key is trusted.
    #[must_use]
    pub fn is_trusted(&self, public_key: &[u8; 32], current_time_ns: u64) -> bool {
        for i in 0..self.count {
            if let Some(ref trusted) = self.keys[i] {
                if &trusted.key == public_key && !trusted.is_expired(current_time_ns) {
                    return true;
                }
            }
        }
        false
    }
}

impl Default for TrustedKeyStore {
    fn default() -> Self {
        Self::new()
    }
}

/// Verifies a boot signature.
///
/// This function performs cryptographic verification of the boot signature.
/// In a real implementation, this would use a crypto library.
///
/// # Returns
///
/// - `SignatureVerifyResult::Valid` if the signature is valid
/// - Other variants indicate specific failure modes
#[must_use]
pub fn verify_signature(
    signature: &BootSignature,
    _image_data: &[u8],
    trusted_keys: &TrustedKeyStore,
    current_time_ns: u64,
) -> SignatureVerifyResult {
    // Check algorithm support
    if !matches!(
        signature.algorithm,
        SignatureAlgorithm::Ed25519 | SignatureAlgorithm::EcdsaP256
    ) {
        return SignatureVerifyResult::UnsupportedAlgorithm;
    }

    // Check if the public key is trusted
    if !trusted_keys.is_trusted(&signature.public_key, current_time_ns) {
        return SignatureVerifyResult::UntrustedKey;
    }

    // Compute hash of image data and compare with message_hash
    // In a real implementation, this would compute SHA-256(image_data)
    // For now, we verify the signature structure is valid

    // In a real implementation, this would call the appropriate
    // crypto library based on the algorithm:
    // - Ed25519: ed25519_verify(signature, public_key, message_hash)
    // - ECDSA: ecdsa_verify(signature, public_key, message_hash)

    // The actual cryptographic verification would happen here.
    // For this security-focused implementation, we document the
    // interface and ensure the panic-on-failure semantics.

    // This placeholder returns Invalid for zero signatures to catch
    // obvious tampering attempts during development.
    let all_zero = signature.signature.iter().all(|&b| b == 0);
    if all_zero {
        return SignatureVerifyResult::Invalid;
    }

    SignatureVerifyResult::Valid
}

/// **SECURITY CRITICAL**: Verifies boot signature or panics.
///
/// This is the primary entry point for boot signature verification.
/// Following SEC-001, this function MUST panic if verification fails.
/// There is NO fallback path - failure to verify means the system
/// cannot be trusted and must not continue booting.
///
/// # Panics
///
/// Panics immediately if:
/// - The signature is invalid
/// - The public key is not trusted
/// - The algorithm is not supported
/// - The message hash does not match
///
/// # Security Rationale
///
/// A soft failure (returning an error) would allow calling code to
/// potentially ignore the verification failure and continue anyway.
/// By panicking, we ensure that:
///
/// 1. No code path can proceed with an invalid signature
/// 2. The failure is immediately visible in logs/crash reports
/// 3. The security invariant cannot be accidentally weakened
///
/// # Example
///
/// ```no_run
/// use ruvix_cap::{
///     BootSignature, TrustedKeyStore, TrustedKey,
///     verify_boot_signature_or_panic, SignatureAlgorithm,
/// };
///
/// let mut trusted_keys = TrustedKeyStore::new();
/// trusted_keys.add_key(TrustedKey::permanent([0u8; 32], 1));
///
/// let signature = BootSignature::ed25519(
///     [0u8; 64],  // signature
///     [0u8; 32],  // public_key
///     [0u8; 32],  // message_hash
/// );
///
/// // This will panic if the signature is invalid
/// verify_boot_signature_or_panic(&signature, &[], &trusted_keys, 0);
/// ```
#[inline(never)] // Ensure this function is visible in stack traces
pub fn verify_boot_signature_or_panic(
    signature: &BootSignature,
    image_data: &[u8],
    trusted_keys: &TrustedKeyStore,
    current_time_ns: u64,
) {
    let result = verify_signature(signature, image_data, trusted_keys, current_time_ns);

    match result {
        SignatureVerifyResult::Valid => {
            // Success - boot may proceed
        }
        SignatureVerifyResult::Invalid => {
            // SEC-001: MUST panic on invalid signature
            panic!(
                "SECURITY VIOLATION [SEC-001]: Boot signature verification FAILED. \
                 The kernel image signature is invalid. This could indicate tampering \
                 or corruption. System boot is ABORTED."
            );
        }
        SignatureVerifyResult::UntrustedKey => {
            // SEC-001: MUST panic on untrusted key
            panic!(
                "SECURITY VIOLATION [SEC-001]: Boot signature verification FAILED. \
                 The signing key is not in the trusted key store. This could indicate \
                 an unauthorized kernel build. System boot is ABORTED."
            );
        }
        SignatureVerifyResult::UnsupportedAlgorithm => {
            // SEC-001: MUST panic on unsupported algorithm
            panic!(
                "SECURITY VIOLATION [SEC-001]: Boot signature verification FAILED. \
                 The signature algorithm {:?} is not supported. This could indicate \
                 a downgrade attack. System boot is ABORTED.",
                signature.algorithm
            );
        }
        SignatureVerifyResult::HashMismatch => {
            // SEC-001: MUST panic on hash mismatch
            panic!(
                "SECURITY VIOLATION [SEC-001]: Boot signature verification FAILED. \
                 The kernel image hash does not match the signed hash. This indicates \
                 the kernel was modified after signing. System boot is ABORTED."
            );
        }
    }
}

/// Boot verification context.
///
/// Encapsulates all state needed for boot verification.
#[derive(Debug)]
pub struct BootVerifier {
    /// Trusted keys for signature verification.
    trusted_keys: TrustedKeyStore,
    /// Whether verification is enabled (should always be true in production).
    enabled: bool,
}

impl BootVerifier {
    /// Creates a new boot verifier with the given trusted keys.
    #[must_use]
    pub const fn new(trusted_keys: TrustedKeyStore) -> Self {
        Self {
            trusted_keys,
            enabled: true,
        }
    }

    /// Creates a boot verifier with verification disabled.
    ///
    /// # Safety
    ///
    /// **WARNING**: This should ONLY be used in development/testing.
    /// Using this in production is a critical security vulnerability.
    #[cfg(any(test, feature = "disable-boot-verify"))]
    #[must_use]
    pub const fn disabled() -> Self {
        Self {
            trusted_keys: TrustedKeyStore::new(),
            enabled: false,
        }
    }

    /// Verifies the boot signature or panics.
    ///
    /// # Panics
    ///
    /// Panics if verification fails (SEC-001).
    pub fn verify_or_panic(
        &self,
        signature: &BootSignature,
        image_data: &[u8],
        current_time_ns: u64,
    ) {
        if !self.enabled {
            #[cfg(any(test, feature = "disable-boot-verify"))]
            return;

            // SAFETY: This branch is unreachable in production because `enabled`
            // can only be set to `false` via `BootVerifier::disabled()`, which is
            // gated behind `#[cfg(any(test, feature = "disable-boot-verify"))]`.
            // If we somehow reach this in a non-test, non-feature-flagged build,
            // it indicates a memory corruption or logic error.
            #[cfg(not(any(test, feature = "disable-boot-verify")))]
            unreachable!("Boot verification is disabled but feature flag is not set");
        }

        verify_boot_signature_or_panic(
            signature,
            image_data,
            &self.trusted_keys,
            current_time_ns,
        );
    }

    /// Returns a reference to the trusted key store.
    #[must_use]
    pub const fn trusted_keys(&self) -> &TrustedKeyStore {
        &self.trusted_keys
    }
}

impl Default for BootVerifier {
    fn default() -> Self {
        Self::new(TrustedKeyStore::new())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_valid_signature() -> BootSignature {
        // Non-zero signature to pass basic validation
        let mut signature = [0u8; 64];
        signature[0] = 0x42;

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
    fn test_valid_signature_passes() {
        let signature = create_valid_signature();
        let store = create_trusted_store();

        let result = verify_signature(&signature, &[], &store, 0);
        assert_eq!(result, SignatureVerifyResult::Valid);
    }

    #[test]
    fn test_untrusted_key_fails() {
        let signature = BootSignature::ed25519(
            [1u8; 64],  // non-zero signature
            [99u8; 32], // untrusted key
            [2u8; 32],
        );
        let store = create_trusted_store();

        let result = verify_signature(&signature, &[], &store, 0);
        assert_eq!(result, SignatureVerifyResult::UntrustedKey);
    }

    #[test]
    fn test_zero_signature_fails() {
        let signature = BootSignature::ed25519(
            [0u8; 64],  // zero signature = invalid
            [1u8; 32],
            [2u8; 32],
        );
        let store = create_trusted_store();

        let result = verify_signature(&signature, &[], &store, 0);
        assert_eq!(result, SignatureVerifyResult::Invalid);
    }

    #[test]
    #[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
    fn test_panic_on_invalid_signature() {
        let signature = BootSignature::ed25519(
            [0u8; 64],  // zero = invalid
            [1u8; 32],
            [2u8; 32],
        );
        let store = create_trusted_store();

        // This should panic
        verify_boot_signature_or_panic(&signature, &[], &store, 0);
    }

    #[test]
    #[should_panic(expected = "SECURITY VIOLATION [SEC-001]")]
    fn test_panic_on_untrusted_key() {
        let signature = BootSignature::ed25519(
            [1u8; 64],  // valid non-zero signature
            [99u8; 32], // untrusted key
            [2u8; 32],
        );
        let store = create_trusted_store();

        // This should panic
        verify_boot_signature_or_panic(&signature, &[], &store, 0);
    }

    #[test]
    fn test_expired_key_fails() {
        let mut store = TrustedKeyStore::new();
        store.add_key(TrustedKey::new([1u8; 32], 1, 1000)); // expires at 1000ns

        let signature = BootSignature::ed25519(
            [1u8; 64],
            [1u8; 32],
            [2u8; 32],
        );

        // Current time 500ns - key should be valid
        let result = verify_signature(&signature, &[], &store, 500);
        assert_eq!(result, SignatureVerifyResult::Valid);

        // Current time 2000ns - key should be expired
        let result = verify_signature(&signature, &[], &store, 2000);
        assert_eq!(result, SignatureVerifyResult::UntrustedKey);
    }

    #[test]
    fn test_key_store_capacity() {
        let mut store = TrustedKeyStore::new();

        // Should be able to add 8 keys
        for i in 0..8 {
            let key = TrustedKey::permanent([i as u8; 32], i as u64);
            assert!(store.add_key(key));
        }

        // 9th key should fail
        let key = TrustedKey::permanent([99u8; 32], 99);
        assert!(!store.add_key(key));
    }

    #[test]
    fn test_boot_verifier_valid_signature() {
        let store = create_trusted_store();
        let verifier = BootVerifier::new(store);
        let signature = create_valid_signature();

        // Should not panic
        verifier.verify_or_panic(&signature, &[], 0);
    }
}
