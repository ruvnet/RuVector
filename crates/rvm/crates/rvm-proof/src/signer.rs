//! Witness signing traits and implementations (ADR-142 Phase 2).
//!
//! Provides [`WitnessSigner`] for cryptographically signing witness
//! records in the RVM proof pipeline. Concrete signers included:
//!
//! - [`HmacSha256WitnessSigner`]: HMAC-SHA256 based signer (default,
//!   no heap allocation, `no_std` compatible).
//! - [`Ed25519WitnessSigner`]: Ed25519 signer using `verify_strict`
//!   per ADR-142 amendment. Gated behind `feature = "ed25519"`.
//! - [`DualHmacSigner`]: Strong symmetric signer producing 64-byte
//!   signatures via dual HMAC-SHA256. Gated behind `crypto-sha256`.
//! - [`NullSigner`]: Zero-signature signer for testing only, gated
//!   behind `#[cfg(any(test, feature = "null-signer"))]`.

use crate::constant_time::ct_eq_64;

#[cfg(feature = "crypto-sha256")]
use sha2::{Digest, Sha256};

#[cfg(feature = "crypto-sha256")]
use hmac::{Hmac, Mac};

#[cfg(feature = "crypto-sha256")]
type HmacSha256 = Hmac<Sha256>;

/// Typed verification failure causes.
///
/// Each variant describes a specific reason for signature or
/// attestation failure, enabling precise error handling in the
/// proof pipeline.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum SignatureError {
    /// The signature bytes do not verify against the digest and key.
    BadSignature,
    /// The key identifier is not recognized by this verifier.
    UnknownKey,
    /// The enclave or platform measurement does not match.
    BadMeasurement,
    /// TCB collateral (CRLs, QE identity, etc.) has expired.
    ExpiredCollateral,
    /// The nonce or sequence number has been seen before.
    Replay,
    /// The requested TEE platform is not available on this hardware.
    UnsupportedPlatform,
    /// The input data is structurally invalid (wrong length, bad encoding).
    MalformedInput,
}

/// Trait for cryptographically signing witness records.
///
/// Implementations must be `Send + Sync` so they can be shared across
/// partitions and scheduler contexts in the hypervisor.
pub trait WitnessSigner: Send + Sync {
    /// Produce a 64-byte signature over the given 32-byte digest.
    fn sign(&self, digest: &[u8; 32]) -> [u8; 64];

    /// Verify a 64-byte signature against a 32-byte digest.
    ///
    /// # Errors
    ///
    /// Returns [`SignatureError::BadSignature`] if the signature is invalid.
    fn verify(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), SignatureError>;

    /// Return the canonical signer identifier.
    ///
    /// Computed as SHA-256 over a typed signer descriptor to prevent
    /// cross-algorithm collisions.
    fn signer_id(&self) -> [u8; 32];
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 signer
// ---------------------------------------------------------------------------

/// HMAC-SHA256 witness signer.
///
/// Uses a stored 32-byte key to produce HMAC-SHA256 tags. The 32-byte
/// MAC output is placed in the first 32 bytes of the 64-byte signature
/// buffer (the trailing 32 bytes are zeroed).
///
/// Verification recomputes the MAC and uses constant-time comparison.
#[cfg(feature = "crypto-sha256")]
pub struct HmacSha256WitnessSigner {
    key: [u8; 32],
}

#[cfg(feature = "crypto-sha256")]
impl HmacSha256WitnessSigner {
    /// Domain tag appended to the signer descriptor for `signer_id()`.
    const DOMAIN_TAG: &'static [u8] = b"rvm-witness-hmac";

    /// Create a new HMAC-SHA256 signer from a 32-byte key.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Compute the raw HMAC-SHA256 tag over the digest.
    fn compute_mac(&self, digest: &[u8; 32]) -> [u8; 32] {
        let mut mac =
            <HmacSha256 as Mac>::new_from_slice(&self.key).expect("HMAC key length is 32 bytes");
        mac.update(digest);
        let result = mac.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result.into_bytes());
        out
    }
}

#[cfg(feature = "crypto-sha256")]
impl WitnessSigner for HmacSha256WitnessSigner {
    fn sign(&self, digest: &[u8; 32]) -> [u8; 64] {
        let mac = self.compute_mac(digest);
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&mac);
        sig
    }

    fn verify(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), SignatureError> {
        let expected = self.sign(digest);
        if ct_eq_64(&expected, signature) {
            Ok(())
        } else {
            Err(SignatureError::BadSignature)
        }
    }

    fn signer_id(&self) -> [u8; 32] {
        // key_id = SHA-256(key)
        let key_id = Sha256::digest(self.key);
        // signer_id = SHA-256(0x02 || key_id || domain_tag)
        let mut hasher = Sha256::new();
        hasher.update([0x02]);
        hasher.update(key_id);
        hasher.update(Self::DOMAIN_TAG);
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }
}

// ---------------------------------------------------------------------------
// Ed25519 signer (ADR-142 amendment: verify_strict)
// ---------------------------------------------------------------------------

/// Ed25519 witness signer.
///
/// Uses an Ed25519 keypair (32-byte seed + 32-byte public key) to produce
/// 64-byte Ed25519 signatures. Verification uses `verify_strict()` per
/// ADR-142 amendment, which rejects non-canonical encodings and small-order
/// public keys.
///
/// Requires the `ed25519` feature.
#[cfg(feature = "ed25519")]
pub struct Ed25519WitnessSigner {
    /// Ed25519 secret seed (32 bytes).
    secret_key: [u8; 32],
    /// Ed25519 public key (32 bytes).
    public_key: [u8; 32],
}

#[cfg(feature = "ed25519")]
impl Ed25519WitnessSigner {
    /// Create a new Ed25519 signer from a 32-byte seed.
    ///
    /// The public key is derived from the seed using the Ed25519 key
    /// derivation algorithm.
    #[must_use]
    pub fn from_seed(seed: [u8; 32]) -> Self {
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&seed);
        let verifying_key = signing_key.verifying_key();
        Self {
            secret_key: seed,
            public_key: verifying_key.to_bytes(),
        }
    }

    /// Create a new Ed25519 signer from a pre-existing seed and public key.
    ///
    /// # Panics
    ///
    /// This does **not** verify that `public_key` corresponds to
    /// `secret_key`. Callers must ensure consistency; mismatched keys
    /// will produce signatures that fail verification.
    #[must_use]
    pub const fn new(secret_key: [u8; 32], public_key: [u8; 32]) -> Self {
        Self {
            secret_key,
            public_key,
        }
    }

    /// Return the raw 32-byte public key.
    #[must_use]
    pub const fn public_key(&self) -> &[u8; 32] {
        &self.public_key
    }
}

#[cfg(feature = "ed25519")]
impl WitnessSigner for Ed25519WitnessSigner {
    fn sign(&self, digest: &[u8; 32]) -> [u8; 64] {
        use ed25519_dalek::Signer as _;
        let signing_key = ed25519_dalek::SigningKey::from_bytes(&self.secret_key);
        let signature = signing_key.sign(digest);
        signature.to_bytes()
    }

    fn verify(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), SignatureError> {
        let verifying_key = ed25519_dalek::VerifyingKey::from_bytes(&self.public_key)
            .map_err(|_| SignatureError::MalformedInput)?;
        let sig = ed25519_dalek::Signature::from_bytes(signature);
        // ADR-142 amendment: use verify_strict to reject non-canonical
        // signatures and small-order public keys.
        verifying_key
            .verify_strict(digest, &sig)
            .map_err(|_| SignatureError::BadSignature)
    }

    fn signer_id(&self) -> [u8; 32] {
        // signer_id = SHA-256(0x01 || public_key)
        let mut hasher = Sha256::new();
        hasher.update([0x01]);
        hasher.update(self.public_key);
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }
}

// ---------------------------------------------------------------------------
// Dual-HMAC signer (no_std / no_alloc fallback)
// ---------------------------------------------------------------------------

/// Strong software signer using dual HMAC-SHA256.
///
/// Produces a 64-byte signature by concatenating two HMAC-SHA256 tags:
///
/// ```text
/// sig = HMAC-SHA256(key, digest) || HMAC-SHA256(key, HMAC-SHA256(key, digest))
/// ```
///
/// Provides 256-bit security strength without requiring Ed25519 or
/// `alloc`. **Not publicly verifiable** (symmetric key) -- use for
/// single trust domain only.
///
/// Requires the `crypto-sha256` feature.
#[cfg(feature = "crypto-sha256")]
pub struct DualHmacSigner {
    /// 32-byte symmetric key.
    key: [u8; 32],
}

#[cfg(feature = "crypto-sha256")]
impl DualHmacSigner {
    /// Domain tag appended to the signer descriptor for `signer_id()`.
    const DOMAIN_TAG: &'static [u8] = b"rvm-dual-hmac";

    /// Create a new dual-HMAC signer from a 32-byte key.
    #[must_use]
    pub const fn new(key: [u8; 32]) -> Self {
        Self { key }
    }

    /// Compute a single HMAC-SHA256 tag over `data`.
    fn hmac(&self, data: &[u8]) -> [u8; 32] {
        let mut mac =
            <HmacSha256 as Mac>::new_from_slice(&self.key).expect("HMAC key length is 32 bytes");
        mac.update(data);
        let result = mac.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result.into_bytes());
        out
    }
}

#[cfg(feature = "crypto-sha256")]
impl WitnessSigner for DualHmacSigner {
    fn sign(&self, digest: &[u8; 32]) -> [u8; 64] {
        let tag1 = self.hmac(digest);
        let tag2 = self.hmac(&tag1);
        let mut sig = [0u8; 64];
        sig[..32].copy_from_slice(&tag1);
        sig[32..].copy_from_slice(&tag2);
        sig
    }

    fn verify(&self, digest: &[u8; 32], signature: &[u8; 64]) -> Result<(), SignatureError> {
        let expected = self.sign(digest);
        if ct_eq_64(&expected, signature) {
            Ok(())
        } else {
            Err(SignatureError::BadSignature)
        }
    }

    fn signer_id(&self) -> [u8; 32] {
        // key_hash = SHA-256(key)
        let key_hash = Sha256::digest(self.key);
        // signer_id = SHA-256(0x04 || key_hash || domain_tag)
        let mut hasher = Sha256::new();
        hasher.update([0x04]);
        hasher.update(key_hash);
        hasher.update(Self::DOMAIN_TAG);
        let result = hasher.finalize();
        let mut out = [0u8; 32];
        out.copy_from_slice(&result);
        out
    }
}

// ---------------------------------------------------------------------------
// Null signer (test-only)
// ---------------------------------------------------------------------------

/// Null signer that produces zero signatures.
///
/// Only available in test builds or when the `null-signer` feature is
/// enabled. Useful for unit testing proof pipelines where
/// cryptographic verification is not the focus.
#[cfg(any(test, feature = "null-signer"))]
pub struct NullSigner;

#[cfg(any(test, feature = "null-signer"))]
impl NullSigner {
    /// Create a new null signer.
    #[must_use]
    pub const fn new() -> Self {
        Self
    }
}

#[cfg(any(test, feature = "null-signer"))]
impl Default for NullSigner {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(any(test, feature = "null-signer"))]
impl WitnessSigner for NullSigner {
    fn sign(&self, _digest: &[u8; 32]) -> [u8; 64] {
        [0u8; 64]
    }

    fn verify(&self, _digest: &[u8; 32], _signature: &[u8; 64]) -> Result<(), SignatureError> {
        Ok(())
    }

    fn signer_id(&self) -> [u8; 32] {
        [0u8; 32]
    }
}

// ---------------------------------------------------------------------------
// TEE key derivation (ADR-142 Phase 4)
// ---------------------------------------------------------------------------

/// A bundle of partition-specific HMAC keys derived from a TEE measurement.
///
/// Each key is derived with a distinct domain separator to ensure
/// cryptographic independence. Using any one key does not reveal
/// information about the others.
#[cfg(feature = "crypto-sha256")]
#[derive(Clone)]
pub struct KeyBundle {
    /// Key for witness record signing (HMAC-SHA256 witness signer).
    pub witness_key: [u8; 32],
    /// Key for attestation chain extension / verification.
    pub attestation_key: [u8; 32],
    /// Key for inter-partition communication authentication.
    pub ipc_key: [u8; 32],
}

/// Derives a witness signing key from a TEE measurement and a partition ID.
///
/// ```text
/// key = SHA-256(measurement || partition_id_le_bytes || "rvm-witness-key-v1")
/// ```
///
/// MUST be called with a real TEE measurement in production.
/// In development, measurement can be `SHA-256(b"rvm-dev-measurement")`.
#[cfg(feature = "crypto-sha256")]
#[must_use]
pub fn derive_witness_key(measurement: &[u8; 32], partition_id: u32) -> [u8; 32] {
    derive_key_with_tag(measurement, partition_id, b"rvm-witness-key-v1")
}

/// Derives unique HMAC keys per partition from a root TEE measurement.
///
/// Returns a [`KeyBundle`] containing keys for: witness signing,
/// attestation chain, and IPC authentication. Each key uses a
/// different domain tag to ensure domain separation.
///
/// ```text
/// witness_key     = SHA-256(measurement || pid || "rvm-witness-key-v1")
/// attestation_key = SHA-256(measurement || pid || "rvm-attestation-key-v1")
/// ipc_key         = SHA-256(measurement || pid || "rvm-ipc-key-v1")
/// ```
#[cfg(feature = "crypto-sha256")]
#[must_use]
pub fn derive_key_bundle(measurement: &[u8; 32], partition_id: u32) -> KeyBundle {
    KeyBundle {
        witness_key: derive_key_with_tag(measurement, partition_id, b"rvm-witness-key-v1"),
        attestation_key: derive_key_with_tag(
            measurement,
            partition_id,
            b"rvm-attestation-key-v1",
        ),
        ipc_key: derive_key_with_tag(measurement, partition_id, b"rvm-ipc-key-v1"),
    }
}

/// Internal helper: `SHA-256(measurement || partition_id_le || domain_tag)`.
#[cfg(feature = "crypto-sha256")]
fn derive_key_with_tag(measurement: &[u8; 32], partition_id: u32, tag: &[u8]) -> [u8; 32] {
    let mut hasher = Sha256::new();
    hasher.update(measurement);
    hasher.update(partition_id.to_le_bytes());
    hasher.update(tag);
    let digest = hasher.finalize();
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

/// Compute the canonical dev measurement: `SHA-256(b"rvm-dev-measurement")`.
///
/// This is a deterministic, publicly known value. It MUST NOT be used
/// in production -- it exists solely for local development and testing.
#[cfg(feature = "crypto-sha256")]
#[must_use]
pub fn dev_measurement() -> [u8; 32] {
    let digest = Sha256::digest(b"rvm-dev-measurement");
    let mut out = [0u8; 32];
    out.copy_from_slice(&digest);
    out
}

// ---------------------------------------------------------------------------
// Witness v2 segment seal bridge (ADR-134 v2)
// ---------------------------------------------------------------------------

/// Adapter exposing any proof-crate [`WitnessSigner`] as a witness-crate
/// [`rvm_witness::SegmentSealSigner`].
///
/// This lets the v2 witness log seal Merkle segment roots with the
/// existing CT/QMDB-style signer infrastructure (HMAC-SHA256,
/// dual-HMAC, Ed25519 `verify_strict`, or TEE-backed signers) without
/// `rvm-witness` depending on `rvm-proof`. The seal digest already
/// binds the root, first sequence, and count
/// (see [`rvm_witness::seal_digest`]); the adapter signs that 32-byte
/// digest directly.
pub struct SealSignerAdapter<'a, S: WitnessSigner> {
    inner: &'a S,
}

impl<'a, S: WitnessSigner> SealSignerAdapter<'a, S> {
    /// Wrap a proof-crate signer for use as a segment seal signer.
    #[must_use]
    pub const fn new(inner: &'a S) -> Self {
        Self { inner }
    }
}

impl<S: WitnessSigner> rvm_witness::SegmentSealSigner for SealSignerAdapter<'_, S> {
    fn sign_root(&self, digest: &[u8; 32]) -> [u8; 64] {
        self.inner.sign(digest)
    }

    fn verify_root(&self, digest: &[u8; 32], signature: &[u8; 64]) -> bool {
        self.inner.verify(digest, signature).is_ok()
    }
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;

    // -- HMAC-SHA256 signer tests ------------------------------------------

    #[cfg(feature = "crypto-sha256")]
    mod hmac_tests {
        use super::*;

        fn test_key() -> [u8; 32] {
            let mut key = [0u8; 32];
            // Deterministic test key.
            for (i, byte) in key.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *byte = (i as u8).wrapping_mul(0x37).wrapping_add(0x42);
                }
            }
            key
        }

        #[test]
        fn sign_verify_round_trip() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest = [0xAAu8; 32];
            let sig = signer.sign(&digest);
            assert!(signer.verify(&digest, &sig).is_ok());
        }

        #[test]
        fn verify_rejects_tampered_signature() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest = [0xBBu8; 32];
            let mut sig = signer.sign(&digest);
            sig[0] ^= 0xFF; // Flip bits in the first byte.
            assert_eq!(signer.verify(&digest, &sig), Err(SignatureError::BadSignature));
        }

        #[test]
        fn verify_rejects_wrong_digest() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest_a = [0xAAu8; 32];
            let digest_b = [0xBBu8; 32];
            let sig = signer.sign(&digest_a);
            assert_eq!(signer.verify(&digest_b, &sig), Err(SignatureError::BadSignature));
        }

        #[test]
        fn verify_rejects_different_key() {
            let signer_a = HmacSha256WitnessSigner::new([0x11u8; 32]);
            let signer_b = HmacSha256WitnessSigner::new([0x22u8; 32]);
            let digest = [0xCCu8; 32];
            let sig = signer_a.sign(&digest);
            assert_eq!(signer_b.verify(&digest, &sig), Err(SignatureError::BadSignature));
        }

        #[test]
        fn signature_is_deterministic() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest = [0xDDu8; 32];
            let sig1 = signer.sign(&digest);
            let sig2 = signer.sign(&digest);
            assert_eq!(sig1, sig2);
        }

        #[test]
        fn signature_trailing_bytes_are_zero() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest = [0xEEu8; 32];
            let sig = signer.sign(&digest);
            // HMAC-SHA256 produces 32 bytes; remaining 32 must be zero.
            assert_eq!(&sig[32..64], &[0u8; 32]);
        }

        #[test]
        fn signer_id_is_deterministic() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let id1 = signer.signer_id();
            let id2 = signer.signer_id();
            assert_eq!(id1, id2);
        }

        #[test]
        fn signer_id_differs_for_different_keys() {
            let signer_a = HmacSha256WitnessSigner::new([0x11u8; 32]);
            let signer_b = HmacSha256WitnessSigner::new([0x22u8; 32]);
            assert_ne!(signer_a.signer_id(), signer_b.signer_id());
        }

        #[test]
        fn signer_id_is_not_zero() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            assert_ne!(signer.signer_id(), [0u8; 32]);
        }

        #[test]
        fn sign_zero_digest() {
            let signer = HmacSha256WitnessSigner::new(test_key());
            let digest = [0u8; 32];
            let sig = signer.sign(&digest);
            assert!(signer.verify(&digest, &sig).is_ok());
            // Signature should not be all zeros (HMAC of zeros is non-zero).
            assert_ne!(&sig[..32], &[0u8; 32]);
        }
    }

    // -- Ed25519 signer tests -----------------------------------------------

    #[cfg(feature = "ed25519")]
    mod ed25519_tests {
        use super::*;

        fn test_seed() -> [u8; 32] {
            let mut seed = [0u8; 32];
            for (i, byte) in seed.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *byte = (i as u8).wrapping_mul(0x5A).wrapping_add(0x13);
                }
            }
            seed
        }

        #[test]
        fn sign_verify_round_trip() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest = [0xAAu8; 32];
            let sig = signer.sign(&digest);
            assert!(signer.verify(&digest, &sig).is_ok());
        }

        #[test]
        fn verify_rejects_tampered_signature() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest = [0xBBu8; 32];
            let mut sig = signer.sign(&digest);
            sig[0] ^= 0xFF;
            assert_eq!(
                signer.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn verify_rejects_wrong_digest() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest_a = [0xAAu8; 32];
            let digest_b = [0xBBu8; 32];
            let sig = signer.sign(&digest_a);
            assert_eq!(
                signer.verify(&digest_b, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn different_seeds_produce_different_signatures() {
            let signer_a = Ed25519WitnessSigner::from_seed([0x11u8; 32]);
            let signer_b = Ed25519WitnessSigner::from_seed([0x22u8; 32]);
            let digest = [0xCCu8; 32];
            let sig_a = signer_a.sign(&digest);
            let sig_b = signer_b.sign(&digest);
            assert_ne!(sig_a, sig_b);
        }

        #[test]
        fn cross_key_verify_fails() {
            let signer_a = Ed25519WitnessSigner::from_seed([0x11u8; 32]);
            let signer_b = Ed25519WitnessSigner::from_seed([0x22u8; 32]);
            let digest = [0xCCu8; 32];
            let sig = signer_a.sign(&digest);
            assert_eq!(
                signer_b.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn signature_is_deterministic() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest = [0xDDu8; 32];
            let sig1 = signer.sign(&digest);
            let sig2 = signer.sign(&digest);
            assert_eq!(sig1, sig2);
        }

        #[test]
        fn signature_fills_all_64_bytes() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest = [0xEEu8; 32];
            let sig = signer.sign(&digest);
            // Ed25519 signatures use all 64 bytes; extremely unlikely
            // that both halves are zero.
            assert_ne!(&sig[..32], &[0u8; 32]);
            assert_ne!(&sig[32..64], &[0u8; 32]);
        }

        #[test]
        fn signer_id_is_deterministic() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let id1 = signer.signer_id();
            let id2 = signer.signer_id();
            assert_eq!(id1, id2);
        }

        #[test]
        fn signer_id_differs_for_different_keys() {
            let signer_a = Ed25519WitnessSigner::from_seed([0x11u8; 32]);
            let signer_b = Ed25519WitnessSigner::from_seed([0x22u8; 32]);
            assert_ne!(signer_a.signer_id(), signer_b.signer_id());
        }

        #[test]
        fn signer_id_is_not_zero() {
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            assert_ne!(signer.signer_id(), [0u8; 32]);
        }

        #[test]
        fn public_key_accessor() {
            let seed = test_seed();
            let signer = Ed25519WitnessSigner::from_seed(seed);
            let pk = signer.public_key();
            // Public key should not be all zeros (valid Ed25519 derivation).
            assert_ne!(pk, &[0u8; 32]);
        }

        #[test]
        fn verify_strict_rejects_non_canonical() {
            // Construct a signature with S >= L (non-canonical).
            // The group order L for Ed25519 starts with 0xED in the
            // high byte of S (bytes 32..64). Setting all S bytes to
            // 0xFF guarantees S > L.
            let signer = Ed25519WitnessSigner::from_seed(test_seed());
            let digest = [0xAAu8; 32];
            let mut sig = signer.sign(&digest);
            // Overwrite the S component (bytes 32..64) with 0xFF.
            for byte in &mut sig[32..64] {
                *byte = 0xFF;
            }
            assert_eq!(
                signer.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn from_seed_and_new_produce_same_results() {
            let seed = test_seed();
            let from_seed = Ed25519WitnessSigner::from_seed(seed);
            let from_new =
                Ed25519WitnessSigner::new(seed, *from_seed.public_key());
            let digest = [0xFFu8; 32];
            assert_eq!(from_seed.sign(&digest), from_new.sign(&digest));
            assert_eq!(from_seed.signer_id(), from_new.signer_id());
        }
    }

    // -- Dual-HMAC signer tests -----------------------------------------------

    #[cfg(feature = "crypto-sha256")]
    mod dual_hmac_tests {
        use super::*;

        fn test_key() -> [u8; 32] {
            let mut key = [0u8; 32];
            for (i, byte) in key.iter_mut().enumerate() {
                #[allow(clippy::cast_possible_truncation)]
                {
                    *byte = (i as u8).wrapping_mul(0x4B).wrapping_add(0x19);
                }
            }
            key
        }

        #[test]
        fn sign_verify_round_trip() {
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xAAu8; 32];
            let sig = signer.sign(&digest);
            assert!(signer.verify(&digest, &sig).is_ok());
        }

        #[test]
        fn verify_rejects_tampered_signature() {
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xBBu8; 32];
            let mut sig = signer.sign(&digest);
            sig[0] ^= 0xFF;
            assert_eq!(
                signer.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn verify_rejects_tampered_second_half() {
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xBBu8; 32];
            let mut sig = signer.sign(&digest);
            sig[32] ^= 0xFF;
            assert_eq!(
                signer.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn different_keys_produce_different_signatures() {
            let signer_a = DualHmacSigner::new([0x11u8; 32]);
            let signer_b = DualHmacSigner::new([0x22u8; 32]);
            let digest = [0xCCu8; 32];
            let sig_a = signer_a.sign(&digest);
            let sig_b = signer_b.sign(&digest);
            assert_ne!(sig_a, sig_b);
        }

        #[test]
        fn cross_key_verify_fails() {
            let signer_a = DualHmacSigner::new([0x11u8; 32]);
            let signer_b = DualHmacSigner::new([0x22u8; 32]);
            let digest = [0xCCu8; 32];
            let sig = signer_a.sign(&digest);
            assert_eq!(
                signer_b.verify(&digest, &sig),
                Err(SignatureError::BadSignature)
            );
        }

        #[test]
        fn signature_is_deterministic() {
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xDDu8; 32];
            let sig1 = signer.sign(&digest);
            let sig2 = signer.sign(&digest);
            assert_eq!(sig1, sig2);
        }

        #[test]
        fn signature_uses_full_64_bytes() {
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xEEu8; 32];
            let sig = signer.sign(&digest);
            // Both halves should be non-zero for any non-trivial key.
            assert_ne!(&sig[..32], &[0u8; 32]);
            assert_ne!(&sig[32..64], &[0u8; 32]);
        }

        #[test]
        fn second_half_is_hmac_of_first_half() {
            // Verify the construction: sig[32..64] = HMAC(key, sig[..32])
            let signer = DualHmacSigner::new(test_key());
            let digest = [0xFFu8; 32];
            let sig = signer.sign(&digest);
            let recomputed_tag2 = signer.hmac(&sig[..32]);
            assert_eq!(&sig[32..64], &recomputed_tag2);
        }

        #[test]
        fn signer_id_is_deterministic() {
            let signer = DualHmacSigner::new(test_key());
            let id1 = signer.signer_id();
            let id2 = signer.signer_id();
            assert_eq!(id1, id2);
        }

        #[test]
        fn signer_id_differs_for_different_keys() {
            let signer_a = DualHmacSigner::new([0x11u8; 32]);
            let signer_b = DualHmacSigner::new([0x22u8; 32]);
            assert_ne!(signer_a.signer_id(), signer_b.signer_id());
        }

        #[test]
        fn signer_id_is_not_zero() {
            let signer = DualHmacSigner::new(test_key());
            assert_ne!(signer.signer_id(), [0u8; 32]);
        }

        #[test]
        fn signer_id_differs_from_hmac_signer() {
            let key = test_key();
            let dual = DualHmacSigner::new(key);
            let hmac = HmacSha256WitnessSigner::new(key);
            // Different domain tags (0x04 vs 0x02) ensure distinct IDs.
            assert_ne!(dual.signer_id(), hmac.signer_id());
        }
    }

    // -- Null signer tests -------------------------------------------------

    #[test]
    fn null_signer_sign_returns_zeros() {
        let signer = NullSigner::new();
        let digest = [0xAAu8; 32];
        let sig = signer.sign(&digest);
        assert_eq!(sig, [0u8; 64]);
    }

    #[test]
    fn null_signer_verify_always_ok() {
        let signer = NullSigner::new();
        let digest = [0xBBu8; 32];
        let sig = [0xFFu8; 64]; // Arbitrary non-zero signature.
        assert!(signer.verify(&digest, &sig).is_ok());
    }

    #[test]
    fn null_signer_id_is_zero() {
        let signer = NullSigner::new();
        assert_eq!(signer.signer_id(), [0u8; 32]);
    }

    #[test]
    fn null_signer_default() {
        #[allow(clippy::default_constructed_unit_structs)]
        let signer = NullSigner::default();
        assert_eq!(signer.sign(&[0u8; 32]), [0u8; 64]);
    }

    // -- TEE key derivation tests -------------------------------------------

    #[cfg(feature = "crypto-sha256")]
    mod key_derivation_tests {
        use super::*;

        fn test_measurement() -> [u8; 32] {
            [0xAA; 32]
        }

        #[test]
        fn same_measurement_same_partition_same_key() {
            let m = test_measurement();
            let k1 = derive_witness_key(&m, 1);
            let k2 = derive_witness_key(&m, 1);
            assert_eq!(k1, k2);
        }

        #[test]
        fn different_partitions_different_keys() {
            let m = test_measurement();
            let k1 = derive_witness_key(&m, 1);
            let k2 = derive_witness_key(&m, 2);
            assert_ne!(k1, k2);
        }

        #[test]
        fn different_measurements_different_keys() {
            let m1 = [0xAA; 32];
            let m2 = [0xBB; 32];
            let k1 = derive_witness_key(&m1, 1);
            let k2 = derive_witness_key(&m2, 1);
            assert_ne!(k1, k2);
        }

        #[test]
        fn key_bundle_keys_are_all_distinct() {
            let m = test_measurement();
            let bundle = derive_key_bundle(&m, 1);
            assert_ne!(bundle.witness_key, bundle.attestation_key);
            assert_ne!(bundle.witness_key, bundle.ipc_key);
            assert_ne!(bundle.attestation_key, bundle.ipc_key);
        }

        #[test]
        fn key_bundle_deterministic() {
            let m = test_measurement();
            let b1 = derive_key_bundle(&m, 5);
            let b2 = derive_key_bundle(&m, 5);
            assert_eq!(b1.witness_key, b2.witness_key);
            assert_eq!(b1.attestation_key, b2.attestation_key);
            assert_eq!(b1.ipc_key, b2.ipc_key);
        }

        #[test]
        fn key_bundle_differs_across_partitions() {
            let m = test_measurement();
            let b1 = derive_key_bundle(&m, 1);
            let b2 = derive_key_bundle(&m, 2);
            assert_ne!(b1.witness_key, b2.witness_key);
            assert_ne!(b1.attestation_key, b2.attestation_key);
            assert_ne!(b1.ipc_key, b2.ipc_key);
        }

        #[test]
        fn dev_measurement_is_nonzero() {
            let m = dev_measurement();
            assert_ne!(m, [0u8; 32]);
        }

        #[test]
        fn dev_measurement_is_deterministic() {
            let m1 = dev_measurement();
            let m2 = dev_measurement();
            assert_eq!(m1, m2);
        }

        #[test]
        fn dev_measurement_produces_nonzero_keys() {
            let m = dev_measurement();
            let bundle = derive_key_bundle(&m, 0);
            assert_ne!(bundle.witness_key, [0u8; 32]);
            assert_ne!(bundle.attestation_key, [0u8; 32]);
            assert_ne!(bundle.ipc_key, [0u8; 32]);
        }

        #[test]
        fn derive_witness_key_matches_bundle_witness_key() {
            let m = test_measurement();
            let standalone = derive_witness_key(&m, 7);
            let bundle = derive_key_bundle(&m, 7);
            assert_eq!(standalone, bundle.witness_key);
        }
    }

    // -- SignatureError tests -----------------------------------------------

    #[test]
    fn signature_error_clone_eq() {
        let a = SignatureError::BadSignature;
        let b = a.clone();
        assert_eq!(a, b);
    }

    #[test]
    fn signature_error_variants_distinct() {
        let variants = [
            SignatureError::BadSignature,
            SignatureError::UnknownKey,
            SignatureError::BadMeasurement,
            SignatureError::ExpiredCollateral,
            SignatureError::Replay,
            SignatureError::UnsupportedPlatform,
            SignatureError::MalformedInput,
        ];
        for i in 0..variants.len() {
            for j in (i + 1)..variants.len() {
                assert_ne!(variants[i], variants[j]);
            }
        }
    }

    #[test]
    fn signature_error_debug_is_implemented() {
        // Verify Debug is implemented by using write! to a fixed buffer.
        use core::fmt::Write;
        struct Buf([u8; 64], usize);
        impl Write for Buf {
            fn write_str(&mut self, s: &str) -> core::fmt::Result {
                for b in s.bytes() {
                    if self.1 < self.0.len() {
                        self.0[self.1] = b;
                        self.1 += 1;
                    }
                }
                Ok(())
            }
        }
        let err = SignatureError::BadSignature;
        let mut buf = Buf([0u8; 64], 0);
        write!(buf, "{err:?}").unwrap();
        // The debug output should contain "BadSignature".
        let written = &buf.0[..buf.1];
        assert!(written.windows(12).any(|w| w == b"BadSignature"));
    }

    // -- SealSignerAdapter tests (ADR-134 v2 segment sealing) ---------------

    #[cfg(feature = "crypto-sha256")]
    mod seal_adapter_tests {
        use super::*;
        use rvm_types::WitnessRecordV2;
        use rvm_witness::{verify_inclusion, verify_seal, WitnessLogV2};

        #[test]
        fn hmac_signer_seals_v2_segments() {
            let log = WitnessLogV2::<64, 8>::new();
            for i in 0..5u64 {
                let mut r = WitnessRecordV2::zeroed();
                r.target_object_id = i;
                log.append(r);
            }

            let signer = HmacSha256WitnessSigner::new([0x5Au8; 32]);
            let adapter = SealSignerAdapter::new(&signer);
            let (sealed, acc) = log.seal_segment(&adapter).unwrap();

            assert_eq!(sealed.first_sequence, 0);
            assert_eq!(sealed.count, 5);
            assert!(verify_seal(&sealed, &adapter));

            // Inclusion proofs verify against the sealed root.
            for seq in 0..5u64 {
                let proof = acc.proof_for_sequence(seq).unwrap();
                #[allow(clippy::cast_possible_truncation)]
                let mac = acc.leaf(seq as usize).unwrap();
                assert!(verify_inclusion(&sealed.root, &mac, &proof));
            }
        }

        #[test]
        fn seal_fails_with_wrong_key() {
            let log = WitnessLogV2::<16, 8>::new();
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = 1;
            log.append(r);

            let signer = HmacSha256WitnessSigner::new([0x5Au8; 32]);
            let adapter = SealSignerAdapter::new(&signer);
            let (sealed, _) = log.seal_segment(&adapter).unwrap();

            let other = HmacSha256WitnessSigner::new([0x5Bu8; 32]);
            let other_adapter = SealSignerAdapter::new(&other);
            assert!(!verify_seal(&sealed, &other_adapter));
        }

        #[test]
        fn dual_hmac_signer_seals_v2_segments() {
            let log = WitnessLogV2::<16, 8>::new();
            let mut r = WitnessRecordV2::zeroed();
            r.target_object_id = 42;
            log.append(r);

            let signer = DualHmacSigner::new([0x77u8; 32]);
            let adapter = SealSignerAdapter::new(&signer);
            let (sealed, _) = log.seal_segment(&adapter).unwrap();
            assert!(verify_seal(&sealed, &adapter));

            let mut bad = sealed;
            bad.root[0] ^= 1;
            assert!(!verify_seal(&bad, &adapter));
        }
    }
}
