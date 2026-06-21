//! Optional witness signing trait (ADR-134 Section 9, ADR-142 Phase 4).
//!
//! Provides pluggable signing for witness records. Production
//! deployments should enable the `crypto-sha256` feature to use
//! [`HmacWitnessSigner`] (HMAC-SHA256 based) or supply a TEE-backed signer.
//! The [`StrictSigner`] (FNV-1a) remains available as a lightweight fallback.

use rvm_types::WitnessRecord;

#[cfg(feature = "crypto-sha256")]
use hmac::{Hmac, Mac};
#[cfg(feature = "crypto-sha256")]
use sha2::{Digest, Sha256};

#[cfg(feature = "crypto-sha256")]
type HmacSha256 = Hmac<Sha256>;

/// Optional cryptographic signing for witness records.
pub trait WitnessSigner {
    /// Sign a witness record. Returns a truncated 8-byte signature.
    fn sign(&self, record: &WitnessRecord) -> [u8; 8];

    /// Verify a signature on a witness record.
    fn verify(&self, record: &WitnessRecord) -> bool;
}

/// No-op signer for deployments without TEE.
///
/// **Security warning:** `NullSigner` accepts all records as valid
/// without performing any integrity check. It is only available in
/// test builds or when the `null-signer` feature is explicitly enabled.
#[cfg(any(test, feature = "null-signer"))]
#[deprecated(note = "Use a real WitnessSigner implementation in production")]
#[derive(Debug, Clone, Copy, Default)]
pub struct NullSigner;

#[cfg(any(test, feature = "null-signer"))]
#[allow(deprecated)]
impl WitnessSigner for NullSigner {
    fn sign(&self, _record: &WitnessRecord) -> [u8; 8] {
        [0u8; 8]
    }

    fn verify(&self, _record: &WitnessRecord) -> bool {
        true
    }
}

/// FNV-1a-based witness signer for non-TEE deployments.
///
/// Computes an FNV-1a hash over the first 52 bytes of the witness
/// record (all fields except `aux` and `pad`) and stores the
/// truncated 8-byte result in the `aux` field as a signature.
///
/// This is not cryptographically strong but provides non-trivial
/// tamper evidence for environments without hardware attestation.
///
/// When the `strict-signing` feature is enabled, this is the
/// recommended default signer.
#[derive(Debug, Clone, Copy, Default)]
pub struct StrictSigner;

impl StrictSigner {
    /// Compute the FNV-1a signature bytes for a witness record.
    ///
    /// We hash the first 52 bytes of the record (all content fields
    /// before `aux` and `pad`).
    fn compute_signature(record: &WitnessRecord) -> [u8; 8] {
        let record_bytes = record.to_bytes();
        // Hash the first 52 bytes (everything before aux + pad).
        let hash = rvm_types::fnv1a_64(&record_bytes[..WitnessRecord::SIGNED_LEN]);
        hash.to_le_bytes()
    }
}

impl WitnessSigner for StrictSigner {
    fn sign(&self, record: &WitnessRecord) -> [u8; 8] {
        Self::compute_signature(record)
    }

    fn verify(&self, record: &WitnessRecord) -> bool {
        let expected = Self::compute_signature(record);
        record.aux == expected
    }
}

/// Compute a 32-byte SHA-256 digest of a witness record's content fields.
///
/// Hashes the first 52 bytes of the serialized record (all fields before
/// `aux` and `pad`, see [`WitnessRecord::to_bytes`]). This digest can be
/// fed to an HMAC signer or used as input to the proof-crate's 64-byte
/// `WitnessSigner` trait.
#[cfg(feature = "crypto-sha256")]
pub fn record_to_digest(record: &WitnessRecord) -> [u8; 32] {
    let buf = record.to_bytes();
    let hash = Sha256::digest(&buf[..WitnessRecord::SIGNED_LEN]);
    let mut out = [0u8; 32];
    out.copy_from_slice(&hash);
    out
}

// ---------------------------------------------------------------------------
// HMAC-SHA256 witness signer (ADR-142 Phase 4)
// ---------------------------------------------------------------------------

/// HMAC-SHA256-based witness signer for the 8-byte `aux` field.
///
/// Computes HMAC-SHA256 over the first 52 bytes of the serialized witness
/// record (all content fields before `aux` and `pad`), then truncates the
/// 32-byte MAC to 8 bytes for storage in the `WitnessRecord.aux` field.
///
/// This is stronger than [`StrictSigner`] (FNV-1a) because HMAC-SHA256
/// is a keyed PRF resistant to forgery. The 8-byte truncation is a
/// constraint of the 64-byte record format.
///
/// # Default Key
///
/// The compile-time default key is `SHA-256(b"rvm-witness-default-key-v1")`.
/// **Production deployments MUST replace this with a TEE-derived key** by
/// calling [`HmacWitnessSigner::new`] with an appropriate secret.
#[cfg(feature = "crypto-sha256")]
#[derive(Clone)]
pub struct HmacWitnessSigner {
    key: [u8; 32],
    /// Pre-keyed HMAC state built once at construction.
    ///
    /// The HMAC key schedule costs two SHA-256 compressions; running it
    /// per record doubles signing cost. Instead we build the keyed Mac
    /// state once and `clone()` it per record (a cheap state copy),
    /// producing byte-identical signatures.
    mac_template: HmacSha256,
}

#[cfg(feature = "crypto-sha256")]
impl HmacWitnessSigner {
    /// Default key derived at compile time: `SHA-256(b"rvm-witness-default-key-v1")`.
    ///
    /// **Security warning:** This key is public. Production deployments
    /// MUST supply a TEE-derived key via [`Self::new`].
    const DEFAULT_KEY_INPUT: &'static [u8] = b"rvm-witness-default-key-v1";

    /// Create a new HMAC-SHA256 witness signer from a 32-byte key.
    ///
    /// The HMAC key schedule is run once here; per-record signing
    /// clones the pre-keyed state instead of re-deriving it.
    #[must_use]
    pub fn new(key: [u8; 32]) -> Self {
        let mac_template = <HmacSha256 as Mac>::new_from_slice(&key)
            .expect("HMAC key length is 32 bytes");
        Self { key, mac_template }
    }

    /// Create a signer using the compile-time default key.
    ///
    /// **Security warning:** The default key is deterministic and public.
    /// Use [`Self::new`] with a TEE-derived key in production.
    #[must_use]
    pub fn with_default_key() -> Self {
        let hash = Sha256::digest(Self::DEFAULT_KEY_INPUT);
        let mut key = [0u8; 32];
        key.copy_from_slice(&hash);
        Self::new(key)
    }

    /// Compute the raw 8-byte truncated HMAC-SHA256 signature.
    fn compute_signature(&self, record: &WitnessRecord) -> [u8; 8] {
        let buf = record.to_bytes();
        // Clone the pre-keyed state: avoids re-running the HMAC key
        // schedule (2 SHA-256 compressions) for every record.
        let mut mac = self.mac_template.clone();
        mac.update(&buf[..WitnessRecord::SIGNED_LEN]);
        let result = mac.finalize();
        let tag = result.into_bytes();
        let mut sig = [0u8; 8];
        sig.copy_from_slice(&tag[..8]);
        sig
    }

    /// Return the 32-byte signing key (for bridging to the proof-crate signer).
    #[must_use]
    pub fn key(&self) -> &[u8; 32] {
        &self.key
    }
}

#[cfg(feature = "crypto-sha256")]
impl WitnessSigner for HmacWitnessSigner {
    fn sign(&self, record: &WitnessRecord) -> [u8; 8] {
        self.compute_signature(record)
    }

    fn verify(&self, record: &WitnessRecord) -> bool {
        let expected = self.compute_signature(record);
        // Constant-time comparison to prevent timing side-channels.
        let mut diff = 0u8;
        let aux_bytes = record.aux;
        let mut i = 0;
        while i < 8 {
            diff |= expected[i] ^ aux_bytes[i];
            i += 1;
        }
        diff == 0
    }
}

/// The default signer type.
///
/// When the `crypto-sha256` feature is enabled, this is
/// [`HmacWitnessSigner`]; otherwise it is [`StrictSigner`].
#[cfg(feature = "crypto-sha256")]
pub type DefaultSigner = HmacWitnessSigner;

/// The default signer type (fallback: FNV-1a based).
#[cfg(not(feature = "crypto-sha256"))]
pub type DefaultSigner = StrictSigner;

/// Return the default signer.
///
/// When the `crypto-sha256` feature is enabled, returns an
/// [`HmacWitnessSigner`] using the compile-time default key
/// `SHA-256(b"rvm-witness-default-key-v1")`.
///
/// **Production deployments MUST replace this** with a signer
/// constructed via [`HmacWitnessSigner::new`] using a TEE-derived key.
///
/// When `crypto-sha256` is not enabled, returns [`StrictSigner`]
/// (FNV-1a based, not cryptographically strong).
#[must_use]
#[cfg(feature = "crypto-sha256")]
pub fn default_signer() -> HmacWitnessSigner {
    HmacWitnessSigner::with_default_key()
}

/// Return the default signer (FNV-1a fallback).
///
/// Returns [`StrictSigner`] when the `crypto-sha256` feature is not enabled.
#[must_use]
#[cfg(not(feature = "crypto-sha256"))]
pub fn default_signer() -> StrictSigner {
    StrictSigner
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    #[allow(deprecated)]
    fn test_null_signer_sign() {
        let signer = NullSigner;
        let record = WitnessRecord::zeroed();
        assert_eq!(signer.sign(&record), [0u8; 8]);
    }

    #[test]
    #[allow(deprecated)]
    fn test_null_signer_verify() {
        let signer = NullSigner;
        let record = WitnessRecord::zeroed();
        assert!(signer.verify(&record));
    }

    #[test]
    fn test_strict_signer_sign_nonzero() {
        let signer = StrictSigner;
        let mut record = WitnessRecord::zeroed();
        record.sequence = 42;
        record.action_kind = 0x01;
        record.actor_partition_id = 7;

        let sig = signer.sign(&record);
        // The signature should not be all zeros for non-zero input.
        assert_ne!(sig, [0u8; 8]);
    }

    #[test]
    fn test_strict_signer_verify_round_trip() {
        let signer = StrictSigner;
        let mut record = WitnessRecord::zeroed();
        record.sequence = 100;
        record.timestamp_ns = 1_000_000;
        record.action_kind = 0x10;
        record.proof_tier = 2;
        record.actor_partition_id = 3;
        record.target_object_id = 99;
        record.capability_hash = 0xDEAD;
        record.prev_hash = 0x1234;
        record.record_hash = 0x5678;

        // Sign: place signature in aux.
        let sig = signer.sign(&record);
        record.aux = sig;

        // Verify should pass.
        assert!(signer.verify(&record));
    }

    #[test]
    fn test_strict_signer_tampered_record_fails() {
        let signer = StrictSigner;
        let mut record = WitnessRecord::zeroed();
        record.sequence = 100;
        record.actor_partition_id = 3;

        // Sign and place in aux.
        let sig = signer.sign(&record);
        record.aux = sig;

        // Tamper with the record.
        record.sequence = 101;

        // Verify should fail.
        assert!(!signer.verify(&record));
    }

    #[test]
    fn test_strict_signer_deterministic() {
        let signer = StrictSigner;
        let mut record = WitnessRecord::zeroed();
        record.sequence = 42;

        let sig1 = signer.sign(&record);
        let sig2 = signer.sign(&record);
        assert_eq!(sig1, sig2);
    }

    #[test]
    fn test_strict_signer_different_records_different_sigs() {
        let signer = StrictSigner;

        let mut r1 = WitnessRecord::zeroed();
        r1.sequence = 1;

        let mut r2 = WitnessRecord::zeroed();
        r2.sequence = 2;

        assert_ne!(signer.sign(&r1), signer.sign(&r2));
    }

    #[test]
    fn test_default_signer_exists() {
        // Just verify the function is callable.
        let _signer = default_signer();
    }

    // -- HMAC witness signer tests (crypto-sha256 feature) -----------------

    #[cfg(feature = "crypto-sha256")]
    mod hmac_witness_tests {
        use super::*;

        #[test]
        fn hmac_signer_sign_nonzero() {
            let signer = HmacWitnessSigner::with_default_key();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;
            record.action_kind = 0x01;
            let sig = signer.sign(&record);
            assert_ne!(sig, [0u8; 8]);
        }

        #[test]
        fn hmac_signer_verify_round_trip() {
            let signer = HmacWitnessSigner::with_default_key();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 100;
            record.timestamp_ns = 1_000_000;
            record.action_kind = 0x10;
            record.proof_tier = 2;
            record.actor_partition_id = 3;
            record.target_object_id = 99;
            record.capability_hash = 0xDEAD;
            record.prev_hash = 0x1234;
            record.record_hash = 0x5678;

            let sig = signer.sign(&record);
            record.aux = sig;
            assert!(signer.verify(&record));
        }

        #[test]
        fn hmac_signer_tampered_record_fails() {
            let signer = HmacWitnessSigner::with_default_key();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 100;
            record.actor_partition_id = 3;

            let sig = signer.sign(&record);
            record.aux = sig;
            record.sequence = 101; // tamper
            assert!(!signer.verify(&record));
        }

        /// Fixed-vector signature stability test.
        ///
        /// The expected bytes are HMAC-SHA256 computed independently
        /// (Python `hmac`/`hashlib`) over the first 52 serialized bytes
        /// of the record with key `SHA-256(b"rvm-witness-default-key-v1")`,
        /// truncated to 8 bytes. This pins the wire format: any change
        /// to record serialization or the HMAC construction (e.g. the
        /// pre-keyed `mac_template` optimization) must keep producing
        /// these exact bytes.
        #[test]
        fn hmac_signer_fixed_vector_stability() {
            let signer = HmacWitnessSigner::with_default_key();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;
            record.action_kind = 0x01;

            let sig = signer.sign(&record);
            assert_eq!(
                sig,
                [0x91, 0x8a, 0x84, 0x19, 0xd6, 0x4c, 0xac, 0x77],
                "HMAC signature output changed: wire-format regression"
            );
        }

        #[test]
        fn hmac_signer_deterministic() {
            let signer = HmacWitnessSigner::with_default_key();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;

            let sig1 = signer.sign(&record);
            let sig2 = signer.sign(&record);
            assert_eq!(sig1, sig2);
        }

        #[test]
        fn hmac_signer_different_records_different_sigs() {
            let signer = HmacWitnessSigner::with_default_key();

            let mut r1 = WitnessRecord::zeroed();
            r1.sequence = 1;

            let mut r2 = WitnessRecord::zeroed();
            r2.sequence = 2;

            assert_ne!(signer.sign(&r1), signer.sign(&r2));
        }

        #[test]
        fn hmac_signer_different_keys_different_sigs() {
            let s1 = HmacWitnessSigner::new([0x11u8; 32]);
            let s2 = HmacWitnessSigner::new([0x22u8; 32]);

            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;

            assert_ne!(s1.sign(&record), s2.sign(&record));
        }

        #[test]
        fn hmac_signer_wrong_key_fails_verify() {
            let s1 = HmacWitnessSigner::new([0x11u8; 32]);
            let s2 = HmacWitnessSigner::new([0x22u8; 32]);

            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;

            let sig = s1.sign(&record);
            record.aux = sig;
            // Verify with different key should fail.
            assert!(!s2.verify(&record));
        }

        #[test]
        fn hmac_default_signer_returns_crypto() {
            // When crypto-sha256 is enabled, default_signer should
            // return an HmacWitnessSigner that produces non-zero sigs.
            let signer = default_signer();
            let mut record = WitnessRecord::zeroed();
            record.sequence = 1;
            let sig = signer.sign(&record);
            assert_ne!(sig, [0u8; 8]);

            // Round-trip should work.
            record.aux = sig;
            assert!(signer.verify(&record));
        }

        #[test]
        fn record_to_digest_is_deterministic() {
            let mut record = WitnessRecord::zeroed();
            record.sequence = 42;
            record.action_kind = 0x05;

            let d1 = record_to_digest(&record);
            let d2 = record_to_digest(&record);
            assert_eq!(d1, d2);
            assert_ne!(d1, [0u8; 32]);
        }

        #[test]
        fn record_to_digest_differs_for_different_records() {
            let mut r1 = WitnessRecord::zeroed();
            r1.sequence = 1;
            let mut r2 = WitnessRecord::zeroed();
            r2.sequence = 2;

            assert_ne!(record_to_digest(&r1), record_to_digest(&r2));
        }
    }
}
