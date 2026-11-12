//! Ed25519 helpers. The registry holds public keys and signatures only.
//!
//! No private key is ever written to the registry directory or held in a
//! registry object. A caller that wants the registry to countersign something —
//! a witness receipt — lends it a [`Signer`], and the key material stays inside
//! the caller's own object for the lifetime of that borrow.
//!
//! Signatures are Ed25519 over the object id (the ASCII bytes of
//! `sha256:<hex>`), per the identity rules in `registry-model.md`. Signing the
//! id rather than the body is what makes the signature independent of the
//! signature array itself, so a countersignature can be added without
//! invalidating the publisher's.

use crate::error::{RegistryError, Result};
use base64::engine::general_purpose::STANDARD as B64;
use base64::Engine;
use ed25519_dalek::{Signature as EdSignature, Verifier, VerifyingKey};
use rvf_forge_core::canonical::hex_lower;

/// The `keyId` prefix for Ed25519 keys.
pub const ED25519_KEY_ID_PREFIX: &str = "ed25519:";

/// The `alg` string for Ed25519 public-key entries.
pub const ED25519_ALG: &str = "ed25519";

/// Base64 (standard alphabet, padded) encoding.
pub fn b64_encode(bytes: &[u8]) -> String {
    B64.encode(bytes)
}

/// Decode standard base64.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] if the input is not valid base64.
pub fn b64_decode(s: &str) -> Result<Vec<u8>> {
    B64.decode(s)
        .map_err(|e| RegistryError::invalid(format!("invalid base64: {e}")))
}

/// Parse a base64-encoded 32-byte Ed25519 public key.
///
/// # Errors
///
/// [`RegistryError::InvalidObject`] if the input is not base64, is not 32
/// bytes, or is not a valid curve point.
pub fn verifying_key_from_b64(s: &str) -> Result<VerifyingKey> {
    let bytes = b64_decode(s)?;
    let raw: [u8; 32] = bytes.as_slice().try_into().map_err(|_| {
        RegistryError::invalid(format!("ed25519 key is {} bytes, want 32", bytes.len()))
    })?;
    VerifyingKey::from_bytes(&raw)
        .map_err(|e| RegistryError::invalid(format!("not a valid ed25519 public key: {e}")))
}

/// The canonical `keyId` for a public key: `ed25519:<hex of the 32 raw bytes>`.
///
/// The registry treats `keyId` as an opaque label and always verifies against
/// the `publicKey` field of the matching publisher entry, so this is a
/// convenience for constructing records rather than a trusted encoding.
pub fn key_id_for(key: &VerifyingKey) -> String {
    format!("{ED25519_KEY_ID_PREFIX}{}", hex_lower(key.as_bytes()))
}

/// Verify a base64 Ed25519 signature over an object id.
///
/// # Errors
///
/// [`RegistryError::SignatureInvalid`] if the signature is malformed or does
/// not verify.
pub fn verify_over_id(object_id: &str, sig_b64: &str, key: &VerifyingKey) -> Result<()> {
    let bytes = b64_decode(sig_b64).map_err(|e| RegistryError::SignatureInvalid {
        detail: format!("signature is not base64: {e}"),
    })?;
    let raw: [u8; 64] =
        bytes
            .as_slice()
            .try_into()
            .map_err(|_| RegistryError::SignatureInvalid {
                detail: format!("signature is {} bytes, want 64", bytes.len()),
            })?;
    let sig = EdSignature::from_bytes(&raw);
    key.verify(object_id.as_bytes(), &sig)
        .map_err(|e| RegistryError::SignatureInvalid {
            detail: format!("ed25519 verification failed for {object_id}: {e}"),
        })
}

/// Something that can produce Ed25519 signatures on the registry's behalf.
///
/// Implemented for [`ed25519_dalek::SigningKey`]. The registry stores an
/// `Arc<dyn Signer>` and never persists what is behind it.
pub trait Signer {
    /// The `keyId` this signer's signatures should be labelled with.
    fn key_id(&self) -> String;
    /// The public half, so the registry can record it and callers can verify.
    fn verifying_key(&self) -> VerifyingKey;
    /// Sign a message, returning the raw 64-byte signature.
    fn sign_bytes(&self, message: &[u8]) -> Vec<u8>;
}

impl Signer for ed25519_dalek::SigningKey {
    fn key_id(&self) -> String {
        key_id_for(&self.verifying_key())
    }

    fn verifying_key(&self) -> VerifyingKey {
        ed25519_dalek::SigningKey::verifying_key(self)
    }

    fn sign_bytes(&self, message: &[u8]) -> Vec<u8> {
        use ed25519_dalek::Signer as _;
        self.sign(message).to_bytes().to_vec()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ed25519_dalek::SigningKey;

    fn key(seed: u8) -> SigningKey {
        SigningKey::from_bytes(&[seed; 32])
    }

    #[test]
    fn key_id_is_prefixed_hex_of_the_public_key() {
        let sk = key(1);
        let id = key_id_for(&sk.verifying_key());
        assert!(id.starts_with(ED25519_KEY_ID_PREFIX));
        assert_eq!(id.len(), ED25519_KEY_ID_PREFIX.len() + 64);
    }

    #[test]
    fn signature_over_id_round_trips() {
        let sk = key(2);
        let id = "sha256:0000000000000000000000000000000000000000000000000000000000000000";
        let sig = b64_encode(&sk.sign_bytes(id.as_bytes()));
        verify_over_id(id, &sig, &sk.verifying_key()).unwrap();
    }

    #[test]
    fn signature_from_another_key_is_refused() {
        let signer = key(3);
        let other = key(4);
        let id = "sha256:1111111111111111111111111111111111111111111111111111111111111111";
        let sig = b64_encode(&signer.sign_bytes(id.as_bytes()));
        let err = verify_over_id(id, &sig, &other.verifying_key()).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::SignatureInvalid);
    }

    #[test]
    fn signature_over_a_different_id_is_refused() {
        let sk = key(5);
        let sig = b64_encode(&sk.sign_bytes(b"sha256:aa"));
        let err = verify_over_id("sha256:bb", &sig, &sk.verifying_key()).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::SignatureInvalid);
    }

    #[test]
    fn public_key_round_trips_through_base64() {
        let sk = key(6);
        let encoded = b64_encode(sk.verifying_key().as_bytes());
        assert_eq!(
            verifying_key_from_b64(&encoded).unwrap().as_bytes(),
            sk.verifying_key().as_bytes()
        );
    }

    #[test]
    fn short_key_is_refused() {
        let err = verifying_key_from_b64(&b64_encode(&[0u8; 16])).unwrap_err();
        assert_eq!(err.code(), crate::RegistryErrorCode::InvalidObject);
    }
}
