//! Kalshi RSA-PSS-SHA256 request signing.
//!
//! Kalshi REST endpoints require three headers on every authenticated call:
//! - `KALSHI-ACCESS-KEY`: the API key UUID
//! - `KALSHI-ACCESS-TIMESTAMP`: unix time in milliseconds
//! - `KALSHI-ACCESS-SIGNATURE`: base64(RSA-PSS-SHA256(pem, ts || method || path))
//!
//! The canonical string is `format!("{ts}{method}{path}")` with no separators.

use std::sync::Arc;

use base64::{engine::general_purpose::STANDARD, Engine};
use rsa::pkcs1::DecodeRsaPrivateKey;
use rsa::pkcs8::DecodePrivateKey;
use rsa::pss::{SigningKey, VerifyingKey};
use rsa::signature::{Keypair, RandomizedSigner, SignatureEncoding, Verifier};
use rsa::traits::PublicKeyParts;
use rsa::RsaPrivateKey;
use sha2::Sha256;

use crate::{KalshiError, Result};

/// Signed headers for a single Kalshi REST request.
#[derive(Debug, Clone)]
pub struct SignedHeaders {
    pub access_key: String,
    pub timestamp_ms: String,
    pub signature_b64: String,
}

impl SignedHeaders {
    /// Apply this signature set onto a [`reqwest::RequestBuilder`].
    pub fn apply(self, rb: reqwest::RequestBuilder) -> reqwest::RequestBuilder {
        rb.header("KALSHI-ACCESS-KEY", self.access_key)
            .header("KALSHI-ACCESS-TIMESTAMP", self.timestamp_ms)
            .header("KALSHI-ACCESS-SIGNATURE", self.signature_b64)
    }
}

/// Kalshi request signer.
///
/// Internally holds an `Arc<SigningKey<Sha256>>` so `Clone` is O(1) — the
/// RSA private key is reference-counted across all clones. The API key is
/// also shared as `Arc<str>` to avoid per-request allocations.
#[derive(Clone)]
pub struct Signer {
    api_key: Arc<str>,
    signing_key: Arc<SigningKey<Sha256>>,
}

impl std::fmt::Debug for Signer {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        // Never leak the API key or key material in Debug output.
        f.debug_struct("Signer")
            .field("api_key_len", &self.api_key.len())
            .field("key_size_bits", &self.signing_key.as_ref().as_ref().size())
            .finish()
    }
}

impl Signer {
    /// Build a signer from a PEM string (accepts PKCS#1 or PKCS#8) and API key.
    pub fn from_pem(api_key: impl Into<String>, pem: &str) -> Result<Self> {
        let private_key = parse_rsa_pem(pem)?;
        let api_key: String = api_key.into();
        Ok(Self {
            api_key: Arc::from(api_key.into_boxed_str()),
            signing_key: Arc::new(SigningKey::<Sha256>::new(private_key)),
        })
    }

    /// Sign for the given method + path using the current wall-clock time.
    pub fn sign_now(&self, method: &str, path: &str) -> SignedHeaders {
        let ts = current_ts_ms();
        self.sign_with_ts(ts, method, path)
    }

    /// Sign with an explicit timestamp (deterministic — for tests/replay).
    pub fn sign_with_ts(&self, ts_ms: u64, method: &str, path: &str) -> SignedHeaders {
        let msg = canonical_string(ts_ms, method, path);
        let mut rng = rand::thread_rng();
        let sig = self.signing_key.sign_with_rng(&mut rng, msg.as_bytes());
        SignedHeaders {
            access_key: self.api_key.as_ref().to_string(),
            timestamp_ms: ts_ms.to_string(),
            signature_b64: STANDARD.encode(sig.to_bytes()),
        }
    }

    /// Derive the matching verifying key (useful for round-trip tests).
    pub fn verifying_key(&self) -> VerifyingKey<Sha256> {
        self.signing_key.verifying_key()
    }
}

fn canonical_string(ts_ms: u64, method: &str, path: &str) -> String {
    // Kalshi spec: concat ts, method (upper-case), path exactly as sent.
    format!("{ts_ms}{}{path}", method.to_uppercase())
}

fn current_ts_ms() -> u64 {
    use std::time::{SystemTime, UNIX_EPOCH};
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .expect("system clock before UNIX epoch")
        .as_millis() as u64
}

fn parse_rsa_pem(pem: &str) -> Result<RsaPrivateKey> {
    // Accept either `-----BEGIN RSA PRIVATE KEY-----` (PKCS#1) or
    // `-----BEGIN PRIVATE KEY-----` (PKCS#8).
    if pem.contains("BEGIN RSA PRIVATE KEY") {
        RsaPrivateKey::from_pkcs1_pem(pem)
            .map_err(|e| KalshiError::InvalidPem(format!("pkcs1: {e}")))
    } else {
        RsaPrivateKey::from_pkcs8_pem(pem)
            .map_err(|e| KalshiError::InvalidPem(format!("pkcs8: {e}")))
    }
}

/// Verify a signature against the public half. Test-only helper.
pub fn verify(
    vk: &VerifyingKey<Sha256>,
    ts_ms: u64,
    method: &str,
    path: &str,
    sig_b64: &str,
) -> Result<()> {
    use rsa::pss::Signature;
    let sig_bytes = STANDARD
        .decode(sig_b64)
        .map_err(|e| KalshiError::Signing(format!("base64: {e}")))?;
    let sig = Signature::try_from(sig_bytes.as_slice())
        .map_err(|e| KalshiError::Signing(format!("sig decode: {e}")))?;
    let msg = canonical_string(ts_ms, method, path);
    vk.verify(msg.as_bytes(), &sig)
        .map_err(|e| KalshiError::Signing(format!("verify: {e}")))
}

#[cfg(test)]
mod tests {
    use super::*;
    use rsa::pkcs1::EncodeRsaPrivateKey;

    /// Generate a fresh 2048-bit RSA key in PKCS#1 PEM form for tests.
    fn gen_pkcs1_pem() -> String {
        let mut rng = rand::thread_rng();
        let key = RsaPrivateKey::new(&mut rng, 2048).expect("keygen");
        key.to_pkcs1_pem(rsa::pkcs1::LineEnding::LF)
            .expect("encode pkcs1")
            .to_string()
    }

    #[test]
    fn signer_roundtrip_pkcs1() {
        let pem = gen_pkcs1_pem();
        let signer = Signer::from_pem("test-key-uuid", &pem).unwrap();

        let headers = signer.sign_with_ts(1_700_000_000_000, "GET", "/trade-api/v2/portfolio");
        assert_eq!(headers.access_key, "test-key-uuid");
        assert_eq!(headers.timestamp_ms, "1700000000000");
        assert!(!headers.signature_b64.is_empty());

        // Round-trip through the verifying key proves the signature is well-formed.
        verify(
            &signer.verifying_key(),
            1_700_000_000_000,
            "GET",
            "/trade-api/v2/portfolio",
            &headers.signature_b64,
        )
        .unwrap();
    }

    #[test]
    fn signer_rejects_tampered_message() {
        let pem = gen_pkcs1_pem();
        let signer = Signer::from_pem("k", &pem).unwrap();
        let headers = signer.sign_with_ts(42, "POST", "/orders");

        // Verifying against a different path must fail.
        let err = verify(
            &signer.verifying_key(),
            42,
            "POST",
            "/orders/tampered",
            &headers.signature_b64,
        );
        assert!(err.is_err(), "tampered path must not verify");
    }

    #[test]
    fn canonical_string_is_stable() {
        assert_eq!(
            canonical_string(1, "get", "/foo"),
            "1GET/foo",
            "method must be upper-cased",
        );
    }

    #[test]
    fn invalid_pem_is_rejected() {
        let err = Signer::from_pem("k", "not a pem").unwrap_err();
        match err {
            KalshiError::InvalidPem(_) => {}
            other => panic!("expected InvalidPem, got {other:?}"),
        }
    }

    #[test]
    fn debug_does_not_leak_key() {
        let pem = gen_pkcs1_pem();
        let signer = Signer::from_pem("super-secret-api-key-1234567890", &pem).unwrap();
        let dbg = format!("{signer:?}");
        assert!(!dbg.contains("super-secret-api-key"));
        assert!(!dbg.contains("BEGIN RSA"));
    }
}
