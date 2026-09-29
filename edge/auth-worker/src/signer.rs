//! ES256 signing key from the `EDGE_AUTH_SIGNING_JWK` Worker secret.
//!
//! The secret is either one private EC P-256 JWK
//! (`{"kty":"EC","crv":"P-256","d":..,"x":..,"y":..}`) or a JWK Set
//! `{"keys":[active, previous...]}` whose **first** key is private and active
//! and whose remaining keys (public or private) are only published in the
//! JWKS during rotation. `x`/`y`/`kid`, when present, must match the key
//! derived from `d`. Any failure leaves the signer keyless, so every
//! signature fails closed. Error values never contain key material.

use p256::ecdsa::{SigningKey, VerifyingKey};
use ruvector_edge_auth::jws::b64url_decode;
use ruvector_edge_auth::Jwk;
use ruvector_edge_authz::{Signer, StoreError};
use serde::Deserialize;
use zeroize::Zeroize;

/// Worker secret holding the private JWK (or JWK Set).
pub const SIGNING_KEY_SECRET: &str = "EDGE_AUTH_SIGNING_JWK";
/// Maximum secret size accepted.
pub const MAX_SECRET_BYTES: usize = 8 * 1024;
/// Maximum keys in the set (active + previous).
pub const MAX_KEYS: usize = 4;

/// Why the secret was refused (static, never echoes the secret).
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct KeyError(pub &'static str);

#[derive(Deserialize)]
struct PrivateJwk {
    kty: String,
    crv: String,
    #[serde(default)]
    d: Option<String>,
    #[serde(default)]
    x: Option<String>,
    #[serde(default)]
    y: Option<String>,
    #[serde(default)]
    kid: Option<String>,
    #[serde(default)]
    alg: Option<String>,
    #[serde(default, rename = "use")]
    use_: Option<String>,
}

impl Drop for PrivateJwk {
    fn drop(&mut self) {
        if let Some(d) = self.d.as_mut() {
            d.zeroize();
        }
    }
}

#[derive(Deserialize)]
#[serde(untagged)]
enum SecretDoc {
    Set { keys: Vec<PrivateJwk> },
    One(PrivateJwk),
}

fn coord(v: &Option<String>) -> Result<Option<[u8; 32]>, KeyError> {
    let Some(s) = v else { return Ok(None) };
    let bytes = b64url_decode(s).map_err(|_| KeyError("bad coordinate encoding"))?;
    <[u8; 32]>::try_from(bytes.as_slice())
        .map(Some)
        .map_err(|_| KeyError("coordinate must be 32 bytes"))
}

/// Parse one JWK into (optional signing key, verifying key).
fn parse_jwk(jwk: &PrivateJwk) -> Result<(Option<SigningKey>, VerifyingKey), KeyError> {
    if jwk.kty != "EC" || jwk.crv != "P-256" {
        return Err(KeyError("key must be EC P-256"));
    }
    if jwk.alg.as_deref().is_some_and(|a| a != "ES256") {
        return Err(KeyError("alg must be ES256"));
    }
    if jwk.use_.as_deref().is_some_and(|u| u != "sig") {
        return Err(KeyError("use must be sig"));
    }
    let (x, y) = (coord(&jwk.x)?, coord(&jwk.y)?);
    let signing = match &jwk.d {
        Some(d) => {
            let mut raw = b64url_decode(d).map_err(|_| KeyError("bad d encoding"))?;
            let key = <[u8; 32]>::try_from(raw.as_slice())
                .map_err(|_| KeyError("d must be 32 bytes"))
                .and_then(|b| {
                    SigningKey::from_bytes(&b.into()).map_err(|_| KeyError("d out of range"))
                });
            raw.zeroize();
            Some(key?)
        }
        None => None,
    };
    let verifying = match (&signing, x, y) {
        (Some(sk), _, _) => {
            let vk = *sk.verifying_key();
            let point = vk.to_encoded_point(false);
            let same = |given: Option<[u8; 32]>, derived: Option<&[u8]>| {
                given.map_or(true, |g| derived == Some(&g[..]))
            };
            if !same(x, point.x().map(|p| &p[..])) || !same(y, point.y().map(|p| &p[..])) {
                return Err(KeyError("x/y do not match d"));
            }
            vk
        }
        (None, Some(x), Some(y)) => {
            let mut sec1 = [0u8; 65];
            sec1[0] = 0x04;
            sec1[1..33].copy_from_slice(&x);
            sec1[33..].copy_from_slice(&y);
            VerifyingKey::from_sec1_bytes(&sec1).map_err(|_| KeyError("point not on curve"))?
        }
        (None, _, _) => return Err(KeyError("public key needs x and y")),
    };
    let expected_kid = Jwk::from_verifying_key(&verifying).kid;
    if jwk.kid.as_deref().is_some_and(|k| k != expected_kid) {
        return Err(KeyError("kid is not the RFC 7638 thumbprint"));
    }
    Ok((signing, verifying))
}

/// The active signing key plus every key to publish.
pub struct KeyRing {
    active: SigningKey,
    active_kid: String,
    published: Vec<VerifyingKey>,
}

impl KeyRing {
    /// Parse the secret document.
    pub fn parse(secret: &str) -> Result<Self, KeyError> {
        if secret.len() > MAX_SECRET_BYTES {
            return Err(KeyError("secret too large"));
        }
        let doc: SecretDoc =
            serde_json::from_str(secret).map_err(|_| KeyError("secret is not a JWK or JWKS"))?;
        let keys = match doc {
            SecretDoc::Set { keys } => keys,
            SecretDoc::One(k) => vec![k],
        };
        if keys.is_empty() || keys.len() > MAX_KEYS {
            return Err(KeyError("expected 1..=4 keys"));
        }
        let mut active = None;
        let mut published = Vec::with_capacity(keys.len());
        for (i, jwk) in keys.iter().enumerate() {
            let (sk, vk) = parse_jwk(jwk)?;
            if i == 0 {
                active = Some(sk.ok_or(KeyError("first key must be private"))?);
            }
            if !published.contains(&vk) {
                published.push(vk);
            }
        }
        let active = active.ok_or(KeyError("no active key"))?;
        let active_kid = Jwk::from_verifying_key(active.verifying_key()).kid;
        Ok(KeyRing {
            active,
            active_kid,
            published,
        })
    }

    /// Keys for the JWKS document (active first).
    pub fn published(&self) -> &[VerifyingKey] {
        &self.published
    }
}

/// [`Signer`] over an optional [`KeyRing`]; keyless means fail closed.
pub struct EnvSigner {
    ring: Option<KeyRing>,
}

impl EnvSigner {
    /// Build from the raw secret value (absent or invalid -> keyless).
    pub fn from_secret(secret: Option<&str>) -> Self {
        EnvSigner {
            ring: secret.and_then(|s| KeyRing::parse(s).ok()),
        }
    }

    /// Whether a usable active key is loaded.
    #[cfg(test)]
    pub fn is_ready(&self) -> bool {
        self.ring.is_some()
    }

    /// Public keys to publish (empty when keyless).
    pub fn public_keys(&self) -> Vec<VerifyingKey> {
        self.ring
            .as_ref()
            .map(|r| r.published().to_vec())
            .unwrap_or_default()
    }
}

impl Signer for EnvSigner {
    fn kid(&self) -> String {
        self.ring
            .as_ref()
            .map(|r| r.active_kid.clone())
            .unwrap_or_default()
    }

    fn sign_es256(&self, signing_input: &[u8]) -> Result<[u8; 64], StoreError> {
        use p256::ecdsa::signature::Signer as _;
        let ring = self
            .ring
            .as_ref()
            .ok_or_else(|| StoreError("signing key unavailable".into()))?;
        let sig: p256::ecdsa::Signature = ring.active.sign(signing_input);
        let mut out = [0u8; 64];
        out.copy_from_slice(&sig.to_bytes());
        Ok(out)
    }

    /// The published keys (active + previous): RFC 8693 subject tokens
    /// signed by any key still in the JWKS verify.
    fn verifying_keys(&self) -> Vec<VerifyingKey> {
        self.public_keys()
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use ruvector_edge_auth::jws::b64url_encode;

    /// Deterministic test key (never a real secret).
    pub(crate) fn test_key(seed: u8) -> SigningKey {
        let mut b = [seed; 32];
        b[0] = 0x01;
        SigningKey::from_bytes(&b.into()).unwrap()
    }

    /// Private JWK JSON for `key`.
    pub(crate) fn private_jwk(key: &SigningKey, with_kid: bool) -> String {
        let pubjwk = Jwk::from_verifying_key(key.verifying_key());
        let d = b64url_encode(&key.to_bytes());
        let kid = if with_kid {
            format!(r#","kid":"{}""#, pubjwk.kid)
        } else {
            String::new()
        };
        format!(
            r#"{{"kty":"EC","crv":"P-256","d":"{d}","x":"{}","y":"{}"{kid}}}"#,
            pubjwk.x, pubjwk.y
        )
    }

    #[test]
    fn signs_verifiable_fixed_width_es256() {
        use p256::ecdsa::signature::Verifier as _;
        let key = test_key(7);
        let s = EnvSigner::from_secret(Some(&private_jwk(&key, true)));
        assert!(s.is_ready());
        assert_eq!(s.kid(), Jwk::from_verifying_key(key.verifying_key()).kid);
        let sig = s.sign_es256(b"header.payload").unwrap();
        let sig = p256::ecdsa::Signature::from_slice(&sig).unwrap();
        key.verifying_key().verify(b"header.payload", &sig).unwrap();
        // RFC 6979: deterministic.
        assert_eq!(
            s.sign_es256(b"header.payload").unwrap(),
            s.sign_es256(b"header.payload").unwrap()
        );
    }

    #[test]
    fn keyless_signer_fails_closed() {
        for secret in [None, Some(""), Some("not json"), Some(r#"{"keys":[]}"#)] {
            let s = EnvSigner::from_secret(secret);
            assert!(!s.is_ready());
            assert_eq!(s.kid(), "");
            assert!(s.sign_es256(b"x").is_err());
            assert!(s.public_keys().is_empty());
        }
    }

    #[test]
    fn rejects_mismatched_or_malformed_keys() {
        let a = test_key(1);
        let b = test_key(2);
        let pa = Jwk::from_verifying_key(a.verifying_key());
        let pb = Jwk::from_verifying_key(b.verifying_key());
        let d = b64url_encode(&a.to_bytes());
        let cases = [
            // x/y from another key.
            format!(
                r#"{{"kty":"EC","crv":"P-256","d":"{d}","x":"{}","y":"{}"}}"#,
                pb.x, pb.y
            ),
            // kid not the thumbprint.
            format!(r#"{{"kty":"EC","crv":"P-256","d":"{d}","kid":"nope"}}"#),
            // wrong curve / alg / use.
            format!(r#"{{"kty":"EC","crv":"P-384","d":"{d}"}}"#),
            format!(r#"{{"kty":"EC","crv":"P-256","d":"{d}","alg":"RS256"}}"#),
            format!(r#"{{"kty":"EC","crv":"P-256","d":"{d}","use":"enc"}}"#),
            // short d; zero scalar.
            r#"{"kty":"EC","crv":"P-256","d":"AAAA"}"#.to_string(),
            format!(
                r#"{{"kty":"EC","crv":"P-256","d":"{}"}}"#,
                b64url_encode(&[0u8; 32])
            ),
            // first key public only.
            format!(
                r#"{{"keys":[{{"kty":"EC","crv":"P-256","x":"{}","y":"{}"}}]}}"#,
                pa.x, pa.y
            ),
        ];
        for c in &cases {
            assert!(KeyRing::parse(c).is_err(), "accepted: {c}");
        }
        assert!(KeyRing::parse(&"x".repeat(MAX_SECRET_BYTES + 1)).is_err());
    }

    #[test]
    fn jwks_rotation_publishes_active_then_previous() {
        let a = test_key(3);
        let b = test_key(4);
        let pb = Jwk::from_verifying_key(b.verifying_key());
        let doc = format!(
            r#"{{"keys":[{},{{"kty":"EC","crv":"P-256","x":"{}","y":"{}","kid":"{}"}}]}}"#,
            private_jwk(&a, false),
            pb.x,
            pb.y,
            pb.kid
        );
        let s = EnvSigner::from_secret(Some(&doc));
        assert_eq!(
            s.public_keys(),
            vec![*a.verifying_key(), *b.verifying_key()]
        );
        assert_eq!(s.kid(), Jwk::from_verifying_key(a.verifying_key()).kid);
    }
}
