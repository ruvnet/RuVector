//! JWK / JWKS handling and the caching key source (ADR-351 §5.1.5).

use crate::error::AuthError;
use p256::ecdsa::VerifyingKey;
use serde::{Deserialize, Serialize};
use std::collections::BTreeMap;

/// A public EC P-256 JWK. Only `kty=EC`, `crv=P-256` keys are usable.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Jwk {
    /// Key type; must be `"EC"`.
    pub kty: String,
    /// Curve; must be `"P-256"`.
    pub crv: String,
    /// base64url X coordinate (32 bytes).
    pub x: String,
    /// base64url Y coordinate (32 bytes).
    pub y: String,
    /// Key id; must equal the RFC 7638 thumbprint.
    pub kid: String,
    /// Optional `alg`; if present must be `"ES256"`.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub alg: Option<String>,
    /// Optional `use`; if present must be `"sig"`.
    #[serde(default, rename = "use", skip_serializing_if = "Option::is_none")]
    pub use_: Option<String>,
}

impl Jwk {
    /// RFC 7638 thumbprint: base64url(SHA-256 of
    /// `{"crv":"P-256","kty":"EC","x":"..","y":".."}`), members in
    /// lexicographic order, no whitespace.
    pub fn thumbprint(&self) -> String {
        use sha2::{Digest, Sha256};
        let canonical = format!(
            r#"{{"crv":"{}","kty":"{}","x":"{}","y":"{}"}}"#,
            self.crv, self.kty, self.x, self.y
        );
        crate::jws::b64url_encode(&Sha256::digest(canonical.as_bytes()))
    }

    /// Build the verifying key.
    ///
    /// Contract: rejects non-EC/P-256 keys, `alg` other than ES256, `use`
    /// other than `sig`, coordinates that are not exactly 32 bytes, points not
    /// on the curve, and a `kid` that differs from [`Jwk::thumbprint`].
    pub fn to_verifying_key(&self) -> Result<VerifyingKey, AuthError> {
        if self.kty != "EC" || self.crv != "P-256" {
            return Err(AuthError::InvalidConfig("jwk kty/crv"));
        }
        if self.alg.as_deref().is_some_and(|a| a != "ES256") {
            return Err(AuthError::InvalidConfig("jwk alg"));
        }
        if self.use_.as_deref().is_some_and(|u| u != "sig") {
            return Err(AuthError::InvalidConfig("jwk use"));
        }
        let coord = |v: &str| -> Result<p256::FieldBytes, AuthError> {
            let bytes: [u8; 32] = crate::jws::b64url_decode(v)?
                .as_slice()
                .try_into()
                .map_err(|_| AuthError::InvalidConfig("jwk coordinate length"))?;
            Ok(p256::FieldBytes::from(bytes))
        };
        let (x, y) = (coord(&self.x)?, coord(&self.y)?);
        let point = p256::EncodedPoint::from_affine_coordinates(&x, &y, false);
        let key = VerifyingKey::from_encoded_point(&point)
            .map_err(|_| AuthError::InvalidConfig("jwk point not on curve"))?;
        if self.kid != self.thumbprint() {
            return Err(AuthError::InvalidConfig("jwk kid != thumbprint"));
        }
        Ok(key)
    }

    /// Export a verifying key as a JWK with `kid` = its thumbprint,
    /// `alg=ES256`, `use=sig` (used by the edge AS to publish its JWKS).
    pub fn from_verifying_key(key: &VerifyingKey) -> Jwk {
        let point = key.to_encoded_point(false);
        let x = point
            .x()
            .map(|b| crate::jws::b64url_encode(b))
            .unwrap_or_default();
        let y = point
            .y()
            .map(|b| crate::jws::b64url_encode(b))
            .unwrap_or_default();
        let mut jwk = Jwk {
            kty: "EC".into(),
            crv: "P-256".into(),
            x,
            y,
            kid: String::new(),
            alg: Some("ES256".into()),
            use_: Some("sig".into()),
        };
        jwk.kid = jwk.thumbprint();
        jwk
    }
}

/// A JWK Set document `{"keys":[...]}`.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct JwkSet {
    /// Member keys (unknown members are ignored by serde).
    pub keys: Vec<Jwk>,
}

impl JwkSet {
    /// Parse a JWKS body and keep only usable keys, indexed by `kid`.
    ///
    /// Contract: body size bounded by the caller; JSON must parse; keys that
    /// fail [`Jwk::to_verifying_key`] are dropped (not fatal); an empty
    /// result is [`AuthError::KeysUnavailable`].
    ///
    /// Entries that are not EC P-256 (e.g. RSA keys without `crv`) are
    /// skipped individually rather than failing the whole document. At most
    /// [`MAX_JWKS_KEYS`] entries are examined.
    pub fn parse_usable(body: &[u8]) -> Result<BTreeMap<String, VerifyingKey>, AuthError> {
        #[derive(Deserialize)]
        struct Doc {
            keys: Vec<serde_json::Value>,
        }
        if !crate::jws::starts_with_object(body) {
            return Err(AuthError::KeysUnavailable);
        }
        let doc: Doc = serde_json::from_slice(body).map_err(|_| AuthError::KeysUnavailable)?;
        let mut out = BTreeMap::new();
        for entry in doc.keys.into_iter().take(MAX_JWKS_KEYS) {
            let Ok(jwk) = serde_json::from_value::<Jwk>(entry) else {
                continue;
            };
            if let Ok(key) = jwk.to_verifying_key() {
                out.insert(jwk.kid, key);
            }
        }
        if out.is_empty() {
            return Err(AuthError::KeysUnavailable);
        }
        Ok(out)
    }
}

/// Upper bound on JWKS entries examined per document.
pub const MAX_JWKS_KEYS: usize = 32;

/// Minimal HTTP response for [`HttpFetch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct HttpResponse {
    /// HTTP status code.
    pub status: u16,
    /// Response body.
    pub body: Vec<u8>,
}

/// Transport failure for [`HttpFetch`].
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct FetchError(pub String);

/// Network port (the Worker implements it with `worker::Fetch`).
#[allow(async_fn_in_trait)]
pub trait HttpFetch {
    /// `GET url`, reading at most `max_body_bytes + 1` body bytes.
    ///
    /// Contract: implementations must not follow redirects to other hosts and
    /// must stop reading the body once it exceeds `max_body_bytes` (return
    /// the truncated `max_body_bytes + 1` bytes or a [`FetchError`]), so an
    /// oversized or unbounded response is never fully buffered in the
    /// isolate. Callers reject any body longer than `max_body_bytes`.
    async fn get(&self, url: &str, max_body_bytes: usize) -> Result<HttpResponse, FetchError>;
}

impl<T: HttpFetch + ?Sized> HttpFetch for &T {
    async fn get(&self, url: &str, max_body_bytes: usize) -> Result<HttpResponse, FetchError> {
        (**self).get(url, max_body_bytes).await
    }
}

/// Source of verifying keys by `kid`. Tests hand-mock this.
#[allow(async_fn_in_trait)]
pub trait KeySource {
    /// Resolve `kid` to a verifying key.
    ///
    /// Errors: [`AuthError::UnknownKid`] when the key set (after at most one
    /// permitted refresh) lacks `kid`; [`AuthError::KeysUnavailable`] when no
    /// key set has ever been obtained.
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError>;
}

impl<T: KeySource + ?Sized> KeySource for &T {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        (**self).verifying_key(kid).await
    }
}

/// Cache policy (ADR §5.1.5). The JWKS URL comes from config, never from
/// runtime discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwksCachePolicy {
    /// Hardcoded JWKS URL (edge AS JWKS for edge tokens; upstream for CLI).
    pub url: String,
    /// In-isolate freshness window (default 600 s).
    pub fresh_ttl_secs: u64,
    /// Minimum spacing between fetch attempts once a key set has been
    /// obtained, including unknown-`kid` refetches (default 30 s).
    pub refetch_min_interval_secs: u64,
    /// Minimum spacing between fetch attempts while no key set has ever been
    /// obtained (default 1 s), so one transient cold-start failure does not
    /// answer 503 for the full `refetch_min_interval_secs`.
    pub cold_retry_interval_secs: u64,
    /// Serve the last good set on fetch failure while it is younger than
    /// this (24 h).
    pub stale_if_error_secs: u64,
    /// Maximum accepted JWKS body size in bytes.
    pub max_body_bytes: usize,
    /// Optional pinned `kid` allowlist. A `kid` outside it is
    /// [`AuthError::UnknownKid`] with no fetch, and fetched keys outside it
    /// are discarded. `None` (default) trusts every thumbprint-valid key in
    /// the configured JWKS, which is right for edge keys (ADR §5.4.5). For the
    /// upstream path the authoritative pin is
    /// [`crate::UpstreamFirstPartyPolicy::accepted_kids`], enforced by the
    /// verifier whatever this cache trusts; setting the same list here also
    /// keeps foreign keys out of the cache.
    pub accepted_kids: Option<Vec<String>>,
}

impl JwksCachePolicy {
    /// Policy with the ADR defaults for `url`.
    pub fn with_defaults(url: impl Into<String>) -> Self {
        JwksCachePolicy {
            url: url.into(),
            fresh_ttl_secs: 600,
            refetch_min_interval_secs: 30,
            cold_retry_interval_secs: 1,
            stale_if_error_secs: 86_400,
            max_body_bytes: 64 * 1024,
            accepted_kids: None,
        }
    }

    /// Pin the accepted `kid` set (builder).
    pub fn with_accepted_kids<I, S>(mut self, kids: I) -> Self
    where
        I: IntoIterator<Item = S>,
        S: Into<String>,
    {
        self.accepted_kids = Some(kids.into_iter().map(Into::into).collect());
        self
    }

    fn kid_accepted(&self, kid: &str) -> bool {
        self.accepted_kids
            .as_ref()
            .map_or(true, |list| list.iter().any(|k| k == kid))
    }
}

mod cache;
pub use cache::JwksCache;

#[cfg(test)]
mod tests;
