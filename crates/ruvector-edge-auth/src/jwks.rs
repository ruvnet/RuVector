//! JWK / JWKS handling and the caching key source (ADR-351 §5.1.5).

use crate::clock::Clock;
use crate::error::AuthError;
use core::cell::RefCell;
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
        Err(AuthError::NotImplemented("jwks::Jwk::to_verifying_key"))
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
    pub fn parse_usable(body: &[u8]) -> Result<BTreeMap<String, VerifyingKey>, AuthError> {
        let _ = body;
        Err(AuthError::NotImplemented("jwks::JwkSet::parse_usable"))
    }
}

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
    /// `GET url`. Implementations must not follow redirects to other hosts.
    async fn get(&self, url: &str) -> Result<HttpResponse, FetchError>;
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

/// Cache policy (ADR §5.1.5). The JWKS URL comes from config, never from
/// runtime discovery.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct JwksCachePolicy {
    /// Hardcoded JWKS URL (edge AS JWKS for edge tokens; upstream for CLI).
    pub url: String,
    /// In-isolate freshness window (default 600 s).
    pub fresh_ttl_secs: u64,
    /// Minimum spacing between unknown-`kid` refetches (default 30 s).
    pub refetch_min_interval_secs: u64,
    /// Serve the last good set on fetch failure for up to this long (24 h).
    pub stale_if_error_secs: u64,
    /// Maximum accepted JWKS body size in bytes.
    pub max_body_bytes: usize,
}

impl JwksCachePolicy {
    /// Policy with the ADR defaults for `url`.
    pub fn with_defaults(url: impl Into<String>) -> Self {
        JwksCachePolicy {
            url: url.into(),
            fresh_ttl_secs: 600,
            refetch_min_interval_secs: 30,
            stale_if_error_secs: 86_400,
            max_body_bytes: 64 * 1024,
        }
    }
}

#[derive(Debug, Default)]
struct CacheState {
    keys: BTreeMap<String, VerifyingKey>,
    fetched_at: Option<u64>,
    last_refetch_attempt: Option<u64>,
}

/// In-isolate JWKS cache implementing [`KeySource`] over an [`HttpFetch`].
/// Single-threaded (wasm isolates), hence `RefCell`.
pub struct JwksCache<F: HttpFetch, C: Clock> {
    fetch: F,
    clock: C,
    policy: JwksCachePolicy,
    state: RefCell<CacheState>,
}

impl<F: HttpFetch, C: Clock> JwksCache<F, C> {
    /// New, empty cache.
    pub fn new(fetch: F, clock: C, policy: JwksCachePolicy) -> Self {
        JwksCache {
            fetch,
            clock,
            policy,
            state: RefCell::new(CacheState::default()),
        }
    }

    /// The configured policy.
    pub fn policy(&self) -> &JwksCachePolicy {
        &self.policy
    }
}

impl<F: HttpFetch, C: Clock> KeySource for JwksCache<F, C> {
    /// Contract: fresh hit -> key; stale or miss -> fetch (single-flight, rate
    /// limited per `refetch_min_interval_secs` for unknown kids); fetch error
    /// -> last good set within `stale_if_error_secs`, else `KeysUnavailable`.
    /// Never holds the `RefCell` borrow across an `.await`.
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        let _ = (
            kid,
            &self.fetch,
            self.clock.now_unix(),
            self.state.borrow().fetched_at,
            self.state.borrow().last_refetch_attempt,
            self.state.borrow().keys.len(),
        );
        Err(AuthError::NotImplemented("jwks::JwksCache::verifying_key"))
    }
}
