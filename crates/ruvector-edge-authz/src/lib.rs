//! # ruvector-edge-authz
//!
//! Pure-Rust core of the ruvector edge **authorization server**
//! (`ruvector-edge-auth` Worker). Per the 2026-09-29 decision superseding
//! ADR-351 §5.2: `auth.cognitum.one` mints `aud = client_id` and ignores RFC
//! 8707, so the edge runs its own AS that
//!
//! 1. federates user login upstream to `auth.cognitum.one` (authorization
//!    code + PKCE S256; upstream tokens verified against its JWKS, ES256)
//!    — [`federation`];
//! 2. registers public clients dynamically (RFC 7591, `none` auth, https or
//!    loopback redirects only) — [`client`];
//! 3. runs authorization-code + PKCE S256 (only) with one-time codes —
//!    [`authorize`], [`pkce`], [`code`];
//! 4. mints its **own** ES256 access tokens (RFC 9068 `at+jwt`) whose `aud`
//!    is the exact canonical resource URL from an allowlist — [`token`],
//!    [`resource`];
//! 5. rotates refresh tokens with family reuse detection — [`refresh`];
//! 6. supports RFC 7009 revocation — [`revoke`];
//! 7. publishes RFC 8414 metadata and its JWKS — [`metadata`];
//! 8. lets operator-registered confidential adapter clients
//!    (`private_key_jwt`, RFC 7523) exchange a user's adapter-resource token
//!    for a gateway `…/v1` token (RFC 8693) — [`confidential`], [`exchange`].
//!
//! All I/O is behind the sync ports in [`ports`] (clock, RNG, signer, stores),
//! which the Worker implements over Durable Object SQLite and which tests mock.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod authorize;
pub mod client;
pub mod code;
pub mod confidential;
pub mod error;
pub mod exchange;
pub mod federation;
pub mod grant;
pub mod metadata;
pub mod params;
pub mod pkce;
pub mod ports;
pub mod refresh;
pub mod resource;
pub mod revoke;
pub mod token;

#[cfg(test)]
mod testing;
#[cfg(test)]
mod tests;

pub use authorize::AuthorizeError;
pub use confidential::ConfidentialClients;
pub use error::{OAuthError, OAuthErrorCode};
pub use grant::TokenEndpoint;
pub use ports::{
    AssertionReplayStore, ClientStore, CodeStore, FederationStore, RefreshStore, Rng, Signer,
    StoreError,
};
pub use resource::ResourceAllowlist;
pub use ruvector_edge_auth::{Clock, ResourceUrl};

/// Placeholder printed by the manual `Debug` impls of secret-bearing types
/// (tokens, codes, verifiers, browser secrets are never logged).
pub(crate) const REDACTED: &str = "<redacted>";

/// Hash a bearer secret (code / refresh token) for storage. Stores never hold
/// plaintext secrets: raw SHA-256 digest (32 bytes).
pub fn secret_hash(secret: &str) -> [u8; 32] {
    use sha2::{Digest, Sha256};
    Sha256::digest(secret.as_bytes()).into()
}

/// Generate a URL-safe random secret of `n_bytes` entropy (base64url, no
/// padding) from `rng`. RNG failure is an error, never a weak secret.
pub fn random_secret<R: Rng + ?Sized>(rng: &R, n_bytes: usize) -> Result<String, StoreError> {
    let mut buf = vec![0u8; n_bytes];
    rng.fill(&mut buf)?;
    Ok(ruvector_edge_auth::jws::b64url_encode(&buf))
}
