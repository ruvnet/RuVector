//! # ruvector-edge-auth
//!
//! Resource-server authentication for ruvector edge (ADR-351 §5), amended by
//! the 2026-09-29 decision: resource servers accept **only** access tokens
//! minted by the edge authorization server `ruvector-edge-auth`, whose `aud`
//! is the exact canonical https URL of the protected resource
//! ([`ResourceUrl`]). Upstream `auth.cognitum.one` tokens for the first-party
//! CLI (ADR §5.2) are accepted only when
//! [`AudiencePolicy::upstream`] is explicitly configured.
//!
//! Pure Rust with no `worker`/`wasm-bindgen` dependency. Time and network are
//! reached only through the [`Clock`] and [`HttpFetch`]/[`KeySource`] traits so
//! every rule is unit-testable natively with mocks. Never calls `std::time`.
//!
//! Module map (ADR §8): [`jws`] compact JWS parsing + ES256 verify, [`jwks`]
//! JWK parsing, RFC 7638 thumbprints and the caching key source, [`claims`]
//! claim validation, [`audience`] audience policy, [`scopes`] scope ->
//! capability and route tables, [`prm`] RFC 9728 metadata and
//! `WWW-Authenticate`, [`verifier`] the end-to-end bearer verifier.

#![forbid(unsafe_code)]
#![warn(missing_docs)]

pub mod audience;
pub mod claims;
pub mod clock;
pub mod error;
pub mod jwks;
pub mod jws;
pub mod prm;
pub mod resource;
pub mod scopes;
pub mod subject;
pub mod verifier;

#[cfg(test)]
mod test_support;

pub use audience::{AudiencePolicy, UpstreamFirstPartyPolicy};
pub use claims::{Audience, ClaimsPolicy, RawClaims, TokenKind, VerifiedClaims};
pub use clock::Clock;
pub use error::AuthError;
pub use jwks::{
    FetchError, HttpFetch, HttpResponse, Jwk, JwkSet, JwksCache, JwksCachePolicy, KeySource,
};
pub use jws::{bearer_token, parse_compact, verify_es256, CompactJws, JwsHeader};
pub use prm::ProtectedResourceMetadata;
pub use resource::ResourceUrl;
pub use scopes::{Capability, CapabilitySet, Method, RouteRequirement, RouteSurface};
pub use verifier::Verifier;

/// Re-export so callers name the same key type the verifier uses.
pub use p256::ecdsa::VerifyingKey;
