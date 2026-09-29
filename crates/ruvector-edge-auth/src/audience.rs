//! Audience policy (supersedes ADR-351 §5.2 per the 2026-09-29 decision).
//!
//! Default: accept only tokens whose `iss` is the edge AS and whose single
//! `aud` is byte-equal to this resource's canonical [`ResourceUrl`]. The
//! upstream first-party path (ADR §5.2 `FIRST_PARTY_AUDS`) exists only when
//! [`AudiencePolicy::upstream`] is `Some`, which must be an explicit config
//! flag in the Worker. No prefix matching anywhere.

use crate::claims::{Audience, TokenKind};
use crate::error::AuthError;
use crate::resource::ResourceUrl;

/// Opt-in acceptance of upstream `auth.cognitum.one` tokens.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamFirstPartyPolicy {
    /// Exact upstream issuer, `https://auth.cognitum.one`.
    pub issuer: String,
    /// Exact first-party client ids accepted as `aud` (never `dcr-` prefixes).
    pub first_party_auds: Vec<String>,
}

/// Audience policy for one protected resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudiencePolicy {
    /// Exact edge AS issuer URL.
    pub edge_issuer: String,
    /// This resource; edge tokens must carry exactly this `aud`.
    pub resource: ResourceUrl,
    /// `None` (default) disables the upstream path entirely.
    pub upstream: Option<UpstreamFirstPartyPolicy>,
}

impl AudiencePolicy {
    /// Edge-only policy (the default posture).
    pub fn edge_only(edge_issuer: impl Into<String>, resource: ResourceUrl) -> Self {
        AudiencePolicy {
            edge_issuer: edge_issuer.into(),
            resource,
            upstream: None,
        }
    }

    /// Decide the token kind from the unverified-but-parsed `iss` and header
    /// `typ` so the verifier can select the key source and claims policy.
    ///
    /// Contract: `iss == edge_issuer` and header `typ == "at+jwt"` ->
    /// `EdgeIssued`; `iss == upstream.issuer` (only if configured) and header
    /// `typ` absent or `JWT` -> `UpstreamFirstParty`; anything else ->
    /// [`AuthError::WrongIssuer`] / [`AuthError::BadTyp`]. The result is
    /// re-checked after signature verification.
    pub fn classify(
        &self,
        iss: Option<&str>,
        header_typ: Option<&str>,
    ) -> Result<TokenKind, AuthError> {
        let _ = (iss, header_typ);
        Err(AuthError::NotImplemented("audience::classify"))
    }

    /// Exact audience check for `kind`.
    ///
    /// Contract: `EdgeIssued` -> `aud.single() == Some(resource.as_str())`
    /// (arrays of length != 1 rejected); `UpstreamFirstParty` -> single `aud`
    /// in `first_party_auds`. Failure is [`AuthError::AudienceNotAllowed`].
    pub fn check_audience(&self, kind: TokenKind, aud: Option<&Audience>) -> Result<(), AuthError> {
        let _ = (kind, aud);
        Err(AuthError::NotImplemented("audience::check_audience"))
    }
}
