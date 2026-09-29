//! Audience policy (supersedes ADR-351 §5.2 per the 2026-09-29 decision).
//!
//! Default: accept only tokens whose `iss` is the edge AS and whose single
//! `aud` is byte-equal to this resource's canonical [`ResourceUrl`]. The
//! upstream first-party path (ADR §5.5 `FIRST_PARTY_AUDS`) exists only when
//! [`AudiencePolicy::upstream`] is `Some`, which must be an explicit config
//! flag in the Worker. No prefix matching anywhere.

use crate::claims::{Audience, TokenKind};
use crate::error::AuthError;
use crate::resource::ResourceUrl;

/// Opt-in acceptance of upstream `auth.cognitum.one` tokens (ADR §5.5).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamFirstPartyPolicy {
    /// Exact upstream issuer, `https://auth.cognitum.one`.
    pub issuer: String,
    /// Exact first-party client ids accepted as `aud` (never `dcr-` prefixes).
    pub first_party_auds: Vec<String>,
    /// Compiled `ACCEPTED_UPSTREAM_KIDS` pin (ADR §5.5 / §5.6 step 3). Must be
    /// non-empty: an empty pin fails closed with
    /// [`AuthError::InvalidConfig`], and a header `kid` outside it is
    /// [`AuthError::UnknownKid`] before any key lookup or fetch. Upstream
    /// key rotation therefore needs a redeploy. Edge `kid`s are never pinned.
    pub accepted_kids: Vec<String>,
}

/// Audience policy for one protected resource.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct AudiencePolicy {
    /// Exact edge AS issuer URL.
    pub edge_issuer: String,
    /// This resource; edge tokens must carry exactly this `aud`.
    pub resource: ResourceUrl,
    /// The other edge resources of this gateway (e.g. `/v1` when this is
    /// `/v1/mcp`). An edge token for one of them is `403
    /// audience_not_allowed`; any other audience is `401` (ADR §5.4.7).
    pub sibling_resources: Vec<ResourceUrl>,
    /// `None` (default) disables the upstream path entirely. Set it only on
    /// the `/v1` REST resource, never on `/v1/mcp` or `/v1/ops` (ADR §5.5;
    /// tenancy also refuses upstream tokens off the REST surface).
    pub upstream: Option<UpstreamFirstPartyPolicy>,
}

impl AudiencePolicy {
    /// Edge-only policy (the default posture) with no sibling resources.
    pub fn edge_only(edge_issuer: impl Into<String>, resource: ResourceUrl) -> Self {
        AudiencePolicy {
            edge_issuer: edge_issuer.into(),
            resource,
            sibling_resources: Vec::new(),
            upstream: None,
        }
    }

    /// Declare the gateway's other edge resources (builder). `resource`
    /// itself is ignored if listed.
    pub fn with_siblings<I: IntoIterator<Item = ResourceUrl>>(mut self, siblings: I) -> Self {
        let own = self.resource.clone();
        self.sibling_resources = siblings.into_iter().filter(|r| *r != own).collect();
        self
    }

    /// Header `typ` value required on edge-issued access tokens (RFC 9068).
    pub const EDGE_HEADER_TYP: &'static str = "at+jwt";

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
        let iss = iss.ok_or(AuthError::WrongIssuer)?;
        if iss == self.edge_issuer {
            return match header_typ {
                Some(t) if is_at_jwt(t) => Ok(TokenKind::EdgeIssued),
                _ => Err(AuthError::BadTyp),
            };
        }
        match &self.upstream {
            Some(up) if iss == up.issuer => match header_typ {
                None => Ok(TokenKind::UpstreamFirstParty),
                Some(t) if t.eq_ignore_ascii_case("JWT") => Ok(TokenKind::UpstreamFirstParty),
                Some(_) => Err(AuthError::BadTyp),
            },
            _ => Err(AuthError::WrongIssuer),
        }
    }

    /// Upstream `kid` pin (ADR §5.5), checked before any key lookup.
    ///
    /// Contract: upstream path not configured -> [`AuthError::WrongIssuer`];
    /// empty `accepted_kids` -> [`AuthError::InvalidConfig`] (fail closed);
    /// `kid` not in the pin -> [`AuthError::UnknownKid`].
    pub fn check_upstream_kid(&self, kid: &str) -> Result<(), AuthError> {
        let up = self.upstream.as_ref().ok_or(AuthError::WrongIssuer)?;
        if up.accepted_kids.is_empty() {
            return Err(AuthError::InvalidConfig("upstream accepted_kids empty"));
        }
        if up.accepted_kids.iter().any(|k| k == kid) {
            Ok(())
        } else {
            Err(AuthError::UnknownKid)
        }
    }

    /// Exact audience check for `kind` (ADR §5.4.7).
    ///
    /// Contract: `aud` must be a single JSON **string**; arrays of any length
    /// and absent `aud` are [`AuthError::InvalidClaim`]`("aud")` (401).
    /// `EdgeIssued`: byte-equal to `resource` -> Ok; byte-equal to a
    /// `sibling_resources` entry -> [`AuthError::AudienceNotAllowed`] (403,
    /// "an edge token for the other resource"); anything else -> 401
    /// `InvalidClaim("aud")`. `UpstreamFirstParty`: in `first_party_auds`
    /// (only when upstream is configured) -> Ok, else 401
    /// `InvalidClaim("aud")`. No prefix matching.
    pub fn check_audience(&self, kind: TokenKind, aud: Option<&Audience>) -> Result<(), AuthError> {
        let aud = match aud {
            Some(Audience::One(a)) => a.as_str(),
            None | Some(Audience::Many(_)) => return Err(AuthError::InvalidClaim("aud")),
        };
        match kind {
            TokenKind::EdgeIssued if aud == self.resource.as_str() => Ok(()),
            TokenKind::EdgeIssued if self.sibling_resources.iter().any(|r| r.as_str() == aud) => {
                Err(AuthError::AudienceNotAllowed)
            }
            TokenKind::UpstreamFirstParty
                if self
                    .upstream
                    .as_ref()
                    .is_some_and(|up| up.first_party_auds.iter().any(|a| a == aud)) =>
            {
                Ok(())
            }
            _ => Err(AuthError::InvalidClaim("aud")),
        }
    }
}

/// RFC 9068 §4: `at+jwt` or `application/at+jwt`, case-insensitive.
fn is_at_jwt(typ: &str) -> bool {
    let t = typ.strip_prefix("application/").unwrap_or(typ);
    t.eq_ignore_ascii_case(AudiencePolicy::EDGE_HEADER_TYP)
}

#[cfg(test)]
mod tests;
