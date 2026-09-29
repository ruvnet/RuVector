//! Claims model and validation (ADR-351 §5.1.6, amended by the edge-AS
//! decision).

use crate::error::AuthError;
use serde::Deserialize;

/// Which authorization server minted the token.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum TokenKind {
    /// Minted by `ruvector-edge-auth`; `aud` is the exact resource URL.
    EdgeIssued,
    /// Minted by `auth.cognitum.one` for the first-party CLI; `aud` is a
    /// client_id on the explicit allowlist. Only accepted when configured.
    UpstreamFirstParty,
}

/// `aud` as it appears on the wire: a string or an array of strings.
#[derive(Debug, Clone, PartialEq, Eq, Deserialize)]
#[serde(untagged)]
pub enum Audience {
    /// Single audience.
    One(String),
    /// Array form. Edge-issued tokens must carry exactly one element.
    Many(Vec<String>),
}

impl Audience {
    /// The single audience value, if the claim holds exactly one.
    pub fn single(&self) -> Option<&str> {
        match self {
            Audience::One(s) => Some(s),
            Audience::Many(v) if v.len() == 1 => Some(&v[0]),
            Audience::Many(_) => None,
        }
    }
}

/// Unvalidated payload as decoded from JSON. Unknown claims are ignored.
#[derive(Debug, Clone, Default, PartialEq, Eq, Deserialize)]
#[allow(missing_docs)]
pub struct RawClaims {
    pub iss: Option<String>,
    pub aud: Option<Audience>,
    pub sub: Option<String>,
    pub exp: Option<u64>,
    pub iat: Option<u64>,
    pub nbf: Option<u64>,
    pub jti: Option<String>,
    pub scope: Option<String>,
    pub client_id: Option<String>,
    pub org_id: Option<String>,
    pub workspace_id: Option<String>,
    /// Upstream `typ` *claim* (`"access"` / `"refresh"`), not the header.
    pub typ: Option<String>,
    pub nonce: Option<String>,
    pub exchanged: Option<bool>,
    pub setup: Option<bool>,
    pub workload: Option<bool>,
}

/// Per-kind claim rules.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimsPolicy {
    /// Exact required `iss`.
    pub issuer: String,
    /// Clock skew tolerance (default 60 s).
    pub skew_secs: u64,
    /// Maximum `exp - iat` (default 3600 s).
    pub max_lifetime_secs: u64,
    /// Required value of the `typ` claim if present (upstream: `"access"`).
    pub typ_claim: Option<String>,
}

impl ClaimsPolicy {
    /// ADR defaults for `issuer`.
    pub fn with_defaults(issuer: impl Into<String>) -> Self {
        ClaimsPolicy {
            issuer: issuer.into(),
            skew_secs: 60,
            max_lifetime_secs: 3600,
            typ_claim: None,
        }
    }
}

/// Claims that passed signature, issuer, audience and time checks. Fields are
/// private: the only constructors are [`validate`] (and `for_tests` behind the
/// `test-util` feature), so `TenantContext::from_verified` can trust them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VerifiedClaims {
    kind: TokenKind,
    iss: String,
    aud: String,
    sub: String,
    client_id: String,
    org_id: Option<String>,
    workspace_id: Option<String>,
    scopes: Vec<String>,
    jti: Option<String>,
    nonce: Option<String>,
    iat: u64,
    exp: u64,
}

#[allow(missing_docs)]
impl VerifiedClaims {
    pub fn kind(&self) -> TokenKind {
        self.kind
    }
    pub fn iss(&self) -> &str {
        &self.iss
    }
    pub fn aud(&self) -> &str {
        &self.aud
    }
    pub fn sub(&self) -> &str {
        &self.sub
    }
    pub fn client_id(&self) -> &str {
        &self.client_id
    }
    pub fn org_id(&self) -> Option<&str> {
        self.org_id.as_deref()
    }
    pub fn workspace_id(&self) -> Option<&str> {
        self.workspace_id.as_deref()
    }
    pub fn scopes(&self) -> &[String] {
        &self.scopes
    }
    pub fn jti(&self) -> Option<&str> {
        self.jti.as_deref()
    }
    pub fn nonce(&self) -> Option<&str> {
        self.nonce.as_deref()
    }
    pub fn iat(&self) -> u64 {
        self.iat
    }
    pub fn exp(&self) -> u64 {
        self.exp
    }

    /// Test constructor (feature `test-util`). Never enable in release builds.
    #[cfg(any(test, feature = "test-util"))]
    #[allow(clippy::too_many_arguments)]
    pub fn for_tests(
        kind: TokenKind,
        iss: &str,
        aud: &str,
        sub: &str,
        client_id: &str,
        org_id: Option<&str>,
        workspace_id: Option<&str>,
        scopes: &[&str],
        iat: u64,
        exp: u64,
    ) -> Self {
        VerifiedClaims {
            kind,
            iss: iss.into(),
            aud: aud.into(),
            sub: sub.into(),
            client_id: client_id.into(),
            org_id: org_id.map(Into::into),
            workspace_id: workspace_id.map(Into::into),
            scopes: scopes.iter().map(|s| (*s).to_string()).collect(),
            jti: None,
            nonce: None,
            iat,
            exp,
        }
    }
}

/// Validate time claims against `now`.
///
/// Contract: `exp` and `iat` required; `exp > now - skew`; `iat <= now +
/// skew`; `exp - iat <= max_lifetime`; `nbf`, if present, `<= now + skew`.
pub fn validate_time(raw: &RawClaims, policy: &ClaimsPolicy, now: u64) -> Result<(), AuthError> {
    let _ = (raw, policy, now);
    Err(AuthError::NotImplemented("claims::validate_time"))
}

/// Full claim validation for an already signature-verified payload whose
/// audience was checked by [`crate::AudiencePolicy`].
///
/// Contract: `iss == policy.issuer`; `sub` required; `client_id` required
/// (edge tokens: the DCR client; upstream: equals `aud`); `typ` claim, if
/// present, equals `policy.typ_claim`; `exchanged`, `setup`, `workload` true
/// => reject; `scope` split on single spaces; time via [`validate_time`].
pub fn validate(
    raw: RawClaims,
    policy: &ClaimsPolicy,
    kind: TokenKind,
    now: u64,
) -> Result<VerifiedClaims, AuthError> {
    let _ = (raw, policy, kind, now);
    Err(AuthError::NotImplemented("claims::validate"))
}
