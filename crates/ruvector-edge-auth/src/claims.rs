//! Claims model and validation (ADR-351 §5.1.6, amended by the edge-AS
//! decision).

use crate::error::AuthError;
use crate::resource::MAX_RESOURCE_URL_LEN;
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
    /// Upstream refresh-family id (ADR §5.6 deny-list key).
    pub family_id: Option<String>,
    /// Edge tokens: the upstream issuer that namespaces the tenant (§4.1).
    pub upstream_iss: Option<String>,
    pub exchanged: Option<bool>,
    pub setup: Option<bool>,
    pub workload: Option<bool>,
}

/// Hard cap on `exp - iat` for edge-issued tokens (ADR §5.4.7; the AS mints
/// 900 s tokens). [`validate`] applies it whatever the policy says.
pub const EDGE_MAX_LIFETIME_SECS: u64 = 900;
/// Hard cap on `exp - iat` for upstream first-party tokens (ADR §5.5).
pub const UPSTREAM_MAX_LIFETIME_SECS: u64 = 3600;
/// `typ` claim every upstream first-party token must carry (ADR §5.5).
/// [`validate`] requires it for [`TokenKind::UpstreamFirstParty`] whatever
/// the policy says, so upstream ID tokens (same key, `aud` = client_id, no
/// `typ`) are always refused.
pub const UPSTREAM_ACCESS_TYP: &str = "access";
/// The upstream issuer the edge AS federates to (ADR §4.1). Only a default
/// for tests; production callers pass it to [`ClaimsPolicy::edge`].
pub const DEFAULT_UPSTREAM_ISSUER: &str = "https://auth.cognitum.one";

/// Per-kind claim rules. Build with [`ClaimsPolicy::edge`] or
/// [`ClaimsPolicy::upstream_access`]; [`validate`] additionally enforces the
/// kind's hard limits, so a looser policy can never widen them.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct ClaimsPolicy {
    /// Exact required `iss`.
    pub issuer: String,
    /// Clock skew tolerance (default 60 s).
    pub skew_secs: u64,
    /// Maximum `exp - iat`. Capped by the kind's hard limit
    /// ([`EDGE_MAX_LIFETIME_SECS`] / [`UPSTREAM_MAX_LIFETIME_SECS`]).
    pub max_lifetime_secs: u64,
    /// Required value of the `typ` claim (upstream: always at least
    /// [`UPSTREAM_ACCESS_TYP`]).
    pub typ_claim: Option<String>,
    /// Edge tokens: when `Some`, the `upstream_iss` claim is **required**
    /// and must equal it exactly (ADR §5.4.7). Ignored for upstream tokens,
    /// whose `upstream_iss` is their own `iss`.
    pub upstream_issuer: Option<String>,
}

impl ClaimsPolicy {
    /// Edge-issued access tokens (ADR §5.4.7): skew 60 s, `exp - iat <= 900`,
    /// no `typ` claim rule, `upstream_iss` required and equal to
    /// `upstream_issuer`.
    pub fn edge(issuer: impl Into<String>, upstream_issuer: impl Into<String>) -> Self {
        ClaimsPolicy {
            issuer: issuer.into(),
            skew_secs: 60,
            max_lifetime_secs: EDGE_MAX_LIFETIME_SECS,
            typ_claim: None,
            upstream_issuer: Some(upstream_issuer.into()),
        }
    }

    /// Upstream first-party access tokens (ADR §5.5): skew 60 s,
    /// `exp - iat <= 3600`, claim `typ == "access"`.
    pub fn upstream_access(issuer: impl Into<String>) -> Self {
        ClaimsPolicy {
            issuer: issuer.into(),
            skew_secs: 60,
            max_lifetime_secs: UPSTREAM_MAX_LIFETIME_SECS,
            typ_claim: Some(UPSTREAM_ACCESS_TYP.into()),
            upstream_issuer: None,
        }
    }

    /// Legacy constructor kept for existing callers: skew 60 s, lifetime
    /// 3600 s, no `typ` / `upstream_iss` rule. Prefer [`ClaimsPolicy::edge`] /
    /// [`ClaimsPolicy::upstream_access`]. Even with this policy, [`validate`]
    /// caps edge tokens at 900 s and requires `typ == "access"` on upstream
    /// tokens.
    pub fn with_defaults(issuer: impl Into<String>) -> Self {
        ClaimsPolicy {
            issuer: issuer.into(),
            skew_secs: 60,
            max_lifetime_secs: UPSTREAM_MAX_LIFETIME_SECS,
            typ_claim: None,
            upstream_issuer: None,
        }
    }

    /// The effective lifetime cap for `kind`: the policy value, never above
    /// the kind's hard limit.
    pub fn lifetime_cap(&self, kind: TokenKind) -> u64 {
        let hard = match kind {
            TokenKind::EdgeIssued => EDGE_MAX_LIFETIME_SECS,
            TokenKind::UpstreamFirstParty => UPSTREAM_MAX_LIFETIME_SECS,
        };
        self.max_lifetime_secs.min(hard)
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
    family_id: Option<String>,
    upstream_iss: Option<String>,
    kid: String,
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
    /// `sub` verbatim. Edge tokens: the edge subject. Upstream first-party
    /// tokens: the raw upstream `sub` (normalise with
    /// [`crate::subject::edge_subject`] before using it as an RS actor).
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
    /// Refresh-family id (required on upstream tokens; carried by edge ones).
    pub fn family_id(&self) -> Option<&str> {
        self.family_id.as_deref()
    }
    /// Upstream issuer that namespaces the tenant (ADR §4.1): the edge
    /// token's `upstream_iss` claim, or the upstream token's own `iss`.
    /// `None` only for an edge token validated under a policy without
    /// [`ClaimsPolicy::upstream_issuer`] that did not carry the claim.
    pub fn upstream_iss(&self) -> Option<&str> {
        self.upstream_iss.as_deref()
    }
    /// JOSE header `kid` of the signing key (ADR §5.4.7 / §5.6 deny-list key).
    /// Always set by [`crate::Verifier::verify`]; empty only for claims built
    /// by [`validate`] directly, which never saw the header.
    pub fn kid(&self) -> &str {
        &self.kid
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

    /// Attach the verified header `kid` (the verifier calls this after the
    /// signature check).
    pub(crate) fn with_kid(mut self, kid: &str) -> Self {
        self.kid = kid.to_string();
        self
    }

    /// Test constructor (feature `test-util`). Never enable in release builds.
    /// `upstream_iss` is the token's `iss` for upstream tokens and
    /// [`DEFAULT_UPSTREAM_ISSUER`] for edge ones; `kid` is `"test-kid"`.
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
        let upstream_iss = match kind {
            TokenKind::EdgeIssued => DEFAULT_UPSTREAM_ISSUER,
            TokenKind::UpstreamFirstParty => iss,
        };
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
            family_id: None,
            upstream_iss: Some(upstream_iss.into()),
            kid: "test-kid".into(),
            nonce: None,
            iat,
            exp,
        }
    }
}

/// Maximum byte length of any string claim we keep.
pub const MAX_CLAIM_LEN: usize = 256;
/// Maximum byte length of the `scope` claim.
pub const MAX_SCOPE_LEN: usize = 2048;
/// Maximum number of scope tokens.
pub const MAX_SCOPES: usize = 64;

/// Validate time claims against `now` with `policy.max_lifetime_secs` as the
/// lifetime cap (kind-agnostic; [`validate`] applies the kind's hard cap).
///
/// Contract: `exp` and `iat` required; `exp > now - skew`; `iat <= now +
/// skew`; `exp - iat <= max_lifetime`; `nbf`, if present, `<= now + skew`.
pub fn validate_time(raw: &RawClaims, policy: &ClaimsPolicy, now: u64) -> Result<(), AuthError> {
    check_time(raw, policy.skew_secs, policy.max_lifetime_secs, now)
}

fn check_time(raw: &RawClaims, skew: u64, max_lifetime: u64, now: u64) -> Result<(), AuthError> {
    let exp = raw.exp.ok_or(AuthError::InvalidClaim("exp"))?;
    let iat = raw.iat.ok_or(AuthError::InvalidClaim("iat"))?;
    let horizon = now.saturating_add(skew);
    if exp.saturating_add(skew) <= now {
        return Err(AuthError::Expired);
    }
    if iat > horizon {
        return Err(AuthError::IssuedInFuture);
    }
    if raw.nbf.is_some_and(|nbf| nbf > horizon) {
        return Err(AuthError::NotYetValid);
    }
    if exp <= iat {
        return Err(AuthError::InvalidClaim("exp <= iat"));
    }
    if exp - iat > max_lifetime {
        return Err(AuthError::LifetimeTooLong);
    }
    Ok(())
}

/// Full claim validation for an already signature-verified payload whose
/// audience was checked by [`crate::AudiencePolicy`].
///
/// Contract: `iss == policy.issuer`; `sub`, `client_id`, `jti`, `org_id`,
/// `workspace_id` and `scope` required, bounded strings (edge tokens:
/// `client_id` is the DCR client; upstream: `client_id == aud` and
/// `family_id` required); `aud` a single string of at most
/// [`MAX_RESOURCE_URL_LEN`] bytes; `exchanged`, `setup`, `workload` true =>
/// reject; `scope` split on single spaces (empty tokens and non RFC 6749
/// scope-token bytes rejected); time via the kind-capped lifetime
/// ([`ClaimsPolicy::lifetime_cap`]: edge <= 900 s, upstream <= 3600 s).
///
/// `typ` claim: for [`TokenKind::UpstreamFirstParty`] it is **required** and
/// must equal [`UPSTREAM_ACCESS_TYP`] regardless of the policy (rejects
/// upstream ID tokens, which carry no `typ`); additionally, when
/// `policy.typ_claim` is `Some(v)` it must equal `v`.
///
/// `upstream_iss`: edge tokens under a policy with `upstream_issuer =
/// Some(u)` must carry `upstream_iss == u`; upstream tokens get
/// `upstream_iss = iss`. Their `sub` is kept verbatim (the edge AS callback
/// needs the raw upstream `sub`); callers normalise it with
/// [`crate::subject::edge_subject`] for the §5.5 resource-server mode.
pub fn validate(
    raw: RawClaims,
    policy: &ClaimsPolicy,
    kind: TokenKind,
    now: u64,
) -> Result<VerifiedClaims, AuthError> {
    match raw.iss.as_deref() {
        None => return Err(AuthError::InvalidClaim("iss")),
        Some(iss) if iss != policy.issuer => return Err(AuthError::WrongIssuer),
        Some(_) => {}
    }
    if kind == TokenKind::UpstreamFirstParty && raw.typ.as_deref() != Some(UPSTREAM_ACCESS_TYP) {
        return Err(AuthError::InvalidClaim("typ"));
    }
    if let Some(want) = policy.typ_claim.as_deref() {
        if raw.typ.as_deref() != Some(want) {
            return Err(AuthError::InvalidClaim("typ"));
        }
    }
    if raw.exchanged == Some(true) {
        return Err(AuthError::InvalidClaim("exchanged"));
    }
    if raw.setup == Some(true) {
        return Err(AuthError::InvalidClaim("setup"));
    }
    if raw.workload == Some(true) {
        return Err(AuthError::InvalidClaim("workload"));
    }
    check_time(&raw, policy.skew_secs, policy.lifetime_cap(kind), now)?;

    // `check_audience` already required byte-equality with a validated
    // `ResourceUrl` (up to 512 bytes) or an allowlisted client id.
    let aud = match raw.aud {
        Some(Audience::One(a)) => bounded_to(Some(a), "aud", MAX_RESOURCE_URL_LEN)?,
        _ => return Err(AuthError::InvalidClaim("aud")),
    };
    let sub = bounded(raw.sub, "sub")?;
    let client_id = bounded(raw.client_id, "client_id")?;
    let jti = bounded(raw.jti, "jti")?;
    let org_id = bounded(raw.org_id, "org_id")?;
    let workspace_id = bounded(raw.workspace_id, "workspace_id")?;
    let scopes = parse_scope(
        raw.scope
            .as_deref()
            .ok_or(AuthError::InvalidClaim("scope"))?,
    )?;
    let nonce = raw.nonce.map(|n| bounded(Some(n), "nonce")).transpose()?;
    let (family_id, upstream_iss) = match kind {
        TokenKind::UpstreamFirstParty => {
            if client_id != aud {
                return Err(AuthError::InvalidClaim("client_id != aud"));
            }
            let family = bounded(raw.family_id, "family_id")?;
            (Some(family), Some(policy.issuer.clone()))
        }
        TokenKind::EdgeIssued => {
            let family = raw
                .family_id
                .map(|f| bounded(Some(f), "family_id"))
                .transpose()?;
            let upstream_iss = match (&policy.upstream_issuer, raw.upstream_iss) {
                (Some(want), Some(got)) if got == *want => Some(got),
                (Some(_), _) => return Err(AuthError::InvalidClaim("upstream_iss")),
                (None, got) => got.map(|u| bounded(Some(u), "upstream_iss")).transpose()?,
            };
            (family, upstream_iss)
        }
    };
    Ok(VerifiedClaims {
        kind,
        iss: policy.issuer.clone(),
        aud,
        sub,
        client_id,
        org_id: Some(org_id),
        workspace_id: Some(workspace_id),
        scopes,
        jti: Some(jti),
        nonce,
        family_id,
        upstream_iss,
        kid: String::new(),
        // check_time guarantees both are present.
        iat: raw.iat.unwrap_or_default(),
        exp: raw.exp.unwrap_or_default(),
    })
}

/// Required, non-empty, length-bounded string without control characters.
fn bounded(value: Option<String>, name: &'static str) -> Result<String, AuthError> {
    bounded_to(value, name, MAX_CLAIM_LEN)
}

fn bounded_to(value: Option<String>, name: &'static str, max: usize) -> Result<String, AuthError> {
    match value {
        Some(v) if !v.is_empty() && v.len() <= max && !v.chars().any(char::is_control) => Ok(v),
        _ => Err(AuthError::InvalidClaim(name)),
    }
}

/// Split a `scope` claim on single spaces (RFC 6749 §3.3). An empty claim
/// is zero scopes; empty tokens (leading/trailing/double spaces) and bytes
/// outside `%x21 / %x23-5B / %x5D-7E` are rejected.
pub fn parse_scope(scope: &str) -> Result<Vec<String>, AuthError> {
    if scope.len() > MAX_SCOPE_LEN {
        return Err(AuthError::InvalidClaim("scope"));
    }
    if scope.is_empty() {
        return Ok(Vec::new());
    }
    let mut out = Vec::new();
    for token in scope.split(' ') {
        let ok = !token.is_empty()
            && token
                .bytes()
                .all(|b| b == 0x21 || (0x23..=0x5B).contains(&b) || (0x5D..=0x7E).contains(&b));
        if !ok || out.len() == MAX_SCOPES {
            return Err(AuthError::InvalidClaim("scope"));
        }
        out.push(token.to_string());
    }
    Ok(out)
}

#[cfg(test)]
mod tests;
