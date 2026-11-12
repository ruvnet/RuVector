//! Upstream federation to `auth.cognitum.one` (authorization code + PKCE
//! S256). The edge AS is itself an OAuth client of the upstream IdP; the
//! upstream token only establishes *who the user is*.
//!
//! Binding: the downstream request is bound to the upstream leg by a random
//! one-time `state` (stored with a TTL), a PKCE verifier only we know, a
//! `nonce`, and a **browser-binding secret** that the Worker sets as a
//! `__Host-` cookie at `/authorize` and presents at `/callback`. The binding
//! stops an attacker from starting a flow and handing the upstream login URL
//! to a victim already signed in upstream (the static-client-id proxy
//! confused-deputy case). It does not replace the consent screen: the Worker
//! must show `client_name`, the redirect host and the scopes before
//! redirecting upstream.

use crate::authorize::ValidatedAuthorization;
use crate::error::{OAuthError, OAuthErrorCode};
use crate::params::Params;
use crate::ports::{Clock, FederationStore, Rng};
use ruvector_edge_auth::{TokenKind, VerifiedClaims};
use serde::{Deserialize, Serialize};
use url::Url;

/// Pending-flow lifetime (seconds).
pub const FLOW_TTL_SECS: u64 = 600;
/// Random bytes in `state`, `nonce`, the upstream PKCE verifier and the
/// browser-binding secret.
pub const FLOW_SECRET_BYTES: usize = 32;
/// Upper bound on a presented `state` / browser secret (bytes).
pub const MAX_FLOW_PARAM_LEN: usize = 128;
/// Upper bound on the upstream `sub` (bytes).
pub const MAX_SUB_LEN: usize = 256;

/// Static upstream configuration (wrangler vars; nothing from discovery).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamConfig {
    /// `https://auth.cognitum.one`.
    pub issuer: String,
    /// Upstream authorization endpoint.
    pub authorization_endpoint: String,
    /// Upstream token endpoint.
    pub token_endpoint: String,
    /// `https://auth.cognitum.one/.well-known/jwks.json`.
    pub jwks_url: String,
    /// The edge AS's own client id at the upstream IdP (= upstream `aud`).
    pub client_id: String,
    /// Our callback, e.g. `https://ruvector-edge-auth.<acct>.workers.dev/callback`.
    pub redirect_uri: String,
    /// Scopes requested upstream (identity + org/workspace claims).
    pub scopes: Vec<String>,
}

impl UpstreamConfig {
    /// Fail closed unless the upstream leg is usable. An empty `client_id`
    /// (no upstream registration yet) is `temporarily_unavailable`; a
    /// non-https endpoint, an unsound trust root
    /// ([`UpstreamConfig::trust_root_ok`]) or an empty scope list is
    /// `server_error`.
    pub fn ensure_ready(&self) -> Result<(), OAuthError> {
        if self.client_id.is_empty() {
            return Err(OAuthError::new(
                OAuthErrorCode::TemporarilyUnavailable,
                "upstream login is not configured",
            ));
        }
        let https = |u: &str| Url::parse(u).is_ok_and(|u| u.scheme() == "https");
        if !self.trust_root_ok()
            || !https(&self.authorization_endpoint)
            || !https(&self.token_endpoint)
            || !https(&self.redirect_uri)
            || self.scopes.is_empty()
        {
            return Err(OAuthError::new(
                OAuthErrorCode::ServerError,
                "upstream configuration invalid",
            ));
        }
        Ok(())
    }

    /// Whether the upstream trust root is sound: `issuer` is a non-empty
    /// `https` URL with a host, no trailing slash (it is compared byte-exact
    /// with the token `iss`), no userinfo/query/fragment; `jwks_url` is
    /// `https` on the **same origin** as `issuer`. A plaintext or foreign
    /// JWKS would let a network attacker substitute the upstream keys.
    pub fn trust_root_ok(&self) -> bool {
        let (Ok(iss), Ok(jwks)) = (Url::parse(&self.issuer), Url::parse(&self.jwks_url)) else {
            return false;
        };
        iss.scheme() == "https"
            && iss.host_str().is_some_and(|h| !h.is_empty())
            && !self.issuer.ends_with('/')
            && iss.username().is_empty()
            && iss.password().is_none()
            && iss.query().is_none()
            && iss.fragment().is_none()
            && jwks.scheme() == "https"
            && jwks.origin() == iss.origin()
    }
}

/// Stored between `/authorize` and `/callback`, keyed by `state`.
/// `Debug` redacts the PKCE verifier.
#[derive(Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamFlowState {
    /// Random upstream `state` (one-time key).
    pub state: String,
    /// Random `nonce` sent upstream.
    pub nonce: String,
    /// Our PKCE verifier for the upstream leg.
    pub upstream_code_verifier: String,
    /// SHA-256 of the browser-binding secret (cookie).
    pub browser_binding: [u8; 32],
    /// The validated downstream request to resume.
    pub downstream: ValidatedAuthorization,
    /// Absolute expiry.
    pub expires_at: u64,
}

impl std::fmt::Debug for UpstreamFlowState {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamFlowState")
            .field("state", &self.state)
            .field("nonce", &self.nonce)
            .field("upstream_code_verifier", &crate::REDACTED)
            .field("browser_binding", &crate::REDACTED)
            .field("downstream", &self.downstream)
            .field("expires_at", &self.expires_at)
            .finish()
    }
}

/// Result of [`begin_upstream`]. `Debug` redacts the browser secret.
#[derive(Clone, PartialEq, Eq)]
pub struct UpstreamStart {
    /// Where to send the user agent.
    pub authorization_url: String,
    /// Set by the Worker at `GET /authorize` as a `__Host-` cookie with
    /// `Secure; HttpOnly; SameSite=Lax; Path=/; Max-Age=600` (browsers drop a
    /// `__Host-` cookie with any other `Path`), and presented to
    /// [`complete_upstream`] at `/callback`. The consent form token is
    /// [`crate::authorize::consent_token`] over this value.
    pub browser_secret: String,
}

impl std::fmt::Debug for UpstreamStart {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("UpstreamStart")
            .field("authorization_url", &self.authorization_url)
            .field("browser_secret", &crate::REDACTED)
            .finish()
    }
}

/// The user identity established upstream. Stored only inside the AS (code
/// and refresh records); tokens carry the derived edge subject
/// ([`UpstreamIdentity::edge_subject`]), never the raw upstream `sub`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamIdentity {
    /// Upstream issuer the identity was verified against (the tenant
    /// namespace, ADR-351 §4.1; the `upstream_iss` token claim).
    pub upstream_iss: String,
    /// Raw upstream `sub` (AS storage only).
    pub sub: String,
    /// Upstream `org_id` (tenant input).
    pub org_id: String,
    /// Upstream `workspace_id` (tenant input).
    pub workspace_id: String,
}

impl UpstreamIdentity {
    /// The ADR-351 §5.2 edge subject for this identity (hash of the upstream
    /// issuer and `sub`; the only user id tokens carry).
    pub fn edge_subject(&self) -> String {
        ruvector_edge_auth::subject::edge_subject(&self.upstream_iss, &self.sub)
    }
}

/// Start the upstream leg.
///
/// Contract: [`UpstreamConfig::ensure_ready`]; generate `state`, `nonce`, a
/// PKCE verifier and a browser secret from `rng`; persist
/// [`UpstreamFlowState`] with `expires_at = now + FLOW_TTL_SECS`; return the
/// upstream authorization URL (`response_type=code`, `client_id`,
/// `redirect_uri`, `scope`, `state`, `nonce`, `code_challenge`,
/// `code_challenge_method=S256`) and the browser secret.
pub fn begin_upstream<S, R, C>(
    store: &S,
    rng: &R,
    clock: &C,
    config: &UpstreamConfig,
    downstream: ValidatedAuthorization,
) -> Result<UpstreamStart, OAuthError>
where
    S: FederationStore + ?Sized,
    R: Rng + ?Sized,
    C: Clock + ?Sized,
{
    config.ensure_ready()?;
    let state = crate::random_secret(rng, FLOW_SECRET_BYTES)?;
    let nonce = crate::random_secret(rng, FLOW_SECRET_BYTES)?;
    let verifier = crate::random_secret(rng, FLOW_SECRET_BYTES)?;
    let browser_secret = crate::random_secret(rng, FLOW_SECRET_BYTES)?;
    let challenge = crate::pkce::challenge_s256(&verifier);
    let mut url = Url::parse(&config.authorization_endpoint)
        .map_err(|_| OAuthError::new(OAuthErrorCode::ServerError, "upstream endpoint"))?;
    url.query_pairs_mut()
        .append_pair("response_type", "code")
        .append_pair("client_id", &config.client_id)
        .append_pair("redirect_uri", &config.redirect_uri)
        .append_pair("scope", &config.scopes.join(" "))
        .append_pair("state", &state)
        .append_pair("nonce", &nonce)
        .append_pair("code_challenge", &challenge)
        .append_pair("code_challenge_method", "S256");
    store.insert_flow(&UpstreamFlowState {
        state,
        nonce,
        upstream_code_verifier: verifier,
        browser_binding: crate::secret_hash(&browser_secret),
        downstream,
        expires_at: clock.now_unix().saturating_add(FLOW_TTL_SECS),
    })?;
    Ok(UpstreamStart {
        authorization_url: url.into(),
        browser_secret,
    })
}

fn denied(desc: &'static str) -> OAuthError {
    OAuthError::new(OAuthErrorCode::AccessDenied, desc)
}

fn bounded(s: &str) -> bool {
    !s.is_empty() && s.len() <= MAX_FLOW_PARAM_LEN
}

/// Resume at `/callback`.
///
/// Contract: `state` and `browser_secret` non-empty and bounded (else
/// `access_denied` without touching the store); `take_flow(state)` (atomic,
/// one-time) -> none or expired => `access_denied`; the browser secret must
/// hash to the stored binding (constant time) => else `access_denied`. The
/// flow is consumed by every outcome after the take. Returns the flow so the
/// Worker can exchange the upstream code (with `upstream_code_verifier`) and
/// verify the result.
pub fn complete_upstream<S, C>(
    store: &S,
    clock: &C,
    state_param: &str,
    browser_secret: &str,
) -> Result<UpstreamFlowState, OAuthError>
where
    S: FederationStore + ?Sized,
    C: Clock + ?Sized,
{
    use subtle::ConstantTimeEq;
    if !bounded(state_param) || !bounded(browser_secret) {
        return Err(denied("missing or malformed flow binding"));
    }
    let flow = store
        .take_flow(state_param)?
        .ok_or(denied("unknown or already used state"))?;
    if clock.now_unix() >= flow.expires_at {
        return Err(denied("login flow expired"));
    }
    let presented = crate::secret_hash(browser_secret);
    if !bool::from(presented.ct_eq(&flow.browser_binding)) {
        return Err(denied("login flow bound to another browser"));
    }
    Ok(flow)
}

/// Query parameters on the upstream redirect to `/callback`.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct UpstreamCallback {
    /// Upstream-echoed `state`.
    pub state: String,
    /// `Ok(code)` or `Err(upstream error code)`.
    pub outcome: Result<String, String>,
    /// RFC 9207 `iss`, if the upstream sent one (see
    /// [`UpstreamCallback::check_issuer`]).
    pub iss: Option<String>,
}

impl UpstreamCallback {
    /// RFC 9207 mix-up defence (ADR-351 §5.6 callback step 1): an `iss`
    /// parameter, if present, must equal `config.issuer` byte-exact, else
    /// `access_denied`. Call before exchanging the code.
    pub fn check_issuer(&self, config: &UpstreamConfig) -> Result<(), OAuthError> {
        match &self.iss {
            Some(iss) if *iss != config.issuer => Err(denied("upstream issuer mismatch")),
            _ => Ok(()),
        }
    }

    /// Parse decoded pairs: `state` required; exactly one of `code` /
    /// `error`; optional `iss`. Anything else is `invalid_request`.
    pub fn from_pairs(pairs: &[(String, String)]) -> Result<Self, OAuthError> {
        let p = Params::from_pairs(pairs)?;
        let state = p.require("state", "state required")?;
        let iss = p.take("iss");
        let outcome = match (p.take("code"), p.take("error")) {
            (Some(code), None) => Ok(code),
            (None, Some(err)) => Err(err),
            _ => {
                return Err(OAuthError::new(
                    OAuthErrorCode::InvalidRequest,
                    "expected exactly one of code or error",
                ))
            }
        };
        Ok(UpstreamCallback {
            state,
            outcome,
            iss,
        })
    }
}

/// Form body for the upstream token request (RFC 6749 §4.1.3 + PKCE).
pub fn upstream_token_form(
    config: &UpstreamConfig,
    flow: &UpstreamFlowState,
    code: &str,
) -> Vec<(String, String)> {
    [
        ("grant_type", "authorization_code"),
        ("code", code),
        ("redirect_uri", config.redirect_uri.as_str()),
        ("client_id", config.client_id.as_str()),
        ("code_verifier", flow.upstream_code_verifier.as_str()),
    ]
    .iter()
    .map(|(k, v)| ((*k).to_string(), (*v).to_string()))
    .collect()
}

/// ADR-351 §4.1 charset for `org_id` / `workspace_id`: `^[A-Za-z0-9_-]{1,64}$`.
pub fn is_tenant_id(s: &str) -> bool {
    (1..=64).contains(&s.len())
        && s.bytes()
            .all(|b| b.is_ascii_alphanumeric() || b == b'_' || b == b'-')
}

/// Extract the identity from upstream claims already verified by
/// `ruvector_edge_auth::Verifier` (upstream issuer, JWKS, ES256).
///
/// Contract (`access_denied` on any failure): `kind == UpstreamFirstParty`;
/// `iss == config.issuer`; `aud == client_id == config.client_id`; `nonce`,
/// if the token carries one, equals `flow.nonce` (upstream access tokens may
/// omit it; the one-time state and our PKCE verifier already bind the code to
/// this flow); `sub` non-empty, <= [`MAX_SUB_LEN`], visible ASCII; `org_id`
/// and `workspace_id` present and [`is_tenant_id`]. The returned identity
/// records `upstream_iss` (the verified issuer).
pub fn identity_from_upstream(
    claims: &VerifiedClaims,
    flow: &UpstreamFlowState,
    config: &UpstreamConfig,
) -> Result<UpstreamIdentity, OAuthError> {
    use subtle::ConstantTimeEq;
    if claims.kind() != TokenKind::UpstreamFirstParty {
        return Err(denied("not an upstream token"));
    }
    if claims.iss() != config.issuer
        || claims.aud() != config.client_id
        || claims.client_id() != config.client_id
    {
        return Err(denied("upstream token not issued to this server"));
    }
    if let Some(n) = claims.nonce() {
        if !bool::from(n.as_bytes().ct_eq(flow.nonce.as_bytes())) {
            return Err(denied("nonce mismatch"));
        }
    }
    let sub = claims.sub();
    if sub.is_empty() || sub.len() > MAX_SUB_LEN || !sub.bytes().all(|b| b.is_ascii_graphic()) {
        return Err(denied("invalid upstream sub"));
    }
    let org_id = claims.org_id().filter(|s| is_tenant_id(s));
    let workspace_id = claims.workspace_id().filter(|s| is_tenant_id(s));
    let (Some(org_id), Some(workspace_id)) = (org_id, workspace_id) else {
        return Err(denied("upstream token lacks tenant claims"));
    };
    Ok(UpstreamIdentity {
        upstream_iss: config.issuer.clone(),
        sub: sub.to_string(),
        org_id: org_id.to_string(),
        workspace_id: workspace_id.to_string(),
    })
}
