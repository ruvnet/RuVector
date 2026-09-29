//! Upstream federation to `auth.cognitum.one` (authorization code + PKCE
//! S256). The edge AS is itself an OAuth client of the upstream IdP; the
//! upstream token only establishes *who the user is*. The downstream request
//! is bound to the upstream flow by `state` (one-time) and `nonce`.

use crate::authorize::ValidatedAuthorization;
use crate::error::OAuthError;
use crate::ports::{Clock, FederationStore, Rng};
use ruvector_edge_auth::VerifiedClaims;
use serde::{Deserialize, Serialize};

/// Pending-flow lifetime (seconds).
pub const FLOW_TTL_SECS: u64 = 600;

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

/// Stored between `/authorize` and `/callback`, keyed by `state`.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamFlowState {
    /// Random upstream `state` (one-time key).
    pub state: String,
    /// Random `nonce` expected in the upstream id_token.
    pub nonce: String,
    /// Our PKCE verifier for the upstream leg.
    pub upstream_code_verifier: String,
    /// The validated downstream request to resume.
    pub downstream: ValidatedAuthorization,
    /// Absolute expiry.
    pub expires_at: u64,
}

/// The user identity carried into edge-minted tokens.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct UpstreamIdentity {
    /// Upstream `sub`.
    pub sub: String,
    /// Upstream `org_id` (tenant input).
    pub org_id: String,
    /// Upstream `workspace_id` (tenant input).
    pub workspace_id: String,
}

/// Start the upstream leg.
///
/// Contract: generate `state`, `nonce` and a PKCE verifier from `rng`;
/// persist [`UpstreamFlowState`] with `expires_at = now + FLOW_TTL_SECS`;
/// return the upstream authorization URL
/// (`response_type=code`, `client_id`, `redirect_uri`, `scope`, `state`,
/// `nonce`, `code_challenge`, `code_challenge_method=S256`).
pub fn begin_upstream<S: FederationStore, R: Rng, C: Clock>(
    store: &S,
    rng: &R,
    clock: &C,
    config: &UpstreamConfig,
    downstream: ValidatedAuthorization,
) -> Result<String, OAuthError> {
    let _ = (store, rng, clock.now_unix(), config, downstream);
    Err(OAuthError::not_implemented("federation::begin_upstream"))
}

/// Resume at `/callback`.
///
/// Contract: `take_flow(state)` (atomic, one-time) -> none or expired =>
/// `access_denied`; returns the flow so the Worker can exchange the upstream
/// code (with `upstream_code_verifier`) over HTTP and verify the result.
pub fn complete_upstream<S: FederationStore, C: Clock>(
    store: &S,
    clock: &C,
    state_param: &str,
) -> Result<UpstreamFlowState, OAuthError> {
    let _ = (store, clock.now_unix(), state_param);
    Err(OAuthError::not_implemented("federation::complete_upstream"))
}

/// Extract the identity from upstream claims verified by
/// `ruvector_edge_auth::Verifier` (upstream issuer, JWKS, ES256, `aud ==
/// config.client_id`).
///
/// Contract: `nonce` claim (id_token) must equal `flow.nonce` when present in
/// the flow; `sub`, `org_id`, `workspace_id` required.
pub fn identity_from_upstream(
    claims: &VerifiedClaims,
    flow: &UpstreamFlowState,
) -> Result<UpstreamIdentity, OAuthError> {
    let _ = (claims, flow);
    Err(OAuthError::not_implemented(
        "federation::identity_from_upstream",
    ))
}
