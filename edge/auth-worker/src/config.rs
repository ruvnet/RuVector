//! Authorization-server configuration from wrangler `[vars]` (validated;
//! nothing from runtime discovery). The ES256 signing key is a secret
//! (`EDGE_AUTH_SIGNING_JWK`), read separately, never from vars.

use ruvector_edge_authz::client::{DcrPolicy, DEFAULT_CLIENT_SCOPE};
use ruvector_edge_authz::federation::UpstreamConfig;
use ruvector_edge_authz::{ConfidentialClients, OAuthError, ResourceAllowlist, ResourceUrl};

/// Default cap on dynamically registered clients (abuse bound on a public,
/// unauthenticated write endpoint).
pub const DEFAULT_MAX_CLIENTS: u64 = 10_000;

/// Validated AS configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// This AS's issuer URL (canonical, no trailing slash).
    pub issuer: String,
    /// Resources tokens may be minted for (exact canonical URLs), each with
    /// its own scopes from one vocabulary family (ADR-351 §5.3 grant rule).
    pub resources: ResourceAllowlist,
    /// Scopes advertised and registrable: the ordered union of every
    /// resource's scopes (derived, never a separate var that could drift):
    /// `ruvector:*`, `team:*` and `offline_access` (ADR-351 §5.3).
    pub scopes_supported: Vec<String>,
    /// Upstream IdP (`auth.cognitum.one`) client configuration.
    pub upstream: UpstreamConfig,
    /// DCR cap (counts live clients; idle ones are purged first).
    pub max_clients: u64,
    /// Registrations accepted per IP bucket per hour.
    pub dcr_rate_per_hour: u64,
    /// Ceiling given to a registration that omits `scope`
    /// ([`DEFAULT_CLIENT_SCOPE`] ∩ supported).
    pub default_scope: Vec<String>,
    /// Upstream RFC 7009 endpoint: the upstream refresh token is revoked
    /// right after the callback (ADR-351 §5.6 step 4).
    pub upstream_revocation_endpoint: String,
    /// Pinned upstream signing `kid`s (`ACCEPTED_UPSTREAM_KIDS`). Empty
    /// means federation is not ready (like an empty `UPSTREAM_CLIENT_ID`).
    pub upstream_accepted_kids: Vec<String>,
    /// Operator-registered confidential (adapter) clients of the RFC 8693
    /// exchange grant (`CONFIDENTIAL_CLIENTS`; empty = none). Never
    /// created by DCR.
    pub confidential_clients: ConfidentialClients,
}

/// `error_description` of `/authorize` while the upstream client is not
/// registered (gate G1, cognitum-one/console#605): static, never echoes
/// input.
pub const FEDERATION_PENDING: &str =
    "sign-in is not available yet: the upstream auth.cognitum.one client registration is pending";

/// Maximum pinned upstream `kid`s and `kid` length.
const MAX_PINNED_KIDS: usize = 8;
const MAX_KID_LEN: usize = 128;

fn kid_list(v: Option<String>) -> Result<Vec<String>, OAuthError> {
    let kids: Vec<String> = v
        .unwrap_or_default()
        .split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect();
    let valid = |k: &String| {
        k.len() <= MAX_KID_LEN
            && k.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    };
    if kids.len() > MAX_PINNED_KIDS || !kids.iter().all(valid) {
        return Err(crate::config_error("ACCEPTED_UPSTREAM_KIDS"));
    }
    Ok(kids)
}

/// The only scopes the edge AS may request upstream (ADR-351 §5.6: never
/// `mcp:*`, which `api.cognitum.one` would honour).
pub const UPSTREAM_IDENTITY_SCOPES: [&str; 3] = ["openid", "profile", "email"];

fn optional_u64(
    get: &dyn Fn(&str) -> Option<String>,
    name: &'static str,
    default: u64,
) -> Result<u64, OAuthError> {
    match get(name) {
        None => Ok(default),
        Some(v) if v.trim().is_empty() => Ok(default),
        Some(v) => v.trim().parse().map_err(|_| crate::config_error(name)),
    }
}

fn same_origin(a: &str, b: &str) -> bool {
    match (url::Url::parse(a), url::Url::parse(b)) {
        (Ok(a), Ok(b)) => a.origin() == b.origin(),
        _ => false,
    }
}

fn https_url(v: &str, name: &'static str) -> Result<(), OAuthError> {
    let u = url::Url::parse(v).map_err(|_| crate::config_error(name))?;
    let ok = u.scheme() == "https"
        && u.host_str().is_some_and(|h| !h.is_empty())
        && u.username().is_empty()
        && u.password().is_none()
        && u.fragment().is_none();
    ok.then_some(()).ok_or(crate::config_error(name))
}

fn scope_list(v: &str, name: &'static str) -> Result<Vec<String>, OAuthError> {
    let scopes: Vec<String> = v.split_whitespace().map(String::from).collect();
    let valid = |s: &String| {
        s.len() <= 64
            && s.bytes()
                .all(|b| b.is_ascii_graphic() && b != b'"' && b != b'\\')
    };
    if scopes.is_empty() || !scopes.iter().all(valid) {
        return Err(crate::config_error(name));
    }
    Ok(scopes)
}

impl AuthConfig {
    /// Build from a variable lookup (natively testable).
    ///
    /// Reads `ISSUER`, `RESOURCE_ALLOWLIST` (entries `<url> <scope>...`, see
    /// [`ResourceAllowlist::from_config`]; `scopes_supported` is their union),
    /// `UPSTREAM_ISSUER`, `UPSTREAM_AUTHORIZATION_ENDPOINT`,
    /// `UPSTREAM_TOKEN_ENDPOINT`, `UPSTREAM_JWKS_URL`, `UPSTREAM_CLIENT_ID`
    /// (may be empty: federation then answers `temporarily_unavailable`),
    /// `UPSTREAM_SCOPES` (identity scopes only, [`UPSTREAM_IDENTITY_SCOPES`])
    /// and optional `MAX_CLIENTS`, `DCR_RATE_PER_HOUR` (>= 1),
    /// `UPSTREAM_REVOCATION_ENDPOINT` (https, same origin as the upstream
    /// issuer; default `<UPSTREAM_ISSUER>/oauth/revoke`) and
    /// `ACCEPTED_UPSTREAM_KIDS` (comma-separated base64url `kid`s) and
    /// `CONFIDENTIAL_CLIENTS` (the operator registry of exchange clients, see
    /// [`ConfidentialClients::from_config`]; invalid fails the load). The upstream
    /// callback is `<ISSUER>/callback`.
    pub fn from_vars(get: &dyn Fn(&str) -> Option<String>) -> Result<Self, OAuthError> {
        let var = |name: &'static str| get(name).ok_or(crate::config_error(name));
        let issuer = var("ISSUER")?;
        let canonical = ResourceUrl::parse(&issuer).map_err(|_| crate::config_error("ISSUER"))?;
        if canonical.as_str() != issuer {
            return Err(crate::config_error("ISSUER must be canonical"));
        }
        let resources = ResourceAllowlist::from_config(&var("RESOURCE_ALLOWLIST")?)
            .map_err(|_| crate::config_error("RESOURCE_ALLOWLIST"))?;
        if resources.entries().is_empty() {
            return Err(crate::config_error("RESOURCE_ALLOWLIST empty"));
        }
        let upstream = UpstreamConfig {
            issuer: var("UPSTREAM_ISSUER")?,
            authorization_endpoint: var("UPSTREAM_AUTHORIZATION_ENDPOINT")?,
            token_endpoint: var("UPSTREAM_TOKEN_ENDPOINT")?,
            jwks_url: var("UPSTREAM_JWKS_URL")?,
            client_id: var("UPSTREAM_CLIENT_ID")?.trim().to_string(),
            redirect_uri: format!("{issuer}{}", ruvector_edge_authz::metadata::paths::CALLBACK),
            scopes: scope_list(&var("UPSTREAM_SCOPES")?, "UPSTREAM_SCOPES")?,
        };
        https_url(&upstream.issuer, "UPSTREAM_ISSUER")?;
        https_url(
            &upstream.authorization_endpoint,
            "UPSTREAM_AUTHORIZATION_ENDPOINT",
        )?;
        https_url(&upstream.token_endpoint, "UPSTREAM_TOKEN_ENDPOINT")?;
        https_url(&upstream.jwks_url, "UPSTREAM_JWKS_URL")?;
        if upstream.client_id.len() > 256
            || upstream.client_id.bytes().any(|b| !b.is_ascii_graphic())
        {
            return Err(crate::config_error("UPSTREAM_CLIENT_ID"));
        }
        if !upstream
            .scopes
            .iter()
            .all(|s| UPSTREAM_IDENTITY_SCOPES.contains(&s.as_str()))
        {
            return Err(crate::config_error(
                "UPSTREAM_SCOPES must be identity scopes only",
            ));
        }
        let max_clients = optional_u64(get, "MAX_CLIENTS", DEFAULT_MAX_CLIENTS)?;
        let dcr_rate_per_hour = optional_u64(
            get,
            "DCR_RATE_PER_HOUR",
            crate::abuse::DEFAULT_DCR_RATE_PER_HOUR,
        )?;
        if dcr_rate_per_hour == 0 {
            return Err(crate::config_error("DCR_RATE_PER_HOUR"));
        }
        let scopes_supported = resources.scopes_supported();
        let default_scope: Vec<String> = DEFAULT_CLIENT_SCOPE
            .iter()
            .filter(|s| scopes_supported.iter().any(|t| t == *s))
            .map(|s| s.to_string())
            .collect();
        if !default_scope.iter().any(|s| s.starts_with("ruvector:")) {
            return Err(crate::config_error(
                "RESOURCE_ALLOWLIST lacks ruvector scopes",
            ));
        }
        let confidential_clients = ConfidentialClients::from_config(
            &get("CONFIDENTIAL_CLIENTS").unwrap_or_default(),
            &resources,
        )
        .map_err(|_| crate::config_error("CONFIDENTIAL_CLIENTS"))?;
        let upstream_revocation_endpoint = match get("UPSTREAM_REVOCATION_ENDPOINT") {
            Some(v) if !v.trim().is_empty() => v.trim().to_string(),
            _ => format!("{}/oauth/revoke", upstream.issuer),
        };
        https_url(
            &upstream_revocation_endpoint,
            "UPSTREAM_REVOCATION_ENDPOINT",
        )?;
        if !same_origin(&upstream_revocation_endpoint, &upstream.issuer) {
            return Err(crate::config_error("UPSTREAM_REVOCATION_ENDPOINT origin"));
        }
        Ok(AuthConfig {
            issuer,
            resources,
            scopes_supported,
            upstream,
            max_clients,
            dcr_rate_per_hour,
            default_scope,
            upstream_revocation_endpoint,
            upstream_accepted_kids: kid_list(get("ACCEPTED_UPSTREAM_KIDS"))?,
            confidential_clients,
        })
    }

    /// Fail closed unless the federation leg is usable. Until gate G1 lands
    /// (`UPSTREAM_CLIENT_ID` and `ACCEPTED_UPSTREAM_KIDS` recorded) this is
    /// `temporarily_unavailable` with [`FEDERATION_PENDING`] as the
    /// description; then [`UpstreamConfig::ensure_ready`].
    pub fn ensure_federation_ready(&self) -> Result<(), OAuthError> {
        if self.upstream.client_id.is_empty() || self.upstream_accepted_kids.is_empty() {
            return Err(OAuthError::new(
                ruvector_edge_authz::OAuthErrorCode::TemporarilyUnavailable,
                FEDERATION_PENDING,
            ));
        }
        self.upstream.ensure_ready()
    }

    /// Read from the Worker environment.
    pub fn from_env(env: &worker::Env) -> Result<Self, OAuthError> {
        AuthConfig::from_vars(&|name| env.var(name).ok().map(|v| v.to_string()))
    }

    /// DCR policy: a registration that omits `scope` gets the
    /// [`AuthConfig::default_scope`] ceiling (`ruvector:read ruvector:write
    /// offline_access`), never the full supported set (ADR-351 §5.3); admin
    /// must be registered explicitly.
    pub fn dcr_policy(&self) -> DcrPolicy {
        DcrPolicy {
            scopes_supported: self.scopes_supported.clone(),
            default_scope: self.default_scope.clone(),
        }
    }
}

#[cfg(test)]
#[path = "config_tests.rs"]
pub(crate) mod tests;
