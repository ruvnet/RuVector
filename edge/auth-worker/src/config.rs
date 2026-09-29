//! Authorization-server configuration from wrangler `[vars]` (validated;
//! nothing from runtime discovery). Secrets (the ES256 signing key) are read
//! separately via `env.secret`, never from vars.

use ruvector_edge_authz::federation::UpstreamConfig;
use ruvector_edge_authz::{OAuthError, ResourceAllowlist, ResourceUrl};
use worker::Env;

/// Validated AS configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// This AS's issuer URL (no trailing slash).
    pub issuer: String,
    /// Resources tokens may be minted for (exact canonical URLs).
    pub resources: ResourceAllowlist,
    /// Scopes advertised and registrable.
    pub scopes_supported: Vec<String>,
    /// Upstream IdP (`auth.cognitum.one`) client configuration.
    pub upstream: UpstreamConfig,
}

fn var(env: &Env, name: &'static str) -> Result<String, OAuthError> {
    env.var(name)
        .map(|v| v.to_string())
        .map_err(|_| crate::config_error(name))
}

impl AuthConfig {
    /// Read `ISSUER`, `RESOURCE_ALLOWLIST`, `SCOPES_SUPPORTED`,
    /// `UPSTREAM_ISSUER`, `UPSTREAM_AUTHORIZATION_ENDPOINT`,
    /// `UPSTREAM_TOKEN_ENDPOINT`, `UPSTREAM_JWKS_URL`, `UPSTREAM_CLIENT_ID`,
    /// `UPSTREAM_SCOPES`. The callback is `<ISSUER>/callback`.
    pub fn from_env(env: &Env) -> Result<Self, OAuthError> {
        let issuer = var(env, "ISSUER")?;
        ResourceUrl::parse(&issuer).map_err(|_| crate::config_error("ISSUER"))?;
        let resources = ResourceAllowlist::from_config(&var(env, "RESOURCE_ALLOWLIST")?)
            .map_err(|_| crate::config_error("RESOURCE_ALLOWLIST"))?;
        let split = |s: String| s.split_whitespace().map(String::from).collect::<Vec<_>>();
        let upstream = UpstreamConfig {
            issuer: var(env, "UPSTREAM_ISSUER")?,
            authorization_endpoint: var(env, "UPSTREAM_AUTHORIZATION_ENDPOINT")?,
            token_endpoint: var(env, "UPSTREAM_TOKEN_ENDPOINT")?,
            jwks_url: var(env, "UPSTREAM_JWKS_URL")?,
            client_id: var(env, "UPSTREAM_CLIENT_ID")?,
            redirect_uri: format!("{issuer}{}", ruvector_edge_authz::metadata::paths::CALLBACK),
            scopes: split(var(env, "UPSTREAM_SCOPES")?),
        };
        Ok(AuthConfig {
            issuer,
            resources,
            scopes_supported: split(var(env, "SCOPES_SUPPORTED")?),
            upstream,
        })
    }
}
