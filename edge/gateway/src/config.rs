//! Gateway configuration from wrangler `[vars]`. Every value is validated at
//! the boundary; nothing comes from runtime discovery.

use ruvector_edge_auth::{AudiencePolicy, AuthError, ResourceUrl, UpstreamFirstPartyPolicy};
use worker::Env;

/// Validated gateway configuration.
#[derive(Debug, Clone)]
pub struct GatewayConfig {
    /// Edge AS issuer (`https://ruvector-edge-auth.<acct>.workers.dev`).
    pub edge_issuer: String,
    /// Edge AS JWKS URL (`<edge_issuer>/.well-known/jwks.json`).
    pub edge_jwks_url: String,
    /// REST resource `https://<host>/v1`.
    pub rest_resource: ResourceUrl,
    /// MCP resource `https://<host>/v1/mcp`.
    pub mcp_resource: ResourceUrl,
    /// Upstream first-party acceptance (explicit opt-in, default off).
    pub upstream: Option<UpstreamFirstPartyPolicy>,
    /// Upstream JWKS URL (used only when `upstream` is `Some`).
    pub upstream_jwks_url: String,
}

fn var(env: &Env, name: &str) -> Result<String, AuthError> {
    env.var(name)
        .map(|v| v.to_string())
        .map_err(|_| AuthError::InvalidConfig("missing var"))
}

impl GatewayConfig {
    /// Read and validate `EDGE_ISSUER`, `EDGE_JWKS_URL`, `PUBLIC_ORIGIN`,
    /// `UPSTREAM_FIRST_PARTY` (`"true"`/`"false"`), `UPSTREAM_ISSUER`,
    /// `UPSTREAM_JWKS_URL`, `FIRST_PARTY_AUDS` (comma-separated exact ids).
    pub fn from_env(env: &Env) -> Result<Self, AuthError> {
        let origin = var(env, "PUBLIC_ORIGIN")?;
        let rest_resource = ResourceUrl::parse(&format!("{origin}/v1"))?;
        let mcp_resource = ResourceUrl::parse(&format!("{origin}/v1/mcp"))?;
        let edge_issuer = var(env, "EDGE_ISSUER")?;
        ResourceUrl::parse(&edge_issuer)?;
        let edge_jwks_url = var(env, "EDGE_JWKS_URL")?;
        let upstream = match var(env, "UPSTREAM_FIRST_PARTY")?.as_str() {
            "true" => {
                let auds: Vec<String> = var(env, "FIRST_PARTY_AUDS")?
                    .split(',')
                    .map(str::trim)
                    .filter(|s| !s.is_empty())
                    .map(String::from)
                    .collect();
                if auds.is_empty() {
                    return Err(AuthError::InvalidConfig("FIRST_PARTY_AUDS empty"));
                }
                Some(UpstreamFirstPartyPolicy {
                    issuer: var(env, "UPSTREAM_ISSUER")?,
                    first_party_auds: auds,
                })
            }
            "false" => None,
            _ => {
                return Err(AuthError::InvalidConfig(
                    "UPSTREAM_FIRST_PARTY must be true|false",
                ))
            }
        };
        let upstream_jwks_url = var(env, "UPSTREAM_JWKS_URL")?;
        Ok(GatewayConfig {
            edge_issuer,
            edge_jwks_url,
            rest_resource,
            mcp_resource,
            upstream,
            upstream_jwks_url,
        })
    }

    /// Audience policy for a resource served by this gateway.
    pub fn audience_for(&self, resource: &ResourceUrl) -> AudiencePolicy {
        AudiencePolicy {
            edge_issuer: self.edge_issuer.clone(),
            resource: resource.clone(),
            upstream: self.upstream.clone(),
        }
    }
}
