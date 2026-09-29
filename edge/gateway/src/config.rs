//! Gateway configuration from wrangler `[vars]`. Every value is validated at
//! the boundary; nothing comes from runtime discovery.

use ruvector_edge_auth::{AudiencePolicy, AuthError, ResourceUrl, UpstreamFirstPartyPolicy};

/// JWKS path on the edge AS (`ruvector_edge_authz::metadata::paths::JWKS`;
/// repeated here so the gateway does not depend on the AS crate).
pub const EDGE_JWKS_PATH: &str = "/.well-known/jwks.json";

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

/// Client-id prefixes minted by DCR (upstream `dcr-`, edge `edc-`): never
/// first-party, so never accepted in `FIRST_PARTY_AUDS`.
const DCR_PREFIXES: [&str; 2] = ["dcr-", "edc-"];
/// Bounds on the pinned upstream `kid` list.
const MAX_PINNED_KIDS: usize = 8;
const MAX_KID_LEN: usize = 128;

fn list(v: &str) -> Vec<String> {
    v.split([',', ' '])
        .map(str::trim)
        .filter(|s| !s.is_empty())
        .map(String::from)
        .collect()
}

/// The §5.5 upstream first-party policy (only when explicitly enabled).
///
/// Trust root (the AS's `UpstreamConfig::trust_root_ok` rule, tightened):
/// `UPSTREAM_ISSUER` canonical `https` (no trailing slash) and distinct from
/// the edge issuer; `UPSTREAM_JWKS_URL` exactly issuer + JWKS path (same
/// origin). `FIRST_PARTY_AUDS`: non-empty exact ids, never a DCR id.
/// `ACCEPTED_UPSTREAM_KIDS`: non-empty base64url `kid` pin (1..=8).
fn upstream_policy(
    var: &dyn Fn(&str) -> Result<String, AuthError>,
    edge_issuer: &str,
    jwks_url: &str,
) -> Result<UpstreamFirstPartyPolicy, AuthError> {
    let issuer = var("UPSTREAM_ISSUER")?;
    if ResourceUrl::parse(&issuer)?.as_str() != issuer {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_ISSUER must be canonical",
        ));
    }
    if issuer == edge_issuer {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_ISSUER equals EDGE_ISSUER",
        ));
    }
    if jwks_url != format!("{issuer}{EDGE_JWKS_PATH}") {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_JWKS_URL must be issuer + JWKS path",
        ));
    }
    let auds = list(&var("FIRST_PARTY_AUDS")?);
    if auds.is_empty() {
        return Err(AuthError::InvalidConfig("FIRST_PARTY_AUDS empty"));
    }
    if auds
        .iter()
        .any(|a| DCR_PREFIXES.iter().any(|p| a.starts_with(p)))
    {
        return Err(AuthError::InvalidConfig(
            "FIRST_PARTY_AUDS must not hold DCR client ids",
        ));
    }
    let kids = list(&var("ACCEPTED_UPSTREAM_KIDS")?);
    let kid_ok = |k: &String| {
        k.len() <= MAX_KID_LEN
            && k.bytes()
                .all(|b| b.is_ascii_alphanumeric() || b == b'-' || b == b'_')
    };
    if kids.is_empty() || kids.len() > MAX_PINNED_KIDS || !kids.iter().all(kid_ok) {
        return Err(AuthError::InvalidConfig("ACCEPTED_UPSTREAM_KIDS"));
    }
    Ok(UpstreamFirstPartyPolicy {
        issuer,
        first_party_auds: auds,
        accepted_kids: kids,
    })
}

impl GatewayConfig {
    /// Build from a variable lookup (natively testable).
    ///
    /// Reads `PUBLIC_ORIGIN`, `EDGE_ISSUER`, `EDGE_JWKS_URL` (must equal
    /// `EDGE_ISSUER` + [`EDGE_JWKS_PATH`]), `UPSTREAM_FIRST_PARTY`
    /// (`"true"`/`"false"`), `UPSTREAM_ISSUER`, `UPSTREAM_JWKS_URL`,
    /// `FIRST_PARTY_AUDS` and `ACCEPTED_UPSTREAM_KIDS` (comma-separated; both
    /// required non-empty when the upstream path is on, see
    /// [`upstream_policy`]).
    pub fn from_vars(get: &dyn Fn(&str) -> Option<String>) -> Result<Self, AuthError> {
        let var = |name: &str| get(name).ok_or(AuthError::InvalidConfig("missing var"));
        let origin = var("PUBLIC_ORIGIN")?;
        let origin_url = ResourceUrl::parse(&origin)?;
        if origin_url.as_str() != origin || !origin_url.path().is_empty() {
            return Err(AuthError::InvalidConfig(
                "PUBLIC_ORIGIN must be a bare origin",
            ));
        }
        let rest_resource = ResourceUrl::parse(&format!("{origin}/v1"))?;
        let mcp_resource = ResourceUrl::parse(&format!("{origin}/v1/mcp"))?;
        let edge_issuer = var("EDGE_ISSUER")?;
        if ResourceUrl::parse(&edge_issuer)?.as_str() != edge_issuer {
            return Err(AuthError::InvalidConfig("EDGE_ISSUER must be canonical"));
        }
        let edge_jwks_url = var("EDGE_JWKS_URL")?;
        if edge_jwks_url != format!("{edge_issuer}{EDGE_JWKS_PATH}") {
            return Err(AuthError::InvalidConfig(
                "EDGE_JWKS_URL must be issuer + JWKS path",
            ));
        }
        let upstream_jwks_url = var("UPSTREAM_JWKS_URL")?;
        let upstream = match var("UPSTREAM_FIRST_PARTY")?.as_str() {
            "true" => Some(upstream_policy(&var, &edge_issuer, &upstream_jwks_url)?),
            "false" => None,
            _ => {
                return Err(AuthError::InvalidConfig(
                    "UPSTREAM_FIRST_PARTY must be true|false",
                ))
            }
        };
        Ok(GatewayConfig {
            edge_issuer,
            edge_jwks_url,
            rest_resource,
            mcp_resource,
            upstream,
            upstream_jwks_url,
        })
    }

    /// Read from the Worker environment.
    pub fn from_env(env: &worker::Env) -> Result<Self, AuthError> {
        GatewayConfig::from_vars(&|name| env.var(name).ok().map(|v| v.to_string()))
    }

    /// Audience policy for a resource served by this gateway: exact `aud`,
    /// the gateway's other resources as siblings (403, not 401), and the
    /// upstream first-party path only on the REST resource (ADR-351 §5.5).
    pub fn audience_for(&self, resource: &ResourceUrl) -> AudiencePolicy {
        let rest = *resource == self.rest_resource;
        AudiencePolicy {
            edge_issuer: self.edge_issuer.clone(),
            resource: resource.clone(),
            sibling_resources: [&self.rest_resource, &self.mcp_resource]
                .into_iter()
                .filter(|r| *r != resource)
                .cloned()
                .collect(),
            upstream: if rest { self.upstream.clone() } else { None },
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn vars() -> HashMap<&'static str, String> {
        HashMap::from([
            (
                "PUBLIC_ORIGIN",
                "https://gw.example.workers.dev".to_string(),
            ),
            ("EDGE_ISSUER", "https://as.example.workers.dev".into()),
            (
                "EDGE_JWKS_URL",
                "https://as.example.workers.dev/.well-known/jwks.json".into(),
            ),
            ("UPSTREAM_FIRST_PARTY", "false".into()),
            ("UPSTREAM_ISSUER", "https://auth.cognitum.one".into()),
            (
                "UPSTREAM_JWKS_URL",
                "https://auth.cognitum.one/.well-known/jwks.json".into(),
            ),
            ("FIRST_PARTY_AUDS", "".into()),
            ("ACCEPTED_UPSTREAM_KIDS", "".into()),
        ])
    }

    fn load(v: &HashMap<&'static str, String>) -> Result<GatewayConfig, AuthError> {
        GatewayConfig::from_vars(&|k| v.get(k).cloned())
    }

    #[test]
    fn default_shape_is_edge_only_with_exact_resources() {
        let c = load(&vars()).unwrap();
        assert!(c.upstream.is_none());
        assert_eq!(
            c.rest_resource.as_str(),
            "https://gw.example.workers.dev/v1"
        );
        assert_eq!(
            c.mcp_resource.as_str(),
            "https://gw.example.workers.dev/v1/mcp"
        );
        let a = c.audience_for(&c.mcp_resource);
        assert_eq!(a.edge_issuer, "https://as.example.workers.dev");
        assert!(a.upstream.is_none());
    }

    #[test]
    fn upstream_path_needs_explicit_opt_in_auds_and_kid_pin() {
        let mut v = vars();
        v.insert("UPSTREAM_FIRST_PARTY", "true".into());
        assert!(load(&v).is_err());
        v.insert("FIRST_PARTY_AUDS", " cli-a , cli-b ,".into());
        assert!(load(&v).is_err(), "no kid pin");
        v.insert("ACCEPTED_UPSTREAM_KIDS", "kid-1, kid_2".into());
        let c = load(&v).unwrap();
        let up = c.upstream.clone().unwrap();
        assert_eq!(up.first_party_auds, vec!["cli-a", "cli-b"]);
        assert_eq!(up.accepted_kids, vec!["kid-1", "kid_2"]);
        // Upstream tokens only on the REST resource; siblings are exact.
        assert!(c.audience_for(&c.rest_resource).upstream.is_some());
        let mcp = c.audience_for(&c.mcp_resource);
        assert!(mcp.upstream.is_none());
        assert_eq!(mcp.sibling_resources, vec![c.rest_resource.clone()]);
        v.insert("UPSTREAM_FIRST_PARTY", "yes".into());
        assert!(load(&v).is_err());
    }

    /// Regression (loose upstream trust root on the RS): with the upstream
    /// path on, a non-canonical issuer, a JWKS on another origin, a DCR
    /// client id as first-party aud, or a malformed kid pin do not load.
    #[test]
    fn upstream_trust_root_is_strict() {
        let mut base = vars();
        base.insert("UPSTREAM_FIRST_PARTY", "true".into());
        base.insert("FIRST_PARTY_AUDS", "cli".into());
        base.insert("ACCEPTED_UPSTREAM_KIDS", "kid-1".into());
        assert!(load(&base).is_ok());
        let cases: [(&str, &str); 8] = [
            ("UPSTREAM_ISSUER", "https://auth.cognitum.one/"),
            ("UPSTREAM_ISSUER", "http://auth.cognitum.one"),
            ("UPSTREAM_ISSUER", "https://as.example.workers.dev"),
            (
                "UPSTREAM_JWKS_URL",
                "https://attacker.example/.well-known/jwks.json",
            ),
            ("FIRST_PARTY_AUDS", "cli,dcr-abc"),
            ("FIRST_PARTY_AUDS", "edc-123"),
            ("ACCEPTED_UPSTREAM_KIDS", "bad/kid"),
            ("ACCEPTED_UPSTREAM_KIDS", "a,b,c,d,e,f,g,h,i"),
        ];
        for (k, bad) in cases {
            let mut v = base.clone();
            v.insert(k, bad.into());
            assert!(load(&v).is_err(), "{k}={bad} accepted");
        }
    }

    #[test]
    fn rejects_inconsistent_trust_roots() {
        let cases: [(&str, &str); 5] = [
            (
                "EDGE_JWKS_URL",
                "https://attacker.example/.well-known/jwks.json",
            ),
            ("EDGE_ISSUER", "https://as.example.workers.dev/"),
            ("EDGE_ISSUER", "http://as.example.workers.dev"),
            ("PUBLIC_ORIGIN", "https://gw.example.workers.dev/app"),
            ("PUBLIC_ORIGIN", "http://gw.example.workers.dev"),
        ];
        for (k, bad) in cases {
            let mut v = vars();
            v.insert(k, bad.into());
            assert!(load(&v).is_err(), "{k}={bad} accepted");
        }
    }
}
