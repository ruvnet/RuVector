//! Gateway configuration. The trust root is compiled in
//! ([`crate::trust_root`]); only the `UPSTREAM_FIRST_PARTY` flag is a
//! wrangler var in release builds. Every value is validated at the boundary;
//! nothing comes from runtime discovery.

use crate::trust_root::TrustRoot;
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

/// Why the configuration could not be built.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum ConfigError {
    /// Release build: a trust-root var contradicts the compiled const
    /// (503 `trust_root_mismatch`, ADR-351 §5.4 item 5).
    TrustRootMismatch(&'static str),
    /// Anything else (500 `server_error`).
    Invalid(AuthError),
}

impl From<AuthError> for ConfigError {
    fn from(e: AuthError) -> Self {
        ConfigError::Invalid(e)
    }
}

/// Client-id prefixes minted by DCR (upstream `dcr-`, edge `edc-`): never
/// first-party, so never accepted in `FIRST_PARTY_AUDS`.
const DCR_PREFIXES: [&str; 2] = ["dcr-", "edc-"];
/// Bounds on the pinned upstream `kid` list.
const MAX_PINNED_KIDS: usize = 8;
const MAX_KID_LEN: usize = 128;

/// The §5.5 upstream first-party policy (only when explicitly enabled).
///
/// Trust root (the AS's `UpstreamConfig::trust_root_ok` rule, tightened):
/// upstream issuer canonical `https` (no trailing slash) and distinct from
/// the edge issuer; upstream JWKS URL exactly issuer + JWKS path (same
/// origin). First-party auds: non-empty exact ids, never a DCR id. Pinned
/// `kid`s: non-empty base64url (1..=8). With the compiled (empty) auds and
/// kids this fails closed, so the flag alone cannot open the path.
fn upstream_policy(root: &TrustRoot) -> Result<UpstreamFirstPartyPolicy, AuthError> {
    let issuer = root.upstream_issuer.clone();
    if ResourceUrl::parse(&issuer)?.as_str() != issuer {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_ISSUER must be canonical",
        ));
    }
    if issuer == root.edge_issuer {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_ISSUER equals EDGE_ISSUER",
        ));
    }
    if root.upstream_jwks_url != format!("{issuer}{EDGE_JWKS_PATH}") {
        return Err(AuthError::InvalidConfig(
            "UPSTREAM_JWKS_URL must be issuer + JWKS path",
        ));
    }
    let auds = root.first_party_auds.clone();
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
    let kids = root.accepted_kids.clone();
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
    /// Build from a trust root and the `UPSTREAM_FIRST_PARTY` flag
    /// (`"true"`/`"false"`).
    ///
    /// Checks: `public_origin` a bare canonical https origin; `edge_issuer`
    /// canonical; `edge_jwks_url` = `edge_issuer` + [`EDGE_JWKS_PATH`]; the
    /// upstream path per [`upstream_policy`] only when the flag is `"true"`.
    pub fn from_trust_root(
        root: &TrustRoot,
        upstream_first_party: &str,
    ) -> Result<Self, AuthError> {
        let origin = &root.public_origin;
        let origin_url = ResourceUrl::parse(origin)?;
        if origin_url.as_str() != origin || !origin_url.path().is_empty() {
            return Err(AuthError::InvalidConfig(
                "PUBLIC_ORIGIN must be a bare origin",
            ));
        }
        let rest_resource = ResourceUrl::parse(&format!("{origin}/v1"))?;
        let mcp_resource = ResourceUrl::parse(&format!("{origin}/v1/mcp"))?;
        let edge_issuer = root.edge_issuer.clone();
        if ResourceUrl::parse(&edge_issuer)?.as_str() != edge_issuer {
            return Err(AuthError::InvalidConfig("EDGE_ISSUER must be canonical"));
        }
        if root.edge_jwks_url != format!("{edge_issuer}{EDGE_JWKS_PATH}") {
            return Err(AuthError::InvalidConfig(
                "EDGE_JWKS_URL must be issuer + JWKS path",
            ));
        }
        let upstream = match upstream_first_party {
            "true" => Some(upstream_policy(root)?),
            "false" => None,
            _ => {
                return Err(AuthError::InvalidConfig(
                    "UPSTREAM_FIRST_PARTY must be true|false",
                ))
            }
        };
        Ok(GatewayConfig {
            edge_issuer,
            edge_jwks_url: root.edge_jwks_url.clone(),
            rest_resource,
            mcp_resource,
            upstream,
            upstream_jwks_url: root.upstream_jwks_url.clone(),
        })
    }

    /// Release configuration: the compiled [`TrustRoot`] plus the
    /// `UPSTREAM_FIRST_PARTY` var (absent -> `"false"`). A trust-root var
    /// that is present and differs from its const is
    /// [`ConfigError::TrustRootMismatch`].
    #[cfg_attr(feature = "dev-issuer", allow(dead_code))]
    pub fn release(get: &dyn Fn(&str) -> Option<String>) -> Result<Self, ConfigError> {
        let root = TrustRoot::compiled();
        if let Some(name) = root.contradiction(get) {
            return Err(ConfigError::TrustRootMismatch(name));
        }
        let flag = get("UPSTREAM_FIRST_PARTY").unwrap_or_else(|| "false".into());
        Ok(GatewayConfig::from_trust_root(&root, flag.trim())?)
    }

    /// Dev/integration configuration (feature `dev-issuer`, and tests): the
    /// whole trust root from vars (`PUBLIC_ORIGIN`, `EDGE_ISSUER`,
    /// `EDGE_JWKS_URL`, `UPSTREAM_ISSUER`, `UPSTREAM_JWKS_URL`,
    /// `FIRST_PARTY_AUDS`, `ACCEPTED_UPSTREAM_KIDS`) plus
    /// `UPSTREAM_FIRST_PARTY`, all required. Never compiled into release.
    #[cfg(any(test, feature = "dev-issuer"))]
    pub fn from_vars(get: &dyn Fn(&str) -> Option<String>) -> Result<Self, AuthError> {
        let missing = AuthError::InvalidConfig("missing var");
        let root = TrustRoot::from_vars(get).ok_or(missing.clone())?;
        let flag = get("UPSTREAM_FIRST_PARTY").ok_or(missing)?;
        GatewayConfig::from_trust_root(&root, &flag)
    }

    /// Read from the Worker environment: [`GatewayConfig::release`], or
    /// [`GatewayConfig::from_vars`] under `dev-issuer`.
    pub fn from_env(env: &worker::Env) -> Result<Self, ConfigError> {
        let get = |name: &str| env.var(name).ok().map(|v| v.to_string());
        #[cfg(feature = "dev-issuer")]
        return Ok(GatewayConfig::from_vars(&get)?);
        #[cfg(not(feature = "dev-issuer"))]
        GatewayConfig::release(&get)
    }

    /// Audience policy for a resource served by this gateway: exact `aud`
    /// (any other, including the gateway's other resource, is a 401
    /// audience mismatch, ADR-351 §5.4.7), and the upstream first-party path
    /// only on the REST resource (§5.5).
    pub fn audience_for(&self, resource: &ResourceUrl) -> AudiencePolicy {
        let rest = *resource == self.rest_resource;
        AudiencePolicy {
            edge_issuer: self.edge_issuer.clone(),
            resource: resource.clone(),
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
        // Upstream tokens only on the REST resource; audiences are exact.
        assert!(c.audience_for(&c.rest_resource).upstream.is_some());
        let mcp = c.audience_for(&c.mcp_resource);
        assert!(mcp.upstream.is_none());
        assert_eq!(mcp.resource, c.mcp_resource);
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

    fn release(v: &HashMap<&'static str, String>) -> Result<GatewayConfig, ConfigError> {
        GatewayConfig::release(&|k| v.get(k).cloned())
    }

    /// `KEY = "value"` lines of the shipped `wrangler.toml` `[vars]` table.
    fn shipped_vars() -> HashMap<&'static str, String> {
        let toml: &'static str = include_str!("../wrangler.toml");
        let mut section = "";
        let mut out = HashMap::new();
        for line in toml.lines().map(str::trim) {
            if line.starts_with('[') {
                section = line;
                continue;
            }
            match line.split_once('=') {
                Some((k, v)) if section == "[vars]" && !line.starts_with('#') => {
                    out.insert(k.trim(), v.trim().trim_matches('"').to_string());
                }
                _ => {}
            }
        }
        out
    }

    /// Regression (ADR-351 §5.4 item 5): release builds use the compiled
    /// trust root; the shipped wrangler carries only the flag.
    #[test]
    fn release_uses_the_compiled_trust_root() {
        let shipped = shipped_vars();
        assert_eq!(
            shipped.keys().copied().collect::<Vec<_>>(),
            ["UPSTREAM_FIRST_PARTY"]
        );
        let c = release(&shipped).unwrap();
        assert_eq!(c.edge_issuer, crate::trust_root::EDGE_ISSUER);
        assert_eq!(c.edge_jwks_url, crate::trust_root::EDGE_JWKS_URL);
        assert_eq!(
            c.rest_resource.as_str(),
            "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1"
        );
        assert_eq!(
            c.mcp_resource.as_str(),
            "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp"
        );
        assert!(c.upstream.is_none());
        assert!(release(&HashMap::new()).unwrap().upstream.is_none());
    }

    /// Regression (§5.4 item 5): a release build that sees a contradicting
    /// trust-root var is 503 `trust_root_mismatch`, never silently re-rooted;
    /// an agreeing var is harmless.
    #[test]
    fn release_refuses_contradicting_trust_root_vars() {
        for (k, bad) in [
            ("EDGE_ISSUER", "https://attacker.example"),
            (
                "EDGE_JWKS_URL",
                "https://attacker.example/.well-known/jwks.json",
            ),
            ("PUBLIC_ORIGIN", "https://gw.example.workers.dev"),
            ("UPSTREAM_ISSUER", "https://evil-idp.example"),
            ("FIRST_PARTY_AUDS", "cli"),
            ("ACCEPTED_UPSTREAM_KIDS", "kid-1"),
        ] {
            let mut v = shipped_vars();
            v.insert(k, bad.into());
            assert_eq!(
                release(&v).unwrap_err(),
                ConfigError::TrustRootMismatch(k),
                "{k}"
            );
        }
        let mut v = shipped_vars();
        v.insert("EDGE_ISSUER", crate::trust_root::EDGE_ISSUER.into());
        assert!(release(&v).is_ok());
    }

    /// With no compiled first-party auds or kid pin, the flag alone cannot
    /// open the §5.5 path: release config fails closed (500).
    #[test]
    fn release_upstream_flag_fails_closed_without_compiled_auds() {
        let mut v = shipped_vars();
        v.insert("UPSTREAM_FIRST_PARTY", "true".into());
        assert!(matches!(release(&v), Err(ConfigError::Invalid(_))));
    }

    /// Regression (Cloudflare 1042): the edge JWKS comes through the
    /// `EDGE_AUTH` Service Binding to `ruvector-edge-auth`, and the gateway
    /// declares no routes.
    #[test]
    fn wrangler_binds_edge_auth_service() {
        let toml = include_str!("../wrangler.toml");
        let services = toml.split("[[services]]").nth(1).expect("[[services]]");
        let services = services.split("\n[").next().unwrap();
        assert!(services.contains("binding = \"EDGE_AUTH\""));
        assert!(services.contains("service = \"ruvector-edge-auth\""));
        assert!(!toml.lines().any(|l| l.trim_start().starts_with("routes")));
        assert!(!toml.lines().any(|l| l.trim() == "[[routes]]"));
    }
}
