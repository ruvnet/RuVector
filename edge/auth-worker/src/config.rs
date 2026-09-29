//! Authorization-server configuration from wrangler `[vars]` (validated;
//! nothing from runtime discovery). The ES256 signing key is a secret
//! (`EDGE_AUTH_SIGNING_JWK`), read separately, never from vars.

use ruvector_edge_authz::client::DcrPolicy;
use ruvector_edge_authz::federation::UpstreamConfig;
use ruvector_edge_authz::{OAuthError, ResourceAllowlist, ResourceUrl};

/// Default cap on dynamically registered clients (abuse bound on a public,
/// unauthenticated write endpoint).
pub const DEFAULT_MAX_CLIENTS: u64 = 10_000;

/// Validated AS configuration.
#[derive(Debug, Clone)]
pub struct AuthConfig {
    /// This AS's issuer URL (canonical, no trailing slash).
    pub issuer: String,
    /// Resources tokens may be minted for (exact canonical URLs).
    pub resources: ResourceAllowlist,
    /// Scopes advertised and registrable.
    pub scopes_supported: Vec<String>,
    /// Upstream IdP (`auth.cognitum.one`) client configuration.
    pub upstream: UpstreamConfig,
    /// DCR cap (counts live clients; idle ones are purged first).
    pub max_clients: u64,
    /// Registrations accepted per IP bucket per hour.
    pub dcr_rate_per_hour: u64,
    /// Ceiling given to a registration that omits `scope` (read-only).
    pub default_scope: Vec<String>,
    /// Upstream RFC 7009 endpoint: the upstream refresh token is revoked
    /// right after the callback (ADR-351 §5.6 step 4).
    pub upstream_revocation_endpoint: String,
    /// Pinned upstream signing `kid`s (`ACCEPTED_UPSTREAM_KIDS`). Empty
    /// means federation is not ready (like an empty `UPSTREAM_CLIENT_ID`).
    pub upstream_accepted_kids: Vec<String>,
}

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

/// Default `DEFAULT_SCOPE` candidates (filtered to `SCOPES_SUPPORTED`).
const DEFAULT_SCOPE_FALLBACK: [&str; 2] = ["ruvector:read", "offline_access"];

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
    /// Reads `ISSUER`, `RESOURCE_ALLOWLIST`, `SCOPES_SUPPORTED`,
    /// `UPSTREAM_ISSUER`, `UPSTREAM_AUTHORIZATION_ENDPOINT`,
    /// `UPSTREAM_TOKEN_ENDPOINT`, `UPSTREAM_JWKS_URL`, `UPSTREAM_CLIENT_ID`
    /// (may be empty: federation then answers `temporarily_unavailable`),
    /// `UPSTREAM_SCOPES` (identity scopes only, [`UPSTREAM_IDENTITY_SCOPES`])
    /// and optional `MAX_CLIENTS`, `DCR_RATE_PER_HOUR` (>= 1),
    /// `DEFAULT_SCOPE` (non-empty subset of `SCOPES_SUPPORTED`; default
    /// `ruvector:read offline_access` ∩ supported) and
    /// `UPSTREAM_REVOCATION_ENDPOINT` (https, same origin as the upstream
    /// issuer; default `<UPSTREAM_ISSUER>/oauth/revoke`) and
    /// `ACCEPTED_UPSTREAM_KIDS` (comma-separated base64url `kid`s). The upstream
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
        if resources.resources().is_empty() {
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
        let scopes_supported = scope_list(&var("SCOPES_SUPPORTED")?, "SCOPES_SUPPORTED")?;
        let default_scope = match get("DEFAULT_SCOPE") {
            Some(v) if !v.trim().is_empty() => scope_list(&v, "DEFAULT_SCOPE")?,
            _ => DEFAULT_SCOPE_FALLBACK
                .iter()
                .filter(|s| scopes_supported.iter().any(|t| t == *s))
                .map(|s| s.to_string())
                .collect(),
        };
        if default_scope.is_empty() || !default_scope.iter().all(|s| scopes_supported.contains(s)) {
            return Err(crate::config_error("DEFAULT_SCOPE"));
        }
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
        })
    }

    /// Fail closed unless the federation leg is usable:
    /// [`UpstreamConfig::ensure_ready`], then a non-empty upstream `kid` pin
    /// (`temporarily_unavailable` until the operator sets it).
    pub fn ensure_federation_ready(&self) -> Result<(), OAuthError> {
        self.upstream.ensure_ready()?;
        if self.upstream_accepted_kids.is_empty() {
            return Err(OAuthError::new(
                ruvector_edge_authz::OAuthErrorCode::TemporarilyUnavailable,
                "upstream signing keys are not pinned",
            ));
        }
        Ok(())
    }

    /// Read from the Worker environment.
    pub fn from_env(env: &worker::Env) -> Result<Self, OAuthError> {
        AuthConfig::from_vars(&|name| env.var(name).ok().map(|v| v.to_string()))
    }

    /// DCR policy: a registration that omits `scope` gets the read-only
    /// [`AuthConfig::default_scope`] ceiling, never the full supported set
    /// (ADR-351 §5.3); write scopes must be registered explicitly.
    pub fn dcr_policy(&self) -> DcrPolicy {
        DcrPolicy {
            scopes_supported: self.scopes_supported.clone(),
            default_scope: self.default_scope.clone(),
        }
    }
}

#[cfg(test)]
pub(crate) mod tests {
    use super::*;
    use std::collections::HashMap;

    pub(crate) const ISSUER: &str = "https://ruvector-edge-auth.example.workers.dev";
    pub(crate) const RESOURCE: &str = "https://ruvector-edge-gateway.example.workers.dev/v1/mcp";

    pub(crate) fn vars() -> HashMap<&'static str, String> {
        HashMap::from([
            ("ISSUER", ISSUER.to_string()),
            (
                "RESOURCE_ALLOWLIST",
                format!("https://ruvector-edge-gateway.example.workers.dev/v1,{RESOURCE}"),
            ),
            (
                "SCOPES_SUPPORTED",
                "ruvector:read ruvector:write offline_access".into(),
            ),
            ("UPSTREAM_ISSUER", "https://auth.cognitum.one".into()),
            (
                "UPSTREAM_AUTHORIZATION_ENDPOINT",
                "https://auth.cognitum.one/oauth/authorize".into(),
            ),
            (
                "UPSTREAM_TOKEN_ENDPOINT",
                "https://auth.cognitum.one/oauth/token".into(),
            ),
            (
                "UPSTREAM_JWKS_URL",
                "https://auth.cognitum.one/.well-known/jwks.json".into(),
            ),
            ("UPSTREAM_CLIENT_ID", "dcr-edge-test".into()),
            ("UPSTREAM_SCOPES", "openid profile email".into()),
            ("ACCEPTED_UPSTREAM_KIDS", upstream_test_kid()),
        ])
    }

    /// `kid` of the fake upstream IdP key the endpoint tests sign with.
    pub(crate) fn upstream_test_kid() -> String {
        let key = crate::signer::tests::test_key(5);
        ruvector_edge_auth::Jwk::from_verifying_key(key.verifying_key()).kid
    }

    pub(crate) fn load(v: &HashMap<&'static str, String>) -> Result<AuthConfig, OAuthError> {
        AuthConfig::from_vars(&|k| v.get(k).cloned())
    }

    #[test]
    fn loads_the_wrangler_shape() {
        let c = load(&vars()).unwrap();
        assert_eq!(c.upstream.redirect_uri, format!("{ISSUER}/callback"));
        assert_eq!(c.resources.resources().len(), 2);
        assert_eq!(c.max_clients, DEFAULT_MAX_CLIENTS);
        assert_eq!(c.dcr_rate_per_hour, crate::abuse::DEFAULT_DCR_RATE_PER_HOUR);
        assert!(c.upstream.ensure_ready().is_ok());
        assert_eq!(
            c.upstream_revocation_endpoint,
            "https://auth.cognitum.one/oauth/revoke"
        );
    }

    /// `KEY = "value"` lines of the `[vars]` table of the shipped
    /// `wrangler.toml` (flat strings only, which is all it holds).
    pub(crate) fn shipped_vars() -> HashMap<&'static str, String> {
        let toml: &'static str = include_str!("../wrangler.toml");
        let mut in_vars = false;
        let mut out = HashMap::new();
        for line in toml.lines().map(str::trim) {
            if line.starts_with('[') {
                in_vars = line == "[vars]";
                continue;
            }
            let Some((k, v)) = line.split_once('=') else {
                continue;
            };
            if in_vars && !line.starts_with('#') {
                let v = v.trim().trim_matches('"').to_string();
                out.insert(k.trim(), v);
            }
        }
        out
    }

    /// Regression (offline_access refused, ADR-351 §5.3; upstream `mcp:*`
    /// credential, §5.6; full-set default grant): the shipped configuration
    /// loads, advertises and accepts `offline_access`, registers without
    /// `scope` into a read-only default, and asks upstream for identity
    /// scopes only.
    #[test]
    fn shipped_wrangler_vars_are_valid_and_safe() {
        let c = load(&shipped_vars()).expect("shipped wrangler.toml vars load");
        assert!(c.scopes_supported.iter().any(|s| s == "offline_access"));
        let p = c.dcr_policy();
        assert!(p.default_scope.iter().any(|s| s == "offline_access"));
        assert!(!p.default_scope.iter().any(|s| s == "ruvector:write"));
        assert_ne!(p.default_scope, c.scopes_supported);
        assert!(c
            .upstream
            .scopes
            .iter()
            .all(|s| UPSTREAM_IDENTITY_SCOPES.contains(&s.as_str())));
    }

    #[test]
    fn upstream_scopes_are_identity_only() {
        for bad in [
            "openid profile email ruvector:read",
            "openid ruvector:write",
            "offline_access",
        ] {
            let mut v = vars();
            v.insert("UPSTREAM_SCOPES", bad.into());
            assert!(load(&v).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn default_scope_is_a_read_only_subset() {
        let c = load(&vars()).unwrap();
        assert_eq!(c.default_scope, vec!["ruvector:read", "offline_access"]);
        let mut v = vars();
        v.insert("DEFAULT_SCOPE", "ruvector:read".into());
        assert_eq!(
            load(&v).unwrap().dcr_policy().default_scope,
            vec!["ruvector:read"]
        );
        v.insert("DEFAULT_SCOPE", "mcp:admin".into());
        assert!(load(&v).is_err());
        let mut v = vars();
        v.insert("SCOPES_SUPPORTED", "ruvector:write".into());
        assert!(load(&v).is_err(), "empty fallback default must not load");
    }

    #[test]
    fn empty_upstream_client_id_loads_but_is_not_ready() {
        let mut v = vars();
        v.insert("UPSTREAM_CLIENT_ID", "".into());
        assert!(load(&v).unwrap().upstream.ensure_ready().is_err());
        assert!(load(&v).unwrap().ensure_federation_ready().is_err());
    }

    #[test]
    fn upstream_kid_pin_is_parsed_bounded_and_required_for_federation() {
        let mut v = vars();
        v.insert("ACCEPTED_UPSTREAM_KIDS", " k1, k-2 ,k_3 ".into());
        let c = load(&v).unwrap();
        assert_eq!(c.upstream_accepted_kids, vec!["k1", "k-2", "k_3"]);
        assert!(c.ensure_federation_ready().is_ok());
        v.insert("ACCEPTED_UPSTREAM_KIDS", String::new());
        let c = load(&v).unwrap();
        let e = c.ensure_federation_ready().unwrap_err();
        assert_eq!(
            e.error,
            ruvector_edge_authz::OAuthErrorCode::TemporarilyUnavailable
        );
        for bad in [
            "a/b",
            "x".repeat(MAX_KID_LEN + 1).as_str(),
            "a,b,c,d,e,f,g,h,i",
        ] {
            v.insert("ACCEPTED_UPSTREAM_KIDS", bad.to_string());
            assert!(load(&v).is_err(), "{bad} accepted");
        }
    }

    #[test]
    fn rejects_bad_values() {
        let cases: [(&str, &str); 13] = [
            ("ISSUER", "http://as.example"),
            ("ISSUER", "https://as.example/"),
            ("RESOURCE_ALLOWLIST", ""),
            ("RESOURCE_ALLOWLIST", "https://x.example/v1?q=1"),
            (
                "UPSTREAM_TOKEN_ENDPOINT",
                "http://auth.cognitum.one/oauth/token",
            ),
            ("UPSTREAM_JWKS_URL", "https://user@auth.cognitum.one/jwks"),
            ("SCOPES_SUPPORTED", "  "),
            ("UPSTREAM_CLIENT_ID", "has space"),
            ("MAX_CLIENTS", "lots"),
            ("DCR_RATE_PER_HOUR", "0"),
            ("DCR_RATE_PER_HOUR", "many"),
            (
                "UPSTREAM_REVOCATION_ENDPOINT",
                "https://evil.example/oauth/revoke",
            ),
            (
                "UPSTREAM_REVOCATION_ENDPOINT",
                "http://auth.cognitum.one/oauth/revoke",
            ),
        ];
        for (k, bad) in cases {
            let mut v = vars();
            v.insert(k, bad.into());
            assert!(load(&v).is_err(), "{k}={bad} accepted");
        }
        let mut v = vars();
        v.remove("UPSTREAM_ISSUER");
        assert!(load(&v).is_err());
    }
}
