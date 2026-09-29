//! Configuration tests (split out of `config.rs` to keep both files under
//! 500 lines).

use super::*;
use std::collections::HashMap;

pub(crate) const ISSUER: &str = "https://ruvector-edge-auth.example.workers.dev";
pub(crate) const RESOURCE: &str =
    "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp";

pub(crate) fn vars() -> HashMap<&'static str, String> {
    HashMap::from([
        ("ISSUER", ISSUER.to_string()),
        (
            "RESOURCE_ALLOWLIST",
            format!(
                "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 \
                 ruvector:read ruvector:write ruvector:admin offline_access, \
                 {RESOURCE} ruvector:read ruvector:write offline_access"
            ),
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
    assert_eq!(c.resources.entries().len(), 2);
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
/// `scope` into `ruvector:read ruvector:write offline_access` (never
/// admin), and asks upstream for identity scopes only.
#[test]
fn shipped_wrangler_vars_are_valid_and_safe() {
    let c = load(&shipped_vars()).expect("shipped wrangler.toml vars load");
    assert!(c.scopes_supported.iter().any(|s| s == "offline_access"));
    let p = c.dcr_policy();
    assert_eq!(
        p.default_scope,
        vec!["ruvector:read", "ruvector:write", "offline_access"]
    );
    assert_ne!(p.default_scope, c.scopes_supported);
    assert!(c
        .upstream
        .scopes
        .iter()
        .all(|s| UPSTREAM_IDENTITY_SCOPES.contains(&s.as_str())));
    for gone in ["SCOPES_SUPPORTED", "DEFAULT_SCOPE"] {
        assert!(!shipped_vars().contains_key(gone), "{gone} is derived");
    }
}

/// Regression (ADR-351 §5.3, §5.7, §16.1): the shipped allowlist has exactly
/// the two gateway resources (`ruvector:*`, admin only on `/v1`) and the
/// team.ruv.io adapter (`team:*` only), and `scopes_supported` is the union
/// of both vocabularies.
#[test]
fn shipped_allowlist_has_per_resource_scopes() {
    let c = load(&shipped_vars()).unwrap();
    let gw = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev";
    let got: Vec<(String, Vec<String>)> = c
        .resources
        .entries()
        .iter()
        .map(|e| (e.url().as_str().to_string(), e.scopes().to_vec()))
        .collect();
    let want = |url: String, s: &[&str]| (url, s.iter().map(|x| x.to_string()).collect());
    assert_eq!(
        got,
        vec![
            want(
                format!("{gw}/v1"),
                &[
                    "ruvector:read",
                    "ruvector:write",
                    "ruvector:admin",
                    "offline_access"
                ]
            ),
            want(
                format!("{gw}/v1/mcp"),
                &["ruvector:read", "ruvector:write", "offline_access"]
            ),
            want(
                "https://team.ruv.io/mcp".into(),
                &["team:read", "team:write", "team:run", "offline_access"]
            ),
        ]
    );
    let mut sup = c.scopes_supported.clone();
    sup.sort_unstable();
    assert_eq!(
        sup,
        [
            "offline_access",
            "ruvector:admin",
            "ruvector:read",
            "ruvector:write",
            "team:read",
            "team:run",
            "team:write",
        ]
    );
    // G1 pending (console#605): no upstream client and no kid pin yet.
    assert!(c.upstream.client_id.is_empty());
    assert!(c.upstream_accepted_kids.is_empty());
    let e = c.ensure_federation_ready().unwrap_err();
    assert_eq!(
        e.error,
        ruvector_edge_authz::OAuthErrorCode::TemporarilyUnavailable
    );
    assert_eq!(e.error_description, FEDERATION_PENDING);
}

/// Regression (ADR-351 §5.3, §16.1): the URL → vocabulary binding is
/// compiled, so the allowlist value deployed at M0.5 (team.ruv.io with
/// `ruvector:*`), an adapter URL that is not compiled, or a gateway resource
/// with `team:*` can no longer load through the mutable var.
#[test]
fn allowlist_var_cannot_rebind_a_resource_vocabulary() {
    let gw = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev";
    let base = format!(
        "{gw}/v1 ruvector:read ruvector:write ruvector:admin offline_access, \
         {gw}/v1/mcp ruvector:read ruvector:write offline_access"
    );
    for tail in [
        "https://team.ruv.io/mcp ruvector:read ruvector:write offline_access".to_string(),
        "https://any-adapter.example/mcp ruvector:read ruvector:admin".to_string(),
        "https://any-adapter.example/mcp team:read".to_string(),
    ] {
        let mut v = shipped_vars();
        v.insert("RESOURCE_ALLOWLIST", format!("{base}, {tail}"));
        assert!(load(&v).is_err(), "{tail} accepted");
    }
    let mut v = shipped_vars();
    v.insert(
        "RESOURCE_ALLOWLIST",
        format!("{gw}/v1 team:read team:write offline_access"),
    );
    assert!(load(&v).is_err(), "/v1 with team:* accepted");
    // Narrowing through the var is still possible.
    let mut v = shipped_vars();
    v.insert("RESOURCE_ALLOWLIST", format!("{gw}/v1/mcp ruvector:read"));
    assert!(load(&v).is_ok());
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

/// The DCR default ceiling is the compiled `DEFAULT_CLIENT_SCOPE` ∩ the
/// allowlist's scopes; an allowlist without any `ruvector:*` scope does not
/// load (a default registration could never be granted anything).
#[test]
fn default_scope_is_the_compiled_ceiling() {
    let c = load(&vars()).unwrap();
    assert_eq!(
        c.default_scope,
        vec!["ruvector:read", "ruvector:write", "offline_access"]
    );
    let mut v = vars();
    v.insert(
        "RESOURCE_ALLOWLIST",
        format!("{RESOURCE} ruvector:read offline_access"),
    );
    assert_eq!(
        load(&v).unwrap().dcr_policy().default_scope,
        vec!["ruvector:read", "offline_access"]
    );
    // A team-only allowlist loads its entry but leaves the compiled default
    // ceiling empty, so it fails; a team entry mixing families fails as
    // resource scopes; so does an allowlist whose entries are all admin-only.
    v.insert(
        "RESOURCE_ALLOWLIST",
        "https://team.ruv.io/mcp team:read".into(),
    );
    assert!(load(&v).is_err(), "no default-ceiling ruvector scope");
    v.insert(
        "RESOURCE_ALLOWLIST",
        format!("{RESOURCE} ruvector:read, https://team.ruv.io/mcp team:read ruvector:write"),
    );
    assert!(load(&v).is_err(), "mixed vocabulary entry");
    v.insert(
        "RESOURCE_ALLOWLIST",
        format!("{RESOURCE} ruvector:read, https://team.ruv.io/mcp team:read"),
    );
    let c = load(&v).unwrap();
    assert_eq!(c.scopes_supported, vec!["ruvector:read", "team:read"]);
    assert_eq!(c.dcr_policy().default_scope, vec!["ruvector:read"]);
    v.insert("RESOURCE_ALLOWLIST", format!("{RESOURCE} ruvector:admin"));
    assert!(load(&v).is_err(), "no default-ceiling ruvector scope");
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
        ("RESOURCE_ALLOWLIST", "https://x.example/v1"),
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

/// ADR-351 §5.6: confidential exchange clients come only from the operator
/// registry var, validated whole at load; the shipped config registers none.
#[test]
fn confidential_clients_are_operator_config_validated_at_load() {
    let shipped = load(&shipped_vars()).unwrap();
    assert!(shipped.confidential_clients.clients().is_empty());
    assert!(load(&vars())
        .unwrap()
        .confidential_clients
        .clients()
        .is_empty());

    let key = crate::signer::tests::test_key(11);
    let j = ruvector_edge_auth::Jwk::from_verifying_key(key.verifying_key());
    let entry = |id: &str, aud: &str| {
        serde_json::json!([{
            "client_id": id,
            "jwk": {"kty": "EC", "crv": "P-256", "x": j.x, "y": j.y, "kid": j.kid},
            "subject_audiences": [aud],
            "scope": "ruvector:read",
        }])
        .to_string()
    };
    let mut v = vars();
    let allow = format!(
        "{}, https://team.ruv.io/mcp team:read team:write",
        v["RESOURCE_ALLOWLIST"]
    );
    v.insert("RESOURCE_ALLOWLIST", allow);
    v.insert(
        "CONFIDENTIAL_CLIENTS",
        entry("team-ruv-io", "https://team.ruv.io/mcp"),
    );
    let c = load(&v).unwrap();
    assert!(c.confidential_clients.get("team-ruv-io").is_some());
    for bad in [
        entry("edc-0123", "https://team.ruv.io/mcp"),
        entry(
            "team-ruv-io",
            "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1",
        ),
        "{not json".to_string(),
    ] {
        v.insert("CONFIDENTIAL_CLIENTS", bad);
        let e = load(&v).unwrap_err();
        assert_eq!(e.error_description, "CONFIDENTIAL_CLIENTS");
    }
}
