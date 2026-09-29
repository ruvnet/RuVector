//! params, pkce, resource allowlist, metadata, error mapping.

use super::assert_code;
use crate::error::{OAuthError, OAuthErrorCode as C};
use crate::metadata::{jwks_document, AuthorizationServerMetadata};
use crate::params::{ensure_subset, split_scope, Params, MAX_PARAMS, MAX_PARAM_LEN};
use crate::pkce;
use crate::testing::*;
use crate::{ResourceAllowlist, StoreError};

#[test]
fn params_empty_value_is_omitted_and_repeat_rejected() {
    let p = Params::from_pairs(&pairs(&[("a", ""), ("a", "x")])).unwrap();
    assert_eq!(p.get("a"), Some("x"));
    assert_code(
        Params::from_pairs(&pairs(&[("a", "1"), ("a", "2")])),
        C::InvalidRequest,
    );
    assert_code(
        Params::from_pairs(&pairs(&[("resource", RESOURCE), ("resource", RESOURCE)])),
        C::InvalidTarget,
    );
}

#[test]
fn params_bounds() {
    let long = "x".repeat(MAX_PARAM_LEN + 1);
    assert_code(
        Params::from_pairs(&pairs(&[("a", &long)])),
        C::InvalidRequest,
    );
    let many: Vec<(String, String)> = (0..=MAX_PARAMS)
        .map(|i| (format!("k{i}"), "v".to_string()))
        .collect();
    assert_code(Params::from_pairs(&many), C::InvalidRequest);
    assert_code(
        Params::default().require("x", "x required"),
        C::InvalidRequest,
    );
}

#[test]
fn scope_parsing() {
    assert_eq!(
        split_scope("ruvector:read ruvector:write ruvector:read").unwrap(),
        vec!["ruvector:read", "ruvector:write"]
    );
    for bad in [
        "",
        " ruvector:read",
        "ruvector:read ",
        "a  b",
        "a\"b",
        "a\\b",
        "a\tb",
        "é",
    ] {
        assert_code(split_scope(bad), C::InvalidScope);
    }
    let too_many: Vec<String> = (0..33).map(|i| format!("s{i}")).collect();
    assert_code(split_scope(&too_many.join(" ")), C::InvalidScope);
    let ceiling = vec!["a".to_string(), "b".to_string()];
    assert!(ensure_subset(&["a".to_string()], &ceiling).is_ok());
    assert_code(ensure_subset(&["c".to_string()], &ceiling), C::InvalidScope);
}

#[test]
fn pkce_rfc7636_vector_and_rules() {
    assert_eq!(
        pkce::challenge_s256(VERIFIER),
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM"
    );
    assert!(pkce::verify_s256(VERIFIER, &challenge()));
    assert!(!pkce::verify_s256(
        VERIFIER,
        "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cN"
    ));
    assert!(!pkce::verify_s256("short", &pkce::challenge_s256("short")));
    assert!(!pkce::verify_s256(
        &"a".repeat(129),
        &pkce::challenge_s256(&"a".repeat(129))
    ));
    assert!(pkce::validate_challenge(&challenge(), Some("S256")).is_ok());
    assert_code(
        pkce::validate_challenge(&challenge(), Some("plain")),
        C::InvalidRequest,
    );
    assert_code(
        pkce::validate_challenge(&challenge(), None),
        C::InvalidRequest,
    );
    assert_code(
        pkce::validate_challenge("abc", Some("S256")),
        C::InvalidRequest,
    );
    assert_code(
        pkce::validate_challenge(&format!("{}+", &challenge()[..42]), Some("S256")),
        C::InvalidRequest,
    );
    assert_code(pkce::validate_verifier("a b"), C::InvalidGrant);
}

#[test]
fn allowlist_resolution() {
    let a = allowlist();
    assert_eq!(a.resolve(Some(RESOURCE)).unwrap(), resource());
    let upper = RESOURCE.replace("ruvector-edge-gateway", "RUVECTOR-edge-gateway");
    assert_eq!(a.resolve(Some(&upper)).unwrap(), resource());
    for bad in [
        None,
        Some("https://evil.example/v1/mcp"),
        Some("http://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp"),
        Some(&format!("{RESOURCE}/")),
        Some(&format!("{RESOURCE}?x=1")),
    ] {
        assert_code(a.resolve(bad), C::InvalidTarget);
    }
    assert!(ResourceAllowlist::from_config(&format!("{RESOURCE}, {OTHER_RESOURCE}")).is_ok());
    assert!(ResourceAllowlist::from_config("https://ok.example,http://bad.example").is_err());
    assert_code(
        ResourceAllowlist::default().resolve(Some(RESOURCE)),
        C::InvalidTarget,
    );
}

#[test]
fn metadata_advertises_s256_none_and_rfc9207() {
    let m = AuthorizationServerMetadata::build(ISSUER, &["ruvector:read".to_string()]);
    let v = serde_json::to_value(&m).unwrap();
    assert_eq!(v["issuer"], ISSUER);
    assert_eq!(v["token_endpoint"], format!("{ISSUER}/token"));
    assert_eq!(v["registration_endpoint"], format!("{ISSUER}/register"));
    assert_eq!(v["revocation_endpoint"], format!("{ISSUER}/revoke"));
    assert_eq!(v["jwks_uri"], format!("{ISSUER}/.well-known/jwks.json"));
    assert_eq!(
        v["code_challenge_methods_supported"],
        serde_json::json!(["S256"])
    );
    assert_eq!(v["response_types_supported"], serde_json::json!(["code"]));
    assert_eq!(
        v["token_endpoint_auth_methods_supported"],
        serde_json::json!(["none"])
    );
    assert_eq!(v["authorization_response_iss_parameter_supported"], true);
    let jwks = jwks_document(&[TestSigner::default().verifying_key()]);
    assert_eq!(jwks.keys.len(), 1);
}

#[test]
fn errors_map_to_status_and_json() {
    assert_eq!(C::InvalidClient.http_status(), 401);
    assert_eq!(C::ServerError.http_status(), 500);
    assert_eq!(C::TemporarilyUnavailable.http_status(), 503);
    assert_eq!(C::InvalidGrant.http_status(), 400);
    let e: OAuthError = StoreError("disk".into()).into();
    assert_eq!(e.error, C::ServerError);
    assert_eq!(
        OAuthError::new(C::InvalidTarget, "x").to_json(),
        r#"{"error":"invalid_target","error_description":"x"}"#
    );
}

#[test]
fn secrets_are_distinct_and_hashed() {
    let rng = SeqRng::default();
    let a = crate::random_secret(&rng, 32).unwrap();
    let b = crate::random_secret(&rng, 32).unwrap();
    assert_ne!(a, b);
    assert_eq!(a.len(), 43);
    assert_ne!(crate::secret_hash(&a), crate::secret_hash(&b));
}
