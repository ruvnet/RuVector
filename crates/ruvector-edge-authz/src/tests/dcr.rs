//! Dynamic Client Registration (RFC 7591).

use super::assert_code;
use crate::client::*;
use crate::error::OAuthErrorCode as C;
use crate::ports::{MockClientStore, MockRng};
use crate::testing::*;
use crate::StoreError;

fn validate(req: &RegistrationRequest) -> Result<ClientRecord, crate::OAuthError> {
    validate_registration(req, &policy(), "edc-x".into(), T0)
}

#[test]
fn valid_registration_builds_record() {
    let r = validate(&reg_request()).unwrap();
    assert_eq!(r.client_id, "edc-x");
    assert_eq!(r.redirect_uris[0].as_str(), REDIRECT);
    assert_eq!(r.grant_types, vec!["authorization_code", "refresh_token"]);
    assert_eq!(r.client_id_issued_at, T0);
    let body = serde_json::to_value(r.to_response()).unwrap();
    assert_eq!(body["token_endpoint_auth_method"], "none");
    assert_eq!(body["response_types"], serde_json::json!(["code"]));
    assert_eq!(body["scope"], "ruvector:read ruvector:write offline_access");
}

#[test]
fn omitted_scope_gets_default_not_full_ceiling() {
    let r = validate(&reg_request()).unwrap();
    assert_eq!(r.scope, policy().default_scope);
    assert_ne!(r.scope, policy().scopes_supported);
}

#[test]
fn explicit_scope_subset_and_defaults() {
    let mut req = reg_request();
    req.scope = Some("ruvector:read ruvector:write".into());
    req.grant_types = None;
    req.response_types = Some(vec!["code".into()]);
    req.token_endpoint_auth_method = Some("none".into());
    let r = validate(&req).unwrap();
    assert_eq!(r.scope, vec!["ruvector:read", "ruvector:write"]);
    // ADR-351 §5.6: omitted grant_types -> both, so connectors refresh.
    assert_eq!(r.grant_types, vec!["authorization_code", "refresh_token"]);
    req.grant_types = Some(vec![
        "authorization_code".into(),
        "authorization_code".into(),
    ]);
    assert_eq!(validate(&req).unwrap().grant_types.len(), 1);
}

#[test]
fn rejected_metadata() {
    let cases: Vec<super::Case<RegistrationRequest>> = vec![
        (
            |r| r.scope = Some("brains:read".into()),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.scope = Some("ruvector:read  x".into()),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.token_endpoint_auth_method = Some("client_secret_basic".into()),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.response_types = Some(vec!["token".into()]),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.response_types = Some(vec!["code".into(), "token".into()]),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.grant_types = Some(vec!["implicit".into()]),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.grant_types = Some(vec!["refresh_token".into()]),
            C::InvalidClientMetadata,
        ),
        (|r| r.grant_types = Some(vec![]), C::InvalidClientMetadata),
        (
            |r| r.grant_types = Some(vec!["client_credentials".into()]),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.client_name = Some(String::new()),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.client_name = Some("a\u{7}b".into()),
            C::InvalidClientMetadata,
        ),
        (
            |r| r.client_name = Some("n".repeat(129)),
            C::InvalidClientMetadata,
        ),
        (|r| r.redirect_uris = vec![], C::InvalidRedirectUri),
        (
            |r| r.redirect_uris = (0..9).map(|i| format!("https://a.example/{i}")).collect(),
            C::InvalidRedirectUri,
        ),
        (
            |r| r.redirect_uris = vec![REDIRECT.into(), REDIRECT.into()],
            C::InvalidRedirectUri,
        ),
        (
            |r| r.redirect_uris = vec!["http://evil.example/cb".into()],
            C::InvalidRedirectUri,
        ),
    ];
    for (mutate, code) in cases {
        let mut req = reg_request();
        mutate(&mut req);
        assert_code(validate(&req), code);
    }
}

#[test]
fn redirect_uri_rules() {
    for ok in [
        "https://claude.ai/api/mcp/auth_callback",
        "https://app.example:8443/cb?x=1",
        "http://127.0.0.1:53682/callback",
        "http://127.0.0.1/callback",
        "http://[::1]:8080/cb",
    ] {
        assert!(validate_redirect_uri(ok).is_ok(), "{ok}");
    }
    let long = format!("https://a.example/{}", "x".repeat(MAX_REDIRECT_URI_LEN));
    for bad in [
        "",
        "/relative/cb",
        "http://localhost:8080/cb",
        "http://example.com/cb",
        "http://127.0.0.2/cb",
        "http://0x7f.0.0.1/cb",
        "http://127.1/cb",
        "http://127.0.0.1.evil.example/cb",
        "http://[0:0:0:0:0:0:0:1]/cb",
        "https://a.example/cb#frag",
        "https://a.example/cb#",
        "https://user:pw@a.example/cb",
        "https://user@a.example/cb",
        "https://a.example/c\tb",
        "https://a.example/c\nb",
        "https://a.example/c b",
        "com.example.app:/cb",
        "javascript:alert(1)",
        "file:///etc/passwd",
        "ftp://a.example/cb",
        long.as_str(),
    ] {
        assert_code(validate_redirect_uri(bad), C::InvalidRedirectUri);
    }
}

#[test]
fn redirect_matching_exact_and_loopback_port_agnostic() {
    let https = validate_redirect_uri(REDIRECT).unwrap();
    assert!(https.matches(REDIRECT));
    assert!(!https.matches("https://claude.ai:444/api/mcp/auth_callback"));
    assert!(!https.matches("https://claude.ai/api/mcp/auth_callback/"));
    assert!(!https.matches("https://CLAUDE.ai/api/mcp/auth_callback"));
    assert!(!https.matches("https://claude.ai/api/mcp/auth_callback?x=1"));
    let lo = validate_redirect_uri(LOOPBACK).unwrap();
    assert!(lo.matches("http://127.0.0.1:12345/callback"));
    assert!(lo.matches("http://127.0.0.1/callback"));
    assert!(!lo.matches("http://127.0.0.1:12345/steal"));
    assert!(!lo.matches("http://localhost:12345/callback"));
    assert!(!lo.matches("http://[::1]:12345/callback"));
    assert!(!lo.matches("https://127.0.0.1:12345/callback"));
    assert!(!lo.matches("http://127.0.0.1:12345/callback?x=1"));
    assert!(!lo.matches("http://127.0.0.1:12345/callback#f"));
    assert!(!lo.matches("http://u@127.0.0.1:12345/callback"));
    assert!(!lo.matches("http://127.0.0.1:12345/call\tback"));
}

#[test]
fn from_json_bounds_and_types() {
    let ok = br#"{"redirect_uris":["https://a.example/cb"],"extra":{"ignored":true}}"#;
    assert_eq!(
        RegistrationRequest::from_json(ok)
            .unwrap()
            .redirect_uris
            .len(),
        1
    );
    assert_code(
        RegistrationRequest::from_json(b"{"),
        C::InvalidClientMetadata,
    );
    assert_code(
        RegistrationRequest::from_json(br#"{"redirect_uris":"https://a.example/cb"}"#),
        C::InvalidClientMetadata,
    );
    let big = vec![b' '; MAX_REGISTRATION_BODY + 1];
    assert_code(
        RegistrationRequest::from_json(&big),
        C::InvalidClientMetadata,
    );
}

#[test]
fn register_client_persists_with_prefixed_random_id() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let a = register_client(&store, &rng, &clock, &policy(), &reg_request()).unwrap();
    let b = register_client(&store, &rng, &clock, &policy(), &reg_request()).unwrap();
    assert!(a.client_id.starts_with(CLIENT_ID_PREFIX));
    assert_eq!(a.client_id.len(), CLIENT_ID_PREFIX.len() + 22);
    assert_ne!(a.client_id, b.client_id);
    assert_eq!(store.clients.borrow().len(), 2);
}

#[test]
fn register_client_stores_nothing_on_invalid_request() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let mut req = reg_request();
    req.redirect_uris = vec!["http://evil.example/cb".into()];
    assert_code(
        register_client(&store, &rng, &clock, &policy(), &req),
        C::InvalidRedirectUri,
    );
    assert!(store.clients.borrow().is_empty());
}

#[test]
fn register_client_rng_failure_aborts_before_store() {
    let mut rng = MockRng::new();
    rng.expect_fill()
        .returning(|_| Err(StoreError("no entropy".into())));
    let mut store = MockClientStore::new();
    store.expect_insert_client().times(0);
    assert_code(
        register_client(&store, &rng, &FixedClock::at(T0), &policy(), &reg_request()),
        C::ServerError,
    );
}

#[test]
fn register_client_store_failure_is_server_error() {
    let mut store = MockClientStore::new();
    store
        .expect_insert_client()
        .times(1)
        .returning(|_| Err(StoreError("sqlite".into())));
    assert_code(
        register_client(
            &store,
            &SeqRng::default(),
            &FixedClock::at(T0),
            &policy(),
            &reg_request(),
        ),
        C::ServerError,
    );
}
