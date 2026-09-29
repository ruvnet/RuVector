//! Authorization request validation and redirects.

use crate::authorize::*;
use crate::error::OAuthErrorCode as C;
use crate::testing::*;

fn request() -> AuthorizationRequest {
    AuthorizationRequest {
        response_type: Some("code".into()),
        client_id: Some(CLIENT_ID.into()),
        redirect_uri: Some(REDIRECT.into()),
        scope: Some("ruvector:read".into()),
        state: Some("xyz".into()),
        code_challenge: Some(challenge()),
        code_challenge_method: Some("S256".into()),
        resource: Some(RESOURCE.into()),
    }
}

fn run(req: &AuthorizationRequest) -> Result<ValidatedAuthorization, AuthorizeError> {
    validate_authorization(req, &client(&["authorization_code"]), &allowlist())
}

#[track_caller]
fn user_agent(r: Result<ValidatedAuthorization, AuthorizeError>, code: C) {
    match r {
        Err(AuthorizeError::UserAgent(e)) => assert_eq!(e.error, code),
        other => panic!("expected user-agent {code:?}, got {other:?}"),
    }
}

#[track_caller]
fn redirected(r: Result<ValidatedAuthorization, AuthorizeError>, code: C) -> AuthorizeError {
    match r {
        Err(e @ AuthorizeError::Redirect { .. }) => {
            assert_eq!(e.oauth().error, code);
            e
        }
        other => panic!("expected redirect {code:?}, got {other:?}"),
    }
}

#[test]
fn valid_request() {
    let v = run(&request()).unwrap();
    assert_eq!(v.client_id, CLIENT_ID);
    assert_eq!(v.redirect_uri, REDIRECT);
    assert_eq!(v.scopes, vec!["ruvector:read"]);
    assert_eq!(v.state.as_deref(), Some("xyz"));
    assert_eq!(v.code_challenge, challenge());
    assert_eq!(v.resource, resource());
}

#[test]
fn omitted_scope_is_client_ceiling_and_state_optional() {
    let mut r = request();
    r.scope = None;
    r.state = None;
    let v = run(&r).unwrap();
    assert_eq!(
        v.scopes,
        vec!["ruvector:read", "ruvector:write", "offline_access"]
    );
    assert_eq!(v.state, None);
}

#[test]
fn loopback_redirect_any_port() {
    let mut r = request();
    r.redirect_uri = Some("http://127.0.0.1:40111/callback".into());
    assert_eq!(
        run(&r).unwrap().redirect_uri,
        "http://127.0.0.1:40111/callback"
    );
}

#[test]
fn untrusted_client_or_redirect_is_never_redirected() {
    let cases: Vec<super::Case<AuthorizationRequest>> = vec![
        (|r| r.client_id = None, C::InvalidClient),
        (|r| r.client_id = Some("edc-other".into()), C::InvalidClient),
        (|r| r.redirect_uri = None, C::InvalidRequest),
        (
            |r| r.redirect_uri = Some("https://evil.example/cb".into()),
            C::InvalidRequest,
        ),
        (
            |r| r.redirect_uri = Some(format!("{REDIRECT}/x")),
            C::InvalidRequest,
        ),
        (
            |r| r.redirect_uri = Some("http://127.0.0.1:1/steal".into()),
            C::InvalidRequest,
        ),
        (
            |r| r.state = Some("s".repeat(MAX_STATE_LEN + 1)),
            C::InvalidRequest,
        ),
        (|r| r.state = Some("a\nb".into()), C::InvalidRequest),
    ];
    for (mutate, code) in cases {
        let mut r = request();
        mutate(&mut r);
        // Even with other errors present, the untrusted check wins.
        r.response_type = Some("token".into());
        user_agent(run(&r), code);
    }
}

#[test]
fn redirected_errors() {
    let cases: Vec<super::Case<AuthorizationRequest>> = vec![
        (|r| r.response_type = None, C::UnsupportedResponseType),
        (
            |r| r.response_type = Some("token".into()),
            C::UnsupportedResponseType,
        ),
        (
            |r| r.response_type = Some("code id_token".into()),
            C::UnsupportedResponseType,
        ),
        (|r| r.code_challenge = None, C::InvalidRequest),
        (|r| r.code_challenge_method = None, C::InvalidRequest),
        (
            |r| r.code_challenge_method = Some("plain".into()),
            C::InvalidRequest,
        ),
        (
            |r| r.code_challenge = Some("short".into()),
            C::InvalidRequest,
        ),
        (|r| r.resource = None, C::InvalidTarget),
        (
            |r| r.resource = Some("https://evil.example/v1/mcp".into()),
            C::InvalidTarget,
        ),
        (
            |r| r.resource = Some(format!("{RESOURCE}/")),
            C::InvalidTarget,
        ),
        (
            |r| r.scope = Some("ruvector:read brains:read".into()),
            C::InvalidScope,
        ),
        (
            |r| r.scope = Some("ruvector:read  ruvector:write".into()),
            C::InvalidScope,
        ),
    ];
    for (mutate, code) in cases {
        let mut r = request();
        mutate(&mut r);
        let e = redirected(run(&r), code);
        let url = e.redirect_url(ISSUER).unwrap();
        let parsed = url::Url::parse(&url).unwrap();
        let q: Vec<(String, String)> = parsed.query_pairs().into_owned().collect();
        assert!(url.starts_with(REDIRECT), "{url}");
        assert!(q.contains(&("state".into(), "xyz".into())));
        assert!(q.contains(&("iss".into(), ISSUER.into())));
        assert!(q.iter().any(|(k, _)| k == "error"));
    }
}

#[test]
fn error_redirect_encodes_code_and_skips_absent_state() {
    let mut r = request();
    r.state = None;
    r.resource = None;
    let e = redirected(run(&r), C::InvalidTarget);
    let url = e.redirect_url(ISSUER).unwrap();
    assert!(url.contains("error=invalid_target"), "{url}");
    assert!(!url.contains("state="));
    assert_eq!(
        AuthorizeError::UserAgent(crate::OAuthError::new(C::InvalidClient, "x"))
            .redirect_url(ISSUER),
        None
    );
}

#[test]
fn success_redirect_carries_code_state_iss_and_keeps_query() {
    let mut v = validated();
    v.state = Some("a b&c=d".into());
    let url = success_redirect(&v, "CODE123", ISSUER);
    let parsed = url::Url::parse(&url).unwrap();
    let q: Vec<(String, String)> = parsed.query_pairs().into_owned().collect();
    assert_eq!(
        q,
        vec![
            ("code".into(), "CODE123".into()),
            ("state".into(), "a b&c=d".into()),
            ("iss".into(), ISSUER.into()),
        ]
    );
    v.redirect_uri = "https://app.example/cb?keep=1".into();
    v.state = None;
    let url = success_redirect(&v, "C", ISSUER);
    assert!(
        url.starts_with("https://app.example/cb?keep=1&code=C&iss="),
        "{url}"
    );
}

#[test]
fn from_pairs_rejects_repeats_at_user_agent() {
    let dup = pairs(&[("client_id", CLIENT_ID), ("client_id", "edc-evil")]);
    assert!(matches!(
        AuthorizationRequest::from_pairs(&dup),
        Err(AuthorizeError::UserAgent(_))
    ));
    let ok = pairs(&[
        ("response_type", "code"),
        ("client_id", CLIENT_ID),
        ("redirect_uri", REDIRECT),
        ("state", "xyz"),
        ("scope", "ruvector:read"),
        ("code_challenge", &challenge()),
        ("code_challenge_method", "S256"),
        ("resource", RESOURCE),
        ("unknown", "ignored"),
    ]);
    assert_eq!(AuthorizationRequest::from_pairs(&ok).unwrap(), request());
}
