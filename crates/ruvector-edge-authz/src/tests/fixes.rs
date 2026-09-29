//! Regression tests for the 2026-09-28 authz review findings.

use super::assert_code;
use crate::authorize::*;
use crate::client::{is_display_name, validate_redirect_uri, validate_registration};
use crate::error::OAuthErrorCode as C;
use crate::federation::{UpstreamCallback, UpstreamStart};
use crate::params::{split_scope, strip_identity_scopes, Params};
use crate::testing::*;
use crate::token::{mint_access_token, MintRequest, TokenRequest, TokenResponse};
use ruvector_edge_auth::jws::b64url_decode;

const EVIL: &str = "https://evil.example/x";

fn evil_client() -> crate::client::ClientRecord {
    let mut c = client(&["authorization_code"]);
    c.redirect_uris = vec![validate_redirect_uri(EVIL).unwrap()];
    c
}

fn request_to(redirect: &str) -> AuthorizationRequest {
    AuthorizationRequest {
        response_type: Some("code".into()),
        client_id: Some(CLIENT_ID.into()),
        redirect_uri: Some(redirect.into()),
        scope: Some("ruvector:read".into()),
        state: Some("s".into()),
        code_challenge: Some(challenge()),
        code_challenge_method: Some("S256".into()),
        resource: Some(RESOURCE.into()),
    }
}

// --- Open redirector via anonymous DCR (RFC 9700 §4.11.2) ---------------

#[test]
fn unverified_redirect_errors_go_to_user_agent() {
    let c = evil_client();
    let mutations: [fn(&mut AuthorizationRequest); 5] = [
        |r| r.response_type = Some("bogus".into()),
        |r| r.code_challenge = None,
        |r| r.code_challenge_method = Some("plain".into()),
        |r| r.resource = Some("https://other.example/".into()),
        |r| r.scope = Some("ruvector:admin".into()),
    ];
    for m in mutations {
        let mut r = request_to(EVIL);
        m(&mut r);
        let e = validate_authorization(&r, &c, &allowlist()).unwrap_err();
        assert!(matches!(e, AuthorizeError::UserAgent(_)), "{e:?}");
        assert_eq!(e.redirect_url(ISSUER), None);
    }
    // Not over-broad: a valid request from the unverified client still
    // proceeds (to the consent page, which marks the host unverified).
    assert!(validate_authorization(&request_to(EVIL), &c, &allowlist()).is_ok());
}

#[test]
fn verified_connector_redirect_still_redirects_errors() {
    let mut r = request_to(REDIRECT);
    r.response_type = Some("bogus".into());
    let e = validate_authorization(&r, &client(&["authorization_code"]), &allowlist());
    assert!(matches!(e, Err(AuthorizeError::Redirect { .. })));
    let mut r = request_to(LOOPBACK);
    r.response_type = Some("bogus".into());
    let e = validate_authorization(&r, &client(&["authorization_code"]), &allowlist());
    assert!(matches!(e, Err(AuthorizeError::Redirect { .. })));
}

#[test]
fn verified_hosts_are_exact_host_and_path_prefix() {
    let v = |u: &str| validate_redirect_uri(u).unwrap().is_verified();
    assert!(v("https://claude.ai/api/mcp/auth_callback"));
    assert!(v("https://claude.com/api/mcp/auth_callback"));
    assert!(v("https://chatgpt.com/connector/oauth/abc"));
    assert!(v("https://chatgpt.com/aip/x/oauth/callback"));
    assert!(v("http://127.0.0.1:9/cb"));
    assert!(!v("https://claude.ai.evil.example/api/mcp/cb"));
    assert!(!v("https://evil.claude.ai/api/mcp/cb"));
    assert!(!v("https://claude.ai/other/cb"));
    assert!(!v("https://claude.ai:8443/api/mcp/cb"));
    assert!(!v("https://claude.ai/api/mcp/../../evil"));
    assert!(!v(EVIL));
}

#[test]
fn post_validation_errors_respect_verification() {
    let mut v = validated();
    let err = crate::OAuthError::new(C::TemporarilyUnavailable, "down");
    assert!(matches!(
        v.error(err.clone()),
        AuthorizeError::Redirect { .. }
    ));
    v.redirect_uri = EVIL.into();
    assert!(matches!(v.error(err.clone()), AuthorizeError::UserAgent(_)));
    // After the user clicked Cancel on the consent page: redirect is fine.
    assert!(matches!(
        v.error_after_consent(err),
        AuthorizeError::Redirect { .. }
    ));
}

// --- Identity scopes accepted and dropped (ADR-351 §5.3) -----------------

#[test]
fn identity_scopes_are_dropped_at_authorization() {
    let c = client(&["authorization_code"]);
    let mut r = request_to(REDIRECT);
    r.scope = Some("openid ruvector:read profile email".into());
    let v = validate_authorization(&r, &c, &allowlist()).unwrap();
    assert_eq!(v.scopes, vec!["ruvector:read"]);
    r.scope = Some("openid profile".into());
    let v = validate_authorization(&r, &c, &allowlist()).unwrap();
    assert_eq!(
        v.scopes,
        vec!["ruvector:read", "offline_access"],
        "nothing left -> resource default"
    );
    r.scope = Some("openid brains:read".into());
    let e = validate_authorization(&r, &c, &allowlist()).unwrap_err();
    assert_eq!(e.oauth().error, C::InvalidScope);
}

#[test]
fn identity_scopes_are_dropped_at_registration() {
    let mut req = reg_request();
    req.scope = Some("openid offline_access".into());
    let rec = validate_registration(&req, &policy(), "edc-x".into(), T0).unwrap();
    assert_eq!(rec.scope, vec!["offline_access"]);
    req.scope = Some("openid profile email".into());
    let rec = validate_registration(&req, &policy(), "edc-x".into(), T0).unwrap();
    assert_eq!(rec.scope, policy().default_scope);
    req.scope = Some("openid bogus".into());
    assert_code(
        validate_registration(&req, &policy(), "edc-x".into(), T0),
        C::InvalidClientMetadata,
    );
    let s = |v: &[&str]| v.iter().map(|x| x.to_string()).collect::<Vec<_>>();
    assert_eq!(
        strip_identity_scopes(s(&["email", "a", "openid"])),
        s(&["a"])
    );
}

// --- Edge subject + upstream_iss (ADR-351 §5.2) ---------------------------

fn mint_payload(identity: &crate::federation::UpstreamIdentity) -> serde_json::Value {
    let scopes = vec!["ruvector:read".to_string()];
    let (jwt, _) = mint_access_token(
        &TestSigner::default(),
        &SeqRng::default(),
        &FixedClock::at(T0),
        &MintRequest {
            issuer: ISSUER,
            resource: &resource(),
            client_id: CLIENT_ID,
            identity,
            family_id: "fam-1",
            scopes: &scopes,
            act: None,
        },
    )
    .unwrap();
    let raw = b64url_decode(jwt.split('.').nth(1).unwrap()).unwrap();
    assert!(
        !String::from_utf8_lossy(&raw).contains(&identity.sub),
        "raw upstream sub leaked into the token"
    );
    serde_json::from_slice(&raw).unwrap()
}

#[test]
fn minted_sub_is_edge_subject_with_upstream_iss() {
    let id = identity();
    let p = mint_payload(&id);
    let sub = p["sub"].as_str().unwrap();
    assert!(sub.starts_with("es1_"));
    assert_eq!(sub.len(), 30);
    assert_eq!(p["upstream_iss"], "https://auth.cognitum.one");
    let mut other = identity();
    other.upstream_iss = "https://other-idp.example".into();
    assert_ne!(mint_payload(&other)["sub"], p["sub"]);
}

#[test]
fn mint_refuses_identity_without_upstream_iss() {
    let mut id = identity();
    id.upstream_iss.clear();
    let scopes = vec!["ruvector:read".to_string()];
    let r = mint_access_token(
        &TestSigner::default(),
        &SeqRng::default(),
        &FixedClock::at(T0),
        &MintRequest {
            issuer: ISSUER,
            resource: &resource(),
            client_id: CLIENT_ID,
            identity: &id,
            family_id: "fam-1",
            scopes: &scopes,
            act: None,
        },
    );
    assert_code(r, C::ServerError);
}

// --- Low findings ---------------------------------------------------------

#[test]
fn client_name_rejects_format_and_separator_chars() {
    for bad in [
        "Chat\u{202E}GPT",
        "Claude\u{200B}",
        "a\u{2066}b",
        "a\u{2028}b",
        "a\u{FEFF}",
        " Claude",
        "Claude ",
        "a\u{00A0}b",
    ] {
        assert!(!is_display_name(bad), "{bad:?}");
        let mut req = reg_request();
        req.client_name = Some(bad.into());
        assert_code(
            validate_registration(&req, &policy(), "edc-x".into(), T0),
            C::InvalidClientMetadata,
        );
    }
    assert!(is_display_name("Claude (Anthropic) - v2.1"));
    // ASCII only: non-ASCII letters (and so homoglyphs) are refused.
    for bad in ["Café", "東京", "Cl\u{0430}ude"] {
        assert!(!is_display_name(bad), "{bad:?}");
    }
}

#[test]
fn param_names_are_bounded() {
    let huge = "k".repeat(1 << 20);
    assert_code(
        Params::from_pairs(&[(huge.clone(), String::new())]),
        C::InvalidRequest,
    );
    assert_code(Params::from_pairs(&[(huge, "v".into())]), C::InvalidRequest);
    assert_code(
        Params::from_pairs(&[("a\u{0}b".into(), "v".into())]),
        C::InvalidRequest,
    );
    assert!(Params::from_pairs(&[("k".repeat(64), "v".into())]).is_ok());
}

#[test]
fn split_scope_is_bounded_early() {
    let many: Vec<String> = (0..33).map(|i| format!("s{i}")).collect();
    assert_code(split_scope(&many.join(" ")), C::InvalidScope);
    assert!(split_scope(&many[..32].join(" ")).is_ok());
    let long = "a".repeat(crate::params::MAX_PARAM_LEN + 1);
    assert_code(split_scope(&long), C::InvalidScope);
    // Duplicates do not count toward the cap.
    assert_eq!(split_scope(&["x"; 100].join(" ")).unwrap(), vec!["x"]);
    let mut req = reg_request();
    req.scope = Some(["a"; 8000].join(" "));
    assert_code(
        validate_registration(&req, &policy(), "edc-x".into(), T0),
        C::InvalidClientMetadata,
    );
}

#[test]
fn debug_never_prints_secrets() {
    let req = TokenRequest::AuthorizationCode {
        code: "SECRET-CODE".into(),
        redirect_uri: REDIRECT.into(),
        client_id: CLIENT_ID.into(),
        code_verifier: "SECRET-VERIFIER".into(),
        resource: None,
    };
    let rt = TokenRequest::RefreshToken {
        refresh_token: "SECRET-RT".into(),
        client_id: CLIENT_ID.into(),
        scope: None,
        resource: None,
    };
    let resp = TokenResponse {
        access_token: "SECRET-AT".into(),
        token_type: "Bearer",
        expires_in: 900,
        refresh_token: Some("SECRET-RT2".into()),
        scope: "ruvector:read".into(),
    };
    let rev = crate::revoke::RevocationRequest {
        token: "SECRET-REV".into(),
        client_id: CLIENT_ID.into(),
        token_type_hint: None,
    };
    let start = UpstreamStart {
        authorization_url: "https://auth.cognitum.one/oauth/authorize".into(),
        browser_secret: "SECRET-COOKIE".into(),
    };
    let flow = crate::federation::UpstreamFlowState {
        state: "st".into(),
        nonce: "n".into(),
        upstream_code_verifier: "SECRET-UPV".into(),
        browser_binding: [9; 32],
        downstream: validated(),
        expires_at: T0,
    };
    let all = format!("{req:?}{rt:?}{resp:?}{rev:?}{start:?}{flow:?}");
    assert!(!all.contains("SECRET"), "{all}");
    assert!(all.contains("<redacted>"));
    assert!(all.contains(CLIENT_ID));
}

#[test]
fn upstream_trust_root_must_be_https_same_origin() {
    let cases: [fn(&mut crate::federation::UpstreamConfig); 6] = [
        |u| u.issuer = "http://auth.cognitum.one".into(),
        |u| u.issuer = String::new(),
        |u| u.issuer = "https://auth.cognitum.one/".into(),
        |u| u.jwks_url = "http://auth.cognitum.one/.well-known/jwks.json".into(),
        |u| u.jwks_url = String::new(),
        |u| u.jwks_url = "https://evil.example/.well-known/jwks.json".into(),
    ];
    for m in cases {
        let mut u = upstream();
        m(&mut u);
        assert_code(u.ensure_ready(), C::ServerError);
    }
    assert!(upstream().ensure_ready().is_ok());
}

#[test]
fn callback_iss_must_match_upstream() {
    let cb = |iss: Option<&str>| {
        let mut kv = vec![("state", "st"), ("code", "c")];
        if let Some(i) = iss {
            kv.push(("iss", i));
        }
        UpstreamCallback::from_pairs(&pairs(&kv)).unwrap()
    };
    assert!(cb(None).check_issuer(&upstream()).is_ok());
    let ok = cb(Some("https://auth.cognitum.one"));
    assert_eq!(ok.iss.as_deref(), Some("https://auth.cognitum.one"));
    assert!(ok.check_issuer(&upstream()).is_ok());
    assert_code(
        cb(Some("https://evil.example")).check_issuer(&upstream()),
        C::AccessDenied,
    );
}

#[test]
fn consent_token_binds_cookie_and_request() {
    let v = validated();
    let t = consent_token("cookie-1", &v);
    assert!(verify_consent_token("cookie-1", &v, &t));
    assert!(!verify_consent_token("cookie-2", &v, &t));
    assert!(!verify_consent_token("", &v, &consent_token("", &v)));
    let mut other = validated();
    other.scopes = vec!["ruvector:write".into()];
    assert!(!verify_consent_token("cookie-1", &other, &t));
    assert!(!verify_consent_token("cookie-1", &v, ""));
}
