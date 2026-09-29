//! Error-channel tests: what goes to the user agent, what is redirected,
//! and what the glue itself refuses (content types, sizes, caps).

use super::*;
use crate::http::MAX_JSON_BODY;

fn assert_page(r: &Reply, error: &str) {
    assert!(r.header("Location").is_none(), "must not redirect");
    assert!(r.status == 400 || r.status >= 500, "status {}", r.status);
    assert_eq!(body_json(r)["error"], error);
}

fn assert_redirect_error(r: &Reply, error: &str) {
    assert_eq!(r.status, 302, "{}", String::from_utf8_lossy(&r.body));
    let loc = r.header("Location").unwrap();
    assert!(loc.starts_with(REDIRECT), "{loc}");
    assert_eq!(param(loc, "error").as_deref(), Some(error));
    assert_eq!(param(loc, "state").as_deref(), Some("client-state"));
    assert_eq!(param(loc, "iss").as_deref(), Some(ISSUER));
}

#[test]
fn register_gates_content_type_size_and_cap() {
    let w = World::new();
    let r = register(&w.ctx(), "ip-test", Some("text/plain"), b"{}");
    assert_eq!(body_json(&r)["error"], "invalid_client_metadata");
    let big = vec![b' '; MAX_JSON_BODY + 1];
    assert_eq!(
        register(&w.ctx(), "ip-test", Some("application/json"), &big).status,
        413
    );
    let r = register(&w.ctx(), "ip-test", Some("application/json"), b"not json");
    assert_eq!(body_json(&r)["error"], "invalid_client_metadata");
    let ok = w.register(json!({"redirect_uris": [REDIRECT]}));
    assert_eq!(ok.status, 201);
    assert!(body_json(&ok)["client_id"]
        .as_str()
        .unwrap()
        .starts_with("edc-"));
    assert_eq!(body_json(&ok)["token_endpoint_auth_method"], "none");
    let mut v = vars();
    v.insert("MAX_CLIENTS", "1".into());
    let capped = World::with_cfg(load(&v).unwrap());
    assert_eq!(
        capped.register(json!({"redirect_uris": [REDIRECT]})).status,
        201
    );
    let r = capped.register(json!({"redirect_uris": [REDIRECT]}));
    assert_eq!(r.status, 503);
    assert_eq!(body_json(&r)["error"], "temporarily_unavailable");
}

#[test]
fn authorize_never_redirects_to_unverified_targets() {
    let w = World::new();
    let client_id = w.client_id();
    let r = authorize(&w.ctx(), Some(&authorize_query("edc-nobody", RESOURCE)));
    assert_page(&r, "invalid_client");
    let q = authorize_query(&client_id, RESOURCE).replace("claude.ai", "evil.example");
    assert_page(&authorize(&w.ctx(), Some(&q)), "invalid_request");
    let q = format!(
        "{}&client_id={client_id}",
        authorize_query(&client_id, RESOURCE)
    );
    assert!(authorize(&w.ctx(), Some(&q)).header("Location").is_none());
    assert_page(&authorize(&w.ctx(), None), "invalid_request");
}

#[test]
fn authorize_redirects_errors_after_verification() {
    let w = World::new();
    let client_id = w.client_id();
    let q = authorize_query(&client_id, "https://elsewhere.example/v1");
    assert_redirect_error(&authorize(&w.ctx(), Some(&q)), "invalid_target");
    let q = authorize_query(&client_id, RESOURCE).replace("S256", "plain");
    assert_redirect_error(&authorize(&w.ctx(), Some(&q)), "invalid_request");
}

#[test]
fn authorize_is_temporarily_unavailable_without_upstream_client() {
    let mut v = vars();
    v.insert("UPSTREAM_CLIENT_ID", String::new());
    let w = World::with_cfg(load(&v).unwrap());
    let client_id = w.client_id();
    let r = authorize(&w.ctx(), Some(&authorize_query(&client_id, RESOURCE)));
    assert_redirect_error(&r, "temporarily_unavailable");
}

/// Regression (open redirect, RFC 9700 §4.11.2): an anonymously registered,
/// unverified https redirect never receives a pre-consent 302 — neither
/// `temporarily_unavailable` (empty `UPSTREAM_CLIENT_ID`, the shipped
/// config) nor any other pre-consent failure.
#[test]
fn pre_consent_errors_never_redirect_to_unverified_targets() {
    const EVIL: &str = "https://evil.example/phish";
    let mut v = vars();
    v.insert("UPSTREAM_CLIENT_ID", String::new());
    let w = World::with_cfg(load(&v).unwrap());
    let r = w.register(json!({"redirect_uris": [EVIL]}));
    assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
    let client_id = body_json(&r)["client_id"].as_str().unwrap().to_string();
    let q = authorize_query(&client_id, RESOURCE).replace(
        &url::form_urlencoded::byte_serialize(REDIRECT.as_bytes()).collect::<String>(),
        &url::form_urlencoded::byte_serialize(EVIL.as_bytes()).collect::<String>(),
    );
    assert!(q.contains("evil.example"), "{q}");
    let r = authorize(&w.ctx(), Some(&q));
    assert_page(&r, "temporarily_unavailable");
    assert!(!String::from_utf8_lossy(&r.body).contains("evil.example"));
}

#[test]
fn consent_page_is_escaped_framing_protected_and_sets_bound_cookie() {
    let w = World::new();
    let client_id = w.client_id();
    let r = authorize(&w.ctx(), Some(&authorize_query(&client_id, RESOURCE)));
    let html = String::from_utf8(r.body.clone()).unwrap();
    assert!(html.contains("Test &lt;b&gt;Client&lt;/b&gt;"));
    assert!(!html.contains("<b>Client"));
    assert!(html.contains("claude.ai"));
    assert!(html.contains("ruvector:read"));
    assert_eq!(r.header("X-Frame-Options"), Some("DENY"));
    assert!(r
        .header("Content-Security-Policy")
        .unwrap()
        .contains("frame-ancestors 'none'"));
    let cookie = r.header("Set-Cookie").unwrap();
    assert!(cookie.starts_with("__Host-eaf-"), "{cookie}");
    let (up, _) = start(&w, &client_id);
    assert!(up.starts_with("https://auth.cognitum.one/oauth/authorize?"));
    assert_eq!(param(&up, "code_challenge_method").as_deref(), Some("S256"));
    assert_eq!(param(&up, "client_id").as_deref(), Some("dcr-edge-test"));
    assert_eq!(
        param(&up, "redirect_uri").as_deref(),
        Some(format!("{ISSUER}/callback").as_str())
    );
}

fn run_callback(w: &World, q: &str, cookie: Option<&str>, up: FakeUpstream) -> Reply {
    block_on(callback(
        &w.ctx(),
        Some(q),
        cookie,
        &up,
        OneKey(*w.upstream_key.verifying_key()),
    ))
}

#[test]
fn callback_requires_the_browser_binding_cookie_and_is_one_time() {
    let w = World::new();
    let client_id = w.client_id();
    let (up, cookie) = start(&w, &client_id);
    let state = param(&up, "state").unwrap();
    let q = format!("state={state}&code=c");
    let at = w.upstream_token(w.good_upstream_claims());
    // Another browser (no cookie) is refused without consuming the flow...
    assert_page(
        &run_callback(&w, &q, None, FakeUpstream::tokens(&at)),
        "access_denied",
    );
    // ...so the binding browser still completes it, exactly once.
    let r = run_callback(&w, &q, Some(&cookie), FakeUpstream::tokens(&at));
    assert_eq!(r.status, 302);
    assert!(param(r.header("Location").unwrap(), "code").is_some());
    let r = run_callback(&w, &q, Some(&cookie), FakeUpstream::tokens(&at));
    assert_page(&r, "access_denied");
    // A wrong secret under the right name consumes the flow.
    let (up, cookie) = start(&w, &client_id);
    let q = format!("state={}&code=c", param(&up, "state").unwrap());
    let forged = format!("{}=forged", cookie.split('=').next().unwrap());
    assert_page(
        &run_callback(&w, &q, Some(&forged), FakeUpstream::tokens(&at)),
        "access_denied",
    );
    assert_page(
        &run_callback(&w, &q, Some(&cookie), FakeUpstream::tokens(&at)),
        "access_denied",
    );
    assert_page(
        &run_callback(&w, "code=c", None, FakeUpstream::tokens(&at)),
        "invalid_request",
    );
}

#[test]
fn callback_redirects_upstream_and_verification_failures() {
    let w = World::new();
    let client_id = w.client_id();
    let at = w.upstream_token(w.good_upstream_claims());
    let cases: Vec<(String, FakeUpstream, &str)> = vec![
        (
            "error=access_denied".into(),
            FakeUpstream::tokens(&at),
            "access_denied",
        ),
        (
            "code=c&iss=https%3A%2F%2Fevil.example".into(),
            FakeUpstream::tokens(&at),
            "access_denied",
        ),
        (
            "code=c".into(),
            FakeUpstream(HttpResponse {
                status: 400,
                body: b"{}".to_vec(),
            }),
            "server_error",
        ),
        (
            "code=c".into(),
            FakeUpstream::tokens("not.a.jwt"),
            "server_error",
        ),
    ];
    for (extra, fake, error) in cases {
        let (up, cookie) = start(&w, &client_id);
        let q = format!("state={}&{extra}", param(&up, "state").unwrap());
        let r = run_callback(&w, &q, Some(&cookie), fake);
        assert_redirect_error(&r, error);
        assert!(r
            .headers
            .iter()
            .any(|(k, v)| *k == "Set-Cookie" && v.contains("Max-Age=0")));
    }
}

#[test]
fn callback_rejects_upstream_tokens_for_other_clients_or_without_tenant() {
    let w = World::new();
    let client_id = w.client_id();
    let mut wrong_aud = w.good_upstream_claims();
    wrong_aud["aud"] = json!("dcr-someone-else");
    wrong_aud["client_id"] = json!("dcr-someone-else");
    let mut id_token_shape = w.good_upstream_claims();
    id_token_shape.as_object_mut().unwrap().remove("typ");
    let mut no_org = w.good_upstream_claims();
    no_org.as_object_mut().unwrap().remove("org_id");
    for claims in [wrong_aud, id_token_shape, no_org] {
        let (up, cookie) = start(&w, &client_id);
        let q = format!("state={}&code=c", param(&up, "state").unwrap());
        let at = w.upstream_token(claims);
        let r = run_callback(&w, &q, Some(&cookie), FakeUpstream::tokens(&at));
        assert_eq!(r.status, 302);
        let loc = r.header("Location").unwrap();
        assert!(param(loc, "code").is_none(), "{loc}");
        assert!(param(loc, "error").is_some());
    }
}

#[test]
fn token_endpoint_gates_media_type_and_grant() {
    let w = World::new();
    let r = token::token(&w.ctx(), Some("application/json"), b"{}");
    assert_eq!(r.status, 415);
    let r = token_form(&w, &[("grant_type", "password"), ("client_id", "x")]);
    assert_eq!(body_json(&r)["error"], "unsupported_grant_type");
    let client_id = w.client_id();
    let code = login(&w, &client_id);
    w.clock.advance(3_600);
    assert_eq!(
        body_json(&redeem(&w, &client_id, &code))["error"],
        "invalid_grant"
    );
}
