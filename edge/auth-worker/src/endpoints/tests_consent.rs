//! Consent POST (ADR-351 §5.6): Continue is a same-origin form bound to the
//! `__Host-` cookie by the consent token; the page marks unverified hosts
//! and look-alike names.

use super::*;
use crate::sql::memory::MemoryDb;
use ruvector_edge_authz::authorize::{validate_authorization, AuthorizationRequest};
use ruvector_edge_authz::federation::begin_upstream;
use ruvector_edge_authz::ClientStore;

fn page_for(w: &World, client_id: &str) -> (Reply, String) {
    let r = authorize(&w.ctx(), Some(&authorize_query(client_id, RESOURCE)));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let cookie = r.header("Set-Cookie").unwrap();
    let pair = cookie.split(';').next().unwrap().to_string();
    (r, pair)
}

fn refused(r: &Reply) {
    assert!(r.header("Location").is_none(), "must not redirect");
    assert_eq!(r.status, 400, "{}", String::from_utf8_lossy(&r.body));
}

/// Regression (GET-link consent): without the binding cookie, with another
/// flow's cookie or with a forged token the Continue POST is refused, and
/// the refusal does not consume the flow.
#[test]
fn consent_post_requires_cookie_and_token() {
    let w = World::new();
    let client_id = w.client_id();
    let (page, cookie) = page_for(&w, &client_id);
    refused(&submit_consent(&w, &page, None));
    let name = cookie.split('=').next().unwrap();
    refused(&submit_consent(&w, &page, Some(&format!("{name}=forged"))));
    let (_, other_cookie) = page_for(&w, &client_id);
    refused(&submit_consent(&w, &page, Some(&other_cookie)));
    let html = String::from_utf8(page.body.clone()).unwrap();
    let forged = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("flow", &form_field(&html, "flow"))
        .append_pair("consent", "AAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAAA")
        .finish();
    refused(&super::super::authorize::consent_submit(
        &w.ctx(),
        Some(&cookie),
        FORM,
        forged.as_bytes(),
    ));
    let ok = submit_consent(&w, &page, Some(&cookie));
    assert_eq!(ok.status, 303);
    assert!(ok
        .header("Location")
        .unwrap()
        .starts_with("https://auth.cognitum.one/oauth/authorize?"));
}

/// The consent POST rebuilds exactly the URL `begin_upstream` would send.
#[test]
fn rebuilt_upstream_url_matches_begin_upstream() {
    let w = World::new();
    let client_id = w.client_id();
    let client = w.store.get_client(&client_id).unwrap().unwrap();
    let q = authorize_query(&client_id, RESOURCE);
    let pairs: Vec<(String, String)> = url::form_urlencoded::parse(q.as_bytes())
        .into_owned()
        .collect();
    let req = AuthorizationRequest::from_pairs(&pairs).unwrap();
    let v = validate_authorization(&req, &client, &w.cfg.resources).unwrap();
    let start = begin_upstream(&w.store, &w.rng, &w.clock, &w.cfg.upstream, v).unwrap();
    let state = param(&start.authorization_url, "state").unwrap();
    let flow = SqlPorts::<MemoryDb>::peek_flow(&w.store, &state, T0)
        .unwrap()
        .unwrap();
    assert_eq!(
        super::super::authorize::upstream_authorization_url(&w.cfg.upstream, &flow).unwrap(),
        start.authorization_url
    );
}

/// Regression (weak anti-phishing signals): unverified redirect hosts and
/// non-ASCII names are called out; the form may only lead to us or upstream.
#[test]
fn consent_page_marks_unverified_hosts_and_lookalike_names() {
    let w = World::new();
    let (page, _) = page_for(&w, &w.client_id());
    let html = String::from_utf8(page.body.clone()).unwrap();
    assert!(!html.contains("Unverified application"));
    assert!(html.contains("method=\"post\""));
    let csp = page.header("Content-Security-Policy").unwrap();
    assert!(
        csp.contains("form-action 'self' https://auth.cognitum.one"),
        "{csp}"
    );
    let r = w.register(json!({
        "redirect_uris": ["https://evil.example/cb"],
        "client_name": "\u{0421}laude",
    }));
    let id = body_json(&r)["client_id"].as_str().unwrap().to_string();
    let q = authorize_query(&id, RESOURCE).replace(
        &url::form_urlencoded::byte_serialize(REDIRECT.as_bytes()).collect::<String>(),
        &url::form_urlencoded::byte_serialize(b"https://evil.example/cb").collect::<String>(),
    );
    let html = String::from_utf8(authorize(&w.ctx(), Some(&q)).body).unwrap();
    assert!(html.contains("Unverified application"), "{html}");
    assert!(html.contains("Unusual characters"));
}

/// Regression (offline_access refused under the shipped config): with the
/// shipped `wrangler.toml` vars (plus an upstream client and kid pin), a
/// connector can register and authorize `offline_access`, and a
/// registration without `scope` gets the read-only default.
#[test]
fn shipped_config_accepts_offline_access_end_to_end() {
    let mut v = crate::config::tests::shipped_vars();
    v.insert("UPSTREAM_CLIENT_ID", "dcr-edge-test".into());
    v.insert(
        "ACCEPTED_UPSTREAM_KIDS",
        crate::config::tests::upstream_test_kid(),
    );
    let cfg = load(&v).unwrap();
    let resource = cfg.resources.resources()[1].as_str().to_string();
    let w = World::with_cfg(cfg);
    let r = w.register(json!({
        "redirect_uris": [REDIRECT],
        "scope": "ruvector:read offline_access",
        "grant_types": ["authorization_code", "refresh_token"],
    }));
    assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(body_json(&r)["scope"], "ruvector:read offline_access");
    let id = body_json(&r)["client_id"].as_str().unwrap().to_string();
    let page = authorize(&w.ctx(), Some(&authorize_query(&id, &resource)));
    assert_eq!(page.status, 200, "{}", String::from_utf8_lossy(&page.body));
    assert!(String::from_utf8_lossy(&page.body).contains("offline_access"));
    let r = w.register(json!({"redirect_uris": [REDIRECT]}));
    assert_eq!(body_json(&r)["scope"], "ruvector:read offline_access");
}
