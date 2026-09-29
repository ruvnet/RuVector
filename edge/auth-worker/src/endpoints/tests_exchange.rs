//! RFC 8693 exchange through the Worker `/token` handler over real SQLite:
//! an operator-registered adapter (`CONFIDENTIAL_CLIENTS`) turns a user's
//! team.ruv.io token into a `…/v1` token; replay and nested `act` refused.

use super::{body_json, World, FORM, T0};
use crate::config::tests::{load, vars};
use crate::endpoints::token;
use crate::http::Reply;
use crate::signer::tests::test_key;
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::SigningKey;
use ruvector_edge_auth::jws::{b64url_decode, b64url_encode};
use ruvector_edge_auth::{Jwk, ResourceUrl};
use ruvector_edge_authz::federation::UpstreamIdentity;
use ruvector_edge_authz::token::{mint_access_token, MintRequest};
use serde_json::{json, Value};

const TEAM: &str = "https://team.ruv.io/mcp";
const V1: &str = "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1";
const ADAPTER: &str = "team-ruv-io";

fn adapter() -> SigningKey {
    test_key(11)
}

fn world() -> World {
    let mut v = vars();
    let allow = format!(
        "{}, {TEAM} team:read team:write team:run offline_access",
        v["RESOURCE_ALLOWLIST"]
    );
    v.insert("RESOURCE_ALLOWLIST", allow);
    let j = Jwk::from_verifying_key(adapter().verifying_key());
    let reg = json!([{
        "client_id": ADAPTER,
        "jwk": {"kty": "EC", "crv": "P-256", "x": j.x, "y": j.y, "kid": j.kid},
        "subject_audiences": [TEAM],
        "scope": "ruvector:read ruvector:write",
    }]);
    v.insert("CONFIDENTIAL_CLIENTS", reg.to_string());
    World::with_cfg(load(&v).unwrap())
}

fn sign(header: &Value, claims: &Value, k: &SigningKey) -> String {
    let input = format!(
        "{}.{}",
        b64url_encode(header.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    let sig: p256::ecdsa::Signature = k.sign(input.as_bytes());
    format!("{input}.{}", b64url_encode(&sig.to_bytes()))
}

fn claims_of(token: &str) -> Value {
    serde_json::from_slice(&b64url_decode(token.split('.').nth(1).unwrap()).unwrap()).unwrap()
}

fn user_token(w: &World, scopes: &[&str]) -> String {
    let scopes: Vec<String> = scopes.iter().map(|s| s.to_string()).collect();
    let identity = UpstreamIdentity {
        upstream_iss: "https://auth.cognitum.one".into(),
        sub: "user-1".into(),
        org_id: "org_1".into(),
        workspace_id: "ws_1".into(),
    };
    let req = MintRequest {
        issuer: &w.cfg.issuer,
        resource: &ResourceUrl::parse(TEAM).unwrap(),
        client_id: "edc-user-client",
        identity: &identity,
        family_id: "fam-1",
        scopes: &scopes,
        act: None,
    };
    mint_access_token(&w.signer, &w.rng, &w.clock, &req)
        .unwrap()
        .0
}

fn assertion(w: &World, jti: &str) -> String {
    let k = adapter();
    let kid = Jwk::from_verifying_key(k.verifying_key()).kid;
    let claims = json!({
        "iss": ADAPTER, "sub": ADAPTER, "aud": format!("{}/token", w.cfg.issuer),
        "iat": T0, "exp": T0 + 60, "jti": jti,
    });
    sign(
        &json!({"alg": "ES256", "typ": "JWT", "kid": kid}),
        &claims,
        &k,
    )
}

fn exchange(w: &World, assertion: &str, subject: &str) -> Reply {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair(
            "grant_type",
            "urn:ietf:params:oauth:grant-type:token-exchange",
        )
        .append_pair(
            "client_assertion_type",
            "urn:ietf:params:oauth:client-assertion-type:jwt-bearer",
        )
        .append_pair("client_assertion", assertion)
        .append_pair("subject_token", subject)
        .append_pair(
            "subject_token_type",
            "urn:ietf:params:oauth:token-type:access_token",
        )
        .append_pair("resource", V1)
        .finish();
    token::token(&w.ctx(), FORM, body.as_bytes())
}

#[test]
fn adapter_exchanges_a_team_token_for_a_v1_token() {
    let w = world();
    let subject = user_token(
        &w,
        &["team:read", "team:write", "team:run", "offline_access"],
    );
    let r = exchange(&w, &assertion(&w, "a1"), &subject);
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.header("Cache-Control"), Some("no-store"));
    let b = body_json(&r);
    assert_eq!(
        b["issued_token_type"],
        "urn:ietf:params:oauth:token-type:access_token"
    );
    assert_eq!(b["scope"], "ruvector:read ruvector:write");
    assert!(b.get("refresh_token").is_none());
    let (s, c) = (
        claims_of(&subject),
        claims_of(b["access_token"].as_str().unwrap()),
    );
    for k in ["sub", "upstream_iss", "org_id", "workspace_id", "family_id"] {
        assert_eq!(c[k], s[k], "{k}");
    }
    assert_eq!(c["aud"], V1);
    assert_eq!(c["act"], json!({"sub": ADAPTER}));
    assert_eq!(c["client_id"], ADAPTER);
    // The mint is audited (logged by the Worker, never in the body).
    let line: Value = serde_json::from_str(r.log.as_deref().expect("audit line")).unwrap();
    assert_eq!(line["event"], "token_exchange");
    for (k, v) in [("client_id", ADAPTER), ("sub", c["sub"].as_str().unwrap())] {
        assert_eq!(line["exchange"][k], v, "{k}");
    }
    assert_eq!(line["exchange"]["jti"], c["jti"]);
    assert_eq!(line["exchange"]["family_id"], c["family_id"]);
    assert!(b.get("audit").is_none());

    // Same assertion again: replay (401 invalid_client).
    let r = exchange(&w, &assertion(&w, "a1"), &subject);
    assert_eq!(r.status, 401);
    assert!(r.log.is_none());
    assert_eq!(body_json(&r)["error"], "invalid_client");

    // The exchanged token cannot be exchanged again (nested act).
    let exchanged = b["access_token"].as_str().unwrap().to_string();
    let r = exchange(&w, &assertion(&w, "a2"), &exchanged);
    assert_eq!(r.status, 400);
    assert_eq!(body_json(&r)["error"], "invalid_request");
}

#[test]
fn exchange_is_refused_without_registered_clients() {
    // Shipped config shape: no CONFIDENTIAL_CLIENTS.
    let w = World::new();
    let r = exchange(&w, &assertion(&w, "a1"), "x.y.z");
    assert_eq!(r.status, 401);
    assert_eq!(body_json(&r)["error"], "invalid_client");
}
