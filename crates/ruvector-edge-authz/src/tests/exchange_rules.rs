//! RFC 8693 exchange request rules: request-shape rejections, the exchange
//! scope rule (map ∩ requested ∩ client ceiling ∩ target scopes), and the
//! separation of confidential and public (DCR) clients.

use super::assert_code;
use super::exchange::{adapter_key, description, sign, thumb, Ex, ADAPTER};
use crate::confidential::JWT_BEARER_ASSERTION;
use crate::error::OAuthErrorCode as C;
use crate::exchange::ACCESS_TOKEN_TYPE;
use crate::testing::*;
use crate::token::TokenRequest;
use serde_json::json;

const ALL_TEAM: [&str; 4] = ["team:read", "team:write", "team:run", "offline_access"];

#[test]
fn request_shape_rejections() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let cases: [(&[(&str, &str)], C); 9] = [
        (
            &[("subject_token_type", "urn:ietf:params:oauth:token-type:jwt")],
            C::InvalidRequest,
        ),
        (
            &[(
                "requested_token_type",
                "urn:ietf:params:oauth:token-type:refresh_token",
            )],
            C::InvalidRequest,
        ),
        (&[("actor_token", "x")], C::InvalidRequest),
        (
            &[("actor_token_type", ACCESS_TOKEN_TYPE)],
            C::InvalidRequest,
        ),
        (&[("audience", OTHER_RESOURCE)], C::InvalidTarget),
        (&[("resource", TEAM_RESOURCE)], C::InvalidTarget),
        (&[("resource", RESOURCE)], C::InvalidTarget),
        (&[("resource", "https://evil.example/v1")], C::InvalidTarget),
        (
            &[
                ("requested_token_type", ACCESS_TOKEN_TYPE),
                ("scope", "nope:x"),
            ],
            C::InvalidScope,
        ),
    ];
    for (extra, code) in cases {
        assert_code(w.exchange(&s, extra), code);
    }
    // Missing members.
    for drop in ["subject_token", "subject_token_type"] {
        let mut f = w.form(&w.assertion(), &s);
        f.retain(|(k, _)| k != drop);
        assert_code(w.run(&f), C::InvalidRequest);
    }
    let mut f = w.form(&w.assertion(), &s);
    f.retain(|(k, _)| k != "resource");
    assert_code(w.run(&f), C::InvalidTarget);
    let mut f = w.form(&w.assertion(), &s);
    f.push(("resource".into(), OTHER_RESOURCE.into()));
    assert_code(w.run(&f), C::InvalidTarget);
    // Accepted: requested_token_type = access_token.
    assert!(w
        .exchange(&s, &[("requested_token_type", ACCESS_TOKEN_TYPE)])
        .is_ok());
}

#[test]
fn scope_rule_is_map_then_requested_ceiling_and_resource() {
    let w = Ex::new();
    let all = w.user_token(TEAM_RESOURCE, &ALL_TEAM);
    let run_only = w.user_token(TEAM_RESOURCE, &["team:run", "offline_access"]);
    assert_code(w.exchange(&run_only, &[]), C::InvalidScope);
    let r = w.exchange(&all, &[("scope", "ruvector:read")]).unwrap();
    assert_eq!(r.scope, "ruvector:read");
    // Asking for more than the map yields is dropped, never minted.
    let r = w
        .exchange(
            &all,
            &[(
                "scope",
                "ruvector:write ruvector:admin offline_access openid",
            )],
        )
        .unwrap();
    assert_eq!(r.scope, "ruvector:write");
    assert_code(
        w.exchange(&all, &[("scope", "ruvector:admin")]),
        C::InvalidScope,
    );
    assert_code(w.exchange(&all, &[("scope", "team:read")]), C::InvalidScope);
    assert_code(w.exchange(&all, &[("scope", "bogus")]), C::InvalidScope);
    assert_code(w.exchange(&all, &[("scope", "a  b")]), C::InvalidScope);

    // Client ceiling narrows.
    let w = Ex::with(ALLOWLIST_CONFIG, "ruvector:read");
    let all = w.user_token(TEAM_RESOURCE, &ALL_TEAM);
    assert_eq!(w.exchange(&all, &[]).unwrap().scope, "ruvector:read");
    let write_only = w.user_token(TEAM_RESOURCE, &["team:write"]);
    assert_code(w.exchange(&write_only, &[]), C::InvalidScope);

    // The target's current scopes narrow too.
    let narrowed = "\
        https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 ruvector:read, \
        https://team.ruv.io/mcp team:read team:write team:run offline_access";
    let w = Ex::with(narrowed, "ruvector:read ruvector:write");
    let all = w.user_token(TEAM_RESOURCE, &ALL_TEAM);
    assert_eq!(w.exchange(&all, &[]).unwrap().scope, "ruvector:read");
}

#[test]
fn confidential_client_cannot_use_public_grants_and_dcr_client_cannot_exchange() {
    let w = Ex::new();
    let code = TokenRequest::AuthorizationCode {
        code: "c".into(),
        redirect_uri: REDIRECT.into(),
        client_id: ADAPTER.into(),
        code_verifier: VERIFIER.into(),
        resource: None,
    };
    assert_code(w.endpoint().handle(&code), C::InvalidClient);
    let refresh = TokenRequest::RefreshToken {
        refresh_token: "r".into(),
        client_id: ADAPTER.into(),
        scope: None,
        resource: None,
    };
    assert_code(w.endpoint().handle(&refresh), C::InvalidClient);
    // A public client's assertion-bearing code request is still refused.
    let f = pairs(&[
        ("grant_type", "authorization_code"),
        ("client_id", ADAPTER),
        ("client_assertion_type", JWT_BEARER_ASSERTION),
        ("client_assertion", &w.assertion()),
        ("code", "c"),
        ("redirect_uri", REDIRECT),
        ("code_verifier", VERIFIER),
    ]);
    assert_code(TokenRequest::from_form(&f), C::InvalidClient);
    // A registered DCR client signing its own assertion is unknown here.
    w.store
        .clients
        .borrow_mut()
        .insert(CLIENT_ID.into(), client(&["authorization_code"]));
    let k = adapter_key();
    let mut c = w.assertion_claims();
    c["iss"] = json!(CLIENT_ID);
    c["sub"] = json!(CLIENT_ID);
    let a = sign(&json!({"alg": "ES256", "kid": thumb(&k)}), &c, &k);
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let r = w.run(&w.form(&a, &s));
    assert_code(r.clone(), C::InvalidClient);
    assert_eq!(description(r), "unknown confidential client");
}

#[test]
fn exchange_is_audited_and_gateway_tokens_are_never_subjects() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let r = w.exchange(&subject, &[]).unwrap();
    let c = super::exchange::claims_of(&r.access_token);
    let a = r.audit.clone().expect("exchange audit");
    assert_eq!(a.client_id, ADAPTER);
    assert_eq!(
        (a.sub.as_str(), a.jti.as_str()),
        (c["sub"].as_str().unwrap(), c["jti"].as_str().unwrap())
    );
    assert_eq!(a.family_id, c["family_id"].as_str().unwrap());
    assert_eq!(
        (a.scope.as_str(), a.exp),
        ("ruvector:read", c["exp"].as_u64().unwrap())
    );
    let line: serde_json::Value = serde_json::from_str(&a.log_line()).unwrap();
    assert_eq!(line["event"], "token_exchange");
    assert_eq!(line["exchange"]["client_id"], ADAPTER);
    // The audit never reaches the client.
    assert!(serde_json::to_value(&r).unwrap().get("audit").is_none());
    // A user's `/v1/mcp` (or `/v1`) token is never exchangeable, whatever
    // its `ruvector:*` scopes.
    for aud in [RESOURCE, OTHER_RESOURCE] {
        let s = w.user_token(aud, &["ruvector:read", "ruvector:write"]);
        let e = w.exchange(&s, &[]).unwrap_err();
        assert_eq!(e.error, C::InvalidRequest, "{aud}");
    }
}
