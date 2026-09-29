//! RFC 7523 `private_key_jwt` authentication of exchange clients: every
//! rejection is `invalid_client`, replay is detected, and a forged
//! assertion never burns a client's `jti`.

use super::assert_code;
use super::exchange::{adapter_key, sign, thumb, Ex, ADAPTER, TOKEN_URL};
use crate::confidential::JWT_BEARER_ASSERTION;
use crate::error::OAuthErrorCode as C;
use crate::ports::MockAssertionReplayStore;
use crate::testing::*;
use crate::StoreError;
use p256::ecdsa::SigningKey;
use serde_json::{json, Value};

type Form = Vec<(String, String)>;

fn header(k: &SigningKey) -> Value {
    json!({"alg": "ES256", "typ": "JWT", "kid": thumb(k)})
}

#[test]
fn client_authentication_rejections() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let k = adapter_key();
    let other = SigningKey::from_bytes(&[4u8; 32].into()).unwrap();
    let now = w.now();
    let with = |f: &dyn Fn(&mut Value)| {
        let mut c = w.assertion_claims();
        f(&mut c);
        sign(&header(&k), &c, &k)
    };
    let cases: Vec<(String, &str)> = vec![
        ("garbage".into(), "malformed client assertion"),
        (
            sign(
                &json!({"alg": "ES256", "typ": "at+jwt", "kid": thumb(&k)}),
                &w.assertion_claims(),
                &k,
            ),
            "client assertion typ not accepted",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("iss");
            }),
            "malformed client assertion",
        ),
        (
            with(&|c| c["iss"] = json!("someone-else")),
            "unknown confidential client",
        ),
        (
            sign(&header(&other), &w.assertion_claims(), &other),
            "client assertion kid not registered",
        ),
        (
            sign(&header(&k), &w.assertion_claims(), &other),
            "bad client assertion signature",
        ),
        (
            with(&|c| c["sub"] = json!("x")),
            "client assertion sub must equal iss",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("sub");
            }),
            "client assertion sub must equal iss",
        ),
        (
            with(&|c| c["aud"] = json!(ISSUER)),
            "client assertion aud must be the token endpoint",
        ),
        (
            with(&|c| c["aud"] = json!([TOKEN_URL, "https://other.example/token"])),
            "client assertion aud must be the token endpoint",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("aud");
            }),
            "client assertion aud must be the token endpoint",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("exp");
            }),
            "client assertion exp required",
        ),
        (with(&|c| c["exp"] = json!(now)), "client assertion expired"),
        (
            with(&|c| c["exp"] = json!(now + 301)),
            "client assertion lifetime too long",
        ),
        (
            with(&|c| c["iat"] = json!(now + 61)),
            "client assertion iat invalid",
        ),
        (
            with(&|c| {
                c["iat"] = json!(now - 200);
                c["exp"] = json!(now + 200);
            }),
            "client assertion iat invalid",
        ),
        (
            with(&|c| c["nbf"] = json!(now + 61)),
            "client assertion not yet valid",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("jti");
            }),
            "client assertion jti required",
        ),
        (
            with(&|c| c["jti"] = json!("")),
            "client assertion jti required",
        ),
        (
            with(&|c| c["jti"] = json!("a b")),
            "client assertion jti required",
        ),
        (
            with(&|c| c["jti"] = json!("j".repeat(129))),
            "client assertion jti required",
        ),
    ];
    for (assertion, desc) in cases {
        let r = w.run(&w.form(&assertion, &s));
        assert_code(r.clone(), C::InvalidClient);
        assert_eq!(r.unwrap_err().error_description, desc, "{desc}");
    }
    // Accepted variants: one-element aud array, no typ, nbf/iat in range.
    let ok = [
        with(&|c| c["aud"] = json!([TOKEN_URL])),
        sign(
            &json!({"alg": "ES256", "kid": thumb(&k)}),
            &w.assertion_claims(),
            &k,
        ),
        with(&|c| {
            c["nbf"] = json!(now + 30);
            c["exp"] = json!(now + 300);
        }),
    ];
    for a in ok {
        w.run(&w.form(&a, &s)).unwrap();
    }
}

#[test]
fn form_level_client_rejections() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let a = w.assertion();
    let edit = |f: &dyn Fn(&mut Form)| {
        let mut form = w.form(&a, &s);
        f(&mut form);
        w.run(&form)
    };
    assert_code(
        edit(&|f| f.push(("client_secret".into(), "s".into()))),
        C::InvalidClient,
    );
    assert_code(
        edit(&|f| f.retain(|(k, _)| k != "client_assertion_type")),
        C::InvalidClient,
    );
    assert_code(
        edit(&|f| {
            f.retain(|(k, _)| k != "client_assertion_type");
            f.push((
                "client_assertion_type".into(),
                "urn:ietf:params:oauth:client-assertion-type:saml2-bearer".into(),
            ));
        }),
        C::InvalidClient,
    );
    assert_code(
        edit(&|f| f.retain(|(k, _)| k != "client_assertion")),
        C::InvalidClient,
    );
    let r = edit(&|f| f.push(("client_id".into(), "another".into())));
    assert_eq!(
        r.unwrap_err().error_description,
        "client_id does not match the assertion"
    );
    // A matching client_id is fine.
    let form = {
        let mut f = w.form(&w.assertion(), &s);
        f.push(("client_id".into(), ADAPTER.into()));
        f
    };
    w.run(&form).unwrap();
}

#[test]
fn assertion_replay_is_refused_until_it_expires() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let a = w.assertion();
    w.run(&w.form(&a, &s)).unwrap();
    let r = w.run(&w.form(&a, &s));
    assert_code(r.clone(), C::InvalidClient);
    assert_eq!(
        r.unwrap_err().error_description,
        "client assertion replayed"
    );
    // A replay fails even if the rest of the request would now fail too.
    let r = w.run(&w.form(&a, "junk"));
    assert_eq!(
        r.unwrap_err().error_description,
        "client assertion replayed"
    );
}

#[test]
fn consumed_assertion_stays_consumed_when_the_exchange_fails() {
    let w = Ex::new();
    let a = w.assertion();
    assert_code(w.run(&w.form(&a, "junk")), C::InvalidRequest);
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    assert_code(w.run(&w.form(&a, &s)), C::InvalidClient);
}

#[test]
fn forged_assertion_does_not_burn_the_jti() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let k = adapter_key();
    let claims = w.assertion_claims();
    let other = SigningKey::from_bytes(&[4u8; 32].into()).unwrap();
    // Correct kid, attacker's key: rejected before the replay cache.
    let forged = sign(&header(&k), &claims, &other);
    assert_code(w.run(&w.form(&forged, &s)), C::InvalidClient);
    assert!(w.store.assertions.borrow().is_empty());
    w.run(&w.form(&sign(&header(&k), &claims, &k), &s)).unwrap();
    assert_eq!(w.store.assertions.borrow().len(), 1);
}

#[test]
fn replay_store_failure_is_server_error() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let mut replay = MockAssertionReplayStore::new();
    replay
        .expect_record_assertion()
        .times(1)
        .returning(|_, _, _| Err(StoreError("down".into())));
    let mut ep = w.endpoint();
    ep.assertions = &replay;
    let f = w.form(&w.assertion(), &s);
    let r = crate::token::TokenRequest::from_form(&f).and_then(|r| ep.handle(&r));
    assert_code(r, C::ServerError);
}

#[test]
fn replay_key_is_per_client_and_expires_with_the_assertion() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let a = w.assertion();
    w.run(&w.form(&a, &s)).unwrap();
    let (key, exp) = {
        let m = w.store.assertions.borrow();
        let (k, e) = m.iter().next().unwrap();
        (*k, *e)
    };
    assert_eq!(exp, w.now() + 120);
    let want = format!("{ADAPTER}|jti-1");
    assert_eq!(key, crate::secret_hash(&want));
    assert_eq!(
        JWT_BEARER_ASSERTION,
        "urn:ietf:params:oauth:client-assertion-type:jwt-bearer"
    );
}
