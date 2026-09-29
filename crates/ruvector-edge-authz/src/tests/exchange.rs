//! RFC 8693 token exchange (ADR-351 §5.6, §16.1): success shape, the
//! subject-token and request rejection paths, nested-`act` prevention and
//! the exchange scope rule. Client authentication (RFC 7523) rejections
//! live in `exchange_client.rs`; registry loading in `confidential.rs`.

use super::assert_code;
use crate::confidential::{ConfidentialClients, JWT_BEARER_ASSERTION};
use crate::error::OAuthErrorCode as C;
use crate::exchange::{ACCESS_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT};
use crate::grant::TokenEndpoint;
use crate::ports::MockSigner;
use crate::testing::*;
use crate::token::{mint_access_token, MintRequest, TokenRequest, TokenResponse};
use crate::{OAuthError, ResourceAllowlist, ResourceUrl};
use p256::ecdsa::{signature::Signer as _, Signature, SigningKey};
use ruvector_edge_auth::jws::{b64url_decode, b64url_encode};
use ruvector_edge_auth::Jwk;
use serde_json::{json, Value};
use std::cell::Cell;

pub(super) const ADAPTER: &str = "team-ruv-io";
pub(super) const TOKEN_URL: &str =
    "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev/token";
const ALL_TEAM: [&str; 4] = ["team:read", "team:write", "team:run", "offline_access"];

pub(super) fn adapter_key() -> SigningKey {
    SigningKey::from_bytes(&[3u8; 32].into()).unwrap()
}

pub(super) fn thumb(k: &SigningKey) -> String {
    Jwk::from_verifying_key(k.verifying_key()).kid
}

/// Compact ES256 JWS over arbitrary JSON (forgeries included).
pub(super) fn sign(header: &Value, claims: &Value, k: &SigningKey) -> String {
    let input = format!(
        "{}.{}",
        b64url_encode(header.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    let sig: Signature = k.sign(input.as_bytes());
    format!("{input}.{}", b64url_encode(&sig.to_bytes()))
}

/// One registry entry for `key` as JSON.
pub(super) fn entry(id: &str, key: &SigningKey, auds: &[&str], scope: &str) -> Value {
    let j = Jwk::from_verifying_key(key.verifying_key());
    json!({
        "client_id": id,
        "jwk": {"kty": "EC", "crv": "P-256", "x": j.x, "y": j.y, "kid": j.kid},
        "subject_audiences": auds,
        "scope": scope,
    })
}

pub(super) struct Ex {
    pub store: MemStore,
    pub rng: SeqRng,
    pub clock: FixedClock,
    pub signer: ThumbSigner,
    pub resources: ResourceAllowlist,
    pub confidential: ConfidentialClients,
    jti: Cell<u64>,
}

impl Ex {
    pub fn new() -> Self {
        Self::with(ALLOWLIST_CONFIG, "ruvector:read ruvector:write")
    }

    /// World with allowlist `allow` and the adapter's `ceiling`.
    pub fn with(allow: &str, ceiling: &str) -> Self {
        let resources = ResourceAllowlist::from_config(allow).unwrap();
        let reg = json!([entry(ADAPTER, &adapter_key(), &[TEAM_RESOURCE], ceiling)]);
        Ex {
            confidential: ConfidentialClients::from_config(&reg.to_string(), &resources).unwrap(),
            resources,
            store: MemStore::default(),
            rng: SeqRng::default(),
            clock: FixedClock::at(T0),
            signer: ThumbSigner::default(),
            jti: Cell::new(0),
        }
    }

    pub fn now(&self) -> u64 {
        self.clock.0.get()
    }

    pub fn endpoint(&self) -> TokenEndpoint<'_> {
        TokenEndpoint {
            issuer: ISSUER,
            resources: &self.resources,
            clients: &self.store,
            codes: &self.store,
            refresh: &self.store,
            signer: &self.signer,
            rng: &self.rng,
            clock: &self.clock,
            confidential: &self.confidential,
            assertions: &self.store,
        }
    }

    /// A genuine edge token for `resource` with `scopes`, family `fam-1`.
    pub fn user_token(&self, resource: &str, scopes: &[&str]) -> String {
        let scopes: Vec<String> = scopes.iter().map(|s| s.to_string()).collect();
        let (t, _) = mint_access_token(
            &self.signer,
            &self.rng,
            &self.clock,
            &MintRequest {
                issuer: ISSUER,
                resource: &ResourceUrl::parse(resource).unwrap(),
                client_id: CLIENT_ID,
                identity: &identity(),
                family_id: "fam-1",
                scopes: &scopes,
                act: None,
            },
        )
        .unwrap();
        t
    }

    /// Valid assertion claims with a fresh `jti`.
    pub fn assertion_claims(&self) -> Value {
        self.jti.set(self.jti.get() + 1);
        json!({
            "iss": ADAPTER, "sub": ADAPTER, "aud": TOKEN_URL,
            "iat": self.now(), "exp": self.now() + 120,
            "jti": format!("jti-{}", self.jti.get()),
        })
    }

    pub fn assertion(&self) -> String {
        let k = adapter_key();
        sign(
            &json!({"alg": "ES256", "typ": "JWT", "kid": thumb(&k)}),
            &self.assertion_claims(),
            &k,
        )
    }

    /// Exchange form for `subject` targeting `…/v1`.
    pub fn form(&self, assertion: &str, subject: &str) -> Vec<(String, String)> {
        pairs(&[
            ("grant_type", TOKEN_EXCHANGE_GRANT),
            ("client_assertion_type", JWT_BEARER_ASSERTION),
            ("client_assertion", assertion),
            ("subject_token", subject),
            ("subject_token_type", ACCESS_TOKEN_TYPE),
            ("resource", OTHER_RESOURCE),
        ])
    }

    pub fn run(&self, form: &[(String, String)]) -> Result<TokenResponse, OAuthError> {
        TokenRequest::from_form(form).and_then(|r| self.endpoint().handle(&r))
    }

    /// Exchange `subject` with a fresh assertion plus `extra` pairs
    /// (replacing same-named members).
    pub fn exchange(
        &self,
        subject: &str,
        extra: &[(&str, &str)],
    ) -> Result<TokenResponse, OAuthError> {
        let mut f = self.form(&self.assertion(), subject);
        for (k, v) in extra {
            f.retain(|(n, _)| n != k);
            f.push((k.to_string(), v.to_string()));
        }
        self.run(&f)
    }
}

pub(super) fn claims_of(token: &str) -> Value {
    let seg = token.split('.').nth(1).unwrap();
    serde_json::from_slice(&b64url_decode(seg).unwrap()).unwrap()
}

pub(super) fn description(r: Result<TokenResponse, OAuthError>) -> &'static str {
    r.unwrap_err().error_description
}

#[test]
fn exchange_keeps_identity_adds_act_and_maps_scopes() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &ALL_TEAM);
    let s = claims_of(&subject);
    let r = w.exchange(&subject, &[]).unwrap();
    assert_eq!(r.token_type, "Bearer");
    assert!(r.refresh_token.is_none());
    assert_eq!(r.issued_token_type, Some(ACCESS_TOKEN_TYPE));
    assert_eq!(r.scope, "ruvector:read ruvector:write");
    assert_eq!(r.expires_in, 900);
    let c = claims_of(&r.access_token);
    for k in [
        "sub",
        "upstream_iss",
        "org_id",
        "workspace_id",
        "family_id",
        "iss",
    ] {
        assert_eq!(c[k], s[k], "{k}");
    }
    assert_eq!(c["aud"], OTHER_RESOURCE);
    assert_eq!(c["client_id"], ADAPTER);
    assert_eq!(c["act"], json!({"sub": ADAPTER}));
    assert_eq!(c["scope"], "ruvector:read ruvector:write");
    assert!(c["exp"].as_u64().unwrap() <= s["exp"].as_u64().unwrap());
    assert_ne!(c["jti"], s["jti"]);
    let v = serde_json::to_value(&r).unwrap();
    assert!(v.get("refresh_token").is_none());
    assert_eq!(v["issued_token_type"], ACCESS_TOKEN_TYPE);
    // The exchanged token is a well-formed edge token under the AS key.
    let jws = ruvector_edge_auth::parse_compact(&r.access_token).unwrap();
    assert_eq!(jws.header.typ.as_deref(), Some("at+jwt"));
    assert_eq!(jws.header.kid, thumb(&w.signer.0 .0));
    ruvector_edge_auth::verify_es256(&jws, &w.signer.0.verifying_key()).unwrap();
}

#[test]
fn exchanged_token_never_outlives_its_subject() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let subject_exp = claims_of(&subject)["exp"].as_u64().unwrap();
    w.clock.advance(600);
    let r = w.exchange(&subject, &[]).unwrap();
    assert_eq!(r.expires_in, 300);
    assert_eq!(
        claims_of(&r.access_token)["exp"].as_u64().unwrap(),
        subject_exp
    );
    w.clock.advance(300);
    assert_eq!(
        description(w.exchange(&subject, &[])),
        "subject_token expired"
    );
}

#[test]
fn nested_act_is_refused() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let exchanged = w.exchange(&subject, &[]).unwrap().access_token;
    // An exchanged token (aud …/v1, act present) is never a subject.
    let r = w.exchange(&exchanged, &[]);
    assert_code(r.clone(), C::InvalidRequest);
    assert_eq!(
        description(r),
        "subject_token is already an exchanged token"
    );
    // Even with an allowed audience, any `act` (object or null) is refused
    // before the audience rule.
    let key = &w.signer.0 .0;
    let header = json!({"alg": "ES256", "typ": "at+jwt", "kid": thumb(key)});
    for act in [json!({"sub": "other"}), Value::Null] {
        let mut c = claims_of(&subject);
        c["act"] = act;
        let r = w.exchange(&sign(&header, &c, key), &[]);
        assert_eq!(
            description(r),
            "subject_token is already an exchanged token"
        );
    }
}

#[test]
fn subject_token_rejections() {
    let w = Ex::new();
    let good = claims_of(&w.user_token(TEAM_RESOURCE, &["team:read"]));
    let as_key = w.signer.0 .0.clone();
    let hdr = |k: &SigningKey, typ: &str| json!({"alg": "ES256", "typ": typ, "kid": thumb(k)});
    let other = SigningKey::from_bytes(&[4u8; 32].into()).unwrap();
    let with = |f: &dyn Fn(&mut Value)| {
        let mut c = good.clone();
        f(&mut c);
        sign(&hdr(&as_key, "at+jwt"), &c, &as_key)
    };
    let cases: Vec<(String, &str)> = vec![
        ("not-a-jwt".into(), "malformed subject_token"),
        (
            sign(&hdr(&as_key, "JWT"), &good, &as_key),
            "subject_token is not an access token",
        ),
        (
            sign(&hdr(&other, "at+jwt"), &good, &other),
            "subject_token key unknown",
        ),
        // Right kid, wrong key: signature check.
        (
            sign(&hdr(&as_key, "at+jwt"), &good, &other),
            "subject_token signature invalid",
        ),
        (
            with(&|c| c["iss"] = json!("https://evil.example")),
            "subject_token issuer mismatch",
        ),
        (
            with(&|c| c["sub"] = json!("raw-upstream-sub")),
            "subject_token sub invalid",
        ),
        (
            with(&|c| {
                c.as_object_mut().unwrap().remove("family_id");
            }),
            "subject_token claims incomplete",
        ),
        (
            with(&|c| c["aud"] = json!([TEAM_RESOURCE])),
            "subject_token claims incomplete",
        ),
        (
            with(&|c| c["org_id"] = json!("")),
            "subject_token claims invalid",
        ),
        (
            with(&|c| c["exp"] = json!(T0 + 3600)),
            "subject_token times invalid",
        ),
        (
            with(&|c| c["iat"] = json!(T0 + 120)),
            "subject_token times invalid",
        ),
        (
            with(&|c| c["aud"] = json!(RESOURCE)),
            "subject_token audience not exchangeable by this client",
        ),
        // A plain `…/v1` user token is not an adapter token.
        (
            with(&|c| c["aud"] = json!(OTHER_RESOURCE)),
            "subject_token audience not exchangeable by this client",
        ),
    ];
    for (token, desc) in cases {
        let r = w.exchange(&token, &[]);
        assert_code(r.clone(), C::InvalidRequest);
        assert_eq!(description(r), desc);
    }
}

#[test]
fn revoked_family_is_refused() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    w.store.revoked.borrow_mut().insert("fam-1".into());
    assert_eq!(
        description(w.exchange(&subject, &[])),
        "subject_token revoked"
    );
}

#[test]
fn subject_audience_removed_from_allowlist_is_refused() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let mut w2 = Ex::new();
    w2.resources = ResourceAllowlist::from_config(
        "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 ruvector:read",
    )
    .unwrap();
    let r = w2.exchange(&subject, &[]);
    assert_eq!(
        description(r),
        "subject_token audience not exchangeable by this client"
    );
}

#[test]
fn keyless_signer_fails_closed() {
    let w = Ex::new();
    let subject = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let mut signer = MockSigner::new();
    signer.expect_verifying_keys().returning(Vec::new);
    signer.expect_sign_es256().times(0);
    let mut ep = w.endpoint();
    ep.signer = &signer;
    let f = w.form(&w.assertion(), &subject);
    let r = TokenRequest::from_form(&f).and_then(|r| ep.handle(&r));
    assert_code(r, C::ServerError);
}

#[test]
fn exchange_request_debug_redacts_tokens() {
    let w = Ex::new();
    let s = w.user_token(TEAM_RESOURCE, &["team:read"]);
    let a = w.assertion();
    let req = TokenRequest::from_form(&w.form(&a, &s)).unwrap();
    let dbg = format!("{req:?}");
    assert!(!dbg.contains(&s) && !dbg.contains(&a), "{dbg}");
    assert!(dbg.contains("<redacted>"));
    assert_eq!(req.client_id(), None);
}
