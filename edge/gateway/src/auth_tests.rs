//! Gateway authentication over real ES256 tokens and mocked key sources.

use super::*;
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{SigningKey, VerifyingKey};
use ruvector_edge_auth::jws::b64url_encode;
use ruvector_edge_auth::{Jwk, UpstreamFirstPartyPolicy};
use serde_json::{json, Value};
use std::future::Future;
use std::sync::Arc;
use std::task::{Context, Poll, Wake, Waker};

const NOW: u64 = 1_800_000_000;
const EDGE: &str = "https://as.example.workers.dev";
const UPSTREAM: &str = "https://auth.cognitum.one";
const REST: &str = "https://gw.example.workers.dev/v1";
const MCP: &str = "https://gw.example.workers.dev/v1/mcp";
/// Edge subject shape required by the tenancy core (`es1_` + 26 base32).
const SUB: &str = "es1_abcdefghijklmnopqrstuvwxyz";

struct NoopWake;
impl Wake for NoopWake {
    fn wake(self: Arc<Self>) {}
}

fn block_on<F: Future>(fut: F) -> F::Output {
    let waker = Waker::from(Arc::new(NoopWake));
    let mut cx = Context::from_waker(&waker);
    let mut fut = std::pin::pin!(fut);
    loop {
        if let Poll::Ready(v) = fut.as_mut().poll(&mut cx) {
            return v;
        }
    }
}

struct Fixed;
impl Clock for Fixed {
    fn now_unix(&self) -> u64 {
        NOW
    }
}

struct Keys(Vec<VerifyingKey>);
impl KeySource for Keys {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        self.0
            .iter()
            .find(|k| Jwk::from_verifying_key(k).kid == kid)
            .copied()
            .ok_or(AuthError::UnknownKid)
    }
}

struct Down;
impl KeySource for Down {
    async fn verifying_key(&self, _kid: &str) -> Result<VerifyingKey, AuthError> {
        Err(AuthError::KeysUnavailable)
    }
}

fn key(seed: u8) -> SigningKey {
    let mut b = [seed; 32];
    b[0] = 1;
    SigningKey::from_bytes(&b.into()).unwrap()
}

fn token(k: &SigningKey, typ: Option<&str>, claims: Value) -> String {
    let mut h = json!({"alg": "ES256", "kid": Jwk::from_verifying_key(k.verifying_key()).kid});
    if let Some(t) = typ {
        h["typ"] = json!(t);
    }
    let input = format!(
        "{}.{}",
        b64url_encode(h.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    let sig: p256::ecdsa::Signature = k.sign(input.as_bytes());
    format!("Bearer {input}.{}", b64url_encode(&sig.to_bytes()))
}

fn edge_claims(aud: &str) -> Value {
    json!({
        "iss": EDGE, "aud": aud, "sub": SUB, "client_id": "edc-1",
        "org_id": "org_1", "workspace_id": "ws_1", "scope": "mcp:read",
        "jti": "AAECAwQFBgcICQoLDA0ODw", "family_id": "EBESExQVFhcYGRobHB0eHw",
        "upstream_iss": UPSTREAM, "iat": NOW - 10, "exp": NOW + 890,
    })
}

fn cfg(upstream: bool) -> GatewayConfig {
    GatewayConfig {
        edge_issuer: EDGE.into(),
        edge_jwks_url: format!("{EDGE}/.well-known/jwks.json"),
        rest_resource: ResourceUrl::parse(REST).unwrap(),
        mcp_resource: ResourceUrl::parse(MCP).unwrap(),
        upstream: upstream.then(|| UpstreamFirstPartyPolicy {
            issuer: UPSTREAM.into(),
            first_party_auds: vec!["cli".into()],
            accepted_kids: vec![Jwk::from_verifying_key(key(2).verifying_key()).kid],
        }),
        upstream_jwks_url: format!("{UPSTREAM}/.well-known/jwks.json"),
    }
}

fn run(c: &GatewayConfig, header: Option<&str>, edge: Keys) -> Result<TenantContext, Denied> {
    let upstream = Keys(vec![*key(2).verifying_key()]);
    block_on(authenticate(
        header,
        c,
        &c.rest_resource,
        RouteSurface::Rest,
        edge,
        || upstream,
        Fixed,
    ))
}

fn edge_keys() -> Keys {
    Keys(vec![*key(1).verifying_key()])
}

#[test]
fn accepts_edge_token_bound_to_this_resource() {
    let t = token(&key(1), Some("at+jwt"), edge_claims(REST));
    let ctx = run(&cfg(false), Some(&t), edge_keys()).unwrap();
    assert_eq!(ctx.org_id(), "org_1");
    assert_eq!(ctx.sub(), SUB);
    assert_eq!(ctx.client_id(), "edc-1");
}

#[test]
fn missing_token_gets_bare_resource_metadata_challenge() {
    let d = run(&cfg(false), None, edge_keys()).unwrap_err();
    assert_eq!(d.code, ProblemCode::InvalidToken);
    assert_eq!(
        d.www_authenticate.as_deref(),
        Some("Bearer resource_metadata=\"https://gw.example.workers.dev/.well-known/oauth-protected-resource/v1\", scope=\"ruvector:read\"")
    );
}

#[test]
fn other_resource_audience_is_403_without_challenge() {
    let t = token(&key(1), Some("at+jwt"), edge_claims(MCP));
    let d = run(&cfg(false), Some(&t), edge_keys()).unwrap_err();
    assert_eq!(
        d,
        Denied {
            code: ProblemCode::AudienceNotAllowed,
            www_authenticate: None
        }
    );
}

#[test]
fn forged_or_mistyped_edge_tokens_are_401_with_challenge() {
    let wrong_key = token(&key(3), Some("at+jwt"), edge_claims(REST));
    let wrong_typ = token(&key(1), Some("JWT"), edge_claims(REST));
    let mut expired = edge_claims(REST);
    expired["exp"] = json!(NOW - 3600);
    let expired = token(&key(1), Some("at+jwt"), expired);
    for t in [wrong_key, wrong_typ, expired] {
        let d = run(&cfg(false), Some(&t), edge_keys()).unwrap_err();
        assert_eq!(d.code, ProblemCode::InvalidToken);
        assert!(d
            .www_authenticate
            .unwrap()
            .contains("error=\"invalid_token\""));
    }
}

/// A real upstream first-party token: UUID `sub`, `typ = access`.
fn upstream_token() -> String {
    let mut up = edge_claims("cli");
    up["iss"] = json!(UPSTREAM);
    up["client_id"] = json!("cli");
    up["typ"] = json!("access");
    up["sub"] = json!(UPSTREAM_SUB);
    up.as_object_mut().unwrap().remove("upstream_iss");
    token(&key(2), Some("JWT"), up)
}

const UPSTREAM_SUB: &str = "8d0c7a52-4f1e-4b8a-9c3d-2e5f6a7b8c9d";

/// Regression (§5.5 path unusable): a real upstream token (UUID `sub`) is
/// accepted behind the flag, with `sub` normalised to the edge subject.
#[test]
fn upstream_tokens_only_behind_the_flag() {
    let t = upstream_token();
    assert_eq!(
        run(&cfg(false), Some(&t), edge_keys()).unwrap_err().code,
        ProblemCode::InvalidToken
    );
    let ctx = run(&cfg(true), Some(&t), edge_keys()).unwrap();
    assert_eq!(
        ctx.sub(),
        ruvector_edge_auth::subject::edge_subject(UPSTREAM, UPSTREAM_SUB)
    );
}

fn run_mcp(c: &GatewayConfig, header: Option<&str>) -> Result<TenantContext, Denied> {
    let route = crate::routes::classify(&worker::Method::Post, "/v1/mcp");
    let (resource, surface) = crate::routes::auth_target(route, c).unwrap();
    let upstream = Keys(vec![*key(2).verifying_key()]);
    block_on(authenticate(
        header,
        c,
        resource,
        surface,
        edge_keys(),
        || upstream,
        Fixed,
    ))
}

/// Regression (/v1/mcp authenticated as REST): on `/v1/mcp` an MCP-audience
/// token is accepted, a `/v1` token is 403, the 401 challenge names the
/// `/v1/mcp` metadata, and upstream first-party tokens are refused.
#[test]
fn mcp_route_uses_the_mcp_resource_audience_and_surface() {
    let c = cfg(true);
    let mcp = token(&key(1), Some("at+jwt"), edge_claims(MCP));
    assert!(run_mcp(&c, Some(&mcp)).is_ok());
    let rest = token(&key(1), Some("at+jwt"), edge_claims(REST));
    assert_eq!(
        run_mcp(&c, Some(&rest)).unwrap_err(),
        Denied {
            code: ProblemCode::AudienceNotAllowed,
            www_authenticate: None
        }
    );
    let d = run_mcp(&c, None).unwrap_err();
    assert!(d
        .www_authenticate
        .unwrap()
        .contains("/.well-known/oauth-protected-resource/v1/mcp\""));
    let d = run_mcp(&c, Some(&upstream_token())).unwrap_err();
    assert_eq!(d.code, ProblemCode::InvalidToken);
}

/// Regression (ADR §5.4.7): edge tokens live at most 900 s.
#[test]
fn edge_tokens_longer_than_900s_are_refused() {
    let mut long = edge_claims(REST);
    long["iat"] = json!(NOW - 10);
    long["exp"] = json!(NOW + 3000);
    let t = token(&key(1), Some("at+jwt"), long);
    assert_eq!(
        run(&cfg(false), Some(&t), edge_keys()).unwrap_err().code,
        ProblemCode::InvalidToken
    );
}

#[test]
fn unreachable_jwks_is_503() {
    let t = token(&key(1), Some("at+jwt"), edge_claims(REST));
    let c = cfg(false);
    let d = block_on(authenticate(
        Some(&t),
        &c,
        &c.rest_resource,
        RouteSurface::Rest,
        Down,
        || Down,
        Fixed,
    ))
    .unwrap_err();
    assert_eq!(d.code, ProblemCode::JwksUnavailable);
    assert!(d.www_authenticate.is_none());
}

#[test]
fn denial_mapping() {
    assert_eq!(
        denial(&AuthError::BadSignature),
        (ProblemCode::InvalidToken, true)
    );
    assert_eq!(
        denial(&AuthError::InvalidConfig("x")),
        (ProblemCode::ServerError, false)
    );
}
