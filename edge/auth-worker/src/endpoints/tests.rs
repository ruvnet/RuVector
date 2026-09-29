//! Endpoint tests over real SQLite (in-memory), a fixed clock, a sequential
//! RNG, a real ES256 signer and a fake upstream IdP.

use super::{authorize::authorize, callback::callback, register::register, token, Ctx};
use crate::config::tests::{load, vars, ISSUER, RESOURCE};
use crate::config::AuthConfig;
use crate::http::Reply;
use crate::signer::tests::{private_jwk, test_key};
use crate::signer::EnvSigner;
use crate::sql::memory::MemoryDb;
use crate::sql_ports::SqlPorts;
use crate::testutil::{block_on, FixedClock, SeqRng};
use crate::upstream::UpstreamHttp;
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{SigningKey, VerifyingKey};
use ruvector_edge_auth::jws::{b64url_decode, b64url_encode};
use ruvector_edge_auth::{AuthError, FetchError, HttpResponse, Jwk, KeySource};
use serde_json::{json, Value};

const T0: u64 = 1_800_000_000;
const REDIRECT: &str = "https://claude.ai/api/mcp/auth_callback";
const VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
const CHALLENGE: &str = "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM";
const FORM: Option<&str> = Some("application/x-www-form-urlencoded");

pub(super) struct World {
    pub cfg: AuthConfig,
    pub store: SqlPorts<MemoryDb>,
    pub rng: SeqRng,
    pub clock: FixedClock,
    pub signer: EnvSigner,
    pub upstream_key: SigningKey,
}

impl World {
    pub fn new() -> Self {
        Self::with_cfg(load(&vars()).unwrap())
    }

    pub fn with_cfg(cfg: AuthConfig) -> Self {
        let store = SqlPorts::new(MemoryDb::new());
        store.migrate().unwrap();
        World {
            cfg,
            store,
            rng: SeqRng::default(),
            clock: FixedClock::at(T0),
            signer: EnvSigner::from_secret(Some(&private_jwk(&test_key(9), true))),
            upstream_key: test_key(5),
        }
    }

    pub fn ctx(&self) -> Ctx<'_, SqlPorts<MemoryDb>, SeqRng, FixedClock, EnvSigner> {
        Ctx {
            cfg: &self.cfg,
            store: &self.store,
            rng: &self.rng,
            clock: &self.clock,
            signer: &self.signer,
        }
    }

    /// Register from a fresh IP bucket (the per-IP window has its own test).
    pub fn register(&self, body: Value) -> Reply {
        static NEXT_IP: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(0);
        let ip = NEXT_IP.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        self.register_from(&format!("ip-{ip}"), body)
    }

    pub fn register_from(&self, client_key: &str, body: Value) -> Reply {
        register(
            &self.ctx(),
            client_key,
            Some("application/json"),
            body.to_string().as_bytes(),
        )
    }

    pub fn client_id(&self) -> String {
        let r = self.register(json!({
            "redirect_uris": [REDIRECT],
            "grant_types": ["authorization_code", "refresh_token"],
            "client_name": "Test <b>Client</b>",
        }));
        assert_eq!(r.status, 201, "{}", String::from_utf8_lossy(&r.body));
        body_json(&r)["client_id"].as_str().unwrap().to_string()
    }

    /// An upstream access token as auth.cognitum.one would mint it.
    pub fn upstream_token(&self, claims: Value) -> String {
        let kid = Jwk::from_verifying_key(self.upstream_key.verifying_key()).kid;
        let h = b64url_encode(
            json!({"alg":"ES256","typ":"JWT","kid":kid})
                .to_string()
                .as_bytes(),
        );
        let p = b64url_encode(claims.to_string().as_bytes());
        let input = format!("{h}.{p}");
        let sig: p256::ecdsa::Signature = self.upstream_key.sign(input.as_bytes());
        format!("{input}.{}", b64url_encode(&sig.to_bytes()))
    }

    pub fn good_upstream_claims(&self) -> Value {
        json!({
            "iss": "https://auth.cognitum.one", "aud": "dcr-edge-test",
            "client_id": "dcr-edge-test", "sub": "user-1", "org_id": "org_1",
            "workspace_id": "ws_1", "scope": "openid ruvector:read", "jti": "j1",
            "family_id": "f1", "typ": "access", "iat": T0, "exp": T0 + 900,
        })
    }
}

pub(super) fn body_json(r: &Reply) -> Value {
    serde_json::from_slice(&r.body).unwrap_or(Value::Null)
}

fn query_of(url: &str) -> Vec<(String, String)> {
    url::Url::parse(url)
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
}

fn param(url: &str, k: &str) -> Option<String> {
    query_of(url)
        .into_iter()
        .find(|(n, _)| n == k)
        .map(|(_, v)| v)
}

/// Fake upstream token endpoint.
pub(super) struct FakeUpstream(pub HttpResponse);

impl UpstreamHttp for FakeUpstream {
    async fn post_form(&self, _url: &str, _body: String) -> Result<HttpResponse, FetchError> {
        Ok(self.0.clone())
    }
}

impl FakeUpstream {
    pub fn tokens(access_token: &str) -> Self {
        let body = json!({"access_token": access_token, "token_type": "Bearer"}).to_string();
        FakeUpstream(HttpResponse {
            status: 200,
            body: body.into_bytes(),
        })
    }
}

/// Upstream JWKS with one key.
pub(super) struct OneKey(pub VerifyingKey);

impl KeySource for OneKey {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        (Jwk::from_verifying_key(&self.0).kid == kid)
            .then_some(self.0)
            .ok_or(AuthError::UnknownKid)
    }
}

fn authorize_query(client_id: &str, resource: &str) -> String {
    url::form_urlencoded::Serializer::new(String::new())
        .append_pair("response_type", "code")
        .append_pair("client_id", client_id)
        .append_pair("redirect_uri", REDIRECT)
        .append_pair("scope", "ruvector:read offline_access")
        .append_pair("state", "client-state")
        .append_pair("code_challenge", CHALLENGE)
        .append_pair("code_challenge_method", "S256")
        .append_pair("resource", resource)
        .finish()
}

/// Run `/authorize`; return (upstream URL, cookie header for the callback).
fn start(w: &World, client_id: &str) -> (String, String) {
    let r = authorize(&w.ctx(), Some(&authorize_query(client_id, RESOURCE)));
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    let cookie = r.header("Set-Cookie").unwrap();
    let pair = cookie.split(';').next().unwrap().to_string();
    let r = submit_consent(w, &r, Some(&pair));
    assert_eq!(r.status, 303, "{}", String::from_utf8_lossy(&r.body));
    (r.header("Location").unwrap().to_string(), pair)
}

/// Hidden `name` field of the consent form.
fn form_field(html: &str, name: &str) -> String {
    let marker = format!("name=\"{name}\" value=\"");
    html.split(&marker)
        .nth(1)
        .unwrap()
        .split('"')
        .next()
        .unwrap()
        .to_string()
}

/// Click Continue on a rendered consent page.
fn submit_consent(w: &World, page: &Reply, cookie: Option<&str>) -> Reply {
    let html = String::from_utf8(page.body.clone()).unwrap();
    let body = url::form_urlencoded::Serializer::new(String::new())
        .append_pair("flow", &form_field(&html, "flow"))
        .append_pair("consent", &form_field(&html, "consent"))
        .finish();
    super::authorize::consent_submit(&w.ctx(), cookie, FORM, body.as_bytes())
}

/// Full login: authorize -> upstream -> callback; returns our code.
fn login(w: &World, client_id: &str) -> String {
    let (up, cookie) = start(w, client_id);
    let state = param(&up, "state").unwrap();
    let q = format!("state={state}&code=up-code&iss=https%3A%2F%2Fauth.cognitum.one");
    let at = w.upstream_token(w.good_upstream_claims());
    let r = block_on(callback(
        &w.ctx(),
        Some(&q),
        Some(&cookie),
        &FakeUpstream::tokens(&at),
        OneKey(*w.upstream_key.verifying_key()),
    ));
    assert_eq!(r.status, 302, "{}", String::from_utf8_lossy(&r.body));
    let loc = r.header("Location").unwrap();
    assert!(loc.starts_with(REDIRECT));
    assert_eq!(param(loc, "state").as_deref(), Some("client-state"));
    assert_eq!(param(loc, "iss").as_deref(), Some(ISSUER));
    param(loc, "code").unwrap()
}

fn token_form(w: &World, pairs: &[(&str, &str)]) -> Reply {
    let body = url::form_urlencoded::Serializer::new(String::new())
        .extend_pairs(pairs)
        .finish();
    token::token(&w.ctx(), FORM, body.as_bytes())
}

fn redeem(w: &World, client_id: &str, code: &str) -> Reply {
    token_form(
        w,
        &[
            ("grant_type", "authorization_code"),
            ("code", code),
            ("redirect_uri", REDIRECT),
            ("client_id", client_id),
            ("code_verifier", VERIFIER),
            ("resource", RESOURCE),
        ],
    )
}

#[test]
fn end_to_end_code_flow_mints_resource_bound_es256_token() {
    use p256::ecdsa::signature::Verifier as _;
    let w = World::new();
    let client_id = w.client_id();
    let code = login(&w, &client_id);
    let r = redeem(&w, &client_id, &code);
    assert_eq!(r.status, 200, "{}", String::from_utf8_lossy(&r.body));
    assert_eq!(r.header("Cache-Control"), Some("no-store"));
    let body = body_json(&r);
    assert_eq!(body["token_type"], "Bearer");
    assert!(body["refresh_token"].is_string());
    let jwt = body["access_token"].as_str().unwrap();
    let parts: Vec<&str> = jwt.split('.').collect();
    let header: Value = serde_json::from_slice(&b64url_decode(parts[0]).unwrap()).unwrap();
    let claims: Value = serde_json::from_slice(&b64url_decode(parts[1]).unwrap()).unwrap();
    assert_eq!(header["alg"], "ES256");
    assert_eq!(header["typ"], "at+jwt");
    assert_eq!(claims["aud"], RESOURCE);
    assert_eq!(claims["iss"], ISSUER);
    // Edge subject derived from (upstream iss, upstream sub), never the raw sub.
    let sub = claims["sub"].as_str().unwrap();
    assert!(sub.starts_with("es1_") && sub.len() == 30, "{sub}");
    assert_eq!(claims["org_id"], "org_1");
    assert_eq!(claims["client_id"], client_id.as_str());
    let sig = p256::ecdsa::Signature::from_slice(&b64url_decode(parts[2]).unwrap()).unwrap();
    let key = test_key(9);
    key.verifying_key()
        .verify(format!("{}.{}", parts[0], parts[1]).as_bytes(), &sig)
        .unwrap();
    // One-time code.
    assert_eq!(
        body_json(&redeem(&w, &client_id, &code))["error"],
        "invalid_grant"
    );
}

#[test]
fn refresh_rotates_and_reuse_revokes_the_family() {
    let w = World::new();
    let client_id = w.client_id();
    let code = login(&w, &client_id);
    let first = body_json(&redeem(&w, &client_id, &code));
    let rt1 = first["refresh_token"].as_str().unwrap().to_string();
    let refresh = |rt: &str| {
        token_form(
            &w,
            &[
                ("grant_type", "refresh_token"),
                ("refresh_token", rt),
                ("client_id", &client_id),
            ],
        )
    };
    let second = refresh(&rt1);
    assert_eq!(
        second.status,
        200,
        "{}",
        String::from_utf8_lossy(&second.body)
    );
    let rt2 = body_json(&second)["refresh_token"]
        .as_str()
        .unwrap()
        .to_string();
    assert_ne!(rt1, rt2);
    // Reuse of rt1 is detected and kills the family, including rt2.
    assert_eq!(body_json(&refresh(&rt1))["error"], "invalid_grant");
    assert_eq!(body_json(&refresh(&rt2))["error"], "invalid_grant");
}

#[test]
fn revoke_kills_refresh_token_and_is_silent_for_unknown() {
    let w = World::new();
    let client_id = w.client_id();
    let code = login(&w, &client_id);
    let rt = body_json(&redeem(&w, &client_id, &code))["refresh_token"]
        .as_str()
        .unwrap()
        .to_string();
    let rev = |t: &str| {
        let body = format!("token={t}&client_id={client_id}");
        token::revoke(&w.ctx(), FORM, body.as_bytes())
    };
    assert_eq!(rev(&rt).status, 200);
    assert_eq!(rev("unknown-token").status, 200);
    let r = token_form(
        &w,
        &[
            ("grant_type", "refresh_token"),
            ("refresh_token", &rt),
            ("client_id", &client_id),
        ],
    );
    assert_eq!(body_json(&r)["error"], "invalid_grant");
    let bad = token::revoke(&w.ctx(), Some("application/json"), b"{}");
    assert_eq!(bad.status, 415);
    assert_eq!(token::revoke(&w.ctx(), FORM, b"client_id=x").status, 400);
}

#[path = "tests_errors.rs"]
mod errors;

#[path = "tests_abuse.rs"]
mod abuse;

#[path = "tests_upstream.rs"]
mod upstream_leg;

#[path = "tests_consent.rs"]
mod consent_form;

#[path = "tests_exchange.rs"]
mod exchange;
