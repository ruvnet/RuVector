//! Shared test fixtures: deterministic ES256 keys, token minting, a settable
//! clock and counting network/key-source doubles. Test-only.

use crate::clock::Clock;
use crate::error::AuthError;
use crate::jwks::{FetchError, HttpFetch, HttpResponse, Jwk, JwkSet, KeySource};
use crate::jws::b64url_encode;
use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use serde_json::{json, Value};
use std::cell::{Cell, RefCell};
use std::collections::BTreeMap;

pub const EDGE_ISS: &str = "https://ruvector-edge-auth.example.workers.dev";
pub const UPSTREAM_ISS: &str = "https://auth.cognitum.one";
pub const RESOURCE: &str = "https://ruvector-edge-gateway.example.workers.dev/v1/mcp";
/// The sibling edge resource of [`RESOURCE`] on the same gateway.
pub const SIBLING: &str = "https://ruvector-edge-gateway.example.workers.dev/v1";
pub const JWKS_URL: &str = "https://ruvector-edge-auth.example.workers.dev/.well-known/jwks.json";
pub const CLI_CLIENT: &str = "ruvector-edge-cli";
pub const NOW: u64 = 1_790_000_000;

/// Deterministic P-256 key from a one-byte seed (`[seed; 32]` scalar).
pub fn key(seed: u8) -> SigningKey {
    SigningKey::from_bytes(&[seed; 32].into()).expect("valid scalar")
}

pub fn kid(k: &SigningKey) -> String {
    Jwk::from_verifying_key(k.verifying_key()).kid
}

/// Compact JWS with an explicit (possibly bogus) signature.
pub fn with_sig(header: &Value, claims: &Value, sig: &[u8]) -> String {
    format!(
        "{}.{}.{}",
        b64url_encode(header.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes()),
        b64url_encode(sig)
    )
}

/// Properly ES256-signed compact JWS.
pub fn sign(header: &Value, claims: &Value, k: &SigningKey) -> String {
    let input = format!(
        "{}.{}",
        b64url_encode(header.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    let sig: Signature = k.sign(input.as_bytes());
    format!("{input}.{}", b64url_encode(&sig.to_bytes()))
}

pub fn edge_header(k: &SigningKey) -> Value {
    json!({"alg": "ES256", "typ": "at+jwt", "kid": kid(k)})
}

/// A valid edge-issued claim set at [`NOW`].
pub fn edge_claims() -> Value {
    json!({
        "iss": EDGE_ISS,
        "aud": RESOURCE,
        "sub": "user-1",
        "client_id": "edge-client-abc",
        "org_id": "org-1",
        "workspace_id": "ws-1",
        "scope": "ruvector:read ruvector:write",
        "jti": "jti-1",
        "family_id": "fam-edge-1",
        "upstream_iss": UPSTREAM_ISS,
        "iat": NOW - 10,
        "exp": NOW + 890,
    })
}

/// A valid upstream first-party claim set at [`NOW`].
pub fn upstream_claims() -> Value {
    json!({
        "iss": UPSTREAM_ISS,
        "aud": CLI_CLIENT,
        "client_id": CLI_CLIENT,
        "sub": "user-1",
        "org_id": "org-1",
        "workspace_id": "ws-1",
        "scope": "openid profile",
        "jti": "jti-up",
        "family_id": "fam-1",
        "typ": "access",
        "iat": NOW - 10,
        "exp": NOW + 890,
    })
}

pub fn edge_token(k: &SigningKey) -> String {
    sign(&edge_header(k), &edge_claims(), k)
}

pub fn bearer(token: &str) -> String {
    format!("Bearer {token}")
}

/// JWKS document publishing `keys`.
pub fn jwks_body(keys: &[&SigningKey]) -> Vec<u8> {
    let set = JwkSet {
        keys: keys
            .iter()
            .map(|k| Jwk::from_verifying_key(k.verifying_key()))
            .collect(),
    };
    serde_json::to_vec(&set).unwrap()
}

/// Settable clock.
pub struct TestClock(pub Cell<u64>);

impl TestClock {
    pub fn at(t: u64) -> Self {
        TestClock(Cell::new(t))
    }
    pub fn advance(&self, secs: u64) {
        self.0.set(self.0.get() + secs);
    }
}

impl Clock for TestClock {
    fn now_unix(&self) -> u64 {
        self.0.get()
    }
}

/// Scripted [`HttpFetch`] that counts calls; `fail` makes it error;
/// `yield_once` makes the next call pend once (to interleave concurrent
/// callers). Honours the `max_body_bytes + 1` read bound of the port.
pub struct CountingFetch {
    pub status: Cell<u16>,
    pub body: RefCell<Vec<u8>>,
    pub fail: Cell<bool>,
    pub calls: Cell<usize>,
    pub last_url: RefCell<String>,
    pub last_max: Cell<usize>,
    pub yield_once: Cell<bool>,
}

impl CountingFetch {
    pub fn serving(body: Vec<u8>) -> Self {
        CountingFetch {
            status: Cell::new(200),
            body: RefCell::new(body),
            fail: Cell::new(false),
            calls: Cell::new(0),
            last_url: RefCell::new(String::new()),
            last_max: Cell::new(0),
            yield_once: Cell::new(false),
        }
    }
}

/// Pends exactly once (waking itself), then completes.
pub struct YieldNow(pub bool);

impl core::future::Future for YieldNow {
    type Output = ();
    fn poll(
        mut self: core::pin::Pin<&mut Self>,
        cx: &mut core::task::Context<'_>,
    ) -> core::task::Poll<()> {
        if self.0 {
            return core::task::Poll::Ready(());
        }
        self.0 = true;
        cx.waker().wake_by_ref();
        core::task::Poll::Pending
    }
}

impl HttpFetch for CountingFetch {
    async fn get(&self, url: &str, max_body_bytes: usize) -> Result<HttpResponse, FetchError> {
        self.calls.set(self.calls.get() + 1);
        *self.last_url.borrow_mut() = url.to_string();
        self.last_max.set(max_body_bytes);
        if self.yield_once.replace(false) {
            YieldNow(false).await;
        }
        if self.fail.get() {
            return Err(FetchError("down".into()));
        }
        let body = self.body.borrow();
        let keep = body.len().min(max_body_bytes.saturating_add(1));
        Ok(HttpResponse {
            status: self.status.get(),
            body: body[..keep].to_vec(),
        })
    }
}

/// Hand-rolled [`KeySource`] over a fixed map, counting lookups.
pub struct StaticKeys {
    pub keys: BTreeMap<String, VerifyingKey>,
    pub calls: Cell<usize>,
}

impl StaticKeys {
    pub fn of(keys: &[&SigningKey]) -> Self {
        StaticKeys {
            keys: keys.iter().map(|k| (kid(k), *k.verifying_key())).collect(),
            calls: Cell::new(0),
        }
    }
}

impl KeySource for StaticKeys {
    async fn verifying_key(&self, kid: &str) -> Result<VerifyingKey, AuthError> {
        self.calls.set(self.calls.get() + 1);
        self.keys.get(kid).copied().ok_or(AuthError::UnknownKid)
    }
}

pub fn block_on<F: core::future::Future>(f: F) -> F::Output {
    futures::executor::block_on(f)
}
