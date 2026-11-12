//! The edge authorization server side of the cross-crate end-to-end tests:
//! the real `ruvector-edge-authz` token endpoint over in-memory ports,
//! configured from the **deployed** `auth-worker/wrangler.toml` vars (issuer
//! and resource allowlist), so drift between the two Workers fails here.

use p256::ecdsa::signature::Signer as _;
use p256::ecdsa::{Signature, SigningKey, VerifyingKey};
use ruvector_edge_auth::jws::b64url_encode;
use ruvector_edge_auth::{Clock, Jwk};
use ruvector_edge_authz::client::ClientRecord;
use ruvector_edge_authz::code::AuthorizationCodeRecord;
use ruvector_edge_authz::confidential::JWT_BEARER_ASSERTION;
use ruvector_edge_authz::exchange::{ACCESS_TOKEN_TYPE, TOKEN_EXCHANGE_GRANT};
use ruvector_edge_authz::federation::UpstreamIdentity;
use ruvector_edge_authz::refresh::RefreshTokenRecord;
use ruvector_edge_authz::resource::TEAM_RESOURCE_URL;
use ruvector_edge_authz::token::{mint_access_token, MintRequest, TokenRequest, TokenResponse};
use ruvector_edge_authz::{
    AssertionReplayStore, ClientStore, CodeStore, ConfidentialClients, OAuthError, RefreshStore,
    ResourceAllowlist, ResourceUrl, Rng, Signer, StoreError, TokenEndpoint,
};
use serde_json::{json, Value as Json};
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

/// The auth Worker's deployed configuration.
pub const AUTH_WRANGLER: &str = include_str!("../../auth-worker/wrangler.toml");
/// The team.ruv.io adapter's confidential `client_id`.
pub const ADAPTER: &str = "team-ruv-io";
/// Upstream issuer of every test identity (the tenant namespace).
pub const UPSTREAM: &str = "https://auth.cognitum.one";

/// A `[vars]` string value of [`AUTH_WRANGLER`].
pub fn wrangler_var(name: &str) -> String {
    let prefix = format!("{name} = \"");
    AUTH_WRANGLER
        .lines()
        .find_map(|l| l.strip_prefix(&prefix)?.strip_suffix('"'))
        .unwrap_or_else(|| panic!("{name} missing from auth-worker/wrangler.toml"))
        .to_string()
}

/// A fixed clock (both Workers read the same instant).
#[derive(Clone, Copy)]
pub struct At(pub u64);
impl Clock for At {
    fn now_unix(&self) -> u64 {
        self.0
    }
}

/// Deterministic, never-repeating bytes (SHA-256 of a counter).
#[derive(Default)]
pub struct SeqRng(Cell<u64>);
impl Rng for SeqRng {
    fn fill(&self, buf: &mut [u8]) -> Result<(), StoreError> {
        use sha2::{Digest, Sha256};
        for chunk in buf.chunks_mut(32) {
            let n = self.0.get();
            self.0.set(n + 1);
            let d = Sha256::digest(n.to_le_bytes());
            chunk.copy_from_slice(&d[..chunk.len()]);
        }
        Ok(())
    }
}

/// The AS signing key; `kid` is its RFC 7638 thumbprint, as in the Worker.
pub struct AsSigner(pub SigningKey);
impl Signer for AsSigner {
    fn kid(&self) -> String {
        Jwk::from_verifying_key(self.0.verifying_key()).kid
    }
    fn sign_es256(&self, input: &[u8]) -> Result<[u8; 64], StoreError> {
        let sig: Signature = self.0.sign(input);
        Ok(sig.to_bytes().into())
    }
    fn verifying_keys(&self) -> Vec<VerifyingKey> {
        vec![*self.0.verifying_key()]
    }
}

/// The ports the exchange grant touches; the others are inert.
#[derive(Default)]
pub struct Stores {
    revoked: RefCell<BTreeSet<String>>,
    assertions: RefCell<BTreeMap<[u8; 32], u64>>,
}
impl ClientStore for Stores {
    fn insert_client(&self, _: &ClientRecord) -> Result<(), StoreError> {
        Ok(())
    }
    fn get_client(&self, _: &str) -> Result<Option<ClientRecord>, StoreError> {
        Ok(None)
    }
}
impl CodeStore for Stores {
    fn insert_code(&self, _: &AuthorizationCodeRecord) -> Result<(), StoreError> {
        Ok(())
    }
    fn take_code(&self, _: &[u8; 32]) -> Result<Option<AuthorizationCodeRecord>, StoreError> {
        Ok(None)
    }
    fn record_redeemed(&self, _: &[u8; 32], _: &str, _: u64) -> Result<(), StoreError> {
        Ok(())
    }
    fn redeemed_family(&self, _: &[u8; 32], _: u64) -> Result<Option<String>, StoreError> {
        Ok(None)
    }
}
impl RefreshStore for Stores {
    fn insert_refresh(&self, _: &RefreshTokenRecord) -> Result<(), StoreError> {
        Ok(())
    }
    fn get_refresh(&self, _: &[u8; 32]) -> Result<Option<RefreshTokenRecord>, StoreError> {
        Ok(None)
    }
    fn mark_rotated(&self, _: &[u8; 32]) -> Result<bool, StoreError> {
        Ok(false)
    }
    fn revoke_family(&self, f: &str) -> Result<(), StoreError> {
        self.revoked.borrow_mut().insert(f.to_string());
        Ok(())
    }
    fn is_family_revoked(&self, f: &str) -> Result<bool, StoreError> {
        Ok(self.revoked.borrow().contains(f))
    }
}
impl AssertionReplayStore for Stores {
    fn record_assertion(&self, h: &[u8; 32], until: u64, now: u64) -> Result<bool, StoreError> {
        let mut m = self.assertions.borrow_mut();
        if m.get(h).is_some_and(|exp| now < *exp) {
            return Ok(false);
        }
        m.insert(*h, until);
        Ok(true)
    }
}

/// Compact ES256 JWS over arbitrary JSON.
pub fn sign_jws(header: &Json, claims: &Json, k: &SigningKey) -> String {
    let input = format!(
        "{}.{}",
        b64url_encode(header.to_string().as_bytes()),
        b64url_encode(claims.to_string().as_bytes())
    );
    let sig: Signature = k.sign(input.as_bytes());
    format!("{input}.{}", b64url_encode(&sig.to_bytes()))
}

/// Decoded JWS payload.
pub fn payload(token: &str) -> Json {
    let p = token.split('.').nth(1).expect("compact JWS");
    serde_json::from_slice(&ruvector_edge_auth::jws::b64url_decode(p).unwrap()).unwrap()
}

/// The edge AS: deployed issuer + allowlist, one registered adapter.
pub struct EdgeAs {
    pub issuer: String,
    pub resources: ResourceAllowlist,
    pub confidential: ConfidentialClients,
    pub signer: AsSigner,
    pub adapter_key: SigningKey,
    pub stores: Stores,
    pub rng: SeqRng,
    pub clock: At,
    jti: Cell<u64>,
}

impl EdgeAs {
    /// The AS at `now`; the adapter may exchange team.ruv.io tokens for
    /// `ceiling`.
    pub fn new(now: u64, ceiling: &str) -> Self {
        let resources = ResourceAllowlist::from_config(&wrangler_var("RESOURCE_ALLOWLIST"))
            .expect("deployed RESOURCE_ALLOWLIST loads");
        let adapter_key = SigningKey::from_bytes(&[3u8; 32].into()).unwrap();
        let j = Jwk::from_verifying_key(adapter_key.verifying_key());
        let registry = json!([{
            "client_id": ADAPTER,
            "jwk": { "kty": "EC", "crv": "P-256", "x": j.x, "y": j.y, "kid": j.kid },
            "subject_audiences": [TEAM_RESOURCE_URL],
            "scope": ceiling,
        }]);
        EdgeAs {
            issuer: wrangler_var("ISSUER"),
            confidential: ConfidentialClients::from_config(&registry.to_string(), &resources)
                .expect("adapter registry loads"),
            resources,
            signer: AsSigner(SigningKey::from_bytes(&[7u8; 32].into()).unwrap()),
            adapter_key,
            stores: Stores::default(),
            rng: SeqRng::default(),
            clock: At(now),
            jti: Cell::new(0),
        }
    }

    /// The AS's published verifying keys (the gateway's JWKS view).
    pub fn jwks(&self) -> Vec<VerifyingKey> {
        self.signer.verifying_keys()
    }

    /// A user's edge access token for `resource`, as the authorization-code
    /// grant mints it (16-byte random `family_id`, like `new_family_id`).
    pub fn user_token(&self, user: &str, org: &str, resource: &str, scopes: &[&str]) -> String {
        let scopes: Vec<String> = scopes.iter().map(|s| s.to_string()).collect();
        let family = ruvector_edge_authz::random_secret(&self.rng, 16).unwrap();
        let identity = UpstreamIdentity {
            upstream_iss: UPSTREAM.into(),
            sub: user.into(),
            org_id: org.into(),
            workspace_id: "ws1".into(),
        };
        let req = MintRequest {
            issuer: &self.issuer,
            resource: &ResourceUrl::parse(resource).unwrap(),
            client_id: "edc-e2e-client",
            identity: &identity,
            family_id: &family,
            scopes: &scopes,
            act: None,
        };
        mint_access_token(&self.signer, &self.rng, &self.clock, &req)
            .unwrap()
            .0
    }

    /// The adapter's RFC 7523 client assertion (fresh `jti`).
    pub fn assertion(&self) -> String {
        self.jti.set(self.jti.get() + 1);
        let now = self.clock.0;
        let kid = Jwk::from_verifying_key(self.adapter_key.verifying_key()).kid;
        sign_jws(
            &json!({ "alg": "ES256", "typ": "JWT", "kid": kid }),
            &json!({
                "iss": ADAPTER, "sub": ADAPTER, "aud": format!("{}/token", self.issuer),
                "iat": now, "exp": now + 120, "jti": format!("e2e-{}", self.jti.get()),
            }),
            &self.adapter_key,
        )
    }

    /// `POST /token` token exchange by the adapter for `resource`.
    pub fn exchange(
        &self,
        subject: &str,
        resource: &str,
        scope: Option<&str>,
    ) -> Result<TokenResponse, OAuthError> {
        let assertion = self.assertion();
        let mut form = vec![
            ("grant_type", TOKEN_EXCHANGE_GRANT),
            ("client_assertion_type", JWT_BEARER_ASSERTION),
            ("client_assertion", assertion.as_str()),
            ("subject_token", subject),
            ("subject_token_type", ACCESS_TOKEN_TYPE),
            ("resource", resource),
        ];
        if let Some(s) = scope {
            form.push(("scope", s));
        }
        let pairs: Vec<(String, String)> = form
            .iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect();
        let ep = TokenEndpoint {
            issuer: &self.issuer,
            resources: &self.resources,
            clients: &self.stores,
            codes: &self.stores,
            refresh: &self.stores,
            signer: &self.signer,
            rng: &self.rng,
            clock: &self.clock,
            confidential: &self.confidential,
            assertions: &self.stores,
        };
        ep.handle(&TokenRequest::from_form(&pairs)?)
    }
}
