//! In-memory port fakes and fixtures for unit tests (cfg(test) only).

use crate::authorize::ValidatedAuthorization;
use crate::client::{ClientRecord, DcrPolicy, RegistrationRequest};
use crate::code::AuthorizationCodeRecord;
use crate::federation::{UpstreamConfig, UpstreamFlowState, UpstreamIdentity};
use crate::ports::{ClientStore, Clock, CodeStore, FederationStore, RefreshStore, Rng, Signer};
use crate::refresh::RefreshTokenRecord;
use crate::resource::ResourceAllowlist;
use crate::StoreError;
use p256::ecdsa::{signature::Signer as _, Signature, SigningKey, VerifyingKey};
use ruvector_edge_auth::ResourceUrl;
use std::cell::{Cell, RefCell};
use std::collections::{BTreeMap, BTreeSet};

pub const ISSUER: &str = "https://ruvector-edge-auth.cognitum-consulting-mail.workers.dev";
pub const RESOURCE: &str =
    "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp";
pub const OTHER_RESOURCE: &str =
    "https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1";
pub const REDIRECT: &str = "https://claude.ai/api/mcp/auth_callback";
pub const LOOPBACK: &str = "http://127.0.0.1:53682/callback";
pub const CLIENT_ID: &str = "edc-client-1";
pub const VERIFIER: &str = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
pub const T0: u64 = 1_800_000_000;

/// Settable clock.
pub struct FixedClock(pub Cell<u64>);
impl FixedClock {
    pub fn at(t: u64) -> Self {
        FixedClock(Cell::new(t))
    }
    pub fn advance(&self, secs: u64) {
        self.0.set(self.0.get() + secs);
    }
}
impl Clock for FixedClock {
    fn now_unix(&self) -> u64 {
        self.0.get()
    }
}

/// Deterministic, never-repeating byte stream (SHA-256 of a counter).
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

/// Every store in one in-memory object (like the Worker's single DO).
#[derive(Default)]
pub struct MemStore {
    pub clients: RefCell<BTreeMap<String, ClientRecord>>,
    pub codes: RefCell<BTreeMap<[u8; 32], AuthorizationCodeRecord>>,
    pub flows: RefCell<BTreeMap<String, UpstreamFlowState>>,
    pub refresh: RefCell<BTreeMap<[u8; 32], RefreshTokenRecord>>,
    pub revoked: RefCell<BTreeSet<String>>,
    pub redeemed: RefCell<BTreeMap<[u8; 32], (String, u64)>>,
}

impl ClientStore for MemStore {
    fn insert_client(&self, r: &ClientRecord) -> Result<(), StoreError> {
        let mut m = self.clients.borrow_mut();
        if m.contains_key(&r.client_id) {
            return Err(StoreError("duplicate client".into()));
        }
        m.insert(r.client_id.clone(), r.clone());
        Ok(())
    }
    fn get_client(&self, id: &str) -> Result<Option<ClientRecord>, StoreError> {
        Ok(self.clients.borrow().get(id).cloned())
    }
}

impl CodeStore for MemStore {
    fn insert_code(&self, r: &AuthorizationCodeRecord) -> Result<(), StoreError> {
        self.codes.borrow_mut().insert(r.code_hash, r.clone());
        Ok(())
    }
    fn take_code(&self, h: &[u8; 32]) -> Result<Option<AuthorizationCodeRecord>, StoreError> {
        Ok(self.codes.borrow_mut().remove(h))
    }
    fn record_redeemed(&self, h: &[u8; 32], f: &str, until: u64) -> Result<(), StoreError> {
        self.redeemed
            .borrow_mut()
            .insert(*h, (f.to_string(), until));
        Ok(())
    }
    fn redeemed_family(&self, h: &[u8; 32], now: u64) -> Result<Option<String>, StoreError> {
        Ok(self
            .redeemed
            .borrow()
            .get(h)
            .filter(|(_, until)| now < *until)
            .map(|(f, _)| f.clone()))
    }
}

impl RefreshStore for MemStore {
    fn insert_refresh(&self, r: &RefreshTokenRecord) -> Result<(), StoreError> {
        self.refresh.borrow_mut().insert(r.token_hash, r.clone());
        Ok(())
    }
    fn get_refresh(&self, h: &[u8; 32]) -> Result<Option<RefreshTokenRecord>, StoreError> {
        Ok(self.refresh.borrow().get(h).cloned())
    }
    fn mark_rotated(&self, h: &[u8; 32]) -> Result<bool, StoreError> {
        match self.refresh.borrow_mut().get_mut(h) {
            Some(r) if !r.rotated => {
                r.rotated = true;
                Ok(true)
            }
            _ => Ok(false),
        }
    }
    fn revoke_family(&self, f: &str) -> Result<(), StoreError> {
        self.revoked.borrow_mut().insert(f.to_string());
        Ok(())
    }
    fn is_family_revoked(&self, f: &str) -> Result<bool, StoreError> {
        Ok(self.revoked.borrow().contains(f))
    }
}

impl FederationStore for MemStore {
    fn insert_flow(&self, s: &UpstreamFlowState) -> Result<(), StoreError> {
        self.flows.borrow_mut().insert(s.state.clone(), s.clone());
        Ok(())
    }
    fn take_flow(&self, state: &str) -> Result<Option<UpstreamFlowState>, StoreError> {
        Ok(self.flows.borrow_mut().remove(state))
    }
}

/// Real ES256 signer over a fixed test key.
pub struct TestSigner(pub SigningKey);
impl Default for TestSigner {
    fn default() -> Self {
        TestSigner(SigningKey::from_bytes(&[7u8; 32].into()).expect("valid scalar"))
    }
}
impl TestSigner {
    pub fn verifying_key(&self) -> VerifyingKey {
        *self.0.verifying_key()
    }
}
impl Signer for TestSigner {
    fn kid(&self) -> String {
        "test-kid".into()
    }
    fn sign_es256(&self, input: &[u8]) -> Result<[u8; 64], StoreError> {
        let sig: Signature = self.0.sign(input);
        Ok(sig.to_bytes().into())
    }
}

pub fn resource() -> ResourceUrl {
    ResourceUrl::parse(RESOURCE).unwrap()
}

/// `RESOURCE` (`/v1/mcp`, no admin), `OTHER_RESOURCE` (`/v1`, with admin)
/// and an adapter resource drawing on the same §5.3 vocabulary (ADR-351
/// §5.3, §16.1: no `team:*` scopes, no admin).
pub const ALLOWLIST_CONFIG: &str = "\
    https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1/mcp \
        ruvector:read ruvector:write offline_access, \
    https://ruvector-edge-gateway.cognitum-consulting-mail.workers.dev/v1 \
        ruvector:read ruvector:write ruvector:admin offline_access, \
    https://team.ruv.io/mcp ruvector:read ruvector:write offline_access";

/// The adapter resource in [`ALLOWLIST_CONFIG`].
pub const TEAM_RESOURCE: &str = "https://team.ruv.io/mcp";

pub fn allowlist() -> ResourceAllowlist {
    ResourceAllowlist::from_config(ALLOWLIST_CONFIG).unwrap()
}

/// Production-shaped DCR policy: registrable = the allowlist's scope union,
/// default ceiling = [`crate::client::DEFAULT_CLIENT_SCOPE`].
pub fn policy() -> DcrPolicy {
    DcrPolicy {
        scopes_supported: allowlist().scopes_supported(),
        default_scope: crate::client::DEFAULT_CLIENT_SCOPE
            .iter()
            .map(|s| s.to_string())
            .collect(),
    }
}

pub fn reg_request() -> RegistrationRequest {
    RegistrationRequest {
        redirect_uris: vec![REDIRECT.into()],
        grant_types: Some(vec!["authorization_code".into(), "refresh_token".into()]),
        client_name: Some("Claude".into()),
        ..Default::default()
    }
}

pub fn client(grants: &[&str]) -> ClientRecord {
    ClientRecord {
        client_id: CLIENT_ID.into(),
        redirect_uris: vec![
            crate::client::validate_redirect_uri(REDIRECT).unwrap(),
            crate::client::validate_redirect_uri(LOOPBACK).unwrap(),
        ],
        grant_types: grants.iter().map(|g| g.to_string()).collect(),
        scope: vec![
            "ruvector:read".into(),
            "ruvector:write".into(),
            "offline_access".into(),
        ],
        client_name: Some("Claude".into()),
        client_id_issued_at: T0,
    }
}

pub fn challenge() -> String {
    crate::pkce::challenge_s256(VERIFIER)
}

pub fn validated() -> ValidatedAuthorization {
    ValidatedAuthorization {
        client_id: CLIENT_ID.into(),
        redirect_uri: REDIRECT.into(),
        scopes: vec!["ruvector:read".into(), "offline_access".into()],
        state: Some("client-state".into()),
        code_challenge: challenge(),
        resource: resource(),
    }
}

pub fn identity() -> UpstreamIdentity {
    UpstreamIdentity {
        upstream_iss: "https://auth.cognitum.one".into(),
        sub: "8d0c7a52-user".into(),
        org_id: "org_123".into(),
        workspace_id: "ws-456".into(),
    }
}

pub fn upstream() -> UpstreamConfig {
    UpstreamConfig {
        issuer: "https://auth.cognitum.one".into(),
        authorization_endpoint: "https://auth.cognitum.one/oauth/authorize".into(),
        token_endpoint: "https://auth.cognitum.one/oauth/token".into(),
        jwks_url: "https://auth.cognitum.one/.well-known/jwks.json".into(),
        client_id: "dcr-edge-as".into(),
        redirect_uri: format!("{ISSUER}/callback"),
        scopes: vec!["openid".into(), "profile".into()],
    }
}

pub fn pairs(kv: &[(&str, &str)]) -> Vec<(String, String)> {
    kv.iter()
        .map(|(k, v)| (k.to_string(), v.to_string()))
        .collect()
}
