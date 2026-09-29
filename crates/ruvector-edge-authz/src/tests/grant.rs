//! End-to-end token endpoint over the in-memory ports: DCR -> /authorize ->
//! upstream federation -> /callback -> code -> /token -> refresh.

use super::assert_code;
use crate::authorize::{validate_authorization, AuthorizationRequest};
use crate::client::register_client;
use crate::error::OAuthErrorCode as C;
use crate::federation::{begin_upstream, complete_upstream, identity_from_upstream};
use crate::grant::TokenEndpoint;
use crate::testing::*;
use crate::token::{TokenRequest, TokenResponse};
use crate::ResourceAllowlist;
use ruvector_edge_auth::jws::b64url_decode;
use ruvector_edge_auth::{TokenKind, VerifiedClaims};

const FULL_SCOPE: &str = "ruvector:read ruvector:write offline_access";

struct World {
    store: MemStore,
    rng: SeqRng,
    clock: FixedClock,
    signer: TestSigner,
    resources: ResourceAllowlist,
}

impl World {
    fn new() -> Self {
        World {
            store: MemStore::default(),
            rng: SeqRng::default(),
            clock: FixedClock::at(T0),
            signer: TestSigner::default(),
            resources: allowlist(),
        }
    }

    fn endpoint(&self) -> TokenEndpoint<'_> {
        TokenEndpoint {
            issuer: ISSUER,
            resources: &self.resources,
            clients: &self.store,
            codes: &self.store,
            refresh: &self.store,
            signer: &self.signer,
            rng: &self.rng,
            clock: &self.clock,
        }
    }

    /// Register a client with `grants` and run the browser leg consenting to
    /// `ruvector:read ruvector:write offline_access`; returns `(client_id, code)`.
    fn login(&self, grants: &[&str]) -> (String, String) {
        self.login_with(grants, FULL_SCOPE)
    }

    /// Like [`World::login`] with an explicit consented `scope`.
    fn login_with(&self, grants: &[&str], scope: &str) -> (String, String) {
        let mut req = reg_request();
        req.grant_types = Some(grants.iter().map(|g| g.to_string()).collect());
        req.scope = Some(FULL_SCOPE.into());
        let client = register_client(&self.store, &self.rng, &self.clock, &policy(), &req).unwrap();
        let areq = AuthorizationRequest {
            response_type: Some("code".into()),
            client_id: Some(client.client_id.clone()),
            redirect_uri: Some(REDIRECT.into()),
            scope: Some(scope.into()),
            state: Some("st".into()),
            code_challenge: Some(challenge()),
            code_challenge_method: Some("S256".into()),
            resource: Some(RESOURCE.into()),
        };
        let v = validate_authorization(&areq, &client, &self.resources).unwrap();
        let start = begin_upstream(&self.store, &self.rng, &self.clock, &upstream(), v).unwrap();
        let q: Vec<(String, String)> = url::Url::parse(&start.authorization_url)
            .unwrap()
            .query_pairs()
            .into_owned()
            .collect();
        let state = &q.iter().find(|(k, _)| k == "state").unwrap().1;
        let flow =
            complete_upstream(&self.store, &self.clock, state, &start.browser_secret).unwrap();
        let up = upstream();
        let claims = VerifiedClaims::for_tests(
            TokenKind::UpstreamFirstParty,
            &up.issuer,
            &up.client_id,
            "user-42",
            &up.client_id,
            Some("org_1"),
            Some("ws_1"),
            &["openid"],
            T0,
            T0 + 900,
        );
        let id = identity_from_upstream(&claims, &flow, &up).unwrap();
        let code =
            crate::code::issue_code(&self.store, &self.rng, &self.clock, flow.downstream, id)
                .unwrap();
        (client.client_id, code)
    }

    fn redeem(&self, client_id: &str, code: &str) -> Result<TokenResponse, crate::OAuthError> {
        self.endpoint().handle(&TokenRequest::AuthorizationCode {
            code: code.into(),
            redirect_uri: REDIRECT.into(),
            client_id: client_id.into(),
            code_verifier: VERIFIER.into(),
            resource: Some(RESOURCE.into()),
        })
    }

    fn refresh(
        &self,
        client_id: &str,
        rt: &str,
        scope: Option<&str>,
    ) -> Result<TokenResponse, crate::OAuthError> {
        self.endpoint().handle(&TokenRequest::RefreshToken {
            refresh_token: rt.into(),
            client_id: client_id.into(),
            scope: scope.map(Into::into),
            resource: None,
        })
    }
}

fn payload(jwt: &str) -> serde_json::Value {
    let p = jwt.split('.').nth(1).unwrap();
    serde_json::from_slice(&b64url_decode(p).unwrap()).unwrap()
}

#[test]
fn full_flow_mints_resource_bound_token_and_refresh() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let t = w.redeem(&cid, &code).unwrap();
    assert_eq!(t.token_type, "Bearer");
    assert_eq!(t.expires_in, 900);
    assert_eq!(t.scope, FULL_SCOPE);
    let rt = t.refresh_token.clone().expect("refresh issued");
    let p = payload(&t.access_token);
    assert_eq!(p["aud"], RESOURCE);
    assert_eq!(p["iss"], ISSUER);
    let edge_sub =
        ruvector_edge_auth::subject::edge_subject("https://auth.cognitum.one", "user-42");
    assert_eq!(p["sub"], edge_sub.as_str());
    assert_eq!(p["upstream_iss"], "https://auth.cognitum.one");
    assert_eq!(p["client_id"], cid.as_str());
    assert_eq!(
        (p["org_id"].as_str(), p["workspace_id"].as_str()),
        (Some("org_1"), Some("ws_1"))
    );
    let fam = w.store.refresh.borrow()[&crate::secret_hash(&rt)]
        .family_id
        .clone();
    assert_eq!(p["family_id"], fam.as_str());
    // Refresh with down-scoping; family id carries over.
    w.clock.advance(60);
    let t2 = w.refresh(&cid, &rt, Some("ruvector:read")).unwrap();
    let p2 = payload(&t2.access_token);
    assert_eq!(p2["scope"], "ruvector:read");
    assert_eq!(p2["family_id"], fam.as_str());
    assert_ne!(p2["jti"], p["jti"]);
    assert_ne!(t2.refresh_token.as_deref(), Some(rt.as_str()));
    // Reusing the first refresh token kills the family.
    assert_code(w.refresh(&cid, &rt, None), C::InvalidGrant);
    assert_code(
        w.refresh(&cid, t2.refresh_token.as_deref().unwrap(), None),
        C::InvalidGrant,
    );
}

/// Regression (code replay, RFC 6749 §4.1.2): replaying a redeemed code
/// revokes the grant it started, so the first redemption's refresh token
/// stops working.
#[test]
fn code_replay_revokes_the_grant_it_started() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let t = w.redeem(&cid, &code).unwrap();
    let rt = t.refresh_token.expect("refresh issued");
    let fam = payload(&t.access_token)["family_id"]
        .as_str()
        .unwrap()
        .to_string();
    assert_code(w.redeem(&cid, &code), C::InvalidGrant);
    assert!(w.store.revoked.borrow().contains(&fam));
    assert_code(w.refresh(&cid, &rt, None), C::InvalidGrant);
}

/// A never-issued code does not revoke anything.
#[test]
fn unknown_code_revokes_nothing() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let rt = w.redeem(&cid, &code).unwrap().refresh_token.unwrap();
    assert_code(w.redeem(&cid, "not-a-code"), C::InvalidGrant);
    assert!(w.store.revoked.borrow().is_empty());
    assert!(w.refresh(&cid, &rt, None).is_ok());
}

/// Regression (offline_access, ADR-351 §5.3): a client with the
/// refresh_token grant whose user consented without `offline_access` gets
/// no refresh token.
#[test]
fn no_refresh_token_without_offline_access() {
    let w = World::new();
    let (cid, code) = w.login_with(&["authorization_code", "refresh_token"], "ruvector:read");
    let t = w.redeem(&cid, &code).unwrap();
    assert!(t.refresh_token.is_none());
    assert_eq!(t.scope, "ruvector:read");
    assert!(w.store.refresh.borrow().is_empty());
}

/// Narrowing `scope` on refresh to drop `offline_access` narrows only the
/// access token; the family ceiling keeps it, so rotation continues.
#[test]
fn narrowing_away_offline_access_keeps_family_ceiling() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let rt = w.redeem(&cid, &code).unwrap().refresh_token.unwrap();
    let t2 = w.refresh(&cid, &rt, Some("ruvector:read")).unwrap();
    assert_eq!(t2.scope, "ruvector:read");
    let rt2 = t2.refresh_token.expect("rotated");
    let rec = w.store.refresh.borrow()[&crate::secret_hash(&rt2)].clone();
    assert!(rec.scopes.iter().any(|s| s == "offline_access"));
    assert!(w.refresh(&cid, &rt2, None).is_ok());
}

/// Regression (rotate-before-mint): a signing failure during refresh must
/// not consume the old token, so the client's retry is not reuse.
#[test]
fn signing_failure_does_not_burn_refresh_token() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let rt = w.redeem(&cid, &code).unwrap().refresh_token.unwrap();
    let mut broken = crate::ports::MockSigner::new();
    broken.expect_kid().returning(String::new);
    let mut ep = w.endpoint();
    ep.signer = &broken;
    let req = TokenRequest::RefreshToken {
        refresh_token: rt.clone(),
        client_id: cid.clone(),
        scope: None,
        resource: None,
    };
    assert_code(ep.handle(&req), C::ServerError);
    assert!(!w.store.refresh.borrow()[&crate::secret_hash(&rt)].rotated);
    assert!(w.store.revoked.borrow().is_empty());
    assert!(w.refresh(&cid, &rt, None).is_ok());
}

#[test]
fn client_without_refresh_grant() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code"]);
    let t = w.redeem(&cid, &code).unwrap();
    assert!(t.refresh_token.is_none());
    assert!(w.store.refresh.borrow().is_empty());
    assert!(payload(&t.access_token)["family_id"]
        .as_str()
        .is_some_and(|f| f.len() == 22));
    assert_code(w.refresh(&cid, "anything", None), C::UnauthorizedClient);
}

#[test]
fn unknown_client_is_invalid_client_and_code_survives() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code"]);
    assert_code(w.redeem("edc-nobody", &code), C::InvalidClient);
    assert!(w.redeem(&cid, &code).is_ok());
}

#[test]
fn another_registered_client_cannot_redeem() {
    let w = World::new();
    let (cid, code) = w.login(&["authorization_code"]);
    let (other, _) = w.login(&["authorization_code"]);
    assert_code(w.redeem(&other, &code), C::InvalidGrant);
    assert_code(w.redeem(&cid, &code), C::InvalidGrant);
}

#[test]
fn resource_removed_from_allowlist_blocks_issuance() {
    let mut w = World::new();
    let (cid, code) = w.login(&["authorization_code", "refresh_token"]);
    let rt = w.redeem(&cid, &code).unwrap().refresh_token.unwrap();
    w.resources = ResourceAllowlist::new(vec![]);
    assert_code(w.refresh(&cid, &rt, None), C::InvalidTarget);
}

#[test]
fn resource_removed_before_code_redemption() {
    let mut w = World::new();
    let (cid, code) = w.login(&["authorization_code"]);
    w.resources = ResourceAllowlist::new(vec![]);
    assert_code(w.redeem(&cid, &code), C::InvalidTarget);
}
