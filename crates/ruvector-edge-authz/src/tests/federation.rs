//! Upstream federation: flow binding, one-time state, identity extraction.

use super::assert_code;
use crate::error::OAuthErrorCode as C;
use crate::federation::*;
use crate::ports::{MockFederationStore, MockRng};
use crate::testing::*;
use crate::StoreError;
use ruvector_edge_auth::{TokenKind, VerifiedClaims};

fn begin(store: &MemStore, rng: &SeqRng, clock: &FixedClock) -> UpstreamStart {
    begin_upstream(store, rng, clock, &upstream(), validated()).unwrap()
}

fn query(url: &str) -> Vec<(String, String)> {
    url::Url::parse(url)
        .unwrap()
        .query_pairs()
        .into_owned()
        .collect()
}

#[test]
fn begin_builds_upstream_url_and_stores_bound_flow() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    assert!(start
        .authorization_url
        .starts_with("https://auth.cognitum.one/oauth/authorize?"));
    let q = query(&start.authorization_url);
    let get = |k: &str| {
        q.iter()
            .find(|(n, _)| n == k)
            .map(|(_, v)| v.clone())
            .unwrap()
    };
    assert_eq!(get("response_type"), "code");
    assert_eq!(get("client_id"), "dcr-edge-as");
    assert_eq!(get("redirect_uri"), format!("{ISSUER}/callback"));
    assert_eq!(get("scope"), "openid profile");
    assert_eq!(get("code_challenge_method"), "S256");
    let flows = store.flows.borrow();
    let flow = flows.get(&get("state")).expect("flow stored under state");
    assert_eq!(flow.nonce, get("nonce"));
    assert_eq!(
        crate::pkce::challenge_s256(&flow.upstream_code_verifier),
        get("code_challenge")
    );
    assert!(crate::pkce::validate_verifier(&flow.upstream_code_verifier).is_ok());
    assert_eq!(
        flow.browser_binding,
        crate::secret_hash(&start.browser_secret)
    );
    assert_eq!(flow.expires_at, T0 + FLOW_TTL_SECS);
    assert_eq!(flow.downstream, validated());
    let distinct = [
        &flow.state,
        &flow.nonce,
        &flow.upstream_code_verifier,
        &start.browser_secret,
    ];
    for (i, a) in distinct.iter().enumerate() {
        for b in &distinct[i + 1..] {
            assert_ne!(a, b);
        }
    }
    // The browser secret itself is never in the URL or the store.
    assert!(!start.authorization_url.contains(&start.browser_secret));
}

#[test]
fn begin_fails_closed_without_upstream_client_id() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let mut cfg = upstream();
    cfg.client_id.clear();
    assert_code(
        begin_upstream(&store, &rng, &clock, &cfg, validated()),
        C::TemporarilyUnavailable,
    );
    assert!(store.flows.borrow().is_empty());
}

#[test]
fn begin_rejects_invalid_config() {
    let muts: Vec<fn(&mut UpstreamConfig)> = vec![
        |c| c.authorization_endpoint = "http://auth.cognitum.one/oauth/authorize".into(),
        |c| c.token_endpoint = "not a url".into(),
        |c| c.redirect_uri = "http://127.0.0.1/callback".into(),
        |c| c.scopes.clear(),
    ];
    for m in muts {
        let mut cfg = upstream();
        m(&mut cfg);
        let store = MemStore::default();
        assert_code(
            begin_upstream(
                &store,
                &SeqRng::default(),
                &FixedClock::at(T0),
                &cfg,
                validated(),
            ),
            C::ServerError,
        );
        assert!(store.flows.borrow().is_empty());
    }
}

#[test]
fn begin_rng_failure_stores_nothing() {
    let mut rng = MockRng::new();
    rng.expect_fill()
        .returning(|_| Err(StoreError("rng".into())));
    let mut store = MockFederationStore::new();
    store.expect_insert_flow().times(0);
    assert_code(
        begin_upstream(&store, &rng, &FixedClock::at(T0), &upstream(), validated()),
        C::ServerError,
    );
}

#[test]
fn complete_is_one_time_and_bound_to_browser() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    let state = store.flows.borrow().keys().next().unwrap().clone();
    let flow = complete_upstream(&store, &clock, &state, &start.browser_secret).unwrap();
    assert_eq!(flow.downstream, validated());
    // Replay of the same state.
    assert_code(
        complete_upstream(&store, &clock, &state, &start.browser_secret),
        C::AccessDenied,
    );
}

#[test]
fn complete_wrong_browser_denies_and_burns_state() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    let state = store.flows.borrow().keys().next().unwrap().clone();
    assert_code(
        complete_upstream(&store, &clock, &state, "attacker-cookie"),
        C::AccessDenied,
    );
    assert_code(
        complete_upstream(&store, &clock, &state, &start.browser_secret),
        C::AccessDenied,
    );
}

#[test]
fn complete_expired_denies_and_burns_state() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    let state = store.flows.borrow().keys().next().unwrap().clone();
    clock.advance(FLOW_TTL_SECS);
    assert_code(
        complete_upstream(&store, &clock, &state, &start.browser_secret),
        C::AccessDenied,
    );
    assert!(store.flows.borrow().is_empty());
}

#[test]
fn complete_just_before_expiry_succeeds() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    let state = store.flows.borrow().keys().next().unwrap().clone();
    clock.advance(FLOW_TTL_SECS - 1);
    assert!(complete_upstream(&store, &clock, &state, &start.browser_secret).is_ok());
}

#[test]
fn complete_rejects_malformed_inputs_without_touching_store() {
    let mut store = MockFederationStore::new();
    store.expect_take_flow().times(0);
    let clock = FixedClock::at(T0);
    let long = "s".repeat(MAX_FLOW_PARAM_LEN + 1);
    for (s, b) in [
        ("", "b"),
        ("s", ""),
        (long.as_str(), "b"),
        ("s", long.as_str()),
    ] {
        assert_code(complete_upstream(&store, &clock, s, b), C::AccessDenied);
    }
    assert_code(
        complete_upstream(&MemStore::default(), &clock, "unknown", "b"),
        C::AccessDenied,
    );
}

#[test]
fn complete_store_failure_is_server_error() {
    let mut store = MockFederationStore::new();
    store
        .expect_take_flow()
        .returning(|_| Err(StoreError("sqlite".into())));
    assert_code(
        complete_upstream(&store, &FixedClock::at(T0), "s", "b"),
        C::ServerError,
    );
}

#[test]
fn callback_parsing() {
    let ok = UpstreamCallback::from_pairs(&pairs(&[("state", "s"), ("code", "c")])).unwrap();
    assert_eq!(ok.outcome, Ok("c".to_string()));
    let err = UpstreamCallback::from_pairs(&pairs(&[("state", "s"), ("error", "access_denied")]))
        .unwrap();
    assert_eq!(err.outcome, Err("access_denied".to_string()));
    for bad in [
        pairs(&[("code", "c")]),
        pairs(&[("state", "s")]),
        pairs(&[("state", "s"), ("code", "c"), ("error", "e")]),
        pairs(&[("state", "s"), ("state", "t"), ("code", "c")]),
    ] {
        assert_code(UpstreamCallback::from_pairs(&bad), C::InvalidRequest);
    }
}

#[test]
fn upstream_token_form_uses_our_verifier() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let start = begin(&store, &rng, &clock);
    let state = store.flows.borrow().keys().next().unwrap().clone();
    let flow = complete_upstream(&store, &clock, &state, &start.browser_secret).unwrap();
    let form = upstream_token_form(&upstream(), &flow, "up-code");
    assert_eq!(
        form,
        pairs(&[
            ("grant_type", "authorization_code"),
            ("code", "up-code"),
            ("redirect_uri", &format!("{ISSUER}/callback")),
            ("client_id", "dcr-edge-as"),
            ("code_verifier", &flow.upstream_code_verifier),
        ])
    );
}

fn claims(
    kind: TokenKind,
    iss: &str,
    aud: &str,
    org: Option<&str>,
    ws: Option<&str>,
) -> VerifiedClaims {
    VerifiedClaims::for_tests(
        kind,
        iss,
        aud,
        "user-1",
        aud,
        org,
        ws,
        &["openid"],
        T0,
        T0 + 900,
    )
}

fn flow() -> UpstreamFlowState {
    UpstreamFlowState {
        state: "s".into(),
        nonce: "n".into(),
        upstream_code_verifier: VERIFIER.into(),
        browser_binding: [0; 32],
        downstream: validated(),
        expires_at: T0 + 600,
    }
}

#[test]
fn identity_from_valid_upstream_claims() {
    let up = TokenKind::UpstreamFirstParty;
    let c = claims(
        up,
        "https://auth.cognitum.one",
        "dcr-edge-as",
        Some("org_1"),
        Some("ws-1"),
    );
    let id = identity_from_upstream(&c, &flow(), &upstream()).unwrap();
    assert_eq!(
        (
            id.sub.as_str(),
            id.org_id.as_str(),
            id.workspace_id.as_str()
        ),
        ("user-1", "org_1", "ws-1")
    );
}

#[test]
fn identity_rejections() {
    let (up, iss, aud) = (
        TokenKind::UpstreamFirstParty,
        "https://auth.cognitum.one",
        "dcr-edge-as",
    );
    let long = "o".repeat(65);
    for c in [
        claims(TokenKind::EdgeIssued, iss, aud, Some("o"), Some("w")),
        claims(up, "https://evil.example", aud, Some("o"), Some("w")),
        claims(up, iss, "dcr-other", Some("o"), Some("w")),
        claims(up, iss, aud, None, Some("w")),
        claims(up, iss, aud, Some("o"), None),
        claims(up, iss, aud, Some("o/../x"), Some("w")),
        claims(up, iss, aud, Some("o"), Some("w w")),
        claims(up, iss, aud, Some(&long), Some("w")),
        claims(up, iss, aud, Some(""), Some("w")),
    ] {
        assert_code(
            identity_from_upstream(&c, &flow(), &upstream()),
            C::AccessDenied,
        );
    }
    let bad_sub = VerifiedClaims::for_tests(
        up,
        iss,
        aud,
        "a b",
        aud,
        Some("o"),
        Some("w"),
        &[],
        T0,
        T0 + 1,
    );
    assert_code(
        identity_from_upstream(&bad_sub, &flow(), &upstream()),
        C::AccessDenied,
    );
    let wrong_client = VerifiedClaims::for_tests(
        up,
        iss,
        aud,
        "u",
        "dcr-x",
        Some("o"),
        Some("w"),
        &[],
        T0,
        T0 + 1,
    );
    assert_code(
        identity_from_upstream(&wrong_client, &flow(), &upstream()),
        C::AccessDenied,
    );
}

#[test]
fn tenant_id_charset() {
    assert!(is_tenant_id("0c8e6a0e-1d2b-4c3a-9f00-1234567890ab"));
    assert!(is_tenant_id("A_z-9"));
    for bad in ["", "a.b", "a b", "ä", &"x".repeat(65)] {
        assert!(!is_tenant_id(bad), "{bad}");
    }
}
