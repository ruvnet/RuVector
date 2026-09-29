//! Authorization codes: issuance, single use, TTL, bindings.

use super::assert_code;
use crate::code::*;
use crate::error::OAuthErrorCode as C;
use crate::ports::{MockCodeStore, MockRng};
use crate::testing::*;
use crate::StoreError;

struct Fx {
    store: MemStore,
    clock: FixedClock,
    code: String,
}

fn fx() -> Fx {
    let (store, clock) = (MemStore::default(), FixedClock::at(T0));
    let code = issue_code(&store, &SeqRng::default(), &clock, validated(), identity()).unwrap();
    Fx { store, clock, code }
}

fn redeem(
    f: &Fx,
    client: &str,
    redirect: &str,
    verifier: &str,
    res: Option<&str>,
) -> Result<AuthorizationCodeRecord, crate::OAuthError> {
    redeem_code(&f.store, &f.clock, &f.code, client, redirect, verifier, res)
}

fn good(f: &Fx) -> Result<AuthorizationCodeRecord, crate::OAuthError> {
    redeem(f, CLIENT_ID, REDIRECT, VERIFIER, None)
}

#[test]
fn issue_stores_only_the_hash_with_60s_ttl() {
    let f = fx();
    assert_eq!(f.code.len(), 43);
    let codes = f.store.codes.borrow();
    let rec = codes
        .get(&crate::secret_hash(&f.code))
        .expect("stored by hash");
    assert_eq!(rec.expires_at, T0 + CODE_TTL_SECS);
    assert_eq!(rec.identity, identity());
    assert!(!serde_json::to_string(rec).unwrap().contains(&f.code));
}

#[test]
fn redeem_happy_path_binds_everything() {
    let f = fx();
    let rec = redeem(&f, CLIENT_ID, REDIRECT, VERIFIER, Some(RESOURCE)).unwrap();
    assert_eq!(rec.authorization, validated());
    assert_eq!(rec.identity.sub, identity().sub);
}

#[test]
fn replay_is_rejected() {
    let f = fx();
    good(&f).unwrap();
    assert_code(good(&f), C::InvalidGrant);
}

#[test]
fn unknown_code_is_invalid_grant() {
    let f = fx();
    assert_code(
        redeem_code(
            &f.store, &f.clock, "nope", CLIENT_ID, REDIRECT, VERIFIER, None,
        ),
        C::InvalidGrant,
    );
    // The real code is untouched by a miss.
    assert!(good(&f).is_ok());
}

#[test]
fn ttl_boundary() {
    let f = fx();
    f.clock.advance(CODE_TTL_SECS - 1);
    assert!(good(&f).is_ok());
    let f = fx();
    f.clock.advance(CODE_TTL_SECS);
    assert_code(good(&f), C::InvalidGrant);
    assert_code(good(&f), C::InvalidGrant);
}

#[test]
fn every_binding_failure_consumes_the_code() {
    let other_verifier = "x".repeat(43);
    let cases: Vec<(&str, &str, &str, Option<&str>, C)> = vec![
        ("edc-other", REDIRECT, VERIFIER, None, C::InvalidGrant),
        (
            CLIENT_ID,
            "https://claude.ai/other",
            VERIFIER,
            None,
            C::InvalidGrant,
        ),
        // Loopback port flexibility applies at /authorize, not here: the
        // token request must repeat the exact redirect_uri.
        (CLIENT_ID, LOOPBACK, VERIFIER, None, C::InvalidGrant),
        (CLIENT_ID, REDIRECT, &other_verifier, None, C::InvalidGrant),
        (CLIENT_ID, REDIRECT, "short", None, C::InvalidGrant),
        (
            CLIENT_ID,
            REDIRECT,
            VERIFIER,
            Some(OTHER_RESOURCE),
            C::InvalidTarget,
        ),
        (
            CLIENT_ID,
            REDIRECT,
            VERIFIER,
            Some("https://evil.example"),
            C::InvalidTarget,
        ),
        (
            CLIENT_ID,
            REDIRECT,
            VERIFIER,
            Some("not a url"),
            C::InvalidTarget,
        ),
    ];
    for (client, redirect, verifier, res, code) in cases {
        let f = fx();
        assert_code(redeem(&f, client, redirect, verifier, res), code);
        assert_code(good(&f), C::InvalidGrant);
    }
}

#[test]
fn rng_failure_stores_nothing() {
    let mut rng = MockRng::new();
    rng.expect_fill()
        .returning(|_| Err(StoreError("rng".into())));
    let mut store = MockCodeStore::new();
    store.expect_insert_code().times(0);
    assert_code(
        issue_code(&store, &rng, &FixedClock::at(T0), validated(), identity()),
        C::ServerError,
    );
}

#[test]
fn store_failures_are_server_errors() {
    let mut store = MockCodeStore::new();
    store
        .expect_insert_code()
        .returning(|_| Err(StoreError("sqlite".into())));
    store
        .expect_take_code()
        .returning(|_| Err(StoreError("sqlite".into())));
    let clock = FixedClock::at(T0);
    assert_code(
        issue_code(&store, &SeqRng::default(), &clock, validated(), identity()),
        C::ServerError,
    );
    assert_code(
        redeem_code(&store, &clock, "c", CLIENT_ID, REDIRECT, VERIFIER, None),
        C::ServerError,
    );
}
