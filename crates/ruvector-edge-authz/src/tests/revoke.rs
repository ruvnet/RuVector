//! RFC 7009 revocation.

use super::assert_code;
use crate::error::OAuthErrorCode as C;
use crate::ports::MockRefreshStore;
use crate::refresh::{issue_refresh, rotate_refresh};
use crate::revoke::*;
use crate::testing::*;
use crate::StoreError;

fn setup() -> (MemStore, SeqRng, FixedClock, String) {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let (tok, _) = issue_refresh(
        &store,
        &rng,
        &clock,
        "fam-9",
        CLIENT_ID,
        &identity(),
        &resource(),
        &["ruvector:read".to_string(), "offline_access".to_string()],
    )
    .unwrap();
    (store, rng, clock, tok)
}

#[test]
fn revoking_own_refresh_token_kills_family() {
    let (store, rng, clock, tok) = setup();
    revoke(&store, &tok, CLIENT_ID, Some("refresh_token")).unwrap();
    assert!(store.revoked.borrow().contains("fam-9"));
    assert_code(
        rotate_refresh(&store, &rng, &clock, &tok, CLIENT_ID, None, None),
        C::InvalidGrant,
    );
}

#[test]
fn silent_success_cases_do_not_revoke() {
    let (store, _, _, tok) = setup();
    revoke(&store, &tok, "edc-other", None).unwrap();
    revoke(&store, "unknown", CLIENT_ID, None).unwrap();
    revoke(
        &store,
        "eyJhbGciOiJFUzI1NiJ9.e30.sig",
        CLIENT_ID,
        Some("access_token"),
    )
    .unwrap();
    revoke(&store, &tok, CLIENT_ID, Some("bogus_hint")).unwrap();
    // Only the final (own-client) call revoked, regardless of the hint.
    assert_eq!(store.revoked.borrow().len(), 1);
}

#[test]
fn other_client_revocation_leaves_token_usable() {
    let (store, rng, clock, tok) = setup();
    revoke(&store, &tok, "edc-other", None).unwrap();
    assert!(store.revoked.borrow().is_empty());
    assert!(rotate_refresh(&store, &rng, &clock, &tok, CLIENT_ID, None, None).is_ok());
}

#[test]
fn storage_failure_is_error() {
    let mut store = MockRefreshStore::new();
    store
        .expect_get_refresh()
        .returning(|_| Err(StoreError("sqlite".into())));
    assert_code(revoke(&store, "t", CLIENT_ID, None), C::ServerError);
}

#[test]
fn request_parsing() {
    let r = RevocationRequest::from_form(&pairs(&[
        ("token", "t"),
        ("client_id", CLIENT_ID),
        ("token_type_hint", "refresh_token"),
    ]))
    .unwrap();
    assert_eq!(r.token_type_hint.as_deref(), Some("refresh_token"));
    assert_code(
        RevocationRequest::from_form(&pairs(&[("client_id", CLIENT_ID)])),
        C::InvalidRequest,
    );
    assert_code(
        RevocationRequest::from_form(&pairs(&[("token", "t")])),
        C::InvalidRequest,
    );
    assert_code(
        RevocationRequest::from_form(&pairs(&[
            ("token", "t"),
            ("client_id", CLIENT_ID),
            ("client_secret", "s"),
        ])),
        C::InvalidClient,
    );
    assert_code(
        RevocationRequest::from_form(&pairs(&[
            ("token", "t"),
            ("token", "u"),
            ("client_id", "c"),
        ])),
        C::InvalidRequest,
    );
}
