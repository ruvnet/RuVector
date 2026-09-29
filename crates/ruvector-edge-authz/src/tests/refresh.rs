//! Refresh rotation and family reuse detection.

use super::assert_code;
use crate::error::OAuthErrorCode as C;
use crate::ports::MockRefreshStore;
use crate::refresh::*;
use crate::testing::*;
use crate::StoreError;

struct Fx {
    store: MemStore,
    rng: SeqRng,
    clock: FixedClock,
    token: String,
}

fn scopes() -> Vec<String> {
    vec![
        "ruvector:read".into(),
        "ruvector:write".into(),
        "offline_access".into(),
    ]
}

fn fx() -> Fx {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let (token, rec) = issue_refresh(
        &store,
        &rng,
        &clock,
        "fam-1",
        CLIENT_ID,
        &identity(),
        &resource(),
        &scopes(),
    )
    .unwrap();
    assert_eq!(rec.family_id, "fam-1");
    Fx {
        store,
        rng,
        clock,
        token,
    }
}

fn rotate(f: &Fx, token: &str) -> Result<RotatedRefresh, crate::OAuthError> {
    rotate_refresh(&f.store, &f.rng, &f.clock, token, CLIENT_ID, None, None)
}

#[test]
fn issue_stores_hash_with_sliding_and_family_expiry() {
    let f = fx();
    let map = f.store.refresh.borrow();
    let rec = map.get(&crate::secret_hash(&f.token)).unwrap();
    assert_eq!(rec.expires_at, T0 + REFRESH_TTL_SECS);
    assert_eq!(rec.family_expires_at, T0 + FAMILY_MAX_LIFETIME_SECS);
    assert!(!rec.rotated);
    assert!(!serde_json::to_string(rec).unwrap().contains(&f.token));
}

#[test]
fn rotation_issues_new_token_in_same_family() {
    let f = fx();
    f.clock.advance(100);
    let r = rotate(&f, &f.token).unwrap();
    assert_ne!(r.token, f.token);
    assert_eq!(r.record.family_id, "fam-1");
    assert_eq!(r.record.scopes, scopes());
    assert_eq!(r.granted_scopes, scopes());
    assert_eq!(r.record.expires_at, T0 + 100 + REFRESH_TTL_SECS);
    assert_eq!(r.record.family_expires_at, T0 + FAMILY_MAX_LIFETIME_SECS);
    assert!(f.store.refresh.borrow()[&crate::secret_hash(&f.token)].rotated);
    // The new token rotates too.
    assert!(rotate(&f, &r.token).is_ok());
}

#[test]
fn reuse_of_rotated_token_revokes_whole_family() {
    let f = fx();
    let next = rotate(&f, &f.token).unwrap();
    assert_code(rotate(&f, &f.token), C::InvalidGrant);
    assert!(f.store.revoked.borrow().contains("fam-1"));
    // The legitimately rotated descendant is dead too.
    assert_code(rotate(&f, &next.token), C::InvalidGrant);
}

#[test]
fn lost_cas_race_revokes_family() {
    let rec = {
        let f = fx();
        let map = f.store.refresh.borrow();
        map.values().next().unwrap().clone()
    };
    let mut store = MockRefreshStore::new();
    store
        .expect_get_refresh()
        .returning(move |_| Ok(Some(rec.clone())));
    store.expect_is_family_revoked().returning(|_| Ok(false));
    store
        .expect_mark_rotated()
        .times(1)
        .returning(|_| Ok(false));
    store
        .expect_revoke_family()
        .withf(|f| f == "fam-1")
        .times(1)
        .returning(|_| Ok(()));
    store.expect_insert_refresh().times(0);
    assert_code(
        rotate_refresh(
            &store,
            &SeqRng::default(),
            &FixedClock::at(T0),
            "t",
            CLIENT_ID,
            None,
            None,
        ),
        C::InvalidGrant,
    );
}

#[test]
fn other_client_cannot_rotate_and_does_not_burn_token() {
    let f = fx();
    assert_code(
        rotate_refresh(
            &f.store,
            &f.rng,
            &f.clock,
            &f.token,
            "edc-other",
            None,
            None,
        ),
        C::InvalidGrant,
    );
    assert!(f.store.revoked.borrow().is_empty());
    assert!(rotate(&f, &f.token).is_ok());
}

#[test]
fn unknown_and_revoked_tokens_fail() {
    let f = fx();
    assert_code(rotate(&f, "unknown"), C::InvalidGrant);
    f.store.revoked.borrow_mut().insert("fam-1".into());
    assert_code(rotate(&f, &f.token), C::InvalidGrant);
}

#[test]
fn token_expiry_and_family_cap() {
    let f = fx();
    f.clock.advance(REFRESH_TTL_SECS);
    assert_code(rotate(&f, &f.token), C::InvalidGrant);

    // Keep rotating inside the sliding window: the family cap still ends it,
    // and a new token's expiry never exceeds the cap.
    let f = fx();
    let mut tok = f.token.clone();
    let step = REFRESH_TTL_SECS - 1;
    while f.clock.0.get() + step < T0 + FAMILY_MAX_LIFETIME_SECS {
        f.clock.advance(step);
        let r = rotate(&f, &tok).unwrap();
        assert!(r.record.expires_at <= T0 + FAMILY_MAX_LIFETIME_SECS);
        tok = r.token;
    }
    f.clock.0.set(T0 + FAMILY_MAX_LIFETIME_SECS);
    assert_code(rotate(&f, &tok), C::InvalidGrant);
}

#[test]
fn downscoping_narrows_access_not_family() {
    let f = fx();
    let r = rotate_refresh(
        &f.store,
        &f.rng,
        &f.clock,
        &f.token,
        CLIENT_ID,
        Some("ruvector:read"),
        None,
    )
    .unwrap();
    assert_eq!(r.granted_scopes, vec!["ruvector:read"]);
    assert_eq!(r.record.scopes, scopes());
}

#[test]
fn request_errors_do_not_consume_token() {
    let f = fx();
    let bad: Vec<(Option<&str>, Option<&str>, C)> = vec![
        (Some("ruvector:read brains:read"), None, C::InvalidScope),
        (Some("ruvector:read  x"), None, C::InvalidScope),
        (None, Some(OTHER_RESOURCE), C::InvalidTarget),
        (None, Some("https://evil.example"), C::InvalidTarget),
    ];
    for (scope, res, code) in bad {
        assert_code(
            rotate_refresh(&f.store, &f.rng, &f.clock, &f.token, CLIENT_ID, scope, res),
            code,
        );
    }
    assert!(f.store.revoked.borrow().is_empty());
    assert!(rotate_refresh(
        &f.store,
        &f.rng,
        &f.clock,
        &f.token,
        CLIENT_ID,
        None,
        Some(RESOURCE)
    )
    .is_ok());
}

#[test]
fn store_failure_is_server_error() {
    let mut store = MockRefreshStore::new();
    store
        .expect_get_refresh()
        .returning(|_| Err(StoreError("sqlite".into())));
    assert_code(
        rotate_refresh(
            &store,
            &SeqRng::default(),
            &FixedClock::at(T0),
            "t",
            CLIENT_ID,
            None,
            None,
        ),
        C::ServerError,
    );
}

/// Regression (offline_access, ADR-351 §5.3): a family whose ceiling lacks
/// `offline_access` cannot rotate, and the failed attempt consumes nothing.
#[test]
fn family_without_offline_access_cannot_rotate() {
    let (store, rng, clock) = (MemStore::default(), SeqRng::default(), FixedClock::at(T0));
    let (token, _) = issue_refresh(
        &store,
        &rng,
        &clock,
        "fam-2",
        CLIENT_ID,
        &identity(),
        &resource(),
        &["ruvector:read".to_string()],
    )
    .unwrap();
    assert_code(
        rotate_refresh(&store, &rng, &clock, &token, CLIENT_ID, None, None),
        C::InvalidGrant,
    );
    assert!(!store.refresh.borrow()[&crate::secret_hash(&token)].rotated);
}

/// Preparing a rotation writes nothing; only commit consumes the token.
#[test]
fn prepare_is_side_effect_free_until_commit() {
    let f = fx();
    let p = prepare_rotation(&f.store, &f.rng, &f.clock, &f.token, CLIENT_ID, None, None).unwrap();
    assert_eq!(f.store.refresh.borrow().len(), 1);
    assert!(!f.store.refresh.borrow()[&crate::secret_hash(&f.token)].rotated);
    assert!(!format!("{p:?}").contains(&f.token));
    let r = p.commit(&f.store).unwrap();
    assert!(f.store.refresh.borrow()[&crate::secret_hash(&f.token)].rotated);
    assert!(!format!("{r:?}").contains(&r.token));
}
