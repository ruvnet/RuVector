use super::*;
use crate::test_support::*;
use serde_json::json;

fn policy() -> JwksCachePolicy {
    JwksCachePolicy::with_defaults(JWKS_URL)
}

// ---- Jwk / JwkSet -------------------------------------------------------

#[test]
fn jwk_round_trip_and_thumbprint() {
    let k = key(1);
    let jwk = Jwk::from_verifying_key(k.verifying_key());
    assert_eq!(jwk.kid, jwk.thumbprint());
    assert_eq!(jwk.kid.len(), 43);
    assert_eq!(jwk.to_verifying_key().unwrap(), *k.verifying_key());
    assert_ne!(kid(&key(1)), kid(&key(2)));
}

#[test]
fn rfc7638_thumbprint_vector() {
    // RFC 7638 §3.1 canonical-member form, checked against an independently
    // computed SHA-256 of the canonical JSON.
    use sha2::{Digest, Sha256};
    let jwk = Jwk::from_verifying_key(key(3).verifying_key());
    let canonical = format!(
        r#"{{"crv":"P-256","kty":"EC","x":"{}","y":"{}"}}"#,
        jwk.x, jwk.y
    );
    let want = crate::jws::b64url_encode(&Sha256::digest(canonical.as_bytes()));
    assert_eq!(jwk.thumbprint(), want);
}

#[test]
fn jwk_rejections() {
    let good = Jwk::from_verifying_key(key(1).verifying_key());
    let mut cases: Vec<Jwk> = Vec::new();
    let mut j = good.clone();
    j.kid = kid(&key(2)); // thumbprint mismatch
    cases.push(j);
    let mut j = good.clone();
    j.kty = "RSA".into();
    cases.push(j);
    let mut j = good.clone();
    j.crv = "P-384".into();
    cases.push(j);
    let mut j = good.clone();
    j.alg = Some("RS256".into());
    cases.push(j);
    let mut j = good.clone();
    j.use_ = Some("enc".into());
    cases.push(j);
    let mut j = good.clone();
    j.x = crate::jws::b64url_encode(&[1u8; 31]); // short coordinate
    cases.push(j);
    let mut j = good.clone();
    j.y = crate::jws::b64url_encode(&[1u8; 32]); // off curve
    j.kid = j.thumbprint();
    cases.push(j);
    let mut j = good.clone();
    j.x.push('='); // non-strict base64
    cases.push(j);
    for (i, j) in cases.iter().enumerate() {
        assert!(j.to_verifying_key().is_err(), "case {i}");
    }
    // alg/use absent is fine.
    let mut j = good;
    j.alg = None;
    j.use_ = None;
    assert!(j.to_verifying_key().is_ok());
}

#[test]
fn parse_usable_skips_foreign_and_bad_keys() {
    let k1 = key(1);
    let mut bad = Jwk::from_verifying_key(key(2).verifying_key());
    bad.kid = "not-the-thumbprint".into();
    let body = json!({"keys": [
        {"kty": "RSA", "n": "abc", "e": "AQAB", "kid": "rsa-1"},
        {"kty": "oct", "k": "c2VjcmV0", "kid": "hmac"},
        bad,
        Jwk::from_verifying_key(k1.verifying_key()),
    ], "extra": true});
    let keys = JwkSet::parse_usable(body.to_string().as_bytes()).unwrap();
    assert_eq!(keys.len(), 1);
    assert_eq!(keys.get(&kid(&k1)), Some(k1.verifying_key()));
}

#[test]
fn parse_usable_errors() {
    for body in [
        &b"not json"[..],
        br#"[{"keys":[]}]"#,
        br#"{"keys":[]}"#,
        br#"{"keys":[{"kty":"RSA","n":"a","e":"b","kid":"x"}]}"#,
        br#"{}"#,
    ] {
        assert_eq!(
            JwkSet::parse_usable(body).unwrap_err(),
            AuthError::KeysUnavailable
        );
    }
}

// ---- JwksCache ----------------------------------------------------------

#[test]
fn fresh_hit_fetches_once() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    for _ in 0..5 {
        assert_eq!(
            block_on(cache.verifying_key(&kid(&k))),
            Ok(*k.verifying_key())
        );
        clock.advance(60);
    }
    assert_eq!(fetch.calls.get(), 1);
    assert_eq!(*fetch.last_url.borrow(), JWKS_URL);
}

#[test]
fn unknown_kid_forces_rate_limited_refresh() {
    let (k1, k2) = (key(1), key(2));
    let fetch = CountingFetch::serving(jwks_body(&[&k1]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert!(block_on(cache.verifying_key(&kid(&k1))).is_ok());
    assert_eq!(fetch.calls.get(), 1);

    // Unknown kid within 30 s of the last attempt: no fetch.
    clock.advance(10);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k2))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 1);

    // Rotation published; after the interval one refetch picks it up.
    *fetch.body.borrow_mut() = jwks_body(&[&k1, &k2]);
    clock.advance(20);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k2))),
        Ok(*k2.verifying_key())
    );
    assert_eq!(fetch.calls.get(), 2);

    // A still-unknown kid right after is rate-limited again.
    clock.advance(5);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&key(9)))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 2);
    clock.advance(30);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&key(9)))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 3);
    // Known keys keep working throughout.
    assert!(block_on(cache.verifying_key(&kid(&k1))).is_ok());
    assert_eq!(fetch.calls.get(), 3);
}

#[test]
fn ttl_expiry_refetches() {
    let (k1, k2) = (key(1), key(2));
    let fetch = CountingFetch::serving(jwks_body(&[&k1]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert!(block_on(cache.verifying_key(&kid(&k1))).is_ok());
    // Key k1 retired at the AS.
    *fetch.body.borrow_mut() = jwks_body(&[&k2]);
    clock.advance(599);
    assert!(block_on(cache.verifying_key(&kid(&k1))).is_ok());
    assert_eq!(fetch.calls.get(), 1);
    clock.advance(1);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k1))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(fetch.calls.get(), 2);
    assert!(block_on(cache.verifying_key(&kid(&k2))).is_ok());
    assert_eq!(fetch.calls.get(), 2);
}

#[test]
fn stale_if_error_then_unavailable() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
    fetch.fail.set(true);
    clock.advance(3_600);
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
    assert_eq!(fetch.calls.get(), 2);
    // Failed attempt is rate-limited too; stale set still served.
    clock.advance(1);
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
    assert_eq!(fetch.calls.get(), 2);
    clock.advance(86_400);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k))),
        Err(AuthError::KeysUnavailable)
    );
    // Recovery.
    fetch.fail.set(false);
    clock.advance(30);
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
}

#[test]
fn never_fetched_is_unavailable() {
    let fetch = CountingFetch::serving(Vec::new());
    fetch.fail.set(true);
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    let err = block_on(cache.verifying_key(&kid(&key(1)))).unwrap_err();
    assert_eq!(err, AuthError::KeysUnavailable);
    assert_eq!(err.http_status(), 503);
}

#[test]
fn bad_responses_never_replace_good_set() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());

    let oversize = {
        let mut v = jwks_body(&[&key(2)]);
        v.resize(64 * 1024 + 1, b' ');
        v
    };
    let bad_bodies: Vec<(u16, Vec<u8>)> = vec![
        (500, jwks_body(&[&key(2)])),
        (200, b"{\"keys\":[]}".to_vec()),
        (200, b"garbage".to_vec()),
        (200, oversize),
    ];
    for (status, body) in bad_bodies {
        fetch.status.set(status);
        *fetch.body.borrow_mut() = body;
        clock.advance(601);
        assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
        assert_eq!(
            block_on(cache.verifying_key(&kid(&key(2)))),
            Err(AuthError::UnknownKid)
        );
    }
}

#[test]
fn pinned_kids_skip_fetch_and_filter_keys() {
    let (k1, k2) = (key(1), key(2));
    let fetch = CountingFetch::serving(jwks_body(&[&k1, &k2]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy().with_accepted_kids([kid(&k1)]));
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k2))),
        Err(AuthError::UnknownKid)
    );
    assert_eq!(
        fetch.calls.get(),
        0,
        "unpinned kid must not trigger a fetch"
    );
    assert!(block_on(cache.verifying_key(&kid(&k1))).is_ok());
    assert_eq!(fetch.calls.get(), 1);
}

/// Regression: concurrent requests on a cold cache share one in-flight
/// fetch instead of the follower getting a spurious 503.
#[test]
fn concurrent_cold_requests_share_one_fetch() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    fetch.yield_once.set(true);
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    let (a, b) = block_on(futures::future::join(
        cache.verifying_key(&kid(&k)),
        cache.verifying_key(&kid(&k)),
    ));
    assert_eq!(a, Ok(*k.verifying_key()));
    assert_eq!(b, Ok(*k.verifying_key()));
    assert_eq!(fetch.calls.get(), 1);
}

/// Followers of a failed in-flight fetch fall back to the stale set and
/// never start a second fetch.
#[test]
fn followers_of_failed_refresh_use_stale_set() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert!(block_on(cache.verifying_key(&kid(&k))).is_ok());
    clock.advance(601);
    fetch.fail.set(true);
    fetch.yield_once.set(true);
    let (a, b) = block_on(futures::future::join(
        cache.verifying_key(&kid(&k)),
        cache.verifying_key(&kid(&k)),
    ));
    assert_eq!((a.is_ok(), b.is_ok()), (true, true));
    assert_eq!(fetch.calls.get(), 2);
}

/// Regression: one transient cold-start failure is retried after the short
/// cold interval, not after the full 30 s refetch interval.
#[test]
fn cold_failure_retries_quickly() {
    let k = key(1);
    let fetch = CountingFetch::serving(jwks_body(&[&k]));
    fetch.fail.set(true);
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k))),
        Err(AuthError::KeysUnavailable)
    );
    // Same second: rate limited, no hammering.
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k))),
        Err(AuthError::KeysUnavailable)
    );
    assert_eq!(fetch.calls.get(), 1);
    fetch.fail.set(false);
    clock.advance(1);
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k))),
        Ok(*k.verifying_key())
    );
    assert_eq!(fetch.calls.get(), 2);
}

/// Regression: the body bound is handed to the port, and an oversized body
/// (the port returns at most `max + 1` bytes) is refused.
#[test]
fn body_limit_is_passed_to_the_port() {
    let k = key(1);
    let mut body = jwks_body(&[&k]);
    body.resize(64 * 1024 + 500, b' ');
    let fetch = CountingFetch::serving(body);
    let clock = TestClock::at(NOW);
    let cache = JwksCache::new(&fetch, &clock, policy());
    assert_eq!(
        block_on(cache.verifying_key(&kid(&k))),
        Err(AuthError::KeysUnavailable)
    );
    assert_eq!(fetch.last_max.get(), 64 * 1024);
}
