//! Abuse controls on the public endpoints: the durable per-IP DCR window
//! and idle-client expiry (ADR-351 §5.6).

use super::*;
use crate::abuse::{CLIENT_IDLE_TTL_SECS, DCR_RATE_WINDOW_SECS, UNUSED_CLIENT_TTL_SECS};
use ruvector_edge_authz::ClientStore;

fn body() -> Value {
    json!({"redirect_uris": [REDIRECT]})
}

/// Regression (DCR flooding): one IP bucket gets `DCR_RATE_PER_HOUR`
/// registrations per window, then 429 with `Retry-After`; other buckets are
/// unaffected and the window resets.
#[test]
fn dcr_is_rate_limited_per_ip_bucket() {
    let mut v = vars();
    v.insert("DCR_RATE_PER_HOUR", "2".into());
    let w = World::with_cfg(load(&v).unwrap());
    assert_eq!(w.register_from("ip-a", body()).status, 201);
    assert_eq!(w.register_from("ip-a", body()).status, 201);
    let r = w.register_from("ip-a", body());
    assert_eq!(r.status, 429);
    assert_eq!(body_json(&r)["error"], "temporarily_unavailable");
    assert!(r.header("Retry-After").is_some());
    // Malformed attempts count too (the window bounds DO work, not rows).
    assert_eq!(w.register_from("ip-a", json!({})).status, 429);
    assert_eq!(w.register_from("ip-b", body()).status, 201);
    w.clock.advance(DCR_RATE_WINDOW_SECS);
    assert_eq!(w.register_from("ip-a", body()).status, 201);
}

/// Regression (permanent DCR lockout): never-used registrations expire
/// after `UNUSED_CLIENT_TTL_SECS`, so junk cannot hold `MAX_CLIENTS`
/// forever; a client that completed a token request is kept until it has
/// been idle for `CLIENT_IDLE_TTL_SECS`.
#[test]
fn abandoned_clients_expire_and_free_the_cap() {
    let mut v = vars();
    v.insert("MAX_CLIENTS", "2".into());
    let w = World::with_cfg(load(&v).unwrap());
    let used = w.client_id();
    let code = login(&w, &used);
    assert_eq!(redeem(&w, &used, &code).status, 200);
    let junk = body_json(&w.register(body()))["client_id"]
        .as_str()
        .unwrap()
        .to_string();
    let r = w.register(body());
    assert_eq!(r.status, 503, "cap reached");
    w.clock.advance(UNUSED_CLIENT_TTL_SECS);
    assert_eq!(
        w.register(body()).status,
        201,
        "junk no longer holds the cap"
    );
    assert!(w.store.get_client(&junk).unwrap().is_none());
    assert!(w.store.get_client(&used).unwrap().is_some());
    w.clock.advance(CLIENT_IDLE_TTL_SECS);
    assert_eq!(w.register(body()).status, 201);
    assert!(
        w.store.get_client(&used).unwrap().is_none(),
        "idle client purged"
    );
}
