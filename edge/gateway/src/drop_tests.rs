//! Collection drop (shard wipe) and the rate-limit guard, over the
//! in-process DO backend.

use super::admin_tests::{admin, bob, call, code, who};
use super::Caller;
use crate::api::guard;
use crate::api::ratelimit::mem::CountingLimiter;
use crate::api::ratelimit::Class;
use crate::backend::mem::MemBackend;
use crate::backend::Backend;
use crate::testkit::*;
use ruvector_edge_store::{OpError, SqlStore};
use serde_json::{json, Value as Json};
use worker::Method;

fn usage<B: Backend>(b: &B, c: &Caller) -> Json {
    let r = call(b, c, Method::Get, "/v1/usage", Json::Null, T0);
    serde_json::from_str::<Json>(&r.body).unwrap()["usage"].clone()
}

#[test]
fn drop_collection_wipes_shards_tombstones_the_uid_and_frees_the_name() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", rw());
    assert_eq!(
        call(&b, &alice, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    b.invite(
        &tenant("org-a"),
        &sub("alice"),
        &sub("bob"),
        ruvector_edge_tenancy::Role::Editor,
        T0,
    )
    .unwrap();
    let spec = json!({ "name": "d", "dim": 2, "metric": "l2", "shards": 2 });
    let r = call(
        &b,
        &alice,
        Method::Post,
        "/v1/collections",
        spec.clone(),
        T0,
    );
    assert_eq!(r.status, 201, "{}", r.body);
    let uid1 = serde_json::from_str::<Json>(&r.body).unwrap()["collection_uid"].clone();
    let rows: Vec<Json> = (0..6)
        .map(|i| json!({ "id": format!("v{i}"), "values": [i as f32, 1.0] }))
        .collect();
    let r = call(
        &b,
        &alice,
        Method::Post,
        "/v1/collections/d/vectors",
        json!({ "vectors": rows }),
        T0,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    assert_eq!(usage(&b, &alice)["vectors"], json!(6));
    // Scope before role: a read-only owner steps up to write; an editor
    // (even with admin scope) is not the owner.
    let r = call(
        &b,
        &who("org-a", "alice", ro()),
        Method::Delete,
        "/v1/collections/d",
        Json::Null,
        T0,
    );
    assert_eq!(code(&r), (403, "insufficient_scope".into()));
    assert!(r.www_authenticate.unwrap().contains("ruvector:write"));
    let r = call(
        &b,
        &bob("org-a", admin()),
        Method::Delete,
        "/v1/collections/d",
        Json::Null,
        T0,
    );
    assert_eq!(code(&r), (403, "role_required".into()));
    // Another tenant's owner: its own (absent) name, 404.
    let eve = who("org-b", "eve", rw());
    assert_eq!(
        call(&b, &eve, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    assert_eq!(
        code(&call(
            &b,
            &eve,
            Method::Delete,
            "/v1/collections/d",
            Json::Null,
            T0
        ))
        .0,
        404
    );
    // Owner drops: every shard wiped, usage released, uid tombstoned.
    let r = call(
        &b,
        &alice,
        Method::Delete,
        "/v1/collections/d",
        Json::Null,
        T0,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        (
            v["collection_uid"].clone(),
            v["released"]["vectors"].clone()
        ),
        (uid1.clone(), json!(6))
    );
    for store in b.shards.borrow().values() {
        assert!(store
            .query("SELECT id FROM vectors", &[])
            .unwrap()
            .is_empty());
    }
    // No resident index survives to flush into the wiped storage.
    assert_eq!(b.host.borrow().resident_count(), 0);
    let u = usage(&b, &alice);
    assert_eq!(
        (
            u["vectors"].clone(),
            u["collections"].clone(),
            u["bytes"].clone()
        ),
        (json!(0), json!(0), json!(0))
    );
    assert_eq!(
        code(&call(
            &b,
            &alice,
            Method::Get,
            "/v1/collections/d",
            Json::Null,
            T0
        ))
        .0,
        404
    );
    let q = json!({ "vector": [0.0, 1.0], "top_k": 3 });
    assert_eq!(
        code(&call(
            &b,
            &alice,
            Method::Post,
            "/v1/collections/d/query",
            q.clone(),
            T0
        ))
        .0,
        404
    );
    assert_eq!(
        code(&call(
            &b,
            &alice,
            Method::Delete,
            "/v1/collections/d",
            Json::Null,
            T0
        ))
        .0,
        404
    );
    // The name is free again, under a new uid (new, empty shards).
    let r = call(&b, &alice, Method::Post, "/v1/collections", spec, T0);
    assert_eq!(r.status, 201, "{}", r.body);
    let uid2 = serde_json::from_str::<Json>(&r.body).unwrap()["collection_uid"].clone();
    assert_ne!(uid1, uid2);
    let r = call(&b, &alice, Method::Post, "/v1/collections/d/query", q, T0);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["matches"], json!([]), "{}", r.body);
}

#[test]
fn wipe_refuses_a_foreign_shard_and_is_zero_on_a_fresh_one() {
    use crate::durable::wipe::{handle as wipe, WipeRequest};
    use ruvector_edge_store::{shard_meta_for, MemSqlStore};
    use ruvector_edge_tenancy::{CollectionUid, ShardIndex};
    let uid = CollectionUid::parse(&"ab".repeat(16)).unwrap();
    let dm = shard_meta_for(&tenant("org-a"), uid, ShardIndex::parse("0").unwrap()).unwrap();
    let other = shard_meta_for(&tenant("org-b"), uid, ShardIndex::parse("0").unwrap()).unwrap();
    let store = MemSqlStore::default();
    let req = WipeRequest::for_shard(&dm);
    assert_eq!(
        wipe(Some(other.do_name().as_str()), &store, &req).unwrap_err(),
        OpError::not_found()
    );
    assert_eq!(
        wipe(Some(dm.do_name().as_str()), &store, &req).unwrap(),
        Default::default()
    );
    let bad = WipeRequest { wipe: false, ..req };
    assert_eq!(wipe(None, &store, &bad).unwrap_err(), OpError::not_found());
}

#[test]
fn rate_limit_is_per_user_then_tenant_per_class_with_retry_after() {
    let b = MemBackend::new();
    let l = CountingLimiter::default();
    let users: Vec<Caller> = (0..6)
        .map(|i| who("org-a", &format!("u{i}"), rw()))
        .collect();
    let gw = |c: &Caller, class| block_on(guard(&b, &l, class, &c.ctx, T0, REST_MD));
    // Per user: 10 writes / 10 s, then 429 with Retry-After = the window.
    for _ in 0..10 {
        assert!(gw(&users[0], Class::Write).is_ok());
    }
    let r = gw(&users[0], Class::Write).unwrap_err();
    assert_eq!(
        (code(&r.reply), r.retry_after),
        ((429, "rate_limited".into()), Some(10))
    );
    assert_eq!(r.reply.content_type, "application/problem+json");
    // Separate budgets: reads, ops and mcp still pass for the same user.
    for class in [Class::Read, Class::Ops, Class::Mcp] {
        assert!(gw(&users[0], class).is_ok(), "{class:?}");
    }
    // Per tenant: 50 writes / 10 s across users (u0's refused attempt was
    // never charged to the tenant).
    for u in &users[1..5] {
        for _ in 0..10 {
            assert!(gw(u, Class::Write).is_ok());
        }
    }
    assert_eq!(
        gw(&users[5], Class::Write).unwrap_err().retry_after,
        Some(10)
    );
    // Another tenant has its own budget.
    assert!(gw(&who("org-b", "u0", rw()), Class::Write).is_ok());
    // A new window admits again.
    l.reset();
    assert!(gw(&users[0], Class::Write).is_ok());
    // A flooding caller is throttled before the deny check runs.
    let seen = l.seen.borrow().len();
    for _ in 0..25 {
        let _ = gw(&users[1], Class::Read);
    }
    assert_eq!(gw(&users[1], Class::Read).unwrap_err().reply.status, 429);
    assert!(l.seen.borrow().len() > seen);
}
