//! Review regressions: collection drop races (tombstone first, by uid,
//! wiped-shard marker) and a member removal at the deny-entry cap.

use super::admin_tests::{admin as admin_caps, call, code, who};
use super::Caller;
use crate::backend::mem::MemBackend;
use crate::backend::{ledger as ledger_call, shard, Backend, ShardErr};
use crate::ledger_core::admin::{AdminCall, AdminRequest, MAX_DENY_ENTRIES};
use crate::testkit::*;
use crate::wire::{ActorWire, CollectionWire, DeltaWire, LedgerCall, LedgerOut, ShardCall};
use ruvector_edge_store::{ErrorCode, SqlStore, UpsertRow};
use ruvector_edge_tenancy::{ledger_do_name, CollectionUid, ShardIndex};
use serde_json::{json, Value as Json};
use worker::Method;

fn usage(b: &MemBackend, c: &Caller) -> Json {
    let r = call(b, c, Method::Get, "/v1/usage", Json::Null, T0);
    serde_json::from_str::<Json>(&r.body).unwrap()["usage"].clone()
}

/// One raw admin call (what a request in flight sends to the ledger).
fn admin_call(b: &MemBackend, org: &str, admin: AdminCall) -> Json {
    let body = serde_json::to_string(&AdminRequest {
        tenant_key: tenant(org).as_str().to_string(),
        admin,
    })
    .unwrap();
    let text = block_on(b.call_ledger(&ledger_do_name(&tenant(org)), body)).unwrap();
    serde_json::from_str(&text).unwrap()
}

fn setup(b: &MemBackend, alice: &Caller, rows: usize, shards: u32) -> String {
    assert_eq!(
        call(b, alice, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    create(b, alice, rows, shards)
}

fn create(b: &MemBackend, alice: &Caller, rows: usize, shards: u32) -> String {
    let spec = json!({ "name": "d", "dim": 2, "metric": "l2", "shards": shards });
    let r = call(b, alice, Method::Post, "/v1/collections", spec, T0);
    assert_eq!(r.status, 201, "{}", r.body);
    let uid = serde_json::from_str::<Json>(&r.body).unwrap()["collection_uid"]
        .as_str()
        .unwrap()
        .to_string();
    let v: Vec<Json> = (0..rows)
        .map(|i| json!({ "id": format!("v{i}"), "values": [i as f32, 1.0] }))
        .collect();
    let path = "/v1/collections/d/vectors";
    let r = call(b, alice, Method::Post, path, json!({ "vectors": v }), T0);
    assert_eq!(r.status, 200, "{}", r.body);
    uid
}

fn query_count(b: &MemBackend, c: &Caller) -> usize {
    let q = json!({ "vector": [0.0, 1.0], "top_k": 10 });
    let r = call(b, c, Method::Post, "/v1/collections/d/query", q, T0);
    assert_eq!(r.status, 200, "{}", r.body);
    serde_json::from_str::<Json>(&r.body).unwrap()["matches"]
        .as_array()
        .unwrap()
        .len()
}

#[test]
fn a_stale_drop_never_tombstones_a_newer_same_name_collection() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", admin_caps());
    let x = setup(&b, &alice, 4, 2);
    // Request A begins the drop (the ledger picks uid X) and stalls.
    let begun = admin_call(
        &b,
        "org-a",
        AdminCall::Drop {
            actor: sub("alice"),
            name: "d".into(),
        },
    );
    assert_eq!(begun["Ok"]["entry"]["uid"], json!(x));
    // X is no longer served; its name stays reserved while it is wiped.
    let r = call(&b, &alice, Method::Get, "/v1/collections/d", Json::Null, T0);
    assert_eq!(code(&r).0, 404);
    let spec = json!({ "name": "d", "dim": 2, "metric": "l2" });
    let r = call(&b, &alice, Method::Post, "/v1/collections", spec, T0);
    assert_eq!(code(&r), (409, "conflict".into()), "{}", r.body);
    // A client retry (request B) finishes X: wipe + purge.
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
        (json!(x), json!(4))
    );
    // An editor recreates the name (uid Y) and writes into it.
    let y = create(&b, &alice, 3, 1);
    assert_ne!(x, y);
    // A finally resumes: its purge is by uid X and never touches Y.
    let late = admin_call(
        &b,
        "org-a",
        AdminCall::Purge {
            actor: sub("alice"),
            uid: x.clone(),
        },
    );
    assert_eq!(late["Ok"]["entry"]["uid"], json!(x));
    assert_eq!(query_count(&b, &alice), 3);
    let u = usage(&b, &alice);
    assert_eq!(
        (u["vectors"].clone(), u["collections"].clone()),
        (json!(3), json!(1))
    );
}

#[test]
fn a_write_racing_a_drop_is_refused_by_the_wiped_shard() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", admin_caps());
    setup(&b, &alice, 2, 1);
    // An editor's upsert looked the collection up before the drop...
    let call_c = LedgerCall::Collection { name: "d".into() };
    let e: CollectionWire = match block_on(ledger_call(&b, &tenant("org-a"), call_c)).unwrap() {
        LedgerOut::Collection { entry } => entry.unwrap(),
        _ => panic!(),
    };
    // ...then the owner drops it.
    let r = call(
        &b,
        &alice,
        Method::Delete,
        "/v1/collections/d",
        Json::Null,
        T0,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    // The stale plan and apply land on the wiped shard: 404, nothing
    // re-initialised, nothing charged.
    let uid = CollectionUid::parse(&e.uid).unwrap();
    let dm = ruvector_edge_store::shard_meta_for(&tenant("org-a"), uid, ShardIndex::ZERO).unwrap();
    let rows = vec![UpsertRow {
        id: "late".into(),
        values: vec![1.0, 1.0],
        metadata: None,
    }];
    let plan = ShardCall::Plan {
        cfg: e.cfg.clone(),
        rows: rows.clone(),
    };
    let apply = ShardCall::Apply {
        cfg: e.cfg.clone(),
        rows,
        admitted: DeltaWire {
            vectors: 1,
            floats: 2,
            bytes: 100,
        },
        actor: ActorWire {
            sub: sub("bob"),
            jti: "j".into(),
            family_id: "f".into(),
            act_sub: None,
        },
        now: T0,
    };
    for c in [plan, apply] {
        match block_on(shard(&b, &dm, c)) {
            Err(ShardErr::Refused(err)) => assert_eq!(err.code, ErrorCode::NotFound),
            other => panic!("{other:?}"),
        }
    }
    let shards = b.shards.borrow();
    let st = shards.get(dm.do_name().as_str()).unwrap();
    assert!(st.query("SELECT id FROM vectors", &[]).unwrap().is_empty());
    let meta = st.query("SELECT k, v FROM meta", &[]).unwrap();
    assert_eq!(meta.len(), 1, "only the wiped marker: {meta:?}");
    drop(shards);
    assert_eq!(usage(&b, &alice)["vectors"], json!(0));
}

#[test]
fn removing_a_member_at_the_deny_cap_still_succeeds() {
    let b = MemBackend::new();
    let alice = who("org-a", "alice", admin_caps());
    assert_eq!(
        call(&b, &alice, Method::Post, "/v1/claim", Json::Null, T0).status,
        201
    );
    let body = json!({ "sub": sub("bob"), "role": "editor" });
    let r = call(&b, &alice, Method::Post, "/v1/tenant/members", body, T0);
    assert_eq!(r.status / 100, 2, "{}", r.body);
    for i in 0..MAX_DENY_ENTRIES {
        let d = json!({ "kind": "client_id", "value": format!("c-{i:04}"), "ttl_s": 3600 });
        let r = call(&b, &alice, Method::Post, "/v1/tenant/deny", d, T0);
        assert_eq!(r.status, 201, "{i}: {}", r.body);
    }
    let d = json!({ "kind": "client_id", "value": "c-full", "ttl_s": 3600 });
    let r = call(&b, &alice, Method::Post, "/v1/tenant/deny", d, T0);
    assert_eq!(code(&r).0, 413, "owner entries stay capped");
    // The removal commits and reports success; its deny is written too.
    let path = format!("/v1/tenant/members/{}", sub("bob"));
    let r = call(&b, &alice, Method::Delete, &path, Json::Null, T0);
    assert_eq!(r.status, 200, "{}", r.body);
    let listed = admin_call(&b, "org-a", AdminCall::DenyList { now: T0 });
    let entries = listed["Ok"]["entries"].as_array().unwrap();
    assert!(entries
        .iter()
        .any(|e| e["kind"] == json!("sub") && e["value"] == json!(sub("bob"))));
    let r = call(&b, &alice, Method::Delete, &path, Json::Null, T0);
    assert_eq!(code(&r).0, 404);
}

#[test]
fn mutating_ops_and_mcp_tools_are_charged_to_the_write_budget() {
    use crate::api::ratelimit::{op_class, Class};
    let ops = |op: &str| json!({ "v": 1, "op": op }).to_string().into_bytes();
    let tool = |name: &str| {
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": name } })
            .to_string()
            .into_bytes()
    };
    for w in ["vector_upsert", "vector_delete", "collection_create"] {
        assert_eq!(op_class(Class::Ops, &ops(w)), Some(Class::Write), "{w}");
        assert_eq!(op_class(Class::Mcp, &tool(w)), Some(Class::Write), "{w}");
    }
    assert_eq!(
        op_class(Class::Mcp, &tool("tenant_claim")),
        Some(Class::Write)
    );
    for r in [
        "vector_query",
        "vector_fetch",
        "collection_list",
        "usage_get",
    ] {
        assert_eq!(op_class(Class::Ops, &ops(r)), None, "{r}");
        assert_eq!(op_class(Class::Mcp, &tool(r)), None, "{r}");
    }
    let init = json!({ "jsonrpc": "2.0", "id": 1, "method": "initialize" }).to_string();
    assert_eq!(op_class(Class::Mcp, init.as_bytes()), None);
    // The op name only counts on its own surface; junk is left to the handler.
    assert_eq!(op_class(Class::Mcp, &ops("vector_upsert")), None);
    assert_eq!(op_class(Class::Ops, &tool("vector_upsert")), None);
    assert_eq!(op_class(Class::Ops, b"not json"), None);
    assert_eq!(op_class(Class::Read, &ops("vector_upsert")), None);
}

#[test]
fn a_query_pays_one_read_token_per_shard() {
    let b = MemBackend::new().with_fanout_limiter();
    let alice = who("org-a", "alice", admin_caps());
    setup(&b, &alice, 6, 6);
    // 6 shards: 5 extra tokens per query (the request's own token is the
    // guard's); the per-user read budget (20 / 10 s) admits 4 of them.
    for _ in 0..4 {
        assert_eq!(query_count(&b, &alice), 6);
    }
    let q = json!({ "vector": [0.0, 1.0], "top_k": 10 });
    let r = call(&b, &alice, Method::Post, "/v1/collections/d/query", q, T0);
    assert_eq!(code(&r), (429, "rate_limited".into()), "{}", r.body);
    // A single-shard collection's query costs nothing extra.
    let bob = who("org-b", "bob", admin_caps());
    let b1 = MemBackend::new().with_fanout_limiter();
    setup(&b1, &bob, 2, 1);
    for _ in 0..30 {
        assert_eq!(query_count(&b1, &bob), 2);
    }
}
