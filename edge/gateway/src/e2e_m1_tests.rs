//! M1 review regressions, end to end over the production authenticator,
//! routes and handlers (see `e2e_tests.rs` for the harness): MCP-only
//! bootstrap, the MCP protocol header, adapter auditing, idempotency keys
//! (REST replay, concurrent `op_id`s) and quota admission before shard
//! fan-out. Plus the streaming body cap.

use crate::api::{read_capped, MAX_BODY_BYTES};
use crate::backend::ledger;
use crate::backend::mem::MemBackend;
use crate::e2e_as::ADAPTER;
use crate::e2e_world::World;
use crate::testkit::{block_on, T0};
use crate::wire::{LedgerCall, LedgerOut};
use ruvector_edge_authz::resource::TEAM_RESOURCE_URL;
use ruvector_edge_store::{schema, ResidentRegistry, SqlStore, Value};
use ruvector_edge_tenancy::{QuotaLimits, TenantKey};
use serde_json::{json, Value as Json};
use sha2::{Digest, Sha256};
use worker::Method;

const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];

fn call(name: &str, id: u64, args: Json) -> Json {
    json!({ "jsonrpc": "2.0", "id": id, "method": "tools/call",
        "params": { "name": name, "arguments": args } })
}

fn result(r: &crate::rest::ApiReply) -> Json {
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["result"]["isError"], json!(false), "{}", r.body);
    v["result"]["structuredContent"].clone()
}

#[test]
fn mcp_only_connector_claims_creates_upserts_and_queries() {
    let w = World::new("ruvector:read ruvector:write");
    let mcp = w.edge_as.user_token("carol", "org-c", &w.mcp(), &RW);
    let send = |m: Json| w.send(&mcp, Method::Post, "/v1/mcp", m, None);
    // Bootstrap with the MCP token alone (no REST `/v1` token exists).
    assert_eq!(
        result(&send(call("tenant_claim", 1, json!({}))))["role"],
        json!("owner")
    );
    let spec = json!({ "name": "notes", "dim": 2, "metric": "l2" });
    assert_eq!(
        result(&send(call("collection_create", 2, spec)))["name"],
        json!("notes")
    );
    let rows = json!({ "collection": "notes", "vectors": [
        { "id": "a", "values": [0.0, 0.0] }, { "id": "b", "values": [3.0, 4.0] } ] });
    assert_eq!(
        result(&send(call("vector_upsert", 3, rows)))["upserted"],
        json!(2)
    );
    let q = json!({ "collection": "notes", "vector": [2.9, 4.1], "top_k": 1 });
    assert_eq!(
        result(&send(call("vector_query", 4, q)))["matches"][0]["id"],
        json!("b")
    );
    // The MCP-Protocol-Version header: supported passes, unsupported is 400.
    let list = call("collection_list", 5, json!({}));
    let ok = w.send_with(
        &mcp,
        Method::Post,
        "/v1/mcp",
        list.clone(),
        None,
        Some("2025-06-18"),
    );
    assert_eq!(ok.status, 200);
    for bad in ["1999-01-01", "garbage", "2025-03-26"] {
        let r = w.send_with(&mcp, Method::Post, "/v1/mcp", list.clone(), None, Some(bad));
        assert_eq!(r.status, 400, "{bad}");
    }
    // Still authenticated first: a bad header without a token is a 401.
    let r = w.send_with("x", Method::Post, "/v1/mcp", list, None, Some("garbage"));
    assert_eq!(r.status, 401);
}

#[test]
fn exchanged_writes_record_the_adapter_in_the_ops_log() {
    let w = World::new("ruvector:read ruvector:write");
    let own = w.edge_as.user_token("alice", "org-a", &w.v1(), &RW);
    assert_eq!(
        w.send(&own, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    let spec = json!({ "name": "log", "dim": 2, "metric": "l2", "shards": 1 });
    w.ok(&own, Method::Post, "/v1/collections", spec);
    let up = json!({ "vectors": [{ "id": "u1", "values": [1.0, 0.0] }] });
    w.ok(&own, Method::Post, "/v1/collections/log/vectors", up);
    let team = w
        .edge_as
        .user_token("alice", "org-a", TEAM_RESOURCE_URL, &["team:write"]);
    let tok = w
        .edge_as
        .exchange(&team, &w.v1(), None)
        .expect("exchange")
        .access_token;
    let del = json!({ "collection": "log", "ids": ["u1"] });
    let (s, r) = w.op(&tok, "01HZX0000000000000000AUD01", "vector_delete", del);
    assert_eq!((s, r["result"]["deleted"].clone()), (200, json!(1)), "{r}");
    let shards = w.b.shards.borrow();
    assert_eq!(shards.len(), 1);
    let store = shards.values().next().unwrap();
    let rows = store
        .query(schema::OPS_ACTORS, &[0i64.into(), 10i64.into()])
        .unwrap();
    let who: Vec<(String, Option<String>)> = rows
        .iter()
        .map(|r| {
            (
                r[1].as_text().unwrap().to_string(),
                r[3].as_text().map(str::to_string),
            )
        })
        .collect();
    assert_eq!(
        who,
        [
            ("upsert".into(), None),
            ("delete".into(), Some(ADAPTER.to_string()))
        ]
    );
    assert!(matches!(rows[0][3], Value::Null));
    // Same user (`actor_sub`) on both: only `act_sub` tells them apart.
    assert_eq!(rows[0][2], rows[1][2]);
}

fn daily_ops(w: &World, tok: &str) -> u64 {
    w.ok(tok, Method::Get, "/v1/usage", Json::Null)["usage"]["daily_ops"]
        .as_u64()
        .unwrap()
}

#[test]
fn rest_idempotency_key_replays_binds_route_and_body() {
    let w = World::new("ruvector:read ruvector:write");
    let own = w.edge_as.user_token("alice", "org-a", &w.v1(), &RW);
    w.send(&own, Method::Post, "/v1/claim", Json::Null, None);
    let spec = json!({ "name": "k1", "dim": 2, "metric": "l2" });
    let create =
        |body: Json, key: &str| w.send(&own, Method::Post, "/v1/collections", body, Some(key));
    let first = create(spec.clone(), "create-k1");
    assert_eq!(first.status, 201, "{}", first.body);
    // A retry after a lost 201 gets the stored 201, not 409 conflict.
    let again = create(spec.clone(), "create-k1");
    assert_eq!(
        (again.status, again.body.clone()),
        (201, first.body.clone())
    );
    // Same key, another body: 409 op_replayed.
    let other = create(
        json!({ "name": "k2", "dim": 2, "metric": "l2" }),
        "create-k1",
    );
    assert_eq!(other.status, 409);
    assert!(other.body.contains("op_replayed"), "{}", other.body);
    // A replayed upsert is neither re-executed nor charged again.
    w.ok(
        &own,
        Method::Post,
        "/v1/collections",
        json!({ "name": "k3", "dim": 2, "metric": "l2" }),
    );
    let up = json!({ "vectors": [{ "id": "a", "values": [1.0, 2.0] }] });
    let path = "/v1/collections/k1/vectors";
    let r1 = w.send(&own, Method::Post, path, up.clone(), Some("up-1"));
    let ops = daily_ops(&w, &own);
    let r2 = w.send(&own, Method::Post, path, up.clone(), Some("up-1"));
    assert_eq!((r2.status, r2.body.clone()), (200, r1.body.clone()));
    assert_eq!(daily_ops(&w, &own), ops + 1, "only the usage read counts");
    // The key is bound to the route: the same body on k3 is a conflict.
    let r3 = w.send(
        &own,
        Method::Post,
        "/v1/collections/k3/vectors",
        up.clone(),
        Some("up-1"),
    );
    assert_eq!(r3.status, 409);
    // Bad keys are 400; read routes and dry runs ignore the header.
    for bad in ["", &"k".repeat(256), "has space"] {
        let r = w.send(&own, Method::Post, path, up.clone(), Some(bad));
        assert_eq!(r.status, 400, "{bad:?}");
    }
    let dry = json!({ "vectors": [{ "id": "z", "values": [1.0, 2.0] }], "dry_run": true });
    assert_eq!(
        w.send(&own, Method::Post, path, dry, Some("up-1")).status,
        200
    );
    let get = w.send(
        &own,
        Method::Get,
        "/v1/collections",
        Json::Null,
        Some("up-1"),
    );
    assert_eq!(get.status, 200);
    // DELETE honours it too: the retry replays the first answer.
    let del = json!({ "ids": ["a"] });
    let d1 = w.send(&own, Method::Delete, path, del.clone(), Some("del-1"));
    let d2 = w.send(&own, Method::Delete, path, del, Some("del-1"));
    assert_eq!((d1.status, d2.body), (200, d1.body.clone()));
    assert!(d1.body.contains("\"deleted\":1"), "{}", d1.body);
}

#[test]
fn concurrent_op_id_is_in_flight_and_never_executes_twice() {
    let w = World::new("ruvector:read ruvector:write");
    let own = w.edge_as.user_token("alice", "org-a", &w.v1(), &RW);
    w.send(&own, Method::Post, "/v1/claim", Json::Null, None);
    let me = w.ok(&own, Method::Get, "/v1/me", Json::Null);
    let tenant = TenantKey::parse(me["tenant_key"].as_str().unwrap()).unwrap();
    let sub = me["sub"].as_str().unwrap().to_string();
    let op_id = "01HZX0000000000000000RACE1";
    let env = json!({
        "v": 1, "op_id": op_id, "target": format!("{}/ops", w.v1()),
        "tenant_key": tenant.as_str(), "op": "collection_create",
        "args": { "name": "race", "dim": 2, "metric": "l2" },
    });
    let sha256: [u8; 32] = Sha256::digest(env.to_string().as_bytes()).into();
    // The first request has reserved the op_id and is still running.
    let reserve = LedgerCall::IdemReserve {
        sub: sub.clone(),
        key: op_id.into(),
        sha256,
        now: T0,
    };
    let out = block_on(ledger(&w.b, &tenant, reserve)).unwrap();
    assert!(matches!(
        out,
        LedgerOut::Idem {
            replay: None,
            conflict: false,
            in_flight: false
        }
    ));
    // Its twin is refused (409 + retry_after_s) instead of executing.
    let twin = w.send(&own, Method::Post, "/v1/ops", env.clone(), Some(op_id));
    let v: Json = serde_json::from_str(&twin.body).unwrap();
    assert_eq!(twin.status, 409, "{}", twin.body);
    assert_eq!(v["error"]["code"], json!("conflict"));
    assert_eq!(v["error"]["retry_after_s"], json!(1));
    assert_eq!(
        w.ok(&own, Method::Get, "/v1/collections", Json::Null)["collections"],
        json!([])
    );
    // Once the first finishes (here: fails and releases), a retry executes
    // once, and later retries replay the stored 201-equivalent response.
    let release = LedgerCall::IdemRelease {
        sub,
        key: op_id.into(),
        sha256,
    };
    block_on(ledger(&w.b, &tenant, release)).unwrap();
    let first = w.send(&own, Method::Post, "/v1/ops", env.clone(), Some(op_id));
    assert_eq!(first.status, 200, "{}", first.body);
    let replay = w.send(&own, Method::Post, "/v1/ops", env, Some(op_id));
    assert_eq!((replay.status, replay.body), (200, first.body));
    // A failed mutating op releases its reservation: the retry runs again.
    let bad = json!({ "collection": "nope", "vectors": [{ "id": "a", "values": [1.0, 0.0] }] });
    let (s, _) = w.op(
        &own,
        "01HZX0000000000000000RACE2",
        "vector_upsert",
        bad.clone(),
    );
    assert_eq!(s, 404);
    let (s, r) = w.op(&own, "01HZX0000000000000000RACE2", "vector_upsert", bad);
    assert_eq!((s, r["error"]["code"].clone()), (404, json!("not_found")));
}

#[test]
fn over_quota_tenants_are_refused_before_any_shard_call() {
    let mut w = World::new("ruvector:read ruvector:write");
    let limits = QuotaLimits {
        max_daily_ops: 4,
        ..crate::ledger_core::FREE_PLAN
    };
    w.b = MemBackend::with(limits, ResidentRegistry::default());
    let own = w.edge_as.user_token("alice", "org-a", &w.v1(), &RW);
    w.send(&own, Method::Post, "/v1/claim", Json::Null, None);
    let spec = json!({ "name": "q", "dim": 2, "metric": "l2", "shards": 3 });
    w.ok(&own, Method::Post, "/v1/collections", spec); // op 1
                                                       // Dry runs are admitted (and counted) before touching shards.
    let up = json!({ "vectors": [{ "id": "a", "values": [1.0, 0.0] }], "dry_run": true });
    w.ok(&own, Method::Post, "/v1/collections/q/vectors", up.clone()); // op 2
    let del = json!({ "ids": ["a"], "dry_run": true });
    w.ok(
        &own,
        Method::Delete,
        "/v1/collections/q/vectors",
        del.clone(),
    ); // op 3
    w.ok(&own, Method::Get, "/v1/usage", Json::Null); // op 4
                                                      // Over quota, with every shard unreachable: each fan-out route answers
                                                      // 413 quota_exceeded (admission first), never 503 shard_unavailable.
    w.b.shard_down.set(true);
    let q = json!({ "vector": [1.0, 0.0], "top_k": 1 });
    let fetch = json!({ "ids": ["a"] });
    let cases = [
        (Method::Post, "/v1/collections/q/query", q),
        (Method::Post, "/v1/collections/q/fetch", fetch),
        (Method::Get, "/v1/collections/q", Json::Null),
        (Method::Post, "/v1/collections/q/vectors", up),
        (Method::Delete, "/v1/collections/q/vectors", del),
    ];
    for (m, path, body) in cases {
        let r = w.send(&own, m, path, body, None);
        assert_eq!(r.status, 413, "{path}: {}", r.body);
        assert!(r.body.contains("quota_exceeded"), "{path}: {}", r.body);
    }
}

#[test]
fn body_reads_stop_at_the_cap_while_streaming() {
    use futures_util::stream;
    let chunk = |n: usize| Ok::<Vec<u8>, ()>(vec![7u8; n]);
    let exact = stream::iter(vec![chunk(MAX_BODY_BYTES / 2), chunk(MAX_BODY_BYTES / 2)]);
    assert_eq!(
        block_on(read_capped(exact, MAX_BODY_BYTES))
            .unwrap()
            .unwrap()
            .len(),
        MAX_BODY_BYTES
    );
    // One byte over is refused without reading what follows (the tail
    // would error if polled).
    let over = stream::iter(vec![chunk(MAX_BODY_BYTES), chunk(1), Err(())]);
    assert_eq!(block_on(read_capped(over, MAX_BODY_BYTES)), Ok(None));
    let huge = stream::iter((0..200).map(|_| chunk(1 << 20)));
    assert_eq!(block_on(read_capped(huge, MAX_BODY_BYTES)), Ok(None));
    assert_eq!(
        block_on(read_capped(stream::iter(vec![Err(())]), 8)),
        Err(())
    );
    assert_eq!(
        block_on(read_capped(
            stream::iter(Vec::<Result<Vec<u8>, ()>>::new()),
            8
        )),
        Ok(Some(vec![]))
    );
}
