//! M2 + M5 + M3 integration regressions: every M3 route is charged to a
//! §10 budget class, M2 / M5 writes are audited, the Worker binds exactly
//! one R2 bucket, snapshot / export / restore keep to the synchronous
//! budget, and a collection drop clears an interrupted restore's staged rows.

use crate::api::ratelimit::{request_class, Class};
use crate::audit_http::{body_event, rest_event, rvf_event};
use crate::m3_mem::*;
use crate::registry_routes::RvfRoute;
use crate::rest::ApiRoute;
use crate::testkit::Rng;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::SqlStore;
use serde_json::{json, Value as Json};
use worker::Method;

#[test]
fn every_m3_route_has_a_budget_class() {
    let (get, post) = (Method::Get, Method::Post);
    for (m, path, want) in [
        (&post, "/v1/collections/c/snapshots", Class::Write),
        (&get, "/v1/collections/c/snapshots", Class::Read),
        (&post, "/v1/collections/c/snapshots/7:restore", Class::Write),
        (&post, "/v1/collections/c:export", Class::Write),
        (&get, "/v1/exports/e1", Class::Read),
        (&post, "/v1/uploads", Class::Write),
        (&post, "/v1/uploads/u1/parts/1", Class::Write),
        (&post, "/v1/uploads/u1:complete", Class::Write),
        (&post, "/v1/collections/c:import", Class::Write),
        (&get, "/v1/jobs/j1", Class::Read),
        (&post, "/v1/collections", Class::Write),
        (&post, "/v1/collections/c/vectors:upsert", Class::Write),
        (&post, "/v1/collections/c/query", Class::Read),
        // M2 / M5 neighbours keep their own classes.
        (&post, "/v1/collections/c:import-rvf", Class::Write),
        (&Method::Delete, "/v1/collections/c", Class::Write),
        (&get, "/v1/tenant/members", Class::Read),
    ] {
        assert_eq!(request_class(false, m, path), want, "{m:?} {path}");
    }
}

#[test]
fn m2_and_m5_writes_are_audited_and_their_reads_are_not() {
    let c = |r| rest_event(&r);
    assert_eq!(
        c(ApiRoute::Drop("d".into())),
        Some(("collection.drop", Capability::CreateCollection))
    );
    assert_eq!(
        c(ApiRoute::Invite),
        Some(("member.invite", Capability::Admin))
    );
    assert_eq!(
        c(ApiRoute::RemoveMember("s".into())),
        Some(("member.remove", Capability::Admin))
    );
    assert_eq!(c(ApiRoute::Deny), Some(("tenant.deny", Capability::Admin)));
    assert_eq!(c(ApiRoute::Members), None);
    let s = || "s".to_string();
    let r = |x: RvfRoute| rvf_event(&x).map(|(n, _)| n);
    assert_eq!(r(RvfRoute::Publish(s(), s())), Some("rvf.publish"));
    assert_eq!(
        r(RvfRoute::Part(s(), s(), s(), s())),
        Some("rvf.upload.part")
    );
    assert_eq!(r(RvfRoute::Import(s())), Some("collection.import_rvf"));
    assert_eq!(
        rvf_event(&RvfRoute::ClaimScope(s())).map(|e| e.1),
        Some(Capability::Admin)
    );
    assert_eq!(r(RvfRoute::Blob(s(), s())), None);
    assert_eq!(r(RvfRoute::ListScopes), None);
    // The M5 MCP write tool is audited like the M1 ones; its reads are not.
    let call = |name: &str| {
        json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                "params": { "name": name, "arguments": {} } })
        .to_string()
    };
    assert_eq!(
        body_event(true, call("rvf_import").as_bytes()),
        Some(("mcp.rvf_import".into(), Capability::Write))
    );
    assert_eq!(body_event(true, call("rvf_get").as_bytes()), None);
}

#[test]
fn wrangler_binds_one_r2_bucket_the_queues_and_ai_and_no_new_classes() {
    let toml = include_str!("../wrangler.toml");
    assert_eq!(crate::m3_ports::DATA_BINDING, crate::blob_r2::R2_BINDING);
    let r2: Vec<&str> = toml.split("[[r2_buckets]]").skip(1).collect();
    assert_eq!(r2.len(), 1, "exactly one R2 binding");
    let r2 = r2[0].split("\n[").next().unwrap();
    assert!(r2.contains("binding = \"EDGE_DATA\""));
    assert!(r2.contains("bucket_name = \"ruvector-edge-data\""));
    for q in ["ruvector-edge-ingest", "ruvector-edge-audit"] {
        assert!(toml.contains(&format!("queue = \"{q}\"")));
        assert!(toml.contains(&format!("dead_letter_queue = \"{q}-dlq\"")));
    }
    assert!(toml.contains("[ai]\nbinding = \"AI\""));
    // Default CPU limits: no `[limits]` table.
    assert!(!toml.lines().any(|l| l.trim() == "[limits]"));
    // M3 adds no Durable Object class: the applied tags only.
    let tags: Vec<&str> = toml
        .lines()
        .filter_map(|l| l.trim().strip_prefix("tag = "))
        .collect();
    assert_eq!(tags, ["\"v1\"", "\"v2-registry\""]);
}

const DIM: usize = 384;

fn seed(w: &World, c: &crate::rest::Caller, n: usize) {
    let mut rng = Rng(5);
    for chunk in (0..n).collect::<Vec<_>>().chunks(500) {
        let rows: Vec<Json> = chunk
            .iter()
            .map(|i| json!({ "id": format!("v{i:05}"), "values": rng.vec(DIM) }))
            .collect();
        let path = "/v1/collections/big/vectors:upsert";
        let (s, v) = w.req(c, Method::Post, path, json!({ "vectors": rows }));
        assert_eq!(s, 200, "{v}");
    }
}

/// Snapshot, export and restore run in one request, so they are refused
/// `413` past the synchronous budget (`sync_budget`, 682 rows × 384) before
/// any side effect: no epoch, no R2 object, no multipart upload, no
/// restore journal. At the budget they still round-trip.
#[test]
fn snapshot_export_and_restore_refuse_past_the_sync_budget_before_side_effects() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "big", "dim": DIM, "metric": "cosine", "shards": 2 });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let max = crate::sync_budget::max_rows(DIM as u32) as usize;
    assert_eq!(max, 682);
    seed(&w, &o, max);
    let snap = "/v1/collections/big/snapshots";
    let (s, v) = w.req(&o, Method::Post, snap, Json::Null);
    assert_eq!(s, 202, "{v}");
    let id = v["snapshot_id"].as_str().unwrap().to_string();
    let epoch = v["epoch"].as_u64().unwrap();
    // One row over the budget.
    let extra = json!({ "vectors": [{ "id": "extra", "values": Rng(9).vec(DIM) }] });
    let up = "/v1/collections/big/vectors:upsert";
    assert_eq!(w.req(&o, Method::Post, up, extra).0, 200);
    let objects = w.blob.keys("");
    let restore = format!("/v1/collections/big/snapshots/{id}:restore");
    for (path, what) in [
        (snap, "snapshot"),
        ("/v1/collections/big:export", "export"),
        (restore.as_str(), "restore"),
    ] {
        let (s, v) = w.req(&o, Method::Post, path, Json::Null);
        assert_eq!(s, 413, "{what}: {v}");
    }
    assert_eq!(w.blob.keys(""), objects, "no R2 side effect");
    // Back at the budget: no epoch was burned and no journal blocks restore.
    let del = "/v1/collections/big/vectors:delete";
    assert_eq!(
        w.req(&o, Method::Post, del, json!({ "ids": ["extra"] })).0,
        200
    );
    let (s, v) = w.req(&o, Method::Post, snap, Json::Null);
    assert_eq!(s, 202, "{v}");
    assert_eq!(v["epoch"].as_u64(), Some(epoch + 1));
    let (s, v) = w.req(&o, Method::Post, &restore, Json::Null);
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["rows"].as_u64(), Some(max as u64));
    let (s, v) = w.req(&o, Method::Post, "/v1/collections/big:export", Json::Null);
    assert_eq!(s, 201, "{v}");
}

/// A drop wipes an interrupted restore's staged rows with the shard.
#[test]
fn a_collection_drop_clears_staged_restore_rows() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "big", "dim": DIM, "metric": "cosine", "shards": 1 });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    seed(&w, &o, 10);
    // Any M3 shard call creates the staging tables; a snapshot pages them.
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/big/snapshots",
        Json::Null,
    );
    assert_eq!(s, 202, "{v}");
    let staged = |w: &World| -> usize {
        let shards = w.b.shards.borrow();
        let q = "SELECT id FROM m3_stage WHERE rid != ?";
        shards
            .values()
            .map(|s| s.query(q, &["".into()]).map(|r| r.len()).unwrap_or(0))
            .sum()
    };
    for s in w.b.shards.borrow().values() {
        let blob = ruvector_edge_store::Value::Blob(vec![0u8; DIM * 4]);
        let p = [
            "r1".into(),
            "x".into(),
            blob,
            ruvector_edge_store::Value::Null,
        ];
        s.exec(
            "INSERT OR REPLACE INTO m3_stage (rid, id, f32, metadata) VALUES (?, ?, ?, ?)",
            &p,
        )
        .unwrap();
    }
    assert_eq!(staged(&w), 1);
    let (s, v) = w.req(&o, Method::Delete, "/v1/collections/big", Json::Null);
    assert!(s < 300, "{s} {v}");
    assert_eq!(staged(&w), 0);
}

/// `POST /v1/collections` now always enters the M3 table first: M2 create
/// options pass through both the plain path and the embedder path.
#[test]
fn m2_index_options_survive_the_m3_create_interception() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let hnsw = json!({ "m": 16, "ef_construction": 100 });
    for body in [
        json!({ "name": "plain", "dim": 8, "metric": "cosine", "index": "hnsw", "hnsw": hnsw }),
        json!({ "name": "text", "metric": "cosine", "embedder": "bge-small-en-v1.5",
                "index": "hnsw", "hnsw": hnsw }),
    ] {
        let (s, v) = w.req(&o, Method::Post, "/v1/collections", body);
        assert_eq!(s, 201, "{v}");
    }
    for name in ["plain", "text"] {
        let path = format!("/v1/collections/{name}");
        let v = w.req(&o, Method::Get, &path, Json::Null).1;
        assert_eq!(v["index"], "hnsw", "{name}: {v}");
    }
}
