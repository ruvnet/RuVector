//! M5 acceptance (ADR-351 §15): an RVF exported by the rvf CLI imports via
//! `upload_id`, is imported into a collection, and queries correctly; plus
//! the MCP registry tools.

use crate::mcp;
use crate::registry_world::*;
use crate::rvf_mcp::RvfTools;
use crate::service;
use crate::testkit::*;
use ruvector_edge_store::{CallerContext, Op};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};
use worker::Method;

fn create(w: &World, c: &CallerContext, name: &str, dim: u32, metric: &str) {
    let spec = json!({ "name": name, "dim": dim, "metric": metric, "shards": 2 });
    block_on(service::execute(
        &call(&w.b, c),
        Op::CollectionCreate,
        &spec.to_string(),
    ))
    .unwrap();
}

fn query(w: &World, c: &CallerContext, coll: &str, q: &[f32]) -> Vec<String> {
    let args = json!({ "collection": coll, "vector": q, "top_k": 10 }).to_string();
    let (r, _) = block_on(service::execute(&call(&w.b, c), Op::VectorQuery, &args)).unwrap();
    r["matches"]
        .as_array()
        .unwrap()
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

fn import(w: &World, c: &CallerContext, coll: &str, pkg: &str) -> (u16, Json) {
    let body = json!({ "package": pkg, "version": "1.0.0" });
    w.json(
        c,
        Method::Post,
        &format!("/v1/collections/{coll}:import-rvf"),
        body,
    )
}

/// M5 acceptance: rvf CLI export → multipart upload → finalize → import
/// into a collection → exact top-k equal to the engine's own answers.
#[test]
fn rvf_cli_export_imports_via_upload_id_and_queries_correctly() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    assert_eq!(w.claim(&alice, "acme").0, 200);
    let (bytes, dir) = rvf_export(300, 32, 5, &[1, 2, 3]);
    let (s, m) = w.push(&alice, "acme/export", "1.0.0", "tenant", &bytes, 4);
    assert_eq!(s, 201, "{m}");
    create(&w, &alice, "docs", 32, "cosine");
    let (s, r) = import(&w, &alice, "docs", "@acme/export");
    assert_eq!(s, 200, "{r}");
    assert_eq!(r["imported"], json!(297));
    let engine = rvf_runtime::RvfStore::open_readonly(&dir.path().join("export.rvf")).unwrap();
    for (_, q) in rows(8, 32, 77) {
        let opts = rvf_runtime::options::QueryOptions {
            force_exact: true,
            ..Default::default()
        };
        let theirs: Vec<String> = engine
            .query(&q, 10, &opts)
            .unwrap()
            .iter()
            .map(|r| r.id.to_string())
            .collect();
        assert_eq!(query(&w, &alice, "docs", &q), theirs);
    }
    // Deleted ids never arrive; re-import is idempotent by id.
    let fetch = json!({ "collection": "docs", "ids": ["1", "4"] }).to_string();
    let (f, _) = block_on(service::execute(
        &call(&w.b, &alice),
        Op::VectorFetch,
        &fetch,
    ))
    .unwrap();
    assert_eq!(f["vectors"].as_array().unwrap().len(), 1);
    assert_eq!(
        import(&w, &alice, "docs", "@acme/export").1["imported"],
        json!(297)
    );
}

#[test]
fn import_checks_shape_visibility_and_write_access() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    w.claim(&alice, "acme");
    let bytes = rvf_export(20, 8, 3, &[]).0;
    assert_eq!(
        w.push(&alice, "acme/p", "1.0.0", "private", &bytes, 2).0,
        201
    );
    create(&w, &alice, "wide", 16, "cosine");
    create(&w, &alice, "flat", 8, "l2");
    create(&w, &alice, "docs", 8, "cosine");
    let (s, b) = import(&w, &alice, "wide", "@acme/p");
    assert_eq!((s, b["code"].as_str()), (400, Some("dimension_mismatch")));
    assert_eq!(import(&w, &alice, "flat", "@acme/p").0, 400);
    assert_eq!(import(&w, &alice, "missing", "@acme/p").0, 404);
    // Read-only member: the collection write is refused first.
    let viewer = w.member(&alice, "vic", Role::Viewer, ro());
    assert_eq!(import(&w, &viewer, "docs", "@acme/p").0, 403);
    // Another tenant cannot import a private package (404), but can import
    // it once public.
    let bob = w.owner("org-b", "bob");
    create(&w, &bob, "docs", 8, "cosine");
    assert_eq!(import(&w, &bob, "docs", "@acme/p").0, 404);
    let (s, _) = w.json(
        &alice,
        Method::Post,
        "/v1/rvf/acme/p/1.0.0:publish",
        Json::Null,
    );
    assert_eq!(s, 200);
    let (s, r) = import(&w, &bob, "docs", "@acme/p");
    assert_eq!((s, r["imported"].clone()), (200, json!(20)));
}

fn rpc(w: &World, c: &CallerContext, method: &str, params: Json) -> crate::rest::ApiReply {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": method, "params": params });
    let d = w.deps();
    let tools = RvfTools(&d);
    block_on(mcp::handle_with(
        &w.b,
        &tools,
        c,
        body.to_string().as_bytes(),
        T0,
        MCP_MD,
    ))
}

fn result(r: &crate::rest::ApiReply) -> Json {
    serde_json::from_str::<Json>(&r.body).unwrap()["result"].clone()
}

#[test]
fn mcp_registry_tools_read_and_import_but_never_publish() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    w.claim(&alice, "acme");
    let bytes = rvf_export(20, 8, 3, &[]).0;
    assert_eq!(
        w.push(&alice, "acme/p", "1.0.0", "tenant", &bytes, 1).0,
        201
    );
    create(&w, &alice, "docs", 8, "cosine");
    let list = result(&rpc(&w, &alice, "tools/list", json!({})));
    let names: Vec<&str> = list["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for t in ["rvf_list", "rvf_get", "rvf_import", "vector_query"] {
        assert!(names.contains(&t), "{t}");
    }
    assert!(!names.iter().any(|n| n.contains("publish")));
    let call = |c: &CallerContext, name: &str, args: Json| {
        rpc(
            &w,
            c,
            "tools/call",
            json!({ "name": name, "arguments": args }),
        )
    };
    let r = call(&alice, "rvf_list", json!({}));
    assert_eq!(
        result(&r)["structuredContent"],
        json!({ "scopes": ["acme"] })
    );
    let r = call(&alice, "rvf_list", json!({ "package": "@acme/p" }));
    assert_eq!(
        result(&r)["structuredContent"]["items"][0]["version"],
        json!("1.0.0")
    );
    let r = call(
        &alice,
        "rvf_get",
        json!({ "package": "@acme/p", "version": "1.0.0" }),
    );
    assert_eq!(result(&r)["structuredContent"]["dim"], json!(8));
    let args = json!({ "collection": "docs", "package": "@acme/p", "version": "1.0.0" });
    let r = call(&alice, "rvf_import", args.clone());
    assert_eq!(result(&r)["structuredContent"]["imported"], json!(20));
    // Import is a write: a read-only token gets the HTTP 403 step-up.
    let reader = w.member(&alice, "rita", Role::Editor, ro());
    let r = call(&reader, "rvf_import", args);
    assert_eq!(r.status, 403);
    assert!(r.www_authenticate.unwrap().contains("ruvector:write"));
    // Not found is a tool error; publish does not exist on MCP.
    let r = call(
        &alice,
        "rvf_get",
        json!({ "package": "@acme/p", "version": "9.0.0" }),
    );
    assert_eq!(result(&r)["isError"], json!(true));
    let r = call(
        &alice,
        "rvf_publish",
        json!({ "package": "@acme/p", "version": "1.0.0" }),
    );
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["error"]["code"], json!(-32602));
}

fn import_window(w: &World, c: &CallerContext, coll: &str, offset: u64, budget: u64) -> Json {
    let rc = block_on(crate::registry_routes::registry_caller(&w.b, c)).unwrap();
    let body = json!({ "package": "@acme/big", "version": "1.0.0", "offset": offset });
    let now = ruvector_edge_auth::Clock::now_unix(&w.r.clock);
    let (s, r) = block_on(crate::rvf_import::import_budgeted(
        &w.deps(),
        c,
        &rc,
        coll,
        body.to_string().as_bytes(),
        now,
        budget,
    ))
    .unwrap();
    assert_eq!(s, 200);
    r
}

/// F3: one request imports a bounded window (by subrequest budget) and
/// answers `next_offset`; repeating with it completes the import, and the
/// collection then answers exactly like the engine.
#[test]
fn import_is_bounded_per_request_and_resumes_from_next_offset() {
    let w = World::new();
    let alice = w.owner("org-a", "alice");
    w.claim(&alice, "acme");
    let (bytes, dir) = rvf_export(1200, 4, 9, &[7]);
    assert_eq!(
        w.push(&alice, "acme/big", "1.0.0", "tenant", &bytes, 3).0,
        201
    );
    create(&w, &alice, "docs", 4, "cosine");
    // 2 shards: 9 subrequests per batch, so a budget of 9 is one batch.
    assert_eq!(crate::rvf_import::rows_per_request(9, 2), 500);
    let (mut offset, mut total, mut windows) = (Some(0u64), 0u64, Vec::new());
    while let Some(o) = offset {
        let r = import_window(&w, &alice, "docs", o, 9);
        assert!(r["imported"].as_u64().unwrap() <= 500);
        total += r["imported"].as_u64().unwrap();
        windows.push(o);
        offset = r["next_offset"].as_u64();
    }
    assert_eq!(windows, vec![0, 500, 1000]);
    assert_eq!(total, 1199);
    let engine = rvf_runtime::RvfStore::open_readonly(&dir.path().join("export.rvf")).unwrap();
    for (_, q) in rows(4, 4, 31) {
        let opts = rvf_runtime::options::QueryOptions {
            force_exact: true,
            ..Default::default()
        };
        let theirs: Vec<String> = engine
            .query(&q, 10, &opts)
            .unwrap()
            .iter()
            .map(|r| r.id.to_string())
            .collect();
        assert_eq!(query(&w, &alice, "docs", &q), theirs);
    }
    // An offset past the end imports nothing and is done.
    let r = import_window(&w, &alice, "docs", 5000, 9);
    assert_eq!(
        (r["imported"].clone(), r["next_offset"].clone()),
        (json!(0), Json::Null)
    );
}

/// F3: the index reproduces the reference decoder (deletions, last record
/// per id wins, id order) and refuses record counts over the cap before
/// indexing anything.
#[test]
fn live_index_matches_the_reference_decoder_and_caps_records() {
    use rvf_runtime::options::{DistanceMetric, RvfOptions};
    let dir = tempfile::tempdir().unwrap();
    let path = dir.path().join("dup.rvf");
    let mut store = rvf_runtime::RvfStore::create(
        &path,
        RvfOptions {
            dimension: 4,
            metric: DistanceMetric::Cosine,
            ..Default::default()
        },
    )
    .unwrap();
    for (seed, ids) in [(1u64, 0..40u64), (2, 10..20)] {
        let data: Vec<(u64, Vec<f32>)> = rows(40, 4, seed)
            .into_iter()
            .zip(ids)
            .map(|((_, v), id)| (id, v))
            .collect();
        let vecs: Vec<&[f32]> = data.iter().map(|(_, v)| v.as_slice()).collect();
        let ids: Vec<u64> = data.iter().map(|(i, _)| *i).collect();
        store.ingest_batch(&vecs, &ids, None).unwrap();
    }
    store.delete(&[3, 12]).unwrap();
    store.close().unwrap();
    let bytes = std::fs::read(&path).unwrap();
    let limits = ruvector_edge_registry::ValidationLimits::default();
    let v = ruvector_edge_registry::validate::validate(&bytes, limits).unwrap();
    let ours = crate::rvf_import::live_rows(&v, &bytes, 1_000).unwrap();
    assert_eq!(ours, v.live_vectors(&bytes).unwrap());
    assert_eq!(ours.len(), 38);
    let e = crate::rvf_import::live_rows(&v, &bytes, 10).unwrap_err();
    assert_eq!(e.status, 413);
}

/// Regression (an import window upserted ~57k vectors for one write
/// token): every upsert batch beyond the first pays a §10 write token;
/// over budget the window stops with `rate_limited` and a resumable
/// `next_offset`.
#[test]
fn import_charges_one_write_token_per_upsert_batch() {
    let w = World {
        b: crate::backend::mem::MemBackend::new().with_fanout_limiter(),
        ..World::new()
    };
    let alice = w.owner("org-a", "alice");
    w.claim(&alice, "acme");
    // 6500 rows = 13 batches of 500; the user write budget is 10 / 10 s.
    let bytes = rvf_export(6500, 8, 9, &[]).0;
    assert_eq!(
        w.push(&alice, "acme/big", "1.0.0", "tenant", &bytes, 2).0,
        201
    );
    create(&w, &alice, "docs", 8, "cosine");
    let (s, r) = import(&w, &alice, "docs", "@acme/big");
    assert_eq!(s, 200, "{r}");
    // Batch 1 rides on the request's token, batches 2-11 pay 10 tokens.
    assert_eq!(
        (&r["imported"], &r["next_offset"], &r["rate_limited"]),
        (&json!(5500), &json!(5500), &json!(true))
    );
    let limiter = w.b.fanout_limiter.as_ref().unwrap();
    let writes: u32 = limiter
        .seen
        .borrow()
        .iter()
        .filter(|((b, _), _)| b == "RL_WRITE_USER")
        .map(|(_, n)| *n)
        .sum();
    assert_eq!(writes, 11, "10 admitted + 1 refused");
    // Next window: resume at next_offset.
    limiter.reset();
    let body = json!({ "package": "@acme/big", "version": "1.0.0", "offset": 5500 });
    let (s, r) = w.json(
        &alice,
        Method::Post,
        "/v1/collections/docs:import-rvf",
        body,
    );
    assert_eq!(s, 200, "{r}");
    assert_eq!(
        (&r["imported"], &r["next_offset"], &r["rate_limited"]),
        (&json!(1000), &Json::Null, &json!(false))
    );
}
