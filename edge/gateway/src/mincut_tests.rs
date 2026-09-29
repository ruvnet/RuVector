//! M4 rv-mincut end to end (ADR-351 §15 M4): inline answers (approximate →
//! `mode: exact`), graph-sourced cuts in node ids, a 50k-edge job through
//! the `AnalyticsJob` alarm path equal to the native crate, jobs needing
//! write, cross-tenant job `404`, and every exhausted budget `413`.

use crate::e2e_world::World;
use crate::graph_tests::owner_world;
use crate::testkit::{sub, tenant, Rng, T0};
use ruvector_edge_analytics::service::{query, Profile};
use ruvector_edge_analytics::{GraphLimits, QueryMode, TenantGraph};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};
use std::time::Instant;
use worker::Method;

fn native(edges: &[(u64, u64, f64)]) -> String {
    let g = TenantGraph::from_edges([0; 16], 0, edges, &GraphLimits::JOB).unwrap();
    query(&g, &QueryMode::Exact, &Profile::JOB)
        .unwrap()
        .digest_hex()
}

fn body(edges: &[(u64, u64, f64)]) -> Json {
    Json::Array(edges.iter().map(|(u, v, w)| json!([u, v, w])).collect())
}

fn post(w: &World, tok: &str, b: Json) -> (u16, Json) {
    let r = w.send(tok, Method::Post, "/v1/mincut", b, None);
    (
        r.status,
        serde_json::from_str(&r.body).unwrap_or(Json::Null),
    )
}

#[test]
fn inline_cuts_edges_and_graphs() {
    let (w, tok) = owner_world();
    // Two triangles joined by a weight-1 bridge.
    let edges = [
        (0, 1, 3.0),
        (1, 2, 3.0),
        (0, 2, 3.0),
        (3, 4, 3.0),
        (4, 5, 3.0),
        (3, 5, 3.0),
        (2, 3, 1.0),
    ];
    let (s, v) = post(
        &w,
        &tok,
        json!({ "edges": body(&edges), "mode": "approximate", "epsilon": 0.1 }),
    );
    assert_eq!(s, 200, "{v}");
    assert_eq!(
        (v["mode"].clone(), v["requested_mode"].clone()),
        (json!("exact"), json!("approximate"))
    );
    assert_eq!(v["value"], json!(1.0));
    assert_eq!(v["digest"], json!(native(&edges)));
    assert_eq!(
        post(
            &w,
            &tok,
            json!({ "edges": body(&edges), "mode": "approximate", "epsilon": 2.0 })
        )
        .0,
        400
    );
    assert_eq!(
        post(&w, &tok, json!({ "edges": body(&edges), "graph": "g" })).0,
        400
    );

    // Over a stored graph: the partition is in node ids.
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "g" }));
    let named = json!([
        ["a", "b", 3.0],
        ["b", "c", 3.0],
        ["a", "c", 3.0],
        ["c", "d", 0.5]
    ]);
    w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/g/edges",
        json!({ "edges": named }),
    );
    let (s, v) = post(&w, &tok, json!({ "graph": "g" }));
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["value"], json!(0.5));
    let sides = v["partition"].as_array().unwrap();
    assert!(sides.iter().any(|s| s == &json!(["d"])), "{v}");
    assert_eq!(post(&w, &tok, json!({ "graph": "nope" })).0, 404);

    // A viewer computes inline (read); MCP `mincut` too.
    w.b.invite(
        &tenant("org-g"),
        &sub("alice"),
        &sub("val"),
        Role::Viewer,
        T0,
    )
    .unwrap();
    let val = w
        .edge_as
        .user_token("val", "org-g", &w.v1(), &["ruvector:read"]);
    assert_eq!(post(&w, &val, json!({ "graph": "g" })).0, 200);
    let vm = w
        .edge_as
        .user_token("val", "org-g", &w.mcp(), &["ruvector:read"]);
    let call = json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": { "name": "mincut", "arguments": { "edges": body(&edges) } } });
    let r: Json =
        serde_json::from_str(&w.send(&vm, Method::Post, "/v1/mcp", call, None).body).unwrap();
    assert_eq!(r["result"]["structuredContent"]["value"], json!(1.0), "{r}");
}

/// K_320 with random integer weights: 51,040 edges, no certificate, so
/// Stoer–Wagner runs — far over the inline budget, inside the job's.
fn k320() -> Vec<(u64, u64, f64)> {
    let mut rng = Rng(0x320);
    let mut e = Vec::new();
    for u in 0..320u64 {
        for v in u + 1..320 {
            e.push((u, v, (1 + rng.next_u64() % 9) as f64));
        }
    }
    e
}

#[test]
fn fifty_k_edges_run_as_a_job_and_equal_native() {
    let (w, tok) = owner_world();
    let edges = k320();
    assert!(edges.len() >= 50_000);
    let b = json!({ "edges": body(&edges) });
    assert!(b.to_string().len() < crate::api::MAX_BODY_BYTES);

    // A viewer cannot queue a job (job = write).
    w.b.invite(
        &tenant("org-g"),
        &sub("alice"),
        &sub("val"),
        Role::Viewer,
        T0,
    )
    .unwrap();
    let val = w.edge_as.user_token(
        "val",
        "org-g",
        &w.v1(),
        &["ruvector:read", "ruvector:write"],
    );
    assert_eq!(post(&w, &val, b.clone()).0, 403);

    // The harness encodes the body before the gateway parses it; report
    // that share separately.
    let t = Instant::now();
    let encoded = b.to_string().len();
    let encode_ms = t.elapsed().as_secs_f64() * 1e3;
    let t = Instant::now();
    let (s, v) = post(&w, &tok, b);
    let submit_ms = t.elapsed().as_secs_f64() * 1e3 - encode_ms;
    assert_eq!((s, v["state"].clone()), (202, json!("queued")), "{v}");
    let id = v["job_id"].as_str().unwrap().to_string();
    let path = format!("/v1/mincut/jobs/{id}");
    assert_eq!(
        w.ok(&tok, Method::Get, &path, Json::Null)["state"],
        json!("queued")
    );

    // Turn 1 (admit: canonicalise, persist chunks, pin), turn 2 (solve).
    let t = Instant::now();
    assert_eq!(w.b.m4.run_alarms(), 1);
    let admit_ms = t.elapsed().as_secs_f64() * 1e3;
    let mid = w.ok(&tok, Method::Get, &path, Json::Null);
    assert_eq!(mid["state"], json!("queued"), "{mid}");
    let t = Instant::now();
    assert_eq!(w.b.m4.run_alarms(), 1);
    let run_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(w.b.m4.run_alarms(), 0);
    let done = w.ok(&tok, Method::Get, &path, Json::Null);
    assert_eq!(done["state"], json!("done"), "{done}");
    let t = Instant::now();
    let want = native(&edges);
    let native_ms = t.elapsed().as_secs_f64() * 1e3;
    assert_eq!(done["result"]["digest"], json!(want));
    assert_eq!(done["result"]["mode"], json!("exact"));
    eprintln!(
        "mincut K_320 ({} edges, {encoded} B body): submit request path {submit_ms:.1} ms, admit turn {admit_ms:.1} ms, \
         solve turn {run_ms:.1} ms (native crate {native_ms:.1} ms); estimate {}",
        edges.len(),
        done["estimate"]
    );

    // Another tenant cannot see the job.
    let bob = w.edge_as.user_token(
        "bob",
        "org-other",
        &w.v1(),
        &["ruvector:read", "ruvector:write"],
    );
    assert_eq!(
        w.send(&bob, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    assert_eq!(
        w.send(&bob, Method::Get, &path, Json::Null, None).status,
        404
    );
    assert_eq!(
        w.send(
            &tok,
            Method::Get,
            "/v1/mincut/jobs/mc_0000000000000000",
            Json::Null,
            None
        )
        .status,
        404
    );
}

#[test]
fn exhausted_budgets_are_413_never_500() {
    let (w, tok) = owner_world();
    // Over the job edge limit: refused before parsing an edge.
    let over = vec![json!([0, 1]); 250_001];
    let (s, v) = post(&w, &tok, json!({ "edges": over }));
    assert_eq!(
        (s, v["code"].clone()),
        (413, json!("payload_too_large")),
        "{v}"
    );
    // Admissible by edge count, but a 250,001-vertex star exceeds the job's
    // vertex limit: the job fails with a 413 code, not a server error.
    let star: Vec<Json> = (1..=250_000u64).map(|i| json!([0, i])).collect();
    let (s, v) = post(&w, &tok, json!({ "edges": star }));
    assert_eq!(s, 202, "{v}");
    let path = format!("/v1/mincut/jobs/{}", v["job_id"].as_str().unwrap());
    w.b.m4.drain_alarms(4);
    let j = w.ok(&tok, Method::Get, &path, Json::Null);
    assert_eq!(
        (j["state"].clone(), j["error"]["status"].clone()),
        (json!("failed"), json!(413)),
        "{j}"
    );
    // The MCP tool is inline only: a job-sized cut is 413 budget_exceeded.
    let om = w
        .edge_as
        .user_token("alice", "org-g", &w.mcp(), &["ruvector:read"]);
    let call = json!({ "jsonrpc": "2.0", "id": 3, "method": "tools/call",
        "params": { "name": "mincut", "arguments": { "edges": body(&k320()[..20_000]) } } });
    let r: Json =
        serde_json::from_str(&w.send(&om, Method::Post, "/v1/mcp", call, None).body).unwrap();
    let text = r["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string();
    assert!(
        text.contains("budget_exceeded") && text.contains("413"),
        "{text}"
    );
}

#[test]
fn probes_create_nothing_and_stalled_jobs_fail_413() {
    use crate::graph_wire::{JobCall, JobRequest};
    use crate::mincut_job::{self, STALL_MS};
    use ruvector_edge_store::{MemSqlStore, SqlStore};
    let (w, tok) = owner_world();
    // Unknown graph / job names are 404 and leave no DO storage behind.
    let before = (
        w.b.m4.graph_stores.borrow().len(),
        w.b.m4.job_stores.borrow().len(),
    );
    assert_eq!(
        w.send(&tok, Method::Get, "/v1/graphs/ghost", Json::Null, None)
            .status,
        404
    );
    let probe = "/v1/mincut/jobs/mc_00000000deadbeef";
    assert_eq!(
        w.send(&tok, Method::Get, probe, Json::Null, None).status,
        404
    );
    let untouched =
        |stores: &std::cell::RefCell<std::collections::BTreeMap<String, MemSqlStore>>| {
            stores.borrow().values().all(|s| {
                s.query("SELECT k, v FROM gmeta", &[]).is_err()
                    && s.query("SELECT k, v FROM jmeta", &[]).is_err()
            })
        };
    assert!(
        untouched(&w.b.m4.graph_stores) && untouched(&w.b.m4.job_stores),
        "{before:?}"
    );

    // A job whose solve never completes (its attempt write lost with the
    // killed isolate) is failed on read after STALL_MS: 413, not stuck.
    let st = MemSqlStore::new();
    let t = tenant("org-g");
    let req = |call| {
        let r = JobRequest {
            tenant_key: t.as_str().into(),
            job_id: "mc_1".into(),
            call,
        };
        serde_json::to_vec(&r).unwrap()
    };
    let submit = JobCall::Submit {
        mode: QueryMode::Exact,
        edges: "[[0,1,1.0],[1,2,1.0]]".into(),
        labels: None,
        graph: None,
        revision: 0,
        now_ms: 1_000,
    };
    let (_, alarm) = mincut_job::serve(None, &st, &req(submit));
    assert!(alarm.is_some());
    // Admission only; the solve turn "dies" (never runs).
    let _ = mincut_job::prepare(&st, 2_000);
    let get = |now_ms| {
        let (out, _) = mincut_job::serve(None, &st, &req(JobCall::Get { now_ms }));
        serde_json::from_str::<Json>(&out).unwrap()["Ok"]["view"].clone()
    };
    assert_eq!(get(2_000 + STALL_MS)["state"], json!("queued"));
    let v = get(2_001 + STALL_MS);
    assert_eq!(
        (v["state"].clone(), v["error"]["status"].clone()),
        (json!("failed"), json!(413)),
        "{v}"
    );
    assert_eq!(mincut_job::alarm(&st, 3_000 + STALL_MS), None);
}
