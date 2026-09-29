//! M4 merge review fixes (ADR-351 §3, §7, §10): the isolate memory budget
//! including `AnalyticsJob` and the M5 registry, job admission (memory
//! precheck, write token, live-job cap), job retention, graph delete and
//! `Idempotency-Key` on graph mutations.

use crate::backend::mem::MemBackend;
use crate::e2e_world::World;
use crate::graph_catalog::{MAX_GRAPHS, MAX_LIVE_JOBS};
use crate::graph_store;
use crate::graph_tests::owner_world;
use crate::graph_wire::{graph_do_name, GraphCall, GraphRequest, CATALOG};
use crate::mincut_core::{self as mc, EDGE_JOB, JOB_MEMORY_BYTES};
use crate::mincut_job::JOB_TTL_MS;
use crate::testkit::{sub, tenant, T0};
use ruvector_edge_analytics::service::plan;
use ruvector_edge_analytics::{GraphLimits, QueryMode, TenantGraph};
use ruvector_edge_store::{ErrorCode, MemSqlStore};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};
use worker::Method;

const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];
/// The Workers isolate memory limit.
const ISOLATE_MEMORY_BYTES: u64 = 128_000_000;
// The job profile is the crate's with a smaller memory budget.
const _: () = assert!(EDGE_JOB.budget.max_memory_bytes < 100 << 20);

fn send(w: &World, tok: &str, m: Method, path: &str, body: Json, key: Option<&str>) -> (u16, Json) {
    let r = w.send(tok, m, path, body, key);
    (
        r.status,
        serde_json::from_str(&r.body).unwrap_or(Json::Null),
    )
}

/// A job-sized (over the inline limit), memory- and work-admissible edge
/// list: the complete graph on 300 vertices (44,850 edges; a sparse path
/// of that many edges would exceed the job's Stoer–Wagner work budget).
fn job_edges(seed: u64) -> Json {
    let k = 300u64;
    let edges: Vec<Json> = (0..k)
        .flat_map(|u| (u + 1..k).map(move |v| json!([u + seed, v + seed])))
        .collect();
    assert!(edges.len() as u64 > mc::INLINE_MAX_EDGES);
    Json::Array(edges)
}

#[test]
fn isolate_budget_counts_jobs_and_registry_parts() {
    let resident = crate::shard_core::VECTOR_RESIDENT_CAP_BYTES
        + crate::quant_shard::QUANT_RESIDENT_CAP_BYTES
        + graph_store::GRAPH_RESIDENT_CAP_BYTES;
    assert_eq!(resident, 56_000_000);
    let part = crate::registry_core::gateway_config().upload.max_part_size;
    assert_eq!(part, 8 << 20);
    // One synchronous analytics turn: a job, or (Workers Paid) the
    // request-path min-cut, never both (neither awaits).
    let analytics = JOB_MEMORY_BYTES.max(mc::EDGE_INLINE.budget.max_memory_bytes);
    assert_eq!(analytics, JOB_MEMORY_BYTES);
    // Workers Paid: the 8 MiB gateway transfers (JS + wasm copies).
    let transfer = [
        crate::uploads::MAX_INLINE_BYTES as u64,
        crate::uploads::MAX_PART_BODY as u64,
        crate::ingest_hash::HASH_BYTES_PER_DELIVERY,
        crate::ingest::TAIL_BYTES,
        crate::ingest::QUEUED_MAX_SEGMENT_PAYLOAD,
    ]
    .into_iter()
    .max()
    .unwrap();
    assert!(transfer <= (8 << 20) + 128, "{transfer}");
    // Resident caps, one analytics turn, one registry part and one 8 MiB
    // transfer (each part / transfer with its JS + wasm copy).
    let total = resident + analytics + 2 * part + 2 * transfer;
    assert!(total <= ISOLATE_MEMORY_BYTES, "{total}");
    assert!(ISOLATE_MEMORY_BYTES - total >= 4_000_000, "{total}");

    // The 51k-edge K_320 acceptance job still fits the job budget.
    let k320: Vec<(u64, u64, f64)> = (0..320u64)
        .flat_map(|u| (u + 1..320).map(move |v| (u, v, 1.0 + (u * v % 9) as f64)))
        .collect();
    let g = TenantGraph::from_edges([0; 16], 0, &k320, &GraphLimits::JOB).unwrap();
    let est = plan(&g, &QueryMode::Exact, &EDGE_JOB).unwrap();
    assert!(est.memory_bytes <= JOB_MEMORY_BYTES, "{est:?}");
}

/// The budget counts an 8 MiB body once plus its JS copy: `read_capped`
/// never reserves past its cap, even when odd-sized chunks would make
/// `Vec` doubling overshoot to ≈ 16 MB.
#[test]
fn eight_mib_bodies_never_reserve_past_the_cap() {
    use futures_util::stream;
    let max = crate::uploads::MAX_INLINE_BYTES;
    let chunks: Vec<Result<Vec<u8>, ()>> = (0..max)
        .step_by(4000)
        .map(|at| Ok(vec![1u8; 4000.min(max - at)]))
        .collect();
    let body = crate::testkit::block_on(crate::api::read_capped(stream::iter(chunks), max))
        .unwrap()
        .unwrap();
    assert_eq!(body.len(), max);
    assert!(body.capacity() <= max, "{}", body.capacity());
}

#[test]
fn a_job_over_the_memory_budget_is_refused_before_anything_is_stored() {
    let (w, tok) = owner_world();
    // 80k edges × 420 B ≥ 33.6 MB > 32 MiB; the body stays under 1 MiB.
    let edges: Vec<Json> = (0..80_000u64)
        .map(|i| json!([i % 997, 1 + i % 991]))
        .collect();
    let b = json!({ "edges": edges });
    assert!(b.to_string().len() < crate::api::MAX_BODY_BYTES);
    let (s, v) = send(&w, &tok, Method::Post, "/v1/mincut", b, None);
    assert_eq!(
        (s, v["code"].clone()),
        (413, json!("budget_exceeded")),
        "{v}"
    );
    assert!(w.b.m4.job_stores.borrow().is_empty());
    assert!(w.b.m4.alarms.borrow().is_empty());
    // The admit turn's streaming parser keeps the old validation.
    assert!(mc::parse_edges("[[0,1],[1,2,0.5]]").is_ok());
    for bad in [
        "[[0,1.5]]",
        "[[0]]",
        "[[0,1,2,3]]",
        "[[0,\"x\"]]",
        "{}",
        "[[0,-1]]",
    ] {
        assert!(mc::parse_edges(bad).is_err(), "{bad}");
    }
}

#[test]
fn a_queued_job_pays_a_write_token_and_the_live_job_cap() {
    let w = World {
        b: MemBackend::new().with_fanout_limiter(),
        ..World::new("ruvector:read ruvector:write")
    };
    let tok = w.edge_as.user_token("alice", "org-g", &w.v1(), &RW);
    assert_eq!(
        w.send(&tok, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    let limiter = w.b.fanout_limiter.as_ref().unwrap();
    let writes = || -> u32 {
        let seen = limiter.seen.borrow();
        seen.iter()
            .filter(|((b, _), _)| b == "RL_WRITE_USER")
            .map(|(_, n)| *n)
            .sum()
    };
    // An inline answer is a read: no write token.
    let (s, _) = send(
        &w,
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": [[0, 1]] }),
        None,
    );
    assert_eq!((s, writes()), (200, 0));
    // A job pays one; the user write budget (10 / 10 s) then refuses: 429.
    for i in 0..10u64 {
        let (s, v) = send(
            &w,
            &tok,
            Method::Post,
            "/v1/mincut",
            json!({ "edges": job_edges(i) }),
            None,
        );
        assert_eq!(s, 202, "{i}: {v}");
    }
    assert_eq!(writes(), 10);
    let (s, v) = send(
        &w,
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": job_edges(99) }),
        None,
    );
    assert_eq!((s, v["code"].clone()), (429, json!("rate_limited")), "{v}");
    assert_eq!(w.b.m4.job_stores.borrow().len(), 10);

    // The live-job cap: at MAX_LIVE_JOBS the next job is 413.
    for i in 10..MAX_LIVE_JOBS as u64 {
        limiter.reset();
        let (s, v) = send(
            &w,
            &tok,
            Method::Post,
            "/v1/mincut",
            json!({ "edges": job_edges(i) }),
            None,
        );
        assert_eq!(s, 202, "{i}: {v}");
    }
    limiter.reset();
    let (s, v) = send(
        &w,
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": job_edges(77) }),
        None,
    );
    assert_eq!(
        (s, v["code"].clone()),
        (413, json!("quota_exceeded")),
        "{v}"
    );
    assert_eq!(w.b.m4.job_stores.borrow().len(), MAX_LIVE_JOBS);
}

#[test]
fn the_catalog_frees_a_job_slot_at_its_expiry() {
    let store = MemSqlStore::new();
    let mut host = graph_store::GraphHost::default();
    let t = tenant("org-g");
    let name = graph_do_name(&t, CATALOG);
    let mut call = |job: &str, now_ms: u64| {
        let req = GraphRequest {
            tenant_key: t.as_str().into(),
            graph: CATALOG.into(),
            call: GraphCall::JobAdd {
                job_id: job.into(),
                now_ms,
                expires_ms: now_ms + JOB_TTL_MS,
            },
        };
        graph_store::handle(&mut host, "k", Some(name.as_str()), &store, req)
    };
    for i in 0..MAX_LIVE_JOBS {
        call(&format!("mc_{i}"), 1_000).unwrap();
    }
    assert_eq!(call("mc_x", 2_000).unwrap_err().code.status(), 413);
    // At the jobs' expiry their rows are pruned and a new job fits.
    call("mc_y", 1_000 + JOB_TTL_MS).unwrap();
}

#[test]
fn a_job_deletes_itself_at_its_expiry() {
    let (w, tok) = owner_world();
    let (s, v) = send(
        &w,
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": job_edges(0) }),
        None,
    );
    assert_eq!(s, 202, "{v}");
    let path = format!("/v1/mincut/jobs/{}", v["job_id"].as_str().unwrap());
    w.b.m4.drain_alarms(4);
    assert_eq!(
        w.ok(&tok, Method::Get, &path, Json::Null)["state"],
        json!("done")
    );
    // Only the retention alarm remains, not due yet.
    assert_eq!(w.b.m4.run_alarms(), 0);
    assert_eq!(w.b.m4.later.borrow().len(), 1);
    w.b.m4.advance(JOB_TTL_MS - 60_000);
    assert_eq!(w.b.m4.run_alarms(), 0);
    w.b.m4.advance(60_000);
    assert_eq!(w.b.m4.run_alarms(), 1);
    assert_eq!(send(&w, &tok, Method::Get, &path, Json::Null, None).0, 404);
    assert!(w.b.m4.later.borrow().is_empty() && w.b.m4.alarms.borrow().is_empty());
}

#[test]
fn graphs_can_be_deleted_and_their_names_reused() {
    let (w, tok) = owner_world();
    for i in 0..MAX_GRAPHS {
        w.ok(
            &tok,
            Method::Post,
            "/v1/graphs",
            json!({ "name": format!("g{i}") }),
        );
    }
    let (s, v) = send(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs",
        json!({ "name": "more" }),
        None,
    );
    assert_eq!((s, v["code"].clone()), (413, json!("quota_exceeded")));
    w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/g0/edges",
        json!({ "edges": [["a", "b", 2.0]] }),
    );
    let q = json!({ "query": "MATCH (n:Node) RETURN n" });
    assert_eq!(
        w.ok(&tok, Method::Post, "/v1/graphs/g0/cypher", q.clone())["rows"]
            .as_array()
            .unwrap()
            .len(),
        2
    );

    // A viewer cannot delete.
    w.b.invite(
        &tenant("org-g"),
        &sub("alice"),
        &sub("val"),
        Role::Viewer,
        T0,
    )
    .unwrap();
    let val = w.edge_as.user_token("val", "org-g", &w.v1(), &RW);
    assert_eq!(
        send(&w, &val, Method::Delete, "/v1/graphs/g0", Json::Null, None).0,
        403
    );

    let d = w.ok(&tok, Method::Delete, "/v1/graphs/g0", Json::Null);
    assert_eq!(d, json!({ "name": "g0", "deleted": true }));
    assert_eq!(
        send(&w, &tok, Method::Get, "/v1/graphs/g0", Json::Null, None).0,
        404
    );
    assert_eq!(
        send(&w, &tok, Method::Delete, "/v1/graphs/g0", Json::Null, None).0,
        404
    );
    let list = w.ok(&tok, Method::Get, "/v1/graphs", Json::Null);
    assert_eq!(list["graphs"].as_array().unwrap().len(), MAX_GRAPHS - 1);
    // The name is free again, and nothing of the old graph survives
    // (storage or the resident copy).
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "g0" }));
    let st = w.ok(&tok, Method::Get, "/v1/graphs/g0", Json::Null);
    assert_eq!(
        (st["nodes"].clone(), st["edges"].clone()),
        (json!(0), json!(0)),
        "{st}"
    );
    assert_eq!(
        w.ok(&tok, Method::Post, "/v1/graphs/g0/cypher", q)["rows"],
        json!([])
    );
    // An unknown graph is 404; another tenant's graph is absent.
    assert_eq!(
        send(
            &w,
            &tok,
            Method::Delete,
            "/v1/graphs/nope",
            Json::Null,
            None
        )
        .0,
        404
    );
}

#[test]
fn graph_mutations_honour_idempotency_keys() {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "g" }));
    let edges = json!({ "edges": [["a", "b", 1.0], ["b", "c", 2.0]] });
    let first = send(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/g/edges",
        edges.clone(),
        Some("k-1"),
    );
    assert_eq!(first.0, 200, "{}", first.1);
    let again = send(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/g/edges",
        edges,
        Some("k-1"),
    );
    assert_eq!(again, first);
    let st = w.ok(&tok, Method::Get, "/v1/graphs/g", Json::Null);
    assert_eq!(
        (st["edges"].clone(), st["revision"].clone()),
        (json!(2), json!(1)),
        "{st}"
    );
    // The same key on another body is a reuse.
    let other = json!({ "edges": [["x", "y"]] });
    assert_eq!(
        send(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs/g/edges",
            other,
            Some("k-1")
        )
        .0,
        409
    );

    let create = json!({ "query": "CREATE (n:Person {name: 'Dora'})" });
    let a = send(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/g/cypher",
        create.clone(),
        Some("k-2"),
    );
    let b = send(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/g/cypher",
        create,
        Some("k-2"),
    );
    assert_eq!((a.0, &a), (200, &b));
    let st = w.ok(&tok, Method::Get, "/v1/graphs/g", Json::Null);
    assert_eq!(st["nodes"], json!(4), "{st}");
    // A read-only query ignores the key (nothing to replay).
    let read = json!({ "query": "MATCH (n:Person) RETURN n.name AS name" });
    assert_eq!(
        send(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs/g/cypher",
            read,
            Some("k-1")
        )
        .0,
        200
    );
    // A malformed key is refused.
    let e = json!({ "edges": [["p", "q"]] });
    assert_eq!(
        send(&w, &tok, Method::Post, "/v1/graphs/g/edges", e, Some("")).0,
        400
    );
}

/// Workers Paid review: an admitted job solve is ≤ [`mc::JOB_MAX_WORK`]
/// (3e9, ≈ 4.5 s of wasm at the crate's ≲ 1 ns/unit × 1.5), so one alarm
/// turn never blocks the DOs co-resident in its isolate for longer; the
/// crate's `Profile::JOB` (20e9) and the earlier 10e9 (≈ 15 s) are refused.
/// A circulant graph (offsets 1 and 2, unit weights: no bridge, min degree
/// 4, so Stoer–Wagner) on 6,782 vertices (≈ 3.00e9 units) is queued; on
/// 6,783 it is `413` on the request path, although the crate's job profile
/// would take it.
#[test]
fn job_work_cap_is_3e9_and_refuses_just_above_on_the_request_path() {
    use ruvector_edge_analytics::service::Profile;
    assert_eq!(EDGE_JOB.budget.max_work, 3_000_000_000);
    let circulant = |n: u64| -> Vec<(u64, u64, f64)> {
        (0..n)
            .flat_map(|i| [(i, (i + 1) % n, 1.0), (i, (i + 2) % n, 1.0)])
            .collect()
    };
    let mode = QueryMode::Exact;
    for (n, admitted) in [(6_782u64, true), (6_783, false), (11_900, false)] {
        let edges = circulant(n);
        assert!(edges.len() as u64 <= mc::INLINE_MAX_EDGES);
        let g = TenantGraph::from_edges([0; 16], 0, &edges, &GraphLimits::JOB).unwrap();
        let est = plan(&g, &mode, &Profile::JOB).unwrap();
        assert!(est.memory_bytes <= JOB_MEMORY_BYTES, "{est:?}");
        assert_eq!(plan(&g, &mode, &EDGE_JOB).is_ok(), admitted, "{n}: {est:?}");
        match mc::route([0; 16], 0, &edges, &mode) {
            Ok(mc::Plan::Job) => assert!(admitted, "{n}"),
            Ok(mc::Plan::Inline(_)) => panic!("{n}: inline"),
            Err(e) => {
                assert!(!admitted, "{n}: {e:?}");
                assert_eq!((e.code.status(), e.code), (413, ErrorCode::BudgetExceeded));
            }
        }
    }
}
