//! rv-mincut on Workers Free (ADR-351 §10, M4 review): a job whose every
//! solve attempt is killed ends `failed` with `413 budget_exceeded` (not
//! the crate's `409` job-state conflict), and a job-sized edge list is
//! handed to the job verbatim — validated in its admit turn, not parsed on
//! the request path.

use crate::graph_tests::owner_world;
use crate::graph_wire::{JobCall, JobRequest};
use crate::mincut_job::{self, Turn};
use crate::testkit::tenant;
use ruvector_edge_analytics::job::MAX_ATTEMPTS;
use ruvector_edge_analytics::QueryMode;
use ruvector_edge_store::MemSqlStore;
use serde_json::{json, Value as Json};
use worker::Method;

#[test]
fn killed_solves_exhaust_attempts_as_413() {
    let st = MemSqlStore::new();
    let t = tenant("org-g");
    let req = |call| {
        let r = JobRequest {
            tenant_key: t.as_str().into(),
            job_id: "mc_2".into(),
            call,
        };
        serde_json::to_vec(&r).unwrap()
    };
    let submit = JobCall::Submit {
        mode: QueryMode::Exact,
        edges: "[[0,1,1.0],[1,2,2.0],[2,0,1.0]]".into(),
        labels: None,
        graph: None,
        revision: 0,
        now_ms: 1_000,
    };
    mincut_job::serve(None, &st, &req(submit));
    // Admission turn.
    assert!(matches!(
        mincut_job::prepare(&st, 2_000),
        Turn::Done(Some(_))
    ));
    // Every attempt is recorded, then its solve is killed (never runs).
    for a in 1..=MAX_ATTEMPTS {
        match mincut_job::prepare(&st, 2_000 + u64::from(a)) {
            Turn::Solve(d) => assert!(!d.is_terminal()),
            Turn::Done(n) => panic!("attempt {a}: {n:?}"),
            Turn::Expire => panic!("attempt {a}: expired"),
        }
    }
    // The next alarm finds the attempts exhausted: 413, terminal; the
    // alarm is re-armed for the job's expiry.
    let expiry = mincut_job::expires_at(1_000);
    assert!(matches!(
        mincut_job::prepare(&st, 3_000),
        Turn::Done(Some(ms)) if ms == expiry - 3_000
    ));
    let (out, _) = mincut_job::serve(None, &st, &req(JobCall::Get { now_ms: 3_001 }));
    let v = serde_json::from_str::<Json>(&out).unwrap()["Ok"]["view"].clone();
    assert_eq!(
        (v["state"].clone(), v["error"]["code"].clone()),
        (json!("failed"), json!("budget_exceeded")),
        "{v}"
    );
    assert_eq!(v["error"]["status"], json!(413));
    assert!(matches!(
        mincut_job::prepare(&st, 4_000),
        Turn::Done(Some(_))
    ));
    // Retention: at the expiry the job erases itself; its id is then 404.
    assert!(matches!(mincut_job::prepare(&st, expiry), Turn::Expire));
    mincut_job::expire(&st).unwrap();
    let (out, _) = mincut_job::serve(None, &st, &req(JobCall::Get { now_ms: expiry }));
    let v: Json = serde_json::from_str(&out).unwrap();
    assert_eq!(v["Err"]["code"], json!("not_found"), "{v}");
    assert!(matches!(
        mincut_job::prepare(&st, expiry + 1),
        Turn::Done(None)
    ));
}

#[test]
fn job_sized_edges_pass_verbatim_and_fail_validation_in_admit() {
    let (w, tok) = owner_world();
    let n = crate::mincut_core::INLINE_MAX_EDGES + 5;
    let mut edges: Vec<Json> = (0..n).map(|i| json!([i, i + 1])).collect();
    edges[7] = json!([1, "x"]);
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": edges }),
        None,
    );
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        (r.status, v["state"].clone()),
        (202, json!("queued")),
        "{v}"
    );
    let path = format!("/v1/mincut/jobs/{}", v["job_id"].as_str().unwrap());
    w.b.m4.drain_alarms(4);
    let j = w.ok(&tok, Method::Get, &path, Json::Null);
    assert_eq!(
        (j["state"].clone(), j["error"]["status"].clone()),
        (json!("failed"), json!(400)),
        "{j}"
    );
    // Inline-sized lists are still validated on the request path.
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": [[0, 1], [1, "x"]] }),
        None,
    );
    assert_eq!(r.status, 400, "{}", r.body);
    // Unknown fields are still refused.
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "edges": [[0, 1]], "nope": 1 }),
        None,
    );
    assert_eq!(r.status, 400, "{}", r.body);
    assert_eq!(crate::mincut_routes::raw_edge_count("[[0,1],[2,3,0.5]]"), 2);
    assert_eq!(crate::mincut_routes::raw_edge_count("[]"), 0);
}
