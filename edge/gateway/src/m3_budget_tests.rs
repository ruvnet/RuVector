//! Post-merge review fixes (ADR-351 M3 on the Workers Free budget): body
//! caps per route, which requests ship audit messages, the synchronous
//! snapshot / export / restore size budget, the drop-vs-restore journal,
//! and queued-import pacing.

use crate::m3_api::{audited, parse};
use crate::m3_http::refusal_audited;
use crate::m3_mem::*;
use crate::m3_ports::QueueName;
use crate::uploads::{MAX_INLINE_BYTES, MAX_PART_BODY};
use serde_json::{json, Value as Json};
use worker::Method;

#[test]
fn only_upload_parts_get_8_mib_and_inline_imports_get_512_kib() {
    let cap = |m: Method, p: &str| parse(&m, p).and_then(|r| r.body_cap());
    assert_eq!(cap(Method::Post, "/v1/uploads/u1/parts/1"), Some(8 << 20));
    assert_eq!(MAX_PART_BODY, 8 << 20);
    assert_eq!(
        cap(Method::Post, "/v1/collections/c:import"),
        Some(512 << 10)
    );
    assert_eq!(MAX_INLINE_BYTES, 512 << 10);
    for p in [
        "/v1/collections/c/snapshots",
        "/v1/collections/c:export",
        "/v1/uploads",
        "/v1/collections/c/vectors:upsert",
    ] {
        assert_eq!(cap(Method::Post, p), None, "{p}");
    }
}

#[test]
fn an_inline_import_over_512_kib_is_refused_before_hashing_or_storing() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let body = json!({ "name": "docs", "dim": 8, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", body).0, 201);
    let path = "/v1/collections/docs:import";
    let big = vec![0u8; MAX_INLINE_BYTES + 1];
    let (s, v) = w.raw(&o, Method::Post, path, big, true, None);
    assert_eq!(s, 413, "{v}");
    assert!(w.blob.keys("staging/").is_empty());
    assert!(w.q.take(QueueName::Ingest).is_empty());
}

#[test]
fn polled_reads_and_throttled_refusals_ship_no_audit_message() {
    let r = |m: Method, p: &str| parse(&m, p).unwrap();
    let job = r(Method::Get, "/v1/jobs/j1");
    let list = r(Method::Get, "/v1/collections/c/snapshots");
    assert!(!audited(&job, b""));
    assert!(!audited(&list, b""));
    let snap = r(Method::Post, "/v1/collections/c/snapshots");
    assert!(audited(&snap, b""));
    // A throttled write is not audited; a denied (401) write still is.
    assert!(!refusal_audited(&snap, 429));
    assert!(refusal_audited(&snap, 401));
    assert!(!refusal_audited(&job, 401));
    assert!(!refusal_audited(&list, 403));
    // End to end: polling a job status ships nothing.
    let w = World::default();
    let o = w.owner("org-a", "alice");
    w.q.take(QueueName::Audit);
    for _ in 0..3 {
        w.req(
            &o,
            Method::Get,
            "/v1/jobs/000000000000000000000000",
            Json::Null,
        );
    }
    assert!(w.q.take(QueueName::Audit).is_empty());
}
