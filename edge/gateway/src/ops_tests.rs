//! `/v1/ops` envelope over the DO backend (ADR-351 §16.3).

use crate::backend::mem::MemBackend;
use crate::ops::{dispatch, OpsReply};
use crate::testkit::*;
use ruvector_edge_store::ops::OpResponse;
use ruvector_edge_store::{CallerContext, ErrorCode};
use serde_json::{json, Value as Json};

const OP1: &str = "01HZX0000000000000000000A1";
const OP2: &str = "01HZX0000000000000000000A2";

fn env(c: &CallerContext, op_id: &str, op: &str, args: Json) -> Vec<u8> {
    json!({
        "v": 1, "op_id": op_id, "target": TARGET, "tenant_key": c.tenant_key().as_str(),
        "op": op, "args": args,
    })
    .to_string()
    .into_bytes()
}

fn send(b: &MemBackend, c: &CallerContext, body: &[u8], key: Option<&str>) -> OpsReply {
    block_on(dispatch(b, TARGET, c, body, key, T0))
}

fn code(r: &OpsReply) -> Option<ErrorCode> {
    serde_json::from_str::<OpResponse>(&r.body)
        .unwrap()
        .error
        .map(|e| e.code)
}

fn owner_with_collection(b: &MemBackend) -> CallerContext {
    let c = ctx(&tenant("org-a"), "alice", rw());
    block_on(crate::service::claim(&call(b, &c))).unwrap();
    let r = send(
        b,
        &c,
        &env(
            &c,
            OP1,
            "collection_create",
            json!({ "name": "d", "dim": 2, "metric": "l2" }),
        ),
        Some(OP1),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    c
}

#[test]
fn target_tenant_and_key_are_bound() {
    let b = MemBackend::new();
    let c = owner_with_collection(&b);
    let mut wrong_target: Json =
        serde_json::from_slice(&env(&c, OP2, "collection_list", json!({}))).unwrap();
    wrong_target["target"] = json!("https://evil.example/v1/ops");
    let r = send(&b, &c, wrong_target.to_string().as_bytes(), None);
    assert_eq!((r.status, code(&r)), (400, Some(ErrorCode::TargetMismatch)));
    let foreign = ctx(&tenant("org-b"), "alice", rw());
    let r = send(
        &b,
        &c,
        &env(&foreign, OP2, "collection_list", json!({})),
        None,
    );
    assert_eq!((r.status, code(&r)), (403, Some(ErrorCode::TenantMismatch)));
    let r = send(
        &b,
        &c,
        &env(&c, OP2, "collection_list", json!({})),
        Some(OP1),
    );
    assert_eq!((r.status, code(&r)), (400, Some(ErrorCode::InvalidRequest)));
    let r = send(&b, &c, &env(&c, OP2, "drop_everything", json!({})), None);
    assert_eq!((r.status, code(&r)), (400, Some(ErrorCode::UnknownOp)));
    let r = send(&b, &c, b"{not json", None);
    assert_eq!((r.status, code(&r)), (400, Some(ErrorCode::InvalidRequest)));
}

#[test]
fn op_id_never_applies_twice() {
    let b = MemBackend::new();
    let c = owner_with_collection(&b);
    let up = env(
        &c,
        OP2,
        "vector_upsert",
        json!({ "collection": "d", "vectors": [{ "id": "a", "values": [1.0, 0.0] }] }),
    );
    let first = send(&b, &c, &up, Some(OP2));
    assert_eq!(first.status, 200, "{}", first.body);
    let again = send(&b, &c, &up, Some(OP2));
    assert!(again.replayed);
    assert_eq!(again.body, first.body);
    let other = env(
        &c,
        OP2,
        "vector_upsert",
        json!({ "collection": "d", "vectors": [{ "id": "b", "values": [0.0, 1.0] }] }),
    );
    let r = send(&b, &c, &other, Some(OP2));
    assert_eq!((r.status, code(&r)), (409, Some(ErrorCode::OpReplayed)));
    let u = send(
        &b,
        &c,
        &env(&c, "01HZX0000000000000000000A3", "usage_get", json!({})),
        None,
    );
    let v: Json = serde_json::from_str(&u.body).unwrap();
    assert_eq!(v["result"]["usage"]["vectors"], json!(1));
}

#[test]
fn scope_is_a_step_up_and_replays_are_reauthorized() {
    let b = MemBackend::new();
    let c = owner_with_collection(&b);
    let ro_owner = ctx(c.tenant_key(), "alice", ro());
    // The create under OP1 was remembered; replaying it without the write
    // scope is still refused (authorization precedes the replay lookup).
    let create = env(
        &c,
        OP1,
        "collection_create",
        json!({ "name": "d", "dim": 2, "metric": "l2" }),
    );
    let r = send(&b, &ro_owner, &create, None);
    assert_eq!(
        (r.status, code(&r)),
        (403, Some(ErrorCode::InsufficientScope))
    );
    assert_eq!(r.step_up, Some("ruvector:write"));
    let r = send(&b, &ro_owner, &env(&c, OP2, "tenant_me", json!({})), None);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        (
            v["result"]["role"].clone(),
            v["result"]["tenant_key"].clone()
        ),
        (json!("owner"), json!(c.tenant_key().as_str()))
    );
}
