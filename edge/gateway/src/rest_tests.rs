//! REST table and handlers over the DO backend.

use crate::backend::mem::MemBackend;
use crate::rest::{handle, op_args, parse, ApiReply, ApiRoute, Caller};
use crate::testkit::*;
use ruvector_edge_auth::scopes::{route_requirement, Method as RouteMethod};
use serde_json::{json, Value as Json};
use worker::Method;

fn caller(org: &str, user: &str, scope: ruvector_edge_auth::CapabilitySet) -> Caller {
    Caller {
        ctx: ctx(&tenant(org), user, scope),
        org_id: org.into(),
        workspace_id: "ws1".into(),
        scopes: vec!["ruvector:read".into()],
    }
}

fn req(b: &MemBackend, c: &Caller, m: Method, path: &str, body: Json) -> ApiReply {
    let route = parse(&m, path).expect(path);
    let bytes = if body.is_null() {
        Vec::new()
    } else {
        body.to_string().into_bytes()
    };
    block_on(handle(b, c, &route, &bytes, None, T0, REST_MD))
}

#[test]
fn route_table_serves_task_and_adr_spellings() {
    let c = |s: &str| s.to_string();
    let cases = [
        (Method::Get, "/v1/me", Some(ApiRoute::Me)),
        (Method::Post, "/v1/claim", Some(ApiRoute::Claim)),
        (Method::Post, "/v1/tenant:claim", Some(ApiRoute::Claim)),
        (Method::Get, "/v1/usage", Some(ApiRoute::Usage)),
        (Method::Post, "/v1/ops", Some(ApiRoute::Ops)),
        (Method::Post, "/v1/collections", Some(ApiRoute::Create)),
        (Method::Get, "/v1/collections", Some(ApiRoute::List)),
        (
            Method::Get,
            "/v1/collections/d",
            Some(ApiRoute::Get(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/vectors",
            Some(ApiRoute::Upsert(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/vectors:upsert",
            Some(ApiRoute::Upsert(c("d"))),
        ),
        (
            Method::Delete,
            "/v1/collections/d/vectors",
            Some(ApiRoute::Delete(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/vectors:delete",
            Some(ApiRoute::Delete(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/query",
            Some(ApiRoute::Query(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/fetch",
            Some(ApiRoute::Fetch(c("d"))),
        ),
        (
            Method::Post,
            "/v1/collections/d/vectors:fetch",
            Some(ApiRoute::Fetch(c("d"))),
        ),
        (Method::Get, "/v1/collections/d/query", None),
        (Method::Post, "/v1/collections//query", None),
        (Method::Post, "/v1/collections/a:b/query", None),
        (Method::Get, "/v1/me/", None),
        (Method::Delete, "/v1/collections/d", None),
    ];
    for (m, p, want) in cases {
        assert_eq!(parse(&m, p), want, "{m:?} {p}");
    }
    // Every ADR spelling served here is also in the auth crate's route table.
    for (m, p) in [
        (RouteMethod::Post, "/v1/tenant:claim"),
        (RouteMethod::Post, "/v1/collections/d/vectors:upsert"),
        (RouteMethod::Post, "/v1/collections/d/vectors:delete"),
        (RouteMethod::Post, "/v1/collections/d/vectors:fetch"),
        (RouteMethod::Post, "/v1/collections/d/query"),
        (RouteMethod::Get, "/v1/usage"),
        (RouteMethod::Post, "/v1/ops"),
    ] {
        assert!(route_requirement(m, p).is_some(), "{p}");
    }
}

#[test]
fn op_args_lift_dry_run_and_take_the_collection_from_the_path() {
    let (a, d) = op_args(br#"{"vectors":[],"dry_run":true}"#, Some("c"), true).unwrap();
    assert!(d);
    assert_eq!(
        serde_json::from_str::<Json>(&a).unwrap(),
        json!({ "vectors": [], "collection": "c" })
    );
    assert!(op_args(br#"{"collection":"x"}"#, Some("c"), false).is_err());
    assert!(op_args(br#"{"dry_run":true}"#, Some("c"), false).is_err());
    assert!(op_args(b"[1]", None, false).is_err());
    assert_eq!(op_args(b"", None, false).unwrap().0, "{}");
}

#[test]
fn me_claim_create_query_and_problems() {
    let b = MemBackend::new();
    let alice = caller("org-a", "alice", rw());
    let me = |c: &Caller| {
        serde_json::from_str::<Json>(&req(&b, c, Method::Get, "/v1/me", Json::Null).body).unwrap()
    };
    let m = me(&alice);
    assert_eq!(
        (
            m["role"].clone(),
            m["claimed"].clone(),
            m["capabilities"].clone()
        ),
        (Json::Null, json!(false), json!([]))
    );
    assert_eq!(m["tenant_key"], json!(tenant("org-a").as_str()));
    // Unclaimed: data routes answer 403 not_claimed.
    let r = req(&b, &alice, Method::Get, "/v1/collections", Json::Null);
    assert_eq!(
        (r.status, r.content_type),
        (403, "application/problem+json")
    );
    assert!(r.body.contains("not_claimed"));
    let r = req(&b, &alice, Method::Post, "/v1/claim", Json::Null);
    assert_eq!((r.status, r.body.as_str()), (201, r#"{"role":"owner"}"#));
    assert_eq!(
        req(&b, &alice, Method::Post, "/v1/tenant:claim", Json::Null).status,
        409
    );
    let m = me(&alice);
    assert_eq!(
        (m["role"].clone(), m["capabilities"].clone()),
        (json!("owner"), json!(["read", "write", "create"]))
    );
    let r = req(
        &b,
        &alice,
        Method::Post,
        "/v1/collections",
        json!({ "name": "d", "dim": 2, "metric": "cosine" }),
    );
    assert_eq!(r.status, 201, "{}", r.body);
    let r = req(
        &b,
        &alice,
        Method::Post,
        "/v1/collections/d/vectors",
        json!({ "vectors": [{ "id": "a", "values": [1.0, 0.0], "metadata": { "k": 1 } }] }),
    );
    assert_eq!(r.status, 200, "{}", r.body);
    let r = req(
        &b,
        &alice,
        Method::Post,
        "/v1/collections/d/query",
        json!({ "vector": [1.0, 0.1], "top_k": 1, "include": ["metadata"] }),
    );
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(
        (
            v["matches"][0]["id"].clone(),
            v["matches"][0]["metadata"].clone()
        ),
        (json!("a"), json!({ "k": 1 }))
    );
    let r = req(
        &b,
        &alice,
        Method::Post,
        "/v1/collections/d/query",
        json!({ "vector": [1.0], "top_k": 1 }),
    );
    assert!(
        r.status == 400 && r.body.contains("dimension_mismatch"),
        "{}",
        r.body
    );
    assert_eq!(
        req(&b, &alice, Method::Get, "/v1/collections/nope", Json::Null).status,
        404
    );
    let r = req(
        &b,
        &alice,
        Method::Delete,
        "/v1/collections/d/vectors",
        json!({ "ids": ["a"], "dry_run": true }),
    );
    assert!(r.body.contains(r#""dry_run":true"#), "{}", r.body);
    // Read-only token: 403 with the /v1 step-up challenge.
    let ro_alice = caller("org-a", "alice", ro());
    let r = req(
        &b,
        &ro_alice,
        Method::Post,
        "/v1/collections/d/vectors:upsert",
        json!({ "vectors": [] }),
    );
    assert_eq!(r.status, 403);
    let www = r.www_authenticate.unwrap();
    assert!(
        www.contains(REST_MD) && www.contains("ruvector:write"),
        "{www}"
    );
    let r = req(&b, &ro_alice, Method::Get, "/v1/usage", Json::Null);
    assert_eq!(r.status, 200, "{}", r.body);
}
