//! M2 + M3 + M5 + M4 integration: the M4 graph / min-cut routes stay
//! disjoint from the M3 (`/v1/jobs/{id}`, snapshots …), M5 registry and
//! M1 REST tables, carry their own rate class and audit events, and the
//! M3 snapshot / export / restore paths (which run over the `VectorShard`
//! `/m3` side channel) refuse an `index = rabitq` collection up front.

use crate::api::ratelimit::{request_class, Class, MUTATING_OPS};
use crate::audit_http::{body_event, graph_event};
use crate::graph_routes::{self, GraphRoute};
use crate::m3_mem::*;
use crate::testkit::Rng;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::ErrorCode;
use serde_json::{json, Value as Json};
use worker::Method;

const GRAPH: [(Method, &str); 7] = [
    (Method::Post, "/v1/graphs"),
    (Method::Get, "/v1/graphs"),
    (Method::Get, "/v1/graphs/g1"),
    (Method::Post, "/v1/graphs/g1/cypher"),
    (Method::Post, "/v1/graphs/g1/edges"),
    (Method::Post, "/v1/mincut"),
    (Method::Get, "/v1/mincut/jobs/j1"),
];

#[test]
fn graph_routes_are_disjoint_from_the_m3_registry_and_rest_tables() {
    for (m, p) in GRAPH {
        assert!(graph_routes::parse(&m, p).is_some(), "{p}");
        assert_eq!(crate::m3_api::parse(&m, p), None, "{p}");
        assert!(crate::registry_routes::parse(&m, p).is_none(), "{p}");
        assert!(crate::rest::parse(&m, p).is_none(), "{p}");
    }
    // M3 import jobs and M4 min-cut jobs live under different prefixes.
    let m3: [(Method, &str); 5] = [
        (Method::Get, "/v1/jobs/j1"),
        (Method::Post, "/v1/collections/c/snapshots"),
        (Method::Post, "/v1/collections/c:import"),
        (Method::Post, "/v1/uploads"),
        (Method::Get, "/v1/exports/e1"),
    ];
    for (m, p) in m3 {
        assert!(crate::m3_api::parse(&m, p).is_some(), "{p}");
        assert_eq!(graph_routes::parse(&m, p), None, "{p}");
    }
    for p in [
        "/v1/rvf/scopes",
        "/v1/collections",
        "/v1/collections/c/query",
    ] {
        assert_eq!(graph_routes::parse(&Method::Get, p), None, "{p}");
        assert_eq!(graph_routes::parse(&Method::Post, p), None, "{p}");
    }
}

#[test]
fn graph_routes_have_their_own_rate_class_and_audit_events() {
    let class = |m: Method, p: &str| request_class(false, &m, p);
    assert_eq!(class(Method::Post, "/v1/graphs"), Class::Write);
    assert_eq!(class(Method::Post, "/v1/graphs/g1/edges"), Class::Write);
    assert_eq!(class(Method::Post, "/v1/graphs/g1/cypher"), Class::Read);
    assert_eq!(class(Method::Get, "/v1/mincut/jobs/j1"), Class::Read);
    // M3's job status is still M3's class (a read), not re-routed.
    assert_eq!(class(Method::Get, "/v1/jobs/j1"), Class::Read);
    assert_eq!(
        class(Method::Post, "/v1/collections/c/snapshots"),
        Class::Write
    );

    let cypher = GraphRoute::Cypher("g1".into());
    let read = json!({ "query": "MATCH (n) RETURN n" }).to_string();
    let write = json!({ "query": "CREATE (n:Person {name: 'a'})" }).to_string();
    assert_eq!(graph_event(&cypher, Some(read.as_bytes())), None);
    assert_eq!(
        graph_event(&cypher, Some(write.as_bytes())),
        Some("graph.cypher_mutate")
    );
    assert_eq!(graph_event(&GraphRoute::Create, None), Some("graph.create"));
    assert_eq!(
        graph_event(&GraphRoute::Edges("g1".into()), None),
        Some("graph.edges")
    );
    for r in [
        GraphRoute::List,
        GraphRoute::Get("g1".into()),
        GraphRoute::Mincut,
        GraphRoute::Job("j1".into()),
    ] {
        assert_eq!(graph_event(&r, None), None, "{r:?}");
    }
    // Both the M5 and M4 mutating MCP tools pay the write budget and audit.
    assert!(MUTATING_OPS.contains(&"rvf_import") && MUTATING_OPS.contains(&"graph_mutate"));
    let call = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
                       "params": { "name": "graph_mutate", "arguments": {} } })
    .to_string();
    assert_eq!(
        body_event(true, call.as_bytes()),
        Some(("mcp.graph_mutate".into(), Capability::Write))
    );
}

#[test]
fn rabitq_collection_refuses_snapshot_export_and_restore() {
    let w = World::default();
    let c = w.owner("org-q", "alice");
    let spec = json!({ "name": "rq", "dim": 64, "metric": "l2", "shards": 2, "index": "rabitq" });
    let (s, v) = w.req(&c, Method::Post, "/v1/collections", spec);
    assert_eq!(s, 201, "{v}");
    let mut rng = Rng(3);
    let rows: Vec<Json> = (0..50)
        .map(|i| json!({ "id": format!("v{i}"), "values": rng.vec(64) }))
        .collect();
    let path = "/v1/collections/rq/vectors:upsert";
    let (s, v) = w.req(&c, Method::Post, path, json!({ "vectors": rows }));
    assert_eq!(s, 200, "{v}");
    let (_, before) = w.req(&c, Method::Get, "/v1/usage", Json::Null);

    for p in [
        "/v1/collections/rq/snapshots",
        "/v1/collections/rq:export",
        "/v1/collections/rq/snapshots/1:restore",
    ] {
        let (s, v) = w.req(&c, Method::Post, p, Json::Null);
        assert_eq!(s, 400, "{p}: {v}");
        assert!(is(&v, ErrorCode::InvalidRequest), "{p}: {v}");
    }
    // Nothing was written: no snapshot record, no R2 object, no charge.
    let (s, v) = w.req(&c, Method::Get, "/v1/collections/rq/snapshots", Json::Null);
    assert_eq!((s, v["snapshots"].as_array().map(Vec::len)), (200, Some(0)));
    assert!(w.blob.keys("snapshots/").is_empty() && w.blob.keys("exports/").is_empty());
    let (_, after) = w.req(&c, Method::Get, "/v1/usage", Json::Null);
    assert_eq!(before["usage"]["vectors"], after["usage"]["vectors"]);

    // The collection itself still serves (over its QuantShards).
    let q = json!({ "vector": rng.vec(64), "top_k": 3 });
    let mut status = 0;
    for _ in 0..5 {
        let (s, _) = w.req(&c, Method::Post, "/v1/collections/rq/query", q.clone());
        status = s;
        if s != 503 {
            break;
        }
    }
    assert_eq!(status, 200);
}
