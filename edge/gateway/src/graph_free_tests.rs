//! rv-graph hardening (ADR-351 §10, M4 review): Cypher text that would hang
//! or overflow rvlite's parser is `400` on every entry point (REST, MCP
//! `graph_query`, `graph_mutate` and its `dry_run`), before any
//! authorization; the Workers-Free graph limits are `413` before the work;
//! and a stored graph's min-cut digest does not depend on the resident
//! graph's history.

use crate::graph_store::{MAX_BULK_EDGES, MAX_STATE_BYTES};
use crate::graph_tests::owner_world;
use crate::graph_wire::graph_do_name;
use crate::testkit::tenant;
use serde_json::{json, Value as Json};
use std::time::{Duration, Instant};
use worker::Method;

fn status(w: &crate::e2e_world::World, tok: &str, path: &str, body: Json) -> (u16, String) {
    let r = w.send(tok, Method::Post, path, body, None);
    let v: Json = serde_json::from_str(&r.body).unwrap_or(Json::Null);
    (r.status, v["code"].as_str().unwrap_or_default().to_string())
}

fn mcp_text(w: &crate::e2e_world::World, tok: &str, name: &str, args: Json) -> String {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": name, "arguments": args } });
    let r: Json =
        serde_json::from_str(&w.send(tok, Method::Post, "/v1/mcp", body, None).body).unwrap();
    assert_eq!(r["result"]["isError"], json!(true), "{r}");
    r["result"]["content"][0]["text"]
        .as_str()
        .unwrap()
        .to_string()
}

#[test]
fn parameters_and_deep_nesting_are_400_everywhere() {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "g" }));
    let deep_paren = format!("RETURN {}1{}", "(".repeat(2_000), ")".repeat(2_000));
    let deep_list = format!("RETURN {}1{}", "[".repeat(2_000), "]".repeat(2_000));
    let bad = [
        "MATCH (n) RETURN $p".to_string(),
        "CREATE$p".to_string(),
        deep_paren,
        deep_list,
    ];
    let t = Instant::now();
    for q in &bad {
        let r = status(&w, &tok, "/v1/graphs/g/cypher", json!({ "query": q }));
        assert_eq!(r, (400, "invalid_request".into()), "{q:.40}");
        // Pre-auth too: a missing graph and another tenant answer 400, not
        // a hung / trapped isolate.
        let r = status(&w, &tok, "/v1/graphs/nope/cypher", json!({ "query": q }));
        assert_eq!(r.0, 400, "{q:.40}");
        let om = w.edge_as.user_token(
            "alice",
            "org-g",
            &w.mcp(),
            &["ruvector:read", "ruvector:write"],
        );
        let t1 = mcp_text(&w, &om, "graph_query", json!({ "graph": "g", "query": q }));
        let t2 = mcp_text(&w, &om, "graph_mutate", json!({ "graph": "g", "query": q }));
        let t3 = mcp_text(
            &w,
            &om,
            "graph_mutate",
            json!({ "graph": "g", "query": q, "dry_run": true }),
        );
        for t in [t1, t2, t3] {
            assert!(t.contains("invalid_request") && t.contains("400"), "{t}");
        }
    }
    let bob = w
        .edge_as
        .user_token("bob", "org-x", &w.v1(), &["ruvector:read"]);
    let r = status(
        &w,
        &bob,
        "/v1/graphs/g/cypher",
        json!({ "query": "RETURN $p" }),
    );
    assert_eq!(r.0, 400);
    assert!(t.elapsed() < Duration::from_secs(10), "{:?}", t.elapsed());
    // `$` inside a string or a backtick name, and moderate nesting, are fine.
    let ok = w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/g/cypher",
        json!({ "query": "CREATE (n:P {name: 'a$b', `x$`: \"c\\\"$\"}) RETURN n.name AS name" }),
    );
    assert_eq!(ok["rows"][0]["name"], json!("a$b"), "{ok}");
    let nested = format!("RETURN {}1{} AS x", "(".repeat(20), ")".repeat(20));
    let ok = w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/g/cypher",
        json!({ "query": nested }),
    );
    assert_eq!(ok["rows"][0]["x"], json!(1), "{ok}");
}

#[test]
fn free_graph_limits_are_413_before_the_work() {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "g" }));
    let too_many: Vec<Json> = (0..=MAX_BULK_EDGES)
        .map(|i| json!([format!("a{i}"), format!("b{i}")]))
        .collect();
    let r = status(&w, &tok, "/v1/graphs/g/edges", json!({ "edges": too_many }));
    assert_eq!(r, (413, "payload_too_large".into()));
    w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/g/edges",
        json!({ "edges": [["a", "b", 1.0], ["b", "c", 2.0]] }),
    );
    // A stored state over the cap (or a graph too large to export for a
    // min-cut) is refused from its counters, before any load.
    let name = graph_do_name(&tenant("org-g"), "g");
    let set = |k: &str, v: u64| {
        let stores = w.b.m4.graph_stores.borrow();
        crate::graph_persist::put(&stores[name.as_str()], k, &v.to_string()).unwrap();
    };
    set("edges", crate::graph_mincut::MAX_MINCUT_EDGES + 1);
    w.b.restart();
    let r = status(&w, &tok, "/v1/mincut", json!({ "graph": "g" }));
    assert_eq!(r, (413, "budget_exceeded".into()));
    set("edges", 2);
    set("bytes", MAX_STATE_BYTES + 1);
    let r = status(
        &w,
        &tok,
        "/v1/graphs/g/cypher",
        json!({ "query": "MATCH (n) RETURN n" }),
    );
    assert_eq!(r, (413, "budget_exceeded".into()));
    assert_eq!(
        status(&w, &tok, "/v1/mincut", json!({ "graph": "g" })),
        (413, "budget_exceeded".into())
    );
}

#[test]
fn stored_graph_mincut_digest_is_history_independent() {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "m" }));
    // Parallel and reverse edges whose float sum depends on the order.
    let mut edges = Vec::new();
    for i in 0..40u32 {
        let (a, b) = (format!("v{}", i % 8), format!("v{}", (i * 3 + 1) % 8));
        let wgt = 0.1 * f64::from(i % 7 + 1);
        edges.push(if i % 2 == 0 {
            json!([a, b, wgt])
        } else {
            json!([b, a, wgt])
        });
    }
    for c in edges.chunks(10) {
        w.ok(
            &tok,
            Method::Post,
            "/v1/graphs/m/edges",
            json!({ "edges": c }),
        );
    }
    let cut = || {
        let r = w.send(
            &tok,
            Method::Post,
            "/v1/mincut",
            json!({ "graph": "m" }),
            None,
        );
        assert_eq!(r.status, 200, "{}", r.body);
        serde_json::from_str::<Json>(&r.body).unwrap()["digest"].clone()
    };
    let warm = cut();
    for _ in 0..5 {
        w.b.restart();
        assert_eq!(cut(), warm);
    }
}
