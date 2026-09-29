//! Cypher developer experience (ADR-351 §3 rv-graph): refusals say what
//! failed (REST problem `detail`, MCP tool-error `detail`) with the status
//! and `code` unchanged, and unaliased `RETURN` items are named by their
//! expression text instead of `?column?` (which also collapsed several
//! such items into one column).

use crate::e2e_world::World;
use crate::graph_cypher_dx::{clamp, column_name, MAX_DETAIL};
use crate::graph_tests::owner_world;
use rvlite::cypher::ast::{AggregationFunction, Expression as E};
use serde_json::{json, Value as Json};
use std::collections::HashMap;
use worker::Method;

const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];

/// `(status, code, detail)` of a Cypher call on graph `g`.
fn refuse(w: &World, tok: &str, g: &str, q: &str) -> (u16, String, String) {
    let r = w.send(
        tok,
        Method::Post,
        &format!("/v1/graphs/{g}/cypher"),
        json!({ "query": q }),
        None,
    );
    let v: Json = serde_json::from_str(&r.body).unwrap_or(Json::Null);
    let s = |k: &str| v[k].as_str().unwrap_or_default().to_string();
    (r.status, s("code"), s("detail"))
}

/// The tool-error body of an MCP `tools/call` (`isError: true`).
fn mcp_err(w: &World, tok: &str, name: &str, args: Json) -> Json {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": name, "arguments": args } });
    let r = w.send(tok, Method::Post, "/v1/mcp", body, None);
    assert_eq!(r.status, 200, "{}", r.body);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["result"]["isError"], json!(true), "{v}");
    serde_json::from_str(v["result"]["content"][0]["text"].as_str().unwrap()).unwrap()
}

fn mcp_ok(w: &World, tok: &str, name: &str, args: Json) -> Json {
    let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call",
        "params": { "name": name, "arguments": args } });
    let r = w.send(tok, Method::Post, "/v1/mcp", body, None);
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!(v["result"]["isError"], json!(false), "{v}");
    v["result"]["structuredContent"].clone()
}

fn world_with_graph(g: &str) -> (World, String) {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": g }));
    (w, tok)
}

#[test]
fn rest_refusals_say_what_failed() {
    let (w, tok) = world_with_graph("p");
    let bad = |q: &str| refuse(&w, &tok, "p", q);

    let (s, c, d) = bad("CREATE (a:N {id:1})-[:E {w:3}]->(b:N {id:2}) RETURN a,b");
    assert_eq!((s, c.as_str()), (400, "invalid_request"), "{d}");
    assert!(d.contains("bound by CREATE") && d.contains("`a`"), "{d}");
    // The refused mutation was discarded.
    let v = w.ok(&tok, Method::Get, "/v1/graphs/p", Json::Null);
    assert_eq!(
        (v["nodes"].clone(), v["edges"].clone()),
        (json!(0), json!(0))
    );

    let (s, c, d) = bad("MATCH (n:N) RETURN count(n)");
    assert_eq!((s, c.as_str()), (400, "invalid_request"));
    assert!(d.contains("aggregation count()"), "{d}");

    let (s, _, d) = bad("MATCH (n:N RETURN n");
    assert_eq!(s, 400);
    assert!(d.starts_with("parse error at line 1, column 12"), "{d}");

    for (q, want) in [
        ("MERGE (n:N {id: 5})", "unsupported: MERGE clause"),
        ("MATCH (n) WITH n RETURN n", "unsupported: WITH clause"),
        ("MATCH (a)-[*1..3]->(b) RETURN a", "variable-length"),
        ("MATCH p = (a)-->(b) RETURN p", "MATCH pattern"),
        ("MATCH (n) RETURN n.id + 1", "operator expressions"),
        ("MATCH (n) RETURN m", "unknown variable `m`"),
        (
            "MATCH (n {id: $id}) RETURN n",
            "query parameters ($ at byte 14)",
        ),
    ] {
        let (s, c, d) = bad(q);
        assert_eq!((s, c.as_str()), (400, "invalid_request"), "{q}: {d}");
        assert!(d.contains(want), "{q}: {d}");
    }
    let deep = format!("RETURN {}1{}", "[".repeat(40), "]".repeat(40));
    let (s, _, d) = bad(&deep);
    assert_eq!(s, 400);
    assert!(d.contains("deeper than 32 levels"), "{d}");
    let long = format!("MATCH (n) WHERE n.x = '{}' RETURN n", "y".repeat(5000));
    let (s, c, d) = bad(&long);
    assert_eq!((s, c.as_str()), (413, "payload_too_large"));
    assert!(d.contains("4096-byte limit"), "{d}");

    // A long identifier in a parse error is bounded.
    let (s, _, d) = bad(&format!("MATCH (n) RETURN n {}", "z".repeat(3000)));
    assert_eq!(s, 400);
    assert!(d.chars().count() <= MAX_DETAIL, "{}", d.len());
}

#[test]
fn budget_refusals_say_which_budget() {
    let (w, tok) = world_with_graph("rows");
    let edges: Vec<Json> = (0..800)
        .map(|i| json!([format!("a{i}"), format!("b{i}")]))
        .collect();
    w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/rows/edges",
        json!({ "edges": edges }),
    );
    let (s, c, d) = refuse(&w, &tok, "rows", "MATCH (n) RETURN n");
    assert_eq!((s, c.as_str()), (413, "budget_exceeded"));
    assert!(d.contains("more than 1000 rows"), "{d}");
    let (s, c, d) = refuse(
        &w,
        &tok,
        "rows",
        "MATCH (a)-[r]-(b), (c)-[s]-(d), (e)-[t]-(f), (g)-[u]-(h) RETURN a",
    );
    assert_eq!((s, c.as_str()), (413, "budget_exceeded"));
    assert!(d.contains("10000-step budget"), "{d}");
}

#[test]
fn unaliased_columns_are_named_by_their_expression() {
    let (w, tok) = world_with_graph("c");
    let cy = |q: &str| {
        w.ok(
            &tok,
            Method::Post,
            "/v1/graphs/c/cypher",
            json!({ "query": q }),
        )
    };
    cy("CREATE (a:N {id: 1, name: 'x'})");
    let v = cy("MATCH (n:N) RETURN n.id, n.name, n");
    assert_eq!(v["columns"], json!(["n.id", "n.name", "n"]), "{v}");
    let row = &v["rows"][0];
    assert_eq!(
        (row["n.id"].clone(), row["n.name"].clone()),
        (json!(1), json!("x"))
    );
    assert_eq!(row["n"]["properties"]["name"], json!("x"));
    // Aliased columns are unchanged.
    let v = cy("MATCH (n:N) RETURN n.name AS name");
    assert_eq!(v["columns"], json!(["name"]));
    assert_eq!(v["rows"], json!([{ "name": "x" }]));
    // Nothing matched: the same names, zero rows.
    let v = cy("MATCH (n:Nope) RETURN n.id, n");
    assert_eq!(v["columns"], json!(["n.id", "n"]), "{v}");
    assert_eq!(v["rows"], json!([]));
}

#[test]
fn mcp_refusals_and_columns() {
    let (w, _) = world_with_graph("m");
    let mt = w.edge_as.user_token("alice", "org-g", &w.mcp(), &RW);

    let e = mcp_err(
        &w,
        &mt,
        "graph_query",
        json!({ "graph": "m", "query": "MATCH (n:N) RETURN count(n)" }),
    );
    assert_eq!(
        (e["code"].clone(), e["status"].clone()),
        (json!("invalid_request"), json!(400))
    );
    assert!(
        e["detail"]
            .as_str()
            .unwrap()
            .contains("aggregation count()"),
        "{e}"
    );

    let q = "CREATE (a:N {id:1})-[:E {w:3}]->(b:N {id:2}) RETURN a,b";
    let e = mcp_err(&w, &mt, "graph_mutate", json!({ "graph": "m", "query": q }));
    assert_eq!(e["code"], json!("invalid_request"));
    assert!(
        e["detail"].as_str().unwrap().contains("bound by CREATE"),
        "{e}"
    );

    let e = mcp_err(
        &w,
        &mt,
        "graph_mutate",
        json!({ "graph": "m", "query": "CREATE (a:N", "dry_run": true }),
    );
    assert!(
        e["detail"].as_str().unwrap().starts_with("parse error"),
        "{e}"
    );

    let e = mcp_err(
        &w,
        &mt,
        "graph_query",
        json!({ "graph": "m", "query": "MATCH (n {id: $x}) RETURN n" }),
    );
    assert!(
        e["detail"].as_str().unwrap().contains("query parameters"),
        "{e}"
    );

    mcp_ok(
        &w,
        &mt,
        "graph_mutate",
        json!({ "graph": "m", "query": "CREATE (a:N {name: 'q'})" }),
    );
    let v = mcp_ok(
        &w,
        &mt,
        "graph_query",
        json!({ "graph": "m", "query": "MATCH (n:N) RETURN n.name" }),
    );
    assert_eq!(v["columns"], json!(["n.name"]));
    assert_eq!(v["rows"], json!([{ "n.name": "q" }]));
}

#[test]
fn column_names_and_clamp() {
    let n = || Box::new(E::Variable("n".into()));
    let count = E::Aggregation {
        function: AggregationFunction::Count,
        expression: n(),
        distinct: true,
    };
    assert_eq!(column_name(&count, 0), "count(DISTINCT n)");
    let mut m = HashMap::new();
    m.insert("b".to_string(), E::Integer(2));
    m.insert("a".to_string(), E::String("s".into()));
    assert_eq!(column_name(&E::Map(m), 0), "{a: 's', b: 2}");
    let call = E::FunctionCall {
        name: "toUpper".into(),
        args: vec![E::Property {
            object: n(),
            property: "name".into(),
        }],
    };
    assert_eq!(column_name(&call, 0), "toUpper(n.name)");
    let case = E::Case {
        expression: None,
        alternatives: vec![],
        default: None,
    };
    assert_eq!(column_name(&case, 3), "column_3");
    assert!(column_name(&E::String("q".repeat(500)), 0).chars().count() <= 64);
    let c = clamp(&format!("a\u{0}b{}", "é".repeat(400)));
    assert!(c.starts_with("ab") && c.ends_with('…') && c.chars().count() == MAX_DETAIL);
}
