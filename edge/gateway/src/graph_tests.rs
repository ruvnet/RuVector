//! M4 rv-graph end to end (ADR-351 §15 M4): graphs and Cypher through the
//! production authenticator and routes over in-process `GraphStore` DOs —
//! read vs write classification, a viewer's mutating query `403`, the
//! read-only MCP tool, cross-tenant `404`, id counters surviving a reload,
//! and the §10 step / row budgets as `413`.

use crate::e2e_world::World;
use crate::testkit::{sub, tenant, T0};
use ruvector_edge_tenancy::Role;
use serde_json::{json, Value as Json};
use worker::Method;

const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];

fn code(w: &World, tok: &str, m: Method, path: &str, body: Json) -> (u16, String) {
    let r = w.send(tok, m, path, body, None);
    let v: Json = serde_json::from_str(&r.body).unwrap_or(Json::Null);
    (r.status, v["code"].as_str().unwrap_or_default().to_string())
}

fn cypher(w: &World, tok: &str, g: &str, q: &str) -> Json {
    w.ok(
        tok,
        Method::Post,
        &format!("/v1/graphs/{g}/cypher"),
        json!({ "query": q }),
    )
}

fn mcp(w: &World, tok: &str, name: &str, args: Json) -> (u16, Json) {
    let body = json!({ "jsonrpc": "2.0", "id": 7, "method": "tools/call", "params": { "name": name, "arguments": args } });
    let r = w.send(tok, Method::Post, "/v1/mcp", body, None);
    (
        r.status,
        serde_json::from_str(&r.body).unwrap_or(Json::Null),
    )
}

pub fn owner_world() -> (World, String) {
    let w = World::new("ruvector:read ruvector:write");
    let tok = w.edge_as.user_token("alice", "org-g", &w.v1(), &RW);
    assert_eq!(
        w.send(&tok, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    (w, tok)
}

#[test]
fn graphs_cypher_roles_tenancy_and_counters() {
    let (w, tok) = owner_world();
    let g = w.ok(
        &tok,
        Method::Post,
        "/v1/graphs",
        json!({ "name": "social" }),
    );
    assert_eq!(g["name"], json!("social"));
    assert_eq!(
        code(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs",
            json!({ "name": "social" })
        )
        .0,
        409
    );
    assert_eq!(
        code(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs",
            json!({ "name": "Bad!" })
        )
        .0,
        400
    );

    let c = cypher(
        &w,
        &tok,
        "social",
        "CREATE (a:Person {name: 'Alice'})-[r:KNOWS]->(b:Person {name: 'Bob'})",
    );
    assert_eq!(c["read_only"], json!(false));
    let m = cypher(&w, &tok, "social", "MATCH (n:Person) RETURN n.name AS name");
    assert_eq!(m["read_only"], json!(true));
    let mut names: Vec<String> = m["rows"]
        .as_array()
        .unwrap()
        .iter()
        .map(|r| {
            r["name"]
                .as_str()
                .unwrap_or_else(|| panic!("{m}"))
                .to_string()
        })
        .collect();
    names.sort();
    assert_eq!(names, ["Alice", "Bob"]);
    let list = w.ok(&tok, Method::Get, "/v1/graphs", Json::Null);
    assert_eq!(list["graphs"][0]["name"], json!("social"));

    // Counters survive an isolate restart: a CREATE after the reload gets a
    // fresh id instead of overwriting `n0` (rvlite's export/load bug).
    w.b.restart();
    cypher(&w, &tok, "social", "CREATE (c:Person {name: 'Carol'})");
    let stats = w.ok(&tok, Method::Get, "/v1/graphs/social", Json::Null);
    assert_eq!(
        (stats["nodes"].clone(), stats["edges"].clone()),
        (json!(3), json!(1)),
        "{stats}"
    );
    w.b.restart();
    let m = cypher(
        &w,
        &tok,
        "social",
        "MATCH (a:Person)-[r:KNOWS]->(b) RETURN a.name AS a, b.name AS b",
    );
    assert_eq!(m["rows"], json!([{ "a": "Alice", "b": "Bob" }]));

    // A viewer (even with ruvector:write) reads but cannot mutate: 403.
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
        cypher(&w, &val, "social", "MATCH (n) RETURN n.name AS name")["rows"]
            .as_array()
            .unwrap()
            .len(),
        3
    );
    for q in [
        "CREATE (x:Person {name: 'Mallory'})",
        "MATCH (n:Person) SET n.name = 'x'",
        "MATCH (n) DELETE n",
    ] {
        let r = code(
            &w,
            &val,
            Method::Post,
            "/v1/graphs/social/cypher",
            json!({ "query": q }),
        );
        assert_eq!(r, (403, "role_required".to_string()), "{q}");
    }
    let r = code(
        &w,
        &val,
        Method::Post,
        "/v1/graphs/social/edges",
        json!({ "edges": [["a", "b"]] }),
    );
    assert_eq!(r, (403, "role_required".to_string()));
    // Without the write scope: 403 insufficient_scope with a step-up.
    let ro = w
        .edge_as
        .user_token("alice", "org-g", &w.v1(), &["ruvector:read"]);
    let r = w.send(
        &ro,
        Method::Post,
        "/v1/graphs/social/cypher",
        json!({ "query": "CREATE (x:X)" }),
        None,
    );
    assert_eq!(r.status, 403);
    assert!(r.www_authenticate.unwrap().contains("insufficient_scope"));
    assert_eq!(
        w.ok(&tok, Method::Get, "/v1/graphs/social", Json::Null)["nodes"],
        json!(3)
    );

    // MCP: graph_query is read-only; graph_mutate needs editor.
    let vm = w.edge_as.user_token("val", "org-g", &w.mcp(), &RW);
    let (s, v) = mcp(
        &w,
        &vm,
        "graph_query",
        json!({ "graph": "social", "query": "MATCH (n:Person) RETURN n.name" }),
    );
    assert_eq!(
        (s, v["result"]["isError"].clone()),
        (200, json!(false)),
        "{v}"
    );
    let (_, v) = mcp(
        &w,
        &vm,
        "graph_query",
        json!({ "graph": "social", "query": "CREATE (x:X)" }),
    );
    assert_eq!(v["result"]["isError"], json!(true));
    let (_, v) = mcp(
        &w,
        &vm,
        "graph_mutate",
        json!({ "graph": "social", "query": "CREATE (x:X)" }),
    );
    assert!(
        v["result"]["content"][0]["text"]
            .as_str()
            .unwrap()
            .contains("role_required"),
        "{v}"
    );
    let om = w.edge_as.user_token("alice", "org-g", &w.mcp(), &RW);
    let (_, v) = mcp(
        &w,
        &om,
        "graph_mutate",
        json!({ "graph": "social", "edges": [["Alice", "Dave", 2.0]] }),
    );
    assert_eq!(v["result"]["structuredContent"]["edges"], json!(2), "{v}");
    // dry_run: validated and authorized, nothing written.
    let dry = json!({ "graph": "social", "query": "CREATE (z:Z)", "dry_run": true });
    let (_, v) = mcp(&w, &om, "graph_mutate", dry.clone());
    assert_eq!(
        v["result"]["structuredContent"]["dry_run"],
        json!(true),
        "{v}"
    );
    let (_, v) = mcp(&w, &vm, "graph_mutate", dry);
    assert_eq!(v["result"]["isError"], json!(true), "{v}");
    let stats = w.ok(&tok, Method::Get, "/v1/graphs/social", Json::Null);
    assert_eq!(stats["nodes"], json!(5), "{stats}");
    let (_, v) = mcp(&w, &om, "graph_list", json!({}));
    assert_eq!(
        v["result"]["structuredContent"]["graphs"][0]["name"],
        json!("social")
    );
    let list = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/list" });
    let v: Json =
        serde_json::from_str(&w.send(&om, Method::Post, "/v1/mcp", list, None).body).unwrap();
    let names: Vec<&str> = v["result"]["tools"]
        .as_array()
        .unwrap()
        .iter()
        .map(|t| t["name"].as_str().unwrap())
        .collect();
    for t in ["graph_list", "graph_query", "graph_mutate", "mincut"] {
        assert!(names.contains(&t), "{t}");
    }

    // Another tenant: the graph does not exist for it (404), its list is empty.
    let bob = w.edge_as.user_token("bob", "org-other", &w.v1(), &RW);
    assert_eq!(
        w.send(&bob, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    assert_eq!(
        code(&w, &bob, Method::Get, "/v1/graphs/social", Json::Null).0,
        404
    );
    let r = code(
        &w,
        &bob,
        Method::Post,
        "/v1/graphs/social/cypher",
        json!({ "query": "MATCH (n) RETURN n" }),
    );
    assert_eq!(r.0, 404);
    assert_eq!(
        w.ok(&bob, Method::Get, "/v1/graphs", Json::Null)["graphs"],
        json!([])
    );
}

#[test]
fn cypher_budgets_are_413() {
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "big" }));
    // 1,000 disjoint edges (2,000 nodes; the Free state cap keeps graphs
    // small): each undirected pattern enumerates 2,000 nodes + 2 × 1,000
    // edges, so three of them are over the 10k-step budget.
    let edges: Vec<Json> = (0..1_000)
        .map(|i| json!([format!("a{i}"), format!("b{i}")]))
        .collect();
    let mut v = Json::Null;
    for c in edges.chunks(crate::graph_store::MAX_BULK_EDGES) {
        v = w.ok(
            &tok,
            Method::Post,
            "/v1/graphs/big/edges",
            json!({ "edges": c, "label": "V" }),
        );
    }
    assert_eq!(
        (v["nodes"].clone(), v["edges"].clone()),
        (json!(2_000), json!(1_000))
    );
    let r = code(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/big/cypher",
        json!({ "query": "MATCH (a)-[r]-(b), (c)-[s]-(d), (e)-[t]-(f) RETURN a" }),
    );
    assert_eq!(r, (413, "budget_exceeded".to_string()));
    // Within the step budget but over 1k rows: 413 too.
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "rows" }));
    let edges: Vec<Json> = (0..800)
        .map(|i| json!([format!("a{i}"), format!("b{i}")]))
        .collect();
    w.ok(
        &tok,
        Method::Post,
        "/v1/graphs/rows/edges",
        json!({ "edges": edges }),
    );
    let r = code(
        &w,
        &tok,
        Method::Post,
        "/v1/graphs/rows/cypher",
        json!({ "query": "MATCH (n) RETURN n" }),
    );
    assert_eq!(r, (413, "budget_exceeded".to_string()));
    // Oversized query text and unsupported patterns are refused up front.
    let long = format!("MATCH (n) WHERE n.x = '{}' RETURN n", "y".repeat(5000));
    assert_eq!(
        code(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs/rows/cypher",
            json!({ "query": long })
        )
        .0,
        413
    );
    assert_eq!(
        code(
            &w,
            &tok,
            Method::Post,
            "/v1/graphs/rows/cypher",
            json!({ "query": "MATCH (" })
        )
        .0,
        400
    );
}

/// Measured `GraphStore` costs at the Workers-Free limits (run with
/// `--nocapture`, ideally `--release`): bulk inserts of `MAX_BULK_EDGES`
/// until the state cap refuses one (`413`, graph unchanged), then a cold
/// load of the capped graph, a label-indexed read query and the stored
/// graph's min-cut.
#[test]
fn graph_store_costs() {
    use crate::graph_store::{MAX_BULK_EDGES, MAX_STATE_BYTES};
    use std::time::Instant;
    let (w, tok) = owner_world();
    w.ok(&tok, Method::Post, "/v1/graphs", json!({ "name": "m" }));
    let ms = |t: Instant| t.elapsed().as_secs_f64() * 1e3;
    let (mut bulk, mut batches, mut last) = (0.0f64, 0u64, Json::Null);
    loop {
        let base = batches * MAX_BULK_EDGES as u64;
        let edges: Vec<Json> = (base..base + MAX_BULK_EDGES as u64)
            .map(|i| {
                json!([
                    format!("v{}", i % 1_000),
                    format!("v{}", (i * 7 + 1) % 1_000),
                    1.0
                ])
            })
            .collect();
        // The harness encodes the body before the gateway parses it; that
        // share is subtracted.
        let body = json!({ "edges": edges });
        let t = Instant::now();
        let _ = body.to_string().len();
        let encode = ms(t);
        let t = Instant::now();
        let r = w.send(&tok, Method::Post, "/v1/graphs/m/edges", body, None);
        let spent = ms(t) - encode;
        if r.status != 200 {
            let v: Json = serde_json::from_str(&r.body).unwrap();
            assert_eq!(
                (r.status, v["code"].clone()),
                (413, json!("budget_exceeded"))
            );
            break;
        }
        bulk = bulk.max(spent);
        batches += 1;
        last = serde_json::from_str(&r.body).unwrap();
        assert!(batches < 20);
    }
    // The refused batch changed nothing.
    let now = w.ok(&tok, Method::Get, "/v1/graphs/m", Json::Null);
    assert_eq!(now, last);
    assert!(last["state_bytes"].as_u64().unwrap() <= MAX_STATE_BYTES);
    w.b.restart();
    // Over LOAD_TURN_BYTES the cold load is the whole turn (503, retry).
    let q = json!({ "query": "MATCH (n:Nope) RETURN n.id AS id" });
    let t = Instant::now();
    let r = code(&w, &tok, Method::Post, "/v1/graphs/m/cypher", q);
    let cold = ms(t);
    assert_eq!(r, (503, "shard_unavailable".to_string()));
    let none = cypher(&w, &tok, "m", "MATCH (n:Nope) RETURN n.id AS id");
    assert_eq!(none["rows"], json!([]), "{none}");
    let t = Instant::now();
    cypher(&w, &tok, "m", "MATCH (n:Nope) RETURN n.id AS id");
    let warm = ms(t);
    let t = Instant::now();
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/mincut",
        json!({ "graph": "m" }),
        None,
    );
    let mincut = ms(t);
    assert!(r.status == 200 || r.status == 202, "{}", r.status);
    eprintln!(
        "graph at the Free cap ({} nodes / {} edges, {} B state): slowest {MAX_BULK_EDGES}-edge \
         bulk insert + persist {bulk:.1} ms; cold-load turn {cold:.1} ms; warm read \
         query {warm:.2} ms; graph min-cut ({}) {mincut:.1} ms",
        last["nodes"],
        last["edges"],
        last["state_bytes"],
        if r.status == 200 {
            "inline"
        } else {
            "job submit"
        },
    );
}
