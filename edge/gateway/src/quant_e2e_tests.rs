//! M4 rv-quant end to end: an `index = rabitq` collection (2 shards)
//! driven through the production authenticator, routes and executors —
//! REST, `/v1/ops` and MCP `vector_*` tools transparently — over the
//! in-process `QuantShard` DOs: recall, fetch, delete, cold load after an
//! isolate restart (rebuild, then snapshot + replay), a quant budget `413`
//! with the ledger refunded, and collection drop wiping every shard.

use crate::e2e_world::World;
use crate::quant_tests::{brute_top10, clustered};
use serde_json::{json, Value as Json};
use worker::Method;

const DIM: usize = 64;
const N: usize = 2_000;
const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];

fn ids(v: &Json) -> Vec<String> {
    v["matches"]
        .as_array()
        .unwrap_or_else(|| panic!("no matches: {v}"))
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

/// Query until no shard answers `503` (a rebuild still in progress); the
/// number of `503`s seen.
fn warm_up(w: &World, tok: &str, q: &[f32]) -> usize {
    for n in 0..5 {
        let r = w.send(
            tok,
            Method::Post,
            "/v1/collections/rq/query",
            json!({ "vector": q, "top_k": 1 }),
            None,
        );
        if r.status == 200 {
            return n;
        }
        assert_eq!(r.status, 503, "{}", r.body);
    }
    panic!("still loading");
}

fn recall(w: &World, tok: &str, data: &[Vec<f32>], qs: &[Vec<f32>]) -> f64 {
    let mut hits = 0;
    for q in qs {
        let truth = brute_top10(data, q);
        let got = w.ok(
            tok,
            Method::Post,
            "/v1/collections/rq/query",
            json!({ "vector": q, "top_k": 10 }),
        );
        assert_eq!(got["shards_queried"], json!(2));
        hits += ids(&got).iter().filter(|i| truth.contains(i)).count();
    }
    hits as f64 / (qs.len() * 10) as f64
}

#[test]
fn rabitq_collection_over_rest_ops_and_mcp() {
    let w = World::new("ruvector:read ruvector:write");
    let tok = w.edge_as.user_token("alice", "org-q", &w.v1(), &RW);
    let mcp = w.edge_as.user_token("alice", "org-q", &w.mcp(), &RW);
    assert_eq!(
        w.send(&tok, Method::Post, "/v1/claim", Json::Null, None)
            .status,
        201
    );
    let spec = json!({ "name": "rq", "dim": DIM, "metric": "l2", "shards": 2, "index": "rabitq" });
    let c = w.ok(&tok, Method::Post, "/v1/collections", spec);
    assert_eq!(c["index"], json!("rabitq"));

    let mut data = clustered(N + 20, DIM, 0xE2E);
    let qs = data.split_off(N);
    for (b, chunk) in data.chunks(250).enumerate() {
        let vectors: Vec<Json> = chunk
            .iter()
            .enumerate()
            .map(|(i, v)| json!({ "id": format!("v{}", b * 250 + i), "values": v, "metadata": { "i": b * 250 + i } }))
            .collect();
        let up = w.ok(
            &tok,
            Method::Post,
            "/v1/collections/rq/vectors",
            json!({ "vectors": vectors }),
        );
        assert_eq!(up["upserted"], json!(chunk.len()));
    }
    let g = w.ok(&tok, Method::Get, "/v1/collections/rq", Json::Null);
    assert_eq!(
        (g["count"].clone(), g["index"].clone()),
        (json!(N), json!("rabitq"))
    );
    let r = recall(&w, &tok, &data, &qs);
    eprintln!("rabitq e2e recall@10 (2 shards, {N} x {DIM}) = {r:.3}");
    assert!(r >= 0.95, "recall {r}");

    // `/v1/ops` and MCP land on the same shards.
    let (st, v) = w.op(
        &tok,
        "01HZX000000000000000QUANT1",
        "vector_query",
        json!({ "collection": "rq", "vector": qs[0], "top_k": 3, "include": ["metadata"] }),
    );
    assert_eq!(st, 200, "{v}");
    let top = v["result"]["matches"][0].clone();
    assert!(top["metadata"]["i"].is_u64(), "{top}");
    let call = |name: &str, args: Json| {
        let body = json!({ "jsonrpc": "2.0", "id": 1, "method": "tools/call", "params": { "name": name, "arguments": args } });
        let r = w.send(&mcp, Method::Post, "/v1/mcp", body, None);
        assert_eq!(r.status, 200, "{}", r.body);
        serde_json::from_str::<Json>(&r.body).unwrap()["result"].clone()
    };
    let got = call(
        "vector_fetch",
        json!({ "collection": "rq", "ids": ["v1", "v2", "nope"], "include_values": true }),
    );
    assert_eq!(
        got["structuredContent"]["vectors"]
            .as_array()
            .unwrap()
            .len(),
        2,
        "{got}"
    );
    let del = call(
        "vector_delete",
        json!({ "collection": "rq", "ids": ["v0", "v1"] }),
    );
    assert_eq!(del["structuredContent"]["deleted"], json!(2), "{del}");
    let q = call(
        "vector_query",
        json!({ "collection": "rq", "vector": data[1], "top_k": 1 }),
    );
    assert_ne!(q["structuredContent"]["matches"][0]["id"], json!("v1"));
    // A filter has no index on rabitq: 400 over REST.
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/collections/rq/query",
        json!({ "vector": qs[0], "top_k": 3, "filter": { "i": 1 } }),
        None,
    );
    assert_eq!(r.status, 400, "{}", r.body);

    // Isolate restart before any flush: re-encoded from the rows (2 × 999
    // rows fit one rebuild turn at 64 dims and, on Workers Paid, the same
    // turn serves the query: no `503`), then flushed by the alarm; a second
    // restart decodes the snapshot and replays nothing.
    w.b.restart();
    assert_eq!(warm_up(&w, &tok, &qs[0]), 0);
    assert!(recall(&w, &tok, &data, &qs) >= 0.95);
    w.b.m4.drain_alarms(4);
    let frames: usize =
        w.b.m4
            .quant_stores
            .borrow()
            .values()
            .map(|s| {
                ruvector_edge_store::SqlStore::query(s, "SELECT idx FROM qframes", &[])
                    .unwrap()
                    .len()
            })
            .sum();
    assert!(
        frames >= 4,
        "two shards, header + frames + footer each: {frames}"
    );
    w.b.restart();
    assert_eq!(warm_up(&w, &tok, &qs[0]), 0);
    assert!(recall(&w, &tok, &data, &qs) >= 0.95);
    let load = w.b.m4.quant_host.borrow().last_load.unwrap();
    assert!(load.snapshot_rows > 900 && !load.corrupt, "{load:?}");

    // A quant budget refusal is 413 and the ledger is refunded.
    let before = w.ok(&tok, Method::Get, "/v1/usage", Json::Null)["usage"]["vectors"].clone();
    w.b.m4.quant_host.borrow_mut().budget.max_vectors = 10;
    let one = json!({ "vectors": [{ "id": "extra", "values": qs[0] }] });
    let r = w.send(&tok, Method::Post, "/v1/collections/rq/vectors", one, None);
    assert_eq!(r.status, 413, "{}", r.body);
    assert!(r.body.contains("budget_exceeded"), "{}", r.body);
    let after = w.ok(&tok, Method::Get, "/v1/usage", Json::Null)["usage"]["vectors"].clone();
    assert_eq!(before, after);
    w.b.m4.quant_host.borrow_mut().budget = crate::quant_shard::edge_budget();

    // Drop: both quant shards wiped (usage released), then 404.
    assert_eq!(
        w.send(&tok, Method::Delete, "/v1/collections/rq", Json::Null, None)
            .status,
        200
    );
    let r = w.send(
        &tok,
        Method::Post,
        "/v1/collections/rq/query",
        json!({ "vector": qs[0], "top_k": 1 }),
        None,
    );
    assert_eq!(r.status, 404);
    let u = w.ok(&tok, Method::Get, "/v1/usage", Json::Null)["usage"].clone();
    assert_eq!(u["vectors"], json!(0), "{u}");
    for s in w.b.m4.quant_stores.borrow().values() {
        let rows = ruvector_edge_store::SqlStore::query(s, "SELECT rk FROM qrows", &[]).unwrap();
        assert!(rows.is_empty());
    }
}
