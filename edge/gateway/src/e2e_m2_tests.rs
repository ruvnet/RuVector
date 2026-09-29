//! M2 end to end (ADR-351 §15 M2 acceptance), ove.map(|s| s.query(&sql, &[]).expect(&sql).len()) the production
//! authenticator, rate guard, routes and handlers and the in-process DO
//! backend (with the `VectorShard` shell's wipe): a default (`flat`: int8
//! scan + f32 rerank) and an opt-in `hnsw` collection of 5k × 384 each,
//! query results against brute force, the §10 per-user write budget, then
//! collection drop (every shard's rows and index chunks gone, `404` after).

use crate::api::MAX_BODY_BYTES;
use crate::e2e_world::World;
use crate::testkit::Rng;
use ruvector_edge_store::SqlStore;
use serde_json::{json, Value as Json};
use worker::Method;

const DIM: usize = 384;
const N: usize = 5_000;
const BATCH: usize = 100;
const QUERIES: usize = 20;
const RW: [&str; 2] = ["ruvector:read", "ruvector:write"];
const VECTORS: (&str, &str) = ("vectors", "id");
const CHUNKS: (&str, &str) = ("index_chunks", "part");

type Rows = [(String, Vec<f32>)];

fn ids(v: &Json) -> Vec<String> {
    v["matches"]
        .as_array()
        .unwrap_or_else(|| panic!("no matches: {v}"))
        .iter()
        .map(|m| m["id"].as_str().unwrap().to_string())
        .collect()
}

fn l2(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b)
        .map(|(x, y)| (f64::from(*x) - f64::from(*y)).powi(2))
        .sum()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f64 {
    let dot: f64 = a
        .iter()
        .zip(b)
        .map(|(x, y)| f64::from(*x) * f64::from(*y))
        .sum();
    let norm = |v: &[f32]| v.iter().map(|x| f64::from(*x).powi(2)).sum::<f64>().sqrt();
    1.0 - dot / (norm(a) * norm(b))
}

/// Exact top-10 ids by `dist` (ties by id).
fn brute_top10(rows: &Rows, q: &[f32], dist: fn(&[f32], &[f32]) -> f64) -> Vec<String> {
    let mut all: Vec<(f64, &str)> = rows
        .iter()
        .map(|(id, v)| (dist(v, q), id.as_str()))
        .collect();
    all.sort_by(|a, b| a.0.total_cmp(&b.0).then(a.1.cmp(b.1)));
    all.iter().take(10).map(|(_, id)| id.to_string()).collect()
}

/// Upsert `rows` into `name` in bodies under the 1 MiB request cap, opening
/// a new rate window before the per-user write budget (10 / 10 s) runs out.
fn ingest(w: &World, token: &str, name: &str, rows: &Rows, writes: &mut u32) {
    for chunk in rows.chunks(BATCH) {
        if *writes == 10 {
            w.new_window();
            *writes = 0;
        }
        let vectors: Vec<Json> = chunk
            .iter()
            .map(|(id, v)| json!({ "id": id, "values": v }))
            .collect();
        let body = json!({ "vectors": vectors });
        assert!(
            body.to_string().len() <= MAX_BODY_BYTES,
            "batch over the body cap"
        );
        let path = format!("/v1/collections/{name}/vectors");
        let up = w.ok(token, Method::Post, &path, body);
        *writes += 1;
        assert_eq!(up["upserted"], json!(chunk.len()), "{up}");
    }
}

fn query(w: &World, token: &str, name: &str, q: &[f32]) -> Json {
    w.new_window();
    let path = format!("/v1/collections/{name}/query");
    w.ok(
        token,
        Method::Post,
        &path,
        json!({ "vector": q, "top_k": 10 }),
    )
}

fn status(w: &World, token: &str, m: Method, path: &str, body: Json) -> (u16, String) {
    w.new_window();
    let r = w.send(token, m, path, body, None);
    let v: Json = serde_json::from_str(&r.body).unwrap_or(Json::Null);
    (r.status, v["code"].as_str().unwrap_or_default().to_string())
}

/// Rows of `table` (selecting `col`) across every shard store.
fn rows_in(w: &World, (table, col): (&str, &str)) -> usize {
    let sql = format!("SELECT {col} FROM {table}");
    w.b.shards
        .borrow()
        .values()
        .map(|s| s.query(&sql, &[]).map(|r| r.len()).unwrap_or(0))
        .sum()
}

#[test]
fn m2_flat_and_hnsw_collections_recall_rate_limit_and_drop() {
    let w = World::rate_limited("ruvector:read ruvector:write");
    let owner = w.edge_as.user_token("alice", "org-a", &w.v1(), &RW);
    let r = w.send(&owner, Method::Post, "/v1/claim", Json::Null, None);
    assert_eq!(r.status, 201, "{}", r.body);

    // `flat` is the default index; `hnsw` is opt-in.
    let flat = json!({ "name": "flat", "dim": DIM, "metric": "l2", "shards": 2 });
    let hnsw = json!({ "name": "graph", "dim": DIM, "metric": "cosine", "shards": 2,
        "index": "hnsw", "hnsw": { "m": 16, "ef_construction": 100 } });
    for spec in [flat, hnsw] {
        let r = w.send(&owner, Method::Post, "/v1/collections", spec, None);
        assert_eq!(r.status, 201, "{}", r.body);
    }

    let mut rng = Rng(0x3512);
    let rows: Vec<(String, Vec<f32>)> = (0..N).map(|i| (format!("v{i}"), rng.vec(DIM))).collect();
    let mut writes = 3; // claim + two creates
    ingest(&w, &owner, "flat", &rows, &mut writes);
    ingest(&w, &owner, "graph", &rows, &mut writes);
    for name in ["flat", "graph"] {
        w.new_window();
        let path = format!("/v1/collections/{name}");
        let g = w.ok(&owner, Method::Get, &path, Json::Null);
        let want = if name == "flat" { "flat" } else { "hnsw" };
        assert_eq!(
            (g["count"].clone(), g["shards"].clone(), g["index"].clone()),
            (json!(N), json!(2), json!(want))
        );
    }

    // Flat (int8 candidates, f32 rerank): exactly the brute-force top-10.
    // HNSW: recall@10 >= 0.95 against exact cosine.
    let mut hits = 0usize;
    for _ in 0..QUERIES {
        let q = rng.vec(DIM);
        let got = query(&w, &owner, "flat", &q);
        assert_eq!(got["shards_queried"], json!(2));
        assert_eq!(ids(&got), brute_top10(&rows, &q, l2), "flat top-10");
        let truth = brute_top10(&rows, &q, cosine_distance);
        let got = ids(&query(&w, &owner, "graph", &q));
        assert_eq!(got.len(), 10);
        hits += got.iter().filter(|id| truth.contains(id)).count();
    }
    let recall = hits as f64 / (QUERIES * 10) as f64;
    eprintln!("hnsw recall@10 over {QUERIES} queries = {recall:.3}");
    assert!(recall >= 0.95, "hnsw recall@10 = {recall}");
    // A client-tuned beam within limits is served; beyond them it is a 400.
    let q = rng.vec(DIM);
    let path = "/v1/collections/graph/query";
    let body = json!({ "vector": q, "top_k": 10, "ef": 2048 });
    assert_eq!(status(&w, &owner, Method::Post, path, body).0, 200);
    let body = json!({ "vector": q, "top_k": 10, "ef": 100_000 });
    assert_eq!(status(&w, &owner, Method::Post, path, body).0, 400);

    // §10 layer 2: 10 writes per user per 10 s, then 429 + Retry-After;
    // reads keep their own budget; a new window admits again.
    w.new_window();
    let one = |i: usize| json!({ "vectors": [{ "id": format!("rl{i}"), "values": rng_row(i) }] });
    for i in 0..10 {
        let r = w.send(
            &owner,
            Method::Post,
            "/v1/collections/flat/vectors",
            one(i),
            None,
        );
        assert_eq!(r.status, 200, "{}", r.body);
    }
    let r = w.send(
        &owner,
        Method::Post,
        "/v1/collections/flat/vectors",
        one(10),
        None,
    );
    let v: Json = serde_json::from_str(&r.body).unwrap();
    assert_eq!((r.status, v["code"].clone()), (429, json!("rate_limited")));
    assert_eq!(w.retry_after.get(), Some(10));
    let r = w.send(
        &owner,
        Method::Get,
        "/v1/collections/flat",
        Json::Null,
        None,
    );
    assert_eq!(r.status, 200, "{}", r.body);
    w.new_window();
    let r = w.send(
        &owner,
        Method::Post,
        "/v1/collections/flat/vectors",
        one(10),
        None,
    );
    assert_eq!(r.status, 200, "{}", r.body);

    // Drop both: every shard (rows, filters, ops, meta, index chunks) is
    // wiped down to its `wiped` marker, and the collection is gone.
    assert!(rows_in(&w, VECTORS) > 0 && rows_in(&w, CHUNKS) > 0);
    for name in ["graph", "flat"] {
        let path = format!("/v1/collections/{name}");
        assert_eq!(status(&w, &owner, Method::Delete, &path, Json::Null).0, 200);
        assert_eq!(status(&w, &owner, Method::Get, &path, Json::Null).0, 404);
        let qp = format!("{path}/query");
        let body = json!({ "vector": rows[0].1, "top_k": 10 });
        assert_eq!(status(&w, &owner, Method::Post, &qp, body).0, 404);
        assert_eq!(status(&w, &owner, Method::Delete, &path, Json::Null).0, 404);
    }
    for table in [VECTORS, ("filter_idx", "id"), ("ops", "seq"), CHUNKS] {
        assert_eq!(rows_in(&w, table), 0, "{} after drop", table.0);
    }
    let shards = w.b.shards.borrow().len();
    assert_eq!(rows_in(&w, ("meta", "k")), shards, "one marker per shard");
    let markers =
        w.b.shards
            .borrow()
            .values()
            .flat_map(|s| s.query("SELECT k FROM meta", &[]).unwrap())
            .all(|r| r[0].as_text() == Some("wiped"));
    assert!(markers);
    w.new_window();
    let u = w.ok(&owner, Method::Get, "/v1/usage", Json::Null)["usage"].clone();
    assert_eq!(
        (u["vectors"].clone(), u["collections"].clone()),
        (json!(0), json!(0))
    );
}

fn rng_row(i: usize) -> Vec<f32> {
    Rng(0xA11CE + i as u64).vec(DIM)
}
