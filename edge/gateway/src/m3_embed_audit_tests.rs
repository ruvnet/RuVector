//! rv-embed (text on upsert/query through the Workers AI port, batch
//! limits, metering, idempotency). Audit shipping: `m3_review_tests`.

use crate::m3_mem::*;
use ruvector_edge_auth::Capability;
use ruvector_edge_store::ErrorCode;
use serde_json::{json, Value as Json};
use worker::Method;

fn text_collection(w: &World, c: &crate::rest::Caller) -> Json {
    let body = json!({ "name": "notes", "metric": "cosine", "embedder": "bge-small-en-v1.5" });
    let (s, v) = w.req(c, Method::Post, "/v1/collections", body);
    assert_eq!(s, 201, "{v}");
    v
}

fn work_units(w: &World, c: &crate::rest::Caller) -> u64 {
    let (s, v) = w.req(c, Method::Get, "/v1/usage", Json::Null);
    assert_eq!(s, 200);
    v["work_units"]
        .as_u64()
        .or_else(|| v["work_units"]["total"].as_u64())
        .unwrap_or_else(|| v["usage"]["daily_ops"].as_u64().unwrap())
}

#[test]
fn embedder_collections_accept_text_on_upsert_and_query() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    let v = text_collection(&w, &o);
    assert_eq!(
        (v["dim"].as_u64(), v["embedder"].as_str()),
        (Some(384), Some("bge-small-en-v1.5"))
    );
    let rows: Vec<Json> = (0..120)
        .map(|i| json!({ "id": format!("n{i}"), "text": format!("note number {i} about topic {}", i % 7) }))
        .chain([json!({ "id": "raw", "values": mock_vec("raw", 0) })])
        .collect();
    let wu0 = work_units(&w, &o);
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/notes/vectors:upsert",
        json!({ "vectors": rows }),
    );
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["upserted"], 121);
    // 120 texts → two model calls (≤ 100 texts each), charged up front.
    assert_eq!((w.ai.calls.get(), w.ai.texts.get()), (2, 120));
    assert!(work_units(&w, &o) >= wu0 + 2);
    let q = json!({ "text": "note number 42 about topic 0", "top_k": 3 });
    let (s, v) = w.req(&o, Method::Post, "/v1/collections/notes/query", q);
    assert_eq!(s, 200, "{v}");
    assert_eq!(v["matches"][0]["id"], "n42");
    assert!(v["matches"][0]["distance"].as_f64().unwrap() < 1e-5);
    assert_eq!(w.ai.calls.get(), 3);
    // Stored values are the model output, bit for bit.
    let body = json!({ "ids": ["n7"], "include_values": true });
    let (_, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/notes/vectors:fetch",
        body,
    );
    let got: Vec<f32> = serde_json::from_value(v["vectors"][0]["values"].clone()).unwrap();
    let want = mock_vec("note number 7 about topic 0", 0);
    assert!(got
        .iter()
        .zip(&want)
        .all(|(a, b)| a.to_bits() == b.to_bits()));
}

#[test]
fn text_is_refused_without_embedder_or_right_before_the_model_runs() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    text_collection(&w, &o);
    let plain = json!({ "name": "plain", "dim": 384, "metric": "cosine" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", plain).0, 201);
    let row = json!({ "vectors": [{ "id": "a", "text": "hello" }] });
    let (s, v) = w.req(
        &o,
        Method::Post,
        "/v1/collections/plain/vectors:upsert",
        row.clone(),
    );
    assert_eq!(s, 400, "{v}");
    let viewer = caller("org-a", "alice", &[Capability::Read]);
    let (s, v) = w.req(
        &viewer,
        Method::Post,
        "/v1/collections/notes/vectors:upsert",
        row,
    );
    assert_eq!(s, 403);
    assert!(is(&v, ErrorCode::InsufficientScope));
    let q = json!({ "text": "hello", "top_k": 1 });
    let stranger = caller("org-a", "mallory", ALL);
    assert_eq!(
        w.req(&stranger, Method::Post, "/v1/collections/notes/query", q)
            .0,
        403
    );
    assert_eq!(w.ai.calls.get(), 0, "Workers AI never reached");
    // Limits: > 8 KiB text, > 500 texts, text + values, text + vector.
    let long = json!({ "vectors": [{ "id": "a", "text": "x".repeat(9000) }] });
    assert_eq!(
        w.req(
            &o,
            Method::Post,
            "/v1/collections/notes/vectors:upsert",
            long
        )
        .0,
        413
    );
    let many: Vec<Json> = (0..501)
        .map(|i| json!({ "id": format!("m{i}"), "text": "t" }))
        .collect();
    let many = json!({ "vectors": many });
    assert_eq!(
        w.req(
            &o,
            Method::Post,
            "/v1/collections/notes/vectors:upsert",
            many
        )
        .0,
        413
    );
    let both = json!({ "vectors": [{ "id": "a", "text": "t", "values": [0.0] }] });
    assert_eq!(
        w.req(
            &o,
            Method::Post,
            "/v1/collections/notes/vectors:upsert",
            both
        )
        .0,
        400
    );
    let q = json!({ "text": "t", "vector": [0.0], "top_k": 1 });
    assert_eq!(
        w.req(&o, Method::Post, "/v1/collections/notes/query", q).0,
        400
    );
    assert_eq!(w.ai.calls.get(), 0);
    // Embedder dimension and model are fixed.
    let bad =
        json!({ "name": "x", "dim": 128, "metric": "cosine", "embedder": "bge-small-en-v1.5" });
    let (s, v) = w.req(&o, Method::Post, "/v1/collections", bad);
    assert!(s == 400 && is(&v, ErrorCode::DimensionMismatch), "{v}");
    let bad = json!({ "name": "y", "metric": "cosine", "embedder": "gpt" });
    assert_eq!(w.req(&o, Method::Post, "/v1/collections", bad).0, 400);
    // The port failing is a retryable 503.
    w.ai.down.set(true);
    let row = json!({ "vectors": [{ "id": "a", "text": "hello" }] });
    assert_eq!(
        w.req(
            &o,
            Method::Post,
            "/v1/collections/notes/vectors:upsert",
            row
        )
        .0,
        503
    );
}

#[test]
fn text_upsert_idempotency_key_replays_despite_a_nondeterministic_model() {
    let w = World::default();
    let o = w.owner("org-a", "alice");
    text_collection(&w, &o);
    w.ai.jitter.set(true);
    let body = json!({ "vectors": [{ "id": "a", "text": "hello" }] })
        .to_string()
        .into_bytes();
    let path = "/v1/collections/notes/vectors:upsert";
    let first = w.raw(&o, Method::Post, path, body.clone(), false, Some("k-1"));
    let again = w.raw(&o, Method::Post, path, body, false, Some("k-1"));
    assert_eq!(first.0, 200, "{}", first.1);
    assert_eq!(first, again);
    assert_eq!(w.ai.calls.get(), 1, "replayed, not re-embedded");
    let other = json!({ "vectors": [{ "id": "b", "text": "bye" }] })
        .to_string()
        .into_bytes();
    let (s, v) = w.raw(&o, Method::Post, path, other, false, Some("k-1"));
    assert_eq!(s, 409);
    assert!(is(&v, ErrorCode::OpReplayed));
}
