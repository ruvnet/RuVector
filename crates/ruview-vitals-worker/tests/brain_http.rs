//! End-to-end HTTP integration test for `ruview-mcp-brain-mini`
//! (ADR-183 Tier 2 iter 12).
//!
//! Spins the brain up in-process on an ephemeral TCP port, drives it
//! with the same `BrainClient` workers use, then asserts the GET path
//! returns what was POSTed and that body / field caps reject misuse
//! with the right status codes.

use std::sync::Arc;

use ruview_vitals_worker::brain::BrainClient;
use ruview_vitals_worker::mcp_brain::{build_app, Store, DEFAULT_BODY_LIMIT_BYTES};
use tokio::net::TcpListener;
use tokio::sync::RwLock;

/// Boot the brain on `127.0.0.1:0` and return its concrete URL plus a
/// [`tokio::task::JoinHandle`] you can drop to stop it.
async fn spawn_brain(
    body_limit: usize,
) -> (String, tokio::task::JoinHandle<()>) {
    let store = Arc::new(RwLock::new(Store::default()));
    let app = build_app(store, body_limit);
    let listener = TcpListener::bind("127.0.0.1:0").await.expect("bind");
    let addr = listener.local_addr().expect("local_addr");
    let url = format!("http://{}", addr);
    let handle = tokio::spawn(async move {
        let _ = axum::serve(listener, app).await;
    });
    (url, handle)
}

#[tokio::test]
async fn post_and_list_roundtrip() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    let client = BrainClient::new(url.clone(), "test-host".into()).unwrap();

    client
        .post_memory("spatial-vitals", "wifi vitals node 7 valid")
        .await
        .expect("POST 1");
    client
        .post_memory("spatial-vitals", "wifi vitals node 8 valid")
        .await
        .expect("POST 2");
    client
        .post_memory("spatial-noise", "should not match category filter")
        .await
        .expect("POST 3");

    // Direct GET via reqwest — BrainClient is POST-only.
    let http = reqwest::Client::new();
    let body: serde_json::Value = http
        .get(format!("{url}/memories"))
        .query(&[("category", "spatial-vitals"), ("limit", "10")])
        .send()
        .await
        .expect("GET")
        .error_for_status()
        .expect("200")
        .json()
        .await
        .expect("json");

    assert_eq!(body["count"], 2);
    assert_eq!(body["total"], 3);
    let memories = body["memories"].as_array().expect("array");
    assert_eq!(memories.len(), 2);
    // Reverse-chronological by spec: newest first.
    assert_eq!(
        memories[0]["content"], "wifi vitals node 8 valid",
        "newest reading should come first"
    );
    assert_eq!(memories[0]["category"], "spatial-vitals");
    assert!(
        memories[0]["content_hash"]
            .as_str()
            .map_or(false, |s| s.len() == 64),
        "content_hash should be 64 hex chars"
    );
    assert!(
        memories[0]["id"]
            .as_str()
            .map_or(false, |s| s.len() == 32),
        "id should be 32 hex chars"
    );
}

#[tokio::test]
async fn rejects_oversize_content_with_413() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    let oversize = "A".repeat(9 * 1024); // > MAX_CONTENT_LEN
    let payload = serde_json::json!({
        "category": "x",
        "content": oversize,
    });
    let http = reqwest::Client::new();
    let r = http
        .post(format!("{url}/memories"))
        .json(&payload)
        .send()
        .await
        .expect("send");
    assert_eq!(
        r.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "expected 413 for oversize content"
    );
}

#[tokio::test]
async fn rejects_huge_body_via_layer() {
    // Body cap is enforced at the layer; pass a payload that exceeds
    // it AND would also fail per-field, so we get a hard 413 from the
    // body limit (not our handler).
    let (url, _h) = spawn_brain(2 * 1024).await;
    let oversize = "A".repeat(10 * 1024);
    let payload = serde_json::json!({
        "category": "x",
        "content": oversize,
    });
    let http = reqwest::Client::new();
    let r = http
        .post(format!("{url}/memories"))
        .json(&payload)
        .send()
        .await
        .expect("send");
    assert_eq!(
        r.status(),
        reqwest::StatusCode::PAYLOAD_TOO_LARGE,
        "expected 413 from DefaultBodyLimit layer"
    );
}

#[tokio::test]
async fn rejects_empty_content_with_400() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    let payload = serde_json::json!({
        "category": "x",
        "content": "",
    });
    let http = reqwest::Client::new();
    let r = http
        .post(format!("{url}/memories"))
        .json(&payload)
        .send()
        .await
        .expect("send");
    assert_eq!(r.status(), reqwest::StatusCode::BAD_REQUEST);
}

#[tokio::test]
async fn rejects_missing_field_with_422() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    // axum's Json extractor returns 422 for malformed body; our
    // handler never runs.
    let payload = serde_json::json!({"category": "x"});
    let http = reqwest::Client::new();
    let r = http
        .post(format!("{url}/memories"))
        .json(&payload)
        .send()
        .await
        .expect("send");
    assert_eq!(r.status(), reqwest::StatusCode::UNPROCESSABLE_ENTITY);
}

#[tokio::test]
async fn health_returns_ok() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    let r = reqwest::get(format!("{url}/health"))
        .await
        .expect("GET /health");
    assert!(r.status().is_success());
    assert_eq!(r.text().await.expect("text"), "ok");
}

#[tokio::test]
async fn category_filter_limits_results() {
    let (url, _h) = spawn_brain(DEFAULT_BODY_LIMIT_BYTES).await;
    let client = BrainClient::new(url.clone(), "test-host".into()).unwrap();
    for i in 0..5 {
        client
            .post_memory("vital", &format!("v{i}"))
            .await
            .unwrap();
    }
    for i in 0..3 {
        client
            .post_memory("noise", &format!("n{i}"))
            .await
            .unwrap();
    }

    let http = reqwest::Client::new();
    let body: serde_json::Value = http
        .get(format!("{url}/memories"))
        .query(&[("category", "vital"), ("limit", "100")])
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["count"], 5);
    assert_eq!(body["total"], 8);

    // No category filter — get the whole tape.
    let body: serde_json::Value = http
        .get(format!("{url}/memories"))
        .query(&[("limit", "100")])
        .send()
        .await
        .unwrap()
        .json()
        .await
        .unwrap();
    assert_eq!(body["count"], 8);
    assert_eq!(body["total"], 8);
}
