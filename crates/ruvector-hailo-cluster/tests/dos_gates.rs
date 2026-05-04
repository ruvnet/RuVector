//! End-to-end test for the iter-180 gRPC byte-cap DoS gate
//! (`max_decoding_message_size`).
//!
//! Stands up an `EmbeddingServer` with a deliberately tight 4 KB cap,
//! sends an 8 KB embed text, and asserts the server rejects with
//! `Code::OutOfRange` and the error string mentions the limit. Locks
//! in iter-180 (and by extension iter-190's encoding cap, iter-181/
//! 182/183/184/192's parity work) so a future change that drops the
//! cap doesn't regress unnoticed.
//!
//! Mirrors the in-process mock pattern from `rate_limit_interceptor.rs`
//! and `tls_roundtrip.rs` — no NPU dependency, no fakeworker
//! subprocess, runs on x86 dev hosts and aarch64 Pi alike.

use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Code, Request, Response, Status};

use ruvector_hailo_cluster::proto::embedding_client::EmbeddingClient;
use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};

#[derive(Default, Clone)]
struct EchoMockWorker;

#[tonic::async_trait]
impl Embedding for EchoMockWorker {
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        // Should never reach the handler — the byte-cap rejects before
        // dispatch — but if it does, we want the test to fail loudly
        // rather than silently succeed.
        Ok(Response::new(EmbedResponse {
            vector: vec![0.0; 384],
            dim: 384,
            latency_us: 0,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "dos-mock".into(),
            device_id: "dos:0".into(),
            model_fingerprint: "fp:dos".into(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        }))
    }

    type EmbedStreamStream = Pin<
        Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>,
    >;

    async fn embed_stream(
        &self,
        _request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(1);
        drop(tx);
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse::default()))
    }
}

/// Stand up an EmbeddingServer with `max_decoding_message_size = cap_bytes`.
/// Returns the bound `SocketAddr` once the listener is accepting.
async fn start_capped_server(cap_bytes: usize) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let svc = EmbeddingServer::new(EchoMockWorker).max_decoding_message_size(cap_bytes);
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

/// Iter 194 — companion fixture for iter-190's
/// `max_encoding_message_size`. Stands up a worker whose `embed`
/// handler returns a deliberately large `Vec<f32>` so the encoding
/// cap is the only gate between the response and the wire.
#[derive(Default, Clone)]
struct OversizedResponseMockWorker;

#[tonic::async_trait]
impl Embedding for OversizedResponseMockWorker {
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        // 4 000 floats × 4 B = 16 KB raw payload, well above the
        // 4 KB encoding cap the test installs below. prost framing
        // adds a few bytes more, so the encoding will trip the cap
        // even after compression hints.
        Ok(Response::new(EmbedResponse {
            vector: vec![0.0; 4_000],
            dim: 4_000,
            latency_us: 0,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "dos-encode-mock".into(),
            device_id: "dos-encode:0".into(),
            model_fingerprint: "fp:dos-encode".into(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        }))
    }

    type EmbedStreamStream = Pin<
        Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>,
    >;

    async fn embed_stream(
        &self,
        _request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(1);
        drop(tx);
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse::default()))
    }
}

/// Iter 195 — companion fixture for iter-182's `Server::timeout`.
/// Stands up a worker whose `embed` handler sleeps `handler_sleep_ms`
/// before returning, so the test can drive the timeout middleware
/// without an actual hang.
#[derive(Clone)]
struct SlowMockWorker {
    handler_sleep_ms: u64,
}

#[tonic::async_trait]
impl Embedding for SlowMockWorker {
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        tokio::time::sleep(Duration::from_millis(self.handler_sleep_ms)).await;
        Ok(Response::new(EmbedResponse {
            vector: vec![0.0; 384],
            dim: 384,
            latency_us: (self.handler_sleep_ms * 1_000) as i64,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "slow-mock".into(),
            device_id: "slow:0".into(),
            model_fingerprint: "fp:slow".into(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        }))
    }

    type EmbedStreamStream = Pin<
        Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>,
    >;

    async fn embed_stream(
        &self,
        _request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(1);
        drop(tx);
        Ok(Response::new(Box::pin(
            tokio_stream::wrappers::ReceiverStream::new(rx),
        )))
    }

    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        Ok(Response::new(StatsResponse::default()))
    }
}

/// Iter 195 — stand up a server with `Server::timeout(timeout_ms)`
/// wrapping a `SlowMockWorker(handler_sleep_ms)`. When
/// handler_sleep > timeout the tonic tower-timeout middleware fires
/// and the client sees `Status::cancelled`. When handler_sleep <
/// timeout the request completes normally.
async fn start_timeout_server(timeout_ms: u64, handler_sleep_ms: u64) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let svc = EmbeddingServer::new(SlowMockWorker { handler_sleep_ms });
    tokio::spawn(async move {
        Server::builder()
            .timeout(Duration::from_millis(timeout_ms))
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

/// Iter 194 — stand up an EmbeddingServer with the encoding cap set
/// (mirrors `start_capped_server` for the iter-180 byte cap). Returns
/// the bound addr once the listener is accepting.
async fn start_encoding_capped_server(cap_bytes: usize) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let svc =
        EmbeddingServer::new(OversizedResponseMockWorker).max_encoding_message_size(cap_bytes);
    tokio::spawn(async move {
        Server::builder()
            .add_service(svc)
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_request_above_decoding_cap_returns_out_of_range() {
    // Cap chosen deliberately small so a tiny test payload trips it.
    // Same code path as iter-180's 64 KB production cap; only the
    // numeric value differs.
    let cap = 4 * 1024;
    let addr = start_capped_server(cap).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    // Build a payload > cap. 8 KB is comfortably over the 4 KB cap
    // even after prost framing strips a few bytes.
    let oversized: String = "x".repeat(8 * 1024);
    let req = tonic::Request::new(EmbedRequest {
        text: oversized,
        max_seq: 128,
        request_id: "dos-gates-test".into(),
    });

    let err = client
        .embed(req)
        .await
        .expect_err("oversized embed must be rejected by the byte cap");

    assert_eq!(
        err.code(),
        Code::OutOfRange,
        "byte-cap rejection should surface as OutOfRange (status code {:?}); \
         got {:?} with message {:?}",
        Code::OutOfRange,
        err.code(),
        err.message(),
    );
    let msg = err.message();
    assert!(
        msg.contains("decoded message length too large") || msg.contains(&cap.to_string()),
        "OutOfRange status should mention the limit; got message {:?}",
        msg
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_request_below_decoding_cap_succeeds() {
    // Companion to the rejection test: an under-cap payload sails
    // through, proving the cap isn't blocking legitimate traffic.
    // Same cap = 4 KB, payload = 1 KB.
    let cap = 4 * 1024;
    let addr = start_capped_server(cap).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let small: String = "x".repeat(1024);
    let req = tonic::Request::new(EmbedRequest {
        text: small,
        max_seq: 128,
        request_id: "dos-gates-test-ok".into(),
    });

    let resp = client
        .embed(req)
        .await
        .expect("under-cap embed should succeed");
    let body = resp.into_inner();
    assert_eq!(body.dim, 384, "echo mock returns dim=384");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_response_above_encoding_cap_returns_error() {
    // Iter 190's max_encoding_message_size cap. Mock worker emits a
    // 16 KB Vec<f32>; cap at 4 KB; the server-side encoder rejects
    // before the response hits the wire and surfaces an error to the
    // client. tonic returns `OutOfRange` (same shape as the decoding
    // cap) once the encoded length would exceed the limit.
    let cap = 4 * 1024;
    let addr = start_encoding_capped_server(cap).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let req = tonic::Request::new(EmbedRequest {
        text: "small request, oversized response".into(),
        max_seq: 128,
        request_id: "dos-encode-test".into(),
    });

    let err = client
        .embed(req)
        .await
        .expect_err("oversized response must be rejected by the encoding cap");

    assert_eq!(
        err.code(),
        Code::OutOfRange,
        "encoding-cap rejection should surface as OutOfRange; got {:?} \
         with message {:?}",
        err.code(),
        err.message(),
    );
    let msg = err.message();
    assert!(
        msg.contains("encoded message length too large") || msg.contains(&cap.to_string()),
        "OutOfRange status should mention the encoding limit; got message {:?}",
        msg
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_response_under_encoding_cap_succeeds() {
    // Counterpart: 16 KB cap with the same 16 KB raw response would
    // be borderline, so cap at 64 KB (production default) and the
    // 16 KB mock response sails through cleanly.
    let cap = 64 * 1024;
    let addr = start_encoding_capped_server(cap).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let req = tonic::Request::new(EmbedRequest {
        text: "small request, under-cap response".into(),
        max_seq: 128,
        request_id: "dos-encode-ok".into(),
    });

    let resp = client
        .embed(req)
        .await
        .expect("response must fit under cap");
    let body = resp.into_inner();
    assert_eq!(body.dim, 4_000, "oversized mock returns dim=4000");
    assert_eq!(body.vector.len(), 4_000, "vector length matches dim");
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_handler_exceeding_timeout_returns_cancelled() {
    // Iter 182's Server::timeout. Stand up a server with a 200 ms
    // RPC bound. Handler sleeps 1 s — well past the cap. Expect the
    // tower-timeout middleware to drop the future at the first await
    // point past the deadline; tonic surfaces this to the client as
    // `Status::cancelled`. The end-to-end path covers the full
    // tonic + tower interaction so a regression in either layer trips
    // this test.
    let timeout_ms = 200;
    let handler_sleep_ms = 1_000;
    let addr = start_timeout_server(timeout_ms, handler_sleep_ms).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2))
        // Generous client-side request timeout so we measure the
        // server-side bound, not the channel's. We expect the server
        // to fail this in ~200 ms anyway.
        .timeout(Duration::from_secs(5));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let req = tonic::Request::new(EmbedRequest {
        text: "slow".into(),
        max_seq: 16,
        request_id: "dos-timeout-test".into(),
    });

    let started = std::time::Instant::now();
    let err = client
        .embed(req)
        .await
        .expect_err("slow handler should be killed by Server::timeout");
    let elapsed = started.elapsed();

    assert_eq!(
        err.code(),
        Code::Cancelled,
        "tonic tower-timeout middleware surfaces as Cancelled; got \
         {:?} with message {:?}",
        err.code(),
        err.message(),
    );

    // Belt-and-suspenders: if the cap fired correctly, we should land
    // well under the handler's 1 s sleep. Allow generous slack for
    // CI scheduler jitter (3× the timeout).
    assert!(
        elapsed < Duration::from_millis(timeout_ms * 3),
        "request returned in {:?}, expected < {:?} (handler would have \
         taken {:?} without the cap)",
        elapsed,
        Duration::from_millis(timeout_ms * 3),
        Duration::from_millis(handler_sleep_ms),
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn embed_handler_within_timeout_succeeds() {
    // Counterpart: 1 s server timeout, 50 ms handler sleep. The
    // happy path completes well under the cap and returns a normal
    // response.
    let addr = start_timeout_server(1_000, 50).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let req = tonic::Request::new(EmbedRequest {
        text: "fast".into(),
        max_seq: 16,
        request_id: "dos-timeout-ok".into(),
    });

    let resp = client
        .embed(req)
        .await
        .expect("fast handler should complete");
    assert_eq!(resp.into_inner().dim, 384);
}
