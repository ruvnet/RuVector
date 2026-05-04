//! End-to-end test for the per-peer rate limit interceptor (ADR-172
//! §3b iter 104).
//!
//! Stands up an `EmbeddingServer` wrapped by `with_interceptor(...)`
//! using the lib's `RateLimiter` (rps=1, burst=2). Fires 3 sequential
//! embed RPCs and asserts the 3rd surfaces as `Code::ResourceExhausted`.
//! Mirrors the mock-worker pattern from `tls_roundtrip.rs` /
//! `mtls_roundtrip.rs` — no NPU dependency, runs on x86 dev hosts.

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
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
use ruvector_hailo_cluster::rate_limit::{peer_identity, RateLimiter};

#[derive(Default, Clone)]
struct CountingMockWorker;

#[tonic::async_trait]
impl Embedding for CountingMockWorker {
    async fn embed(
        &self,
        _request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        Ok(Response::new(EmbedResponse {
            vector: vec![1.0],
            dim: 1,
            latency_us: 1,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "rl-mock".into(),
            device_id: "rl:0".into(),
            model_fingerprint: "fp:rl".into(),
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

/// Spin up a server with the rate-limit interceptor active. Returns the
/// bound `SocketAddr` once the listener is accepting.
async fn start_rate_limited_server(rps: u32, burst: u32) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let limiter = Arc::new(RateLimiter::new(rps, burst).expect("non-zero quota"));
    #[allow(clippy::result_large_err)]
    let interceptor = move |req: Request<()>| -> Result<Request<()>, Status> {
        let peer = peer_identity(&req);
        if limiter.check(&peer).is_err() {
            return Err(Status::resource_exhausted(format!(
                "rate limit exceeded for {} (ADR-172 §3b)",
                peer
            )));
        }
        Ok(req)
    };
    let intercepted = EmbeddingServer::with_interceptor(CountingMockWorker, interceptor);

    tokio::spawn(async move {
        Server::builder()
            .add_service(intercepted)
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn third_request_in_burst_2_returns_resource_exhausted() {
    let addr = start_rate_limited_server(1, 2).await;
    let endpoint = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2));
    let channel = endpoint.connect().await.expect("connect");
    let mut client = EmbeddingClient::new(channel);

    let req = || {
        tonic::Request::new(EmbedRequest {
            text: "hello".into(),
            max_seq: 16,
            request_id: "".into(),
        })
    };

    // Burst capacity of 2 → first two through.
    for i in 0..2 {
        let resp = client.embed(req()).await;
        assert!(
            resp.is_ok(),
            "request {} should succeed within burst, got {:?}",
            i,
            resp.err()
        );
    }

    // 3rd request — same TCP connection, same peer IP — gets capped.
    let third = client.embed(req()).await;
    let status = third.expect_err("3rd request must be rate-limited");
    assert_eq!(
        status.code(),
        Code::ResourceExhausted,
        "expected ResourceExhausted, got {:?}",
        status.code()
    );
    assert!(
        status.message().contains("ADR-172 §3b"),
        "error message should reference the ADR for grep-ability: {:?}",
        status.message()
    );
}

#[tokio::test(flavor = "multi_thread", worker_threads = 2)]
async fn rate_limit_off_path_passes_unrestricted_traffic() {
    // Sanity check: when no rate limiter is installed, the same
    // burst sequence runs without any ResourceExhausted.
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    tokio::spawn(async move {
        Server::builder()
            .add_service(EmbeddingServer::new(CountingMockWorker))
            .serve_with_incoming(incoming)
            .await
            .ok();
    });
    tokio::time::sleep(Duration::from_millis(50)).await;

    let channel = tonic::transport::Endpoint::from_shared(format!("http://{}", addr))
        .unwrap()
        .connect_timeout(Duration::from_secs(2))
        .connect()
        .await
        .expect("connect");
    let mut client = EmbeddingClient::new(channel);

    for _ in 0..10 {
        let resp = client
            .embed(tonic::Request::new(EmbedRequest {
                text: "hello".into(),
                max_seq: 16,
                request_id: "".into(),
            }))
            .await;
        assert!(
            resp.is_ok(),
            "no rate limiter -> never throttled: {:?}",
            resp.err()
        );
    }
}
