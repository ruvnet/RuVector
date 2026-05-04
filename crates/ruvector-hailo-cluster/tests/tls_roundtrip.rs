//! TLS end-to-end test for `GrpcTransport` (ADR-172 §1a HIGH, iter 99).
//!
//! Generates a self-signed cert at runtime, stands up an
//! `EmbeddingServer` over rustls, dials it through `GrpcTransport::with_tls`,
//! and asserts the embed roundtrip works. Mirrors the plaintext mock-
//! worker pattern in `grpc_transport::tests` but exercises the full
//! TLS handshake (rustls server <-> rustls client with a custom CA bundle).

#![cfg(feature = "tls")]

use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use rcgen::{generate_simple_self_signed, CertifiedKey};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Request, Response, Status};

use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use ruvector_hailo_cluster::tls::{TlsClient, TlsServer};
use ruvector_hailo_cluster::transport::{EmbeddingTransport, WorkerEndpoint};
use ruvector_hailo_cluster::GrpcTransport;

#[derive(Default, Clone)]
struct TlsMockWorker;

#[tonic::async_trait]
impl Embedding for TlsMockWorker {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(EmbedResponse {
            vector: vec![req.text.len() as f32, req.max_seq as f32, 7.0, 11.0],
            dim: 4,
            latency_us: 23,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "tls-mock".into(),
            device_id: "tls:0".into(),
            model_fingerprint: "fp:tls".into(),
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

/// Generate a fresh self-signed cert/key for `127.0.0.1` + `localhost`.
/// Returns `(cert_pem, key_pem)` — both are also valid as a CA bundle
/// since the cert self-signs itself.
fn issue_self_signed() -> (String, String) {
    let CertifiedKey { cert, key_pair } =
        generate_simple_self_signed(vec!["localhost".into(), "127.0.0.1".into()])
            .expect("rcgen self-signed");
    (cert.pem(), key_pair.serialize_pem())
}

/// Stand up an `EmbeddingServer` over TLS on a random localhost port.
/// Returns the bound `SocketAddr` once the server is accepting.
async fn start_tls_mock(cert_pem: &str, key_pem: &str) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let tls = TlsServer::from_pem_bytes(cert_pem.as_bytes(), key_pem.as_bytes());
    let server = Server::builder()
        .tls_config(tls.into_inner())
        .expect("apply server tls")
        .add_service(EmbeddingServer::new(TlsMockWorker));

    tokio::spawn(async move {
        server.serve_with_incoming(incoming).await.ok();
    });
    // Give the server a moment to start accepting before tests dial it.
    tokio::time::sleep(Duration::from_millis(50)).await;
    addr
}

#[test]
fn grpc_transport_with_tls_embeds_against_tls_mock() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (cert_pem, key_pem) = issue_self_signed();
    let addr = server_rt.block_on(async { start_tls_mock(&cert_pem, &key_pem).await });

    // Trust the same self-signed cert as a CA — it's its own issuer.
    // SNI must be one of the SANs we issued for ("localhost" or
    // "127.0.0.1"); rustls validates the SAN bytewise.
    let tls_client =
        TlsClient::from_pem_bytes(cert_pem.as_bytes(), "localhost").expect("build client tls");
    let transport =
        GrpcTransport::with_tls(Duration::from_secs(2), Duration::from_secs(2), tls_client)
            .expect("build tls transport");

    // Use the rustls SAN literal for the dialed address so SNI matches.
    let endpoint = format!("localhost:{}", addr.port());
    let worker = WorkerEndpoint::new("tls-mock", endpoint);

    let (vec, latency) = transport
        .embed(&worker, "hello", 64)
        .expect("embed should succeed over TLS");
    assert_eq!(vec, vec![5.0, 64.0, 7.0, 11.0]);
    assert_eq!(latency, 23);

    // Health roundtrip too — exercises the same channel cache.
    let h = transport.health(&worker).expect("health over TLS");
    assert_eq!(h.version, "tls-mock");
    assert!(h.ready);
}

#[test]
fn grpc_transport_plaintext_against_tls_server_fails_cleanly() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (cert_pem, key_pem) = issue_self_signed();
    let addr = server_rt.block_on(async { start_tls_mock(&cert_pem, &key_pem).await });

    // Plain GrpcTransport — no TLS configured. Dialing a TLS-only server
    // must surface as a transport error, not a panic or silent hang.
    let transport =
        GrpcTransport::with_timeouts(Duration::from_secs(2), Duration::from_secs(2)).unwrap();
    let worker = WorkerEndpoint::new("tls-mock", addr.to_string());

    let res = transport.embed(&worker, "hello", 64);
    assert!(
        res.is_err(),
        "plaintext client must fail against TLS-only server, got {:?}",
        res
    );
}
