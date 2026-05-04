//! End-to-end mTLS test for `GrpcTransport` (ADR-172 §1b HIGH, iter 100).
//!
//! Generates a runtime CA, issues a server cert + a valid client cert
//! signed by it, and a rogue client cert signed by an unrelated CA.
//! Stands up an `EmbeddingServer` configured with `with_client_ca(CA)`
//! (= hard mTLS, since `ServerTlsConfig::new` defaults
//! `client_auth_optional` to false), then asserts:
//!
//! 1. The valid client (cert chains to server's CA) succeeds.
//! 2. A client with NO identity fails the handshake.
//! 3. A client with a self-signed identity outside the CA's chain fails.
//!
//! Companion to `tests/tls_roundtrip.rs` (iter 99) which exercises
//! server-only TLS. Together they cover both halves of ADR-172 §1.

#![cfg(feature = "tls")]

use std::net::SocketAddr;
use std::pin::Pin;
use std::time::Duration;

use rcgen::{
    generate_simple_self_signed, BasicConstraints, CertificateParams, CertifiedKey, IsCa, KeyPair,
};
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
struct MtlsMockWorker;

#[tonic::async_trait]
impl Embedding for MtlsMockWorker {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(EmbedResponse {
            vector: vec![req.text.len() as f32, req.max_seq as f32, 13.0, 17.0],
            dim: 4,
            latency_us: 29,
        }))
    }

    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "mtls-mock".into(),
            device_id: "mtls:0".into(),
            model_fingerprint: "fp:mtls".into(),
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

/// Owned material for an issued cert (PEM-encoded cert + matching key).
struct Issued {
    cert_pem: String,
    key_pem: String,
}

/// Mint a CA, then issue a server cert (with localhost SANs) + a valid
/// client cert under that CA. Returns (ca_cert_pem, server, client).
/// Both leaf certs trace back to `ca_cert_pem`, so a server configured
/// with `with_client_ca(ca_cert_pem)` accepts the client.
fn issue_chain(server_sans: Vec<String>, client_cn: &str) -> (String, Issued, Issued) {
    // CA: self-signed, marked is_ca = Ca(unconstrained).
    let ca_key = KeyPair::generate().expect("ca keypair");
    let mut ca_params = CertificateParams::new(vec!["ruvector-test-ca".into()]).expect("ca params");
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca_cert = ca_params.self_signed(&ca_key).expect("ca self-sign");
    let ca_pem = ca_cert.pem();

    // Server cert: signed by CA, SAN = localhost / 127.0.0.1.
    let server_key = KeyPair::generate().expect("server keypair");
    let server_params = CertificateParams::new(server_sans).expect("server params");
    let server_cert = server_params
        .signed_by(&server_key, &ca_cert, &ca_key)
        .expect("server signed_by ca");

    // Client cert: signed by the same CA. The CN goes into a DNS SAN
    // because rcgen wants at least one SAN even for client certs.
    let client_key = KeyPair::generate().expect("client keypair");
    let client_params = CertificateParams::new(vec![client_cn.into()]).expect("client params");
    let client_cert = client_params
        .signed_by(&client_key, &ca_cert, &ca_key)
        .expect("client signed_by ca");

    (
        ca_pem,
        Issued {
            cert_pem: server_cert.pem(),
            key_pem: server_key.serialize_pem(),
        },
        Issued {
            cert_pem: client_cert.pem(),
            key_pem: client_key.serialize_pem(),
        },
    )
}

/// Stand up an mTLS-required `EmbeddingServer` on a random localhost port.
async fn start_mtls_mock(server: &Issued, ca_pem: &str) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let tls = TlsServer::from_pem_bytes(server.cert_pem.as_bytes(), server.key_pem.as_bytes())
        .with_client_ca_bytes(ca_pem.as_bytes());

    let server = Server::builder()
        .tls_config(tls.into_inner())
        .expect("server tls_config")
        .add_service(EmbeddingServer::new(MtlsMockWorker));

    tokio::spawn(async move {
        server.serve_with_incoming(incoming).await.ok();
    });
    tokio::time::sleep(Duration::from_millis(75)).await;
    addr
}

#[test]
fn mtls_valid_client_succeeds() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (ca_pem, server, client) = issue_chain(
        vec!["localhost".into(), "127.0.0.1".into()],
        "ruvector-test-coordinator",
    );
    let addr = server_rt.block_on(async { start_mtls_mock(&server, &ca_pem).await });

    // Client trusts the CA and presents its own CA-signed identity.
    let tls_client = TlsClient::from_pem_bytes(ca_pem.as_bytes(), "localhost")
        .expect("build client tls")
        .with_client_identity_bytes(client.cert_pem.as_bytes(), client.key_pem.as_bytes());
    let transport =
        GrpcTransport::with_tls(Duration::from_secs(2), Duration::from_secs(2), tls_client)
            .unwrap();

    let endpoint = format!("localhost:{}", addr.port());
    let worker = WorkerEndpoint::new("mtls-mock", endpoint);

    let (vec, latency) = transport
        .embed(&worker, "auth", 32)
        .expect("mTLS embed should succeed for valid client");
    assert_eq!(vec, vec![4.0, 32.0, 13.0, 17.0]);
    assert_eq!(latency, 29);
}

#[test]
fn mtls_no_client_identity_fails() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (ca_pem, server, _client) = issue_chain(
        vec!["localhost".into(), "127.0.0.1".into()],
        "ruvector-test-coordinator",
    );
    let addr = server_rt.block_on(async { start_mtls_mock(&server, &ca_pem).await });

    // Client trusts the CA but does NOT present an identity. Server
    // is configured with hard mTLS (client_auth_optional = false), so
    // the handshake must fail.
    let tls_client =
        TlsClient::from_pem_bytes(ca_pem.as_bytes(), "localhost").expect("build client tls");
    let transport =
        GrpcTransport::with_tls(Duration::from_secs(2), Duration::from_secs(2), tls_client)
            .unwrap();

    let endpoint = format!("localhost:{}", addr.port());
    let worker = WorkerEndpoint::new("mtls-mock", endpoint);

    let res = transport.embed(&worker, "auth", 32);
    assert!(
        res.is_err(),
        "mTLS server must reject anonymous client, got {:?}",
        res
    );
}

#[test]
fn mtls_rogue_client_identity_fails() {
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let (ca_pem, server, _client) = issue_chain(
        vec!["localhost".into(), "127.0.0.1".into()],
        "ruvector-test-coordinator",
    );
    let addr = server_rt.block_on(async { start_mtls_mock(&server, &ca_pem).await });

    // Rogue identity: a self-signed cert *not* under the server's CA.
    // Even though the client TLS layer happily presents it, the server
    // must reject the chain.
    let CertifiedKey {
        cert: rogue_cert,
        key_pair: rogue_key,
    } = generate_simple_self_signed(vec!["rogue.local".into()]).expect("rogue self-signed");

    let tls_client = TlsClient::from_pem_bytes(ca_pem.as_bytes(), "localhost")
        .expect("build client tls")
        .with_client_identity_bytes(
            rogue_cert.pem().as_bytes(),
            rogue_key.serialize_pem().as_bytes(),
        );
    let transport =
        GrpcTransport::with_tls(Duration::from_secs(2), Duration::from_secs(2), tls_client)
            .unwrap();

    let endpoint = format!("localhost:{}", addr.port());
    let worker = WorkerEndpoint::new("mtls-mock", endpoint);

    let res = transport.embed(&worker, "auth", 32);
    assert!(
        res.is_err(),
        "mTLS server must reject untrusted client cert, got {:?}",
        res
    );
}
