//! Composition test for the full security stack (iter 111).
//!
//! Each ADR-172 mitigation has its own focused test:
//!
//!   §1a TLS                  -> tests/tls_roundtrip.rs
//!   §1b mTLS                 -> tests/mtls_roundtrip.rs
//!   §1c manifest signing     -> tests/stats_cli.rs (CLI-level)
//!                               + src/manifest_sig.rs (unit)
//!   §3b per-peer rate limit  -> tests/rate_limit_interceptor.rs
//!
//! What none of those exercise is **all four together** — a regression
//! in any single mitigation could break a cross-cutting interaction
//! that only surfaces when they all stack. This test stands up an
//! `EmbeddingServer` wrapped with mTLS + rate-limit interceptor +
//! reads a signed manifest, fires a real embed RPC end-to-end, and
//! then asserts a burst-induced rate-limit denial on the *same* cert
//! identity that authenticated the call.

#![cfg(feature = "tls")]

use std::net::SocketAddr;
use std::pin::Pin;
use std::sync::Arc;
use std::time::Duration;

use rcgen::{BasicConstraints, CertificateParams, IsCa, KeyPair};
use tokio::net::TcpListener;
use tokio_stream::wrappers::TcpListenerStream;
use tonic::{transport::Server, Code, Request, Response, Status};

use ruvector_hailo_cluster::manifest_sig;
use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use ruvector_hailo_cluster::rate_limit::{peer_identity, RateLimiter};
use ruvector_hailo_cluster::tls::{TlsClient, TlsServer};

#[derive(Default, Clone)]
struct ComposedMockWorker;

#[tonic::async_trait]
impl Embedding for ComposedMockWorker {
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req = request.into_inner();
        Ok(Response::new(EmbedResponse {
            vector: vec![req.text.len() as f32, 1.0, 2.0, 3.0],
            dim: 4,
            latency_us: 11,
        }))
    }
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: "compose-mock".into(),
            device_id: "compose:0".into(),
            model_fingerprint: "fp:compose".into(),
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

struct CertChain {
    ca_pem: String,
    server_cert_pem: String,
    server_key_pem: String,
    client_cert_pem: String,
    client_key_pem: String,
}

/// Mint a CA + a server cert (with localhost SANs) + a client cert.
/// Both leaf certs trace back to the CA, so a server configured with
/// `with_client_ca(ca_pem)` accepts the client.
fn issue_chain() -> CertChain {
    let ca_key = KeyPair::generate().expect("ca key");
    let mut ca_params = CertificateParams::new(vec!["compose-test-ca".into()]).expect("ca params");
    ca_params.is_ca = IsCa::Ca(BasicConstraints::Unconstrained);
    let ca_cert = ca_params.self_signed(&ca_key).expect("ca self-sign");

    let server_key = KeyPair::generate().expect("server key");
    let server_cert = CertificateParams::new(vec!["localhost".into(), "127.0.0.1".into()])
        .expect("server params")
        .signed_by(&server_key, &ca_cert, &ca_key)
        .expect("server signed_by ca");

    let client_key = KeyPair::generate().expect("client key");
    let client_cert = CertificateParams::new(vec!["compose-test-coordinator".into()])
        .expect("client params")
        .signed_by(&client_key, &ca_cert, &ca_key)
        .expect("client signed_by ca");

    CertChain {
        ca_pem: ca_cert.pem(),
        server_cert_pem: server_cert.pem(),
        server_key_pem: server_key.serialize_pem(),
        client_cert_pem: client_cert.pem(),
        client_key_pem: client_key.serialize_pem(),
    }
}

/// Stand up a server with TLS + mTLS + a rate-limit interceptor (rps=1
/// burst=2). Returns the bound addr after the listener is accepting.
async fn start_secure_server(chain: &CertChain, limiter: Arc<RateLimiter>) -> SocketAddr {
    let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
    let addr = listener.local_addr().unwrap();
    let incoming = TcpListenerStream::new(listener);

    let tls = TlsServer::from_pem_bytes(
        chain.server_cert_pem.as_bytes(),
        chain.server_key_pem.as_bytes(),
    )
    .with_client_ca_bytes(chain.ca_pem.as_bytes());

    #[allow(clippy::result_large_err)]
    let interceptor = move |req: Request<()>| -> Result<Request<()>, Status> {
        let peer = peer_identity(&req);
        if limiter.check(&peer).is_err() {
            return Err(Status::resource_exhausted(format!(
                "rate limit exceeded for {} (ADR-172 §3b composition test)",
                peer
            )));
        }
        Ok(req)
    };
    let svc = EmbeddingServer::with_interceptor(ComposedMockWorker, interceptor);

    let server = Server::builder()
        .tls_config(tls.into_inner())
        .expect("server tls_config")
        .add_service(svc);
    tokio::spawn(async move {
        server.serve_with_incoming(incoming).await.ok();
    });
    tokio::time::sleep(Duration::from_millis(75)).await;
    addr
}

#[test]
fn full_security_stack_composes_correctly() {
    // The composition under test: a coordinator authenticated via mTLS
    // makes embed calls that hit the rate-limit interceptor, all over
    // a TLS connection. Plus we reuse the iter-107 manifest_sig API to
    // prove the operator-side discovery path works alongside the
    // server-side mitigations.
    let server_rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()
        .unwrap();

    let chain = issue_chain();
    let limiter = Arc::new(RateLimiter::new(1, 2).expect("non-zero quota"));
    let addr =
        server_rt.block_on(async { start_secure_server(&chain, Arc::clone(&limiter)).await });

    // §1c side-trip: manifest_sig still works under the same crypto
    // assumptions even with the secure stack live elsewhere.
    let body = format!("pi-0 = 127.0.0.1:{}\n", addr.port());
    let sk = ed25519_dalek::SigningKey::from_bytes(&[7u8; 32]);
    let pk_hex = {
        use std::fmt::Write as _;
        let mut s = String::new();
        for b in sk.verifying_key().as_bytes() {
            write!(&mut s, "{:02x}", b).unwrap();
        }
        s
    };
    use ed25519_dalek::Signer;
    let sig_hex = {
        use std::fmt::Write as _;
        let mut s = String::new();
        for b in sk.sign(body.as_bytes()).to_bytes() {
            write!(&mut s, "{:02x}", b).unwrap();
        }
        s
    };
    manifest_sig::verify_detached(body.as_bytes(), &sig_hex, &pk_hex)
        .expect("manifest sig must verify alongside the secure server stack");

    // §1a + §1b path: build a TlsClient with our CA-issued identity.
    let tls_client = TlsClient::from_pem_bytes(chain.ca_pem.as_bytes(), "localhost")
        .expect("client tls")
        .with_client_identity_bytes(
            chain.client_cert_pem.as_bytes(),
            chain.client_key_pem.as_bytes(),
        );
    let endpoint =
        tonic::transport::Endpoint::from_shared(format!("https://localhost:{}", addr.port()))
            .unwrap()
            .connect_timeout(Duration::from_secs(2))
            .tls_config(tls_client.into_inner())
            .expect("apply tls_config");

    // First two RPCs: handshake + interceptor + embed all succeed.
    server_rt.block_on(async {
        let channel = endpoint.connect().await.expect("connect over mTLS");
        let mut client =
            ruvector_hailo_cluster::proto::embedding_client::EmbeddingClient::new(channel);
        for i in 0..2 {
            let resp = client
                .embed(tonic::Request::new(EmbedRequest {
                    text: "compose".into(),
                    max_seq: 16,
                    request_id: format!("compose-{}", i),
                }))
                .await;
            assert!(
                resp.is_ok(),
                "embed {} should succeed within burst, got {:?}",
                i,
                resp.err()
            );
            let inner = resp.unwrap().into_inner();
            assert_eq!(inner.vector, vec![7.0, 1.0, 2.0, 3.0]);
            assert_eq!(inner.dim, 4);
        }
        // Third RPC: rate-limit interceptor must trip on the SAME cert
        // identity. peer_identity hashes the leaf cert DER, so the same
        // (TLS conn, mTLS cert) tuple consistently maps to one bucket.
        let third = client
            .embed(tonic::Request::new(EmbedRequest {
                text: "compose".into(),
                max_seq: 16,
                request_id: "compose-3".into(),
            }))
            .await;
        let status = third.expect_err("3rd request must hit rate limit");
        assert_eq!(
            status.code(),
            Code::ResourceExhausted,
            "expected ResourceExhausted from §3b interceptor, got {:?}: {}",
            status.code(),
            status.message()
        );
        assert!(
            status.message().contains("ADR-172 §3b"),
            "rate-limit error should reference the ADR (helps grep): {}",
            status.message()
        );
    });

    // The limiter should be tracking exactly one peer (the cert hash).
    assert_eq!(
        limiter.tracked_peers(),
        1,
        "single client cert should map to exactly one bucket"
    );
}

#[test]
fn full_stack_still_rejects_tampered_manifest() {
    // Even with the full server stack live, the §1c gate happens on
    // the operator side — verify it short-circuits before any wire
    // traffic is attempted.
    let chain = issue_chain();
    let body = "pi-0 = 127.0.0.1:50051\n";
    let sk = ed25519_dalek::SigningKey::from_bytes(&[9u8; 32]);
    let pk_hex = {
        use std::fmt::Write as _;
        let mut s = String::new();
        for b in sk.verifying_key().as_bytes() {
            write!(&mut s, "{:02x}", b).unwrap();
        }
        s
    };
    use ed25519_dalek::Signer;
    let sig_hex = {
        use std::fmt::Write as _;
        let mut s = String::new();
        for b in sk.sign(body.as_bytes()).to_bytes() {
            write!(&mut s, "{:02x}", b).unwrap();
        }
        s
    };
    let _ = chain; // chain isn't used here; the assertion is operator-side only.

    // Original sig + pubkey verify the legit body fine.
    manifest_sig::verify_detached(body.as_bytes(), &sig_hex, &pk_hex).expect("legit body verifies");
    // Same sig + pubkey rejected when body is tampered.
    let tampered = b"pi-0 = 127.0.0.1:50051\npi-rogue = 10.0.0.99:50051\n";
    let err = manifest_sig::verify_detached(tampered, &sig_hex, &pk_hex)
        .expect_err("tampered body must fail verification");
    let msg = err.to_string();
    assert!(
        msg.contains("signature verification failed"),
        "unexpected error: {}",
        msg
    );
}
