//! `ruvector-hailo-fakeworker` — runs a configurable mock embedding
//! worker as a real binary. Lets you demo the full cluster path
//! (`ruvector-hailo-embed --workers …` → coordinator → tonic gRPC →
//! worker) on localhost today, before the actual HEF lands.
//!
//! Returns deterministic, content-derived vectors so a coordinator
//! pointed at multiple fakeworkers gets *consistent* output across
//! workers (same fingerprint, same dim) — exercises the fleet-integrity
//! path realistically.
//!
//! Env vars:
//!
//!   RUVECTOR_FAKE_BIND        listen addr           (default 0.0.0.0:50052)
//!   RUVECTOR_FAKE_DIM         vector dimensionality (default 384)
//!   RUVECTOR_FAKE_LATENCY_MS  artificial delay      (default 0)
//!   RUVECTOR_FAKE_NAME        worker name in logs   (default fakeworker)
//!   RUVECTOR_FAKE_FINGERPRINT model fp string       (default fp:fakeworker)

use std::sync::Arc;
use std::time::Duration;

use std::pin::Pin;

use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{info, instrument};

struct FakeWorker {
    name: String,
    dim: usize,
    latency: Duration,
    fingerprint: String,
    /// Iter 203 — backport of iter-199's batch-size cap for parity
    /// with the real worker. Without this, fakeworker silently
    /// processes batches of any size while the real worker rejects
    /// them — hiding regressions from any integration test that uses
    /// fakeworker as a stand-in. Same env (RUVECTOR_MAX_BATCH_SIZE)
    /// + same default (256) so deploys stay consistent.
    max_batch_size: usize,
}

#[tonic::async_trait]
impl Embedding for FakeWorker {
    #[instrument(skip(self, request), fields(text_len, latency_us, request_id))]
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        let req_id_owned = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        let req = request.into_inner();
        let req_id_field: &str = if req_id_owned.is_empty() {
            "-"
        } else {
            &req_id_owned
        };
        tracing::Span::current()
            .record("text_len", req.text.len())
            .record("request_id", req_id_field);

        if !self.latency.is_zero() {
            tokio::time::sleep(self.latency).await;
        }

        let v = deterministic_vector(&req.text, self.dim);
        let latency_us = self.latency.as_micros() as i64;
        tracing::Span::current().record("latency_us", latency_us);
        info!("fakeworker embed");
        Ok(Response::new(EmbedResponse {
            vector: v,
            dim: self.dim as u32,
            latency_us,
        }))
    }

    #[instrument(skip_all)]
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        Ok(Response::new(HealthResponse {
            version: format!("ruvector-hailo-fakeworker {}", env!("CARGO_PKG_VERSION")),
            device_id: format!("fake:{}", self.name),
            model_fingerprint: self.fingerprint.clone(),
            ready: true,
            npu_temp_ts0_celsius: 0.0,
            npu_temp_ts1_celsius: 0.0,
        }))
    }

    type EmbedStreamStream = Pin<
        Box<dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>> + Send + 'static>,
    >;

    #[instrument(skip(self, request), fields(batch_size, request_id))]
    async fn embed_stream(
        &self,
        request: Request<EmbedBatchRequest>,
    ) -> Result<Response<Self::EmbedStreamStream>, Status> {
        let req_id_owned = ruvector_hailo_cluster::proto::extract_request_id(
            &request,
            &request.get_ref().request_id,
        );
        let req = request.into_inner();
        let n = req.texts.len();
        let req_id_field: &str = if req_id_owned.is_empty() {
            "-"
        } else {
            &req_id_owned
        };
        tracing::Span::current()
            .record("batch_size", n)
            .record("request_id", req_id_field);
        // Iter 203 — same iter-199 batch-size cap as the real worker.
        // Without this, fakeworker accepted unbounded batches while
        // the real worker rejected them — hiding parity regressions
        // from integration tests that use fakeworker as a stand-in.
        if n > self.max_batch_size {
            tracing::warn!(
                batch_size = n,
                max_batch_size = self.max_batch_size,
                "fakeworker embed_stream batch too large — rejecting"
            );
            return Err(Status::invalid_argument(format!(
                "batch size {} exceeds max {} (ADR-172 §3a iter 199; \
                 tune via RUVECTOR_MAX_BATCH_SIZE)",
                n, self.max_batch_size
            )));
        }
        // Top-level info on entry — single embed has one too, so both
        // RPCs leave a visible audit trail with the same fields. The
        // span's `request_id` field is the cross-system correlation key.
        info!("fakeworker embed_stream");

        let dim = self.dim as u32;
        let latency = self.latency;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(
            req.texts.len().max(1),
        );

        tokio::task::spawn(async move {
            for (i, text) in req.texts.into_iter().enumerate() {
                if !latency.is_zero() {
                    tokio::time::sleep(latency).await;
                }
                let v = deterministic_vector(&text, dim as usize);
                let item = Ok(EmbedStreamResponse {
                    index: i as u32,
                    vector: v,
                    dim,
                    latency_us: latency.as_micros() as i64,
                });
                if tx.send(item).await.is_err() {
                    break;
                }
            }
        });

        Ok(Response::new(Box::pin(ReceiverStream::new(rx))))
    }

    #[instrument(skip_all)]
    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        // Fakeworker is for demos; doesn't track real counters.
        Ok(Response::new(StatsResponse::default()))
    }
}

/// Build a `dim`-element f32 vector that's content-derived but cheap to
/// compute — same input across fakeworkers yields the same vector,
/// which is what a real worker fleet does.
fn deterministic_vector(text: &str, dim: usize) -> Vec<f32> {
    // Seed an LCG with a non-cryptographic hash of the text bytes.
    let mut seed: u64 = 0xcbf2_9ce4_8422_2325;
    for &b in text.as_bytes() {
        seed = seed.wrapping_mul(0x100_0000_01b3) ^ (b as u64);
    }
    (0..dim)
        .map(|_| {
            seed = seed
                .wrapping_mul(6364136223846793005)
                .wrapping_add(1442695040888963407);
            // Map u64 → f32 in [-1, 1)
            ((seed >> 32) as i32 as f32) / (i32::MAX as f32)
        })
        .collect()
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let bind: std::net::SocketAddr = std::env::var("RUVECTOR_FAKE_BIND")
        .unwrap_or_else(|_| "0.0.0.0:50052".into())
        .parse()?;
    let dim: usize = std::env::var("RUVECTOR_FAKE_DIM")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(384);
    let latency_ms: u64 = std::env::var("RUVECTOR_FAKE_LATENCY_MS")
        .ok()
        .and_then(|s| s.parse().ok())
        .unwrap_or(0);
    let name = std::env::var("RUVECTOR_FAKE_NAME").unwrap_or_else(|_| "fakeworker".into());
    let fingerprint =
        std::env::var("RUVECTOR_FAKE_FINGERPRINT").unwrap_or_else(|_| "fp:fakeworker".into());

    info!(
        bind = %bind, dim, latency_ms, name = %name, fingerprint = %fingerprint,
        "ruvector-hailo-fakeworker starting"
    );

    // Iter 203 — read RUVECTOR_MAX_BATCH_SIZE so fakeworker honors the
    // same cap as the real worker (iter 199). Defaulting to 256 with a
    // floor of 1 mirrors the worker behavior exactly.
    let max_batch_size: usize = std::env::var("RUVECTOR_MAX_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256)
        .max(1);

    let svc = FakeWorker {
        name,
        dim,
        latency: Duration::from_millis(latency_ms),
        fingerprint,
        max_batch_size,
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        // Iter 192 — DoS-gate parity with the real worker. iter-180
        // through iter-184 + iter-190 layered six caps onto the gRPC
        // server (byte cap, stream cap, RPC timeout, rapid-reset cap,
        // keepalive, encode cap). fakeworker is the test-fleet stand-
        // in and was running with all defaults wide open, which meant
        // no integration test exercised the gate behavior — a future
        // change that loosened a cap on the real worker but tightened
        // it on fakeworker (or vice versa) would have escaped review.
        // Mirror the same env vars + the same defaults so a deploy
        // that runs both in the same env stays consistent.
        let max_req_bytes: usize = std::env::var("RUVECTOR_MAX_REQUEST_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024)
            .max(4 * 1024);
        let max_resp_bytes: usize = std::env::var("RUVECTOR_MAX_RESPONSE_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16 * 1024)
            .max(4 * 1024);
        let max_streams: u32 = std::env::var("RUVECTOR_MAX_CONCURRENT_STREAMS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(256)
            .max(8);
        let request_timeout_secs: u64 = std::env::var("RUVECTOR_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(30)
            .max(2);
        let max_pending_resets: usize = std::env::var("RUVECTOR_MAX_PENDING_RESETS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32)
            .max(8);
        let keepalive_secs: u64 = std::env::var("RUVECTOR_HTTP2_KEEPALIVE_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60);
        let keepalive = if keepalive_secs == 0 {
            None
        } else {
            Some(Duration::from_secs(keepalive_secs.max(10)))
        };
        info!(
            max_request_bytes = max_req_bytes,
            max_response_bytes = max_resp_bytes,
            max_concurrent_streams = max_streams,
            request_timeout_secs,
            max_pending_resets,
            http2_keepalive_secs = keepalive_secs,
            max_batch_size = svc.max_batch_size,
            "fakeworker DoS-gate parity (iter 192/203)"
        );
        let mut server = Server::builder()
            .max_concurrent_streams(Some(max_streams))
            .timeout(Duration::from_secs(request_timeout_secs))
            .http2_max_pending_accept_reset_streams(Some(max_pending_resets))
            .http2_keepalive_interval(keepalive);
        // Iter 121: TLS parity with the real worker (iter 99). Same
        // env-var contract: RUVECTOR_TLS_CERT + RUVECTOR_TLS_KEY both
        // set → TLS active. Either one alone is a misconfig and
        // halts (matches the worker's loud-fail pattern). Lets the
        // bridge integration tests stand up a TLS-enabled worker
        // without needing a separate test bin.
        #[cfg(feature = "tls")]
        {
            let cert = std::env::var("RUVECTOR_TLS_CERT").ok();
            let key = std::env::var("RUVECTOR_TLS_KEY").ok();
            match (cert, key) {
                (Some(c), Some(k)) => {
                    let mut tls = ruvector_hailo_cluster::tls::TlsServer::from_pem_files(&c, &k)
                        .map_err(|e| format!("tls server config: {}", e))?;
                    if let Ok(ca) = std::env::var("RUVECTOR_TLS_CLIENT_CA") {
                        tls = tls
                            .with_client_ca(&ca)
                            .map_err(|e| format!("client_ca: {}", e))?;
                    }
                    server = server
                        .tls_config(tls.into_inner())
                        .map_err(|e| format!("apply tls: {}", e))?;
                    info!(cert = %c, "fakeworker TLS enabled");
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "RUVECTOR_TLS_CERT and RUVECTOR_TLS_KEY must both be set or both unset"
                            .to_string(),
                    );
                }
                (None, None) => {}
            }
        }
        info!(addr = %bind, "ruvector-hailo-fakeworker serving");
        // Iter 192 — apply the byte/encode caps to the generated
        // EmbeddingServer (same pattern as iter 180/190 on the real
        // worker). `with_interceptor` would re-build the inner with
        // defaults, but fakeworker has no interceptor, so we just
        // hand the configured server straight to add_service.
        let embed_server = EmbeddingServer::new(svc)
            .max_decoding_message_size(max_req_bytes)
            .max_encoding_message_size(max_resp_bytes);
        server
            .add_service(embed_server)
            .serve_with_shutdown(bind, shutdown_signal())
            .await
            .map_err(|e| format!("serve: {}", e))?;
        Ok::<(), String>(())
    })?;

    Ok(())
}

async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let _ = Arc::new(()); // keep the unused-import lint quiet without bringing prelude in
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT");
    tokio::select! {
        _ = sigterm.recv() => info!("SIGTERM received"),
        _ = sigint.recv()  => info!("SIGINT received"),
    }
}
