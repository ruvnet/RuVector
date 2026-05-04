//! gRPC implementation of `EmbeddingTransport`.
//!
//! ADR-167 §5 step 14. The cluster coordinator's pick → call → record-
//! latency loop is sync (the trait), so we wrap the async tonic call in
//! a tokio runtime owned by `GrpcTransport`. Per-worker `Channel`s are
//! lazily constructed and cached to avoid TCP handshake on every call.
//!
//! End-to-end is exercised by the unit tests below: a mock worker that
//! implements `embedding_server::Embedding` is spun up on localhost,
//! `GrpcTransport` dials it, and `embed` returns the mock vector.

use crate::error::ClusterError;
use crate::proto::embedding_client::EmbeddingClient;
use crate::proto::{EmbedBatchRequest, EmbedRequest, HealthRequest, StatsRequest};
use crate::transport::{
    EmbedStreamItem, EmbeddingTransport, HealthReport, StatsSnapshot, WorkerEndpoint,
};

// `random_request_id` lives in `crate::proto` (public API); this file
// imports it via the path so existing call sites keep their bare-name
// reference.
use crate::proto::random_request_id;
use std::collections::HashMap;
use std::sync::Mutex;
use std::time::Duration;
use tokio::runtime::Runtime;
use tonic::transport::{Channel, Endpoint};

/// gRPC-backed transport. Owns a multi-thread tokio runtime + a per-worker
/// `Channel` cache so repeated calls reuse the same TCP/HTTP-2 connection.
pub struct GrpcTransport {
    runtime: Runtime,
    /// `worker.address` → connected channel. Lazily populated.
    channels: Mutex<HashMap<String, Channel>>,
    /// Connect timeout for the first call to a fresh worker.
    connect_timeout: Duration,
    /// Per-RPC deadline.
    rpc_timeout: Duration,
    /// Optional TLS config — when `Some`, every channel dialed uses
    /// `https://` scheme and rustls (ADR-172 §1a HIGH mitigation, iter 99).
    /// Constructed via [`Self::with_tls`] under the `tls` feature.
    #[cfg(feature = "tls")]
    tls: Option<crate::tls::TlsClient>,
}

impl GrpcTransport {
    /// Construct with default timeouts. Reads two env-var overrides
    /// for ops who want to tune without rebuilding clients:
    ///   RUVECTOR_CLIENT_CONNECT_TIMEOUT_MS — default 5000, floor 100
    ///   RUVECTOR_CLIENT_RPC_TIMEOUT_MS     — default 10000, floor 100
    ///
    /// Iter 208 — the previous default (2 s per RPC) was set when the
    /// worker only handled unary embeds at ~14 ms each. iter-199 raised
    /// the worker's `max_batch_size` to 256, which means a single
    /// streaming RPC can legitimately need `256 × 14 ms ≈ 3.6 s` of NPU
    /// time. The 2 s client cap turned every b≥128 batch into a guaranteed
    /// `Status::deadline_exceeded`, even though the worker would have
    /// completed the work cleanly. 10 s default gives 2.7× headroom over
    /// the worst legit batch and is well below the worker's iter-182
    /// `request_timeout=30 s` outer bound — so a real hang still surfaces
    /// to the client within the worker's own timeout window.
    pub fn new() -> Result<Self, ClusterError> {
        let connect_ms: u64 = std::env::var("RUVECTOR_CLIENT_CONNECT_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(5_000)
            .max(100);
        let rpc_ms: u64 = std::env::var("RUVECTOR_CLIENT_RPC_TIMEOUT_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(10_000)
            .max(100);
        Self::with_timeouts(
            Duration::from_millis(connect_ms),
            Duration::from_millis(rpc_ms),
        )
    }

    /// Construct with explicit connect / per-RPC timeouts. `new()` uses
    /// the defaults (5s connect, 2s per RPC) — this lets callers tune
    /// for slower-than-LAN deployments.
    pub fn with_timeouts(connect: Duration, rpc: Duration) -> Result<Self, ClusterError> {
        let runtime = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .map_err(|e| ClusterError::Transport {
                worker: "<runtime>".into(),
                reason: format!("tokio runtime: {}", e),
            })?;
        Ok(Self {
            runtime,
            channels: Mutex::new(HashMap::new()),
            connect_timeout: connect,
            rpc_timeout: rpc,
            #[cfg(feature = "tls")]
            tls: None,
        })
    }

    /// TLS-enabled constructor (ADR-172 §1a HIGH mitigation, iter 99).
    /// Available only under `--features tls`. Every channel dialed
    /// after this is constructed will be `https://` and validated against
    /// the supplied [`crate::tls::TlsClient`] CA bundle.
    #[cfg(feature = "tls")]
    pub fn with_tls(
        connect: Duration,
        rpc: Duration,
        tls: crate::tls::TlsClient,
    ) -> Result<Self, ClusterError> {
        let mut t = Self::with_timeouts(connect, rpc)?;
        t.tls = Some(tls);
        Ok(t)
    }

    /// Get-or-create a channel for the given worker address. Channel
    /// creation is async, so we block_on the runtime.
    fn channel_for(&self, worker: &WorkerEndpoint) -> Result<Channel, ClusterError> {
        // Fast path: cache hit.
        if let Some(c) = self.channels.lock().unwrap().get(&worker.address) {
            return Ok(c.clone());
        }
        // Slow path: dial. Default plaintext http://; when TLS is
        // configured we coerce to https:// regardless of how the address
        // was specified, so a stray `http://` prefix can't downgrade us.
        let raw = worker.address.as_str();
        let stripped = raw
            .strip_prefix("https://")
            .or_else(|| raw.strip_prefix("http://"))
            .unwrap_or(raw);
        #[cfg(feature = "tls")]
        let url = if self.tls.is_some() {
            format!("https://{}", stripped)
        } else {
            format!("http://{}", stripped)
        };
        #[cfg(not(feature = "tls"))]
        let url = format!("http://{}", stripped);
        let connect_to = self.connect_timeout;
        #[cfg(feature = "tls")]
        let tls_cfg = self.tls.clone();
        let channel = self
            .runtime
            .block_on(async move {
                let endpoint = Endpoint::from_shared(url)
                    .map_err(|e| format!("bad endpoint: {}", e))?
                    .connect_timeout(connect_to);
                #[cfg(feature = "tls")]
                let endpoint = if let Some(tls) = tls_cfg {
                    endpoint
                        .tls_config(tls.into_inner())
                        .map_err(|e| format!("tls_config: {}", e))?
                } else {
                    endpoint
                };
                endpoint
                    .connect()
                    .await
                    .map_err(|e| format!("connect: {}", e))
            })
            .map_err(|reason| ClusterError::Transport {
                worker: worker.name.clone(),
                reason,
            })?;
        self.channels
            .lock()
            .unwrap()
            .insert(worker.address.clone(), channel.clone());
        Ok(channel)
    }
}

impl EmbeddingTransport for GrpcTransport {
    fn embed(
        &self,
        worker: &WorkerEndpoint,
        text: &str,
        max_seq: u32,
    ) -> Result<(Vec<f32>, u64), ClusterError> {
        let request_id = random_request_id();
        self.embed_with_request_id(worker, text, max_seq, &request_id)
    }

    fn embed_with_request_id(
        &self,
        worker: &WorkerEndpoint,
        text: &str,
        max_seq: u32,
        request_id: &str,
    ) -> Result<(Vec<f32>, u64), ClusterError> {
        let channel = self.channel_for(worker)?;
        let request_id = request_id.to_string();
        // tracing's structured field — coordinator-side correlation.
        tracing::debug!(request_id = %request_id, worker = %worker.name, "dispatch embed");
        let req = EmbedRequest {
            text: text.to_string(),
            max_seq,
            request_id: request_id.clone(),
        };
        let timeout = self.rpc_timeout;
        self.runtime.block_on(async move {
            let mut client = EmbeddingClient::new(channel);
            let mut tonic_req = tonic::Request::new(req);
            tonic_req.set_timeout(timeout);
            // Mirror the proto field into the gRPC metadata header so
            // intermediaries (logging proxies, tracing middleware) can
            // see the correlation ID without parsing protobuf.
            crate::proto::inject_request_id(&mut tonic_req, &request_id);
            let resp = client
                .embed(tonic_req)
                .await
                .map_err(|s| ClusterError::Transport {
                    worker: worker.name.clone(),
                    reason: format!("embed RPC: {}", s),
                })?
                .into_inner();
            Ok((resp.vector, resp.latency_us as u64))
        })
    }

    fn health(&self, worker: &WorkerEndpoint) -> Result<HealthReport, ClusterError> {
        let channel = self.channel_for(worker)?;
        let timeout = self.rpc_timeout;
        self.runtime.block_on(async move {
            let mut client = EmbeddingClient::new(channel);
            let mut tonic_req = tonic::Request::new(HealthRequest {});
            tonic_req.set_timeout(timeout);
            let resp = client
                .health(tonic_req)
                .await
                .map_err(|s| ClusterError::Transport {
                    worker: worker.name.clone(),
                    reason: format!("health RPC: {}", s),
                })?
                .into_inner();
            // Treat 0.0 as "no reading" — older workers (pre-iter-96)
            // don't populate these fields so they default to 0.0; same
            // for workers whose chip_temperature() returned None.
            let to_opt = |c: f32| if c == 0.0 { None } else { Some(c) };
            Ok(HealthReport {
                version: resp.version,
                device_id: resp.device_id,
                model_fingerprint: resp.model_fingerprint,
                ready: resp.ready,
                npu_temp_ts0_celsius: to_opt(resp.npu_temp_ts0_celsius),
                npu_temp_ts1_celsius: to_opt(resp.npu_temp_ts1_celsius),
            })
        })
    }

    fn embed_stream(
        &self,
        worker: &WorkerEndpoint,
        texts: &[String],
        max_seq: u32,
    ) -> Result<Vec<EmbedStreamItem>, ClusterError> {
        let request_id = random_request_id();
        self.embed_stream_with_request_id(worker, texts, max_seq, &request_id)
    }

    fn embed_stream_with_request_id(
        &self,
        worker: &WorkerEndpoint,
        texts: &[String],
        max_seq: u32,
        request_id: &str,
    ) -> Result<Vec<EmbedStreamItem>, ClusterError> {
        let channel = self.channel_for(worker)?;
        let request_id = request_id.to_string();
        let req = EmbedBatchRequest {
            texts: texts.to_vec(),
            max_seq,
            request_id: request_id.clone(),
        };
        // Per-RPC deadline scaled by batch size — each item gets the
        // configured rpc_timeout, but never less than the original.
        let timeout = self.rpc_timeout * (texts.len().max(1) as u32);
        self.runtime.block_on(async move {
            use futures_core::Stream;
            use std::pin::Pin;
            let mut client = EmbeddingClient::new(channel);
            let mut tonic_req = tonic::Request::new(req);
            tonic_req.set_timeout(timeout);
            crate::proto::inject_request_id(&mut tonic_req, &request_id);
            let resp =
                client
                    .embed_stream(tonic_req)
                    .await
                    .map_err(|s| ClusterError::Transport {
                        worker: worker.name.clone(),
                        reason: format!("embed_stream RPC: {}", s),
                    })?;
            let mut stream: Pin<Box<dyn Stream<Item = _> + Send>> = Box::pin(resp.into_inner());

            let mut out = Vec::with_capacity(texts.len());
            // tokio_stream::StreamExt provides .next() on Pin<Box<Stream>>
            use tokio_stream::StreamExt;
            while let Some(item) = stream.next().await {
                let item = item.map_err(|s| ClusterError::Transport {
                    worker: worker.name.clone(),
                    reason: format!("embed_stream item: {}", s),
                })?;
                out.push(EmbedStreamItem {
                    index: item.index,
                    vector: item.vector,
                    latency_us: item.latency_us as u64,
                });
            }
            Ok(out)
        })
    }

    fn stats(&self, worker: &WorkerEndpoint) -> Result<StatsSnapshot, ClusterError> {
        let channel = self.channel_for(worker)?;
        let timeout = self.rpc_timeout;
        self.runtime.block_on(async move {
            let mut client = EmbeddingClient::new(channel);
            let mut tonic_req = tonic::Request::new(StatsRequest {});
            tonic_req.set_timeout(timeout);
            let resp = client
                .get_stats(tonic_req)
                .await
                .map_err(|s| ClusterError::Transport {
                    worker: worker.name.clone(),
                    reason: format!("stats RPC: {}", s),
                })?
                .into_inner();
            Ok(StatsSnapshot {
                embed_count: resp.embed_count,
                error_count: resp.error_count,
                health_count: resp.health_count,
                latency_sum: Duration::from_micros(resp.latency_us_sum),
                latency_min: if resp.latency_us_min == 0 && resp.embed_count == 0 {
                    None
                } else {
                    Some(Duration::from_micros(resp.latency_us_min))
                },
                latency_max: if resp.latency_us_max == 0 && resp.embed_count == 0 {
                    None
                } else {
                    Some(Duration::from_micros(resp.latency_us_max))
                },
                uptime: Duration::from_secs(resp.uptime_seconds),
                rate_limit_denials: resp.rate_limit_denials,
                rate_limit_tracked_peers: resp.rate_limit_tracked_peers,
            })
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::proto::embedding_server::{Embedding, EmbeddingServer};
    use crate::proto::{
        EmbedBatchRequest, EmbedResponse, EmbedStreamResponse, HealthResponse, StatsRequest,
        StatsResponse,
    };

    // `random_request_id` tests live in `proto::tests` since the
    // function moved there in iter 65; keep this file focused on
    // GrpcTransport channel + dispatch behavior.
    use std::net::SocketAddr;
    use tokio::net::TcpListener;
    use tonic::{transport::Server, Request, Response, Status};

    /// Mock worker that returns a deterministic vector, used to validate
    /// the GrpcTransport end-to-end without needing the actual NPU.
    #[derive(Clone, Default)]
    struct MockWorker {
        device_id: String,
        fingerprint: String,
    }

    #[tonic::async_trait]
    impl Embedding for MockWorker {
        async fn embed(
            &self,
            request: Request<EmbedRequest>,
        ) -> Result<Response<EmbedResponse>, Status> {
            let req = request.into_inner();
            // Deterministic: vector of length 4, components encode the
            // text length and max_seq for the test to verify.
            let v = vec![
                req.text.len() as f32,
                req.text.chars().count() as f32,
                req.max_seq as f32,
                42.0,
            ];
            Ok(Response::new(EmbedResponse {
                vector: v,
                dim: 4,
                latency_us: 17,
            }))
        }
        async fn health(
            &self,
            _request: Request<HealthRequest>,
        ) -> Result<Response<HealthResponse>, Status> {
            Ok(Response::new(HealthResponse {
                version: "mock 0.0.1".into(),
                device_id: self.device_id.clone(),
                model_fingerprint: self.fingerprint.clone(),
                ready: true,
                npu_temp_ts0_celsius: 0.0,
                npu_temp_ts1_celsius: 0.0,
            }))
        }
        async fn get_stats(
            &self,
            _request: Request<StatsRequest>,
        ) -> Result<Response<StatsResponse>, Status> {
            Ok(Response::new(StatsResponse::default()))
        }
        type EmbedStreamStream = std::pin::Pin<
            Box<
                dyn futures_core::Stream<Item = Result<EmbedStreamResponse, Status>>
                    + Send
                    + 'static,
            >,
        >;
        async fn embed_stream(
            &self,
            _request: Request<EmbedBatchRequest>,
        ) -> Result<Response<Self::EmbedStreamStream>, Status> {
            // Mock test workers don't drive batched flows; return empty stream.
            let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(1);
            drop(tx);
            Ok(Response::new(Box::pin(
                tokio_stream::wrappers::ReceiverStream::new(rx),
            )))
        }
    }

    /// Spin up a mock worker on a random localhost port. Returns the
    /// SocketAddr so the test can dial it. Server runs on the supplied
    /// runtime for the lifetime of the test.
    fn start_mock(rt: &Runtime) -> SocketAddr {
        rt.block_on(async {
            let listener = TcpListener::bind("127.0.0.1:0").await.unwrap();
            let addr = listener.local_addr().unwrap();
            let listener_stream = tokio_stream_from_listener(listener);
            tokio::spawn(async move {
                Server::builder()
                    .add_service(EmbeddingServer::new(MockWorker {
                        device_id: "mock:0".into(),
                        fingerprint: "fp:mock".into(),
                    }))
                    .serve_with_incoming(listener_stream)
                    .await
                    .ok();
            });
            // Give the server a moment to start accepting.
            tokio::time::sleep(Duration::from_millis(50)).await;
            addr
        })
    }

    /// Adapter: tokio TcpListener → tonic-friendly Stream. Uses
    /// `tokio-stream::wrappers::TcpListenerStream` which yields
    /// `Result<TcpStream, io::Error>` items as tonic expects.
    fn tokio_stream_from_listener(
        listener: TcpListener,
    ) -> tokio_stream::wrappers::TcpListenerStream {
        tokio_stream::wrappers::TcpListenerStream::new(listener)
    }

    /// Each test owns its own runtime so the mock server lifetime is
    /// scoped — avoids cross-test interference.
    fn fresh_runtime() -> Runtime {
        tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap()
    }

    #[test]
    fn grpc_transport_embed_roundtrip_against_mock() {
        let server_rt = fresh_runtime();
        let addr = start_mock(&server_rt);

        let transport = GrpcTransport::new().unwrap();
        let worker = WorkerEndpoint::new("mock", addr.to_string());

        let (vec, latency) = transport
            .embed(&worker, "hello world", 128)
            .expect("embed should succeed against mock worker");
        assert_eq!(vec, vec![11.0, 11.0, 128.0, 42.0]);
        assert_eq!(latency, 17);
    }

    #[test]
    fn grpc_transport_health_returns_mock_metadata() {
        let server_rt = fresh_runtime();
        let addr = start_mock(&server_rt);

        let transport = GrpcTransport::new().unwrap();
        let worker = WorkerEndpoint::new("mock", addr.to_string());

        let h = transport.health(&worker).unwrap();
        assert_eq!(h.version, "mock 0.0.1");
        assert_eq!(h.device_id, "mock:0");
        assert_eq!(h.model_fingerprint, "fp:mock");
        assert!(h.ready);
    }

    #[test]
    fn grpc_transport_caches_channel_across_calls() {
        let server_rt = fresh_runtime();
        let addr = start_mock(&server_rt);

        let transport = GrpcTransport::new().unwrap();
        let worker = WorkerEndpoint::new("mock", addr.to_string());

        for _ in 0..5 {
            transport.embed(&worker, "x", 4).unwrap();
        }
        // After 5 calls, only one cached channel.
        let n = transport.channels.lock().unwrap().len();
        assert_eq!(n, 1);
    }
}
