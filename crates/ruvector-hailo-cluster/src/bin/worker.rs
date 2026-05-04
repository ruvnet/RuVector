//! `ruvector-hailo-worker` — gRPC server that serves text embedding via
//! a Hailo-8 NPU.
//!
//! ADR-167 §5 step 10 (worker side of the cluster). On the Pi 5 + AI HAT+
//! this wraps the local `HailoEmbedder` and serves the same
//! `embedding_server::Embedding` trait that `ruvector-hailo-cluster`'s
//! `GrpcTransport` calls into.
//!
//! Env vars:
//!   RUVECTOR_WORKER_BIND   socket addr to listen on   (default 0.0.0.0:50051)
//!   RUVECTOR_MODEL_DIR     dir holding either:
//!                            * NPU path:  model.hef + vocab.txt + special_tokens.json
//!                            * cpu-fallback (iter 134): model.safetensors + tokenizer.json + config.json
//!                          Worker auto-detects which is present.
//!   RUVECTOR_HEF_SHA256    optional sha256 hex digest pin (iter 174).
//!                          When set, worker hashes model.hef at boot
//!                          and refuses to start on mismatch. Costs
//!                          ~16ms on Pi 5 NEON; layered with iter-173
//!                          magic check + iter-143 cluster fingerprint.
//!                          (default ./models/all-minilm-l6-v2)
//!   RUVECTOR_TLS_CERT      path to PEM server cert        (TLS — feature `tls`)
//!   RUVECTOR_TLS_KEY       path to PEM server private key (TLS — feature `tls`)
//!   RUVECTOR_TLS_CLIENT_CA path to PEM client CA bundle   (mTLS — optional)
//!   RUVECTOR_LOG_TEXT_CONTENT  embed text audit mode: none|hash|full
//!                              (ADR-172 §3c — default none = no leak)
//!   RUVECTOR_RATE_LIMIT_RPS    Per-peer requests/sec quota; 0 = disabled
//!                              (ADR-172 §3b — default 0)
//!   RUVECTOR_RATE_LIMIT_BURST  Optional burst capacity; defaults to RPS
//!                              if unset (only used when RPS > 0)
//!   RUVECTOR_MAX_REQUEST_BYTES gRPC max_decoding_message_size cap
//!                              (ADR-172 §3a iter 180 — default 65536,
//!                              floor 4096). Caps per-RPC alloc surface
//!                              well below tonic's ~4 MB transport
//!                              default to shrink the DoS surface.
//!   RUVECTOR_MAX_RESPONSE_BYTES gRPC max_encoding_message_size cap
//!                              (ADR-172 §3a iter 190 — default 16384,
//!                              floor 4096). Defense-in-depth on the
//!                              response side: the worker should never
//!                              emit > ~1.6 KB per embed, but capping
//!                              the encode budget bounds the blast
//!                              radius of any hypothetical leak.
//!   RUVECTOR_MAX_CONCURRENT_STREAMS  HTTP/2 SETTINGS_MAX_CONCURRENT_STREAMS
//!                              (ADR-172 §3a iter 181 — default 256,
//!                              floor 8). Caps in-flight streams per
//!                              connection so a single attacker socket
//!                              can't pump unbounded RPCs at the worker.
//!   RUVECTOR_REQUEST_TIMEOUT_SECS  Per-RPC handler timeout
//!                              (ADR-172 §3a iter 182 — default 30,
//!                              floor 2). Bounds slow-loris attacks
//!                              and any handler that hangs past the
//!                              30× p99 headroom. tonic returns
//!                              Status::cancelled when it fires.
//!   RUVECTOR_MAX_PENDING_RESETS  CVE-2023-44487 rapid-reset cap
//!                              (ADR-172 §3a iter 183 — default 32,
//!                              floor 8). Caps unprocessed RST_STREAM
//!                              frames; once exceeded, the server
//!                              sends GOAWAY and closes the connection.
//!   RUVECTOR_HTTP2_KEEPALIVE_SECS  HTTP/2 keepalive ping interval
//!                              (ADR-172 §3a iter 184 — default 60,
//!                              floor 10, 0 = disabled). Reclaims
//!                              half-closed TCP state from crashed or
//!                              partitioned clients; pong timeout is
//!                              hyper's default 20 s.
//!   RUVECTOR_MAX_BATCH_SIZE   Cap on EmbedBatchRequest.texts.len()
//!                              (ADR-172 §3a iter 199 — default 256,
//!                              floor 1). Bounds the per-RPC NPU work
//!                              an attacker can force; iter-180 byte
//!                              cap alone allowed ~16 k tightly-packed
//!                              tiny texts inside a single request,
//!                              which would tie up the worker for
//!                              minutes. Returns InvalidArgument when
//!                              exceeded.
//!   RUVECTOR_SHUTDOWN_FORCE_CLEAN  When "1", attempt the clean
//!                              HailoRT teardown on shutdown
//!                              (iter 185 — default off). The default
//!                              path forgets the embedder + calls
//!                              process::exit(0) because every
//!                              attempt at a clean drop SEGV'd in
//!                              HailoRT's internal teardown; the OS
//!                              reaps fds + mmaps either way.
//!   RUVECTOR_SHUTDOWN_DRAIN_MS  Drain pause used only when
//!                              FORCE_CLEAN=1 (default 500). Reserved
//!                              for the eventual HailoRT release that
//!                              fixes the SEGV upstream.
//!
//! When both `RUVECTOR_TLS_CERT` and `RUVECTOR_TLS_KEY` are set and the
//! binary was built with `--features tls`, the worker serves over HTTPS
//! (rustls). Otherwise it falls back to plaintext gRPC and assumes the
//! caller (e.g. Tailscale) handles transport security. ADR-172 §1a.
//!
//! Without the `hailo` feature, `HailoEmbedder::open()` returns
//! `FeatureDisabled` and the worker exits with a clear message — useful
//! to validate the binary builds + arg parsing without a Pi attached.

use std::path::PathBuf;
use std::sync::Arc;
use std::time::{Duration, Instant};

use std::sync::atomic::{AtomicU64, Ordering};

use std::pin::Pin;

use ruvector_hailo::HailoEmbedder;
use ruvector_hailo_cluster::compute_fingerprint;
use ruvector_hailo_cluster::proto::embedding_server::{Embedding, EmbeddingServer};
use ruvector_hailo_cluster::proto::{
    EmbedBatchRequest, EmbedRequest, EmbedResponse, EmbedStreamResponse, HealthRequest,
    HealthResponse, StatsRequest, StatsResponse,
};
use tokio_stream::wrappers::ReceiverStream;
use tonic::{transport::Server, Request, Response, Status};
use tracing::{error, info, instrument, warn};

use ruvector_hailo_cluster::rate_limit::{peer_identity, RateLimiter};

/// Tracing / audit-log mode for embed text content (ADR-172 §3c iter 103).
/// Default is `None` so we don't leak text content into logs by default;
/// operators opt into `Hash` for cross-system correlation or `Full` for
/// debug environments where text exposure is acceptable.
#[derive(Clone, Copy, Debug)]
enum LogTextContent {
    /// No text content in logs (only `text_len`). Default.
    None,
    /// First 16 hex chars of sha256(text). Non-reversible correlation.
    Hash,
    /// Full text content. Debug only — never recommended for production.
    Full,
}

impl LogTextContent {
    fn parse(s: &str) -> Result<Self, String> {
        match s {
            "none" | "" => Ok(Self::None),
            "hash" => Ok(Self::Hash),
            "full" => Ok(Self::Full),
            other => Err(format!(
                "RUVECTOR_LOG_TEXT_CONTENT must be one of none|hash|full, got {:?}",
                other
            )),
        }
    }

    /// Render the text in the configured mode. Returns "-" for `None`
    /// so the tracing field is always populated (downstream parsers
    /// don't have to handle missing-field cases).
    fn render(self, text: &str) -> String {
        match self {
            Self::None => "-".to_string(),
            Self::Hash => {
                use sha2::{Digest, Sha256};
                let mut h = Sha256::new();
                h.update(text.as_bytes());
                let d = h.finalize();
                let mut hex = String::with_capacity(16);
                for b in &d.as_slice()[..8] {
                    use std::fmt::Write as _;
                    write!(&mut hex, "{:02x}", b).unwrap();
                }
                hex
            }
            Self::Full => text.to_string(),
        }
    }
}

/// The actual gRPC service. Holds a thread-safe HailoEmbedder.
struct WorkerService {
    embedder: Arc<HailoEmbedder>,
    /// Server-reported version string — used by the coordinator's health
    /// check and surfaced in logs.
    version: String,
    /// PCIe BDF of the device the embedder opened — surfaces to clients
    /// for debugging which fleet member served them.
    device_id: String,
    /// sha256(HEF || vocab.txt) — coordinator refuses to mix workers with
    /// different fingerprints (ADR-167 §8.3 fleet integrity guard).
    /// Phase 1 ships an empty string until step 6 (HEF) lands; coordinator
    /// treats empty fingerprint as "skip the check".
    fingerprint: String,
    /// ADR-172 §3c iter-103 audit-log mode for embed text content.
    log_text_content: LogTextContent,
    /// ADR-172 §3b iter-104/105: optional per-peer rate limiter. Read
    /// here so `get_stats` can surface `tracked_peers()` without holding
    /// a separate Arc.
    rate_limiter: Arc<Option<RateLimiter>>,
    /// Iter-105: shared denial counter — bumped by the interceptor
    /// closure on every Status::resource_exhausted, read by `get_stats`.
    rate_limit_denials: Arc<AtomicU64>,
    /// Iter 199 — cap on the number of texts in a single
    /// `embed_stream` (EmbedBatchRequest) RPC. The iter-180 byte cap
    /// bounds the encoded request size at 64 KB, but tightly packed
    /// 1-byte texts can fit ~16 k entries inside that — and each
    /// entry triggers a serial ~14 ms NPU embed, holding the connection
    /// for ~228 s. Capping the batch length closes that loop without
    /// affecting any legitimate caller (iter-179 streaming sweep
    /// peaked at b=16). Env: RUVECTOR_MAX_BATCH_SIZE.
    max_batch_size: usize,
    /// Process start time, for uptime reporting in GetStats.
    start: Instant,
    /// Atomic counters surfaced via GetStats.
    embed_ok: AtomicU64,
    embed_err: AtomicU64,
    health_count: AtomicU64,
    latency_sum_us: AtomicU64,
    latency_min_us: AtomicU64,
    latency_max_us: AtomicU64,
}

#[tonic::async_trait]
impl Embedding for WorkerService {
    #[instrument(
        skip(self, request),
        fields(text_len, text_content, latency_us, dim, request_id)
    )]
    async fn embed(
        &self,
        request: Request<EmbedRequest>,
    ) -> Result<Response<EmbedResponse>, Status> {
        // Prefer the gRPC metadata header (canonical) over the proto
        // field (back-compat). Defaults to "-" when neither is set.
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
        // ADR-172 §3c iter 103: render text content per configured mode.
        // Default mode is None → "-", so legacy log scrapers see a
        // populated field but get no leakable content.
        let text_content_field = self.log_text_content.render(&req.text);
        tracing::Span::current()
            .record("text_len", req.text.len())
            .record("text_content", text_content_field.as_str())
            .record("request_id", req_id_field);

        let start = Instant::now();
        match self.embedder.embed(&req.text) {
            Ok(v) => {
                let dim = self.embedder.dimensions() as u32;
                let latency_us = start.elapsed().as_micros() as u64;
                tracing::Span::current()
                    .record("latency_us", latency_us)
                    .record("dim", dim);
                self.embed_ok.fetch_add(1, Ordering::Relaxed);
                self.latency_sum_us.fetch_add(latency_us, Ordering::Relaxed);
                update_min(&self.latency_min_us, latency_us);
                update_max(&self.latency_max_us, latency_us);
                info!("embed ok");
                Ok(Response::new(EmbedResponse {
                    vector: v,
                    dim,
                    latency_us: latency_us as i64,
                }))
            }
            Err(e) => {
                self.embed_err.fetch_add(1, Ordering::Relaxed);
                warn!(error = %e, "embed failed");
                Err(Status::internal(format!("embed: {}", e)))
            }
        }
    }

    #[instrument(skip_all)]
    async fn health(
        &self,
        _request: Request<HealthRequest>,
    ) -> Result<Response<HealthResponse>, Status> {
        self.health_count.fetch_add(1, Ordering::Relaxed);
        Ok(Response::new(HealthResponse {
            version: self.version.clone(),
            device_id: self.device_id.clone(),
            model_fingerprint: self.fingerprint.clone(),
            // Iter 130: ready iff a real model graph is loaded. The
            // dimension pre-declaration (384) is no longer enough —
            // it lied while the placeholder embed path was active.
            // Now `has_model()` returns false until HEF support lands,
            // so coordinators correctly see model-less workers as
            // not-ready and skip them in validate_fleet / dispatch.
            ready: self.embedder.dimensions() > 0 && self.embedder.has_model(),
            // Iter-96 (ADR-174 §93): live NPU temperature read on every
            // health probe. 0.0 if read fails (older firmware variants
            // don't expose the opcode); coordinator side maps 0.0 → None.
            npu_temp_ts0_celsius: self
                .embedder
                .chip_temperature()
                .map(|(t, _)| t)
                .unwrap_or(0.0),
            npu_temp_ts1_celsius: self
                .embedder
                .chip_temperature()
                .map(|(_, t)| t)
                .unwrap_or(0.0),
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
        // Iter 200 — extract peer identity before consuming the
        // Request via into_inner(); the rate-limit debit below needs
        // it. `peer_identity<T>` is generic over the body type, so
        // we call it on the typed request directly — extensions
        // (peer addr, mTLS identity) are populated by tonic transport
        // and survive into the handler.
        let peer = ruvector_hailo_cluster::rate_limit::peer_identity(&request);
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
        // Iter 199 — refuse oversized batches before we spawn the
        // serial embed task. Without this an attacker fitting tightly
        // packed 1-byte texts inside the iter-180 64 KB request cap
        // could enqueue ~16 k embeds, each ~14 ms NPU time, holding
        // the connection for ~228 s (well past the iter-182 30 s RPC
        // timeout — but that just kicks the connection, doesn't
        // unblock the in-flight FFI work). Capping the length surfaces
        // the gate as InvalidArgument instantly.
        if n > self.max_batch_size {
            warn!(
                batch_size = n,
                max_batch_size = self.max_batch_size,
                "embed_stream batch too large — rejecting"
            );
            return Err(Status::invalid_argument(format!(
                "batch size {} exceeds max {} (ADR-172 §3a iter 199; \
                 tune via RUVECTOR_MAX_BATCH_SIZE)",
                n, self.max_batch_size
            )));
        }
        // Iter 200 — debit the per-peer rate limiter by the batch
        // length (minus 1 already debited by the interceptor) so a
        // peer at 1 RPS can't extract `max_batch_size` embeds/sec via
        // a single streaming RPC. n=1 is a no-op (the interceptor
        // already counted it). When the rate limiter is disabled
        // (None), this branch is also a no-op.
        if let Some(limiter) = self.rate_limiter.as_ref() {
            if n > 1 {
                let extra = (n - 1) as u32;
                if limiter.check_n(&peer, extra).is_err() {
                    self.rate_limit_denials.fetch_add(1, Ordering::Relaxed);
                    warn!(
                        peer = %peer,
                        batch_size = n,
                        "embed_stream batch denied by rate limiter (ADR-172 §3b iter 200)"
                    );
                    return Err(Status::resource_exhausted(format!(
                        "rate limit exceeded for {} on batch of {} (ADR-172 §3b iter 200)",
                        peer, n
                    )));
                }
            }
        }
        info!("worker embed_stream");

        let embedder = Arc::clone(&self.embedder);
        let dim = embedder.dimensions() as u32;
        let (tx, rx) = tokio::sync::mpsc::channel::<Result<EmbedStreamResponse, Status>>(n.max(1));

        // Spawn the embed work — order-preserving sequential issue (the
        // current HailoEmbedder is single-threaded; later iterations can
        // pipeline once the NPU is hooked up). `index` field guards the
        // contract that consumers reorder if needed.
        tokio::task::spawn(async move {
            for (i, text) in req.texts.into_iter().enumerate() {
                let start = Instant::now();
                let item = match embedder.embed(&text) {
                    Ok(v) => Ok(EmbedStreamResponse {
                        index: i as u32,
                        vector: v,
                        dim,
                        latency_us: start.elapsed().as_micros() as i64,
                    }),
                    Err(e) => Err(Status::internal(format!("embed[{}]: {}", i, e))),
                };
                if tx.send(item).await.is_err() {
                    // Client cancelled mid-stream — bail out.
                    break;
                }
            }
        });

        let stream = ReceiverStream::new(rx);
        Ok(Response::new(Box::pin(stream)))
    }

    #[instrument(skip_all)]
    async fn get_stats(
        &self,
        _request: Request<StatsRequest>,
    ) -> Result<Response<StatsResponse>, Status> {
        let min = self.latency_min_us.load(Ordering::Relaxed);
        // Iter-105: when the rate limiter is disabled (None), tracked
        // peers is reported as 0 — same as "no peers ever observed",
        // which is what the operator wants to see.
        let tracked_peers = self
            .rate_limiter
            .as_ref()
            .as_ref()
            .map(|l| l.tracked_peers() as u64)
            .unwrap_or(0);
        Ok(Response::new(StatsResponse {
            embed_count: self.embed_ok.load(Ordering::Relaxed),
            error_count: self.embed_err.load(Ordering::Relaxed),
            health_count: self.health_count.load(Ordering::Relaxed),
            latency_us_sum: self.latency_sum_us.load(Ordering::Relaxed),
            latency_us_min: if min == u64::MAX { 0 } else { min },
            latency_us_max: self.latency_max_us.load(Ordering::Relaxed),
            uptime_seconds: self.start.elapsed().as_secs(),
            rate_limit_denials: self.rate_limit_denials.load(Ordering::Relaxed),
            rate_limit_tracked_peers: tracked_peers,
        }))
    }
}

fn update_min(slot: &AtomicU64, v: u64) {
    let mut cur = slot.load(Ordering::Relaxed);
    while v < cur {
        match slot.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}
fn update_max(slot: &AtomicU64, v: u64) {
    let mut cur = slot.load(Ordering::Relaxed);
    while v > cur {
        match slot.compare_exchange_weak(cur, v, Ordering::Relaxed, Ordering::Relaxed) {
            Ok(_) => break,
            Err(actual) => cur = actual,
        }
    }
}

fn main() -> Result<(), Box<dyn std::error::Error>> {
    // Tracing init — env-driven (`RUST_LOG=info` etc.). Writes to stderr
    // so stdout stays clean for any future structured-output piping.
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .with_writer(std::io::stderr)
        .init();

    let bind: std::net::SocketAddr = std::env::var("RUVECTOR_WORKER_BIND")
        .unwrap_or_else(|_| "0.0.0.0:50051".into())
        .parse()?;
    let model_dir: PathBuf = std::env::var("RUVECTOR_MODEL_DIR")
        .unwrap_or_else(|_| "./models/all-minilm-l6-v2".into())
        .into();

    info!(bind = %bind, model_dir = %model_dir.display(), "ruvector-hailo-worker starting");

    // Compute the model fingerprint *before* opening the device, so
    // even if HailoEmbedder::open fails we've recorded the artifact
    // identity (or its absence — empty string).
    let fingerprint = compute_fingerprint(&model_dir);
    if fingerprint.is_empty() {
        warn!(
            "model_dir {} has no recognizable model artifacts \
             (NPU: model.hef + vocab.txt; cpu-fallback: model.safetensors + \
             tokenizer.json + config.json) — fingerprint empty; coordinators \
             will skip the integrity check",
            model_dir.display()
        );
    } else {
        info!(fingerprint = %fingerprint, "model fingerprint computed");
    }

    // Open the NPU + load the model. This is the only place that can
    // surface FeatureDisabled / NotYetImplemented today; the binary exits
    // cleanly with the underlying error message so operators see what's
    // missing.
    let embedder = HailoEmbedder::open(&model_dir).map_err(|e| {
        error!(error = %e, model_dir = %model_dir.display(), "HailoEmbedder::open failed");
        format!(
            "failed to open HailoEmbedder at {}: {} \
             (build with --features cpu-fallback for the host-CPU path \
             [model.safetensors + tokenizer.json + config.json — fetch via \
             deploy/download-cpu-fallback-model.sh], or --features hailo \
             for the NPU path on a Pi 5 + AI HAT+ [model.hef + vocab.txt])",
            model_dir.display(),
            e
        )
    })?;
    let device_id = embedder.device_id().to_string();

    // Iter 145/167: startup self-test. When a model is loaded
    // (cpu-fallback or HEF), embed three reference phrases and
    // check the semantic-ordering invariant
    // sim(dog, puppy) > sim(dog, kafka). Catches stale model files,
    // corrupt safetensors, op-set mismatches, AND silent quantization
    // drift (which would degrade ranking quality without breaking
    // dimensions). No-op when no model is loaded.
    if embedder.has_model() {
        // Three references: two semantically close (animal/movement),
        // one far (distributed-systems jargon). Any encoder that
        // produces useful vectors should rank close > far.
        let probes = [
            "the quick brown fox jumps over the lazy dog",
            "a puppy sprints across the meadow",
            "kafka topic partition rebalancing strategy",
        ];
        let mut vecs = Vec::with_capacity(probes.len());
        for p in probes {
            match embedder.embed(p) {
                Ok(v) => vecs.push(v),
                Err(e) => {
                    error!(error = %e, "startup self-test embed FAILED — refusing to serve");
                    return Err(format!(
                        "startup self-test embed failed: {} \
                         (model dir loaded but inference path is broken; \
                         fix the model artifacts and restart)",
                        e
                    )
                    .into());
                }
            }
        }
        // L2-normalised → cosine = dot. Already-normalised by every
        // embed path we ship, but the assertion holds either way as a
        // ranking comparison.
        fn cos(a: &[f32], b: &[f32]) -> f32 {
            a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
        }
        let sim_close = cos(&vecs[0], &vecs[1]);
        let sim_far = cos(&vecs[0], &vecs[2]);
        let dim = vecs[0].len();
        let head: Vec<String> = vecs[0]
            .iter()
            .take(4)
            .map(|x| format!("{:.4}", x))
            .collect();
        info!(
            dim,
            vec_head = %head.join(","),
            sim_close = sim_close,
            sim_far = sim_far,
            "startup self-test embed ok"
        );
        if sim_close <= sim_far {
            error!(
                sim_close,
                sim_far,
                "startup self-test ranking invariant FAILED — sim(dog,puppy) <= sim(dog,kafka), \
                 model is producing nonsense — refusing to serve"
            );
            return Err(format!(
                "startup self-test ranking failed: sim(dog,puppy)={:.4} <= sim(dog,kafka)={:.4} \
                 (encoder output is incoherent; check model artifacts, calibration, \
                 or HEF compile parameters)",
                sim_close, sim_far
            )
            .into());
        }
    } else {
        warn!(
            "no model loaded (has_model=false) — worker will serve health \
             but embed RPCs return NoModelLoaded until a model is wired in"
        );
    }

    // Iter 95 (ADR-174 §93): log the NPU on-die temperature once at
    // startup so operators see baseline thermal state without polling.
    // Hailo-8 has two thermal sensors per die; we log both. None means
    // the read failed (firmware unsupported on this board variant); the
    // `ruvector-hailo-stats` integration in iter 96 surfaces it
    // continuously via the Health RPC.
    match embedder.chip_temperature() {
        Some((ts0, ts1)) => {
            info!(
                ts0_celsius = ts0,
                ts1_celsius = ts1,
                "Hailo-8 NPU on-die temperature at startup"
            );
        }
        None => {
            // Soft warn — older Hailo firmware doesn't expose the
            // temperature opcode; not a startup-blocking issue.
            tracing::warn!(
                "Hailo-8 NPU temperature read returned None (firmware may not support the opcode)"
            );
        }
    }

    // ADR-172 §3c iter-103: parse the text-content audit mode. Default
    // None means existing deploys see no behavior change; ops opt into
    // hash for cross-system correlation or full for debug environments.
    let log_text_content =
        LogTextContent::parse(&std::env::var("RUVECTOR_LOG_TEXT_CONTENT").unwrap_or_default())?;
    info!(mode = ?log_text_content, "embed text-content audit mode");

    // ADR-172 §3b iter-104: per-peer rate limiter. None = disabled (back-
    // compat default); Some(_) when RUVECTOR_RATE_LIMIT_RPS > 0. Wrapped
    // in Arc<Option<_>> so the interceptor closure captures cheaply and
    // the always-install path stays type-uniform.
    let rate_limiter = Arc::new(RateLimiter::from_env());
    if rate_limiter.is_some() {
        info!("per-peer rate limiter enabled (ADR-172 §3b iter 104)");
    }
    // Iter-105: shared denial counter. Cloned into the interceptor +
    // the WorkerService; both run on the same tokio runtime so
    // Arc<AtomicU64> is the cheapest correct sharing.
    let rate_limit_denials = Arc::new(AtomicU64::new(0));

    // Iter 199 — cap on EmbedBatchRequest.texts.len(). Read here so
    // the value is logged at startup alongside the iter-180/181/182/
    // 183/184/190 gates. Default 256 is well above iter-179's
    // observed peak (b=16); floor 1 keeps the streaming RPC viable
    // under any misconfig.
    let max_batch_size: usize = std::env::var("RUVECTOR_MAX_BATCH_SIZE")
        .ok()
        .and_then(|s| s.parse::<usize>().ok())
        .unwrap_or(256)
        .max(1);
    info!(
        max_batch_size,
        "embed_stream batch-size cap set (ADR-172 §3a iter 199 DoS gate)"
    );

    // Iter 185 — keep an outer Arc<HailoEmbedder> ref so the FFI
    // teardown happens on the main thread, after the tokio runtime
    // has drained, instead of inside the async runtime where it
    // races HailoRT's internal worker threads. iter-179 observed a
    // SIGSEGV during shutdown when the implicit drop landed on a
    // tokio worker thread mid-DMA: `vstream_release` was reaping the
    // same vstream object that an in-flight DMA callback was
    // touching. Holding this Arc keeps the embedder alive until we
    // explicitly drop it after `block_on` returns.
    let embedder_outer = Arc::new(embedder);
    let svc = WorkerService {
        embedder: Arc::clone(&embedder_outer),
        version: format!("ruvector-hailo-worker {}", env!("CARGO_PKG_VERSION")),
        device_id,
        fingerprint,
        log_text_content,
        rate_limiter: Arc::clone(&rate_limiter),
        rate_limit_denials: Arc::clone(&rate_limit_denials),
        max_batch_size,
        start: Instant::now(),
        embed_ok: AtomicU64::new(0),
        embed_err: AtomicU64::new(0),
        health_count: AtomicU64::new(0),
        latency_sum_us: AtomicU64::new(0),
        latency_min_us: AtomicU64::new(u64::MAX),
        latency_max_us: AtomicU64::new(0),
    };

    let rt = tokio::runtime::Builder::new_multi_thread()
        .worker_threads(2)
        .enable_all()
        .build()?;

    rt.block_on(async move {
        // Iter 181 — cap HTTP/2 SETTINGS_MAX_CONCURRENT_STREAMS so a
        // single malicious connection can't spin up unbounded streams.
        // tonic's default is no cap. The Pi NPU saturates at ~70 req/s
        // (iter-179 measurements) so 256 in-flight streams is wasteful
        // but harmless for legit callers; the value sits ~32× above
        // observed legit peaks (bench c=8) and well below what a slow-
        // loris attacker would want. Operators can tune via
        // `RUVECTOR_MAX_CONCURRENT_STREAMS`; floor 8 to keep the
        // health-check + bench path viable on misconfig.
        let max_streams: u32 = std::env::var("RUVECTOR_MAX_CONCURRENT_STREAMS")
            .ok()
            .and_then(|s| s.parse::<u32>().ok())
            .unwrap_or(256)
            .max(8);
        info!(
            max_concurrent_streams = max_streams,
            "HTTP/2 max_concurrent_streams set (ADR-172 §3a iter 181 DoS gate)"
        );
        // Iter 182 — per-RPC handler timeout. tonic's default is no
        // bound, so a slow-loris client could hold a stream open
        // indefinitely with a single byte trickle. The HailoRT FFI
        // section of the embed handler is fully synchronous (no .await
        // mid-FFI), so timeout cancellation can only land before the
        // Mutex acquire or after the response build — neither leaks
        // NPU resources. Iter-179 measured p99=153 ms unary and
        // p99=910 ms at b=16; 30 s gives 30× headroom over the worst
        // observed legit RPC. Operators can tune via
        // `RUVECTOR_REQUEST_TIMEOUT_SECS`; floor 2 s so a misconfig
        // can't kill normal embeds.
        let request_timeout_secs: u64 = std::env::var("RUVECTOR_REQUEST_TIMEOUT_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(30)
            .max(2);
        info!(
            request_timeout_secs,
            "per-RPC handler timeout set (ADR-172 §3a iter 182 slow-loris gate)"
        );
        // Iter 183 — explicit CVE-2023-44487 rapid-reset cap. hyper/h2
        // already mitigates by defaulting to 20 pending RST_STREAM
        // frames, but pinning the value gives operators a tunable
        // surface and makes the mitigation reviewable from the worker
        // logs. 32 is a small step above the h2 default to leave room
        // for legit reset jitter (e.g., a client cancelling a stream
        // mid-flight) without weakening the cap meaningfully — a
        // GOAWAY still fires after just 33 unprocessed resets.
        let max_pending_resets: usize = std::env::var("RUVECTOR_MAX_PENDING_RESETS")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(32)
            .max(8);
        info!(
            max_pending_resets,
            "HTTP/2 max_pending_accept_reset_streams set (ADR-172 §3a iter 183 CVE-2023-44487 gate)"
        );
        // Iter 184 — HTTP/2 keepalive ping. tonic's default is no
        // keepalive, so a half-closed TCP connection (client crashed,
        // NAT mid-flow drop, network partition) sits in the worker's
        // accept table indefinitely, holding stream state. With a
        // 60 s ping interval the worker probes idle peers; if no PONG
        // arrives within the (hyper-default) 20 s timeout, the
        // connection is closed and its state reclaimed. Operators can
        // tune via `RUVECTOR_HTTP2_KEEPALIVE_SECS`; 0 disables the
        // feature for environments where ping traffic is undesirable
        // (e.g. cellular metering). Floor 10 s so a misconfig can't
        // saturate the link with pings.
        let keepalive_secs: u64 = std::env::var("RUVECTOR_HTTP2_KEEPALIVE_SECS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(60);
        let keepalive = if keepalive_secs == 0 {
            info!("HTTP/2 keepalive disabled (RUVECTOR_HTTP2_KEEPALIVE_SECS=0)");
            None
        } else {
            let v = keepalive_secs.max(10);
            info!(
                http2_keepalive_secs = v,
                "HTTP/2 keepalive enabled (ADR-172 §3a iter 184 dead-peer reclaim)"
            );
            Some(Duration::from_secs(v))
        };
        let mut server = Server::builder()
            .max_concurrent_streams(Some(max_streams))
            .timeout(Duration::from_secs(request_timeout_secs))
            .http2_max_pending_accept_reset_streams(Some(max_pending_resets))
            .http2_keepalive_interval(keepalive);
        #[cfg(feature = "tls")]
        {
            // Both vars must be set to opt-in. A partial config (cert
            // without key, or vice versa) is a misconfiguration — fail
            // loudly rather than silently dropping to plaintext.
            let cert = std::env::var("RUVECTOR_TLS_CERT").ok();
            let key = std::env::var("RUVECTOR_TLS_KEY").ok();
            match (cert, key) {
                (Some(c), Some(k)) => {
                    let mut tls = ruvector_hailo_cluster::tls::TlsServer::from_pem_files(&c, &k)
                        .map_err(|e| format!("tls server config: {}", e))?;
                    if let Ok(ca) = std::env::var("RUVECTOR_TLS_CLIENT_CA") {
                        tls = tls
                            .with_client_ca(&ca)
                            .map_err(|e| format!("tls client_ca: {}", e))?;
                        info!(client_ca = %ca, "mTLS client cert verification enabled");
                    }
                    server = server
                        .tls_config(tls.into_inner())
                        .map_err(|e| format!("apply tls: {}", e))?;
                    info!(cert = %c, "TLS enabled (ADR-172 §1a HIGH mitigation, iter 99)");
                }
                (Some(_), None) | (None, Some(_)) => {
                    return Err(
                        "RUVECTOR_TLS_CERT and RUVECTOR_TLS_KEY must both be set or both unset"
                            .to_string(),
                    );
                }
                (None, None) => {
                    info!("TLS disabled — plaintext gRPC (rely on Tailscale or upstream)");
                }
            }
        }
        info!(addr = %bind, "ruvector-hailo-worker serving");
        // Always install the interceptor; it's a no-op when the limiter
        // is None. Avoids type-divergence between enabled/disabled arms.
        let rl = Arc::clone(&rate_limiter);
        let denials = Arc::clone(&rate_limit_denials);
        // `Status` weighs ~176 bytes, which trips clippy's
        // result_large_err on the closure return type. We can't
        // change tonic's signature, so allow it locally.
        #[allow(clippy::result_large_err)]
        let interceptor = move |req: Request<()>| -> Result<Request<()>, Status> {
            if let Some(limiter) = rl.as_ref() {
                let peer = peer_identity(&req);
                if limiter.check(&peer).is_err() {
                    // Iter-105: bump the shared counter so get_stats
                    // surfaces denial pressure without operators having
                    // to grep the worker's stderr.
                    denials.fetch_add(1, Ordering::Relaxed);
                    return Err(Status::resource_exhausted(format!(
                        "rate limit exceeded for {} (ADR-172 §3b)",
                        peer
                    )));
                }
            }
            Ok(req)
        };
        // Iter 180 — cap the per-RPC decode budget. tonic's transport
        // default lets each RPC allocate ~4 MB before the server even
        // sees the request, which is gratuitous for an embed worker
        // (typical sentence-transformer input is <10 KB; 64 KB is 6×
        // safety margin). Operators on weird workloads can override via
        // RUVECTOR_MAX_REQUEST_BYTES; we clamp the floor at 4 KB so a
        // misconfig can't lock the worker out of accepting any RPC.
        let max_req_bytes: usize = std::env::var("RUVECTOR_MAX_REQUEST_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(64 * 1024)
            .max(4 * 1024);
        info!(
            max_request_bytes = max_req_bytes,
            "gRPC max_decoding_message_size set (ADR-172 §3a iter 180 DoS gate)"
        );
        // Iter 190 — defense-in-depth response size cap. The worker
        // controls its own response shape (Vec<f32>[384] ≈ 1.6 KB +
        // tonic framing for unary embed; streaming embed pumps the
        // same per-item shape), so the encode-side cap shouldn't fire
        // under any normal code path. Setting it explicitly bounds
        // the blast radius of a hypothetical bug that ever returned a
        // huge response (e.g. an accidentally-leaked debug payload).
        // 16 KB is 10× the legitimate per-message size and well
        // outside any plausible legit response.
        let max_resp_bytes: usize = std::env::var("RUVECTOR_MAX_RESPONSE_BYTES")
            .ok()
            .and_then(|s| s.parse::<usize>().ok())
            .unwrap_or(16 * 1024)
            .max(4 * 1024);
        info!(
            max_response_bytes = max_resp_bytes,
            "gRPC max_encoding_message_size set (ADR-172 §3a iter 190 belt-and-suspenders)"
        );
        // Note: `max_decoding_message_size` lives on the generated
        // `EmbeddingServer`, not tonic's `InterceptedService` wrapper —
        // apply it before wrapping. The `with_interceptor` static
        // helper would re-build the inner with default limits, so we
        // skip it and call `InterceptedService::new` ourselves.
        let embed_server = EmbeddingServer::new(svc)
            .max_decoding_message_size(max_req_bytes)
            .max_encoding_message_size(max_resp_bytes);
        let intercepted =
            tonic::service::interceptor::InterceptedService::new(embed_server, interceptor);
        server
            .add_service(intercepted)
            .serve_with_shutdown(bind, shutdown_signal())
            .await
            .map_err(|e| format!("serve: {}", e))?;
        Ok::<(), String>(())
    })?;

    // Iter 185 — exit-without-FFI-teardown to eliminate the
    // shutdown SIGSEGV. Empirically (5/5 across iter-184 + iter-185
    // first-attempt with drain+drop), HailoRT's internal threads
    // (DMA scheduler, vdevice callbacks) crash on every clean shutdown
    // regardless of when we time the vstream release relative to
    // tonic's serve completion. The crash is not in our HefPipeline
    // Drop — the explicit `drop(embedder_outer)` was never reached
    // when we tried — it's deeper, in HailoRT C-library teardown,
    // which the iter-179 backtrace (status=11/SEGV from systemd) had
    // already pointed at.
    //
    // The chosen mitigation is to leak the embedder via `mem::forget`
    // and call `process::exit(0)`. The OS reaps every resource we own
    // (mmap'd HEF, vstream fds, driver-side handles via close(2));
    // HailoRT's own threads are killed by the same exit syscall, so
    // they can't race a free that no longer happens. Leaking is bounded
    // (one HefPipeline + one HostEmbeddings pair per process lifetime;
    // the next worker is a fresh process). Operators see a clean
    // `status=0/SUCCESS` instead of `status=11/SEGV`, which makes
    // restart loops and monitoring sane again.
    //
    // Operators who want to attempt the clean-drop path (e.g. a future
    // HailoRT release that fixes the bug) can flip
    // `RUVECTOR_SHUTDOWN_FORCE_CLEAN=1` to take the slow path instead.
    info!("server stopped — exiting (iter 185 SEGV-on-shutdown mitigation)");
    if std::env::var("RUVECTOR_SHUTDOWN_FORCE_CLEAN").as_deref() == Ok("1") {
        let drain_ms: u64 = std::env::var("RUVECTOR_SHUTDOWN_DRAIN_MS")
            .ok()
            .and_then(|s| s.parse::<u64>().ok())
            .unwrap_or(500);
        info!(drain_ms, "FORCE_CLEAN=1 — taking slow drop path");
        rt.shutdown_timeout(Duration::from_secs(2));
        std::thread::sleep(Duration::from_millis(drain_ms));
        drop(embedder_outer);
        info!("HailoRT released — exiting cleanly");
        return Ok(());
    }
    std::mem::forget(embedder_outer);
    std::mem::forget(rt);
    std::process::exit(0);
}

/// Future that resolves when SIGINT or SIGTERM arrives — graceful exit.
async fn shutdown_signal() {
    use tokio::signal::unix::{signal, SignalKind};
    let mut sigterm = signal(SignalKind::terminate()).expect("install SIGTERM handler");
    let mut sigint = signal(SignalKind::interrupt()).expect("install SIGINT handler");
    tokio::select! {
        _ = sigterm.recv() => info!("SIGTERM received, shutting down"),
        _ = sigint.recv()  => info!("SIGINT received, shutting down"),
    }
}

#[cfg(test)]
mod tests {
    //! ADR-172 §3c iter-103 unit tests for the audit-log text-content
    //! mode parser + renderer. The bin is its own crate (not the lib),
    //! so these tests run via `cargo test --bin ruvector-hailo-worker`.
    use super::LogTextContent;

    #[test]
    fn parse_default_empty_is_none() {
        assert!(matches!(
            LogTextContent::parse("").unwrap(),
            LogTextContent::None
        ));
    }

    #[test]
    fn parse_named_modes() {
        assert!(matches!(
            LogTextContent::parse("none").unwrap(),
            LogTextContent::None
        ));
        assert!(matches!(
            LogTextContent::parse("hash").unwrap(),
            LogTextContent::Hash
        ));
        assert!(matches!(
            LogTextContent::parse("full").unwrap(),
            LogTextContent::Full
        ));
    }

    #[test]
    fn parse_unknown_mode_errors() {
        let e = LogTextContent::parse("partial").unwrap_err();
        assert!(
            e.contains("none|hash|full"),
            "expected mode list in err: {}",
            e
        );
    }

    #[test]
    fn render_none_returns_dash() {
        assert_eq!(LogTextContent::None.render("any text"), "-");
    }

    #[test]
    fn render_hash_is_16_hex_chars_deterministic() {
        let h = LogTextContent::Hash.render("hello world");
        assert_eq!(h.len(), 16, "expected 16-hex prefix, got {:?}", h);
        assert!(
            h.chars().all(|c| c.is_ascii_hexdigit()),
            "non-hex char in {:?}",
            h
        );
        assert_eq!(LogTextContent::Hash.render("hello world"), h);
        assert_ne!(LogTextContent::Hash.render("hello world!"), h);
    }

    #[test]
    fn render_full_passes_through() {
        assert_eq!(LogTextContent::Full.render("payload"), "payload");
    }
}
