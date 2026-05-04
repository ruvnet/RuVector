//! Transport abstraction — what the coordinator uses to call a worker.
//!
//! Step 14 (gRPC via tonic) implements `EmbeddingTransport` against a
//! generated client. Today we ship a `NullTransport` that lets the rest
//! of the crate compile + be unit-tested.

use crate::error::ClusterError;
use std::sync::Arc;

/// Identity for one worker in the fleet. `name` is for logs/metrics;
/// `address` is the wire-level endpoint (e.g. `100.77.59.83:50051`).
#[derive(Clone, Debug)]
pub struct WorkerEndpoint {
    /// Display name for logs / stats / fleet manifests.
    pub name: String,
    /// Wire-level endpoint, typically `host:port` for gRPC.
    pub address: String,
}

impl WorkerEndpoint {
    /// Construct an endpoint from a (name, address) pair.
    pub fn new(name: impl Into<String>, address: impl Into<String>) -> Self {
        Self {
            name: name.into(),
            address: address.into(),
        }
    }
}

/// What the coordinator actually does to a worker. Sync return values for
/// now; `embed_one_blocking` calls into this. Step 14 introduces an async
/// tonic-backed impl + an `embed_one` async sibling.
pub trait EmbeddingTransport {
    /// Synchronous embed call against a single worker. Returns the
    /// L2-normalised f32 vector and the server-reported latency in µs.
    fn embed(
        &self,
        worker: &WorkerEndpoint,
        text: &str,
        max_seq: u32,
    ) -> Result<(Vec<f32>, u64), ClusterError>;

    /// Same as `embed` but with a caller-supplied `request_id` for log
    /// correlation. Default impl calls `embed` (losing the supplied id —
    /// transports that pre-date this method still compile). Real
    /// transports (`GrpcTransport`) override to wire `request_id` into
    /// the gRPC metadata header.
    fn embed_with_request_id(
        &self,
        worker: &WorkerEndpoint,
        text: &str,
        max_seq: u32,
        _request_id: &str,
    ) -> Result<(Vec<f32>, u64), ClusterError> {
        self.embed(worker, text, max_seq)
    }

    /// Cheap health probe used by the EWMA tracker.
    fn health(&self, worker: &WorkerEndpoint) -> Result<HealthReport, ClusterError>;

    /// Per-worker counters + latency stats. Default impl errors with
    /// `Transport{reason: "stats not supported"}` so transports that
    /// pre-date the GetStats RPC keep compiling without an override.
    fn stats(&self, worker: &WorkerEndpoint) -> Result<StatsSnapshot, ClusterError> {
        Err(ClusterError::Transport {
            worker: worker.name.clone(),
            reason: "stats not supported by this transport".into(),
        })
    }

    /// Batched server-streaming embed. Returns one (index, vector,
    /// latency_us) tuple per input slot. Order is *not* guaranteed
    /// across workers that may pipeline; use `index` to put results
    /// back in input order.
    ///
    /// Default impl errors with `Transport{reason: "stream not supported"}`.
    fn embed_stream(
        &self,
        worker: &WorkerEndpoint,
        _texts: &[String],
        _max_seq: u32,
    ) -> Result<Vec<EmbedStreamItem>, ClusterError> {
        Err(ClusterError::Transport {
            worker: worker.name.clone(),
            reason: "stream not supported by this transport".into(),
        })
    }

    /// Batched embed with caller-supplied `request_id`. Default impl
    /// delegates to `embed_stream`, losing the supplied id. Real
    /// transports override to wire `request_id` into the gRPC metadata
    /// header. Mirror of `embed_with_request_id`.
    fn embed_stream_with_request_id(
        &self,
        worker: &WorkerEndpoint,
        texts: &[String],
        max_seq: u32,
        _request_id: &str,
    ) -> Result<Vec<EmbedStreamItem>, ClusterError> {
        self.embed_stream(worker, texts, max_seq)
    }
}

/// One reply slot from a streaming embed batch.
#[derive(Clone, Debug)]
pub struct EmbedStreamItem {
    /// Position in the input batch (0..N-1) — used to restore order.
    pub index: u32,
    /// L2-normalised f32 embedding from the worker.
    pub vector: Vec<f32>,
    /// Server-reported per-item latency in microseconds.
    pub latency_us: u64,
}

/// Server-reported counters. Mirrors `proto::StatsResponse` but with
/// safer types (Duration / Option-wrapped extrema).
#[derive(Clone, Debug, serde::Serialize)]
pub struct StatsSnapshot {
    /// Successful `embed` RPCs since worker boot.
    pub embed_count: u64,
    /// Failed `embed` RPCs since worker boot.
    pub error_count: u64,
    /// Health probes received since worker boot.
    pub health_count: u64,
    /// Serialised as microseconds for tool-friendliness (`stats --json`
    /// pipes cleanly into jq, prometheus exporters, etc.).
    #[serde(serialize_with = "serialize_duration_us")]
    pub latency_sum: std::time::Duration,
    /// Smallest latency observed since boot. `None` if no embeds yet.
    #[serde(serialize_with = "serialize_opt_duration_us")]
    pub latency_min: Option<std::time::Duration>,
    /// Largest latency observed since boot. `None` if no embeds yet.
    #[serde(serialize_with = "serialize_opt_duration_us")]
    pub latency_max: Option<std::time::Duration>,
    /// Serialised as seconds (uptime is naturally coarse).
    #[serde(serialize_with = "serialize_duration_secs")]
    pub uptime: std::time::Duration,
    /// Iter-105 (ADR-172 §3b follow-up): cumulative count of
    /// `Status::resource_exhausted` returned by the worker's rate-limit
    /// interceptor. 0 = limiter disabled or no denials yet.
    #[serde(default)]
    pub rate_limit_denials: u64,
    /// Distinct peers (mTLS cert hashes or IPs) the limiter has seen
    /// since boot. 0 = limiter disabled.
    #[serde(default)]
    pub rate_limit_tracked_peers: u64,
}

fn serialize_duration_us<S: serde::Serializer>(
    d: &std::time::Duration,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.as_micros() as u64)
}
fn serialize_opt_duration_us<S: serde::Serializer>(
    d: &Option<std::time::Duration>,
    s: S,
) -> Result<S::Ok, S::Error> {
    match d {
        Some(d) => s.serialize_u64(d.as_micros() as u64),
        None => s.serialize_none(),
    }
}
fn serialize_duration_secs<S: serde::Serializer>(
    d: &std::time::Duration,
    s: S,
) -> Result<S::Ok, S::Error> {
    s.serialize_u64(d.as_secs())
}

impl StatsSnapshot {
    /// Mean per-embed latency, or `None` if no embeds have completed yet.
    pub fn average_latency(&self) -> Option<std::time::Duration> {
        if self.embed_count == 0 {
            None
        } else {
            Some(self.latency_sum / self.embed_count as u32)
        }
    }
    /// Lifetime throughput: successful embeds divided by uptime seconds.
    pub fn embeds_per_second(&self) -> f64 {
        let secs = self.uptime.as_secs_f64();
        if secs <= 0.0 {
            0.0
        } else {
            (self.embed_count as f64) / secs
        }
    }
}

/// Health-probe response from a worker.
#[derive(Clone, Debug)]
pub struct HealthReport {
    /// Server version string (e.g. `"ruvector-hailo-worker 0.1.0"`).
    pub version: String,
    /// Device identifier — typically PCIe BDF for Hailo NPU workers.
    pub device_id: String,
    /// sha256 fingerprint of the loaded model artifacts; empty if unset.
    pub model_fingerprint: String,
    /// `true` when the worker has loaded a model and is accepting embeds.
    pub ready: bool,
    /// Iter-96 (ADR-174 §93 follow-up): on-die NPU temperature in Celsius
    /// from sensor 0 (front-of-die). `None` if the worker couldn't read,
    /// the firmware doesn't support the opcode, or the worker is older
    /// than iter 96. Coordinator treats `None` as "skip the temp gauge".
    pub npu_temp_ts0_celsius: Option<f32>,
    /// Iter-96: on-die NPU temperature from sensor 1 (back-of-die).
    pub npu_temp_ts1_celsius: Option<f32>,
}

/// Trivial transport that always errors `NotYetImplemented`. Used by the
/// scaffold to keep the type system honest; replaced by the real gRPC
/// transport in step 14.
pub struct NullTransport;

impl EmbeddingTransport for NullTransport {
    fn embed(
        &self,
        worker: &WorkerEndpoint,
        _text: &str,
        _max_seq: u32,
    ) -> Result<(Vec<f32>, u64), ClusterError> {
        Err(ClusterError::Transport {
            worker: worker.name.clone(),
            reason: "NullTransport — gRPC not wired (step 14)".into(),
        })
    }
    fn health(&self, worker: &WorkerEndpoint) -> Result<HealthReport, ClusterError> {
        Err(ClusterError::Transport {
            worker: worker.name.clone(),
            reason: "NullTransport — gRPC not wired (step 14)".into(),
        })
    }
}

/// Convenience constructor.
pub fn null_transport() -> Arc<dyn EmbeddingTransport + Send + Sync> {
    Arc::new(NullTransport)
}
