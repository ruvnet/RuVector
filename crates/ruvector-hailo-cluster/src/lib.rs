//! Multi-Pi cluster coordinator for ruvector's Hailo embedding workers.
//!
//! ADR-167 §8 (`hailo-backend` branch). The crate distributes embed
//! requests across a fleet of Pi 5 + Hailo-8 workers, with built-in
//! P2C+EWMA load balancing, fingerprint enforcement, optional in-process
//! caching, and Tailscale-tag-based discovery.
//!
//! # Example
//!
//! Build a coordinator pointed at two static workers, validate the
//! fleet, and embed one text. The full async batch path and the
//! caller-supplied `request_id` variants live in [`HailoClusterEmbedder`].
//!
//! ```no_run
//! use std::sync::Arc;
//! use ruvector_hailo_cluster::{
//!     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
//!     transport::EmbeddingTransport,
//! };
//!
//! # fn main() -> Result<(), Box<dyn std::error::Error>> {
//! let workers = vec![
//!     WorkerEndpoint::new("pi-a", "100.77.59.83:50051"),
//!     WorkerEndpoint::new("pi-b", "100.77.59.84:50051"),
//! ];
//! let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
//!     Arc::new(GrpcTransport::new()?);
//!
//! let cluster = HailoClusterEmbedder::new(workers, transport, 384, "fp:abc")?
//!     .with_cache(4096);
//!
//! // Boot-time fingerprint enforcement; ejects mismatched workers.
//! let report = cluster.validate_fleet()?;
//! assert!(!report.healthy.is_empty());
//!
//! let vector = cluster.embed_one_blocking("hello world")?;
//! assert_eq!(vector.len(), 384);
//! # Ok(()) }
//! ```
//!
//! # Operator binaries
//!
//! Three CLI tools wrap this library:
//!
//! - `ruvector-hailo-embed` — stdin / `--text` → JSONL embeddings
//! - `ruvector-hailo-stats` — fleet observability (TSV / JSON / Prom)
//! - `ruvector-hailo-cluster-bench` — sustained-load harness
//!
//! All three share `--workers` / `--workers-file` / `--tailscale-tag`
//! discovery and `--auto-fingerprint` / `--validate-fleet` safety flags.

#![allow(dead_code)]
// Iter 75: locks in the doc audit. Future pub additions trigger a
// warning at build time so docs don't bit-rot back to the iter-73 baseline.
#![warn(missing_docs)]

pub mod cache;
pub mod discovery;
pub mod error;
pub mod fingerprint;
pub mod grpc_transport;
pub mod health;
pub mod manifest_sig;
pub mod pool;
pub mod proto;
pub mod rate_limit;
pub mod shard;
#[cfg(feature = "tls")]
pub mod tls;
pub mod transport;

pub use health::{HealthChecker, HealthCheckerConfig};

pub use fingerprint::compute_fingerprint;

pub use discovery::{Discovery, FileDiscovery, StaticDiscovery, TailscaleDiscovery};
pub use grpc_transport::GrpcTransport;

pub use error::ClusterError;
pub use pool::{P2cPool, WorkerStats};
pub use shard::{HashShardRouter, ShardRouter};
pub use transport::{EmbeddingTransport, WorkerEndpoint};

use std::sync::Arc;

/// Per-worker outcome of `HailoClusterEmbedder::validate_fleet()`.
///
/// Categorizes every worker into one of four buckets so an operator
/// can see the full state of the fleet at boot — what's healthy, what
/// the actual fingerprints are vs expected, what's down. Successful
/// boot requires `healthy` non-empty; everything else is observability
/// (and `fingerprint_mismatched` workers are ejected from the pool
/// before this returns).
#[derive(Default, Debug, Clone)]
pub struct FleetValidation {
    /// Workers that returned a successful, fingerprint-matching health probe.
    pub healthy: Vec<String>,
    /// Workers reachable but reporting `ready=false` (initializing, draining…).
    pub not_ready: Vec<String>,
    /// Workers ejected from the dispatch pool because their `model_fingerprint`
    /// didn't match the coordinator's `expected_model_fingerprint`.
    pub fingerprint_mismatched: Vec<FleetMismatch>,
    /// `(name, error_string)` so the operator can see why a worker was unreachable.
    pub unreachable: Vec<(String, String)>,
}

/// One worker's fingerprint mismatch detail surfaced by `validate_fleet`.
#[derive(Debug, Clone)]
pub struct FleetMismatch {
    /// Worker name from `WorkerEndpoint::name`.
    pub worker: String,
    /// Fingerprint string the coordinator was configured to require.
    pub expected: String,
    /// Fingerprint string the worker reported.
    pub actual: String,
}

/// One worker's combined health + stats result. Returned by
/// `HailoClusterEmbedder::fleet_state()` so an ops display can show
/// fingerprint alongside counters in a single table without two
/// passes.
pub struct FleetMemberState {
    /// The discovered worker (name + address).
    pub endpoint: transport::WorkerEndpoint,
    /// `None` if the health probe failed; the stats RPC may still
    /// have succeeded (and vice versa).
    pub fingerprint: Option<String>,
    /// On-die NPU temperature sensor 0 in Celsius. `None` if the worker
    /// didn't populate it (older firmware) or the health probe failed.
    /// Iter-96b deliverable from ADR-174 §93b.
    pub npu_temp_ts0_celsius: Option<f32>,
    /// On-die NPU temperature sensor 1 in Celsius.
    pub npu_temp_ts1_celsius: Option<f32>,
    /// Per-worker counters from the GetStats RPC, or the transport
    /// error encountered (kept as a `Result` so a single bad worker
    /// doesn't poison the rest of the fleet snapshot).
    pub stats: Result<transport::StatsSnapshot, ClusterError>,
}

/// Cluster coordinator that distributes embed requests across multiple
/// Pi 5 + Hailo-8 workers. Implements `EmbeddingProvider` (once iteration
/// 14 brings the path dep on `ruvector-core`) so callers can swap a
/// single-device `HailoEmbedder` for a fleet without code changes.
pub struct HailoClusterEmbedder {
    /// `Arc` so the background health-checker (when spawned via
    /// `spawn_health_checker`) shares the same pool the dispatcher
    /// uses — runtime fingerprint mismatch ejections actually remove
    /// the worker from the dispatch pool, not a sidecar copy.
    pool: Arc<P2cPool>,
    transport: Arc<dyn EmbeddingTransport + Send + Sync>,
    /// Embedding dimensionality (must match across all workers; coordinator
    /// rejects fleet members reporting a different dim during health check).
    dim: usize,
    /// Compiled-in compatibility marker — coordinator refuses to dispatch
    /// to workers reporting a different model fingerprint, preventing
    /// silent vector-space drift across a heterogeneous fleet.
    expected_model_fingerprint: String,
    /// Optional in-process LRU cache. Capacity 0 ≡ disabled; check is
    /// O(1) lock + lookup, so leaving a small cache on always is cheap.
    /// `Arc` so the background health-checker (spawned via
    /// `spawn_health_checker`) can hold its own reference and clear the
    /// cache on a fingerprint-mismatch event without unsafe lifetimes.
    cache: Arc<cache::EmbeddingCache>,
}

impl HailoClusterEmbedder {
    /// Build a new coordinator from a list of worker endpoints + a
    /// transport. The transport is what actually talks to the workers
    /// (gRPC / HTTP / in-memory mock) — the coordinator stays
    /// transport-agnostic.
    pub fn new(
        workers: Vec<WorkerEndpoint>,
        transport: Arc<dyn EmbeddingTransport + Send + Sync>,
        dim: usize,
        expected_model_fingerprint: impl Into<String>,
    ) -> Result<Self, ClusterError> {
        if workers.is_empty() {
            return Err(ClusterError::NoWorkers);
        }
        Ok(Self {
            pool: Arc::new(P2cPool::new(workers)),
            transport,
            dim,
            expected_model_fingerprint: expected_model_fingerprint.into(),
            cache: Arc::new(cache::EmbeddingCache::new(0)),
        })
    }

    /// Enable an in-process LRU cache of size `cap`. Cache key is
    /// `(expected_model_fingerprint, text)` so a model fingerprint
    /// change invalidates everything for free. `cap == 0` keeps the
    /// cache disabled.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "fp:abc",
    /// )?
    /// .with_cache(8192);
    /// assert_eq!(cluster.cache_stats().capacity, 8192);
    /// # Ok(()) }
    /// ```
    pub fn with_cache(mut self, cap: usize) -> Self {
        self.cache = Arc::new(cache::EmbeddingCache::new(cap));
        self
    }

    /// Like `with_cache(cap)`, but bounds entry lifetime by `ttl`.
    /// Hits older than `ttl` count as misses + evictions and force a
    /// fresh RPC. Useful for long-running coordinators where you want
    /// a hard staleness ceiling regardless of access pattern.
    pub fn with_cache_ttl(mut self, cap: usize, ttl: std::time::Duration) -> Self {
        self.cache = Arc::new(cache::EmbeddingCache::with_ttl(cap, Some(ttl)));
        self
    }

    /// Read-only cache stats (hits / misses / size / evictions).
    pub fn cache_stats(&self) -> cache::CacheStats {
        self.cache.stats()
    }

    /// Drop every cached entry. Useful during controlled model rollovers
    /// when the operator wants the next embed for every text to hit the
    /// (new) worker fresh, even if the fingerprint label matches by
    /// coincidence. Returns the number of entries dropped; hits/misses
    /// counters are preserved across the call.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "",
    /// )?
    /// .with_cache(1024);
    /// // Force-drop after a model rollover.
    /// let dropped = cluster.invalidate_cache();
    /// assert_eq!(dropped, 0); // empty cache, nothing to drop
    /// # Ok(()) }
    /// ```
    pub fn invalidate_cache(&self) -> usize {
        self.cache.clear()
    }

    /// Build a `HealthCheckerConfig` pre-wired with this cluster's
    /// expected fingerprint and cache-invalidation callback. Convenience
    /// for the common case: spawn the checker, get fingerprint
    /// enforcement + auto-cache-clear on mismatch, no manual closure
    /// plumbing required. Override fields on the returned config before
    /// passing to `HealthChecker::spawn` if defaults aren't right.
    pub fn health_checker_config(&self) -> health::HealthCheckerConfig {
        let cache = Arc::clone(&self.cache);
        health::HealthCheckerConfig {
            expected_fingerprint: if self.expected_model_fingerprint.is_empty() {
                None
            } else {
                Some(self.expected_model_fingerprint.clone())
            },
            on_fingerprint_mismatch: Some(Arc::new(move || {
                cache.clear();
            })),
            ..Default::default()
        }
    }

    /// Spawn a background `HealthChecker` on `rt_handle`, pre-wired
    /// with this cluster's pool, transport, and (via
    /// `health_checker_config`) the expected fingerprint + auto-cache-
    /// invalidate callback. The checker shares the dispatch pool —
    /// ejection actually removes workers from the dispatch path. Drop
    /// the returned `HealthChecker` to stop.
    pub fn spawn_health_checker(
        &self,
        rt_handle: &tokio::runtime::Handle,
        cfg: health::HealthCheckerConfig,
    ) -> health::HealthChecker {
        let workers = self.pool.all_endpoints();
        health::HealthChecker::spawn(
            rt_handle,
            Arc::clone(&self.pool),
            workers,
            Arc::clone(&self.transport),
            cfg,
        )
    }

    /// Probe the first reachable worker for its `model_fingerprint` and
    /// return it. Useful for the `--auto-fingerprint` discovery flow:
    /// build the cluster with an empty fingerprint (no enforcement),
    /// call `discover_fingerprint()` to see what the fleet is actually
    /// running, then rebuild the cluster with that fingerprint as the
    /// enforced expectation.
    ///
    /// Errors with `AllWorkersFailed` if every health probe fails. Stops
    /// at the first success — operator can re-run with `validate_fleet()`
    /// to confirm the rest of the fleet agrees.
    ///
    /// **Trust note (ADR-172 §2b):** this trusts the first reachable
    /// worker, so a single hostile or out-of-date worker can poison the
    /// discovered fingerprint. Production deploys with ≥2 workers should
    /// prefer [`Self::discover_fingerprint_with_quorum`] (the CLIs do
    /// this automatically when the fleet has ≥2 workers).
    pub fn discover_fingerprint(&self) -> Result<String, ClusterError> {
        let mut last_err: Option<ClusterError> = None;
        for endpoint in self.pool.all_endpoints() {
            match self.transport.health(&endpoint) {
                Ok(report) => return Ok(report.model_fingerprint),
                Err(e) => last_err = Some(e),
            }
        }
        Err(ClusterError::AllWorkersFailed(format!(
            "discover_fingerprint: every worker's health probe failed; last: {}",
            match last_err {
                Some(e) => e.to_string(),
                None => "no workers in pool".into(),
            }
        )))
    }

    /// Quorum-based fingerprint discovery (ADR-172 §2b mitigation, iter
    /// 102). Probes every worker in the pool, tallies the most-reported
    /// fingerprint, and returns it iff at least `min_agree` workers
    /// reported it. A single hostile worker therefore can't establish
    /// "the" fingerprint when `min_agree >= 2`.
    ///
    /// `min_agree == 1` is the legacy single-witness mode (equivalent
    /// to picking the most popular reported fp); use it for single-
    /// worker dev fleets where quorum is impossible.
    ///
    /// Empty fingerprints are excluded from the tally — they mean "the
    /// worker hasn't loaded a model yet" and aren't a meaningful vote.
    pub fn discover_fingerprint_with_quorum(
        &self,
        min_agree: usize,
    ) -> Result<String, ClusterError> {
        use std::collections::HashMap;
        let endpoints = self.pool.all_endpoints();
        let mut tally: HashMap<String, usize> = HashMap::new();
        let mut probed = 0usize;
        let mut errs: Vec<String> = Vec::new();
        for endpoint in &endpoints {
            match self.transport.health(endpoint) {
                Ok(report) => {
                    probed += 1;
                    if !report.model_fingerprint.is_empty() {
                        *tally.entry(report.model_fingerprint).or_insert(0) += 1;
                    }
                }
                Err(e) => errs.push(format!("{}: {}", endpoint.name, e)),
            }
        }
        if probed == 0 {
            return Err(ClusterError::AllWorkersFailed(format!(
                "discover_fingerprint_with_quorum: every worker's health probe failed: {}",
                errs.join("; ")
            )));
        }
        let (best_fp, best_count) = tally
            .iter()
            .max_by_key(|(_, n)| *n)
            .map(|(fp, n)| (fp.clone(), *n))
            .unwrap_or_else(|| ("".into(), 0));
        if best_count < min_agree.max(1) {
            return Err(ClusterError::AllWorkersFailed(format!(
                "discover_fingerprint_with_quorum: best fp had {} agreeing workers, \
                 need {}; tally={:?} (probed={}, unreachable={})",
                best_count,
                min_agree.max(1),
                tally,
                probed,
                errs.len()
            )));
        }
        Ok(best_fp)
    }

    /// Synchronous startup validation. Probes every worker via
    /// `transport.health()`, ejects any whose `model_fingerprint`
    /// doesn't match `expected_model_fingerprint`, and returns a
    /// structured per-worker outcome report. Empty
    /// `expected_model_fingerprint` skips the fingerprint check
    /// (legacy / opt-out mode).
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "fp:expected",
    /// )?;
    /// let report = cluster.validate_fleet()?;
    /// for member in &report.fingerprint_mismatched {
    ///     eprintln!("ejected {}: had {:?}", member.worker, member.actual);
    /// }
    /// # Ok(()) }
    /// ```
    ///
    /// Errors with `AllWorkersFailed` if zero workers come back as
    /// healthy + matching — fail-fast at boot beats discovering it
    /// mid-traffic with a noise-vector-search-results bug.
    pub fn validate_fleet(&self) -> Result<FleetValidation, ClusterError> {
        let mut out = FleetValidation::default();
        for endpoint in self.pool.all_endpoints() {
            match self.transport.health(&endpoint) {
                Ok(report) => {
                    let fp_check = self.expected_model_fingerprint.is_empty()
                        || report.model_fingerprint == self.expected_model_fingerprint;
                    if !fp_check {
                        // Hard disqualification — this worker is on a
                        // different model. Eject so dispatch never picks
                        // it. Report the actual fingerprint so the
                        // operator can see the drift.
                        self.pool.eject(&endpoint.name);
                        out.fingerprint_mismatched.push(FleetMismatch {
                            worker: endpoint.name.clone(),
                            expected: self.expected_model_fingerprint.clone(),
                            actual: report.model_fingerprint,
                        });
                    } else if !report.ready {
                        out.not_ready.push(endpoint.name.clone());
                    } else {
                        out.healthy.push(endpoint.name.clone());
                    }
                }
                Err(e) => {
                    out.unreachable.push((endpoint.name.clone(), e.to_string()));
                }
            }
        }
        if out.healthy.is_empty() {
            return Err(ClusterError::AllWorkersFailed(format!(
                "validate_fleet: 0 healthy workers ({} mismatched fp, {} not ready, {} unreachable)",
                out.fingerprint_mismatched.len(),
                out.not_ready.len(),
                out.unreachable.len(),
            )));
        }
        Ok(out)
    }

    /// Number of workers currently in the pool (live + ejected).
    pub fn worker_count(&self) -> usize {
        self.pool.size()
    }

    /// Embedding dimensionality the coordinator was configured for.
    pub fn dim(&self) -> usize {
        self.dim
    }

    /// Batched embed via the streaming RPC. Picks a single worker via
    /// P2C (the entire batch goes to that worker; coordinator-side
    /// re-fanning across workers would require the worker to be
    /// stateless per-item, which it is, but adds latency vs sticking
    /// the batch on one socket).
    ///
    /// Returns an ordered `Vec<Vec<f32>>` matching input order. Worker
    /// streams items back in arbitrary order; we sort by `index`
    /// before returning.
    pub fn embed_batch_blocking(&self, texts: &[String]) -> Result<Vec<Vec<f32>>, ClusterError> {
        // Convenience wrapper: random correlation ID. Use
        // `embed_batch_blocking_with_request_id` to pass through an
        // upstream tracing token.
        self.embed_batch_blocking_with_request_id(texts, "")
    }

    /// Batch embed with caller-supplied `request_id` (mirror of
    /// `embed_one_blocking_with_request_id`). Empty string ≡ generate
    /// random. The whole batch shares one id — the worker logs one
    /// span covering all items in the streaming RPC.
    pub fn embed_batch_blocking_with_request_id(
        &self,
        texts: &[String],
        request_id: &str,
    ) -> Result<Vec<Vec<f32>>, ClusterError> {
        if texts.is_empty() {
            return Ok(vec![]);
        }

        // Cache pre-pass — copy hits straight into the output slot,
        // collect misses to send to a worker. The whole pre-pass is
        // O(N) cache lookups; a cache disabled at cap=0 returns None
        // for every call so this collapses to a no-op N-step branch.
        let mut output: Vec<Option<Vec<f32>>> = vec![None; texts.len()];
        let mut miss_indices: Vec<usize> = Vec::new();
        let mut miss_texts: Vec<String> = Vec::new();
        for (i, t) in texts.iter().enumerate() {
            match self.cache.get(&self.expected_model_fingerprint, t) {
                Some(v) => output[i] = Some(v),
                None => {
                    miss_indices.push(i);
                    miss_texts.push(t.clone());
                }
            }
        }

        // All-cached fast path: no RPC, no EWMA update, no pool pick.
        if miss_texts.is_empty() {
            return Ok(output.into_iter().map(|o| o.expect("all hits")).collect());
        }

        let endpoint = self
            .pool
            .choose_two_random()
            .ok_or_else(|| ClusterError::AllWorkersFailed("no healthy workers in pool".into()))?;
        let start = std::time::Instant::now();
        // Caller-supplied request_id wins; empty falls through to the
        // random-id path inside the transport (mirrors embed_one).
        let mut items = if request_id.is_empty() {
            self.transport.embed_stream(&endpoint, &miss_texts, 0)?
        } else {
            self.transport
                .embed_stream_with_request_id(&endpoint, &miss_texts, 0, request_id)?
        };

        // Validate per-item against MISS batch (not full input batch).
        for it in &items {
            if it.vector.len() != self.dim {
                return Err(ClusterError::DimMismatch {
                    worker: endpoint.name.clone(),
                    expected: self.dim,
                    actual: it.vector.len(),
                });
            }
            if (it.index as usize) >= miss_texts.len() {
                return Err(ClusterError::Transport {
                    worker: endpoint.name.clone(),
                    reason: format!(
                        "stream emitted index {} for batch size {}",
                        it.index,
                        miss_texts.len()
                    ),
                });
            }
        }
        if items.len() != miss_texts.len() {
            return Err(ClusterError::Transport {
                worker: endpoint.name.clone(),
                reason: format!(
                    "stream returned {} items for batch size {}",
                    items.len(),
                    miss_texts.len()
                ),
            });
        }
        items.sort_by_key(|it| it.index);
        for w in items.windows(2) {
            if w[0].index == w[1].index {
                return Err(ClusterError::Transport {
                    worker: endpoint.name.clone(),
                    reason: format!("stream emitted duplicate index {}", w[0].index),
                });
            }
        }

        // Place each miss-result into its original input slot, populate
        // the cache for next time.
        for (it, &orig_idx) in items.iter().zip(miss_indices.iter()) {
            self.cache.insert(
                &self.expected_model_fingerprint,
                &miss_texts[it.index as usize],
                it.vector.clone(),
            );
            output[orig_idx] = Some(it.vector.clone());
        }

        self.pool
            .record_latency(&endpoint.name, start.elapsed(), 0.3);

        Ok(output
            .into_iter()
            .map(|o| o.expect("output filled"))
            .collect())
    }

    /// Async version of `embed_one_blocking`. The current GrpcTransport
    /// is sync (block_on internally); this wrapper hands the call to
    /// `tokio::task::spawn_blocking` so tokio-native callers can `await`
    /// it without stalling the executor. ~µs of overhead per call vs
    /// the sync path; dominated by the actual NPU inference time.
    ///
    /// Use when calling from inside a `#[tokio::main]` or any tokio runtime.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # async fn run() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = Arc::new(HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "",
    /// )?);
    /// // self: Arc<Self> consuming receiver — Arc::clone before each call.
    /// let v = Arc::clone(&cluster).embed_one("hello".into()).await?;
    /// assert_eq!(v.len(), 384);
    /// # Ok(()) }
    /// ```
    pub async fn embed_one(self: Arc<Self>, text: String) -> Result<Vec<f32>, ClusterError> {
        let me = self;
        tokio::task::spawn_blocking(move || me.embed_one_blocking(&text))
            .await
            .map_err(|e| ClusterError::Transport {
                worker: "<async-wrapper>".into(),
                reason: format!("spawn_blocking join: {}", e),
            })?
    }

    /// Async embed with caller-supplied `request_id`. Async sibling of
    /// `embed_one_blocking_with_request_id` — same spawn_blocking wrapper.
    pub async fn embed_one_with_request_id(
        self: Arc<Self>,
        text: String,
        request_id: String,
    ) -> Result<Vec<f32>, ClusterError> {
        let me = self;
        tokio::task::spawn_blocking(move || {
            me.embed_one_blocking_with_request_id(&text, &request_id)
        })
        .await
        .map_err(|e| ClusterError::Transport {
            worker: "<async-wrapper>".into(),
            reason: format!("spawn_blocking join: {}", e),
        })?
    }

    /// Async version of `embed_batch_blocking`. Uses the same
    /// spawn_blocking wrapper pattern as `embed_one`. The whole batch
    /// goes to one worker via the streaming RPC, so tokio-native
    /// concurrency at the *batch* level (multiple in-flight batches)
    /// runs many RPCs in parallel naturally.
    pub async fn embed_batch(
        self: Arc<Self>,
        texts: Vec<String>,
    ) -> Result<Vec<Vec<f32>>, ClusterError> {
        let me = self;
        tokio::task::spawn_blocking(move || me.embed_batch_blocking(&texts))
            .await
            .map_err(|e| ClusterError::Transport {
                worker: "<async-wrapper>".into(),
                reason: format!("spawn_blocking join: {}", e),
            })?
    }

    /// Async batch embed with caller-supplied `request_id`. Async
    /// sibling of `embed_batch_blocking_with_request_id`.
    pub async fn embed_batch_with_request_id(
        self: Arc<Self>,
        texts: Vec<String>,
        request_id: String,
    ) -> Result<Vec<Vec<f32>>, ClusterError> {
        let me = self;
        tokio::task::spawn_blocking(move || {
            me.embed_batch_blocking_with_request_id(&texts, &request_id)
        })
        .await
        .map_err(|e| ClusterError::Transport {
            worker: "<async-wrapper>".into(),
            reason: format!("spawn_blocking join: {}", e),
        })?
    }

    /// Fan out a `GetStats` RPC to every worker in the pool (regardless
    /// of healthy state — operators want to see ejected workers' last
    /// known counters too) and return per-worker results.
    ///
    /// Per-worker errors don't fail the call — they're captured in the
    /// `Result<StatsSnapshot>` element so callers can render them in a
    /// status table without a single bad worker spoiling the batch.
    pub fn fleet_stats(
        &self,
    ) -> Vec<(
        transport::WorkerEndpoint,
        Result<transport::StatsSnapshot, ClusterError>,
    )> {
        self.pool
            .all_endpoints()
            .into_iter()
            .map(|w| {
                let r = self.transport.stats(&w);
                (w, r)
            })
            .collect()
    }

    /// Richer-than-`fleet_stats` snapshot: paired health + stats RPC per
    /// worker, returning fingerprint alongside the counters. Two RPCs
    /// per worker per call — fine for the ops binary `ruvector-hailo-stats`,
    /// avoid in hot paths.
    pub fn fleet_state(&self) -> Vec<FleetMemberState> {
        self.pool
            .all_endpoints()
            .into_iter()
            .map(|w| {
                // One health() RPC per worker — pulls fingerprint + NPU
                // temps from the same call (iter 96 wired temps in).
                let health = self.transport.health(&w).ok();
                let fingerprint = health.as_ref().map(|h| h.model_fingerprint.clone());
                let npu_temp_ts0_celsius = health.as_ref().and_then(|h| h.npu_temp_ts0_celsius);
                let npu_temp_ts1_celsius = health.as_ref().and_then(|h| h.npu_temp_ts1_celsius);
                let stats = self.transport.stats(&w);
                FleetMemberState {
                    endpoint: w,
                    fingerprint,
                    npu_temp_ts0_celsius,
                    npu_temp_ts1_celsius,
                    stats,
                }
            })
            .collect()
    }

    /// Embed one text by picking a worker via P2C random + EWMA latency
    /// and dispatching through the transport. Retries up to
    /// `MAX_DISPATCH_RETRIES` times across different workers on transport
    /// failure (embedding is idempotent — safe to retry). Each successful
    /// call updates the pool's per-worker EWMA so subsequent picks favour
    /// the faster path.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "",
    /// )?;
    /// let v: Vec<f32> = cluster.embed_one_blocking("hello world")?;
    /// assert_eq!(v.len(), 384);
    /// # Ok(()) }
    /// ```
    pub fn embed_one_blocking(&self, text: &str) -> Result<Vec<f32>, ClusterError> {
        // Convenience wrapper: random correlation ID. Most callers want
        // this. Use `embed_one_blocking_with_request_id` to thread an
        // upstream tracing ID through the RPC.
        self.embed_one_blocking_with_request_id(text, "")
    }

    /// Embed with a caller-supplied `request_id` for cross-system log
    /// correlation. Empty string ≡ generate a random ID. The supplied
    /// id is propagated via gRPC metadata header (preferred) and the
    /// proto field (back-compat) so a worker's tracing span shows the
    /// same correlation token your caller logged.
    ///
    /// ```no_run
    /// # use std::sync::Arc;
    /// # use ruvector_hailo_cluster::{
    /// #     GrpcTransport, HailoClusterEmbedder, WorkerEndpoint,
    /// #     transport::EmbeddingTransport,
    /// # };
    /// # fn main() -> Result<(), Box<dyn std::error::Error>> {
    /// let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
    ///     Arc::new(GrpcTransport::new()?);
    /// let cluster = HailoClusterEmbedder::new(
    ///     vec![WorkerEndpoint::new("pi-a", "100.77.59.83:50051")],
    ///     transport, 384, "",
    /// )?;
    /// // Caller-supplied tracing token surfaces in worker logs.
    /// let trace_id = "ci-build-2741.user-42";
    /// let v = cluster.embed_one_blocking_with_request_id("hello", trace_id)?;
    /// assert_eq!(v.len(), 384);
    /// # Ok(()) }
    /// ```
    pub fn embed_one_blocking_with_request_id(
        &self,
        text: &str,
        request_id: &str,
    ) -> Result<Vec<f32>, ClusterError> {
        const MAX_DISPATCH_RETRIES: usize = 2;
        const EWMA_ALPHA: f64 = 0.3;
        const HEALTH_FAIL_THRESHOLD: u32 = 3;

        // Fast path: cache hit returns without touching the pool or wire.
        // Disabled cache (cap=0) is a single branch + lock-free atomic
        // capacity check, so this is ~ns-scale when off.
        if let Some(v) = self.cache.get(&self.expected_model_fingerprint, text) {
            return Ok(v);
        }

        let mut last_err: Option<ClusterError> = None;

        for _attempt in 0..=MAX_DISPATCH_RETRIES {
            let endpoint = match self.pool.choose_two_random() {
                Some(e) => e,
                None => {
                    return Err(ClusterError::AllWorkersFailed(
                        "no healthy workers in pool".into(),
                    ));
                }
            };

            let start = std::time::Instant::now();
            // If the caller passed a request_id, route through the
            // tagged transport call so the metadata header carries
            // *their* id. Empty string falls through to the default
            // (random ID generated inside the transport).
            let result = if request_id.is_empty() {
                self.transport.embed(&endpoint, text, 0)
            } else {
                self.transport
                    .embed_with_request_id(&endpoint, text, 0, request_id)
            };
            match result {
                Ok((vec, _server_latency_us)) => {
                    if vec.len() != self.dim {
                        return Err(ClusterError::DimMismatch {
                            worker: endpoint.name,
                            expected: self.dim,
                            actual: vec.len(),
                        });
                    }
                    self.pool
                        .record_latency(&endpoint.name, start.elapsed(), EWMA_ALPHA);
                    // Populate the cache on success. No-op if cap=0.
                    self.cache
                        .insert(&self.expected_model_fingerprint, text, vec.clone());
                    return Ok(vec);
                }
                Err(e) => {
                    // Iter 209 — short-circuit on deterministic errors
                    // that won't change on retry (iter-180 OutOfRange,
                    // iter-199 InvalidArgument, iter-104/200
                    // ResourceExhausted, plus dim/fingerprint
                    // mismatches). Without this, every byte-cap or
                    // batch-cap rejection burns the full 3-attempt
                    // retry budget — and for rate limiting it actively
                    // makes things worse: each retry consumes another
                    // token from the same peer's bucket, deepening
                    // the rate-limit hole the caller is already in.
                    if e.is_terminal() {
                        return Err(e);
                    }
                    self.pool
                        .record_health_failure(&endpoint.name, HEALTH_FAIL_THRESHOLD);
                    last_err = Some(e);
                }
            }
        }

        Err(ClusterError::AllWorkersFailed(format!(
            "after {} attempts: {}",
            MAX_DISPATCH_RETRIES + 1,
            match last_err {
                Some(e) => e.to_string(),
                None => "unknown".into(),
            }
        )))
    }
}

/// Iter 218 — closes ADR-178 Gap B (HIGH) part 1. Implements
/// `ruvector_core::embeddings::EmbeddingProvider` for
/// `HailoClusterEmbedder`, the headline integration ADR-167 §8.4
/// promised. The cluster doc-comment at lib.rs line 140 had been
/// honestly admitting this gap ("Implements `EmbeddingProvider` once
/// iteration 14 brings the path dep"); iter-218 finally lands it.
///
/// All three trait methods delegate to existing inherent methods.
/// `embed` folds `ClusterError → RuvectorError::ModelInferenceError`
/// (the iter-209 terminal-error short-circuit still fires inside
/// `embed_one_blocking` before we hit this conversion). `name()`
/// returns the static crate identifier since the cluster doesn't
/// otherwise carry a single-name handle (it's a fleet of workers).
///
/// Effect: callers can now hold `Arc<dyn EmbeddingProvider>` and
/// transparently swap a single-Pi `HailoEmbedder` for a fleet
/// `HailoClusterEmbedder` without code changes — the contract
/// ADR-167 §8.4 promised end-to-end.
impl ruvector_core::embeddings::EmbeddingProvider for HailoClusterEmbedder {
    fn embed(&self, text: &str) -> ruvector_core::Result<Vec<f32>> {
        HailoClusterEmbedder::embed_one_blocking(self, text)
            .map_err(|e| ruvector_core::RuvectorError::ModelInferenceError(e.to_string()))
    }

    fn dimensions(&self) -> usize {
        HailoClusterEmbedder::dim(self)
    }

    fn name(&self) -> &str {
        // The cluster is a fleet, not a single named device — the
        // worker-level `device_id` lives on each WorkerEndpoint.
        // Return a static identifier so callers can distinguish a
        // cluster provider from a `HailoEmbedder` in logs.
        "ruvector-hailo-cluster"
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::atomic::{AtomicUsize, Ordering};
    use transport::HealthReport;

    #[test]
    fn empty_worker_list_rejected() {
        let r =
            HailoClusterEmbedder::new(vec![], transport::null_transport(), 384, "fingerprint:test");
        assert!(matches!(r, Err(ClusterError::NoWorkers)));
    }

    #[test]
    fn coordinator_carries_dim_and_fingerprint() {
        let workers = vec![
            WorkerEndpoint::new("pi-a", "100.77.59.83:50051"),
            WorkerEndpoint::new("pi-b", "100.77.59.84:50051"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport::null_transport(), 384, "fp:abc")
            .expect("two workers");
        assert_eq!(c.dim(), 384);
        assert_eq!(c.worker_count(), 2);
    }

    /// In-process fake transport for unit testing the dispatch loop without
    /// gRPC. Records call counts; programmable success/failure per call.
    struct FakeTransport {
        ok_vector: Option<Vec<f32>>,
        bad_vector_dim: Option<usize>, // if set, return a vec of this length
        fail_first_n: usize,
        calls: AtomicUsize,
    }
    impl FakeTransport {
        fn always_ok(v: Vec<f32>) -> Self {
            Self {
                ok_vector: Some(v),
                bad_vector_dim: None,
                fail_first_n: 0,
                calls: AtomicUsize::new(0),
            }
        }
        fn fail_then_ok(fail_count: usize, then_v: Vec<f32>) -> Self {
            Self {
                ok_vector: Some(then_v),
                bad_vector_dim: None,
                fail_first_n: fail_count,
                calls: AtomicUsize::new(0),
            }
        }
        fn always_wrong_dim(d: usize) -> Self {
            Self {
                ok_vector: None,
                bad_vector_dim: Some(d),
                fail_first_n: 0,
                calls: AtomicUsize::new(0),
            }
        }
    }
    impl EmbeddingTransport for FakeTransport {
        fn embed(
            &self,
            worker: &WorkerEndpoint,
            _text: &str,
            _max_seq: u32,
        ) -> Result<(Vec<f32>, u64), ClusterError> {
            let n = self.calls.fetch_add(1, Ordering::SeqCst);
            if n < self.fail_first_n {
                return Err(ClusterError::Transport {
                    worker: worker.name.clone(),
                    reason: "fake fail".into(),
                });
            }
            if let Some(d) = self.bad_vector_dim {
                return Ok((vec![0.0; d], 1));
            }
            if let Some(v) = &self.ok_vector {
                return Ok((v.clone(), 1));
            }
            Err(ClusterError::Transport {
                worker: worker.name.clone(),
                reason: "fake exhausted".into(),
            })
        }
        fn health(&self, _worker: &WorkerEndpoint) -> Result<HealthReport, ClusterError> {
            Ok(HealthReport {
                version: "fake".into(),
                device_id: "fake:0".into(),
                model_fingerprint: "fp:fake".into(),
                ready: true,
                npu_temp_ts0_celsius: None,
                npu_temp_ts1_celsius: None,
            })
        }
    }

    fn workers(n: usize) -> Vec<WorkerEndpoint> {
        (0..n)
            .map(|i| WorkerEndpoint::new(format!("pi-{}", i), format!("10.0.0.{}:50051", i)))
            .collect()
    }

    #[test]
    fn dispatch_succeeds_on_first_try_returns_vector() {
        let transport = Arc::new(FakeTransport::always_ok(vec![1.0, 2.0, 3.0]));
        let c =
            HailoClusterEmbedder::new(workers(2), transport.clone(), 3, "fp:test").expect("init");
        let v = c.embed_one_blocking("hello").expect("embed should succeed");
        assert_eq!(v, vec![1.0, 2.0, 3.0]);
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn dispatch_retries_on_transient_failure_and_succeeds() {
        // Fake fails twice, succeeds on the 3rd call. Coordinator's retry
        // budget is 3 attempts (initial + MAX_DISPATCH_RETRIES=2) so it
        // should reach the 3rd call.
        let transport = Arc::new(FakeTransport::fail_then_ok(2, vec![1.0, 2.0]));
        let c =
            HailoClusterEmbedder::new(workers(3), transport.clone(), 2, "fp:test").expect("init");
        let v = c.embed_one_blocking("hello").expect("retry should land");
        assert_eq!(v, vec![1.0, 2.0]);
        assert_eq!(transport.calls.load(Ordering::SeqCst), 3);
    }

    #[test]
    fn dispatch_returns_all_workers_failed_after_budget_exhausted() {
        // Fake fails 99 times — coordinator gives up after MAX+1 attempts.
        let transport = Arc::new(FakeTransport::fail_then_ok(99, vec![]));
        let c =
            HailoClusterEmbedder::new(workers(3), transport.clone(), 384, "fp:test").expect("init");
        let r = c.embed_one_blocking("hello");
        assert!(matches!(r, Err(ClusterError::AllWorkersFailed(_))));
    }

    #[test]
    fn dispatch_rejects_dim_mismatch_immediately() {
        // Worker returns 7-dim vector but coordinator expects 384.
        let transport = Arc::new(FakeTransport::always_wrong_dim(7));
        let c =
            HailoClusterEmbedder::new(workers(2), transport.clone(), 384, "fp:test").expect("init");
        match c.embed_one_blocking("hello") {
            Err(ClusterError::DimMismatch {
                expected, actual, ..
            }) => {
                assert_eq!(expected, 384);
                assert_eq!(actual, 7);
            }
            other => panic!("expected DimMismatch, got {:?}", other.map(|_| "ok")),
        }
        // DimMismatch is fatal — exactly one call.
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);
    }

    #[test]
    fn cache_hits_skip_transport_after_first_call() {
        let transport = Arc::new(FakeTransport::always_ok(vec![0.1, 0.2, 0.3]));
        let c = HailoClusterEmbedder::new(workers(2), transport.clone(), 3, "fp:cache")
            .expect("init")
            .with_cache(16);

        // First call → miss → transport hit (count = 1)
        let v1 = c.embed_one_blocking("warm me up").expect("first embed");
        assert_eq!(v1, vec![0.1, 0.2, 0.3]);
        assert_eq!(transport.calls.load(Ordering::SeqCst), 1);

        // Second call, same text → hit → transport NOT touched.
        let v2 = c.embed_one_blocking("warm me up").expect("cached embed");
        assert_eq!(v2, vec![0.1, 0.2, 0.3]);
        assert_eq!(
            transport.calls.load(Ordering::SeqCst),
            1,
            "cache must skip transport"
        );

        // Third call, different text → miss → transport count grows.
        let _ = c.embed_one_blocking("cold").expect("second text");
        assert_eq!(transport.calls.load(Ordering::SeqCst), 2);

        let s = c.cache_stats();
        assert_eq!(s.size, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 2);
        assert_eq!(s.evictions, 0);
        assert_eq!(s.capacity, 16);
    }

    /// Test transport whose health() output varies per-worker based on a
    /// supplied map. Used by validate_fleet tests to exercise mixed
    /// fleets (some healthy, some on a different model, some unreachable).
    struct PerWorkerHealth {
        outcomes: std::collections::HashMap<String, ValidationOutcome>,
    }
    enum ValidationOutcome {
        Ready { fingerprint: String },
        NotReady { fingerprint: String },
        Unreachable,
    }
    impl EmbeddingTransport for PerWorkerHealth {
        fn embed(
            &self,
            _: &WorkerEndpoint,
            _: &str,
            _: u32,
        ) -> Result<(Vec<f32>, u64), ClusterError> {
            Ok((vec![0.0; 4], 1))
        }
        fn health(&self, w: &WorkerEndpoint) -> Result<HealthReport, ClusterError> {
            match self.outcomes.get(&w.name) {
                Some(ValidationOutcome::Ready { fingerprint }) => Ok(HealthReport {
                    version: "test".into(),
                    device_id: format!("test:{}", w.name),
                    model_fingerprint: fingerprint.clone(),
                    ready: true,
                    npu_temp_ts0_celsius: None,
                    npu_temp_ts1_celsius: None,
                }),
                Some(ValidationOutcome::NotReady { fingerprint }) => Ok(HealthReport {
                    version: "test".into(),
                    device_id: format!("test:{}", w.name),
                    model_fingerprint: fingerprint.clone(),
                    ready: false,
                    npu_temp_ts0_celsius: None,
                    npu_temp_ts1_celsius: None,
                }),
                _ => Err(ClusterError::Transport {
                    worker: w.name.clone(),
                    reason: "test: unreachable".into(),
                }),
            }
        }
    }

    #[test]
    fn validate_fleet_accepts_homogeneous_fleet() {
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:abc".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:abc".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "fp:abc").expect("init");
        let report = c.validate_fleet().expect("should pass");
        assert_eq!(report.healthy.len(), 2);
        assert_eq!(report.fingerprint_mismatched.len(), 0);
        assert_eq!(report.not_ready.len(), 0);
        assert_eq!(report.unreachable.len(), 0);
    }

    #[test]
    fn validate_fleet_ejects_fingerprint_mismatch() {
        // pi-0 on the right model, pi-1 on a stale build.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:current".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:stale".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c =
            HailoClusterEmbedder::new(workers, transport.clone(), 4, "fp:current").expect("init");
        let report = c.validate_fleet().expect("at least one healthy → ok");
        assert_eq!(report.healthy, vec!["pi-0"]);
        assert_eq!(report.fingerprint_mismatched.len(), 1);
        assert_eq!(report.fingerprint_mismatched[0].worker, "pi-1");
        assert_eq!(report.fingerprint_mismatched[0].expected, "fp:current");
        assert_eq!(report.fingerprint_mismatched[0].actual, "fp:stale");

        // Confirm pi-1 was actually ejected — embed_one_blocking should
        // never pick it. Run 20 calls; FakeTransport is single-fingerprint
        // so every Ok() proves we hit pi-0 (the only healthy worker).
        for _ in 0..20 {
            let v = c
                .embed_one_blocking("test")
                .expect("should always land on pi-0");
            assert_eq!(v, vec![0.0; 4]);
        }
    }

    #[test]
    fn validate_fleet_fails_when_no_workers_healthy() {
        // Every worker on the wrong fingerprint.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:wrong".into(),
            },
        );
        outcomes.insert("pi-1".into(), ValidationOutcome::Unreachable);
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "fp:current").expect("init");
        match c.validate_fleet() {
            Err(ClusterError::AllWorkersFailed(msg)) => {
                assert!(msg.contains("0 healthy"));
                assert!(msg.contains("1 mismatched fp"));
                assert!(msg.contains("1 unreachable"));
            }
            other => panic!("expected AllWorkersFailed, got {:?}", other.map(|_| "ok")),
        }
    }

    #[test]
    fn discover_fingerprint_returns_first_workers_fingerprint() {
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:discovered".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:other".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        let fp = c.discover_fingerprint().expect("first worker reachable");
        // First worker in pool order = pi-0 → returns fp:discovered.
        // We don't enforce homogeneity here; that's validate_fleet's job.
        assert_eq!(fp, "fp:discovered");
    }

    #[test]
    fn discover_fingerprint_falls_through_unreachable_workers() {
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert("pi-0".into(), ValidationOutcome::Unreachable);
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:second".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        let fp = c.discover_fingerprint().expect("pi-1 reachable");
        assert_eq!(
            fp, "fp:second",
            "should skip unreachable pi-0 and land on pi-1"
        );
    }

    #[test]
    fn discover_fingerprint_errors_when_all_unreachable() {
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert("pi-0".into(), ValidationOutcome::Unreachable);
        outcomes.insert("pi-1".into(), ValidationOutcome::Unreachable);
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        match c.discover_fingerprint() {
            Err(ClusterError::AllWorkersFailed(msg)) => {
                assert!(msg.contains("discover_fingerprint"));
            }
            other => panic!(
                "expected AllWorkersFailed, got {:?}",
                other.map(|s| s.to_string())
            ),
        }
    }

    // ---- ADR-172 §2b iter-102 quorum tests ----

    #[test]
    fn quorum_majority_agrees_returns_majority_fp() {
        // 3 workers; 2 report fp:A, 1 reports fp:B. quorum=2 should
        // pick fp:A and ignore the lone fp:B.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:A".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:A".into(),
            },
        );
        outcomes.insert(
            "pi-2".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:B".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });
        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
            WorkerEndpoint::new("pi-2", "10.0.0.2:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        let fp = c.discover_fingerprint_with_quorum(2).expect("majority hit");
        assert_eq!(fp, "fp:A");
    }

    #[test]
    fn quorum_no_majority_errors_with_tally() {
        // 3 workers, 3 different fingerprints — no quorum possible.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:A".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:B".into(),
            },
        );
        outcomes.insert(
            "pi-2".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:C".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });
        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
            WorkerEndpoint::new("pi-2", "10.0.0.2:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        match c.discover_fingerprint_with_quorum(2) {
            Err(ClusterError::AllWorkersFailed(msg)) => {
                assert!(msg.contains("need 2"), "expected quorum=2 message: {}", msg);
                assert!(msg.contains("tally="), "expected tally in error: {}", msg);
            }
            other => panic!(
                "expected AllWorkersFailed, got {:?}",
                other.map(|s| s.to_string())
            ),
        }
    }

    #[test]
    fn quorum_one_acts_like_single_witness() {
        // min_agree=1 lets a single-worker dev fleet still use quorum
        // discovery without changing semantics from discover_fingerprint().
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:solo".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });
        let workers = vec![WorkerEndpoint::new("pi-0", "10.0.0.0:1")];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        let fp = c.discover_fingerprint_with_quorum(1).expect("solo worker");
        assert_eq!(fp, "fp:solo");
    }

    #[test]
    fn quorum_excludes_empty_fingerprints_from_tally() {
        // 3 workers, all return empty fingerprint (legacy fleet).
        // Tally is empty → best_count=0 → quorum>=1 fails. This
        // protects against treating "no model" as quorum agreement.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "".into(),
            },
        );
        outcomes.insert(
            "pi-2".into(),
            ValidationOutcome::Ready {
                fingerprint: "".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });
        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
            WorkerEndpoint::new("pi-2", "10.0.0.2:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        match c.discover_fingerprint_with_quorum(1) {
            Err(ClusterError::AllWorkersFailed(msg)) => {
                assert!(
                    msg.contains("0 agreeing"),
                    "expected zero-agreement msg: {}",
                    msg
                );
            }
            other => panic!(
                "expected error for empty-fp tally, got {:?}",
                other.map(|s| s.to_string())
            ),
        }
    }

    #[test]
    fn quorum_all_unreachable_errors_with_per_worker_reasons() {
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert("pi-0".into(), ValidationOutcome::Unreachable);
        outcomes.insert("pi-1".into(), ValidationOutcome::Unreachable);
        let transport = Arc::new(PerWorkerHealth { outcomes });
        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        match c.discover_fingerprint_with_quorum(2) {
            Err(ClusterError::AllWorkersFailed(msg)) => {
                assert!(msg.contains("pi-0"), "expected per-worker err: {}", msg);
                assert!(msg.contains("pi-1"), "expected per-worker err: {}", msg);
            }
            other => panic!(
                "expected AllWorkersFailed, got {:?}",
                other.map(|s| s.to_string())
            ),
        }
    }

    #[test]
    fn validate_fleet_skips_check_with_empty_expected_fingerprint() {
        // expected_model_fingerprint = "" means "trust everyone, only
        // care about ready=true". Useful for legacy fleets where the
        // operator hasn't pinned a fingerprint yet.
        let mut outcomes = std::collections::HashMap::new();
        outcomes.insert(
            "pi-0".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:any".into(),
            },
        );
        outcomes.insert(
            "pi-1".into(),
            ValidationOutcome::Ready {
                fingerprint: "fp:other".into(),
            },
        );
        let transport = Arc::new(PerWorkerHealth { outcomes });

        let workers = vec![
            WorkerEndpoint::new("pi-0", "10.0.0.0:1"),
            WorkerEndpoint::new("pi-1", "10.0.0.1:1"),
        ];
        let c = HailoClusterEmbedder::new(workers, transport, 4, "").expect("init");
        let report = c.validate_fleet().expect("empty fp → no check");
        assert_eq!(report.healthy.len(), 2, "both should pass without fp check");
    }

    #[test]
    fn invalidate_cache_drops_all_cached_entries() {
        let transport = Arc::new(FakeTransport::always_ok(vec![0.5, 0.5, 0.5]));
        let c = HailoClusterEmbedder::new(workers(2), transport.clone(), 3, "fp:invalidate")
            .expect("init")
            .with_cache(16);

        // Warm 2 entries.
        let _ = c.embed_one_blocking("hello").unwrap();
        let _ = c.embed_one_blocking("world").unwrap();
        assert_eq!(transport.calls.load(Ordering::SeqCst), 2);
        assert_eq!(c.cache_stats().size, 2);

        // Invalidate — explicit "drop everything" call.
        let dropped = c.invalidate_cache();
        assert_eq!(dropped, 2);
        let s = c.cache_stats();
        assert_eq!(s.size, 0);
        assert_eq!(s.evictions, 2, "explicit drop counts as eviction");

        // Same input now misses → transport call count grows.
        let _ = c.embed_one_blocking("hello").unwrap();
        assert_eq!(
            transport.calls.load(Ordering::SeqCst),
            3,
            "post-invalidate hit must reach the transport"
        );
    }

    #[test]
    fn cache_disabled_by_default() {
        let transport = Arc::new(FakeTransport::always_ok(vec![1.0; 3]));
        let c = HailoClusterEmbedder::new(workers(2), transport.clone(), 3, "fp:nocache")
            .expect("init");

        // No `.with_cache(...)` — capacity 0, every call hits the transport.
        for _ in 0..5 {
            let _ = c.embed_one_blocking("same text").unwrap();
        }
        assert_eq!(transport.calls.load(Ordering::SeqCst), 5);

        let s = c.cache_stats();
        assert_eq!(s.capacity, 0);
        assert_eq!(s.size, 0);
        // Disabled cache stays at zero across the board.
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
    }
}
