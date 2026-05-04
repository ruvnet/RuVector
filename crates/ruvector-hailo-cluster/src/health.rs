//! Background health-check loop.
//!
//! ADR-167 §8.3: each worker probed periodically (default 5 s); 3
//! consecutive failures eject. A successful health probe of an ejected
//! worker re-promotes it. The loop runs on the supplied tokio runtime
//! and is owned by the cluster coordinator.
//!
//! The pool already exposes `record_health_failure` / `mark_healthy`
//! (used opportunistically by the dispatch loop). This module turns that
//! into a *steady-state* signal independent of traffic — important for
//! correctness when a worker comes back from a network blip and there
//! are no in-flight embeds to discover that.

use crate::pool::P2cPool;
use crate::transport::{EmbeddingTransport, WorkerEndpoint};
use std::sync::Arc;
use std::time::{Duration, Instant};
use tokio::task::JoinHandle;
use tracing::{debug, info, instrument, warn};

/// Configuration for the health checker. `Debug` is intentionally NOT
/// derived because `on_fingerprint_mismatch` is a `dyn Fn` trait object
/// — opaque by nature. If you need to inspect a config, log its fields
/// individually.
#[derive(Clone)]
pub struct HealthCheckerConfig {
    /// How often to probe each worker.
    pub interval: Duration,
    /// Failures before ejection.
    pub fail_threshold: u32,
    /// Per-probe deadline.
    pub probe_timeout: Duration,
    /// EWMA smoothing factor used for latency observations from health
    /// probes. Smaller than the dispatch-side α (0.3) because health
    /// probes are far cheaper than full embed calls and shouldn't
    /// dominate the latency estimate.
    pub ewma_alpha: f64,
    /// If `Some(fp)`, every successful probe also enforces fingerprint
    /// equality. A worker reporting a different fingerprint is hard-
    /// ejected (`pool.eject`) regardless of `fail_threshold` — that's
    /// a hard model-skew event, not a transient blip. `None` skips the
    /// runtime check (matches `validate_fleet` semantics for empty fp).
    pub expected_fingerprint: Option<String>,
    /// Optional callback invoked after a worker is ejected for
    /// fingerprint mismatch. Used by the cluster to clear caches that
    /// might contain vectors served from the now-suspect worker. Kept
    /// as a generic callback so the health module stays independent of
    /// the cache implementation.
    pub on_fingerprint_mismatch: Option<Arc<dyn Fn() + Send + Sync>>,
}

impl Default for HealthCheckerConfig {
    fn default() -> Self {
        Self {
            interval: Duration::from_secs(5),
            fail_threshold: 3,
            probe_timeout: Duration::from_secs(2),
            ewma_alpha: 0.1,
            expected_fingerprint: None,
            on_fingerprint_mismatch: None,
        }
    }
}

/// Owns the background probe task. Drop the handle to stop the loop.
pub struct HealthChecker {
    handle: JoinHandle<()>,
}

impl HealthChecker {
    /// Spawn the loop on the supplied tokio runtime handle.
    /// The supplied `transport` is wrapped in `Arc` so the task can hold
    /// it across the lifetime of the runtime.
    pub fn spawn(
        rt: &tokio::runtime::Handle,
        pool: Arc<P2cPool>,
        workers: Vec<WorkerEndpoint>,
        transport: Arc<dyn EmbeddingTransport + Send + Sync>,
        cfg: HealthCheckerConfig,
    ) -> Self {
        let handle = rt.spawn(async move {
            run_loop(pool, workers, transport, cfg).await;
        });
        Self { handle }
    }

    /// Stop the background loop. Idempotent.
    pub fn stop(self) {
        self.handle.abort();
    }

    /// `true` if the background loop is still running. Becomes `false`
    /// after `stop()` or after the spawning runtime is dropped.
    pub fn is_running(&self) -> bool {
        !self.handle.is_finished()
    }
}

#[instrument(skip(pool, transport, workers, cfg), fields(workers = workers.len()))]
async fn run_loop(
    pool: Arc<P2cPool>,
    workers: Vec<WorkerEndpoint>,
    transport: Arc<dyn EmbeddingTransport + Send + Sync>,
    cfg: HealthCheckerConfig,
) {
    info!(
        interval_ms = cfg.interval.as_millis() as u64,
        threshold = cfg.fail_threshold,
        "health-check loop started"
    );

    let mut tick = tokio::time::interval(cfg.interval);
    // Skip the first immediate tick — let the dispatcher settle first.
    tick.tick().await;

    loop {
        tick.tick().await;
        for w in &workers {
            probe_one(&pool, &transport, w, &cfg).await;
        }
    }
}

#[instrument(skip(pool, transport, cfg), fields(worker = %w.name))]
async fn probe_one(
    pool: &P2cPool,
    transport: &Arc<dyn EmbeddingTransport + Send + Sync>,
    w: &WorkerEndpoint,
    cfg: &HealthCheckerConfig,
) {
    // Clone what the spawn_blocking closure needs to own. `pool` and `cfg`
    // stay borrowed because they're used only after the await returns.
    let w_owned: WorkerEndpoint = w.clone();
    let transport_owned: Arc<dyn EmbeddingTransport + Send + Sync> = Arc::clone(transport);
    let timeout = cfg.probe_timeout;
    let alpha = cfg.ewma_alpha;
    let threshold = cfg.fail_threshold;
    let expected_fp = cfg.expected_fingerprint.clone();
    let on_mismatch = cfg.on_fingerprint_mismatch.clone();
    // The transport's health() is sync (blocking); run on a blocking
    // pool so it doesn't stall the timer.
    let outcome = tokio::task::spawn_blocking(move || {
        let start = Instant::now();
        let r = transport_owned.health(&w_owned);
        (r, start.elapsed(), w_owned)
    })
    .await;
    match outcome {
        Ok((Ok(report), elapsed, worker)) => {
            // Hard fingerprint mismatch — model swapped under us.
            // Eject regardless of ready / latency state; the operator
            // sees a clear log line and stale dispatch stops immediately.
            if let Some(expected) = &expected_fp {
                if !expected.is_empty() && &report.model_fingerprint != expected {
                    warn!(
                        expected = %expected,
                        actual = %report.model_fingerprint,
                        "fingerprint mismatch on live worker — ejecting"
                    );
                    pool.eject(&worker.name);
                    // Cache invalidation hook: callbacks set by the
                    // cluster wipe stale vectors served from this
                    // worker. Best-effort; failures don't unwind the
                    // ejection (which already happened).
                    if let Some(cb) = &on_mismatch {
                        cb();
                    }
                    return;
                }
            }
            if report.ready && elapsed <= timeout {
                debug!(
                    elapsed_us = elapsed.as_micros() as u64,
                    fingerprint = %report.model_fingerprint,
                    "health ok"
                );
                pool.mark_healthy(&worker.name);
                pool.record_latency(&worker.name, elapsed, alpha);
            } else if !report.ready {
                warn!("worker reports ready=false");
                pool.record_health_failure(&worker.name, threshold);
            } else {
                warn!(
                    elapsed_us = elapsed.as_micros() as u64,
                    "health probe timed out"
                );
                pool.record_health_failure(&worker.name, threshold);
            }
        }
        Ok((Err(e), _, worker)) => {
            warn!(error = %e, "health probe failed");
            pool.record_health_failure(&worker.name, threshold);
        }
        Err(join_err) => {
            warn!(error = %join_err, "health probe task join failed");
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::transport::HealthReport;
    use std::sync::atomic::{AtomicU64, Ordering};

    /// Toy transport whose health() return value can be flipped at runtime
    /// — lets a single test drive ready→not-ready→ready transitions.
    struct ToggleTransport {
        ready: std::sync::atomic::AtomicBool,
        probes: AtomicU64,
    }
    impl ToggleTransport {
        fn new(ready: bool) -> Self {
            Self {
                ready: std::sync::atomic::AtomicBool::new(ready),
                probes: AtomicU64::new(0),
            }
        }
        fn set_ready(&self, b: bool) {
            self.ready.store(b, Ordering::SeqCst);
        }
        fn probe_count(&self) -> u64 {
            self.probes.load(Ordering::SeqCst)
        }
    }
    impl EmbeddingTransport for ToggleTransport {
        fn embed(
            &self,
            _: &WorkerEndpoint,
            _: &str,
            _: u32,
        ) -> Result<(Vec<f32>, u64), crate::error::ClusterError> {
            Ok((vec![0.0; 4], 1))
        }
        fn health(&self, _: &WorkerEndpoint) -> Result<HealthReport, crate::error::ClusterError> {
            self.probes.fetch_add(1, Ordering::SeqCst);
            Ok(HealthReport {
                version: "toggle".into(),
                device_id: "toggle:0".into(),
                model_fingerprint: "fp:toggle".into(),
                ready: self.ready.load(Ordering::SeqCst),
                npu_temp_ts0_celsius: None,
                npu_temp_ts1_celsius: None,
            })
        }
    }

    #[test]
    fn health_loop_marks_unhealthy_after_threshold_failures() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let workers = vec![WorkerEndpoint::new("w0", "127.0.0.1:1")];
        let pool = Arc::new(P2cPool::new(workers.clone()));

        // Always-not-ready transport.
        let toggle = Arc::new(ToggleTransport::new(false));
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> = toggle.clone();

        let cfg = HealthCheckerConfig {
            interval: Duration::from_millis(20),
            probe_timeout: Duration::from_millis(50),
            fail_threshold: 3,
            ewma_alpha: 0.1,
            expected_fingerprint: None,
            on_fingerprint_mismatch: None,
        };

        let checker = HealthChecker::spawn(rt.handle(), pool.clone(), workers, transport, cfg);

        // Wait long enough for >=3 probes (interval 20ms × 4 = 80ms).
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(150)).await });
        checker.stop();

        // After 3 not-ready probes, worker should be ejected.
        assert!(toggle.probe_count() >= 3);
        // Pool size is unchanged (workers don't get removed, just ejected
        // from the healthy pick-pool).
        assert_eq!(pool.size(), 1);
        // Verify ejection by checking choose_two_random returns None.
        assert!(pool.choose_two_random().is_none());
    }

    /// Simple transport that always reports a configurable fingerprint
    /// and `ready=true`. Used by the fingerprint-runtime-check test.
    struct FixedFingerprintTransport {
        fingerprint: String,
    }
    impl EmbeddingTransport for FixedFingerprintTransport {
        fn embed(
            &self,
            _: &WorkerEndpoint,
            _: &str,
            _: u32,
        ) -> Result<(Vec<f32>, u64), crate::error::ClusterError> {
            Ok((vec![0.0; 4], 1))
        }
        fn health(&self, _: &WorkerEndpoint) -> Result<HealthReport, crate::error::ClusterError> {
            Ok(HealthReport {
                version: "fixed".into(),
                device_id: "fixed:0".into(),
                model_fingerprint: self.fingerprint.clone(),
                ready: true,
                npu_temp_ts0_celsius: None,
                npu_temp_ts1_celsius: None,
            })
        }
    }

    #[test]
    fn health_loop_ejects_runtime_fingerprint_mismatch() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let workers = vec![WorkerEndpoint::new("w0", "127.0.0.1:1")];
        let pool = Arc::new(P2cPool::new(workers.clone()));

        // Worker reports fp:stale, but coordinator expects fp:current.
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
            Arc::new(FixedFingerprintTransport {
                fingerprint: "fp:stale".into(),
            });

        let cfg = HealthCheckerConfig {
            interval: Duration::from_millis(20),
            probe_timeout: Duration::from_millis(50),
            fail_threshold: 3,
            ewma_alpha: 0.1,
            expected_fingerprint: Some("fp:current".into()),
            on_fingerprint_mismatch: None,
        };

        let checker = HealthChecker::spawn(rt.handle(), pool.clone(), workers, transport, cfg);

        // One probe is enough — fingerprint mismatch is a hard ejection,
        // not a threshold-based one. Wait ~50ms to ensure ≥1 tick fired.
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(60)).await });
        checker.stop();

        // Worker should be ejected from the healthy pick-pool.
        assert!(
            pool.choose_two_random().is_none(),
            "worker with stale fingerprint should be ejected after first probe"
        );
        // Pool size unchanged — eject is healthy=false, not removal.
        assert_eq!(pool.size(), 1);
    }

    #[test]
    fn health_loop_invokes_callback_on_fingerprint_mismatch() {
        // Wire a counter — the closure increments it on every invocation.
        // Health loop should call it exactly when the worker is ejected
        // for fingerprint mismatch.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let workers = vec![WorkerEndpoint::new("w0", "127.0.0.1:1")];
        let pool = Arc::new(P2cPool::new(workers.clone()));
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
            Arc::new(FixedFingerprintTransport {
                fingerprint: "fp:stale".into(),
            });

        let invoked = Arc::new(AtomicU64::new(0));
        let invoked_clone = invoked.clone();
        let cb: Arc<dyn Fn() + Send + Sync> = Arc::new(move || {
            invoked_clone.fetch_add(1, Ordering::SeqCst);
        });

        let cfg = HealthCheckerConfig {
            interval: Duration::from_millis(20),
            probe_timeout: Duration::from_millis(50),
            fail_threshold: 3,
            ewma_alpha: 0.1,
            expected_fingerprint: Some("fp:current".into()),
            on_fingerprint_mismatch: Some(cb),
        };

        let checker = HealthChecker::spawn(rt.handle(), pool.clone(), workers, transport, cfg);
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(60)).await });
        checker.stop();

        // Callback should have fired at least once (one mismatch detected).
        assert!(
            invoked.load(Ordering::SeqCst) >= 1,
            "callback should fire on fingerprint mismatch, got {}",
            invoked.load(Ordering::SeqCst)
        );
        // Worker still ejected (separate hard-eject path verified by
        // `health_loop_ejects_runtime_fingerprint_mismatch`).
        assert!(pool.choose_two_random().is_none());
    }

    #[test]
    fn health_loop_skips_fingerprint_check_when_expected_is_none() {
        // Pre-iter-47 behavior: no expected fingerprint → loop only
        // checks ready flag. A worker reporting any fingerprint stays
        // healthy as long as ready=true.
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let workers = vec![WorkerEndpoint::new("w0", "127.0.0.1:1")];
        let pool = Arc::new(P2cPool::new(workers.clone()));
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> =
            Arc::new(FixedFingerprintTransport {
                fingerprint: "fp:whatever".into(),
            });

        let cfg = HealthCheckerConfig {
            interval: Duration::from_millis(20),
            probe_timeout: Duration::from_millis(50),
            fail_threshold: 3,
            ewma_alpha: 0.1,
            expected_fingerprint: None,
            on_fingerprint_mismatch: None,
        };

        let checker = HealthChecker::spawn(rt.handle(), pool.clone(), workers, transport, cfg);
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(60)).await });
        checker.stop();

        // Worker stays healthy — no fingerprint enforcement at runtime.
        assert!(
            pool.choose_two_random().is_some(),
            "no expected fp ⇒ worker stays healthy regardless of reported fp"
        );
    }

    #[test]
    fn health_loop_re_promotes_recovered_worker() {
        let rt = tokio::runtime::Builder::new_multi_thread()
            .worker_threads(2)
            .enable_all()
            .build()
            .unwrap();

        let workers = vec![WorkerEndpoint::new("w0", "127.0.0.1:1")];
        let pool = Arc::new(P2cPool::new(workers.clone()));
        // Pre-eject the worker so we can verify re-promotion.
        pool.record_health_failure("w0", 1);
        assert!(pool.choose_two_random().is_none());

        let toggle = Arc::new(ToggleTransport::new(true)); // healthy from start
        let transport: Arc<dyn EmbeddingTransport + Send + Sync> = toggle.clone();

        let cfg = HealthCheckerConfig {
            interval: Duration::from_millis(20),
            probe_timeout: Duration::from_millis(50),
            fail_threshold: 3,
            ewma_alpha: 0.1,
            expected_fingerprint: None,
            on_fingerprint_mismatch: None,
        };

        let checker = HealthChecker::spawn(rt.handle(), pool.clone(), workers, transport, cfg);
        rt.block_on(async { tokio::time::sleep(Duration::from_millis(120)).await });
        checker.stop();

        // After at least one successful probe, the ejected worker is back.
        assert!(toggle.probe_count() >= 2);
        assert!(
            pool.choose_two_random().is_some(),
            "worker should be re-promoted"
        );
    }
}
