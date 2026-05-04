//! Worker pool: power-of-two random selection with EWMA latency tracking.
//!
//! ADR-167 §8.3 — P2C balances load near-optimally for uniform-cost
//! tasks (embedding inference is the canonical example) and is much
//! cheaper than weighted least-loaded. Each worker carries an EWMA of
//! its observed latency; the picker chooses two at random and routes
//! the request to the lower-latency one.

use crate::transport::WorkerEndpoint;
use std::sync::Mutex;
use std::time::Duration;

/// Per-worker live stats. Kept inside `Mutex` because pick + update
/// happen from multiple threads. Embedding workloads are coarse enough
/// (>1 ms per embed) that lock contention isn't material.
#[derive(Debug)]
pub struct WorkerStats {
    /// The worker this stats record describes.
    pub endpoint: WorkerEndpoint,
    /// EWMA of observed latency in microseconds. Higher = slower.
    pub ewma_latency_us: f64,
    /// Consecutive failed health checks; ejected from the pick pool when
    /// this exceeds the configured threshold (3 by default).
    pub failed_health_checks: u32,
    /// Whether this worker is currently eligible for new requests.
    pub healthy: bool,
}

impl WorkerStats {
    /// Construct a fresh stats record for `endpoint`. EWMA starts at a
    /// neutral 1000µs so the first few picks are approximately random.
    pub fn new(endpoint: WorkerEndpoint) -> Self {
        Self {
            endpoint,
            // Seed EWMA at a "neutral" value so the first few picks are
            // approximately random (no pre-existing bias).
            ewma_latency_us: 1_000.0,
            failed_health_checks: 0,
            healthy: true,
        }
    }
}

/// Fixed pool of workers with P2C random selection.
pub struct P2cPool {
    inner: Mutex<Vec<WorkerStats>>,
}

impl P2cPool {
    /// Build a pool from a static worker list. All workers start
    /// `healthy=true` with a neutral EWMA seed.
    pub fn new(workers: Vec<WorkerEndpoint>) -> Self {
        let stats = workers.into_iter().map(WorkerStats::new).collect();
        Self {
            inner: Mutex::new(stats),
        }
    }

    /// Total worker count in the pool, regardless of healthy state.
    pub fn size(&self) -> usize {
        self.inner.lock().unwrap().len()
    }

    /// Snapshot of every worker's endpoint, healthy or not. Cheap clone.
    /// Used by `fleet_stats()` so operators can see ejected workers too.
    pub fn all_endpoints(&self) -> Vec<WorkerEndpoint> {
        self.inner
            .lock()
            .unwrap()
            .iter()
            .map(|w| w.endpoint.clone())
            .collect()
    }

    /// Pick two workers at random (without replacement) from the healthy
    /// pool, return the one with lower EWMA latency. If only one healthy
    /// worker exists, returns that. If none, returns None.
    pub fn choose_two_random(&self) -> Option<WorkerEndpoint> {
        let inner = self.inner.lock().unwrap();
        let healthy: Vec<&WorkerStats> = inner.iter().filter(|w| w.healthy).collect();
        match healthy.len() {
            0 => None,
            1 => Some(healthy[0].endpoint.clone()),
            n => {
                let i = pseudo_rand_index(n);
                let mut j = pseudo_rand_index(n);
                while j == i {
                    j = pseudo_rand_index(n);
                }
                let a = healthy[i];
                let b = healthy[j];
                let chosen = if a.ewma_latency_us <= b.ewma_latency_us {
                    a
                } else {
                    b
                };
                Some(chosen.endpoint.clone())
            }
        }
    }

    /// Update the EWMA for a worker after observing its latency.
    /// `alpha` controls memory: 0.1 = slow / smoother, 0.5 = fast / reactive.
    pub fn record_latency(&self, name: &str, observed: Duration, alpha: f64) {
        let observed_us = observed.as_micros() as f64;
        let mut inner = self.inner.lock().unwrap();
        if let Some(w) = inner.iter_mut().find(|w| w.endpoint.name == name) {
            w.ewma_latency_us = (1.0 - alpha) * w.ewma_latency_us + alpha * observed_us;
            w.failed_health_checks = 0;
        }
    }

    /// Increment failed-health counter; eject if threshold reached.
    pub fn record_health_failure(&self, name: &str, threshold: u32) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(w) = inner.iter_mut().find(|w| w.endpoint.name == name) {
            w.failed_health_checks += 1;
            if w.failed_health_checks >= threshold {
                w.healthy = false;
            }
        }
    }

    /// Mark a worker healthy again (used by the health-check loop after a
    /// successful probe of a previously-ejected worker).
    pub fn mark_healthy(&self, name: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(w) = inner.iter_mut().find(|w| w.endpoint.name == name) {
            w.healthy = true;
            w.failed_health_checks = 0;
        }
    }

    /// Force-eject a worker regardless of failure-threshold. Used by the
    /// startup `validate_fleet` check when a worker's fingerprint
    /// doesn't match the coordinator's expected model — that's a hard
    /// disqualification, not a transient blip.
    pub fn eject(&self, name: &str) {
        let mut inner = self.inner.lock().unwrap();
        if let Some(w) = inner.iter_mut().find(|w| w.endpoint.name == name) {
            w.healthy = false;
        }
    }
}

/// Cheap pseudo-random index for P2C. Uses a thread-local LCG seeded from
/// the system clock — good enough for load balancing (we don't need
/// crypto-grade randomness here). Avoids pulling `rand` as a dep just
/// for this.
fn pseudo_rand_index(n: usize) -> usize {
    use std::cell::Cell;
    thread_local! {
        static STATE: Cell<u64> = const { Cell::new(0x9E3779B97F4A7C15) };
    }
    STATE.with(|s| {
        let mut x = s.get();
        if x == 0 {
            // First call on this thread — seed from clock.
            x = std::time::SystemTime::now()
                .duration_since(std::time::UNIX_EPOCH)
                .map(|d| d.as_nanos() as u64)
                .unwrap_or(1);
            if x == 0 {
                x = 1;
            }
        }
        // xorshift64*
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        s.set(x);
        (x as usize) % n
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn endpoints(n: usize) -> Vec<WorkerEndpoint> {
        (0..n)
            .map(|i| WorkerEndpoint::new(format!("pi-{}", i), format!("10.0.0.{}:50051", i)))
            .collect()
    }

    #[test]
    fn pool_with_zero_healthy_returns_none() {
        let p = P2cPool::new(endpoints(2));
        // Eject both.
        p.record_health_failure("pi-0", 1);
        p.record_health_failure("pi-1", 1);
        assert!(p.choose_two_random().is_none());
    }

    #[test]
    fn single_healthy_worker_picked_deterministically() {
        let p = P2cPool::new(endpoints(2));
        p.record_health_failure("pi-1", 1);
        let picked = p.choose_two_random().unwrap();
        assert_eq!(picked.name, "pi-0");
    }

    #[test]
    fn ewma_picker_prefers_lower_latency() {
        let p = P2cPool::new(endpoints(2));
        // Make pi-0 slow.
        for _ in 0..50 {
            p.record_latency("pi-0", Duration::from_millis(50), 0.5);
            p.record_latency("pi-1", Duration::from_micros(500), 0.5);
        }
        // With only 2 workers, P2C always picks both — winner is pi-1.
        let picked = p.choose_two_random().unwrap();
        assert_eq!(picked.name, "pi-1");
    }

    #[test]
    fn mark_healthy_restores_ejected_worker() {
        let p = P2cPool::new(endpoints(1));
        p.record_health_failure("pi-0", 1);
        assert!(p.choose_two_random().is_none());
        p.mark_healthy("pi-0");
        assert!(p.choose_two_random().is_some());
    }
}
