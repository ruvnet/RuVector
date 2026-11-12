//! Latency and throughput measurement utilities.

use std::time::{Duration, Instant};

/// Collect per-query latencies and compute statistics.
pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub throughput_qps: f64,
}

impl LatencyStats {
    pub fn measure<F, R>(n_queries: usize, mut f: F) -> (Vec<R>, Self)
    where
        F: FnMut(usize) -> R,
    {
        let mut latencies: Vec<u64> = Vec::with_capacity(n_queries);
        let mut results: Vec<R> = Vec::with_capacity(n_queries);

        let wall_start = Instant::now();
        for i in 0..n_queries {
            let t0 = Instant::now();
            let r = f(i);
            latencies.push(t0.elapsed().as_nanos() as u64);
            results.push(r);
        }
        let wall: Duration = wall_start.elapsed();

        latencies.sort_unstable();
        let n = latencies.len() as f64;
        let mean_ns = latencies.iter().sum::<u64>() as f64 / n;
        let p50_ns = latencies[(latencies.len() as f64 * 0.50) as usize] as f64;
        let p95_ns = latencies[(latencies.len() as f64 * 0.95) as usize] as f64;

        let stats = LatencyStats {
            mean_us: mean_ns / 1_000.0,
            p50_us: p50_ns / 1_000.0,
            p95_us: p95_ns / 1_000.0,
            throughput_qps: n_queries as f64 / wall.as_secs_f64(),
        };
        (results, stats)
    }
}

/// Track distance computation count per query (call count proxy for beam work).
#[derive(Default, Clone)]
pub struct WorkCounter {
    pub total: u64,
    pub queries: u64,
}

impl WorkCounter {
    pub fn record(&mut self, dist_computations: u64) {
        self.total += dist_computations;
        self.queries += 1;
    }

    pub fn mean(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.total as f64 / self.queries as f64
        }
    }
}
