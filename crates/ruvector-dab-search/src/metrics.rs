//! Latency, throughput, and search-work measurement utilities.

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
        let p95_ns =
            latencies[((latencies.len() as f64 * 0.95) as usize).min(latencies.len() - 1)] as f64;

        let stats = LatencyStats {
            mean_us: mean_ns / 1_000.0,
            p50_us: p50_ns / 1_000.0,
            p95_us: p95_ns / 1_000.0,
            throughput_qps: n_queries as f64 / wall.as_secs_f64(),
        };
        (results, stats)
    }
}

/// Per-query "work" sample set (distance computations, expansions, ...).
///
/// Used both to report mean/spread and, critically, to test whether a
/// variant's work actually *varies* across queries of different difficulty —
/// the property that ADR-303 (entropy-adaptive beam search) measured as
/// absent (heap-distance entropy saturated to the same value for every
/// query, so `EntropyScaledEf`'s ef_actual was constant).
#[derive(Default, Clone)]
pub struct WorkStats {
    samples: Vec<u64>,
}

impl WorkStats {
    pub fn new() -> Self {
        Self {
            samples: Vec::new(),
        }
    }

    pub fn record(&mut self, v: u64) {
        self.samples.push(v);
    }

    pub fn mean(&self) -> f64 {
        if self.samples.is_empty() {
            0.0
        } else {
            self.samples.iter().sum::<u64>() as f64 / self.samples.len() as f64
        }
    }

    pub fn min(&self) -> u64 {
        self.samples.iter().copied().min().unwrap_or(0)
    }

    pub fn max(&self) -> u64 {
        self.samples.iter().copied().max().unwrap_or(0)
    }

    pub fn stddev(&self) -> f64 {
        let m = self.mean();
        if self.samples.len() < 2 {
            return 0.0;
        }
        let var = self
            .samples
            .iter()
            .map(|&x| {
                let d = x as f64 - m;
                d * d
            })
            .sum::<f64>()
            / (self.samples.len() as f64 - 1.0);
        var.sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn work_stats_mean_and_spread() {
        let mut w = WorkStats::new();
        for v in [10u64, 20, 30, 40] {
            w.record(v);
        }
        assert!((w.mean() - 25.0).abs() < 1e-9);
        assert_eq!(w.min(), 10);
        assert_eq!(w.max(), 40);
        assert!(w.stddev() > 0.0);
    }

    #[test]
    fn work_stats_constant_has_zero_stddev() {
        let mut w = WorkStats::new();
        for _ in 0..5 {
            w.record(124);
        }
        assert_eq!(w.stddev(), 0.0);
    }
}
