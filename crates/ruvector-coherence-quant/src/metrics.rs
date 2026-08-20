//! Latency measurement helpers.

use std::time::Instant;

pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub throughput_qps: f64,
}

impl LatencyStats {
    /// Runs `f(i)` for `i in 0..n`, timing each call individually.
    pub fn measure<T>(n: usize, mut f: impl FnMut(usize) -> T) -> (Vec<T>, LatencyStats) {
        let mut results = Vec::with_capacity(n);
        let mut times_us = Vec::with_capacity(n);
        let total_start = Instant::now();
        for i in 0..n {
            let t0 = Instant::now();
            results.push(f(i));
            times_us.push(t0.elapsed().as_secs_f64() * 1e6);
        }
        let total_elapsed = total_start.elapsed().as_secs_f64().max(1e-9);

        let mut sorted = times_us.clone();
        sorted.sort_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal));
        let n_f = n.max(1) as f64;
        let mean = times_us.iter().sum::<f64>() / n_f;
        let p50_idx = (n.saturating_sub(1)) / 2;
        let p95_idx = (((n as f64) * 0.95) as usize).min(n.saturating_sub(1));
        let p50 = sorted.get(p50_idx).copied().unwrap_or(0.0);
        let p95 = sorted.get(p95_idx).copied().unwrap_or(0.0);

        (
            results,
            LatencyStats {
                mean_us: mean,
                p50_us: p50,
                p95_us: p95,
                throughput_qps: n_f / total_elapsed,
            },
        )
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn measure_returns_n_results() {
        let (results, stats) = LatencyStats::measure(10, |i| i * 2);
        assert_eq!(results.len(), 10);
        assert!(stats.mean_us >= 0.0);
        assert!(stats.throughput_qps > 0.0);
    }
}
