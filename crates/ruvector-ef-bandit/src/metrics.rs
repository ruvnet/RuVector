//! Recall, latency percentile, and throughput computation.

/// Recall@k: fraction of true k-NN found in `results`.
///
/// `results` and `ground_truth` need not be the same length;
/// only the first `k` elements of each are considered.
pub fn recall_at_k(results: &[usize], ground_truth: &[usize], k: usize) -> f64 {
    let k = k.min(results.len()).min(ground_truth.len());
    if k == 0 {
        return 0.0;
    }
    let found = results[..k]
        .iter()
        .filter(|&&r| ground_truth[..k].contains(&r))
        .count();
    found as f64 / k as f64
}

/// Compute mean, p50, and p95 latencies from a slice of nanosecond measurements.
///
/// The slice is sorted in place (no allocation).
pub fn latency_stats_ns(latencies: &mut [u64]) -> (f64, u64, u64) {
    if latencies.is_empty() {
        return (0.0, 0, 0);
    }
    latencies.sort_unstable();
    let mean = latencies.iter().map(|&x| x as f64).sum::<f64>() / latencies.len() as f64;
    let p50 = latencies[latencies.len() / 2];
    let p95 = latencies[(latencies.len() * 95) / 100];
    (mean, p50, p95)
}

/// QPS from total query count and total wall-clock nanoseconds.
pub fn throughput_qps(n_queries: usize, total_ns: u64) -> f64 {
    if total_ns == 0 {
        return 0.0;
    }
    n_queries as f64 / (total_ns as f64 * 1e-9)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        let r = recall_at_k(&[0, 1, 2, 3, 4], &[0, 1, 2, 3, 4], 5);
        assert!((r - 1.0).abs() < 1e-9);
    }

    #[test]
    fn recall_zero() {
        let r = recall_at_k(&[5, 6, 7], &[0, 1, 2], 3);
        assert!((r - 0.0).abs() < 1e-9);
    }

    #[test]
    fn recall_partial() {
        let r = recall_at_k(&[0, 99, 2, 99, 4], &[0, 1, 2, 3, 4], 5);
        assert!((r - 0.6).abs() < 1e-9, "expected 3/5=0.6, got {r}");
    }

    #[test]
    fn latency_stats_sorted_correctly() {
        let mut v = vec![100u64, 10, 50, 200, 30];
        let (mean, p50, p95) = latency_stats_ns(&mut v);
        assert_eq!(p50, 50);
        assert_eq!(p95, 200);
        assert!((mean - 78.0).abs() < 1.0);
    }

    #[test]
    fn throughput_qps_is_reasonable() {
        let qps = throughput_qps(1000, 1_000_000_000);
        assert!((qps - 1000.0).abs() < 1.0);
    }
}
