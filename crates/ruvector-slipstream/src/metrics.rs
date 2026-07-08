//! Measurement utilities for recall, latency, and memory.

/// Recall@k: fraction of ground-truth neighbours found in results.
pub fn recall_at_k(results: &[usize], ground_truth: &[usize], k: usize) -> f32 {
    let gt: std::collections::HashSet<usize> = ground_truth.iter().take(k).copied().collect();
    let found = results.iter().take(k).filter(|id| gt.contains(id)).count();
    found as f32 / k.min(gt.len()) as f32
}

/// Latency statistics over a slice of nanosecond durations.
pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub throughput_qps: f64,
}

impl LatencyStats {
    pub fn compute(durations_ns: &mut Vec<u64>, total_elapsed_ns: u64) -> Self {
        durations_ns.sort_unstable();
        let n = durations_ns.len();
        let mean_us = durations_ns.iter().sum::<u64>() as f64 / (n as f64 * 1_000.0);
        let p50_us = durations_ns[n / 2] as f64 / 1_000.0;
        let p95_us = durations_ns[(n * 95) / 100] as f64 / 1_000.0;
        let throughput_qps = n as f64 / (total_elapsed_ns as f64 / 1e9);
        LatencyStats {
            mean_us,
            p50_us,
            p95_us,
            throughput_qps,
        }
    }
}

/// Rough memory estimate in bytes.
///
/// Accounts for: vector storage (N × D × 4 bytes) and adjacency lists
/// (N × M × 4 bytes for node IDs).
pub fn memory_estimate_bytes(n: usize, dims: usize, m: usize) -> usize {
    n * dims * 4 + n * m * 4
}

/// Summarise a set of per-query recall values.
pub fn mean_recall(recalls: &[f32]) -> f32 {
    if recalls.is_empty() {
        0.0
    } else {
        recalls.iter().sum::<f32>() / recalls.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_perfect() {
        let r = recall_at_k(&[0, 1, 2], &[0, 1, 2], 3);
        assert!((r - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_zero() {
        let r = recall_at_k(&[3, 4, 5], &[0, 1, 2], 3);
        assert_eq!(r, 0.0);
    }

    #[test]
    fn recall_half() {
        let r = recall_at_k(&[0, 3], &[0, 1], 2);
        assert!((r - 0.5).abs() < 1e-6);
    }

    #[test]
    fn latency_stats_basic() {
        let mut durations = vec![1_000u64, 2_000, 3_000, 4_000, 5_000];
        let stats = LatencyStats::compute(&mut durations, 15_000);
        assert!((stats.mean_us - 3.0).abs() < 0.1);
    }
}
