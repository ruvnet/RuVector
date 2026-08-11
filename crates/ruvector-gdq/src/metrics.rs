use std::collections::HashSet;
use std::time::{Duration, Instant};

/// Compute recall@k: fraction of true neighbors returned.
pub fn recall_at_k(results: &[Vec<usize>], ground_truth: &[Vec<usize>], k: usize) -> f32 {
    assert_eq!(results.len(), ground_truth.len());
    let total: f32 = results.len() as f32;
    let hit_sum: f32 = results
        .iter()
        .zip(ground_truth.iter())
        .map(|(res, gt)| {
            let true_set: HashSet<usize> = gt[..k.min(gt.len())].iter().copied().collect();
            let found = res[..k.min(res.len())]
                .iter()
                .filter(|&&r| true_set.contains(&r))
                .count();
            found as f32 / k.min(gt.len()) as f32
        })
        .sum();
    hit_sum / total
}

/// Run a search and measure latency distribution.
pub struct LatencyReport {
    pub mean_us: f32,
    pub p50_us: f32,
    pub p95_us: f32,
    pub throughput_qps: f32,
    pub total_queries: usize,
}

pub fn measure_latency<F>(queries: usize, mut search_fn: F) -> LatencyReport
where
    F: FnMut(usize) -> Vec<usize>,
{
    let mut latencies: Vec<u64> = Vec::with_capacity(queries);
    let total_start = Instant::now();
    for i in 0..queries {
        let t0 = Instant::now();
        let _res = search_fn(i);
        let elapsed = t0.elapsed();
        latencies.push(elapsed.as_micros() as u64);
    }
    let total_elapsed: Duration = total_start.elapsed();

    latencies.sort_unstable();
    let mean_us = latencies.iter().sum::<u64>() as f32 / queries as f32;
    let p50_us = latencies[queries / 2] as f32;
    let p95_idx = ((queries as f32 * 0.95) as usize).min(queries - 1);
    let p95_us = latencies[p95_idx] as f32;
    let throughput_qps = queries as f32 / total_elapsed.as_secs_f32();

    LatencyReport {
        mean_us,
        p50_us,
        p95_us,
        throughput_qps,
        total_queries: queries,
    }
}

/// Estimated memory in bytes for a quantization regime.
pub fn estimated_memory_bytes(n_high: usize, n_low: usize, dim: usize) -> usize {
    let high_bytes = n_high * dim; // 8-bit: 1 byte per dim
    let low_bytes = n_low * ((dim + 1) / 2); // 4-bit: 0.5 bytes per dim
    high_bytes + low_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_recall_perfect() {
        let gt = vec![vec![0, 1, 2], vec![3, 4, 5]];
        let res = gt.clone();
        assert!((recall_at_k(&res, &gt, 3) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_recall_zero() {
        let gt = vec![vec![0, 1, 2]];
        let res = vec![vec![9, 8, 7]];
        assert!(recall_at_k(&res, &gt, 3) < 1e-6);
    }

    #[test]
    fn test_memory_estimate() {
        // 1000 high (8-bit, dim=128) + 4000 low (4-bit, dim=128)
        let mem = estimated_memory_bytes(1000, 4000, 128);
        assert_eq!(mem, 1000 * 128 + 4000 * 64);
        assert_eq!(mem, 128_000 + 256_000);
        assert_eq!(mem, 384_000);
    }
}
