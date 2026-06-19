//! Recall computation and benchmark measurement utilities.

use crate::search::SearchResult;

/// Recall@k: fraction of true top-k neighbors found in the search results.
pub fn recall_at_k(result: &SearchResult, ground_truth: &[u32]) -> f32 {
    if ground_truth.is_empty() {
        return 1.0;
    }
    let found = result
        .neighbors
        .iter()
        .filter(|(id, _)| ground_truth.contains(id))
        .count();
    found as f32 / ground_truth.len() as f32
}

/// Latency statistics from a slice of nanosecond measurements.
pub struct LatencyStats {
    pub mean_ns: f64,
    pub p50_ns: f64,
    pub p95_ns: f64,
    pub min_ns: u64,
    pub max_ns: u64,
}

impl LatencyStats {
    pub fn compute(mut samples: Vec<u64>) -> Self {
        samples.sort_unstable();
        let n = samples.len();
        let mean_ns = samples.iter().map(|&x| x as f64).sum::<f64>() / n as f64;
        let p50_ns = samples[n / 2] as f64;
        let p95_ns = samples[(n * 95) / 100] as f64;
        LatencyStats {
            mean_ns,
            p50_ns,
            p95_ns,
            min_ns: samples[0],
            max_ns: samples[n - 1],
        }
    }

    pub fn mean_us(&self) -> f64 {
        self.mean_ns / 1_000.0
    }
    pub fn p50_us(&self) -> f64 {
        self.p50_ns / 1_000.0
    }
    pub fn p95_us(&self) -> f64 {
        self.p95_ns / 1_000.0
    }

    pub fn throughput_qps(&self) -> f64 {
        if self.mean_ns == 0.0 {
            0.0
        } else {
            1_000_000_000.0 / self.mean_ns
        }
    }
}

/// Memory estimate for the flat graph: nodes × M × 4 bytes (u32 per neighbor)
/// plus the vector store: nodes × dims × 4 bytes (f32 per component).
pub fn memory_estimate_bytes(n: usize, dims: usize, m: usize) -> usize {
    let vector_store = n * dims * std::mem::size_of::<f32>();
    let neighbor_store = n * m * std::mem::size_of::<u32>();
    vector_store + neighbor_store
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::search::SearchResult;

    fn make_result(ids: &[u32]) -> SearchResult {
        SearchResult {
            neighbors: ids.iter().map(|&id| (id, id as f32)).collect(),
            pops: ids.len(),
            expansions: ids.len(),
        }
    }

    #[test]
    fn perfect_recall() {
        let r = make_result(&[0, 1, 2, 3, 4]);
        let gt = vec![0u32, 1, 2, 3, 4];
        assert!((recall_at_k(&r, &gt) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn half_recall() {
        let r = make_result(&[0, 1, 2, 9, 8]);
        let gt = vec![0u32, 1, 2, 3, 4];
        assert!((recall_at_k(&r, &gt) - 0.6).abs() < 1e-6);
    }

    #[test]
    fn zero_recall() {
        let r = make_result(&[10, 11, 12, 13, 14]);
        let gt = vec![0u32, 1, 2, 3, 4];
        assert!(recall_at_k(&r, &gt) < 1e-6);
    }

    #[test]
    fn latency_stats_percentiles() {
        // 100 samples: 1_000, 2_000, …, 100_000
        let samples: Vec<u64> = (1u64..=100).map(|x| x * 1_000).collect();
        let stats = LatencyStats::compute(samples);
        // mean = 50_500
        assert!((stats.mean_ns - 50_500.0).abs() < 1.0);
        // p50 index = n/2 = 50 → samples[50] = 51_000 (0-indexed, sorted)
        assert_eq!(stats.p50_ns, 51_000.0);
        // p95 index = (100*95)/100 = 95 → samples[95] = 96_000
        assert_eq!(stats.p95_ns, 96_000.0);
    }
}
