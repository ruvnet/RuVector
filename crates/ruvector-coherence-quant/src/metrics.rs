//! Recall and latency measurement helpers shared by the benchmark example
//! and the integration tests, so both compute the same numbers the same way.

use std::collections::HashSet;

/// Recall@k: fraction of `truth`'s top-k that also appear in `approx`'s
/// top-k (both already truncated to length k by the caller).
pub fn recall_at_k(approx: &[usize], truth: &[usize]) -> f32 {
    let truth_set: HashSet<usize> = truth.iter().copied().collect();
    let hits = approx.iter().filter(|i| truth_set.contains(i)).count();
    hits as f32 / truth.len().max(1) as f32
}

/// `p`-th percentile (0.0..=1.0) of a slice of nanosecond durations. Sorts
/// a copy; callers measuring many queries should batch this once.
pub fn percentile_ns(durations_ns: &[u64], p: f64) -> u64 {
    if durations_ns.is_empty() {
        return 0;
    }
    let mut sorted = durations_ns.to_vec();
    sorted.sort_unstable();
    let idx = ((sorted.len() as f64 - 1.0) * p).round() as usize;
    sorted[idx.min(sorted.len() - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn recall_counts_overlap() {
        assert_eq!(recall_at_k(&[1, 2, 3], &[1, 2, 3]), 1.0);
        assert_eq!(recall_at_k(&[1, 2, 9], &[1, 2, 3]), 2.0 / 3.0);
        assert_eq!(recall_at_k(&[9, 8, 7], &[1, 2, 3]), 0.0);
    }

    #[test]
    fn percentile_matches_known_distribution() {
        let d: Vec<u64> = (1..=100).collect();
        assert_eq!(percentile_ns(&d, 0.0), 1);
        assert_eq!(percentile_ns(&d, 0.99), 99);
        assert_eq!(percentile_ns(&d, 0.5), 51);
    }
}
