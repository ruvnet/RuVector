//! Turning a sample of entry→query distances into a routing threshold.
//!
//! This module is deliberately data-only: it never runs a search and never
//! sees ground truth. The bounded search over *which* percentile to use
//! (the "Darwin-lite" generations in the benchmark binary) lives in the
//! benchmark, where recall and latency are measured on a calibration query
//! set that is disjoint from the evaluation set — this module just answers
//! "what distance sits at percentile P of this sample".

/// Returns the value at the given percentile (0.0..=100.0) of a distance
/// sample, using nearest-rank (no interpolation) for a deterministic result.
pub fn percentile_threshold(distances: &[f32], pct: f32) -> f32 {
    assert!(!distances.is_empty(), "distances must not be empty");
    assert!((0.0..=100.0).contains(&pct), "pct must be in [0, 100]");
    let mut sorted = distances.to_vec();
    sorted.sort_unstable_by(|a, b| a.total_cmp(b));
    let n = sorted.len();
    let rank = ((pct / 100.0) * (n as f32 - 1.0)).round() as usize;
    sorted[rank.min(n - 1)]
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn zero_percentile_is_minimum() {
        let d = [5.0, 1.0, 3.0, 9.0, 2.0];
        assert_eq!(percentile_threshold(&d, 0.0), 1.0);
    }

    #[test]
    fn hundred_percentile_is_maximum() {
        let d = [5.0, 1.0, 3.0, 9.0, 2.0];
        assert_eq!(percentile_threshold(&d, 100.0), 9.0);
    }

    #[test]
    fn fifty_percentile_is_median_for_odd_count() {
        // sorted: 1, 2, 3, 5, 9 -> median index 2 -> 3.0
        let d = [5.0, 1.0, 3.0, 9.0, 2.0];
        assert_eq!(percentile_threshold(&d, 50.0), 3.0);
    }

    #[test]
    fn single_element_returns_that_element_at_any_percentile() {
        let d = [42.0];
        assert_eq!(percentile_threshold(&d, 0.0), 42.0);
        assert_eq!(percentile_threshold(&d, 100.0), 42.0);
        assert_eq!(percentile_threshold(&d, 37.0), 42.0);
    }

    #[test]
    #[should_panic(expected = "distances must not be empty")]
    fn empty_slice_panics() {
        percentile_threshold(&[], 50.0);
    }

    #[test]
    #[should_panic(expected = "pct must be in")]
    fn out_of_range_percentile_panics() {
        percentile_threshold(&[1.0, 2.0], 150.0);
    }
}
