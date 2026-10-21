//! Latency percentile accounting for the benchmark binary.

#[derive(Debug, Clone, Copy)]
pub struct LatencyStats {
    pub mean_us: f64,
    pub p50_us: f64,
    pub p95_us: f64,
    pub p99_us: f64,
}

impl LatencyStats {
    /// `samples_us` need not be pre-sorted; this function sorts a copy.
    pub fn from_samples(samples_us: &[f64]) -> Self {
        assert!(!samples_us.is_empty(), "no latency samples");
        let mut sorted = samples_us.to_vec();
        sorted.sort_unstable_by(|a, b| a.partial_cmp(b).unwrap());
        let n = sorted.len();
        let pct = |p: f64| -> f64 {
            let idx = ((p * n as f64).ceil() as usize)
                .saturating_sub(1)
                .min(n - 1);
            sorted[idx]
        };
        let mean = sorted.iter().sum::<f64>() / n as f64;
        LatencyStats {
            mean_us: mean,
            p50_us: pct(0.50),
            p95_us: pct(0.95),
            p99_us: pct(0.99),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn percentiles_of_known_distribution() {
        let samples: Vec<f64> = (1..=100).map(|x| x as f64).collect();
        let stats = LatencyStats::from_samples(&samples);
        assert_eq!(stats.p50_us, 50.0);
        assert_eq!(stats.p95_us, 95.0);
        assert_eq!(stats.p99_us, 99.0);
        assert!((stats.mean_us - 50.5).abs() < 1e-9);
    }
}
