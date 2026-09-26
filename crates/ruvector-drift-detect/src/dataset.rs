//! Deterministic synthetic dataset generation for benchmarks and tests.

use rand::{rngs::SmallRng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Generate `n` random vectors of `dims` dimensions sampled from N(`mean`, `std`).
pub fn sample_normal(n: usize, dims: usize, mean: f32, std: f32, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let dist = Normal::new(mean, std).expect("invalid normal parameters");
    (0..n)
        .map(|_| (0..dims).map(|_| dist.sample(&mut rng)).collect())
        .collect()
}

/// Generate `n` vectors where the first `drifted_dims` dimensions are sampled
/// from N(`drift_mean`, `drift_std`) and remaining dimensions from N(0, 1).
///
/// This simulates a partial distribution shift — common in real embedding drift
/// where only a subset of semantic dimensions change after a model update.
pub fn sample_partial_drift(
    n: usize,
    dims: usize,
    drifted_dims: usize,
    drift_mean: f32,
    drift_std: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let base_dist = Normal::new(0.0f32, 1.0).unwrap();
    let drift_dist = Normal::new(drift_mean, drift_std).unwrap();

    (0..n)
        .map(|_| {
            (0..dims)
                .map(|d| {
                    if d < drifted_dims {
                        drift_dist.sample(&mut rng)
                    } else {
                        base_dist.sample(&mut rng)
                    }
                })
                .collect()
        })
        .collect()
}

/// Interleave baseline and drifted vectors to simulate a gradual drift scenario.
pub fn sample_gradual_drift(
    n: usize,
    dims: usize,
    drift_start_pct: f64,
    drift_mean: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed);
    let base_dist = Normal::new(0.0f32, 1.0).unwrap();
    let drift_dist = Normal::new(drift_mean, 1.0).unwrap();
    let drift_start = (n as f64 * drift_start_pct) as usize;

    (0..n)
        .map(|i| {
            let progress = if i < drift_start {
                0.0f32
            } else {
                (i - drift_start) as f32 / (n - drift_start).max(1) as f32
            };
            (0..dims)
                .map(|_| {
                    let base = base_dist.sample(&mut rng);
                    let drift = drift_dist.sample(&mut rng);
                    base * (1.0 - progress) + drift * progress
                })
                .collect()
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sample_normal_dimensions() {
        let vecs = sample_normal(100, 64, 0.0, 1.0, 42);
        assert_eq!(vecs.len(), 100);
        assert!(vecs.iter().all(|v| v.len() == 64));
    }

    #[test]
    fn partial_drift_first_dims_shifted() {
        let vecs = sample_partial_drift(1000, 128, 64, 5.0, 0.1, 7);
        let mean_first: f32 = vecs.iter().map(|v| v[0]).sum::<f32>() / 1000.0;
        let mean_last: f32 = vecs.iter().map(|v| v[127]).sum::<f32>() / 1000.0;
        assert!(
            mean_first > 3.0,
            "drifted dims should have mean ≈ 5, got {mean_first:.2}"
        );
        assert!(
            mean_last.abs() < 1.0,
            "baseline dims should have mean ≈ 0, got {mean_last:.2}"
        );
    }

    #[test]
    fn gradual_drift_monotone_mean() {
        let vecs = sample_gradual_drift(1000, 8, 0.5, 4.0, 99);
        let early_mean: f32 = vecs[..200].iter().map(|v| v[0]).sum::<f32>() / 200.0;
        let late_mean: f32 = vecs[800..].iter().map(|v| v[0]).sum::<f32>() / 200.0;
        assert!(
            late_mean > early_mean,
            "gradual drift: late mean {late_mean:.2} should exceed early {early_mean:.2}"
        );
    }
}
