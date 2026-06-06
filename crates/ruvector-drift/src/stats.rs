//! Online streaming statistics for D-dimensional vectors.

/// Welford-style online running statistics for a D-dimensional vector stream.
///
/// Tracks per-dimension mean and variance without storing all samples.
/// Used by [`MeanShiftDetector`] and [`CusumDetector`].
#[derive(Clone, Debug)]
pub struct OnlineStats {
    /// Dimension of tracked vectors.
    pub dim: usize,
    /// Per-dimension running mean.
    pub mean: Vec<f64>,
    /// Per-dimension running M2 (for variance: M2 / n).
    pub m2: Vec<f64>,
    /// Number of samples ingested.
    pub n: usize,
}

impl OnlineStats {
    /// Create a new tracker for `dim`-dimensional vectors.
    pub fn new(dim: usize) -> Self {
        Self {
            dim,
            mean: vec![0.0; dim],
            m2: vec![0.0; dim],
            n: 0,
        }
    }

    /// Ingest one sample, updating running mean and M2.
    pub fn push(&mut self, v: &[f32]) {
        debug_assert_eq!(v.len(), self.dim);
        self.n += 1;
        let n = self.n as f64;
        for (i, &x) in v.iter().enumerate() {
            let x = x as f64;
            let delta = x - self.mean[i];
            self.mean[i] += delta / n;
            let delta2 = x - self.mean[i];
            self.m2[i] += delta * delta2;
        }
    }

    /// Per-dimension variance (sample variance, n-1 denominator).
    pub fn variance(&self) -> Vec<f64> {
        if self.n < 2 {
            return vec![0.0; self.dim];
        }
        let denom = (self.n - 1) as f64;
        self.m2.iter().map(|m| m / denom).collect()
    }

    /// L2 distance between this mean and another mean vector.
    pub fn mean_l2_to(&self, other: &[f64]) -> f64 {
        debug_assert_eq!(other.len(), self.dim);
        self.mean
            .iter()
            .zip(other.iter())
            .map(|(a, b)| (a - b).powi(2))
            .sum::<f64>()
            .sqrt()
    }

    /// Approximate heap bytes used.
    pub fn memory_bytes(&self) -> usize {
        2 * self.dim * std::mem::size_of::<f64>()
    }
}
