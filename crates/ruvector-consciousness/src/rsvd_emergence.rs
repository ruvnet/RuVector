//! Randomized SVD-based causal emergence.
//!
//! Implements Zhang et al. (2025) "Dynamical reversibility and causal emergence
//! based on SVD" — computes causal emergence via the singular value spectrum
//! of the transition probability matrix.
//!
//! Key insight: the SVD of a TPM encodes its causal structure. Systems with
//! high causal emergence have a "sharp" singular value spectrum (few dominant
//! singular values), indicating effective coarse-graining.
//!
//! Uses the Halko-Martinsson-Tropp randomized SVD algorithm for O(nnz·k)
//! complexity instead of O(n³) for full SVD.
//!
//! # References
//!
//! - Zhang, J., et al. (2025). "Dynamical reversibility and causal emergence
//!   based on SVD." npj Complexity.
//! - Halko, N., Martinsson, P.-G., Tropp, J. (2011). "Finding structure with
//!   randomness: Probabilistic algorithms for constructing approximate matrix
//!   decompositions." SIAM Review.

use crate::error::{ConsciousnessError, ValidationError};
use crate::simd::entropy;
use crate::types::{ComputeBudget, TransitionMatrix};

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use std::time::Instant;

// ---------------------------------------------------------------------------
// Randomized SVD
// ---------------------------------------------------------------------------

/// Compute the top-k singular values of a matrix via randomized SVD.
///
/// Algorithm (Halko-Martinsson-Tropp):
/// 1. Draw random Gaussian matrix Ω (n × (k+p))
/// 2. Form Y = A * Ω (range sketch)
/// 3. QR decompose Y = Q * R
/// 4. Form B = Q^T * A (small (k+p) × n matrix)
/// 5. SVD of B gives approximate singular values
///
/// Complexity: O(n²·(k+p)) vs O(n³) for full SVD.
pub fn randomized_svd(tpm: &TransitionMatrix, k: usize, oversampling: usize, seed: u64) -> Vec<f64> {
    let n = tpm.n;
    let rank = (k + oversampling).min(n);
    let mut rng = StdRng::seed_from_u64(seed);

    // Step 1: Random Gaussian matrix Ω (n × rank).
    let mut omega = vec![0.0f64; n * rank];
    for val in &mut omega {
        // Box-Muller for approximate Gaussian.
        let u1: f64 = rng.gen::<f64>().max(1e-15);
        let u2: f64 = rng.gen::<f64>();
        *val = (-2.0 * u1.ln()).sqrt() * (2.0 * std::f64::consts::PI * u2).cos();
    }

    // Step 2: Y = A * Ω (n × rank).
    let mut y = vec![0.0f64; n * rank];
    for i in 0..n {
        for j in 0..rank {
            let mut sum = 0.0;
            for l in 0..n {
                sum += tpm.get(i, l) * omega[l * rank + j];
            }
            y[i * rank + j] = sum;
        }
    }

    // Step 3: QR decomposition via modified Gram-Schmidt.
    let mut q = y.clone();
    let mut r = vec![0.0f64; rank * rank];

    for j in 0..rank {
        // Normalize column j.
        let mut norm = 0.0;
        for i in 0..n {
            norm += q[i * rank + j] * q[i * rank + j];
        }
        norm = norm.sqrt();
        r[j * rank + j] = norm;

        if norm > 1e-15 {
            let inv_norm = 1.0 / norm;
            for i in 0..n {
                q[i * rank + j] *= inv_norm;
            }
        }

        // Orthogonalize subsequent columns.
        for jj in (j + 1)..rank {
            let mut dot = 0.0;
            for i in 0..n {
                dot += q[i * rank + j] * q[i * rank + jj];
            }
            r[j * rank + jj] = dot;
            for i in 0..n {
                q[i * rank + jj] -= dot * q[i * rank + j];
            }
        }
    }

    // Step 4: B = Q^T * A (rank × n).
    let mut b = vec![0.0f64; rank * n];
    for i in 0..rank {
        for j in 0..n {
            let mut sum = 0.0;
            for l in 0..n {
                sum += q[l * rank + i] * tpm.get(l, j);
            }
            b[i * n + j] = sum;
        }
    }

    // Step 5: Compute singular values of B via power iteration on B*B^T.
    // B*B^T is rank × rank, small enough for direct computation.
    let mut bbt = vec![0.0f64; rank * rank];
    for i in 0..rank {
        for j in 0..rank {
            let mut sum = 0.0;
            for l in 0..n {
                sum += b[i * n + l] * b[j * n + l];
            }
            bbt[i * rank + j] = sum;
        }
    }

    // Extract eigenvalues of B*B^T via power iteration with deflation.
    let mut eigenvalues = Vec::with_capacity(k);
    let mut matrix = bbt;

    for _ in 0..k {
        let ev = largest_eigenvalue_power(&matrix, rank, 200, &mut rng);
        eigenvalues.push(ev.sqrt().max(0.0)); // singular value = sqrt(eigenvalue)

        // Deflate: remove this eigenvalue's contribution.
        // v = dominant eigenvector (we just need the eigenvalue here).
        let v = dominant_eigenvector(&matrix, rank, 200, &mut rng);
        for i in 0..rank {
            for j in 0..rank {
                matrix[i * rank + j] -= ev * v[i] * v[j];
            }
        }
    }

    eigenvalues.sort_by(|a, b| b.partial_cmp(a).unwrap());
    eigenvalues
}

/// Power iteration to find the largest eigenvalue.
fn largest_eigenvalue_power(matrix: &[f64], n: usize, max_iter: usize, rng: &mut StdRng) -> f64 {
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();

    // Normalize.
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for vi in &mut v {
            *vi /= norm;
        }
    }

    let mut eigenvalue = 0.0;
    let mut w = vec![0.0f64; n];

    for _ in 0..max_iter {
        // w = M * v
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += matrix[i * n + j] * v[j];
            }
            w[i] = sum;
        }

        // Rayleigh quotient.
        let mut dot = 0.0;
        for i in 0..n {
            dot += v[i] * w[i];
        }
        eigenvalue = dot;

        // Normalize w.
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        let inv_norm = 1.0 / norm;
        for i in 0..n {
            v[i] = w[i] * inv_norm;
        }
    }

    eigenvalue.max(0.0)
}

/// Dominant eigenvector via power iteration.
fn dominant_eigenvector(matrix: &[f64], n: usize, max_iter: usize, rng: &mut StdRng) -> Vec<f64> {
    let mut v: Vec<f64> = (0..n).map(|_| rng.gen::<f64>() - 0.5).collect();
    let norm: f64 = v.iter().map(|x| x * x).sum::<f64>().sqrt();
    if norm > 1e-15 {
        for vi in &mut v {
            *vi /= norm;
        }
    }

    let mut w = vec![0.0f64; n];
    for _ in 0..max_iter {
        for i in 0..n {
            let mut sum = 0.0;
            for j in 0..n {
                sum += matrix[i * n + j] * v[j];
            }
            w[i] = sum;
        }
        let norm: f64 = w.iter().map(|x| x * x).sum::<f64>().sqrt();
        if norm < 1e-15 {
            break;
        }
        let inv_norm = 1.0 / norm;
        for i in 0..n {
            v[i] = w[i] * inv_norm;
        }
    }
    v
}

// ---------------------------------------------------------------------------
// SVD-based Causal Emergence
// ---------------------------------------------------------------------------

/// Compute causal emergence via the singular value spectrum.
///
/// The SVD-based emergence metric measures how "compressible" the TPM is:
/// - **Effective rank**: number of significant singular values
/// - **Spectral entropy**: entropy of normalized singular value distribution
/// - **Dynamical reversibility**: ratio of forward to backward information flow
///
/// A system with high causal emergence has a low effective rank (few
/// macro-level degrees of freedom capture most of the causal structure).
pub struct RsvdEmergenceEngine {
    /// Number of top singular values to compute.
    pub k: usize,
    /// Oversampling parameter for randomized SVD.
    pub oversampling: usize,
    /// Random seed.
    pub seed: u64,
}

impl RsvdEmergenceEngine {
    pub fn new(k: usize, oversampling: usize, seed: u64) -> Self {
        Self { k, oversampling, seed }
    }
}

impl Default for RsvdEmergenceEngine {
    fn default() -> Self {
        Self {
            k: 10,
            oversampling: 5,
            seed: 42,
        }
    }
}

impl RsvdEmergenceEngine {
    /// Compute SVD-based emergence metrics.
    pub fn compute(
        &self,
        tpm: &TransitionMatrix,
        _budget: &ComputeBudget,
    ) -> Result<RsvdEmergenceResult, ConsciousnessError> {
        let start = Instant::now();
        let n = tpm.n;

        if n < 2 {
            return Err(ValidationError::EmptySystem.into());
        }

        let k = self.k.min(n);
        let singular_values = randomized_svd(tpm, k, self.oversampling, self.seed);

        // Effective rank: count singular values above threshold.
        let max_sv = singular_values.first().copied().unwrap_or(0.0);
        let threshold = max_sv * 1e-6;
        let effective_rank = singular_values.iter().filter(|&&s| s > threshold).count();

        // Spectral entropy: entropy of the normalized singular value distribution.
        let sv_sum: f64 = singular_values.iter().sum();
        let spectral_entropy = if sv_sum > 1e-15 {
            let normalized: Vec<f64> = singular_values.iter().map(|&s| s / sv_sum).collect();
            entropy(&normalized)
        } else {
            0.0
        };

        // Maximum possible spectral entropy for k singular values.
        let max_spectral_entropy = (k as f64).ln();

        // Causal emergence proxy: 1 - (spectral_entropy / max_entropy).
        // Low spectral entropy = high compressibility = high emergence.
        let emergence_index = if max_spectral_entropy > 1e-15 {
            1.0 - (spectral_entropy / max_spectral_entropy)
        } else {
            0.0
        };

        // Dynamical reversibility via singular value ratio.
        let reversibility = if singular_values.len() >= 2 && max_sv > 1e-15 {
            singular_values.last().copied().unwrap_or(0.0) / max_sv
        } else {
            0.0
        };

        Ok(RsvdEmergenceResult {
            singular_values,
            effective_rank,
            spectral_entropy,
            emergence_index,
            reversibility,
            elapsed: start.elapsed(),
        })
    }
}

/// Result of SVD-based causal emergence analysis.
#[derive(Debug, Clone)]
pub struct RsvdEmergenceResult {
    /// Top-k singular values (descending).
    pub singular_values: Vec<f64>,
    /// Number of significant singular values.
    pub effective_rank: usize,
    /// Entropy of the normalized singular value distribution.
    pub spectral_entropy: f64,
    /// Emergence index: 1 - spectral_entropy/max_entropy (0 = no emergence, 1 = max).
    pub emergence_index: f64,
    /// Dynamical reversibility: min_sv / max_sv.
    pub reversibility: f64,
    /// Computation time.
    pub elapsed: std::time::Duration,
}

#[cfg(test)]
mod tests {
    use super::*;

    fn identity_tpm(n: usize) -> TransitionMatrix {
        TransitionMatrix::identity(n)
    }

    fn uniform_tpm(n: usize) -> TransitionMatrix {
        let val = 1.0 / n as f64;
        TransitionMatrix::new(n, vec![val; n * n])
    }

    #[test]
    fn rsvd_identity_singular_values() {
        let tpm = identity_tpm(4);
        let svs = randomized_svd(&tpm, 4, 2, 42);
        // Identity matrix has all singular values = 1.
        for sv in &svs {
            assert!((*sv - 1.0).abs() < 0.1, "identity sv should be ≈ 1, got {sv}");
        }
    }

    #[test]
    fn rsvd_uniform_low_rank() {
        let tpm = uniform_tpm(4);
        let svs = randomized_svd(&tpm, 4, 2, 42);
        // Uniform matrix has rank 1: one sv ≈ 1, rest ≈ 0.
        assert!(svs[0] > 0.1, "first sv should be significant");
        for sv in &svs[1..] {
            assert!(*sv < 0.2, "remaining svs should be small, got {sv}");
        }
    }

    #[test]
    fn rsvd_emergence_identity() {
        let tpm = identity_tpm(4);
        let engine = RsvdEmergenceEngine::default();
        let budget = ComputeBudget::fast();
        let result = engine.compute(&tpm, &budget).unwrap();
        // Identity: all singular values equal → high spectral entropy → low emergence.
        assert!(result.emergence_index < 0.5, "identity should have low emergence, got {}", result.emergence_index);
    }

    #[test]
    fn rsvd_emergence_uniform() {
        let tpm = uniform_tpm(4);
        let engine = RsvdEmergenceEngine::default();
        let budget = ComputeBudget::fast();
        let result = engine.compute(&tpm, &budget).unwrap();
        // Uniform: rank 1 → low spectral entropy → high emergence index.
        assert!(result.effective_rank <= 2, "uniform should have low effective rank, got {}", result.effective_rank);
    }

    #[test]
    fn rsvd_reversibility_bounded() {
        let tpm = identity_tpm(8);
        let engine = RsvdEmergenceEngine::new(5, 3, 42);
        let budget = ComputeBudget::fast();
        let result = engine.compute(&tpm, &budget).unwrap();
        assert!(result.reversibility >= 0.0 && result.reversibility <= 1.0);
    }
}
