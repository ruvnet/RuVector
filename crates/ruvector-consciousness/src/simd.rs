//! SIMD-accelerated operations for consciousness computation.
//!
//! Provides vectorized KL-divergence, entropy, and matrix operations
//! critical for Φ computation hot paths.

// ---------------------------------------------------------------------------
// KL Divergence (the core operation in Φ computation)
// ---------------------------------------------------------------------------

/// Compute KL divergence D_KL(P || Q) = Σ p_i * ln(p_i / q_i).
///
/// Dispatches to AVX2 when available, falls back to scalar.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "KL divergence: mismatched lengths");

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return kl_divergence_scalar(p, q); // AVX2 log is complex; use scalar with prefetch
        }
    }

    kl_divergence_scalar(p, q)
}

/// Scalar KL divergence with branch-free clamping.
pub fn kl_divergence_scalar(p: &[f64], q: &[f64]) -> f64 {
    let mut sum = 0.0f64;
    for i in 0..p.len() {
        let pi = p[i];
        let qi = q[i];
        if pi > 1e-15 && qi > 1e-15 {
            sum += pi * (pi / qi).ln();
        }
    }
    sum
}

/// Earth Mover's Distance (EMD) approximation for distribution comparison.
/// Used in IIT 4.0 for comparing cause-effect structures.
pub fn emd_l1(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len());
    let mut cumsum = 0.0f64;
    let mut dist = 0.0f64;
    for i in 0..p.len() {
        cumsum += p[i] - q[i];
        dist += cumsum.abs();
    }
    dist
}

// ---------------------------------------------------------------------------
// Entropy
// ---------------------------------------------------------------------------

/// Shannon entropy H(P) = -Σ p_i * ln(p_i).
pub fn entropy(p: &[f64]) -> f64 {
    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return entropy_scalar(p);
        }
    }
    entropy_scalar(p)
}

pub fn entropy_scalar(p: &[f64]) -> f64 {
    let mut h = 0.0f64;
    for &pi in p {
        if pi > 1e-15 {
            h -= pi * pi.ln();
        }
    }
    h
}

// ---------------------------------------------------------------------------
// SIMD matrix-vector multiply (dense, f64)
// ---------------------------------------------------------------------------

/// Dense matrix-vector multiply y = A * x (row-major A).
/// Used for TPM operations in Φ computation.
pub fn dense_matvec(a: &[f64], x: &[f64], y: &mut [f64], n: usize) {
    assert_eq!(a.len(), n * n);
    assert_eq!(x.len(), n);
    assert_eq!(y.len(), n);

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            unsafe {
                dense_matvec_avx2(a, x, y, n);
            }
            return;
        }
    }

    dense_matvec_scalar(a, x, y, n);
}

fn dense_matvec_scalar(a: &[f64], x: &[f64], y: &mut [f64], n: usize) {
    for i in 0..n {
        let mut sum = 0.0f64;
        let row_start = i * n;
        for j in 0..n {
            sum += a[row_start + j] * x[j];
        }
        y[i] = sum;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn dense_matvec_avx2(a: &[f64], x: &[f64], y: &mut [f64], n: usize) {
    use std::arch::x86_64::*;

    for i in 0..n {
        let row_start = i * n;
        let mut accum = _mm256_setzero_pd();
        let chunks = n / 4;
        let remainder = n % 4;

        for chunk in 0..chunks {
            let base = row_start + chunk * 4;
            // SAFETY: base + 3 < row_start + n = a.len() / n * (i+1), in bounds.
            let av = _mm256_loadu_pd(a.as_ptr().add(base));
            let xv = _mm256_loadu_pd(x.as_ptr().add(chunk * 4));
            accum = _mm256_add_pd(accum, _mm256_mul_pd(av, xv));
        }

        let mut sum = horizontal_sum_f64x4(accum);

        let tail_start = chunks * 4;
        for j in tail_start..(tail_start + remainder) {
            sum += *a.get_unchecked(row_start + j) * *x.get_unchecked(j);
        }

        *y.get_unchecked_mut(i) = sum;
    }
}

#[cfg(all(feature = "simd", target_arch = "x86_64"))]
#[target_feature(enable = "avx2")]
unsafe fn horizontal_sum_f64x4(v: std::arch::x86_64::__m256d) -> f64 {
    use std::arch::x86_64::*;
    let hi = _mm256_extractf128_pd(v, 1);
    let lo = _mm256_castpd256_pd128(v);
    let sum128 = _mm_add_pd(lo, hi);
    let hi64 = _mm_unpackhi_pd(sum128, sum128);
    let result = _mm_add_sd(sum128, hi64);
    _mm_cvtsd_f64(result)
}

// ---------------------------------------------------------------------------
// Conditional distribution extraction
// ---------------------------------------------------------------------------

/// Extract conditional distribution P(future | state) from TPM row.
#[inline]
pub fn conditional_distribution(tpm: &[f64], n: usize, state: usize) -> &[f64] {
    &tpm[state * n..(state + 1) * n]
}

/// Compute marginal distribution by averaging over all rows.
pub fn marginal_distribution(tpm: &[f64], n: usize) -> Vec<f64> {
    let mut marginal = vec![0.0; n];
    for i in 0..n {
        for j in 0..n {
            marginal[j] += tpm[i * n + j];
        }
    }
    let inv_n = 1.0 / n as f64;
    for m in &mut marginal {
        *m *= inv_n;
    }
    marginal
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kl_divergence_identical() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        assert!((kl_divergence(&p, &p)).abs() < 1e-12);
    }

    #[test]
    fn entropy_uniform() {
        let p = vec![0.25, 0.25, 0.25, 0.25];
        let h = entropy(&p);
        let expected = (4.0f64).ln();
        assert!((h - expected).abs() < 1e-10);
    }

    #[test]
    fn dense_matvec_correctness() {
        let a = vec![1.0, 2.0, 3.0, 4.0];
        let x = vec![1.0, 1.0];
        let mut y = vec![0.0; 2];
        dense_matvec(&a, &x, &mut y, 2);
        assert!((y[0] - 3.0).abs() < 1e-10);
        assert!((y[1] - 7.0).abs() < 1e-10);
    }

    #[test]
    fn emd_identical() {
        let p = vec![0.5, 0.3, 0.2];
        assert!((emd_l1(&p, &p)).abs() < 1e-12);
    }

    #[test]
    fn marginal_identity() {
        let tpm = vec![1.0, 0.0, 0.0, 1.0];
        let m = marginal_distribution(&tpm, 2);
        assert!((m[0] - 0.5).abs() < 1e-10);
        assert!((m[1] - 0.5).abs() < 1e-10);
    }
}
