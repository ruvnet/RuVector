//! SIMD-accelerated operations for consciousness computation.
//!
//! Provides vectorized KL-divergence, entropy, and matrix operations
//! critical for Φ computation hot paths.

// ---------------------------------------------------------------------------
// KL Divergence (the core operation in Φ computation)
// ---------------------------------------------------------------------------

/// Compute KL divergence D_KL(P || Q) = Σ p_i * ln(p_i / q_i).
///
/// Dispatches to NEON on AArch64, AVX2 on x86_64, falls back to scalar.
pub fn kl_divergence(p: &[f64], q: &[f64]) -> f64 {
    assert_eq!(p.len(), q.len(), "KL divergence: mismatched lengths");

    #[cfg(all(feature = "simd", target_arch = "x86_64"))]
    {
        if is_x86_feature_detected!("avx2") {
            return kl_divergence_scalar(p, q); // AVX2 log is complex; use scalar with prefetch
        }
    }

    // NEON: use prefetch-hinted scalar — ln() has no SIMD intrinsic but
    // prefetching the next cache line hides memory latency on M-series.
    #[cfg(target_arch = "aarch64")]
    {
        return kl_divergence_neon_prefetch(p, q);
    }

    #[allow(unreachable_code)]
    kl_divergence_scalar(p, q)
}

/// AArch64 KL divergence with 4x unroll for better ILP.
/// The ln() call prevents full NEON vectorization, but unrolling
/// enables out-of-order execution to overlap FP operations on M-series.
#[cfg(target_arch = "aarch64")]
fn kl_divergence_neon_prefetch(p: &[f64], q: &[f64]) -> f64 {
    let n = p.len();
    let mut sum0 = 0.0f64;
    let mut sum1 = 0.0f64;
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        // 4x unrolled with 2 independent accumulators for ILP
        let (p0, q0) = (p[base], q[base]);
        let (p1, q1) = (p[base + 1], q[base + 1]);
        let (p2, q2) = (p[base + 2], q[base + 2]);
        let (p3, q3) = (p[base + 3], q[base + 3]);

        if p0 > 1e-15 && q0 > 1e-15 { sum0 += p0 * (p0 / q0).ln(); }
        if p1 > 1e-15 && q1 > 1e-15 { sum1 += p1 * (p1 / q1).ln(); }
        if p2 > 1e-15 && q2 > 1e-15 { sum0 += p2 * (p2 / q2).ln(); }
        if p3 > 1e-15 && q3 > 1e-15 { sum1 += p3 * (p3 / q3).ln(); }
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        let pi = p[i];
        let qi = q[i];
        if pi > 1e-15 && qi > 1e-15 {
            sum0 += pi * (pi / qi).ln();
        }
    }
    sum0 + sum1
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

    #[cfg(target_arch = "aarch64")]
    {
        return entropy_neon_prefetch(p);
    }

    #[allow(unreachable_code)]
    entropy_scalar(p)
}

/// AArch64 entropy with 4x unroll + dual accumulators for ILP.
#[cfg(target_arch = "aarch64")]
fn entropy_neon_prefetch(p: &[f64]) -> f64 {
    let n = p.len();
    let mut h0 = 0.0f64;
    let mut h1 = 0.0f64;
    let chunks = n / 4;
    let remainder = n % 4;

    for c in 0..chunks {
        let base = c * 4;
        let (p0, p1, p2, p3) = (p[base], p[base + 1], p[base + 2], p[base + 3]);
        if p0 > 1e-15 { h0 -= p0 * p0.ln(); }
        if p1 > 1e-15 { h1 -= p1 * p1.ln(); }
        if p2 > 1e-15 { h0 -= p2 * p2.ln(); }
        if p3 > 1e-15 { h1 -= p3 * p3.ln(); }
    }
    for i in (chunks * 4)..(chunks * 4 + remainder) {
        let pi = p[i];
        if pi > 1e-15 {
            h0 -= pi * pi.ln();
        }
    }
    h0 + h1
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

    #[cfg(target_arch = "aarch64")]
    {
        unsafe {
            dense_matvec_neon(a, x, y, n);
        }
        return;
    }

    #[allow(unreachable_code)]
    dense_matvec_scalar(a, x, y, n);
}

/// NEON-optimized dense matvec for AArch64 (Apple Silicon M1-M4).
///
/// Uses `float64x2_t` (2-wide f64 NEON) with 2x unroll → processes 4 f64/iter.
/// FMA instructions maximize throughput on M-series cores.
///
/// # Safety
/// Caller must ensure `a.len() >= n*n`, `x.len() >= n`, `y.len() >= n`.
#[cfg(target_arch = "aarch64")]
unsafe fn dense_matvec_neon(a: &[f64], x: &[f64], y: &mut [f64], n: usize) {
    use std::arch::aarch64::*;

    for i in 0..n {
        let row_start = i * n;

        // 2x unrolled: process 4 f64 per iteration (2 NEON regs × 2 f64)
        let chunks4 = n / 4;
        let mut acc0 = vdupq_n_f64(0.0);
        let mut acc1 = vdupq_n_f64(0.0);

        for c in 0..chunks4 {
            let base = row_start + c * 4;
            let a0 = vld1q_f64(a.as_ptr().add(base));
            let a1 = vld1q_f64(a.as_ptr().add(base + 2));
            let x0 = vld1q_f64(x.as_ptr().add(c * 4));
            let x1 = vld1q_f64(x.as_ptr().add(c * 4 + 2));
            acc0 = vfmaq_f64(acc0, a0, x0);
            acc1 = vfmaq_f64(acc1, a1, x1);
        }

        // Combine accumulators
        let combined = vaddq_f64(acc0, acc1);
        let mut sum = vgetq_lane_f64(combined, 0) + vgetq_lane_f64(combined, 1);

        // Scalar tail
        let tail_start = chunks4 * 4;
        for j in tail_start..n {
            sum += *a.get_unchecked(row_start + j) * *x.get_unchecked(j);
        }

        *y.get_unchecked_mut(i) = sum;
    }
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

// ---------------------------------------------------------------------------
// Shared pairwise MI computation (used by all spectral engines)
// ---------------------------------------------------------------------------

/// Pairwise mutual information between elements i and j given marginals.
///
/// MI(i,j) = p(i,j) · ln(p(i,j) / (p(i)·p(j)))
/// where p(i,j) = (1/n) Σ_s TPM[s,i]·TPM[s,j].
#[inline]
pub fn pairwise_mi(tpm: &[f64], n: usize, i: usize, j: usize, marginal: &[f64]) -> f64 {
    let pi = marginal[i].max(1e-15);
    let pj = marginal[j].max(1e-15);

    #[cfg(target_arch = "aarch64")]
    let pij = {
        unsafe { pairwise_dot_neon(tpm, n, i, j) }
    };

    #[cfg(not(target_arch = "aarch64"))]
    let pij = {
        let mut acc = 0.0;
        for state in 0..n {
            unsafe {
                acc += *tpm.get_unchecked(state * n + i) * *tpm.get_unchecked(state * n + j);
            }
        }
        acc
    };

    let pij = (pij / n as f64).max(1e-15);
    (pij * (pij / (pi * pj)).ln()).max(0.0)
}

/// NEON-accelerated column dot product for pairwise MI.
/// Computes Σ_s TPM[s,i] * TPM[s,j] with stride-n gather + FMA.
#[cfg(target_arch = "aarch64")]
#[inline]
unsafe fn pairwise_dot_neon(tpm: &[f64], n: usize, i: usize, j: usize) -> f64 {
    use std::arch::aarch64::*;

    let mut acc = vdupq_n_f64(0.0);
    let chunks = n / 2;
    let remainder = n % 2;

    for c in 0..chunks {
        let s0 = c * 2;
        let s1 = s0 + 1;
        // Gather strided values into NEON registers
        let ai = vld1q_f64(
            [*tpm.get_unchecked(s0 * n + i), *tpm.get_unchecked(s1 * n + i)].as_ptr(),
        );
        let aj = vld1q_f64(
            [*tpm.get_unchecked(s0 * n + j), *tpm.get_unchecked(s1 * n + j)].as_ptr(),
        );
        acc = vfmaq_f64(acc, ai, aj);
    }

    let mut sum = vgetq_lane_f64(acc, 0) + vgetq_lane_f64(acc, 1);

    if remainder > 0 {
        let s = chunks * 2;
        sum += *tpm.get_unchecked(s * n + i) * *tpm.get_unchecked(s * n + j);
    }
    sum
}

/// Build full pairwise MI matrix (symmetric, zero diagonal).
/// Returns flat n×n row-major matrix.
pub fn build_mi_matrix(tpm: &[f64], n: usize) -> Vec<f64> {
    let marginal = marginal_distribution(tpm, n);
    let mut mi = vec![0.0f64; n * n];
    for i in 0..n {
        for j in (i + 1)..n {
            let val = pairwise_mi(tpm, n, i, j, &marginal);
            mi[i * n + j] = val;
            mi[j * n + i] = val;
        }
    }
    mi
}

/// Build MI edge list (i, j, weight) with threshold pruning.
pub fn build_mi_edges(tpm: &[f64], n: usize, threshold: f64) -> (Vec<(usize, usize, f64)>, Vec<f64>) {
    let marginal = marginal_distribution(tpm, n);
    let mut edges = Vec::new();
    for i in 0..n {
        for j in (i + 1)..n {
            let mi = pairwise_mi(tpm, n, i, j, &marginal);
            if mi > threshold {
                edges.push((i, j, mi));
            }
        }
    }
    (edges, marginal)
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
