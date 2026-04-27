//! Eigensolvers used by the Fiedler detector.
//!
//! For small graphs (`n ≤ 96`) we compute all Laplacian eigenvalues
//! via cyclic Jacobi rotations, which is robust at this scale and
//! trivially finds the Fiedler value as the second smallest
//! eigenvalue. For larger windows we fall back to a shifted
//! power-iteration approximation.

/// Full eigendecomposition of a symmetric `n × n` matrix by cyclic
/// Jacobi rotations. Accurate and robust for small `n` (≤ 96); O(n³)
/// per sweep. Returns the `n` eigenvalues (order is not guaranteed;
/// caller sorts).
pub fn jacobi_symmetric(a_in: &[f32], n: usize) -> Vec<f32> {
    let mut a: Vec<f32> = a_in.to_vec();
    let max_sweeps = 50;
    for _ in 0..max_sweeps {
        let mut off = 0.0_f32;
        for p in 0..n {
            for q in (p + 1)..n {
                let x = a[p * n + q];
                off += x * x;
            }
        }
        if off < 1e-10 {
            break;
        }
        for p in 0..n {
            for q in (p + 1)..n {
                let apq = a[p * n + q];
                if apq.abs() < 1e-10 {
                    continue;
                }
                let app = a[p * n + p];
                let aqq = a[q * n + q];
                let theta = (aqq - app) / (2.0 * apq);
                let t = if theta >= 0.0 {
                    1.0 / (theta + (1.0 + theta * theta).sqrt())
                } else {
                    1.0 / (theta - (1.0 + theta * theta).sqrt())
                };
                let c = 1.0 / (1.0 + t * t).sqrt();
                let s = t * c;
                for i in 0..n {
                    let aip = a[i * n + p];
                    let aiq = a[i * n + q];
                    a[i * n + p] = c * aip - s * aiq;
                    a[i * n + q] = s * aip + c * aiq;
                }
                for j in 0..n {
                    let apj = a[p * n + j];
                    let aqj = a[q * n + j];
                    a[p * n + j] = c * apj - s * aqj;
                    a[q * n + j] = s * apj + c * aqj;
                }
                a[p * n + q] = 0.0;
                a[q * n + p] = 0.0;
            }
        }
    }
    (0..n).map(|i| a[i * n + i]).collect()
}

/// Shifted power-iteration fallback for windows with more than 96
/// active neurons. Estimates the smallest non-zero eigenvalue of
/// `L = D - A` by iterating on `(c·I − L)` with deflation against the
/// constant eigenvector.
pub fn approx_fiedler_power(a: &[f32], n: usize) -> f32 {
    let mut deg = vec![0.0_f32; n];
    for i in 0..n {
        let mut d = 0.0_f32;
        for j in 0..n {
            d += a[i * n + j];
        }
        deg[i] = d;
    }
    // λ_max(L) estimate by power iteration with constant-vector
    // deflation.
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 31 + 7) as f32).sin()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut lambda_max = 0.0_f32;
    for _ in 0..32 {
        let y = mul_l(&deg, a, n, &x);
        let mut y = y;
        deflate_const(&mut y);
        normalize(&mut y);
        let lam = rayleigh_l(&deg, a, n, &y);
        let converged = (lam - lambda_max).abs() < 1e-4 * lam.abs().max(1.0);
        lambda_max = lam;
        x = y;
        if converged {
            break;
        }
    }
    let c = lambda_max * 1.1 + 1e-3;
    let mut x: Vec<f32> = (0..n).map(|i| ((i * 19 + 11) as f32).cos()).collect();
    deflate_const(&mut x);
    normalize(&mut x);
    let mut mu = 0.0_f32;
    for _ in 0..64 {
        let lx = mul_l(&deg, a, n, &x);
        let mut y: Vec<f32> = (0..n).map(|i| c * x[i] - lx[i]).collect();
        deflate_const(&mut y);
        normalize(&mut y);
        let ly = mul_l(&deg, a, n, &y);
        let mut m2 = 0.0_f32;
        for i in 0..n {
            m2 += y[i] * (c * y[i] - ly[i]);
        }
        if (m2 - mu).abs() < 1e-4 * m2.abs().max(1.0) {
            mu = m2;
            break;
        }
        mu = m2;
        x = y;
    }
    (lambda_max - mu).max(0.0)
}

fn mul_l(deg: &[f32], a: &[f32], n: usize, x: &[f32]) -> Vec<f32> {
    let mut y = vec![0.0_f32; n];
    for i in 0..n {
        let mut s = deg[i] * x[i];
        for j in 0..n {
            s -= a[i * n + j] * x[j];
        }
        y[i] = s;
    }
    y
}

fn rayleigh_l(deg: &[f32], a: &[f32], n: usize, y: &[f32]) -> f32 {
    let mut lam = 0.0_f32;
    for i in 0..n {
        let mut s = deg[i] * y[i];
        for j in 0..n {
            s -= a[i * n + j] * y[j];
        }
        lam += y[i] * s;
    }
    lam
}

fn deflate_const(x: &mut [f32]) {
    let m: f32 = x.iter().sum::<f32>() / x.len() as f32;
    for v in x.iter_mut() {
        *v -= m;
    }
}

fn normalize(x: &mut [f32]) {
    let norm: f32 = x.iter().map(|v| v * v).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in x.iter_mut() {
            *v /= norm;
        }
    }
}
