//! Minimal top-`m` PCA via power iteration + deflation — for BET 4's **low-dimensional control**.
//!
//! Projecting the real arxiv features onto their top principal components gives the *same data*
//! at low intrinsic dimensionality, where the triangle-inequality cluster bound should be tight
//! and the B&B kernel is expected to WIN — proving the kernel/harness are sound and isolating
//! high-dimensional distance concentration as the cause of any 128-d NO-GO. No linalg dependency.

/// Project `data` (n × dim) onto its top `m` principal components, returning n × m coordinates.
/// Data is mean-centered first; components found by power iteration with deflation (`iters` steps
/// each). f64 accumulation for numerical stability.
pub fn project_topm(data: &[Vec<f32>], m: usize, iters: usize) -> Vec<Vec<f32>> {
    let n = data.len();
    if n == 0 {
        return Vec::new();
    }
    let dim = data[0].len();

    let mut mean = vec![0.0f64; dim];
    for v in data {
        for (d, &x) in v.iter().enumerate() {
            mean[d] += x as f64;
        }
    }
    for x in &mut mean {
        *x /= n as f64;
    }
    let centered: Vec<Vec<f64>> = data
        .iter()
        .map(|v| (0..dim).map(|d| v[d] as f64 - mean[d]).collect())
        .collect();

    let mut comps: Vec<Vec<f64>> = Vec::with_capacity(m.min(dim));
    for c in 0..m.min(dim) {
        let mut v = vec![0.0f64; dim];
        v[c % dim] = 1.0;
        for _ in 0..iters {
            // u = Σ_i (x_i · v) x_i  — covariance-times-v without forming the covariance matrix.
            let mut u = vec![0.0f64; dim];
            for x in &centered {
                let dot: f64 = x.iter().zip(&v).map(|(a, b)| a * b).sum();
                for (d, &xd) in x.iter().enumerate() {
                    u[d] += dot * xd;
                }
            }
            // Deflate against already-found components (Gram–Schmidt).
            for prev in &comps {
                let proj: f64 = u.iter().zip(prev).map(|(a, b)| a * b).sum();
                for (d, &pd) in prev.iter().enumerate() {
                    u[d] -= proj * pd;
                }
            }
            let norm = u.iter().map(|x| x * x).sum::<f64>().sqrt();
            if norm < 1e-12 {
                break;
            }
            for x in &mut u {
                *x /= norm;
            }
            v = u;
        }
        comps.push(v);
    }

    centered
        .iter()
        .map(|x| {
            comps
                .iter()
                .map(|comp| x.iter().zip(comp).map(|(a, b)| a * b).sum::<f64>() as f32)
                .collect()
        })
        .collect()
}
