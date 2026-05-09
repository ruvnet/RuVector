//! Per-subspace k-means used by both uniform and anisotropic PQ.
//!
//! Two flavors are exposed:
//!   * `kmeans_mse` — standard Lloyd iterations on Euclidean distance.
//!   * `kmeans_aniso` — assigns by *anisotropic loss* and re-centers
//!     by closed-form weighted mean (see `aniso.rs`).

use rand::Rng;

/// Standard Lloyd k-means on `points` with rows of length `dim`.
/// Returns flat centroids `k * dim` and assignments per row.
pub fn kmeans_mse(
    points: &[f32],
    n: usize,
    dim: usize,
    k: usize,
    iters: usize,
    rng: &mut impl Rng,
) -> (Vec<f32>, Vec<u8>) {
    assert!(k <= 256, "k must fit in u8");
    let mut centroids = init_kpp(points, n, dim, k, rng);
    let mut assign = vec![0u8; n];
    for _ in 0..iters {
        // assign
        for i in 0..n {
            let row = &points[i * dim..(i + 1) * dim];
            let mut best = 0u8;
            let mut best_d = f32::INFINITY;
            for c in 0..k {
                let cent = &centroids[c * dim..(c + 1) * dim];
                let mut d = 0.0f32;
                for j in 0..dim {
                    let diff = row[j] - cent[j];
                    d += diff * diff;
                }
                if d < best_d {
                    best_d = d;
                    best = c as u8;
                }
            }
            assign[i] = best;
        }
        // update
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0u32; k];
        for i in 0..n {
            let a = assign[i] as usize;
            counts[a] += 1;
            let row = &points[i * dim..(i + 1) * dim];
            let s = &mut sums[a * dim..(a + 1) * dim];
            for j in 0..dim {
                s[j] += row[j];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // re-seed empty cluster from a random point
                let r = rng.gen_range(0..n);
                centroids[c * dim..(c + 1) * dim]
                    .copy_from_slice(&points[r * dim..(r + 1) * dim]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for j in 0..dim {
                    centroids[c * dim + j] = sums[c * dim + j] * inv;
                }
            }
        }
    }
    (centroids, assign)
}

fn init_kpp(points: &[f32], n: usize, dim: usize, k: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut centroids = Vec::with_capacity(k * dim);
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&points[first * dim..(first + 1) * dim]);
    let mut d2 = vec![f32::INFINITY; n];
    for c in 1..k {
        let last = &centroids[(c - 1) * dim..c * dim];
        for i in 0..n {
            let row = &points[i * dim..(i + 1) * dim];
            let mut d = 0.0f32;
            for j in 0..dim {
                let diff = row[j] - last[j];
                d += diff * diff;
            }
            if d < d2[i] {
                d2[i] = d;
            }
        }
        let total: f32 = d2.iter().sum();
        if total <= 0.0 {
            let r = rng.gen_range(0..n);
            centroids.extend_from_slice(&points[r * dim..(r + 1) * dim]);
            continue;
        }
        let mut t = rng.gen::<f32>() * total;
        let mut pick = n - 1;
        for i in 0..n {
            t -= d2[i];
            if t <= 0.0 {
                pick = i;
                break;
            }
        }
        centroids.extend_from_slice(&points[pick * dim..(pick + 1) * dim]);
    }
    centroids
}
