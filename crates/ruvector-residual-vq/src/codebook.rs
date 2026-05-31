//! Single-codebook k-means quantizer used as one RVQ stage.

use rand::Rng;

/// A codebook of K centroids in R^dim trained by Lloyd's algorithm.
///
/// Encoded values are `u8` (0–255), so K ≤ 256.
#[derive(Debug, Clone)]
pub struct Codebook {
    pub k: usize,
    pub dim: usize,
    /// Row-major: centroids[i*dim .. (i+1)*dim] = centroid i.
    centroids: Vec<f32>,
    /// ||c_i||² precomputed for fast ADC.
    centroid_sq_norms: Vec<f32>,
}

impl Codebook {
    /// Train a codebook on `data` (flat row-major, n×dim).
    ///
    /// Uses k-means++ seeding then Lloyd's iterations. Empty clusters keep
    /// their previous centroid (stable without thrashing).
    pub fn train(data: &[f32], dim: usize, k: usize, n_iter: usize, rng: &mut impl Rng) -> Self {
        assert!(!data.is_empty() && dim > 0 && k > 0);
        let n = data.len() / dim;
        let k = k.min(n);

        let mut centroids = kmeans_pp_init(data, dim, n, k, rng);

        for _ in 0..n_iter {
            if !kmeans_lloyd_step(data, dim, n, &mut centroids, k) {
                break;
            }
        }

        let centroid_sq_norms = (0..k)
            .map(|i| {
                let c = &centroids[i * dim..(i + 1) * dim];
                c.iter().map(|x| x * x).sum()
            })
            .collect();

        Self { k, dim, centroids, centroid_sq_norms }
    }

    /// Find the nearest centroid index. Returns 0..k-1 as u8.
    #[inline]
    pub fn quantize(&self, v: &[f32]) -> u8 {
        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for i in 0..self.k {
            let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
            let d: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        best_i as u8
    }

    /// Return (code, sq_distance) for the nearest centroid.
    #[inline]
    pub fn quantize_with_dist(&self, v: &[f32]) -> (u8, f32) {
        let mut best_i = 0usize;
        let mut best_d = f32::MAX;
        for i in 0..self.k {
            let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
            let d: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best_i = i;
            }
        }
        (best_i as u8, best_d)
    }

    /// Return the top-`n` nearest centroids as (index, sq_distance) sorted ascending.
    pub fn top_n(&self, v: &[f32], n: usize) -> Vec<(u8, f32)> {
        let mut scores: Vec<(u8, f32)> = (0..self.k)
            .map(|i| {
                let c = &self.centroids[i * self.dim..(i + 1) * self.dim];
                let d: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
                (i as u8, d)
            })
            .collect();
        scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        scores.truncate(n);
        scores
    }

    #[inline]
    pub fn centroid(&self, code: u8) -> &[f32] {
        let i = code as usize;
        &self.centroids[i * self.dim..(i + 1) * self.dim]
    }

    #[inline]
    pub fn centroid_sq_norm(&self, code: u8) -> f32 {
        self.centroid_sq_norms[code as usize]
    }

    /// Compute v − centroid[code] (the residual for the next RVQ stage).
    #[inline]
    pub fn residual(&self, v: &[f32], code: u8) -> Vec<f32> {
        let c = self.centroid(code);
        v.iter().zip(c).map(|(a, b)| a - b).collect()
    }

    /// ⟨q, centroid[code]⟩ — used in ADC inner-product table.
    #[inline]
    pub fn inner_product(&self, q: &[f32], code: u8) -> f32 {
        let c = self.centroid(code);
        q.iter().zip(c).map(|(a, b)| a * b).sum()
    }

    pub fn memory_bytes(&self) -> usize {
        self.centroids.len() * 4 + self.centroid_sq_norms.len() * 4 + std::mem::size_of::<Self>()
    }
}

// ── k-means helpers ──────────────────────────────────────────────────────────

fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// k-means++ seeding: probabilistic selection proportional to D² distance.
fn kmeans_pp_init(data: &[f32], dim: usize, n: usize, k: usize, rng: &mut impl Rng) -> Vec<f32> {
    let mut centroids: Vec<f32> = Vec::with_capacity(k * dim);

    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&data[first * dim..(first + 1) * dim]);

    let mut min_dists: Vec<f32> = (0..n)
        .map(|i| l2_sq(&data[i * dim..(i + 1) * dim], &centroids[0..dim]))
        .collect();

    for c_idx in 1..k {
        let total: f32 = min_dists.iter().sum();
        let pick = if total <= 0.0 {
            rng.gen_range(0..n)
        } else {
            let mut r = rng.gen::<f32>() * total;
            let mut p = n - 1;
            for (i, &d) in min_dists.iter().enumerate() {
                r -= d;
                if r <= 0.0 {
                    p = i;
                    break;
                }
            }
            p
        };
        centroids.extend_from_slice(&data[pick * dim..(pick + 1) * dim]);

        let new_c = &centroids[c_idx * dim..(c_idx + 1) * dim];
        for (i, md) in min_dists.iter_mut().enumerate() {
            let d = l2_sq(&data[i * dim..(i + 1) * dim], new_c);
            if d < *md {
                *md = d;
            }
        }
    }

    centroids
}

/// One Lloyd iteration. Returns true if any assignment changed.
fn kmeans_lloyd_step(
    data: &[f32],
    dim: usize,
    n: usize,
    centroids: &mut [f32],
    k: usize,
) -> bool {
    let mut assignments = vec![0usize; n];
    let mut changed = false;

    // Assignment step
    for i in 0..n {
        let v = &data[i * dim..(i + 1) * dim];
        let mut best_j = 0usize;
        let mut best_d = f32::MAX;
        for j in 0..k {
            let c = &centroids[j * dim..(j + 1) * dim];
            let d: f32 = v.iter().zip(c).map(|(a, b)| (a - b) * (a - b)).sum();
            if d < best_d {
                best_d = d;
                best_j = j;
            }
        }
        if assignments[i] != best_j {
            changed = true;
        }
        assignments[i] = best_j;
    }

    // Update step: accumulate sums
    let mut sums = vec![0.0f64; k * dim];
    let mut counts = vec![0usize; k];
    for i in 0..n {
        let j = assignments[i];
        counts[j] += 1;
        for d in 0..dim {
            sums[j * dim + d] += data[i * dim + d] as f64;
        }
    }

    // Divide — empty clusters keep old centroid
    for j in 0..k {
        if counts[j] > 0 {
            let inv = 1.0 / counts[j] as f64;
            for d in 0..dim {
                centroids[j * dim + d] = (sums[j * dim + d] * inv) as f32;
            }
        }
    }

    changed
}

#[cfg(test)]
mod tests {
    use super::*;
    use rand::SeedableRng;

    fn make_clustered(n: usize, dim: usize, k_clusters: usize, seed: u64) -> Vec<f32> {
        use rand_distr::{Distribution, Normal};
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let noise = Normal::new(0.0f32, 0.1).unwrap();
        let centers: Vec<Vec<f32>> = (0..k_clusters)
            .map(|c| {
                let base = c as f32;
                (0..dim).map(|d| base + d as f32 * 0.1).collect()
            })
            .collect();
        let mut out = Vec::with_capacity(n * dim);
        for i in 0..n {
            let c = &centers[i % k_clusters];
            for &x in c {
                out.push(x + noise.sample(&mut rng));
            }
        }
        out
    }

    #[test]
    fn codebook_basic() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(0);
        let data = make_clustered(200, 8, 4, 0);
        let cb = Codebook::train(&data, 8, 16, 15, &mut rng);
        assert_eq!(cb.k, 16);
        assert_eq!(cb.dim, 8);
        // Every quantized code should be valid
        for i in 0..20 {
            let v = &data[i * 8..(i + 1) * 8];
            let code = cb.quantize(v);
            assert!((code as usize) < cb.k);
        }
    }

    #[test]
    fn residual_shrinks() {
        let mut rng = rand::rngs::StdRng::seed_from_u64(1);
        let data = make_clustered(100, 16, 4, 1);
        let cb = Codebook::train(&data, 16, 32, 10, &mut rng);
        let v = &data[0..16];
        let code = cb.quantize(v);
        let res = cb.residual(v, code);
        let orig_norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        let res_norm: f32 = res.iter().map(|x| x * x).sum::<f32>().sqrt();
        // Residual should typically be smaller than original
        assert!(res_norm <= orig_norm + 1e-3, "residual {res_norm:.4} > orig {orig_norm:.4}");
    }
}
