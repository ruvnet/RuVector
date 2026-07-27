//! Single-stage k-means codebook (Lloyd's algorithm with K-means++ init).

use rand::SeedableRng;
use rand::Rng as _;

/// One quantization codebook: K centroids in `dim`-dimensional space.
#[derive(Debug, Clone)]
pub struct Codebook {
    /// Flat layout: centroid c occupies `centroids[c * dim .. (c+1) * dim]`.
    pub centroids: Vec<f32>,
    pub k: usize,
    pub dim: usize,
}

impl Codebook {
    /// Train via Lloyd's algorithm with K-means++ initialization.
    ///
    /// `data` is a slice of row-major f32 vectors, each of length `dim`.
    pub fn train(data: &[Vec<f32>], k: usize, dim: usize, max_iter: usize, seed: u64) -> Self {
        assert!(!data.is_empty(), "codebook training requires data");
        assert!(k >= 1 && k <= 256, "k must be 1..=256");
        let k = k.min(data.len()); // can't have more centroids than points

        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centroids = kmeans_plusplus_init(data, k, dim, &mut rng);
        lloyd(data, centroids, k, dim, max_iter, &mut rng)
    }

    /// Return the index of the nearest centroid (L2 distance).
    #[inline]
    pub fn encode(&self, v: &[f32]) -> u8 {
        debug_assert_eq!(v.len(), self.dim);
        let mut best_idx = 0usize;
        let mut best_dist = f32::MAX;
        for c in 0..self.k {
            let d = l2_sq(v, self.centroid(c));
            if d < best_dist {
                best_dist = d;
                best_idx = c;
            }
        }
        best_idx as u8
    }

    /// View centroid `c` as a slice.
    #[inline]
    pub fn centroid(&self, c: usize) -> &[f32] {
        &self.centroids[c * self.dim..(c + 1) * self.dim]
    }

    /// Compute the residual: `v - centroid[encode(v)]`.
    pub fn residual(&self, v: &[f32]) -> Vec<f32> {
        let c = self.encode(v) as usize;
        let centroid = self.centroid(c);
        v.iter().zip(centroid).map(|(a, b)| a - b).collect()
    }

    /// Precompute squared norms of all centroids (for ADC distance tables).
    pub fn centroid_norms_sq(&self) -> Vec<f32> {
        (0..self.k).map(|c| l2_sq_self(self.centroid(c))).collect()
    }
}

// ── K-means++ initialisation ─────────────────────────────────────────────────

fn kmeans_plusplus_init(
    data: &[Vec<f32>],
    k: usize,
    dim: usize,
    rng: &mut rand::rngs::StdRng,
) -> Vec<f32> {
    let n = data.len();
    let mut centroids = Vec::<f32>::with_capacity(k * dim);
    // Pick first centroid uniformly at random.
    let first = rng.gen_range(0..n);
    centroids.extend_from_slice(&data[first]);

    let mut dists: Vec<f32> = vec![f32::MAX; n];
    for num_chosen in 1..k {
        // Update min-distances to the most recently added centroid.
        let last_centroid = &centroids[(num_chosen - 1) * dim..num_chosen * dim];
        for (i, v) in data.iter().enumerate() {
            let d = l2_sq(v, last_centroid);
            if d < dists[i] {
                dists[i] = d;
            }
        }
        // Sample proportional to distance².
        let total: f32 = dists.iter().sum();
        let mut threshold = rng.gen::<f32>() * total;
        let mut chosen = n - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.extend_from_slice(&data[chosen]);
    }
    centroids
}

// ── Lloyd's algorithm ─────────────────────────────────────────────────────────

fn lloyd(
    data: &[Vec<f32>],
    mut centroids: Vec<f32>,
    k: usize,
    dim: usize,
    max_iter: usize,
    rng: &mut rand::rngs::StdRng,
) -> Codebook {
    let n = data.len();
    let mut assignments = vec![0u8; n];

    for _iter in 0..max_iter {
        // Assignment step.
        let mut changed = false;
        for (i, v) in data.iter().enumerate() {
            let mut best = 0u8;
            let mut best_d = f32::MAX;
            for c in 0..k {
                let d = l2_sq(v, &centroids[c * dim..(c + 1) * dim]);
                if d < best_d {
                    best_d = d;
                    best = c as u8;
                }
            }
            if assignments[i] != best {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update step.
        let mut sums = vec![0.0f32; k * dim];
        let mut counts = vec![0usize; k];
        for (i, v) in data.iter().enumerate() {
            let c = assignments[i] as usize;
            counts[c] += 1;
            for d in 0..dim {
                sums[c * dim + d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] == 0 {
                // Reinitialise empty centroid to a random data point.
                let r = rng.gen_range(0..n);
                centroids[c * dim..(c + 1) * dim].copy_from_slice(&data[r]);
            } else {
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c * dim + d] = sums[c * dim + d] * inv;
                }
            }
        }
    }
    Codebook { centroids, k, dim }
}

// ── Distance helpers ──────────────────────────────────────────────────────────

#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[inline]
pub fn l2_sq_self(a: &[f32]) -> f32 {
    a.iter().map(|x| x * x).sum()
}

#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| x * y).sum()
}
