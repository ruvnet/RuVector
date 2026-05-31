//! K-means++ clustering used by SOAR for IVF partition training.

use crate::error::{Result, SoarError};
use rand::SeedableRng;
use rand::prelude::*;

/// Euclidean squared distance between two equal-length slices.
#[inline]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot product of two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// K-means model: holds trained centroids.
pub struct Kmeans {
    pub centroids: Vec<Vec<f32>>,
    pub dim: usize,
}

impl Kmeans {
    /// Train k-means++ on `vectors`. Panics if `nlist` > `vectors.len()`.
    pub fn train(vectors: &[Vec<f32>], nlist: usize, max_iter: usize, seed: u64) -> Result<Self> {
        if vectors.is_empty() {
            return Err(SoarError::Empty);
        }
        if nlist == 0 || nlist > vectors.len() {
            return Err(SoarError::InvalidConfig(format!(
                "nlist={nlist} must be in 1..={}",
                vectors.len()
            )));
        }
        let dim = vectors[0].len();
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let centroids = kmeans_plus_plus_init(vectors, nlist, &mut rng);
        let centroids = lloyd(vectors, centroids, max_iter);
        Ok(Self { centroids, dim })
    }

    /// Return the index of the closest centroid to `v`.
    pub fn assign(&self, v: &[f32]) -> usize {
        self.centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(v, c)))
            .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
            .map(|(i, _)| i)
            .unwrap()
    }

    /// Return the top-k closest centroids as `(index, sq_distance)`, ascending.
    pub fn top_k(&self, v: &[f32], k: usize) -> Vec<(usize, f32)> {
        let mut dists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(v, c)))
            .collect();
        dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
        dists.truncate(k);
        dists
    }
}

// ── k-means++ initialisation ────────────────────────────────────────────────

fn kmeans_plus_plus_init(
    vectors: &[Vec<f32>],
    k: usize,
    rng: &mut impl Rng,
) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let first = rng.gen_range(0..n);
    let mut centers: Vec<Vec<f32>> = vec![vectors[first].clone()];

    // For each subsequent centroid: sample proportional to min squared distance.
    let mut min_dists: Vec<f32> = vectors.iter().map(|v| l2_sq(v, &centers[0])).collect();

    for _ in 1..k {
        let total: f32 = min_dists.iter().sum();
        let threshold = rng.gen::<f32>() * total;
        let mut cumsum = 0.0f32;
        let mut chosen = n - 1;
        for (i, &d) in min_dists.iter().enumerate() {
            cumsum += d;
            if cumsum >= threshold {
                chosen = i;
                break;
            }
        }
        let new_c = vectors[chosen].clone();
        // Update min distances
        for (i, v) in vectors.iter().enumerate() {
            let d = l2_sq(v, &new_c);
            if d < min_dists[i] {
                min_dists[i] = d;
            }
        }
        centers.push(new_c);
    }
    centers
}

// ── Lloyd iterations ─────────────────────────────────────────────────────────

fn lloyd(vectors: &[Vec<f32>], mut centers: Vec<Vec<f32>>, max_iter: usize) -> Vec<Vec<f32>> {
    let n = vectors.len();
    let k = centers.len();
    let dim = vectors[0].len();
    let mut assignments = vec![0usize; n];

    for _ in 0..max_iter {
        // Assign step
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = centers
                .iter()
                .enumerate()
                .map(|(j, c)| (j, l2_sq(v, c)))
                .min_by(|a, b| a.1.partial_cmp(&b.1).unwrap())
                .map(|(j, _)| j)
                .unwrap();
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
        // Update step: recompute centroids as mean of assigned vectors
        let mut sums: Vec<Vec<f32>> = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            for (d, x) in sums[c].iter_mut().zip(v.iter()) {
                *d += x;
            }
            counts[c] += 1;
        }
        for j in 0..k {
            if counts[j] > 0 {
                for d in 0..dim {
                    centers[j][d] = sums[j][d] / counts[j] as f32;
                }
            }
        }
    }
    centers
}
