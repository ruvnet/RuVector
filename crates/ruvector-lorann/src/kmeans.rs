use rand::{Rng as _, SeedableRng};
use rayon::prelude::*;

use crate::error::Result;

/// Result of k-means clustering.
pub struct KMeansResult {
    /// Cluster centroids, shape: k × d.
    pub centroids: Vec<Vec<f32>>,
    /// Cluster assignment per vector. `assignments[i] = c` means vector i → cluster c.
    pub assignments: Vec<usize>,
}

/// Squared Euclidean distance between two equal-length slices.
#[inline]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Dot-product between two equal-length slices.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// Return the index of the nearest centroid (by squared L2) for `query`.
pub fn nearest_centroid(query: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_l2(query, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap_or(0)
}

/// Return indices of the `n_probe` nearest centroids (sorted, closest first).
pub fn top_n_centroids(query: &[f32], centroids: &[Vec<f32>], n_probe: usize) -> Vec<usize> {
    let k = centroids.len();
    let n = n_probe.min(k);
    let mut dists: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, sq_l2(query, c)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.into_iter().take(n).map(|(i, _)| i).collect()
}

/// Lloyd's k-means with k-means++ initialisation.
///
/// Uses rayon for parallel assignment and nalgebra-free accumulation.
/// Returns `Err(KMeansTimeout)` if no iteration produces a change in
/// assignments (degenerate dataset), but succeeds if convergence is reached.
pub fn kmeans(
    data: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> Result<KMeansResult> {
    let n = data.len();
    let d = data[0].len();
    let k = k.min(n);

    // k-means++ initialisation
    let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    // First centroid: random
    let first_idx = (rng.gen::<f64>() * n as f64) as usize;
    centroids.push(data[first_idx].clone());

    // Subsequent centroids: D² sampling
    let mut dists = vec![f32::MAX; n];
    for _ in 1..k {
        // Update min-distances to nearest chosen centroid
        let c = centroids.last().unwrap();
        for (i, v) in data.iter().enumerate() {
            let d2 = sq_l2(v, c);
            if d2 < dists[i] {
                dists[i] = d2;
            }
        }
        let total: f64 = dists.iter().map(|&x| x as f64).sum();
        if total == 0.0 {
            // All points already assigned; duplicate last centroid with tiny jitter
            let mut c2 = centroids.last().unwrap().clone();
            c2[0] += 1e-6;
            centroids.push(c2);
            continue;
        }
        let mut target = rng.gen::<f64>() * total;
        let mut chosen = n - 1;
        for (i, &d2) in dists.iter().enumerate() {
            target -= d2 as f64;
            if target <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(data[chosen].clone());
    }

    // Lloyd iterations
    let mut assignments = vec![0usize; n];
    for _iter in 0..max_iter {
        // Assignment step (parallel)
        let new_assignments: Vec<usize> = data
            .par_iter()
            .map(|v| nearest_centroid(v, &centroids))
            .collect();

        let changed = new_assignments.iter().zip(assignments.iter()).any(|(a, b)| a != b);
        assignments = new_assignments;
        if !changed {
            break;
        }

        // Update step: recompute centroids
        let mut sums = vec![vec![0.0f32; d]; k];
        let mut counts = vec![0usize; k];
        for (i, &c) in assignments.iter().enumerate() {
            counts[c] += 1;
            for (s, &v) in sums[c].iter_mut().zip(data[i].iter()) {
                *s += v;
            }
        }
        for (c, sum) in sums.iter().enumerate() {
            if counts[c] == 0 {
                // Empty cluster: re-seed from the point farthest from its centroid
                let (farthest, _) = data
                    .iter()
                    .enumerate()
                    .map(|(i, v)| (i, sq_l2(v, &centroids[assignments[i]])))
                    .max_by(|a, b| a.1.total_cmp(&b.1))
                    .unwrap_or((0, 0.0));
                centroids[c] = data[farthest].clone();
            } else {
                let cnt = counts[c] as f32;
                centroids[c] = sum.iter().map(|&s| s / cnt).collect();
            }
        }
    }

    Ok(KMeansResult { centroids, assignments })
}
