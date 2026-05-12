//! Lloyd's k-means clustering used for IVF centroid training.
//!
//! Returns `k` centroids and the cluster assignment for every input vector.
//! Uses kmeans++ seeding for stable convergence.

use crate::index::l2sq;
use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Train k centroids on `vectors` for up to `max_iter` iterations.
/// Returns `(centroids, assignments)`.
pub fn train(
    vectors: &[Vec<f32>],
    k: usize,
    max_iter: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    assert!(!vectors.is_empty());
    assert!(k <= vectors.len());
    let dim = vectors[0].len();
    let mut rng = StdRng::seed_from_u64(seed);

    // kmeans++ seeding
    let mut centroids = kmeanspp_seed(vectors, k, &mut rng);

    let mut assignments = vec![0usize; vectors.len()];
    for _ in 0..max_iter {
        // Assignment step
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = nearest_centroid(v, &centroids);
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }

        // Update step
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            for d in 0..dim {
                sums[c][d] += v[d];
            }
            counts[c] += 1;
        }
        for c in 0..k {
            if counts[c] > 0 {
                let n = counts[c] as f32;
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / n;
                }
            } else {
                // empty cluster: reinitialise to a random vector
                let idx = rng.gen_range(0..vectors.len());
                centroids[c] = vectors[idx].clone();
            }
        }
    }

    // Final assignment pass
    for (i, v) in vectors.iter().enumerate() {
        assignments[i] = nearest_centroid(v, &centroids);
    }

    (centroids, assignments)
}

/// Find the index of the centroid nearest to `v`.
#[inline]
pub fn nearest_centroid(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2sq(v, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// Return the two nearest centroid indices for `v`.
pub fn two_nearest(v: &[f32], centroids: &[Vec<f32>]) -> (usize, f32, usize, f32) {
    let mut best = (0usize, f32::INFINITY);
    let mut second = (0usize, f32::INFINITY);
    for (i, c) in centroids.iter().enumerate() {
        let d = l2sq(v, c);
        if d < best.1 {
            second = best;
            best = (i, d);
        } else if d < second.1 {
            second = (i, d);
        }
    }
    (best.0, best.1, second.0, second.1)
}

// ─── kmeans++ seeding ─────────────────────────────────────────────────────────

fn kmeanspp_seed(vectors: &[Vec<f32>], k: usize, rng: &mut StdRng) -> Vec<Vec<f32>> {
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    // Pick first centroid uniformly at random
    centroids.push(vectors[rng.gen_range(0..vectors.len())].clone());

    for _ in 1..k {
        // For each vector compute min-distance to existing centroids (D² weighting)
        let dists: Vec<f32> = vectors
            .iter()
            .map(|v| {
                centroids
                    .iter()
                    .map(|c| l2sq(v, c))
                    .fold(f32::INFINITY, f32::min)
            })
            .collect();
        let total: f32 = dists.iter().sum();
        let threshold = rng.gen::<f32>() * total;
        let mut cum = 0.0f32;
        let mut chosen = vectors.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            cum += d;
            if cum >= threshold {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen].clone());
    }
    centroids
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_separated() {
        let mut vecs: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32 * 0.01, 0.0]).collect();
        let far: Vec<Vec<f32>> = (0..50).map(|i| vec![10.0 + i as f32 * 0.01, 0.0]).collect();
        vecs.extend(far);
        let (centroids, assignments) = train(&vecs, 2, 50, 42);
        assert_eq!(centroids.len(), 2);
        // All first 50 should share one cluster, last 50 the other
        let cluster_a = assignments[0];
        for a in &assignments[..50] {
            assert_eq!(*a, cluster_a);
        }
        let cluster_b = assignments[50];
        assert_ne!(cluster_a, cluster_b);
        for a in &assignments[50..] {
            assert_eq!(*a, cluster_b);
        }
    }

    #[test]
    fn nearest_centroid_correct() {
        let centroids = vec![vec![0.0f32, 0.0], vec![10.0, 10.0]];
        assert_eq!(nearest_centroid(&[0.1, 0.1], &centroids), 0);
        assert_eq!(nearest_centroid(&[9.9, 9.9], &centroids), 1);
    }
}
