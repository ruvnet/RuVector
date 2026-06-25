//! Lloyd's k-means for centroid computation.
//!
//! Produces `k` centroids from a slice of f32 vectors.
//! Uses a deterministic seeding strategy (pick every n/k-th vector)
//! so benchmarks are reproducible without RNG.

use crate::distance::l2_squared;

/// Run Lloyd's k-means for `iters` iterations.
/// Returns a Vec of k centroids, each of length `dim`.
pub fn kmeans(vectors: &[Vec<f32>], k: usize, dim: usize, iters: usize) -> Vec<Vec<f32>> {
    assert!(!vectors.is_empty());
    assert!(k <= vectors.len());

    // Deterministic seed: pick evenly spaced vectors.
    let step = vectors.len() / k;
    let mut centroids: Vec<Vec<f32>> = (0..k).map(|i| vectors[i * step].clone()).collect();

    let mut assignments = vec![0usize; vectors.len()];

    for _iter in 0..iters {
        // Assign each vector to nearest centroid.
        for (i, v) in vectors.iter().enumerate() {
            let nearest = centroids
                .iter()
                .enumerate()
                .min_by(|(_, ca), (_, cb)| {
                    l2_squared(v, ca)
                        .partial_cmp(&l2_squared(v, cb))
                        .unwrap_or(std::cmp::Ordering::Equal)
                })
                .map(|(idx, _)| idx)
                .unwrap_or(0);
            assignments[i] = nearest;
        }

        // Recompute centroids as mean of assigned vectors.
        let mut sums = vec![vec![0.0f32; dim]; k];
        let mut counts = vec![0usize; k];
        for (i, v) in vectors.iter().enumerate() {
            let c = assignments[i];
            counts[c] += 1;
            for d in 0..dim {
                sums[c][d] += v[d];
            }
        }
        for c in 0..k {
            if counts[c] > 0 {
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] / counts[c] as f32;
                }
            }
            // If a centroid is empty, keep it (rare with evenly-spaced seeding).
        }
    }

    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn kmeans_two_clusters() {
        // 20 vectors clearly in two clusters around [0,0] and [10,10].
        let mut vecs: Vec<Vec<f32>> = Vec::new();
        for i in 0..10 {
            let x = i as f32 * 0.1;
            vecs.push(vec![x, x]);
        }
        for i in 0..10 {
            let x = 10.0 + i as f32 * 0.1;
            vecs.push(vec![x, x]);
        }
        let centroids = kmeans(&vecs, 2, 2, 10);
        assert_eq!(centroids.len(), 2);
        // One centroid near [0.45, 0.45] and one near [10.45, 10.45].
        let c0_mean = (centroids[0][0] + centroids[1][0]) / 2.0;
        assert!(
            c0_mean > 2.0 && c0_mean < 15.0,
            "centroids diverged: {c0_mean}"
        );
    }
}
