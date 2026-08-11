//! Lloyd's k-means for codebook training.
//!
//! Intentionally minimal: no parallelism, no k-means++, 20 iterations.
//! This is sufficient for a research PoC on N≤50K, D≤256.

use crate::{dataset::Lcg, nearest_centroid};

/// Train `n_centroids` centroids on `data` using Lloyd's algorithm.
///
/// Initialization: sample `n_centroids` distinct data points uniformly.
/// Empty clusters keep their previous centroid (avoids NaN from zero-count div).
pub fn kmeans(data: &[Vec<f32>], n_centroids: usize, max_iter: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(!data.is_empty(), "kmeans: empty dataset");
    assert!(
        n_centroids <= data.len(),
        "kmeans: more centroids than data points"
    );

    let dim = data[0].len();
    let n = data.len();
    let mut rng = Lcg::new(seed);

    // Initialise centroids by sampling without replacement (Fisher-Yates).
    let mut indices: Vec<usize> = (0..n).collect();
    for i in 0..n_centroids {
        let j = i + rng.next_usize(n - i);
        indices.swap(i, j);
    }
    let mut centroids: Vec<Vec<f32>> = indices[..n_centroids]
        .iter()
        .map(|&i| data[i].clone())
        .collect();

    // Initialise with sentinel so the first iteration always runs an update.
    let mut assignments = vec![usize::MAX; n];

    for _iter in 0..max_iter {
        // Assignment step.
        let mut changed = false;
        for (i, v) in data.iter().enumerate() {
            let (c, _) = nearest_centroid(v, &centroids);
            if assignments[i] != c {
                assignments[i] = c;
                changed = true;
            }
        }
        // Update step (always run after the assignment, then check convergence).
        let mut sums = vec![vec![0.0f32; dim]; n_centroids];
        let mut counts = vec![0u32; n_centroids];
        for (v, &c) in data.iter().zip(assignments.iter()) {
            for (s, &vi) in sums[c].iter_mut().zip(v.iter()) {
                *s += vi;
            }
            counts[c] += 1;
        }
        for (c, (&cnt, sum)) in counts.iter().zip(sums.iter()).enumerate() {
            if cnt > 0 {
                centroids[c] = sum.iter().map(|&s| s / cnt as f32).collect();
            }
            // If cnt == 0, keep previous centroid (avoids degenerate empty cluster).
        }

        if !changed {
            break;
        }
    }

    centroids
}

/// Quantisation error: mean squared L2 distance from each vector to its centroid.
pub fn quantisation_error(data: &[Vec<f32>], centroids: &[Vec<f32>]) -> f32 {
    let total: f32 = data.iter().map(|v| nearest_centroid(v, centroids).1).sum();
    total / data.len() as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let n = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        v.into_iter().map(|x| x / n).collect()
    }

    #[test]
    fn kmeans_two_clusters() {
        // Two clearly separated clusters; k-means must find both.
        let mut data: Vec<Vec<f32>> = Vec::new();
        for _ in 0..50 {
            data.push(unit(vec![1.0, 0.0, 0.0]));
        }
        for _ in 0..50 {
            data.push(unit(vec![-1.0, 0.0, 0.0]));
        }
        let centroids = kmeans(&data, 2, 20, 1);
        assert_eq!(centroids.len(), 2);
        // Each centroid should be near one of the two cluster centres.
        let qe = quantisation_error(&data, &centroids);
        assert!(qe < 0.01, "unexpected quantisation error: {qe}");
    }

    #[test]
    fn kmeans_single_point_clusters() {
        let data: Vec<Vec<f32>> = (0..4).map(|i| vec![i as f32, 0.0]).collect();
        let centroids = kmeans(&data, 4, 10, 42);
        assert_eq!(centroids.len(), 4);
    }
}
