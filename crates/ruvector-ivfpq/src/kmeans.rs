//! k-means++ clustering for IVF centroid and PQ codebook training.

use rand::{rngs::StdRng, Rng, SeedableRng};

/// Squared Euclidean distance between two equal-length slices.
#[inline]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(&x, &y)| (x - y) * (x - y)).sum()
}

/// Index of the nearest centroid to `v`.
pub fn nearest(v: &[f32], centroids: &[Vec<f32>]) -> usize {
    centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2sq(v, c)))
        .min_by(|a, b| a.1.total_cmp(&b.1))
        .map(|(i, _)| i)
        .unwrap()
}

/// Indices of the `n` nearest centroids to `v`, sorted closest-first.
pub fn n_nearest(v: &[f32], centroids: &[Vec<f32>], n: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2sq(v, c)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.total_cmp(&b.1));
    dists.iter().take(n).map(|(i, _)| *i).collect()
}

/// k-means++ clustering on `vectors`. Returns `k` centroids.
pub fn train(vectors: &[Vec<f32>], k: usize, max_iter: usize, seed: u64) -> Vec<Vec<f32>> {
    assert!(!vectors.is_empty());
    assert!(k <= vectors.len(), "k={k} > n={}", vectors.len());
    let dim = vectors[0].len();
    let mut rng = StdRng::seed_from_u64(seed);

    // k-means++ seeding: D² weighted selection
    let mut centroids: Vec<Vec<f32>> = Vec::with_capacity(k);
    centroids.push(vectors[rng.gen_range(0..vectors.len())].clone());
    while centroids.len() < k {
        let dists: Vec<f32> = vectors
            .iter()
            .map(|v| centroids.iter().map(|c| l2sq(v, c)).fold(f32::INFINITY, f32::min))
            .collect();
        let total: f32 = dists.iter().sum();
        let mut threshold = rng.gen::<f32>() * total;
        let mut chosen = vectors.len() - 1;
        for (i, &d) in dists.iter().enumerate() {
            threshold -= d;
            if threshold <= 0.0 {
                chosen = i;
                break;
            }
        }
        centroids.push(vectors[chosen].clone());
    }

    // Lloyd's iterations.
    // usize::MAX ensures the first iteration always detects a change and runs an update,
    // even when k=1 and the initial seed happens to equal centroid index 0.
    let mut assignments = vec![usize::MAX; vectors.len()];
    for _ in 0..max_iter {
        let mut changed = false;
        for (i, v) in vectors.iter().enumerate() {
            let best = nearest(v, &centroids);
            if best != assignments[i] {
                assignments[i] = best;
                changed = true;
            }
        }
        if !changed {
            break;
        }
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
                let inv = 1.0 / counts[c] as f32;
                for d in 0..dim {
                    centroids[c][d] = sums[c][d] * inv;
                }
            } else {
                centroids[c] = vectors[rng.gen_range(0..vectors.len())].clone();
            }
        }
    }
    centroids
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn two_clusters_separated() {
        let mut vecs: Vec<Vec<f32>> = (0..50).map(|i| vec![i as f32 * 0.01]).collect();
        vecs.extend((0..50).map(|i| vec![10.0 + i as f32 * 0.01f32]));
        let centroids = train(&vecs, 2, 50, 42);
        assert_eq!(centroids.len(), 2);
        let c0 = centroids[0][0];
        let c1 = centroids[1][0];
        let (lo, hi) = if c0 < c1 { (c0, c1) } else { (c1, c0) };
        assert!(lo < 1.0, "low centroid {lo:.3} should be near 0.25");
        assert!(hi > 9.0, "high centroid {hi:.3} should be near 10.25");
    }

    #[test]
    fn n_nearest_ordering() {
        let cs = vec![vec![0.0f32], vec![3.0], vec![7.0], vec![10.0]];
        let res = n_nearest(&[3.1], &cs, 2);
        assert_eq!(res[0], 1, "nearest to 3.1 is index 1 (value 3.0)");
        assert_eq!(res[1], 0, "second nearest to 3.1 is index 0 (value 0.0)");
    }

    #[test]
    fn single_centroid() {
        let vecs: Vec<Vec<f32>> = (0..20).map(|i| vec![i as f32]).collect();
        let cs = train(&vecs, 1, 10, 0);
        assert_eq!(cs.len(), 1);
        // Centroid should be mean ≈ 9.5
        assert!((cs[0][0] - 9.5).abs() < 0.1, "centroid={}", cs[0][0]);
    }
}
