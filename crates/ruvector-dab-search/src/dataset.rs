//! Deterministic synthetic dataset generation.
//!
//! Identical generator to `ruvector-entropy-ann` (ADR-303) by design: this
//! crate's benchmark reuses the same corpus/query construction so results are
//! directly comparable across the two nightly experiments, controlling for
//! the dataset as a confound.

/// A deterministic pseudo-random f32 in [0, 1).
fn lcg_rand(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6364136223846793005)
        .wrapping_add(1442695040888963407);
    ((*state >> 33) as f32) / (u32::MAX as f32)
}

/// Generate `n` random unit-normalised vectors of dimension `dim`.
pub fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed;
    (0..n)
        .map(|_| {
            let mut v: Vec<f32> = (0..dim).map(|_| lcg_rand(&mut state) * 2.0 - 1.0).collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

/// Cluster-structured dataset: `num_clusters` Gaussian-ish blobs.
pub fn clustered_vectors(
    n: usize,
    dim: usize,
    num_clusters: usize,
    noise: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut state = seed;

    let centroids: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dim).map(|_| lcg_rand(&mut state) * 2.0 - 1.0).collect();
            let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            c.iter_mut().for_each(|x| *x /= norm);
            c
        })
        .collect();

    (0..n)
        .map(|i| {
            let c = &centroids[i % num_clusters];
            let mut v: Vec<f32> = c
                .iter()
                .map(|&x| {
                    let n_val = (lcg_rand(&mut state) * 2.0 - 1.0) * noise;
                    x + n_val
                })
                .collect();
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.iter_mut().for_each(|x| *x /= norm);
            v
        })
        .collect()
}

/// Brute-force exact top-k nearest neighbours by squared L2 distance.
pub fn ground_truth(query: &[f32], corpus: &[Vec<f32>], k: usize) -> Vec<usize> {
    let mut dists: Vec<(usize, f32)> = corpus
        .iter()
        .enumerate()
        .map(|(i, v)| (i, l2sq(query, v)))
        .collect();
    dists.sort_unstable_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists.into_iter().map(|(i, _)| i).collect()
}

pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_generation() {
        let a = random_unit_vectors(10, 8, 42);
        let b = random_unit_vectors(10, 8, 42);
        assert_eq!(a, b);
    }

    #[test]
    fn unit_norm() {
        let vecs = random_unit_vectors(20, 16, 99);
        for v in &vecs {
            let norm = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm={norm}");
        }
    }

    #[test]
    fn ground_truth_returns_k() {
        let corpus = random_unit_vectors(100, 8, 1);
        let query = &corpus[0];
        let gt = ground_truth(query, &corpus, 10);
        assert_eq!(gt.len(), 10);
        assert_eq!(gt[0], 0);
    }
}
