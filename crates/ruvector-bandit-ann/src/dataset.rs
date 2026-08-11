//! Deterministic dataset generation for benchmarks and tests.
//!
//! Uses a fixed LCG RNG so benchmarks are reproducible without an external seed file.

use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};

/// Generate `n` random unit-sphere vectors of dimension `dim`.
/// Seed is deterministic so results are reproducible.
pub fn random_unit_vectors(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    (0..n)
        .map(|_| {
            let v: Vec<f32> = (0..dim).map(|_| rng.gen::<f32>() * 2.0 - 1.0).collect();
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            v.into_iter().map(|x| x / norm).collect()
        })
        .collect()
}

/// Generate `n_queries` query vectors with moderate overlap with the data distribution.
pub fn random_queries(n_queries: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    random_unit_vectors(n_queries, dim, seed.wrapping_add(0xDEAD_BEEF))
}

/// Brute-force ground truth: return top-k indices (sorted by ascending sq_l2 distance).
pub fn ground_truth(data: &[Vec<f32>], query: &[f32], k: usize) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = data
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b) * (a - b))
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists.truncate(k);
    dists
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn vectors_are_unit_norm() {
        let vecs = random_unit_vectors(50, 32, 42);
        for v in &vecs {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!((norm - 1.0).abs() < 1e-5, "norm = {}", norm);
        }
    }

    #[test]
    fn ground_truth_sorted_ascending() {
        let data = random_unit_vectors(100, 16, 1);
        let q = random_queries(1, 16, 1);
        let gt = ground_truth(&data, &q[0], 10);
        for w in gt.windows(2) {
            assert!(w[0].1 <= w[1].1);
        }
    }

    #[test]
    fn ground_truth_exact_hit() {
        let mut data = random_unit_vectors(50, 8, 5);
        let target = vec![1.0_f32; 8];
        // Normalize.
        let norm: f32 = (8.0_f32).sqrt();
        let target: Vec<f32> = target.into_iter().map(|x| x / norm).collect();
        data.push(target.clone()); // id = 50
        let gt = ground_truth(&data, &target, 1);
        assert_eq!(gt[0].0, 50, "ground truth should return exact match");
        assert!(gt[0].1 < 1e-9);
    }
}
