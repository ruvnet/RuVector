//! Deterministic synthetic dataset generator simulating Matryoshka embedding structure.
//!
//! **Matryoshka property**: the first `signal_dim` elements carry the dominant
//! semantic signal; later elements add fine-grained discriminative information.
//! This matches how MRL-trained models (e.g., OpenAI text-embedding-3, Nomic-Embed)
//! encode meaning: prefix truncation is lossy but approximately rank-preserving.
//!
//! Generation strategy:
//! - Draw `num_clusters` cluster centres in `signal_dim`-space.
//! - Assign each vector to a cluster deterministically.
//! - Signal dims: centre + small noise (scale 0.08) → strong coarse similarity.
//! - Remaining dims: moderate noise (scale 0.25) → fine-grained full-dim similarity.
//! - L2-normalise every vector before returning.

use crate::l2_normalize;

/// Simple LCG PRNG; returns a value in [0, 1).
fn lcg_f32(state: &mut u64) -> f32 {
    *state = state
        .wrapping_mul(6_364_136_223_846_793_005)
        .wrapping_add(1_442_695_040_888_963_407);
    ((*state >> 33) as f32) / ((1u64 << 31) as f32)
}

/// Returns a value uniformly in [-1, 1).
fn lcg_uniform(state: &mut u64) -> f32 {
    lcg_f32(state) * 2.0 - 1.0
}

/// Generate `n` Matryoshka-structured vectors and `n_queries` query vectors.
///
/// All vectors are L2-normalised and stored at `full_dim` dimensions.
/// The first `signal_dim` dimensions carry the dominant cluster signal;
/// the remainder carry discriminative noise.
pub fn generate_matryoshka_dataset(
    n: usize,
    n_queries: usize,
    full_dim: usize,
    signal_dim: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<Vec<f32>>) {
    let mut rng = seed;
    let num_clusters: usize = (n / 10).clamp(10, 100);

    // Draw cluster centres in the signal subspace.
    let centres: Vec<Vec<f32>> = (0..num_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..signal_dim).map(|_| lcg_uniform(&mut rng)).collect();
            l2_normalize(&mut c);
            c
        })
        .collect();

    let make_vec = |rng: &mut u64, cluster: usize| -> Vec<f32> {
        let centre = &centres[cluster % centres.len()];
        let mut v: Vec<f32> = Vec::with_capacity(full_dim);

        // Signal dims: cluster centre + tight noise.
        for &c in centre {
            v.push(c + lcg_uniform(rng) * 0.08);
        }
        // Remaining dims: moderate noise (no cluster structure).
        for _ in signal_dim..full_dim {
            v.push(lcg_uniform(rng) * 0.25);
        }

        l2_normalize(&mut v);
        v
    };

    let vectors: Vec<Vec<f32>> = (0..n).map(|i| make_vec(&mut rng, i)).collect();
    let queries: Vec<Vec<f32>> = (0..n_queries).map(|i| make_vec(&mut rng, i * 3)).collect();

    (vectors, queries)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_is_normalised() {
        let (vecs, queries) = generate_matryoshka_dataset(50, 5, 128, 32, 999);
        for v in vecs.iter().chain(queries.iter()) {
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            assert!(
                (norm - 1.0).abs() < 1e-4,
                "vector not unit-normalised: norm={norm}"
            );
        }
    }

    #[test]
    fn dataset_is_deterministic() {
        let (a, _) = generate_matryoshka_dataset(10, 3, 64, 16, 12345);
        let (b, _) = generate_matryoshka_dataset(10, 3, 64, 16, 12345);
        for (av, bv) in a.iter().zip(b.iter()) {
            for (ai, bi) in av.iter().zip(bv.iter()) {
                assert!((ai - bi).abs() < 1e-8);
            }
        }
    }

    #[test]
    fn prefix_captures_cluster_structure() {
        // Verify that truncating to signal_dim does not scramble nearest neighbours
        // (coarse-dim recall should be > 0.5 over brute-force full-dim ground truth).
        use crate::{brute_force_knn, recall_at_k};
        let (vecs, queries) = generate_matryoshka_dataset(200, 10, 128, 32, 77777);
        let mut coarse_recall_sum = 0.0f32;
        for q in &queries {
            let gt_full = brute_force_knn(&vecs, q, 10, 128);
            let gt_coarse = brute_force_knn(&vecs, q, 50, 32);
            coarse_recall_sum += recall_at_k(&gt_coarse, &gt_full);
        }
        let avg = coarse_recall_sum / queries.len() as f32;
        assert!(
            avg >= 0.40,
            "coarse-dim recall@10 = {:.3} < 0.40; dataset may lack signal",
            avg
        );
    }
}
