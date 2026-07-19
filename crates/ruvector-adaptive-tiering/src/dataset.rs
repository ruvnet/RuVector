//! Deterministic synthetic dataset generation.
//!
//! Produces a labelled set of 64-dimensional vectors from three clusters:
//!
//! | Label | Name | Cluster characteristics |
//! |-------|------|------------------------|
//! | 0 | Important | Tight (σ=0.05), high intra-cluster coherence |
//! | 1 | Moderate  | Medium spread (σ=0.25), moderate coherence |
//! | 2 | Noise     | Wide spread (σ=1.20), low coherence |

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

/// Generate a dataset of `n_total` vectors of dimension `dims`.
///
/// Cluster sizes:
/// * Important (label 0): 10 % of n_total (min 1).
/// * Moderate  (label 1): 30 % of n_total (min 1).
/// * Noise     (label 2): remainder.
///
/// Returns `(vectors, labels)` where `labels[i]` is the cluster index.
pub fn generate_clustered(
    n_total: usize,
    dims: usize,
    rng: &mut impl rand::Rng,
) -> (Vec<Vec<f32>>, Vec<u8>) {
    let n_important = (n_total / 10).max(1);
    let n_moderate = (n_total * 3 / 10).max(1);
    let n_noise = n_total.saturating_sub(n_important + n_moderate);

    let mut vecs: Vec<Vec<f32>> = Vec::with_capacity(n_total);
    let mut labels: Vec<u8> = Vec::with_capacity(n_total);

    // Centroid for important cluster: all-ones direction (normalised below).
    let centroid_val = 1.0_f32 / (dims as f32).sqrt();

    // Important cluster: tight σ=0.05
    let tight = Normal::new(0.0_f32, 0.05).unwrap();
    for _ in 0..n_important {
        let v: Vec<f32> = (0..dims)
            .map(|_| centroid_val + tight.sample(rng))
            .collect();
        vecs.push(v);
        labels.push(0);
    }

    // Moderate cluster: centroid at opposite direction, σ=0.25
    let moderate = Normal::new(0.0_f32, 0.25).unwrap();
    for _ in 0..n_moderate {
        let v: Vec<f32> = (0..dims)
            .map(|_| -centroid_val + moderate.sample(rng))
            .collect();
        vecs.push(v);
        labels.push(1);
    }

    // Noise cluster: centroid at zero, σ=1.20
    let noise_dist = Normal::new(0.0_f32, 1.20).unwrap();
    for _ in 0..n_noise {
        let v: Vec<f32> = (0..dims).map(|_| noise_dist.sample(rng)).collect();
        vecs.push(v);
        labels.push(2);
    }

    (vecs, labels)
}

/// Representative query vector for the given cluster label.
///
/// Returns a centroid-like query that should retrieve members of that cluster.
pub fn cluster_query(label: u8, dims: usize) -> Vec<f32> {
    let centroid_val = 1.0_f32 / (dims as f32).sqrt();
    (0..dims)
        .map(|_| match label {
            0 => centroid_val,
            1 => -centroid_val,
            _ => 0.0_f32,
        })
        .collect()
}

/// Convenience: seed a StdRng from a fixed u64.
pub fn seeded_rng(seed: u64) -> rand::rngs::StdRng {
    rand::rngs::StdRng::seed_from_u64(seed)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn label_counts_are_correct() {
        let mut rng = seeded_rng(1);
        let n = 100;
        let (vecs, labels) = generate_clustered(n, 8, &mut rng);
        assert_eq!(vecs.len(), n);
        assert_eq!(labels.len(), n);
        let n0 = labels.iter().filter(|&&l| l == 0).count();
        let n1 = labels.iter().filter(|&&l| l == 1).count();
        let n2 = labels.iter().filter(|&&l| l == 2).count();
        assert_eq!(n0 + n1 + n2, n);
        assert!(n0 > 0 && n1 > 0 && n2 > 0);
    }

    #[test]
    fn important_cluster_is_tighter_than_noise() {
        let mut rng = seeded_rng(2);
        let (vecs, labels) = generate_clustered(200, 16, &mut rng);

        let important: Vec<&Vec<f32>> = vecs
            .iter()
            .zip(labels.iter())
            .filter(|(_, &l)| l == 0)
            .map(|(v, _)| v)
            .collect();
        let noise: Vec<&Vec<f32>> = vecs
            .iter()
            .zip(labels.iter())
            .filter(|(_, &l)| l == 2)
            .map(|(v, _)| v)
            .collect();

        let mean_l2 = |group: &[&Vec<f32>]| -> f32 {
            if group.len() < 2 {
                return 0.0;
            }
            let mut total = 0.0;
            let mut cnt = 0;
            for i in 0..group.len().min(10) {
                for j in (i + 1)..group.len().min(10) {
                    total += crate::distance::l2_sq(group[i], group[j]).sqrt();
                    cnt += 1;
                }
            }
            if cnt == 0 {
                0.0
            } else {
                total / cnt as f32
            }
        };

        let imp_spread = mean_l2(&important);
        let noise_spread = mean_l2(&noise);
        assert!(
            imp_spread < noise_spread,
            "important cluster ({imp_spread:.3}) must be tighter than noise ({noise_spread:.3})"
        );
    }
}
