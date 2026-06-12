use rand::rngs::SmallRng;
/// Deterministic pseudorandom Gaussian dataset for benchmarking.
///
/// Generates N vectors with D dimensions. The space is divided into
/// `n_signal_dims` "signal" dimensions (K-cluster Gaussian structure)
/// and the remaining "noise" dimensions (pure isotropic Gaussian).
/// This lets us test whether coherence fusion correctly up-weights
/// the informative subspaces.
use rand::{Rng, SeedableRng};

/// Generate a clustered dataset.
///
/// * `n`              – number of vectors
/// * `dim`            – total dimensions
/// * `n_clusters`     – number of Gaussian clusters
/// * `n_signal_dims`  – leading dimensions that encode cluster membership;
///                      the remaining dims are pure noise
/// * `seed`           – reproducibility seed
pub fn generate_clustered(
    n: usize,
    dim: usize,
    n_clusters: usize,
    n_signal_dims: usize,
    seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize> /* cluster labels */) {
    assert!(n_signal_dims <= dim);
    let mut rng = SmallRng::seed_from_u64(seed);

    // Build cluster centres in signal space.
    let centres: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| {
            (0..n_signal_dims)
                .map(|_| rng.gen_range(-4.0_f32..4.0))
                .collect()
        })
        .collect();

    let mut vectors = Vec::with_capacity(n);
    let mut labels = Vec::with_capacity(n);

    for i in 0..n {
        let cluster = i % n_clusters;
        let c = &centres[cluster];
        let mut v = Vec::with_capacity(dim);

        // Signal dims: Gaussian around cluster centre (σ = 0.4).
        for d in 0..n_signal_dims {
            v.push(c[d] + sample_gaussian(&mut rng) * 0.4);
        }
        // Noise dims: isotropic Gaussian (σ = 1.0).
        for _ in n_signal_dims..dim {
            v.push(sample_gaussian(&mut rng));
        }

        vectors.push(v);
        labels.push(cluster);
    }

    (vectors, labels)
}

/// Generate `n_queries` random query vectors (uniform in [-3, 3]).
pub fn generate_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SmallRng::seed_from_u64(seed.wrapping_add(0xdead_beef));
    (0..n)
        .map(|_| (0..dim).map(|_| rng.gen_range(-3.0_f32..3.0)).collect())
        .collect()
}

/// Box-Muller transform for N(0,1) samples.
fn sample_gaussian(rng: &mut SmallRng) -> f32 {
    let u1: f32 = rng.gen_range(f32::EPSILON..1.0);
    let u2: f32 = rng.gen::<f32>();
    let r = (-2.0_f32 * u1.ln()).sqrt();
    let theta = 2.0 * std::f32::consts::PI * u2;
    r * theta.cos()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn dataset_shapes() {
        let (vecs, labels) = generate_clustered(1000, 64, 10, 32, 42);
        assert_eq!(vecs.len(), 1000);
        assert_eq!(labels.len(), 1000);
        assert_eq!(vecs[0].len(), 64);
    }

    #[test]
    fn queries_shape() {
        let q = generate_queries(50, 64, 99);
        assert_eq!(q.len(), 50);
        assert_eq!(q[0].len(), 64);
    }

    #[test]
    fn cluster_labels_in_range() {
        let (_, labels) = generate_clustered(200, 32, 5, 16, 7);
        assert!(labels.iter().all(|&l| l < 5));
    }
}
