use rand::rngs::StdRng;
use rand::{Rng, SeedableRng};
use rand_distr::{Distribution, Normal};

/// Generate a deterministic Gaussian dataset with optional cluster structure.
pub fn generate_clustered(n: usize, dim: usize, n_clusters: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed);
    let normal = Normal::new(0.0f32, 1.0).unwrap();

    // Generate cluster centroids
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) * 10.0).collect())
        .collect();

    // Assign vectors to clusters with small noise
    (0..n)
        .map(|i| {
            let c = i % n_clusters;
            (0..dim)
                .map(|d| centroids[c][d] + normal.sample(&mut rng) * 1.5)
                .collect()
        })
        .collect()
}

/// Generate deterministic random query vectors.
pub fn generate_queries(n: usize, dim: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(99999));
    let normal = Normal::new(0.0f32, 1.0).unwrap();
    (0..n)
        .map(|_| (0..dim).map(|_| normal.sample(&mut rng) * 10.0).collect())
        .collect()
}

/// Brute-force ground truth: returns top-k indices for each query.
pub fn brute_force_knn(data: &[Vec<f32>], queries: &[Vec<f32>], k: usize) -> Vec<Vec<usize>> {
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, usize)> = data
                .iter()
                .enumerate()
                .map(|(i, v)| (l2_sq(q, v), i))
                .collect();
            dists.sort_by(|a, b| a.0.partial_cmp(&b.0).unwrap());
            dists[..k].iter().map(|(_, i)| *i).collect()
        })
        .collect()
}

pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

pub fn simulate_access_freq(n: usize, seed: u64) -> Vec<usize> {
    // Zipf-like: most accesses go to a small fraction of vectors
    let mut rng = StdRng::seed_from_u64(seed.wrapping_add(77777));
    (0..n)
        .map(|i| {
            // Higher rank (lower i) gets more access
            let rank = (i + 1) as f32;
            let base = (n as f32 / rank).ceil() as usize;
            base + rng.gen_range(0..5)
        })
        .collect()
}
