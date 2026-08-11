//! Deterministic dataset generation with optional distribution shift.
//!
//! Phase A: Gaussian clusters centred around the origin.
//! Phase B: Same clusters shifted by `shift` along every dimension.
//! The shift simulates topic drift in agent memory embeddings.

use rand::SeedableRng;
use rand_distr::{Distribution, Normal};

pub struct DatasetConfig {
    pub dims: usize,
    pub clusters: usize,
    pub cluster_std: f32,
    /// Shift applied to Phase-B vectors along every dimension.
    pub shift: f32,
    pub seed: u64,
}

impl Default for DatasetConfig {
    fn default() -> Self {
        Self {
            dims: 64,
            clusters: 8,
            cluster_std: 0.5,
            shift: 3.0,
            seed: 42,
        }
    }
}

/// Generate `n` vectors from Phase-A distribution (unshifted Gaussian clusters).
pub fn generate_phase_a(n: usize, cfg: &DatasetConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed);
    let normal = Normal::new(0.0_f32, cfg.cluster_std).unwrap();
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        // cycle through clusters, fixed centroid per cluster index
        let cluster = i % cfg.clusters;
        let centroid = cluster_centroid(cluster, cfg.dims, cfg.clusters);
        let v: Vec<f32> = centroid
            .iter()
            .map(|&c| c + normal.sample(&mut rng))
            .collect();
        vecs.push(v);
    }
    vecs
}

/// Generate `n` vectors from Phase-B distribution (shifted by `cfg.shift`).
pub fn generate_phase_b(n: usize, cfg: &DatasetConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed + 1);
    let normal = Normal::new(0.0_f32, cfg.cluster_std).unwrap();
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let cluster = i % cfg.clusters;
        let centroid = cluster_centroid(cluster, cfg.dims, cfg.clusters);
        let v: Vec<f32> = centroid
            .iter()
            .map(|&c| c + cfg.shift + normal.sample(&mut rng))
            .collect();
        vecs.push(v);
    }
    vecs
}

/// Generate `n` query vectors from Phase-A distribution (different seed).
pub fn generate_queries_a(n: usize, cfg: &DatasetConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed + 100);
    let normal = Normal::new(0.0_f32, cfg.cluster_std).unwrap();
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let cluster = i % cfg.clusters;
        let centroid = cluster_centroid(cluster, cfg.dims, cfg.clusters);
        let v: Vec<f32> = centroid
            .iter()
            .map(|&c| c + normal.sample(&mut rng))
            .collect();
        vecs.push(v);
    }
    vecs
}

/// Generate `n` query vectors from Phase-B distribution (different seed).
pub fn generate_queries_b(n: usize, cfg: &DatasetConfig) -> Vec<Vec<f32>> {
    let mut rng = rand::rngs::StdRng::seed_from_u64(cfg.seed + 200);
    let normal = Normal::new(0.0_f32, cfg.cluster_std).unwrap();
    let mut vecs = Vec::with_capacity(n);
    for i in 0..n {
        let cluster = i % cfg.clusters;
        let centroid = cluster_centroid(cluster, cfg.dims, cfg.clusters);
        let v: Vec<f32> = centroid
            .iter()
            .map(|&c| c + cfg.shift + normal.sample(&mut rng))
            .collect();
        vecs.push(v);
    }
    vecs
}

/// Fixed centroid for cluster `c` in `dims` dimensions across `num_clusters`.
/// Spreads centroids uniformly so codebook training can distinguish them.
fn cluster_centroid(c: usize, dims: usize, num_clusters: usize) -> Vec<f32> {
    // Spread clusters along the first dimension so they are clearly separated.
    let spacing = 4.0_f32;
    (0..dims)
        .map(|d| {
            if d == 0 {
                c as f32 * spacing
            } else {
                // Small fixed offset per cluster to break symmetry
                ((c * 7 + d * 3) % num_clusters) as f32 * 0.3
            }
        })
        .collect()
}
