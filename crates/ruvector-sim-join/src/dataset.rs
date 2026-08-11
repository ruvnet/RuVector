//! Deterministic clustered dataset for similarity-join benchmarks.
//!
//! Generates two sets A and B of vectors drawn from K Gaussian clusters.
//! Vectors in the same cluster are semantically similar; vectors from
//! different clusters are dissimilar. The similarity threshold is set
//! so that intra-cluster pairs are discovered and inter-cluster pairs
//! are not.

use crate::l2_normalize;

/// A pair of vector sets with a known similarity threshold.
pub struct ClusteredDataset {
    /// Set A: n_per_set vectors of `dims` dimensions (L2-normalised).
    pub a: Vec<Vec<f32>>,
    /// Set B: n_per_set vectors of `dims` dimensions (L2-normalised).
    pub b: Vec<Vec<f32>>,
    /// Cosine similarity threshold for this dataset.
    pub threshold: f32,
    /// Number of clusters used to generate the data.
    pub n_clusters: usize,
}

impl ClusteredDataset {
    /// Generate a reproducible clustered dataset.
    ///
    /// # Parameters
    /// - `n_per_set`: Number of vectors per set (A and B).
    /// - `dims`: Vector dimensionality.
    /// - `n_clusters`: Number of semantic clusters.
    /// - `noise`: Std-dev of Gaussian noise added to each vector.
    /// - `seed`: Deterministic seed.
    pub fn generate(
        n_per_set: usize,
        dims: usize,
        n_clusters: usize,
        noise: f32,
        seed: u64,
    ) -> Self {
        let mut rng = Lcg64::new(seed);

        // Generate cluster centroids (L2-normalised unit vectors)
        let centroids: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                let mut v: Vec<f32> = (0..dims).map(|_| rng.randn()).collect();
                l2_normalize(&mut v);
                v
            })
            .collect();

        let make_set = |set_seed: u64| -> Vec<Vec<f32>> {
            let mut r = Lcg64::new(set_seed);
            (0..n_per_set)
                .map(|i| {
                    let cluster_idx = i % n_clusters;
                    let centroid = &centroids[cluster_idx];
                    let mut v: Vec<f32> = centroid.iter().map(|c| c + noise * r.randn()).collect();
                    l2_normalize(&mut v);
                    v
                })
                .collect()
        };

        let a = make_set(seed.wrapping_add(1_000_000));
        let b = make_set(seed.wrapping_add(2_000_000));

        // Compute a data-driven threshold from actual intra-cluster cosines.
        // Sample pairs from the same cluster and set threshold at the 20th percentile,
        // guaranteeing real GT pairs regardless of noise level or dimensionality.
        let threshold = data_driven_threshold(&a, &b, n_clusters);

        Self {
            a,
            b,
            threshold,
            n_clusters,
        }
    }
}

/// Sample intra-cluster pairs from A×B and return their 20th-percentile cosine similarity.
fn data_driven_threshold(a: &[Vec<f32>], b: &[Vec<f32>], n_clusters: usize) -> f32 {
    use crate::cosine_similarity;
    // Collect up to 200 intra-cluster similarities
    let mut sims: Vec<f32> = Vec::with_capacity(200);
    'outer: for cluster in 0..n_clusters {
        for a_mult in 0..4 {
            let ai = cluster + a_mult * n_clusters;
            if ai >= a.len() {
                break;
            }
            for b_mult in 0..4 {
                let bi = cluster + b_mult * n_clusters;
                if bi >= b.len() {
                    break;
                }
                sims.push(cosine_similarity(&a[ai], &b[bi]));
                if sims.len() >= 200 {
                    break 'outer;
                }
            }
        }
    }
    if sims.is_empty() {
        return 0.50;
    }
    sims.sort_by(|x, y| x.partial_cmp(y).unwrap());
    // 20th percentile: captures the majority of intra-cluster pairs
    let idx = (sims.len() as f32 * 0.20) as usize;
    (sims[idx] - 0.02).max(0.01)
}

impl ClusteredDataset {
    /// Total memory of both sets in bytes.
    pub fn memory_bytes(&self) -> usize {
        let dims = self.a.first().map(|v| v.len()).unwrap_or(0);
        (self.a.len() + self.b.len()) * dims * std::mem::size_of::<f32>()
    }
}

/// Minimal deterministic LCG PRNG with Box-Muller normal sampling.
pub struct Lcg64 {
    state: u64,
}

impl Lcg64 {
    pub fn new(seed: u64) -> Self {
        Self {
            state: seed ^ 0x6c62272e07bb0142,
        }
    }

    pub fn next_u64(&mut self) -> u64 {
        // LCG: state = a*state + c (mod 2^64)
        self.state = self
            .state
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        self.state
    }

    pub fn rand_f32(&mut self) -> f32 {
        (self.next_u64() >> 11) as f32 / (1u64 << 53) as f32
    }

    /// Box-Muller normal sample (returns one of two; discards the other).
    pub fn randn(&mut self) -> f32 {
        let u1 = (self.rand_f32() + 1e-9).max(1e-9);
        let u2 = self.rand_f32() * 2.0 * std::f32::consts::PI;
        (-2.0 * u1.ln()).sqrt() * u2.cos()
    }

    pub fn rand_usize(&mut self, max: usize) -> usize {
        (self.next_u64() % max as u64) as usize
    }
}
