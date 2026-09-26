//! Deterministic dataset generation (zero external dependencies).
//!
//! Uses an LCG PRNG and sum-of-uniforms for Gaussian approximation.
//! All seeds are fixed so benchmark numbers reproduce across runs.

/// Minimal LCG with period 2^64.
pub struct Lcg(u64);

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg(seed ^ 0x6c62272e07bb0142)
    }

    #[inline]
    pub fn next_u64(&mut self) -> u64 {
        self.0 = self
            .0
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        self.0
    }

    /// Uniform [0, 1).
    #[inline]
    pub fn next_f32(&mut self) -> f32 {
        (self.next_u64() >> 40) as f32 / (1u64 << 24) as f32
    }

    /// Approximate Normal(0, 1) via sum of 8 uniforms.
    #[inline]
    pub fn next_normal(&mut self) -> f32 {
        let s: f32 = (0..8).map(|_| self.next_f32()).sum();
        (s - 4.0) / 1.633
    }

    /// Uniform integer in [0, n).
    #[inline]
    pub fn next_usize(&mut self, n: usize) -> usize {
        (self.next_u64() % n as u64) as usize
    }
}

/// Generate clustered unit-sphere vectors.
///
/// Returns `(flat_f32_buffer, cluster_assignments)`. Each of the `n_clusters`
/// clusters has a random centroid on the unit sphere; points are perturbed by
/// Gaussian noise scaled by `std`.
pub fn clustered_vectors(
    n_clusters: usize,
    n_per_cluster: usize,
    dims: usize,
    std: f32,
    seed: u64,
) -> (Vec<f32>, Vec<usize>) {
    let mut rng = Lcg::new(seed);

    let centroids = gen_centroids(&mut rng, n_clusters, dims);
    let n = n_clusters * n_per_cluster;
    let mut vectors = Vec::with_capacity(n * dims);
    let mut assignments = Vec::with_capacity(n);

    for (ci, centroid) in centroids.iter().enumerate() {
        for _ in 0..n_per_cluster {
            for d in 0..dims {
                vectors.push(centroid[d] + rng.next_normal() * std);
            }
            assignments.push(ci);
        }
    }

    (vectors, assignments)
}

/// Generate query vectors using centroids derived from the SAME seed as the dataset.
///
/// This ensures queries are near the data's actual cluster centres, giving
/// realistic recall numbers.
pub fn clustered_queries(
    n_queries: usize,
    dims: usize,
    std: f32,
    data_seed: u64,
    query_seed: u64,
) -> (Vec<Vec<f32>>, Vec<usize>) {
    // Regenerate the SAME centroids used when building the dataset.
    let n_clusters = 8usize;
    let mut rng_cent = Lcg::new(data_seed);
    let centroids = gen_centroids(&mut rng_cent, n_clusters, dims);

    let half_std = std * 0.5;
    let mut rng = Lcg::new(query_seed);
    let mut queries = Vec::with_capacity(n_queries);
    let mut cluster_ids = Vec::with_capacity(n_queries);

    for i in 0..n_queries {
        let ci = i % n_clusters;
        let q: Vec<f32> = centroids[ci]
            .iter()
            .map(|&x| x + rng.next_normal() * half_std)
            .collect();
        queries.push(q);
        cluster_ids.push(ci);
    }

    (queries, cluster_ids)
}

/// Generate centroids deterministically from a seeded RNG.
fn gen_centroids(rng: &mut Lcg, n_clusters: usize, dims: usize) -> Vec<Vec<f32>> {
    (0..n_clusters)
        .map(|_| {
            let mut c: Vec<f32> = (0..dims).map(|_| rng.next_normal()).collect();
            let norm = c.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-9);
            c.iter_mut().for_each(|x| *x /= norm);
            c
        })
        .collect()
}

/// Brute-force ground truth: k nearest neighbours per query.
pub fn ground_truth(
    vectors: &[f32],
    queries: &[Vec<f32>],
    dims: usize,
    k: usize,
) -> Vec<Vec<u32>> {
    let n = vectors.len() / dims;
    queries
        .iter()
        .map(|q| {
            let mut dists: Vec<(f32, u32)> = (0..n)
                .map(|i| {
                    let v = &vectors[i * dims..(i + 1) * dims];
                    let d: f32 = q.iter().zip(v.iter()).map(|(a, b)| (a - b) * (a - b)).sum();
                    (d, i as u32)
                })
                .collect();
            dists.sort_unstable_by(|a, b| a.0.total_cmp(&b.0));
            dists.truncate(k);
            dists.into_iter().map(|(_, id)| id).collect()
        })
        .collect()
}
