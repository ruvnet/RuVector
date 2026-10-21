//! Deterministic dataset generation: a simple LCG-seeded Gaussian-like
//! corpus and query set, plus brute-force ground truth. Mirrors the
//! generation approach used by `ruvector-speculative-ann` so the two crates'
//! numbers are produced under comparable conditions.

use crate::{sq_l2_f32, Hit};
use rand::rngs::StdRng;
use rand::SeedableRng;
use rand_distr::{Distribution, Normal, Uniform};

/// Minimal splitmix64-style generator: deterministic, dependency-free.
pub struct Lcg {
    state: u64,
}

impl Lcg {
    pub fn new(seed: u64) -> Self {
        Lcg { state: seed }
    }

    #[inline]
    fn next_u64(&mut self) -> u64 {
        self.state = self.state.wrapping_add(0x9E3779B97F4A7C15);
        let mut z = self.state;
        z = (z ^ (z >> 30)).wrapping_mul(0xBF58476D1CE4E5B9);
        z = (z ^ (z >> 27)).wrapping_mul(0x94D049BB133111EB);
        z ^ (z >> 31)
    }

    /// Approximately standard-normal via a 12-uniform sum (Irwin-Hall), fast
    /// and deterministic without external distribution crates in the hot
    /// generation path.
    pub fn next_gaussian(&mut self) -> f32 {
        let mut sum = 0.0f64;
        for _ in 0..12 {
            sum += (self.next_u64() >> 11) as f64 / (1u64 << 53) as f64;
        }
        (sum - 6.0) as f32
    }
}

/// Flat row-major corpus: `n` vectors of `dims` dimensions each.
pub struct Dataset {
    pub corpus: Vec<f32>,
    pub queries: Vec<f32>,
    pub n: usize,
    pub dims: usize,
    pub n_queries: usize,
}

impl Dataset {
    pub fn generate(n: usize, dims: usize, n_queries: usize, seed: u64) -> Self {
        let mut rng = Lcg::new(seed);
        let corpus: Vec<f32> = (0..n * dims).map(|_| rng.next_gaussian()).collect();
        let queries: Vec<f32> = (0..n_queries * dims).map(|_| rng.next_gaussian()).collect();
        Dataset {
            corpus,
            queries,
            n,
            dims,
            n_queries,
        }
    }

    pub fn corpus_vectors(&self) -> Vec<Vec<f32>> {
        (0..self.n)
            .map(|i| self.corpus[i * self.dims..(i + 1) * self.dims].to_vec())
            .collect()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.corpus[i * self.dims..(i + 1) * self.dims]
    }

    pub fn query(&self, i: usize) -> &[f32] {
        &self.queries[i * self.dims..(i + 1) * self.dims]
    }

    /// Exact top-k ground truth for a query via brute-force f32 scan.
    pub fn ground_truth(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = (0..self.n)
            .map(|i| Hit {
                id: i,
                dist: sq_l2_f32(query, self.row(i)),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }
}

/// Clustered benchmark dataset: `n_clusters` Gaussian blobs on the unit
/// sphere, `n_per_cluster` points each. Mirrors `ruvector-coherence-hnsw`'s
/// `clustered_unit_vectors` methodology — the traversal-coherence gate's
/// premise (an "on-path toward the query" direction exists) only holds when
/// the corpus has real cluster structure, so a fair test of the gate must use
/// clustered rather than isotropic data.
pub struct ClusteredDataset {
    pub corpus: Vec<f32>,
    pub queries: Vec<Vec<f32>>,
    pub n: usize,
    pub dims: usize,
}

impl ClusteredDataset {
    pub fn generate(
        n_clusters: usize,
        n_per_cluster: usize,
        dims: usize,
        std_dev: f32,
        n_queries: usize,
        seed: u64,
    ) -> Self {
        let mut rng = StdRng::seed_from_u64(seed);
        let normal_center = Normal::new(0.0f32, 1.0).expect("normal");
        let noise = Normal::new(0.0f32, std_dev).expect("noise normal");

        let centers: Vec<Vec<f32>> = (0..n_clusters)
            .map(|_| {
                let mut v: Vec<f32> = (0..dims).map(|_| normal_center.sample(&mut rng)).collect();
                normalize(&mut v);
                v
            })
            .collect();

        let n = n_clusters * n_per_cluster;
        let mut corpus = Vec::with_capacity(n * dims);
        for center in &centers {
            for _ in 0..n_per_cluster {
                let mut v: Vec<f32> = center.iter().map(|&c| c + noise.sample(&mut rng)).collect();
                normalize(&mut v);
                corpus.extend_from_slice(&v);
            }
        }

        let query_noise = Normal::new(0.0f32, std_dev * 0.5).expect("noise normal");
        let cluster_pick = Uniform::new(0usize, n_clusters);
        let queries: Vec<Vec<f32>> = (0..n_queries)
            .map(|_| {
                let ci = cluster_pick.sample(&mut rng);
                let mut v: Vec<f32> = centers[ci]
                    .iter()
                    .map(|&c| c + query_noise.sample(&mut rng))
                    .collect();
                normalize(&mut v);
                v
            })
            .collect();

        ClusteredDataset {
            corpus,
            queries,
            n,
            dims,
        }
    }

    pub fn corpus_vectors(&self) -> Vec<Vec<f32>> {
        (0..self.n)
            .map(|i| self.corpus[i * self.dims..(i + 1) * self.dims].to_vec())
            .collect()
    }

    pub fn row(&self, i: usize) -> &[f32] {
        &self.corpus[i * self.dims..(i + 1) * self.dims]
    }

    pub fn ground_truth(&self, query: &[f32], k: usize) -> Vec<Hit> {
        let mut hits: Vec<Hit> = (0..self.n)
            .map(|i| Hit {
                id: i,
                dist: sq_l2_f32(query, self.row(i)),
            })
            .collect();
        hits.sort_unstable_by(|a, b| a.dist.partial_cmp(&b.dist).unwrap());
        hits.truncate(k);
        hits
    }
}

fn normalize(v: &mut [f32]) {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-8 {
        for x in v.iter_mut() {
            *x /= norm;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn deterministic_for_same_seed() {
        let a = Dataset::generate(10, 4, 2, 42);
        let b = Dataset::generate(10, 4, 2, 42);
        assert_eq!(a.corpus, b.corpus);
        assert_eq!(a.queries, b.queries);
    }

    #[test]
    fn different_seeds_differ() {
        let a = Dataset::generate(10, 4, 2, 1);
        let b = Dataset::generate(10, 4, 2, 2);
        assert_ne!(a.corpus, b.corpus);
    }

    #[test]
    fn ground_truth_is_sorted_and_sized() {
        let ds = Dataset::generate(200, 8, 1, 7);
        let gt = ds.ground_truth(ds.query(0), 10);
        assert_eq!(gt.len(), 10);
        for w in gt.windows(2) {
            assert!(w[0].dist <= w[1].dist);
        }
    }
}
