//! Deterministic clustered dataset generation for benchmarking.
//!
//! Generates a mix of tight and loose Gaussian clusters in N-dim space using
//! a seeded LCG — no external entropy source, fully reproducible.

/// A generated dataset with per-vector cluster labels.
pub struct Dataset {
    /// Raw f32 vectors [N × dim].
    pub vectors: Vec<Vec<f32>>,
    /// Cluster label per vector (0..n_clusters).
    pub labels: Vec<usize>,
    /// Number of tight clusters.
    pub n_tight: usize,
    /// Number of loose clusters.
    pub n_loose: usize,
    pub dim: usize,
}

impl Dataset {
    pub fn n(&self) -> usize {
        self.vectors.len()
    }

    /// Vectors whose label maps to a tight cluster (label < n_tight).
    pub fn tight_indices(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l < self.n_tight)
            .map(|(i, _)| i)
            .collect()
    }

    /// Vectors whose label maps to a loose cluster (label >= n_tight).
    pub fn loose_indices(&self) -> Vec<usize> {
        self.labels
            .iter()
            .enumerate()
            .filter(|(_, &l)| l >= self.n_tight)
            .map(|(i, _)| i)
            .collect()
    }
}

/// Minimal seeded pseudo-random number generator (xorshift64).
struct Xsr64(u64);

impl Xsr64 {
    fn new(seed: u64) -> Self {
        Xsr64(seed | 1)
    }

    fn next_u64(&mut self) -> u64 {
        let mut x = self.0;
        x ^= x << 13;
        x ^= x >> 7;
        x ^= x << 17;
        self.0 = x;
        x
    }

    /// Sample from U(0,1).
    fn uniform(&mut self) -> f32 {
        self.next_u64() as f32 / u64::MAX as f32
    }

    /// Box-Muller transform: two U(0,1) → one N(0,1).
    fn normal(&mut self) -> f32 {
        let u1 = self.uniform().max(1e-12);
        let u2 = self.uniform();
        (-2.0 * u1.ln()).sqrt() * (2.0 * std::f32::consts::PI * u2).cos()
    }

    /// U(-bound, +bound).
    fn uniform_range(&mut self, bound: f32) -> f32 {
        (self.uniform() * 2.0 - 1.0) * bound
    }
}

/// Generate a clustered dataset for benchmarking adaptive SQ.
///
/// Parameters:
/// - `n_vectors`: total number of stored vectors
/// - `dim`: embedding dimension
/// - `n_tight`: number of tight (dense) clusters
/// - `n_loose`: number of loose (sparse) clusters
/// - `tight_frac`: fraction of vectors assigned to tight clusters
/// - `tight_sigma`: per-dimension std-dev for tight clusters
/// - `loose_sigma`: per-dimension std-dev for loose clusters
/// - `seed`: deterministic seed
pub fn generate(
    n_vectors: usize,
    dim: usize,
    n_tight: usize,
    n_loose: usize,
    tight_frac: f32,
    tight_sigma: f32,
    loose_sigma: f32,
    seed: u64,
) -> Dataset {
    let mut rng = Xsr64::new(seed);

    // Sample cluster centroids from N(0,1) to spread them across the space.
    let n_clusters = n_tight + n_loose;
    let centroids: Vec<Vec<f32>> = (0..n_clusters)
        .map(|_| (0..dim).map(|_| rng.normal() * 2.0).collect())
        .collect();

    let n_tight_vecs = (n_vectors as f32 * tight_frac).round() as usize;
    let n_loose_vecs = n_vectors - n_tight_vecs;

    let mut vectors = Vec::with_capacity(n_vectors);
    let mut labels = Vec::with_capacity(n_vectors);

    // Tight cluster vectors.
    for i in 0..n_tight_vecs {
        let c = i % n_tight;
        let v: Vec<f32> = centroids[c]
            .iter()
            .map(|&cx| cx + rng.normal() * tight_sigma)
            .collect();
        vectors.push(v);
        labels.push(c);
    }

    // Loose cluster vectors.
    for i in 0..n_loose_vecs {
        let c = n_tight + (i % n_loose);
        let v: Vec<f32> = centroids[c]
            .iter()
            .map(|&cx| cx + rng.normal() * loose_sigma)
            .collect();
        vectors.push(v);
        labels.push(c);
    }

    Dataset {
        vectors,
        labels,
        n_tight,
        n_loose,
        dim,
    }
}

/// Generate query vectors — one per requested cluster label.
///
/// Returns (queries, true_cluster_labels) where each query is drawn from
/// its assigned cluster centroid so we know the expected answer region.
pub fn generate_queries(
    base: &Dataset,
    n_queries: usize,
    sigma_scale: f32,
    seed: u64,
) -> Vec<Vec<f32>> {
    let mut rng = Xsr64::new(seed.wrapping_add(0xDEAD_BEEF));
    let n = base.n();
    (0..n_queries)
        .map(|_| {
            // Pick a random base vector as anchor.
            let anchor_idx = rng.next_u64() as usize % n;
            base.vectors[anchor_idx]
                .iter()
                .map(|&x| x + rng.normal() * sigma_scale)
                .collect()
        })
        .collect()
}

/// Brute-force ground truth: sorted list of (id, l2_sq) for a query.
pub fn ground_truth(query: &[f32], dataset: &[Vec<f32>]) -> Vec<(usize, f32)> {
    let mut dists: Vec<(usize, f32)> = dataset
        .iter()
        .enumerate()
        .map(|(i, v)| {
            let d: f32 = v
                .iter()
                .zip(query.iter())
                .map(|(a, b)| (a - b).powi(2))
                .sum();
            (i, d)
        })
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
    dists
}

/// Recall@k: fraction of true top-k IDs found in the candidate list.
pub fn recall_at_k(candidates: &[(usize, f32)], truth: &[(usize, f32)], k: usize) -> f32 {
    let k = k.min(candidates.len()).min(truth.len());
    let truth_ids: std::collections::HashSet<usize> =
        truth[..k].iter().map(|(id, _)| *id).collect();
    let found = candidates[..k]
        .iter()
        .filter(|(id, _)| truth_ids.contains(id))
        .count();
    found as f32 / k as f32
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn generate_correct_count() {
        let ds = generate(200, 32, 3, 5, 0.3, 0.02, 0.3, 42);
        assert_eq!(ds.n(), 200);
        assert_eq!(ds.labels.len(), 200);
        let tight = ds.tight_indices();
        // ~30% should be tight
        assert!(tight.len() >= 40 && tight.len() <= 80);
    }

    #[test]
    fn ground_truth_sorted() {
        let ds = generate(50, 8, 2, 2, 0.5, 0.05, 0.3, 7);
        let q = ds.vectors[0].clone();
        let gt = ground_truth(&q, &ds.vectors);
        assert_eq!(gt[0].0, 0, "Query itself should be the closest (d=0)");
        for w in gt.windows(2) {
            assert!(w[0].1 <= w[1].1, "Ground truth must be sorted");
        }
    }

    #[test]
    fn recall_at_k_perfect_when_correct() {
        let truth: Vec<(usize, f32)> = vec![(0, 0.0), (1, 1.0), (2, 2.0)];
        let candidates = truth.clone();
        let r = recall_at_k(&candidates, &truth, 3);
        assert!((r - 1.0).abs() < 1e-6, "Perfect recall should be 1.0");
    }
}
