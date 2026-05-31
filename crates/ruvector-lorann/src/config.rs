/// Tunable hyper-parameters for a `LorannIndex`.
///
/// Defaults are calibrated for high-dimensional embeddings (d ≈ 768–1536)
/// at a corpus size of ≈ 100 K vectors. Tune `n_clusters`, `rank`, and
/// `n_probe` to navigate the recall–QPS Pareto frontier.
#[derive(Debug, Clone)]
pub struct LorannConfig {
    /// Number of IVF clusters (≈ √n is a safe default).
    pub n_clusters: usize,

    /// Rank r of the per-cluster SVD approximation.
    /// Higher rank → better recall, slower query. r=32 is the paper's default.
    pub rank: usize,

    /// Number of clusters probed per query.
    /// Larger → better recall, more work. n_probe=8 gives ≈80% recall.
    pub n_probe: usize,

    /// After approximate scoring, keep this many candidates for exact rerank.
    /// Oversampling relative to k; the paper uses candidate_set ≈ 20k.
    pub candidate_set: usize,

    /// Max k-means iterations.
    pub kmeans_max_iter: usize,

    /// Random seed for k-means initialisation and reproducibility.
    pub seed: u64,
}

impl Default for LorannConfig {
    fn default() -> Self {
        Self {
            n_clusters: 128,
            rank: 32,
            n_probe: 8,
            candidate_set: 200,
            kmeans_max_iter: 20,
            seed: 42,
        }
    }
}

impl LorannConfig {
    /// Create a config tuned for a corpus of size `n`.
    pub fn for_corpus(n: usize) -> Self {
        let n_clusters = ((n as f64).sqrt().round() as usize).clamp(16, 4096);
        Self {
            n_clusters,
            ..Default::default()
        }
    }
}
