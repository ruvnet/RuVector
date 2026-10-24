//! Deterministic synthetic agent-memory dataset: Gaussian clusters with a
//! controlled intra-cluster coherence gradient (points near a cluster's
//! centroid are "coherent"; points near the cluster boundary are not).
//!
//! This is a synthetic benchmark, not a claim about any production workload —
//! the research doc states this explicitly. Clustered-Gaussian corpora are a
//! standard, honest substitute for labeled ANN corpora (SIFT/GIST) when the
//! property under test (local coherence structure) needs to be controlled
//! and known at generation time, which no public ANN corpus labels.

use crate::rng::SplitMix64;

/// One synthetic corpus: flattened vectors, their cluster id, and the
/// centroid used to synthesize each cluster (useful for building queries).
pub struct Corpus {
    pub dim: usize,
    pub vectors: Vec<Vec<f32>>,
    pub cluster_id: Vec<u32>,
    pub centroids: Vec<Vec<f32>>,
}

/// Generate `n_clusters` Gaussian blobs of `per_cluster` points each in
/// `dim` dimensions. Cluster spread is randomized per cluster (dense vs.
/// diffuse clusters) so the corpus has real local-coherence variation
/// instead of being uniformly easy or uniformly hard.
pub fn generate_corpus(n_clusters: usize, per_cluster: usize, dim: usize, seed: u64) -> Corpus {
    let mut rng = SplitMix64(seed);
    let mut centroids = Vec::with_capacity(n_clusters);
    let mut vectors = Vec::with_capacity(n_clusters * per_cluster);
    let mut cluster_id = Vec::with_capacity(n_clusters * per_cluster);

    for c in 0..n_clusters {
        let centroid: Vec<f32> = rng.gaussian_vec(dim).iter().map(|x| x * 3.0).collect();
        // Spread varies 0.3x-1.1x across clusters: some tight, some diffuse.
        let spread = 0.3 + 0.8 * rng.next_unit();
        for _ in 0..per_cluster {
            let noise = rng.gaussian_vec(dim);
            let v: Vec<f32> = centroid
                .iter()
                .zip(noise.iter())
                .map(|(&cv, &n)| cv + n * spread)
                .collect();
            vectors.push(v);
            cluster_id.push(c as u32);
        }
        centroids.push(centroid);
    }

    Corpus {
        dim,
        vectors,
        cluster_id,
        centroids,
    }
}

/// Query workload: for each query, perturb a randomly chosen corpus point by
/// small Gaussian noise — the standard "retrieve the memory most like this
/// new observation" access pattern for agent memory, not a copy of a corpus
/// vector (which would make exact-match trivial).
pub fn generate_queries(corpus: &Corpus, n_queries: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut rng = SplitMix64(seed);
    let n = corpus.vectors.len();
    (0..n_queries)
        .map(|_| {
            let idx = rng.next_u64() as usize % n;
            let base = &corpus.vectors[idx];
            let noise = rng.gaussian_vec(corpus.dim);
            base.iter()
                .zip(noise.iter())
                .map(|(&b, &n)| b + n * 0.15)
                .collect()
        })
        .collect()
}
