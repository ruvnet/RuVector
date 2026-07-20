//! Three IVF index variants sharing the same `AnnIndex` trait.
//!
//! All three variants expose an identical build/search interface, enabling
//! direct latency and recall comparison without changing the calling code.

use crate::distance::{cosine_distance, l2_sq};
use crate::graph::KnnGraph;
use crate::kmeans::kmeans;
use crate::spectral::fiedler_bisect;

/// A single ANN search result.
#[derive(Debug, Clone)]
pub struct SearchResult {
    /// Original vector index in the build corpus.
    pub id: usize,
    /// Distance to the query (lower = closer).
    pub distance: f32,
}

/// Shared interface for all three IVF variants.
pub trait AnnIndex {
    /// Build the index from the given corpus.
    fn build(&mut self, vectors: &[Vec<f32>]);

    /// Return the `k` approximate nearest neighbours of `query`.
    /// `nprobe` controls how many partitions are visited.
    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult>;

    /// Approximate memory footprint in bytes.
    fn memory_bytes(&self) -> usize;

    /// Human-readable variant name.
    fn name(&self) -> &str;
}

// ── Variant 1: KMeansIvf (baseline) ─────────────────────────────────────────

/// Baseline IVF with Lloyd's k-means partitioning.
pub struct KMeansIvf {
    pub n_centroids: usize,
    pub kmeans_iters: usize,
    pub seed: u64,
    centroids: Vec<Vec<f32>>,
    /// partitions[c] = list of (vector_id, vector) in centroid c.
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
    dim: usize,
}

impl KMeansIvf {
    pub fn new(n_centroids: usize) -> Self {
        Self {
            n_centroids,
            kmeans_iters: 50,
            seed: 42,
            centroids: vec![],
            partitions: vec![],
            dim: 0,
        }
    }
}

impl AnnIndex for KMeansIvf {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        self.dim = vectors[0].len();
        let k = self.n_centroids.min(vectors.len());
        let (assignments, centroids) = kmeans(vectors, k, self.kmeans_iters, self.seed);
        self.centroids = centroids;
        self.partitions = vec![vec![]; k];
        for (i, &c) in assignments.iter().enumerate() {
            self.partitions[c].push((i, vectors[i].clone()));
        }
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        let nprobe = nprobe.min(self.centroids.len());
        let mut cdists: Vec<(usize, f32)> = self
            .centroids
            .iter()
            .enumerate()
            .map(|(i, c)| (i, l2_sq(query, c)))
            .collect();
        cdists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut candidates: Vec<SearchResult> = cdists[..nprobe]
            .iter()
            .flat_map(|&(c, _)| {
                self.partitions[c].iter().map(|(id, v)| SearchResult {
                    id: *id,
                    distance: l2_sq(query, v),
                })
            })
            .collect();

        candidates.sort_by(|a, b| {
            a.distance
                .partial_cmp(&b.distance)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        candidates.dedup_by_key(|r| r.id);
        candidates.truncate(k);
        candidates
    }

    fn memory_bytes(&self) -> usize {
        let centroid_bytes: usize = self.centroids.iter().map(|c| c.len() * 4).sum();
        let partition_bytes: usize = self
            .partitions
            .iter()
            .flat_map(|p| p.iter().map(|(_, v)| v.len() * 4 + 8))
            .sum();
        centroid_bytes + partition_bytes
    }

    fn name(&self) -> &str {
        "KMeansIvf"
    }
}

// ── Variant 2: SpectralIvf ───────────────────────────────────────────────────

/// Spectral IVF: recursive Fiedler bisection with cosine-similarity kNN graph.
pub struct SpectralIvf {
    pub n_partitions: usize,
    pub knn_k: usize,
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
    /// One representative vector per partition (mean of members), used for probing.
    representatives: Vec<Vec<f32>>,
    dim: usize,
}

impl SpectralIvf {
    pub fn new(n_partitions: usize, knn_k: usize) -> Self {
        Self {
            n_partitions,
            knn_k,
            partitions: vec![],
            representatives: vec![],
            dim: 0,
        }
    }
}

impl AnnIndex for SpectralIvf {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        self.dim = vectors[0].len();
        let n = vectors.len();
        let all_indices: Vec<usize> = (0..n).collect();
        let labels = recursive_spectral_partition(
            vectors,
            &all_indices,
            self.n_partitions,
            self.knn_k,
            false,
        );

        // Group by label.
        let np = self.n_partitions;
        let mut parts: Vec<Vec<(usize, Vec<f32>)>> = vec![vec![]; np];
        for (i, &label) in labels.iter().enumerate() {
            parts[label.min(np - 1)].push((i, vectors[i].clone()));
        }
        self.representatives = parts.iter().map(|p| mean_vector(p, self.dim)).collect();
        self.partitions = parts;
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        search_partitions(query, &self.representatives, &self.partitions, k, nprobe)
    }

    fn memory_bytes(&self) -> usize {
        partition_memory(&self.partitions, &self.representatives)
    }

    fn name(&self) -> &str {
        "SpectralIvf"
    }
}

// ── Variant 3: CoherenceSpectralIvf ─────────────────────────────────────────

/// Spectral IVF with **squared** cosine-similarity kNN edges.
/// Edges between semantically similar vectors are emphasised, so Fiedler cuts
/// prefer to sever weakly-related pairs — producing coherence-aligned cells.
pub struct CoherenceSpectralIvf {
    pub n_partitions: usize,
    pub knn_k: usize,
    partitions: Vec<Vec<(usize, Vec<f32>)>>,
    representatives: Vec<Vec<f32>>,
    dim: usize,
}

impl CoherenceSpectralIvf {
    pub fn new(n_partitions: usize, knn_k: usize) -> Self {
        Self {
            n_partitions,
            knn_k,
            partitions: vec![],
            representatives: vec![],
            dim: 0,
        }
    }
}

impl AnnIndex for CoherenceSpectralIvf {
    fn build(&mut self, vectors: &[Vec<f32>]) {
        self.dim = vectors[0].len();
        let n = vectors.len();
        let all_indices: Vec<usize> = (0..n).collect();
        let labels = recursive_spectral_partition(
            vectors,
            &all_indices,
            self.n_partitions,
            self.knn_k,
            true, // coherence-weighted
        );

        let np = self.n_partitions;
        let mut parts: Vec<Vec<(usize, Vec<f32>)>> = vec![vec![]; np];
        for (i, &label) in labels.iter().enumerate() {
            parts[label.min(np - 1)].push((i, vectors[i].clone()));
        }
        self.representatives = parts.iter().map(|p| mean_vector(p, self.dim)).collect();
        self.partitions = parts;
    }

    fn search(&self, query: &[f32], k: usize, nprobe: usize) -> Vec<SearchResult> {
        search_partitions(query, &self.representatives, &self.partitions, k, nprobe)
    }

    fn memory_bytes(&self) -> usize {
        partition_memory(&self.partitions, &self.representatives)
    }

    fn name(&self) -> &str {
        "CoherenceSpectralIvf"
    }
}

// ── Shared helpers ────────────────────────────────────────────────────────────

/// Recursively bisect `indices` using the Fiedler vector until `n_target`
/// partitions are reached. Returns a label in `[0, n_target)` for each index.
fn recursive_spectral_partition(
    vectors: &[Vec<f32>],
    indices: &[usize],
    n_target: usize,
    knn_k: usize,
    coherence_weighted: bool,
) -> Vec<usize> {
    if n_target <= 1 || indices.len() <= 1 {
        return vec![0; indices.len()];
    }

    let sub_vecs: Vec<Vec<f32>> = indices.iter().map(|&i| vectors[i].clone()).collect();
    let k = knn_k.min(sub_vecs.len().saturating_sub(1));

    let graph = if coherence_weighted {
        KnnGraph::build_coherence_weighted(&sub_vecs, k)
    } else {
        KnnGraph::build(&sub_vecs, k)
    };

    let binary = fiedler_bisect(&graph);

    // Gather local indices for each half.
    let left: Vec<usize> = (0..indices.len()).filter(|&i| binary[i] == 0).collect();
    let right: Vec<usize> = (0..indices.len()).filter(|&i| binary[i] == 1).collect();

    let n_left = n_target / 2;
    let n_right = n_target - n_left;

    let left_global: Vec<usize> = left.iter().map(|&i| indices[i]).collect();
    let right_global: Vec<usize> = right.iter().map(|&i| indices[i]).collect();

    let left_labels =
        recursive_spectral_partition(vectors, &left_global, n_left, knn_k, coherence_weighted);
    let right_labels =
        recursive_spectral_partition(vectors, &right_global, n_right, knn_k, coherence_weighted);

    let mut result = vec![0usize; indices.len()];
    for (li, &local_i) in left.iter().enumerate() {
        result[local_i] = left_labels[li];
    }
    for (ri, &local_i) in right.iter().enumerate() {
        result[local_i] = right_labels[ri] + n_left;
    }
    result
}

/// Probe `nprobe` closest partitions by representative distance, collect
/// candidates, sort by distance, dedup, and return top-k.
fn search_partitions(
    query: &[f32],
    representatives: &[Vec<f32>],
    partitions: &[Vec<(usize, Vec<f32>)>],
    k: usize,
    nprobe: usize,
) -> Vec<SearchResult> {
    let nprobe = nprobe.min(representatives.len());
    let mut rep_dists: Vec<(usize, f32)> = representatives
        .iter()
        .enumerate()
        .map(|(i, r)| (i, cosine_distance(query, r)))
        .collect();
    rep_dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

    let mut candidates: Vec<SearchResult> = rep_dists[..nprobe]
        .iter()
        .flat_map(|&(p, _)| {
            partitions[p].iter().map(|(id, v)| SearchResult {
                id: *id,
                distance: cosine_distance(query, v),
            })
        })
        .collect();

    candidates.sort_by(|a, b| {
        a.distance
            .partial_cmp(&b.distance)
            .unwrap_or(std::cmp::Ordering::Equal)
    });
    candidates.dedup_by_key(|r| r.id);
    candidates.truncate(k);
    candidates
}

fn mean_vector(members: &[(usize, Vec<f32>)], dim: usize) -> Vec<f32> {
    if members.is_empty() {
        return vec![0.0; dim];
    }
    let mut sum = vec![0.0f32; dim];
    for (_, v) in members {
        for (d, &x) in v.iter().enumerate() {
            sum[d] += x;
        }
    }
    let n = members.len() as f32;
    sum.iter_mut().for_each(|x| *x /= n);
    sum
}

fn partition_memory(partitions: &[Vec<(usize, Vec<f32>)>], representatives: &[Vec<f32>]) -> usize {
    let part_bytes: usize = partitions
        .iter()
        .flat_map(|p| p.iter().map(|(_, v)| v.len() * 4 + 8))
        .sum();
    let rep_bytes: usize = representatives.iter().map(|r| r.len() * 4).sum();
    part_bytes + rep_bytes
}

#[cfg(test)]
mod tests {
    use super::*;

    fn two_cluster_corpus(n_per_cluster: usize, dim: usize) -> Vec<Vec<f32>> {
        let mut vecs = Vec::new();
        // Cluster A: first dimension dominant
        for i in 0..n_per_cluster {
            let mut v = vec![0.05f32; dim];
            v[0] = 1.0 + i as f32 * 0.01;
            vecs.push(v);
        }
        // Cluster B: second dimension dominant
        for i in 0..n_per_cluster {
            let mut v = vec![0.05f32; dim];
            v[1] = 1.0 + i as f32 * 0.01;
            vecs.push(v);
        }
        vecs
    }

    fn brute_force_top_k(query: &[f32], vectors: &[Vec<f32>], k: usize) -> Vec<usize> {
        let mut dists: Vec<(usize, f32)> = vectors
            .iter()
            .enumerate()
            .map(|(i, v)| (i, cosine_distance(query, v)))
            .collect();
        dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));
        dists.truncate(k);
        dists.into_iter().map(|(i, _)| i).collect()
    }

    #[test]
    fn kmeans_ivf_returns_k_results() {
        let corpus = two_cluster_corpus(20, 8);
        let mut idx = KMeansIvf::new(4);
        idx.build(&corpus);
        let query = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let results = idx.search(&query, 5, 2);
        assert!(results.len() <= 5);
        assert!(!results.is_empty());
    }

    #[test]
    fn spectral_ivf_recall_at_5_above_threshold() {
        let corpus = two_cluster_corpus(30, 8);
        let mut idx = SpectralIvf::new(4, 5);
        idx.build(&corpus);
        let query = vec![1.0f32, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        let ground_truth = brute_force_top_k(&query, &corpus, 5);
        let results = idx.search(&query, 5, 2);
        let result_ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        let recall = ground_truth
            .iter()
            .filter(|id| result_ids.contains(id))
            .count();
        // At least 3/5 of true top-5 should be in results (60% recall threshold).
        assert!(
            recall >= 3,
            "SpectralIvf recall@5={recall}/5 below threshold (nprobe=2, n_partitions=4)"
        );
    }

    #[test]
    fn coherence_spectral_ivf_recall_at_5_above_threshold() {
        let corpus = two_cluster_corpus(30, 8);
        let mut idx = CoherenceSpectralIvf::new(4, 5);
        idx.build(&corpus);
        let query = vec![1.0f32, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05, 0.05];
        let ground_truth = brute_force_top_k(&query, &corpus, 5);
        let results = idx.search(&query, 5, 2);
        let result_ids: Vec<usize> = results.iter().map(|r| r.id).collect();
        let recall = ground_truth
            .iter()
            .filter(|id| result_ids.contains(id))
            .count();
        assert!(
            recall >= 3,
            "CoherenceSpectralIvf recall@5={recall}/5 below threshold"
        );
    }

    #[test]
    fn memory_bytes_is_nonzero_after_build() {
        let corpus = two_cluster_corpus(10, 4);
        let mut idx = SpectralIvf::new(2, 3);
        idx.build(&corpus);
        assert!(idx.memory_bytes() > 0);
    }

    /// Acceptance test: all three variants must achieve recall@10 ≥ 0.60 on a
    /// well-clustered corpus with nprobe = n_partitions/2.
    #[test]
    fn acceptance_recall_60pct_all_variants() {
        let corpus = two_cluster_corpus(50, 16);
        let n = corpus.len();
        let queries: Vec<Vec<f32>> = (0..10).map(|i| corpus[i * (n / 10)].clone()).collect();

        let mut km = KMeansIvf::new(4);
        km.build(&corpus);
        let mut sp = SpectralIvf::new(4, 5);
        sp.build(&corpus);
        let mut csp = CoherenceSpectralIvf::new(4, 5);
        csp.build(&corpus);

        let k = 10;
        let nprobe = 2;

        for (name, idx) in [
            ("KMeansIvf", &km as &dyn AnnIndex),
            ("SpectralIvf", &sp as &dyn AnnIndex),
            ("CoherenceSpectralIvf", &csp as &dyn AnnIndex),
        ] {
            let mut total_recall = 0;
            for q in &queries {
                let gt = brute_force_top_k(q, &corpus, k);
                let res = idx.search(q, k, nprobe);
                let ids: Vec<usize> = res.iter().map(|r| r.id).collect();
                total_recall += gt.iter().filter(|id| ids.contains(id)).count();
            }
            let recall_rate = total_recall as f64 / (queries.len() * k) as f64;
            assert!(
                recall_rate >= 0.60,
                "{name}: recall@{k}={:.2} < 0.60 (nprobe={nprobe})",
                recall_rate
            );
        }
    }
}
