use std::collections::BinaryHeap;
use std::cmp::Ordering;

use rayon::prelude::*;

use crate::config::LorannConfig;
use crate::error::{LorannError, Result};
use crate::kmeans::{dot, kmeans, top_n_centroids, KMeansResult};
use crate::regression::ClusterModel;

/// A single ANN result.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    pub id: usize,
    /// Higher is more similar (negated L2 or raw inner-product approximation).
    pub score: f32,
}

/// Shared trait for all index variants in this crate.
pub trait AnnIndex: Send + Sync {
    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>>;
    fn len(&self) -> usize;
    fn is_empty(&self) -> bool { self.len() == 0 }
    fn dim(&self) -> usize;
    fn memory_bytes(&self) -> usize;
    fn name(&self) -> &'static str;
}

// ---------------------------------------------------------------------------
// Variant 1: FlatExactIndex — brute-force f32 exact inner-product baseline
// ---------------------------------------------------------------------------

/// Baseline: computes exact inner products in O(n·d) per query.
pub struct FlatExactIndex {
    data: Vec<Vec<f32>>,
}

impl FlatExactIndex {
    pub fn build(data: Vec<Vec<f32>>) -> Result<Self> {
        if data.is_empty() {
            return Err(LorannError::EmptyDataset);
        }
        Ok(Self { data })
    }
}

impl AnnIndex for FlatExactIndex {
    fn name(&self) -> &'static str { "FlatExact" }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let d = self.data[0].len();
        if query.len() != d {
            return Err(LorannError::DimMismatch { expected: d, got: query.len() });
        }
        let mut heap: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);
        for (id, v) in self.data.iter().enumerate() {
            let score = dot(query, v);
            if heap.len() < k {
                heap.push(MinEntry { score, id });
            } else if let Some(worst) = heap.peek() {
                if score > worst.score {
                    heap.pop();
                    heap.push(MinEntry { score, id });
                }
            }
        }
        let mut results: Vec<SearchResult> = heap
            .into_iter()
            .map(|e| SearchResult { id: e.id, score: e.score })
            .collect();
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        Ok(results)
    }

    fn len(&self) -> usize { self.data.len() }
    fn dim(&self) -> usize { self.data[0].len() }
    fn memory_bytes(&self) -> usize {
        self.data.len() * self.data[0].len() * 4
    }
}

// ---------------------------------------------------------------------------
// Variant 2 & 3: LorannIndex — IVF with per-cluster RRR score approximation
// ---------------------------------------------------------------------------

/// IVF-based ANN index with reduced-rank regression per cluster (LoRANN).
///
/// Build is O(n · k · max_iter · d) for k-means + O(k · m · d · r) for SVDs.
/// Query is O(n_probe · r · (d + m_avg)) + O(candidate_set · d) for rerank.
pub struct LorannIndex {
    /// k-means result: centroids and per-vector assignments.
    km: KMeansResult,
    /// Per-cluster model (one per centroid).
    models: Vec<ClusterModel>,
    /// Cluster membership lists: `members[c]` = global IDs in cluster c.
    members: Vec<Vec<usize>>,
    /// Raw f32 vectors for exact reranking.
    raw: Vec<Vec<f32>>,
    config: LorannConfig,
}

impl LorannIndex {
    /// Build a LoRANN index from `data`.
    ///
    /// Steps:
    /// 1. k-means clustering
    /// 2. Per-cluster truncated SVD to produce `ClusterModel`
    /// 3. Store raw vectors for exact reranking
    pub fn build(data: Vec<Vec<f32>>, config: LorannConfig) -> Result<Self> {
        if data.is_empty() {
            return Err(LorannError::EmptyDataset);
        }
        let d = data[0].len();
        for (_i, v) in data.iter().enumerate() {
            if v.len() != d {
                return Err(LorannError::DimMismatch { expected: d, got: v.len() });
            }
        }
        if config.n_probe > config.n_clusters {
            return Err(LorannError::NProbeExceedsClusters {
                n_probe: config.n_probe,
                n_clusters: config.n_clusters,
            });
        }

        let n_clusters = config.n_clusters.min(data.len());
        let km = kmeans(&data, n_clusters, config.kmeans_max_iter, config.seed)?;

        // Group member indices by cluster
        let mut members: Vec<Vec<usize>> = vec![vec![]; n_clusters];
        for (i, &c) in km.assignments.iter().enumerate() {
            members[c].push(i);
        }

        // Build per-cluster RRR models (parallel over clusters)
        let models: Vec<Result<ClusterModel>> = members
            .par_iter()
            .enumerate()
            .map(|(c, member_ids)| {
                let cluster_docs: Vec<Vec<f32>> = member_ids.iter().map(|&id| data[id].clone()).collect();
                ClusterModel::fit(c, &cluster_docs, config.rank)
            })
            .collect();

        let models: Vec<ClusterModel> = models.into_iter().collect::<Result<Vec<_>>>()?;

        Ok(Self { km, models, members, raw: data, config })
    }

    /// Perform a LoRANN approximate search.
    pub fn search_internal(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        let n_probe = self.config.n_probe.min(self.km.centroids.len());
        let probe_clusters = top_n_centroids(query, &self.km.centroids, n_probe);

        let candidates_per_cluster = (self.config.candidate_set / n_probe).max(1);
        let mut candidates: Vec<(usize, f32)> = Vec::with_capacity(self.config.candidate_set);

        for &c in &probe_clusters {
            let model = &self.models[c];
            let member_ids = &self.members[c];
            if member_ids.is_empty() {
                continue;
            }
            // Approximate scores via RRR
            let approx = model.approximate_scores(query);
            // Take top candidates_per_cluster from this cluster
            let take = candidates_per_cluster.min(approx.len());
            let mut indexed: Vec<(usize, f32)> = approx
                .into_iter()
                .enumerate()
                .map(|(local_idx, score)| (member_ids[local_idx], score))
                .collect();
            indexed.sort_unstable_by(|a, b| b.1.total_cmp(&a.1));
            for (global_id, score) in indexed.into_iter().take(take) {
                candidates.push((global_id, score));
            }
        }

        // Deduplicate by global_id (keep highest approximate score)
        candidates.sort_unstable_by(|a, b| a.0.cmp(&b.0));
        candidates.dedup_by(|a, b| {
            if a.0 == b.0 {
                if a.1 > b.1 { b.1 = a.1; }
                true
            } else {
                false
            }
        });

        // Exact rerank
        let mut reranked: BinaryHeap<MinEntry> = BinaryHeap::with_capacity(k + 1);
        for (id, _) in &candidates {
            let exact_score = dot(query, &self.raw[*id]);
            if reranked.len() < k {
                reranked.push(MinEntry { score: exact_score, id: *id });
            } else if let Some(worst) = reranked.peek() {
                if exact_score > worst.score {
                    reranked.pop();
                    reranked.push(MinEntry { score: exact_score, id: *id });
                }
            }
        }

        let mut results: Vec<SearchResult> = reranked
            .into_iter()
            .map(|e| SearchResult { id: e.id, score: e.score })
            .collect();
        results.sort_unstable_by(|a, b| b.score.total_cmp(&a.score));
        Ok(results)
    }
}

impl AnnIndex for LorannIndex {
    fn name(&self) -> &'static str { "LoRANN" }

    fn search(&self, query: &[f32], k: usize) -> Result<Vec<SearchResult>> {
        self.search_internal(query, k)
    }

    fn len(&self) -> usize { self.raw.len() }

    fn dim(&self) -> usize {
        self.raw.first().map(|v| v.len()).unwrap_or(0)
    }

    fn memory_bytes(&self) -> usize {
        let raw_bytes = self.raw.len() * self.dim() * 4;
        let model_bytes: usize = self.models.iter().map(|m| m.memory_bytes()).sum();
        let centroid_bytes = self.km.centroids.len() * self.dim() * 4;
        let member_bytes: usize = self.members.iter().map(|v| v.len() * 8).sum();
        raw_bytes + model_bytes + centroid_bytes + member_bytes
    }
}

// ---------------------------------------------------------------------------
// Internal heap entry (min-heap on score, so we evict the worst of top-k)
// ---------------------------------------------------------------------------

#[derive(Debug, Clone, Copy)]
struct MinEntry {
    score: f32,
    id: usize,
}

impl PartialEq for MinEntry {
    fn eq(&self, other: &Self) -> bool { self.score.total_cmp(&other.score) == Ordering::Equal }
}
impl Eq for MinEntry {}
impl PartialOrd for MinEntry {
    fn partial_cmp(&self, other: &Self) -> Option<Ordering> { Some(self.cmp(other)) }
}
impl Ord for MinEntry {
    fn cmp(&self, other: &Self) -> Ordering {
        // Reverse so BinaryHeap (max-heap) acts as min-heap on score
        other.score.total_cmp(&self.score)
    }
}
