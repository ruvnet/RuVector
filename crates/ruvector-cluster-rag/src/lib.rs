//! Hierarchical Cluster-Summary Retrieval for RuVector
//!
//! Implements a two-level cluster tree over an agent memory corpus, inspired by
//! RAPTOR (Chen et al. 2024). At query time, clusters are scored by a combination
//! of query–centroid similarity and per-cluster cohesion, so tight, relevant
//! clusters are searched first. Three measurable variants:
//!
//! - `FlatBrute`      – O(n·d) baseline brute-force (ground truth)
//! - `ClusterSearch`  – IVF-style: score centroids by L2, expand top-nprobe
//! - `CoherenceTree`  – score centroids by λ·sim(q,c) + (1-λ)·cohesion(c)

pub mod bench;
pub mod cluster;
pub mod dataset;
pub mod search;
pub mod tree;

// ─── shared types ────────────────────────────────────────────────────────────

/// One nearest-neighbour result.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist_sq: f32,
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist_sq
            .partial_cmp(&other.dist_sq)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

/// All search backends implement this trait.
pub trait AnnVariant: Send + Sync {
    fn name(&self) -> &'static str;
    /// Return the k approximate nearest neighbours (ascending distance).
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;
    /// Bytes consumed by internal data structures.
    fn mem_bytes(&self) -> usize;
}

/// Compute squared L2 distance between two equal-length slices.
#[inline(always)]
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity (dot product of normalised vectors).
#[inline(always)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na == 0.0 || nb == 0.0 {
        0.0
    } else {
        dot / (na * nb)
    }
}

/// Recall@k: fraction of ground-truth top-k ids present in candidate top-k.
pub fn recall_at_k(candidate: &[Hit], ground_truth: &[Hit]) -> f32 {
    let gt_ids: std::collections::HashSet<usize> = ground_truth.iter().map(|h| h.id).collect();
    let hits = candidate.iter().filter(|h| gt_ids.contains(&h.id)).count();
    hits as f32 / ground_truth.len() as f32
}
