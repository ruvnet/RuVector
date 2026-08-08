//! # RuVector Namespace-Merge MinCut
//!
//! S-T mincut namespace routing for multi-namespace agent memory.
//!
//! Agent memory is partitioned into named namespaces. A query may span multiple
//! namespaces; searching all of them is expensive. This crate provides three
//! routing strategies that decide *which* namespaces to search:
//!
//! 1. [`AllSearch`]      – baseline: scan every namespace unconditionally.
//! 2. [`CentroidFilter`] – heuristic: skip namespaces whose centroid cosine
//!                         similarity to the query falls below a threshold.
//! 3. [`MinCutRoute`]    – principled: build a flow graph where source→namespace
//!                         capacity = query relevance, namespace→sink capacity =
//!                         query irrelevance, and inter-namespace edges = semantic
//!                         similarity. Find the min S-T cut; search namespaces on
//!                         the source side.
//!
//! All three implement the [`NamespaceRouter`] trait so they can be swapped
//! transparently by benchmark or production code.

pub mod dataset;
pub mod flow;
pub mod router;

pub use dataset::{Dataset, DatasetConfig, Namespace};
pub use router::{AllSearch, CentroidFilter, MinCutRoute, NamespaceRouter, RouteResult};

use std::collections::HashSet;

// ─── hit ─────────────────────────────────────────────────────────────────────

/// A nearest-neighbour result: global vector id and squared-L2 distance.
#[derive(Debug, Clone, PartialEq)]
pub struct Hit {
    pub id: usize,
    pub dist: f32,
}

impl Eq for Hit {}

impl PartialOrd for Hit {
    fn partial_cmp(&self, other: &Self) -> Option<std::cmp::Ordering> {
        Some(self.cmp(other))
    }
}

impl Ord for Hit {
    fn cmp(&self, other: &Self) -> std::cmp::Ordering {
        self.dist
            .partial_cmp(&other.dist)
            .unwrap_or(std::cmp::Ordering::Equal)
    }
}

// ─── distances ───────────────────────────────────────────────────────────────

/// Squared L2 distance (no sqrt; monotone for ranking).
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity for normalised vectors (dot product suffices).
#[inline(always)]
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

// ─── recall ──────────────────────────────────────────────────────────────────

/// Recall@k: fraction of ground-truth ids present in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let res_ids: HashSet<usize> = results.iter().take(k).map(|h| h.id).collect();
    let gt_ids: HashSet<usize> = ground_truth.iter().take(k).map(|h| h.id).collect();
    if gt_ids.is_empty() {
        return 1.0;
    }
    let hits = res_ids.intersection(&gt_ids).count();
    hits as f32 / k.min(gt_ids.len()) as f32
}
