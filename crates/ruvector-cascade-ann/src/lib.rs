//! Graph-Neighbour Cascade ANN for RuVector
//!
//! Three-stage approximate nearest-neighbour search:
//!   1. INT8 compressed scan over all vectors for an initial candidate set.
//!   2. Identify the "uncertain zone": candidates within a configurable margin
//!      of the k-th candidate's score, where recall errors are most likely.
//!   3. Expand via graph neighbours of uncertain candidates, then exact f32
//!      re-rank the expanded pool to recover missed true neighbours.
//!
//! Three measurable variants:
//!  - `LinearFull`           – brute-force f32 scan (ground truth)
//!  - `QuantizedCascade`     – INT8 scan → exact f32 verify of top-k' (baseline cascade)
//!  - `GraphNeighbourCascade`– INT8 scan → uncertain-zone graph expansion → exact f32 verify

pub mod dataset;
pub mod graph;
pub mod quantize;
pub mod variants;

// ─── core types ──────────────────────────────────────────────────────────────

/// A single nearest-neighbour hit.
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

// ─── trait ───────────────────────────────────────────────────────────────────

/// Uniform search interface for all three variants.
pub trait AnnVariant: Send + Sync {
    /// Return the `k` approximate nearest neighbours to `query`.
    fn search(&self, query: &[f32], k: usize) -> Vec<Hit>;

    /// Human-readable variant name for reporting.
    fn name(&self) -> &str;

    /// Estimated heap memory used by this index, in bytes.
    fn memory_bytes(&self) -> usize;
}

// ─── recall metric ───────────────────────────────────────────────────────────

/// Recall@k: fraction of ground-truth top-k that appear in the result set.
pub fn recall_at_k(result: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let gt_ids: std::collections::HashSet<usize> =
        ground_truth.iter().take(k).map(|h| h.id).collect();
    let found = result
        .iter()
        .take(k)
        .filter(|h| gt_ids.contains(&h.id))
        .count();
    found as f32 / k as f32
}

// ─── distance ────────────────────────────────────────────────────────────────

/// Squared Euclidean distance between two f32 slices.
#[inline(always)]
pub fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}
