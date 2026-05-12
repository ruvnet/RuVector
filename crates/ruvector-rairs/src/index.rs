//! Shared ANN index trait and search result type.

use crate::error::RairsError;

/// A nearest-neighbor result from any index variant.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Original vector ID (0-based insertion order).
    pub id: usize,
    /// Approximate L2 distance to the query.
    pub distance: f32,
}

/// Common interface for all three RAIRS index variants.
pub trait AnnIndex {
    /// Add a slice of f32 vectors to the index.
    fn add(&mut self, vectors: &[Vec<f32>]) -> Result<(), RairsError>;

    /// Search for the `k` approximate nearest neighbors of `query`.
    /// `nprobe` controls how many inverted lists are visited.
    fn search(
        &self,
        query: &[f32],
        k: usize,
        nprobe: usize,
    ) -> Result<Vec<SearchResult>, RairsError>;

    /// Return the number of indexed vectors.
    fn len(&self) -> usize;

    /// Return true if the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Return the number of inverted lists (clusters).
    fn num_lists(&self) -> usize;
}

// ─── shared distance helpers ─────────────────────────────────────────────────

/// Number of independent FP accumulators in the manually-unrolled reductions
/// below. f32 addition is not associative, so the naïve `iter().sum()` form
/// won't auto-vectorise — splitting the reduction into `LANES` parallel partial
/// sums lets LLVM emit packed SIMD on every target without any `unsafe`.
const LANES: usize = 8;

/// Squared Euclidean distance between two equal-length f32 slices.
#[inline(always)]
pub fn l2sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            let d = xa[l] - xb[l];
            acc[l] += d * d;
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        let d = x - y;
        sum += d * d;
    }
    sum
}

/// Dot product of two equal-length f32 slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut acc = [0.0f32; LANES];
    let mut ca = a.chunks_exact(LANES);
    let mut cb = b.chunks_exact(LANES);
    for (xa, xb) in ca.by_ref().zip(cb.by_ref()) {
        for l in 0..LANES {
            acc[l] += xa[l] * xb[l];
        }
    }
    let mut sum: f32 = acc.iter().sum();
    for (x, y) in ca.remainder().iter().zip(cb.remainder()) {
        sum += x * y;
    }
    sum
}

/// Reduce a candidate set to its `k` smallest-distance entries, ascending.
///
/// Uses `select_nth_unstable` (O(n) average) to partition off the top-`k`
/// before sorting only those — instead of fully sorting every candidate.
/// Ordering on distances uses [`f32::total_cmp`], so NaNs can't panic.
pub(crate) fn finalize_topk(mut cands: Vec<SearchResult>, k: usize) -> Vec<SearchResult> {
    let k = k.min(cands.len());
    if k == 0 {
        return Vec::new();
    }
    if cands.len() > k {
        cands.select_nth_unstable_by(k - 1, |a, b| a.distance.total_cmp(&b.distance));
        cands.truncate(k);
    }
    cands.sort_unstable_by(|a, b| a.distance.total_cmp(&b.distance));
    cands
}

/// Indices of the `nprobe` centroids closest to `query`, in arbitrary order.
/// O(n) average via `select_nth_unstable` rather than a full O(n log n) sort —
/// the probe order doesn't affect the result set.
pub(crate) fn top_nprobe_centroids(
    query: &[f32],
    centroids: &[Vec<f32>],
    nprobe: usize,
) -> Vec<usize> {
    let mut cd: Vec<(usize, f32)> = centroids
        .iter()
        .enumerate()
        .map(|(i, c)| (i, l2sq(query, c)))
        .collect();
    let nprobe = nprobe.min(cd.len());
    if nprobe > 0 && cd.len() > nprobe {
        cd.select_nth_unstable_by(nprobe - 1, |a, b| a.1.total_cmp(&b.1));
        cd.truncate(nprobe);
    }
    cd.into_iter().map(|(i, _)| i).collect()
}
