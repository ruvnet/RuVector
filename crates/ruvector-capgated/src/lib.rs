pub mod cap_graph;
/// Capability-gated ANN search.
///
/// Each stored vector carries a `CapMask` stating which capabilities are
/// *required* to retrieve it.  A querier presents their own `CapMask`
/// (the capabilities they *hold*), and only vectors whose required mask is
/// a subset of the querier's mask are returned.
///
/// Three variants measure the security-vs-performance tradeoff:
///
/// | Variant        | Strategy                         | Recall | Cost          |
/// |----------------|----------------------------------|--------|---------------|
/// | PostFilter     | scan all, discard unauthorised   | 100%   | O(n)          |
/// | EagerMask      | skip unauthorised before dot-prod| 100%   | O(auth_frac·n)|
/// | CapGraph       | graph walk, prune unauth nodes   | ≤100%  | O(deg·steps)  |
pub mod dataset;
pub mod eager_mask;
pub mod oracle;
pub mod post_filter;

/// 64-bit bitset representing a set of capabilities.
///
/// For a querier to access a vector:
///   `(querier.0 & required.0) == required.0`
///
/// i.e. the querier must hold *all* bits required by the vector.
#[derive(Clone, Copy, Debug, PartialEq, Eq, Hash)]
pub struct CapMask(pub u64);

impl CapMask {
    pub const NONE: CapMask = CapMask(0);
    pub const ALL: CapMask = CapMask(u64::MAX);

    /// Create a mask with exactly one capability bit set.
    #[inline]
    pub fn single(bit: u8) -> Self {
        CapMask(1u64 << (bit & 63))
    }

    /// Combine two masks (union).
    #[inline]
    pub fn union(self, other: CapMask) -> CapMask {
        CapMask(self.0 | other.0)
    }

    /// Returns true if `self` (querier) satisfies `required`.
    #[inline]
    pub fn satisfies(self, required: CapMask) -> bool {
        (self.0 & required.0) == required.0
    }

    /// Number of bits set.
    #[inline]
    pub fn count(self) -> u32 {
        self.0.count_ones()
    }
}

/// A single vector stored in the index.
#[derive(Clone, Debug)]
pub struct VecEntry {
    pub id: usize,
    pub vector: Vec<f32>,
    /// Capabilities a querier must hold to retrieve this entry.
    pub required: CapMask,
}

/// One result returned by a capability-gated search.
#[derive(Clone, Debug)]
pub struct SearchResult {
    pub id: usize,
    /// Squared Euclidean distance to the query.
    pub dist_sq: f32,
}

/// Unified interface for all capability-gated ANN backends.
pub trait CapGatedIndex {
    /// Store a vector with the given required capability mask.
    fn insert(&mut self, id: usize, vector: Vec<f32>, required: CapMask);

    /// Return the k nearest *authorised* vectors for a querier holding `holder`.
    fn search(&self, query: &[f32], k: usize, holder: CapMask) -> Vec<SearchResult>;

    /// Human-readable variant name for benchmark output.
    fn name(&self) -> &'static str;
}

/// Squared Euclidean distance between two equal-length f32 slices.
#[inline]
pub fn dist_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Compute recall@k: fraction of oracle results present in candidate results.
pub fn recall_at_k(oracle: &[SearchResult], candidates: &[SearchResult], k: usize) -> f32 {
    if oracle.is_empty() {
        return 1.0; // vacuously correct
    }
    let oracle_ids: std::collections::HashSet<usize> =
        oracle.iter().take(k).map(|r| r.id).collect();
    let hits = candidates
        .iter()
        .take(k)
        .filter(|r| oracle_ids.contains(&r.id))
        .count();
    let denom = oracle.len().min(k);
    if denom == 0 {
        1.0
    } else {
        hits as f32 / denom as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cap_mask_satisfies() {
        let holder = CapMask(0b1110);
        assert!(holder.satisfies(CapMask(0b0010)));
        assert!(holder.satisfies(CapMask(0b1100)));
        assert!(!holder.satisfies(CapMask(0b0001)));
        assert!(!holder.satisfies(CapMask(0b1111)));
    }

    #[test]
    fn cap_mask_none_always_satisfies_none() {
        assert!(CapMask::NONE.satisfies(CapMask::NONE));
    }

    #[test]
    fn dist_sq_zero_on_equal() {
        let v = vec![1.0f32, 2.0, 3.0];
        assert_eq!(dist_sq(&v, &v), 0.0);
    }

    #[test]
    fn dist_sq_known() {
        let a = vec![0.0f32, 0.0];
        let b = vec![3.0f32, 4.0];
        assert!((dist_sq(&a, &b) - 25.0).abs() < 1e-5);
    }

    #[test]
    fn recall_perfect() {
        let oracle = vec![
            SearchResult {
                id: 1,
                dist_sq: 0.1,
            },
            SearchResult {
                id: 2,
                dist_sq: 0.2,
            },
        ];
        assert_eq!(recall_at_k(&oracle, &oracle, 2), 1.0);
    }

    #[test]
    fn recall_zero() {
        let oracle = vec![SearchResult {
            id: 1,
            dist_sq: 0.1,
        }];
        let cands = vec![SearchResult {
            id: 99,
            dist_sq: 0.5,
        }];
        assert_eq!(recall_at_k(&oracle, &cands, 1), 0.0);
    }
}
