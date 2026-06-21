//! LSM-ANN: write-optimized streaming vector index for agent memory.
//!
//! Organises vectors in a Log-Structured Merge (LSM) tier hierarchy:
//!
//! ```text
//! ┌───────────────────────────────────────┐
//! │ L0  MemTable  (mutable, brute-force)  │  ← all writes land here first
//! ├───────────────────────────────────────┤
//! │ L1  Small frozen segments (NSW graph) │  ← compacted from L0
//! ├───────────────────────────────────────┤
//! │ L2  Large merged segment  (NSW graph) │  ← compacted from L1 segments
//! └───────────────────────────────────────┘
//! ```
//!
//! Queries merge candidate lists from all tiers and re-rank by exact distance.

pub mod lsm;
pub mod memtable;
pub mod segment;

pub use lsm::{BaselineLsm, FullLsm, TwoTierLsm};
pub use memtable::MemTable;
pub use segment::FrozenSegment;

/// Core trait for all LSM-ANN index variants.
pub trait LsmIndex {
    /// Insert a vector with the given id. Overwrites if id already exists.
    fn insert(&mut self, id: u64, vector: Vec<f32>);

    /// Return the k approximate nearest neighbours to `query`.
    /// Results are sorted by ascending distance (closest first).
    fn search(&self, query: &[f32], k: usize) -> Vec<(u64, f32)>;

    /// Total number of live vectors in the index.
    fn len(&self) -> usize;

    /// Whether the index is empty.
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Number of segments (L1/L2) not counting the active memtable.
    fn segment_count(&self) -> usize;

    /// Trigger a compaction pass (variant-dependent semantics).
    fn compact(&mut self);
}

/// Configuration shared across variants.
#[derive(Clone, Debug)]
pub struct LsmConfig {
    /// Dimension of every vector. Must match all inserts.
    pub dims: usize,
    /// Max neighbours per node in NSW graph segments.
    pub m: usize,
    /// Beam width used during segment graph construction.
    pub ef_construction: usize,
    /// Beam width used at query time across frozen segments.
    pub ef_search: usize,
    /// L0 size threshold that triggers a compaction to L1.
    pub l0_max: usize,
    /// Number of L1 segments that triggers a merge into L2.
    pub l1_merge_threshold: usize,
}

impl Default for LsmConfig {
    fn default() -> Self {
        Self {
            dims: 128,
            m: 16,
            ef_construction: 64,
            ef_search: 64,
            l0_max: 1_000,
            l1_merge_threshold: 5,
        }
    }
}

/// Euclidean squared distance (no sqrt — sufficient for ranking).
#[inline(always)]
pub fn sq_dist(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Brute-force k-nearest on a slice of (id, vector) pairs.
pub fn brute_force_knn(haystack: &[(u64, Vec<f32>)], query: &[f32], k: usize) -> Vec<(u64, f32)> {
    let mut dists: Vec<(u64, f32)> = haystack
        .iter()
        .map(|(id, v)| (*id, sq_dist(v, query)))
        .collect();
    dists.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());
    dists.truncate(k);
    dists
}

/// Merge two candidate lists and keep the top-k by distance.
pub fn merge_candidates(
    mut a: Vec<(u64, f32)>,
    mut b: Vec<(u64, f32)>,
    k: usize,
) -> Vec<(u64, f32)> {
    a.append(&mut b);
    // de-duplicate by id (keep smallest distance)
    a.sort_by(|x, y| x.0.cmp(&y.0).then(x.1.partial_cmp(&y.1).unwrap()));
    a.dedup_by(|newer, older| {
        if newer.0 == older.0 {
            if newer.1 < older.1 {
                older.1 = newer.1;
            }
            true
        } else {
            false
        }
    });
    a.sort_by(|x, y| x.1.partial_cmp(&y.1).unwrap());
    a.truncate(k);
    a
}

/// Compute recall@k between result ids and ground-truth ids.
pub fn recall_at_k(results: &[(u64, f32)], ground_truth: &[(u64, f32)], k: usize) -> f64 {
    let gt_ids: std::collections::HashSet<u64> =
        ground_truth.iter().take(k).map(|(id, _)| *id).collect();
    let res_ids: std::collections::HashSet<u64> =
        results.iter().take(k).map(|(id, _)| *id).collect();
    let intersection = gt_ids.intersection(&res_ids).count();
    intersection as f64 / k.min(gt_ids.len()) as f64
}

// ---------------------------------------------------------------------------
// Tests
// ---------------------------------------------------------------------------

#[cfg(test)]
mod tests {
    use super::*;
    use crate::lsm::{BaselineLsm, FullLsm, TwoTierLsm};
    use rand::SeedableRng;
    use rand_distr::{Distribution, Normal};

    fn make_vecs(n: usize, dims: usize, seed: u64) -> Vec<Vec<f32>> {
        let mut rng = rand::rngs::StdRng::seed_from_u64(seed);
        let normal = Normal::<f32>::new(0.0, 1.0).unwrap();
        (0..n)
            .map(|_| (0..dims).map(|_| normal.sample(&mut rng)).collect())
            .collect()
    }

    fn cfg(dims: usize) -> LsmConfig {
        LsmConfig {
            dims,
            m: 8,
            ef_construction: 32,
            ef_search: 32,
            l0_max: 50,
            l1_merge_threshold: 3,
        }
    }

    // Insert 200 vectors and confirm len() is correct for each variant.
    #[test]
    fn test_baseline_len() {
        let vecs = make_vecs(200, 32, 1);
        let mut idx = BaselineLsm::new(cfg(32));
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        assert_eq!(idx.len(), 200);
    }

    #[test]
    fn test_twotier_len() {
        let vecs = make_vecs(200, 32, 2);
        let mut idx = TwoTierLsm::new(cfg(32));
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        idx.compact();
        assert_eq!(idx.len(), 200);
    }

    #[test]
    fn test_fulllsm_len() {
        let vecs = make_vecs(200, 32, 3);
        let mut idx = FullLsm::new(cfg(32));
        for (i, v) in vecs.iter().enumerate() {
            idx.insert(i as u64, v.clone());
        }
        idx.compact();
        assert_eq!(idx.len(), 200);
    }

    // Baseline brute-force must achieve recall@5 = 1.0 (exact search).
    #[test]
    fn test_baseline_perfect_recall() {
        let vecs = make_vecs(500, 32, 4);
        let mut idx = BaselineLsm::new(cfg(32));
        let all: Vec<(u64, Vec<f32>)> = vecs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (i as u64, v))
            .collect();
        for (id, v) in &all {
            idx.insert(*id, v.clone());
        }
        let query = &vecs[7];
        let result = idx.search(query, 5);
        let gt = brute_force_knn(&all, query, 5);
        let recall = recall_at_k(&result, &gt, 5);
        assert!(
            recall >= 0.999,
            "Baseline should achieve perfect recall; got {recall:.4}"
        );
    }

    // TwoTier with NSW should achieve recall@5 ≥ 0.70 on 500×32 random data.
    #[test]
    fn test_twotier_recall_threshold() {
        let vecs = make_vecs(500, 32, 5);
        let mut idx = TwoTierLsm::new(cfg(32));
        let all: Vec<(u64, Vec<f32>)> = vecs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (i as u64, v))
            .collect();
        for (id, v) in &all {
            idx.insert(*id, v.clone());
        }
        idx.compact();

        let queries = make_vecs(50, 32, 99);
        let total_recall: f64 = queries
            .iter()
            .map(|q| {
                let res = idx.search(q, 5);
                let gt = brute_force_knn(&all, q, 5);
                recall_at_k(&res, &gt, 5)
            })
            .sum::<f64>()
            / 50.0;

        assert!(
            total_recall >= 0.70,
            "TwoTier should achieve recall@5 ≥ 0.70; got {total_recall:.4}"
        );
    }

    // FullLsm should achieve recall@5 ≥ 0.70 after full compaction.
    #[test]
    fn test_fulllsm_recall_threshold() {
        let vecs = make_vecs(500, 32, 6);
        let mut idx = FullLsm::new(cfg(32));
        let all: Vec<(u64, Vec<f32>)> = vecs
            .iter()
            .cloned()
            .enumerate()
            .map(|(i, v)| (i as u64, v))
            .collect();
        for (id, v) in &all {
            idx.insert(*id, v.clone());
        }
        idx.compact();

        let queries = make_vecs(50, 32, 100);
        let total_recall: f64 = queries
            .iter()
            .map(|q| {
                let res = idx.search(q, 5);
                let gt = brute_force_knn(&all, q, 5);
                recall_at_k(&res, &gt, 5)
            })
            .sum::<f64>()
            / 50.0;

        assert!(
            total_recall >= 0.70,
            "FullLsm should achieve recall@5 ≥ 0.70; got {total_recall:.4}"
        );
    }

    // merge_candidates must de-duplicate and keep the k smallest.
    #[test]
    fn test_merge_dedup() {
        let a = vec![(1u64, 0.1f32), (2u64, 0.5f32), (3u64, 0.9f32)];
        let b = vec![(2u64, 0.4f32), (3u64, 1.0f32), (4u64, 0.2f32)];
        let merged = merge_candidates(a, b, 3);
        // Expect: (1, 0.1), (4, 0.2), (2, 0.4) — id 2 keeps dist=0.4 (minimum), id 3 dropped
        let ids: Vec<u64> = merged.iter().map(|(id, _)| *id).collect();
        assert_eq!(ids, vec![1, 4, 2]);
    }

    // recall_at_k of identical sets should be 1.0.
    #[test]
    fn test_recall_perfect() {
        let r = vec![(0u64, 0.1), (1u64, 0.2), (2u64, 0.3)];
        let g = vec![(0u64, 0.1), (1u64, 0.2), (2u64, 0.3)];
        assert!((recall_at_k(&r, &g, 3) - 1.0).abs() < 1e-6);
    }
}
