//! Semantic Query Cache for RuVector ANN
//!
//! Agents issue statistically similar queries. A semantic cache avoids fresh ANN
//! computation by returning cached results when the incoming query is close enough
//! to a previously-answered query.
//!
//! Three measurable variants:
//!  - `NoCache`       – fresh brute-force scan every call (ground truth baseline)
//!  - `ExactCache`    – bitwise-exact query hash match; rarely hits in practice
//!  - `SemanticCache` – cosine-similarity cache lookup; tunes hit rate vs. quality
//!
//! The semantic cache lookup itself is O(n_cache × dim) brute force over stored
//! query vectors. At typical agent-memory cache sizes (≤2048 entries) this is
//! dominated by the underlying ANN cost, so the net result is positive when the
//! hit rate is high enough.

pub mod dataset;
pub mod exact_cache;
pub mod no_cache;
pub mod semantic_cache;

use std::collections::HashSet;

// ─── core types ──────────────────────────────────────────────────────────────

/// A single nearest-neighbour hit (id, squared-L2 distance).
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

/// Whether a query was answered from the cache or required a fresh scan.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum CacheDecision {
    Hit { similarity: f32 },
    Miss,
}

/// Running totals updated on every call.
#[derive(Debug, Default, Clone)]
pub struct CacheStats {
    pub hits: u64,
    pub misses: u64,
}

impl CacheStats {
    pub fn total(&self) -> u64 {
        self.hits + self.misses
    }

    pub fn hit_rate(&self) -> f32 {
        let t = self.total();
        if t == 0 {
            0.0
        } else {
            self.hits as f32 / t as f32
        }
    }
}

// ─── trait ───────────────────────────────────────────────────────────────────

/// Common interface for all three caching variants.
pub trait CachedAnn {
    /// Search for the k approximate nearest neighbours.
    /// Returns the results and whether they came from the cache.
    fn search(&mut self, query: &[f32], k: usize) -> (Vec<Hit>, CacheDecision);

    /// Human-readable variant name.
    fn name(&self) -> &str;

    /// Snapshot of hit/miss counters.
    fn stats(&self) -> CacheStats;

    /// Estimated heap memory in bytes.
    fn memory_bytes(&self) -> usize;
}

// ─── distance helpers ─────────────────────────────────────────────────────────

/// Squared L2 distance — monotone with L2 so safe for ranking.
#[inline(always)]
pub fn sq_l2(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Cosine similarity in [−1, 1].
/// Clamps dot product to avoid NaN on zero-norm vectors.
#[inline]
pub fn cosine(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let na: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let nb: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if na < 1e-9 || nb < 1e-9 {
        return 0.0;
    }
    (dot / (na * nb)).clamp(-1.0, 1.0)
}

// ─── recall metric ───────────────────────────────────────────────────────────

/// Recall@k: fraction of ground-truth top-k ids present in `results`.
pub fn recall_at_k(results: &[Hit], ground_truth: &[Hit], k: usize) -> f32 {
    let res_ids: HashSet<usize> = results.iter().take(k).map(|h| h.id).collect();
    let gt_ids: HashSet<usize> = ground_truth.iter().take(k).map(|h| h.id).collect();
    if gt_ids.is_empty() {
        return 1.0;
    }
    let intersection = res_ids.intersection(&gt_ids).count();
    intersection as f32 / k.min(gt_ids.len()) as f32
}

// ─── brute-force linear scan (shared primitive) ──────────────────────────────

/// Exact brute-force top-k over `corpus` for a single `query`.
/// Used internally by NoCache and as the miss-path in caching variants.
pub fn brute_force_topk(corpus: &[Vec<f32>], query: &[f32], k: usize) -> Vec<Hit> {
    let mut hits: Vec<Hit> = corpus
        .iter()
        .enumerate()
        .map(|(id, v)| Hit {
            id,
            dist: sq_l2(query, v),
        })
        .collect();
    hits.sort_unstable();
    hits.truncate(k);
    hits
}

// ─── tests ───────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn unit_vec(dim: usize, val: f32) -> Vec<f32> {
        vec![val; dim]
    }

    #[test]
    fn sq_l2_zero_on_equal() {
        let a = unit_vec(8, 1.0);
        assert_eq!(sq_l2(&a, &a), 0.0);
    }

    #[test]
    fn sq_l2_known_value() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        assert!((sq_l2(&a, &b) - 2.0).abs() < 1e-6);
    }

    #[test]
    fn cosine_identical_is_one() {
        let a = vec![1.0f32, 2.0, 3.0];
        let cos = cosine(&a, &a);
        assert!((cos - 1.0).abs() < 1e-6, "cos={cos}");
    }

    #[test]
    fn cosine_orthogonal_is_zero() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let cos = cosine(&a, &b);
        assert!(cos.abs() < 1e-6, "cos={cos}");
    }

    #[test]
    fn brute_force_returns_k_sorted() {
        let corpus: Vec<Vec<f32>> = (0..20u32).map(|i| vec![i as f32, 0.0]).collect();
        let query = vec![0.0f32, 0.0];
        let hits = brute_force_topk(&corpus, &query, 3);
        assert_eq!(hits.len(), 3);
        assert!(hits[0].dist <= hits[1].dist);
        assert!(hits[1].dist <= hits[2].dist);
        assert_eq!(hits[0].id, 0);
    }

    #[test]
    fn recall_at_k_perfect() {
        let gt: Vec<Hit> = (0..10)
            .map(|i| Hit {
                id: i,
                dist: i as f32,
            })
            .collect();
        let same = gt.clone();
        assert!((recall_at_k(&same, &gt, 10) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn recall_at_k_zero() {
        let gt: Vec<Hit> = (0..10)
            .map(|i| Hit {
                id: i,
                dist: i as f32,
            })
            .collect();
        let wrong: Vec<Hit> = (10..20)
            .map(|i| Hit {
                id: i,
                dist: i as f32,
            })
            .collect();
        assert!(recall_at_k(&wrong, &gt, 10).abs() < 1e-6);
    }
}
