//! Semantic Query Cache for ANN Search
//!
//! Three-variant experiment: NoCache (baseline), SemanticCacheCoarse (threshold=0.90),
//! SemanticCacheFine (threshold=0.97). Demonstrates that caching ANN result sets
//! keyed by query cosine similarity drastically cuts latency at the cost of a
//! bounded recall drop tunable via the similarity threshold.
//!
//! Reference: QVCache (arXiv 2602.02057, Feb 2026) — first paper to cache ANN
//! result sets (not LLM responses). No Rust native implementation existed before.

pub mod cache;
pub mod corpus;

/// A single search result: vector ID and its distance to the query.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

/// Statistics collected over a benchmark run.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    /// Sum of latencies in nanoseconds for cache hits.
    pub hit_latency_ns_sum: u64,
    /// Sum of latencies in nanoseconds for cache misses (corpus scan).
    pub miss_latency_ns_sum: u64,
    /// Recall quality sum over hits (|cached ∩ fresh| / k), accumulated for averaging.
    pub hit_recall_sum: f64,
    /// Number of hit recall measurements taken.
    pub hit_recall_count: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.hits as f64 / self.queries as f64
        }
    }

    pub fn mean_hit_latency_us(&self) -> f64 {
        if self.hits == 0 {
            0.0
        } else {
            self.hit_latency_ns_sum as f64 / self.hits as f64 / 1_000.0
        }
    }

    pub fn mean_miss_latency_us(&self) -> f64 {
        if self.misses == 0 {
            0.0
        } else {
            self.miss_latency_ns_sum as f64 / self.misses as f64 / 1_000.0
        }
    }

    pub fn mean_overall_latency_us(&self) -> f64 {
        if self.queries == 0 {
            return 0.0;
        }
        let total_ns = self.hit_latency_ns_sum + self.miss_latency_ns_sum;
        total_ns as f64 / self.queries as f64 / 1_000.0
    }

    pub fn mean_hit_recall(&self) -> f64 {
        if self.hit_recall_count == 0 {
            1.0
        } else {
            self.hit_recall_sum / self.hit_recall_count as f64
        }
    }

    pub fn throughput_qps(&self, elapsed_ns: u64) -> f64 {
        if elapsed_ns == 0 {
            0.0
        } else {
            self.queries as f64 / (elapsed_ns as f64 / 1_000_000_000.0)
        }
    }
}

/// Core trait for all cache variants.
pub trait SemanticCacheLayer: Send {
    /// Look up cached results for a query vector. Returns None on cache miss.
    fn lookup(&mut self, query: &[f32]) -> Option<Vec<SearchResult>>;

    /// Insert query vector and its result set into the cache.
    fn insert(&mut self, query: Vec<f32>, results: Vec<SearchResult>);

    /// Invalidate the entire cache. Must be called after any corpus mutation
    /// (insert, delete, update) to prevent stale results.
    fn invalidate_all(&mut self);

    /// Total number of entries currently in the cache.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Human-readable variant name.
    fn name(&self) -> &str;
}

/// Compute cosine similarity between two equal-length f32 slices.
/// Returns a value in [-1, 1]. Panics if slices differ in length.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    let mut dot = 0.0_f32;
    let mut norm_a = 0.0_f32;
    let mut norm_b = 0.0_f32;
    for i in 0..a.len() {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }
    let denom = norm_a.sqrt() * norm_b.sqrt();
    if denom < 1e-9 {
        0.0
    } else {
        dot / denom
    }
}

/// L2 (Euclidean) distance squared between two equal-length f32 slices.
pub fn l2_distance_sq(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| (x - y).powi(2)).sum()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn cosine_identical_vectors_returns_one() {
        let v = vec![1.0f32, 2.0, 3.0, 4.0];
        let sim = cosine_similarity(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "identical vectors: sim={sim}");
    }

    #[test]
    fn cosine_orthogonal_vectors_returns_zero() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![0.0f32, 1.0, 0.0];
        let sim = cosine_similarity(&a, &b);
        assert!(sim.abs() < 1e-5, "orthogonal vectors: sim={sim}");
    }

    #[test]
    fn cosine_opposite_vectors_returns_neg_one() {
        let a = vec![1.0f32, 1.0, 1.0];
        let b = vec![-1.0f32, -1.0, -1.0];
        let sim = cosine_similarity(&a, &b);
        assert!((sim + 1.0).abs() < 1e-5, "opposite: sim={sim}");
    }
}
