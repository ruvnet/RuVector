//! Semantic Query Cache for RuVector.
//!
//! Agents repeatedly ask semantically equivalent questions. This crate caches
//! (query_vector → result_ids) pairs and uses cosine similarity to detect cache
//! hits for queries that are "close enough" to a previously answered query.
//!
//! Three backends are provided:
//! - [`NoCache`]        – always misses; establishes baseline DB throughput.
//! - [`LinearScanCache`] – O(C·D) scan over cached query vectors.
//! - [`ShardedCache`]   – LSH-bucketed cache; O(C/B·D) lookup for large caches.
//!
//! All query vectors are expected to be **pre-normalized** (unit L2 norm),
//! which reduces cosine similarity to a dot product.

pub mod dataset;
pub mod linear;
pub mod metrics;
pub mod sharded;

pub use metrics::CacheStats;

/// One cached (query → results) pair.
#[derive(Debug, Clone)]
pub struct CacheEntry {
    /// Pre-normalized query vector that produced these results.
    pub query: Vec<f32>,
    /// Ordered list of result vector IDs (top-k).
    pub results: Vec<u64>,
    /// Monotonic tick at insertion time (for TTL eviction).
    pub tick: u64,
}

/// Common interface for all cache backends.
pub trait QueryCache {
    /// Look up a cached result for `query`.
    ///
    /// Returns `Some(result_ids)` when a cached query exists with cosine
    /// similarity ≥ `threshold` and was inserted within `ttl_ticks` of
    /// `now_tick`. Returns `None` on miss.
    fn lookup(
        &self,
        query: &[f32],
        threshold: f32,
        now_tick: u64,
        ttl_ticks: u64,
    ) -> Option<Vec<u64>>;

    /// Insert a query–result pair into the cache.
    fn insert(&mut self, query: Vec<f32>, results: Vec<u64>, tick: u64);

    /// Remove all entries older than `ttl_ticks`. Returns number of evictions.
    fn evict_expired(&mut self, now_tick: u64, ttl_ticks: u64) -> usize;

    /// Snapshot of hit/miss counters.
    fn stats(&self) -> CacheStats;

    /// Current number of cached entries.
    fn len(&self) -> usize;

    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Estimated heap bytes used by stored query vectors and result IDs.
    fn memory_bytes(&self, dims: usize) -> usize;
}

/// Dot product of two pre-normalized (unit) vectors — equals cosine similarity.
#[inline]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    debug_assert_eq!(a.len(), b.len());
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2-normalize `v` in place. Returns `false` when the vector is near-zero.
pub fn normalize(v: &mut Vec<f32>) -> bool {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm < 1e-10 {
        return false;
    }
    v.iter_mut().for_each(|x| *x /= norm);
    true
}

/// Always-miss placeholder that forwards every query to the DB.
/// Used as the "no caching" baseline variant in benchmarks.
pub struct NoCache {
    stats: CacheStats,
}

impl NoCache {
    pub fn new() -> Self {
        Self {
            stats: CacheStats::default(),
        }
    }
}

impl Default for NoCache {
    fn default() -> Self {
        Self::new()
    }
}

impl QueryCache for NoCache {
    fn lookup(&self, _q: &[f32], _threshold: f32, _now: u64, _ttl: u64) -> Option<Vec<u64>> {
        self.stats.record_miss();
        None
    }

    fn insert(&mut self, _q: Vec<f32>, _r: Vec<u64>, _tick: u64) {}

    fn evict_expired(&mut self, _now: u64, _ttl: u64) -> usize {
        0
    }

    fn stats(&self) -> CacheStats {
        self.stats.snapshot()
    }

    fn len(&self) -> usize {
        0
    }

    fn memory_bytes(&self, _dims: usize) -> usize {
        0
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn normalize_unit_vector() {
        let mut v = vec![3.0f32, 4.0];
        normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-5, "norm = {norm}");
    }

    #[test]
    fn dot_identical_unit_vectors() {
        let v = vec![1.0f32 / 2.0f32.sqrt(), 1.0 / 2.0f32.sqrt()];
        let sim = dot(&v, &v);
        assert!((sim - 1.0).abs() < 1e-5, "dot = {sim}");
    }

    #[test]
    fn dot_orthogonal_vectors() {
        let a = vec![1.0f32, 0.0];
        let b = vec![0.0f32, 1.0];
        let sim = dot(&a, &b);
        assert!(sim.abs() < 1e-5, "dot = {sim}");
    }

    #[test]
    fn no_cache_always_misses() {
        let cache = NoCache::new();
        let q = vec![1.0f32, 0.0, 0.0];
        assert!(cache.lookup(&q, 0.9, 0, u64::MAX).is_none());
        let s = cache.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.hits, 0);
    }
}
