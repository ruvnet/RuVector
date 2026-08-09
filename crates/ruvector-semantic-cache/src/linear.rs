//! Linear-scan semantic cache backend.
//!
//! Stores cached entries in a `Vec` and performs an exhaustive dot-product
//! scan to find the highest-similarity prior query. O(C·D) per lookup where
//! C is cache size and D is dimensionality.
//!
//! **Trade-offs**:
//! - Simple, no setup cost, optimal for small caches (C < ~1 000).
//! - Scan cost grows linearly; use [`crate::sharded::ShardedCache`] for C > 10 000.

use crate::{dot, CacheEntry, CacheStats, QueryCache};

/// Semantic cache backed by an exhaustive cosine-similarity scan.
pub struct LinearScanCache {
    entries: Vec<CacheEntry>,
    stats: CacheStats,
    /// Maximum number of entries; oldest are evicted on overflow.
    capacity: usize,
}

impl LinearScanCache {
    /// Create a cache with the given maximum capacity.
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: Vec::with_capacity(capacity.min(4096)),
            stats: CacheStats::new(),
            capacity,
        }
    }

    /// Find the entry with highest cosine similarity to `query`, returning
    /// `(similarity, index)` or `None` if the cache is empty.
    fn best_match(&self, query: &[f32], now_tick: u64, ttl_ticks: u64) -> Option<(f32, usize)> {
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_idx = usize::MAX;
        for (i, e) in self.entries.iter().enumerate() {
            // TTL check.
            if now_tick.saturating_sub(e.tick) > ttl_ticks {
                continue;
            }
            let sim = dot(query, &e.query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        if best_idx == usize::MAX {
            None
        } else {
            Some((best_sim, best_idx))
        }
    }
}

impl QueryCache for LinearScanCache {
    fn lookup(
        &self,
        query: &[f32],
        threshold: f32,
        now_tick: u64,
        ttl_ticks: u64,
    ) -> Option<Vec<u64>> {
        match self.best_match(query, now_tick, ttl_ticks) {
            Some((sim, idx)) if sim >= threshold => {
                self.stats.record_hit();
                Some(self.entries[idx].results.clone())
            }
            _ => {
                self.stats.record_miss();
                None
            }
        }
    }

    fn insert(&mut self, query: Vec<f32>, results: Vec<u64>, tick: u64) {
        if self.entries.len() >= self.capacity {
            // Evict the oldest entry (index 0).
            self.entries.remove(0);
        }
        self.entries.push(CacheEntry {
            query,
            results,
            tick,
        });
    }

    fn evict_expired(&mut self, now_tick: u64, ttl_ticks: u64) -> usize {
        let before = self.entries.len();
        self.entries
            .retain(|e| now_tick.saturating_sub(e.tick) <= ttl_ticks);
        let evicted = before - self.entries.len();
        self.stats.record_evictions(evicted as u64);
        evicted
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn memory_bytes(&self, dims: usize) -> usize {
        self.entries.len()
            * (dims * 4 + self.entries.first().map(|e| e.results.len()).unwrap_or(0) * 8 + 16)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;

    fn unit(v: Vec<f32>) -> Vec<f32> {
        let mut v = v;
        normalize(&mut v);
        v
    }

    #[test]
    fn miss_on_empty_cache() {
        let cache = LinearScanCache::new(100);
        let q = unit(vec![1.0, 0.0, 0.0]);
        assert!(cache.lookup(&q, 0.9, 0, u64::MAX).is_none());
    }

    #[test]
    fn hit_on_identical_query() {
        let mut cache = LinearScanCache::new(100);
        let q = unit(vec![1.0, 1.0, 0.0]);
        let results = vec![1u64, 2, 3];
        cache.insert(q.clone(), results.clone(), 0);
        let got = cache.lookup(&q, 0.99, 0, u64::MAX);
        assert!(got.is_some(), "identical query must hit");
        assert_eq!(got.unwrap(), results);
    }

    #[test]
    fn miss_when_similarity_below_threshold() {
        let mut cache = LinearScanCache::new(100);
        let q1 = unit(vec![1.0, 0.0]);
        let q2 = unit(vec![0.0, 1.0]); // orthogonal
        cache.insert(q1, vec![1], 0);
        assert!(cache.lookup(&q2, 0.5, 0, u64::MAX).is_none());
    }

    #[test]
    fn ttl_evicts_old_entries() {
        let mut cache = LinearScanCache::new(100);
        let q = unit(vec![1.0, 0.0]);
        cache.insert(q.clone(), vec![1], 0); // inserted at tick 0
                                             // now_tick=100, ttl=50 → entry age = 100 > 50 → expired
        let evicted = cache.evict_expired(100, 50);
        assert_eq!(evicted, 1);
        assert!(cache.is_empty());
    }

    #[test]
    fn capacity_evicts_oldest() {
        let mut cache = LinearScanCache::new(3);
        for i in 0u64..5 {
            let q = unit(vec![i as f32, 1.0]);
            cache.insert(q, vec![i], i);
        }
        assert_eq!(cache.len(), 3, "capacity must be enforced");
    }

    #[test]
    fn hit_rate_computation() {
        let mut cache = LinearScanCache::new(100);
        let q = unit(vec![1.0f32, 0.0]);
        cache.insert(q.clone(), vec![1], 0);
        // 1 hit.
        cache.lookup(&q, 0.99, 0, u64::MAX);
        // 1 miss (orthogonal).
        let q2 = unit(vec![0.0f32, 1.0]);
        cache.lookup(&q2, 0.5, 0, u64::MAX);
        let s = cache.stats();
        let rate = s.hit_rate();
        assert!((rate - 0.5).abs() < 1e-5, "hit_rate = {rate}");
    }
}
