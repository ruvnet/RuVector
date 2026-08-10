//! [`LinearCache`]: cosine-similarity scan over a bounded LRU cache.
//!
//! A new query is compared against every stored (query, result) pair.
//! If the maximum cosine similarity exceeds `threshold`, the cached result
//! is returned.  The cache uses a ring-buffer eviction policy: when full,
//! the oldest entry is overwritten.

use crate::{cosine_sim_unit, normalize, CacheStats, SearchResult, SemanticCache};

/// Fixed-threshold cosine cache with ring-buffer eviction.
pub struct LinearCache {
    queries: Vec<Vec<f32>>, // unit-normalised stored queries
    results: Vec<Vec<SearchResult>>,
    head: usize,  // next write position (ring buffer)
    filled: bool, // true once ring has wrapped
    capacity: usize,
    /// Cosine similarity threshold for a cache hit.
    pub threshold: f32,
    stats: CacheStats,
}

impl LinearCache {
    /// `capacity` — maximum number of (query, result) pairs to store.
    /// `threshold` — minimum cosine similarity to consider a hit (0.0–1.0).
    pub fn new(capacity: usize, threshold: f32) -> Self {
        assert!(capacity > 0, "capacity must be > 0");
        assert!(
            (0.0..=1.0).contains(&threshold),
            "threshold must be in [0.0, 1.0]"
        );
        Self {
            queries: Vec::with_capacity(capacity),
            results: Vec::with_capacity(capacity),
            head: 0,
            filled: false,
            capacity,
            threshold,
            stats: CacheStats::default(),
        }
    }

    fn active_len(&self) -> usize {
        if self.filled {
            self.capacity
        } else {
            self.head
        }
    }
}

impl SemanticCache for LinearCache {
    fn query(&mut self, q: &[f32]) -> Option<Vec<SearchResult>> {
        let t0 = std::time::Instant::now();
        self.stats.queries += 1;

        // Normalise the incoming query.
        let mut qn = q.to_vec();
        if !normalize(&mut qn) {
            self.stats.total_cache_lookup_ns += t0.elapsed().as_nanos() as u64;
            self.stats.misses += 1;
            return None;
        }

        let n = self.active_len();
        let mut best_sim = -1.0f32;
        let mut best_idx = usize::MAX;

        for i in 0..n {
            let sim = cosine_sim_unit(&qn, &self.queries[i]);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        self.stats.total_cache_lookup_ns += t0.elapsed().as_nanos() as u64;

        if best_sim >= self.threshold && best_idx < n {
            self.stats.hits += 1;
            Some(self.results[best_idx].clone())
        } else {
            self.stats.misses += 1;
            None
        }
    }

    fn insert(&mut self, q: Vec<f32>, results: Vec<SearchResult>) {
        let mut qn = q;
        if !normalize(&mut qn) {
            return;
        }

        if self.head < self.queries.len() {
            // Overwrite ring slot.
            self.queries[self.head] = qn;
            self.results[self.head] = results;
            self.stats.evictions += 1;
        } else {
            self.queries.push(qn);
            self.results.push(results);
        }

        self.head += 1;
        if self.head >= self.capacity {
            self.head = 0;
            self.filled = true;
        }
    }

    fn record_ann_latency(&mut self, ann_latency_ns: u64) {
        self.stats.total_ann_latency_ns += ann_latency_ns;
    }

    fn stats(&self) -> &CacheStats {
        &self.stats
    }

    fn capacity(&self) -> usize {
        self.capacity
    }

    fn len(&self) -> usize {
        self.active_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchResult;

    fn result(id: u32) -> SearchResult {
        SearchResult { id, distance: 0.0 }
    }

    #[test]
    fn linear_cache_exact_hit() {
        let mut c = LinearCache::new(8, 0.99);
        let q = vec![1.0f32, 0.0, 0.0];
        c.insert(q.clone(), vec![result(0)]);
        let hit = c.query(&q);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap()[0].id, 0);
    }

    #[test]
    fn linear_cache_near_duplicate_hit() {
        let mut c = LinearCache::new(8, 0.95);
        // Base vector
        let q0 = vec![1.0f32, 0.0, 0.0];
        c.insert(q0, vec![result(42)]);
        // Near duplicate: small noise in one dimension.
        let q1 = vec![1.0f32, 0.05, 0.0]; // angle ~3° off
        let hit = c.query(&q1);
        assert!(hit.is_some(), "near-duplicate should hit at threshold=0.95");
        assert_eq!(hit.unwrap()[0].id, 42);
    }

    #[test]
    fn linear_cache_orthogonal_miss() {
        let mut c = LinearCache::new(8, 0.95);
        c.insert(vec![1.0f32, 0.0, 0.0], vec![result(0)]);
        // Orthogonal query — cosine sim = 0.0
        let hit = c.query(&[0.0f32, 1.0, 0.0]);
        assert!(hit.is_none(), "orthogonal query must not hit");
    }

    #[test]
    fn linear_cache_ring_eviction() {
        let mut c = LinearCache::new(2, 0.99);
        c.insert(vec![1.0f32, 0.0], vec![result(0)]);
        c.insert(vec![0.0f32, 1.0], vec![result(1)]);
        c.insert(vec![0.5f32, 0.5], vec![result(2)]); // evicts slot 0
        assert_eq!(c.len(), 2);
    }

    #[test]
    fn linear_cache_hit_rate_on_duplicated_workload() {
        // 50% of queries are near-duplicates; expect hit rate > 30%.
        let mut c = LinearCache::new(32, 0.97);
        let base = vec![0.6f32, 0.8, 0.0];
        c.insert(base.clone(), vec![result(0)]);

        let mut hits = 0u32;
        let total = 100u32;
        for i in 0..total {
            let q = if i % 2 == 0 {
                // Near-duplicate: normalised base + tiny offset
                vec![0.6f32 + 0.001 * (i as f32), 0.8, 0.0]
            } else {
                // Orthogonal
                vec![0.0f32, 0.0, 1.0 + i as f32]
            };
            if c.query(&q).is_some() {
                hits += 1;
            } else {
                c.insert(q, vec![result(i)]);
            }
        }
        let hit_rate = hits as f64 / total as f64;
        // At least 30% of queries should hit (the near-duplicate half).
        assert!(
            hit_rate >= 0.30,
            "expected hit_rate >= 0.30, got {:.2}",
            hit_rate
        );
    }
}
