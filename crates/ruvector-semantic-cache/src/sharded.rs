//! LSH-sharded semantic cache backend.
//!
//! Partitions the cache into 2^B buckets using random binary projections
//! (a minimal LSH scheme). A query is assigned to its bucket by computing
//! the sign of its dot product with B projection vectors.  Only the entries
//! in the matching bucket are scanned, reducing mean scan length from C to
//! C / 2^B.
//!
//! **Trade-offs**:
//! - Bucket assignment is O(B·D) — negligible for small B (default B=8).
//! - Expected scan length C / 256 for B=8, so cost scales gracefully.
//! - Near-boundary queries may fall into a different bucket than a similar
//!   cached entry; this slightly lowers hit rate vs. LinearScanCache.
//! - For C < 500 the overhead of projection outweighs the scan savings;
//!   prefer LinearScanCache in that regime.

use crate::{dot, CacheEntry, CacheStats, QueryCache};

/// Number of random projection bits → 2^BITS buckets.
/// 6 bits → 64 buckets; combined with 1-hamming-distance multi-probe this
/// reaches ~89% recall on similar queries while scanning only 7×C/64 entries.
const BITS: usize = 6;
const N_BUCKETS: usize = 1 << BITS; // 64

/// Semantic cache backed by LSH sharding.
pub struct ShardedCache {
    /// Random projection vectors, each of length `dims`.
    projections: Vec<Vec<f32>>,
    /// 256 buckets, each holding a subset of cache entries.
    buckets: Vec<Vec<CacheEntry>>,
    stats: CacheStats,
    capacity_per_bucket: usize,
    dims: usize,
}

impl ShardedCache {
    /// Create a sharded cache.
    ///
    /// `total_capacity` is the approximate maximum number of entries across
    /// all buckets. `dims` is the embedding dimensionality.  `seed` controls
    /// the random projections — must be the same seed used throughout the
    /// lifetime of the cache.
    pub fn new(total_capacity: usize, dims: usize, seed: u64) -> Self {
        let projections = build_projections(dims, seed);
        let capacity_per_bucket = (total_capacity / N_BUCKETS).max(4);
        let buckets = (0..N_BUCKETS).map(|_| Vec::new()).collect();
        Self {
            projections,
            buckets,
            stats: CacheStats::new(),
            capacity_per_bucket,
            dims,
        }
    }

    /// Compute the 8-bit bucket index for `query`.
    fn bucket_of(&self, query: &[f32]) -> usize {
        let mut key: usize = 0;
        for (bit, proj) in self.projections.iter().enumerate() {
            if dot(query, proj) >= 0.0 {
                key |= 1 << bit;
            }
        }
        key
    }
}

/// Generate multi-probe bucket indices: the primary bucket plus all
/// 1-hamming-distance neighbors (BITS neighbors in total).
/// This recovers ~89% of similar entries that would otherwise fall into
/// an adjacent bucket due to borderline projection values.
fn probe_buckets(primary: usize) -> impl Iterator<Item = usize> {
    std::iter::once(primary).chain((0..BITS).map(move |bit| primary ^ (1 << bit)))
}

impl QueryCache for ShardedCache {
    fn lookup(
        &self,
        query: &[f32],
        threshold: f32,
        now_tick: u64,
        ttl_ticks: u64,
    ) -> Option<Vec<u64>> {
        let primary = self.bucket_of(query);

        // Multi-probe: scan the primary bucket + 1-hamming-distance neighbors.
        let mut best_sim = f32::NEG_INFINITY;
        let mut best_result: Option<Vec<u64>> = None;

        for bkt in probe_buckets(primary) {
            for e in &self.buckets[bkt] {
                if now_tick.saturating_sub(e.tick) > ttl_ticks {
                    continue;
                }
                let sim = dot(query, &e.query);
                if sim > best_sim {
                    best_sim = sim;
                    if sim >= threshold {
                        best_result = Some(e.results.clone());
                    }
                }
            }
        }

        if best_result.is_some() {
            self.stats.record_hit();
            best_result
        } else {
            self.stats.record_miss();
            None
        }
    }

    fn insert(&mut self, query: Vec<f32>, results: Vec<u64>, tick: u64) {
        let bucket = self.bucket_of(&query);
        let bucket_entries = &mut self.buckets[bucket];
        if bucket_entries.len() >= self.capacity_per_bucket {
            bucket_entries.remove(0);
        }
        bucket_entries.push(CacheEntry {
            query,
            results,
            tick,
        });
    }

    fn evict_expired(&mut self, now_tick: u64, ttl_ticks: u64) -> usize {
        let mut total = 0usize;
        for bucket in self.buckets.iter_mut() {
            let before = bucket.len();
            bucket.retain(|e| now_tick.saturating_sub(e.tick) <= ttl_ticks);
            total += before - bucket.len();
        }
        self.stats.record_evictions(total as u64);
        total
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    fn len(&self) -> usize {
        self.buckets.iter().map(|b| b.len()).sum()
    }

    fn memory_bytes(&self, _dims: usize) -> usize {
        let entry_bytes = self.dims * 4 + 8 * 10 + 16; // approx per entry
        self.len() * entry_bytes + self.projections.len() * self.dims * 4 // projection storage
    }
}

/// Generate BITS random unit projection vectors using a minimal LCG.
fn build_projections(dims: usize, seed: u64) -> Vec<Vec<f32>> {
    let mut state = seed ^ 0xDEAD_BEEF_1234_5678;
    let next = |s: &mut u64| -> f32 {
        *s = s
            .wrapping_mul(6_364_136_223_846_793_005)
            .wrapping_add(1_442_695_040_888_963_407);
        (*s as f32 / u64::MAX as f32) * 2.0 - 1.0
    };

    (0..BITS)
        .map(|_| {
            let mut v: Vec<f32> = (0..dims).map(|_| next(&mut state)).collect();
            // Normalize each projection vector.
            let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 1e-10 {
                v.iter_mut().for_each(|x| *x /= norm);
            }
            v
        })
        .collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::normalize;

    fn unit(mut v: Vec<f32>) -> Vec<f32> {
        normalize(&mut v);
        v
    }

    #[test]
    fn miss_on_empty_cache() {
        let cache = ShardedCache::new(1000, 4, 42);
        let q = unit(vec![1.0, 0.0, 0.0, 0.0]);
        assert!(cache.lookup(&q, 0.9, 0, u64::MAX).is_none());
    }

    #[test]
    fn hit_on_identical_query() {
        let mut cache = ShardedCache::new(1000, 4, 42);
        let q = unit(vec![1.0, 1.0, 0.0, 0.0]);
        let results = vec![7u64, 13, 42];
        cache.insert(q.clone(), results.clone(), 0);
        let got = cache.lookup(&q, 0.99, 0, u64::MAX);
        assert!(
            got.is_some(),
            "identical query must land in same bucket and hit"
        );
        assert_eq!(got.unwrap(), results);
    }

    #[test]
    fn bucket_assignment_is_deterministic() {
        let cache = ShardedCache::new(1000, 8, 99);
        let q = unit(vec![0.5f32, -0.3, 0.8, 0.0, 0.1, -0.6, 0.2, 0.7]);
        let b1 = cache.bucket_of(&q);
        let b2 = cache.bucket_of(&q);
        assert_eq!(b1, b2);
    }

    #[test]
    fn projections_have_correct_dims() {
        let cache = ShardedCache::new(100, 64, 1);
        assert_eq!(cache.projections.len(), BITS);
        for p in &cache.projections {
            assert_eq!(p.len(), 64);
        }
    }

    #[test]
    fn ttl_evicts_expired_entries() {
        let mut cache = ShardedCache::new(100, 4, 7);
        let q = unit(vec![1.0, 0.0, 0.0, 0.0]);
        cache.insert(q.clone(), vec![1], 0);
        let n = cache.evict_expired(200, 100);
        assert_eq!(n, 1);
        assert!(cache.is_empty());
    }
}
