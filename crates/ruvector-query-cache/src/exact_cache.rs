//! Variant 2 — ExactCache: bitwise-exact query match via FNV-like hash.
//!
//! Only returns a cached result when the incoming query vector is bitwise
//! identical to a stored query (same bit pattern on every f32 component).
//! In practice this hits only when the same query object is passed twice.
//! It is a useful lower bound: any hit-rate above this comes from semantic
//! approximation, not exact repetition.

use crate::{brute_force_topk, CacheDecision, CacheStats, CachedAnn, Hit};

/// Lightweight non-cryptographic hash over f32 bit patterns.
fn hash_query(q: &[f32]) -> u64 {
    let mut h: u64 = 0xcbf2_9ce4_8422_2325; // FNV-1a offset basis
    for &x in q {
        let bits = x.to_bits() as u64;
        h ^= bits;
        h = h.wrapping_mul(0x0000_0100_0000_01b3); // FNV-1a prime
        h ^= bits >> 32;
        h = h.wrapping_mul(0x0000_0100_0000_01b3);
    }
    h
}

struct CacheEntry {
    hash: u64,
    query: Vec<f32>,
    results: Vec<Hit>,
}

/// Fixed-capacity exact hash cache with LRU-style eviction (newest-in, oldest-out).
pub struct ExactCache {
    corpus: Vec<Vec<f32>>,
    capacity: usize,
    entries: Vec<CacheEntry>,
    stats: CacheStats,
}

impl ExactCache {
    pub fn new(corpus: Vec<Vec<f32>>, capacity: usize) -> Self {
        Self {
            corpus,
            capacity,
            entries: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    fn lookup(&self, query: &[f32]) -> Option<Vec<Hit>> {
        let h = hash_query(query);
        for e in &self.entries {
            if e.hash == h && e.query == query {
                return Some(e.results.clone());
            }
        }
        None
    }

    fn store(&mut self, query: Vec<f32>, results: Vec<Hit>) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0); // evict oldest
        }
        let hash = hash_query(&query);
        self.entries.push(CacheEntry {
            hash,
            query,
            results,
        });
    }
}

impl CachedAnn for ExactCache {
    fn search(&mut self, query: &[f32], k: usize) -> (Vec<Hit>, CacheDecision) {
        if let Some(cached) = self.lookup(query) {
            self.stats.hits += 1;
            return (cached, CacheDecision::Hit { similarity: 1.0 });
        }
        self.stats.misses += 1;
        let results = brute_force_topk(&self.corpus, query, k);
        self.store(query.to_vec(), results.clone());
        (results, CacheDecision::Miss)
    }

    fn name(&self) -> &str {
        "ExactCache"
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    fn memory_bytes(&self) -> usize {
        let corpus_bytes = self.corpus.len()
            * self.corpus.first().map(|v| v.len()).unwrap_or(0)
            * std::mem::size_of::<f32>();
        let cache_bytes = self
            .entries
            .iter()
            .map(|e| {
                e.query.len() * std::mem::size_of::<f32>()
                    + e.results.len() * std::mem::size_of::<Hit>()
                    + std::mem::size_of::<CacheEntry>()
            })
            .sum::<usize>();
        corpus_bytes + cache_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn tiny_corpus() -> Vec<Vec<f32>> {
        (0..20).map(|i| vec![i as f32, 0.0]).collect()
    }

    #[test]
    fn first_query_is_miss() {
        let mut c = ExactCache::new(tiny_corpus(), 64);
        let q = vec![0.0f32, 0.0];
        let (_, dec) = c.search(&q, 3);
        assert_eq!(dec, CacheDecision::Miss);
        assert_eq!(c.stats().misses, 1);
        assert_eq!(c.stats().hits, 0);
    }

    #[test]
    fn identical_query_hits() {
        let mut c = ExactCache::new(tiny_corpus(), 64);
        let q = vec![1.5f32, 0.0];
        c.search(&q, 3);
        let (_, dec) = c.search(&q, 3);
        assert_eq!(dec, CacheDecision::Hit { similarity: 1.0 });
        assert_eq!(c.stats().hits, 1);
    }

    #[test]
    fn slightly_different_query_is_miss() {
        let mut c = ExactCache::new(tiny_corpus(), 64);
        let q1 = vec![1.0f32, 0.0];
        let q2 = vec![1.0f32 + 1e-7, 0.0];
        c.search(&q1, 3);
        let (_, dec) = c.search(&q2, 3);
        assert_eq!(dec, CacheDecision::Miss);
    }

    #[test]
    fn capacity_eviction_works() {
        let mut c = ExactCache::new(tiny_corpus(), 2);
        let q1 = vec![0.1f32];
        let q2 = vec![0.2f32];
        let q3 = vec![0.3f32];
        c.search(&q1, 1);
        c.search(&q2, 1);
        c.search(&q3, 1); // should evict q1
        assert!(c.entries.len() <= 2);
    }
}
