//! Variant 1 — NoCache: fresh brute-force scan for every query.
//!
//! This is the ground-truth baseline. Every query is answered by an exact
//! O(n × dim) linear scan over the corpus. Hit rate is always 0%; recall is 1.0.

use crate::{brute_force_topk, CacheDecision, CacheStats, CachedAnn, Hit};

pub struct NoCache {
    corpus: Vec<Vec<f32>>,
    stats: CacheStats,
}

impl NoCache {
    pub fn new(corpus: Vec<Vec<f32>>) -> Self {
        Self {
            corpus,
            stats: CacheStats::default(),
        }
    }
}

impl CachedAnn for NoCache {
    fn search(&mut self, query: &[f32], k: usize) -> (Vec<Hit>, CacheDecision) {
        self.stats.misses += 1;
        let hits = brute_force_topk(&self.corpus, query, k);
        (hits, CacheDecision::Miss)
    }

    fn name(&self) -> &str {
        "NoCache"
    }

    fn stats(&self) -> CacheStats {
        self.stats.clone()
    }

    fn memory_bytes(&self) -> usize {
        self.corpus.len()
            * self.corpus.first().map(|v| v.len()).unwrap_or(0)
            * std::mem::size_of::<f32>()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn no_cache_always_miss() {
        let corpus: Vec<Vec<f32>> = (0..10).map(|i| vec![i as f32]).collect();
        let mut nc = NoCache::new(corpus);
        let (hits, dec) = nc.search(&[0.0], 3);
        assert_eq!(dec, CacheDecision::Miss);
        assert_eq!(hits.len(), 3);
        assert_eq!(nc.stats().hits, 0);
        assert_eq!(nc.stats().misses, 1);
    }

    #[test]
    fn no_cache_returns_sorted_hits() {
        let corpus: Vec<Vec<f32>> = vec![vec![5.0f32], vec![1.0f32], vec![3.0f32]];
        let mut nc = NoCache::new(corpus);
        let (hits, _) = nc.search(&[0.0], 2);
        assert!(hits[0].dist <= hits[1].dist);
        assert_eq!(hits[0].id, 1); // [1.0] is closest to [0.0]
    }
}
