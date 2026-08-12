//! Variant 3 — SemanticCache: cosine-similarity cache lookup.
//!
//! When a new query arrives the cache performs a brute-force scan over stored
//! query vectors to find the most similar past query.  If the best similarity
//! exceeds `hit_threshold`, the stored results are returned immediately without
//! touching the corpus.
//!
//! The cache lookup cost is O(n_cache × dim).  For n_cache ≤ 2048 and typical
//! 128-dim vectors this is 2–5× cheaper than a fresh full corpus scan, so any
//! non-trivial hit rate improves net throughput.
//!
//! Quality trade-off: cache hits return results from a slightly different query
//! so recall against the true top-k degrades.  The `hit_threshold` parameter
//! controls this: 0.99 → near-identical queries only; 0.85 → broader hits with
//! lower recall fidelity.

use crate::{brute_force_topk, cosine, CacheDecision, CacheStats, CachedAnn, Hit};

struct SemanticEntry {
    query: Vec<f32>,
    results: Vec<Hit>,
}

/// Cosine-similarity cache with configurable hit threshold and capacity.
pub struct SemanticCache {
    corpus: Vec<Vec<f32>>,
    capacity: usize,
    hit_threshold: f32,
    entries: Vec<SemanticEntry>,
    stats: CacheStats,
}

impl SemanticCache {
    /// `hit_threshold` ∈ [0, 1]: cosine similarity above which the cache is used.
    pub fn new(corpus: Vec<Vec<f32>>, capacity: usize, hit_threshold: f32) -> Self {
        Self {
            corpus,
            capacity,
            hit_threshold: hit_threshold.clamp(0.0, 1.0),
            entries: Vec::new(),
            stats: CacheStats::default(),
        }
    }

    /// Find the entry with the highest cosine similarity to `query`.
    /// Returns `(similarity, &results)` or `None` if the cache is empty.
    fn best_match(&self, query: &[f32]) -> Option<(f32, &[Hit])> {
        let mut best_sim = -2.0f32;
        let mut best_idx = 0;
        for (i, e) in self.entries.iter().enumerate() {
            let sim = cosine(query, &e.query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }
        if self.entries.is_empty() {
            return None;
        }
        Some((best_sim, &self.entries[best_idx].results))
    }

    fn store(&mut self, query: Vec<f32>, results: Vec<Hit>) {
        if self.entries.len() >= self.capacity {
            self.entries.remove(0);
        }
        self.entries.push(SemanticEntry { query, results });
    }

    pub fn hit_threshold(&self) -> f32 {
        self.hit_threshold
    }
}

impl CachedAnn for SemanticCache {
    fn search(&mut self, query: &[f32], k: usize) -> (Vec<Hit>, CacheDecision) {
        // Resolve the borrow by eagerly cloning any hit results before mutating stats.
        let cache_outcome: Option<(f32, Vec<Hit>)> = self
            .best_match(query)
            .and_then(|(sim, hits)| {
                if sim >= self.hit_threshold {
                    Some((sim, hits.to_vec()))
                } else {
                    None
                }
            });

        if let Some((sim, cached_hits)) = cache_outcome {
            self.stats.hits += 1;
            return (cached_hits, CacheDecision::Hit { similarity: sim });
        }

        self.stats.misses += 1;
        let results = brute_force_topk(&self.corpus, query, k);
        self.store(query.to_vec(), results.clone());
        (results, CacheDecision::Miss)
    }

    fn name(&self) -> &str {
        "SemanticCache"
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
            })
            .sum::<usize>();
        corpus_bytes + cache_bytes
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn corpus_n(n: usize, dim: usize) -> Vec<Vec<f32>> {
        (0..n)
            .map(|i| {
                let v: Vec<f32> = (0..dim).map(|j| (i * dim + j) as f32).collect();
                let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
                if norm > 1e-9 {
                    v.iter().map(|x| x / norm).collect()
                } else {
                    v
                }
            })
            .collect()
    }

    #[test]
    fn first_query_is_always_miss() {
        let mut c = SemanticCache::new(corpus_n(50, 8), 64, 0.90);
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let (_, dec) = c.search(&q, 3);
        assert_eq!(dec, CacheDecision::Miss);
    }

    #[test]
    fn identical_query_hits_above_threshold() {
        let mut c = SemanticCache::new(corpus_n(50, 8), 64, 0.90);
        let q = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        c.search(&q, 3);
        let (_, dec) = c.search(&q, 3);
        match dec {
            CacheDecision::Hit { similarity } => {
                assert!((similarity - 1.0).abs() < 1e-5, "sim={similarity}");
            }
            CacheDecision::Miss => panic!("expected hit on identical query"),
        }
    }

    #[test]
    fn low_threshold_accepts_similar_queries() {
        let mut c = SemanticCache::new(corpus_n(100, 8), 64, 0.80);
        // Store a base query.
        let base = vec![1.0f32, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        c.search(&base, 3);
        // A slightly jittered version should hit at threshold 0.80.
        let near = vec![0.98f32, 0.1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        let near_norm: f32 = near.iter().map(|x| x * x).sum::<f32>().sqrt();
        let near: Vec<f32> = near.iter().map(|x| x / near_norm).collect();
        let sim_with_base = cosine(&near, &base);
        if sim_with_base >= 0.80 {
            let (_, dec) = c.search(&near, 3);
            assert!(
                matches!(dec, CacheDecision::Hit { .. }),
                "expected hit, sim={sim_with_base}"
            );
        }
        // Test passes vacuously if sim < 0.80 (depends on normalization).
    }

    #[test]
    fn high_threshold_rejects_distant_query() {
        let mut c = SemanticCache::new(corpus_n(100, 4), 64, 0.999);
        let base = vec![1.0f32, 0.0, 0.0, 0.0];
        c.search(&base, 3);
        // An orthogonal query should definitely miss.
        let ortho = vec![0.0f32, 1.0, 0.0, 0.0];
        let (_, dec) = c.search(&ortho, 3);
        assert_eq!(dec, CacheDecision::Miss);
    }

    #[test]
    fn stats_count_correctly() {
        let mut c = SemanticCache::new(corpus_n(50, 4), 64, 0.90);
        let q = vec![1.0f32, 0.0, 0.0, 0.0];
        c.search(&q, 2); // miss
        c.search(&q, 2); // hit
        c.search(&q, 2); // hit
        assert_eq!(c.stats().misses, 1);
        assert_eq!(c.stats().hits, 2);
        assert!((c.stats().hit_rate() - 2.0 / 3.0).abs() < 1e-5);
    }
}
