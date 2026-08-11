//! Three semantic cache variants used in the benchmark.
//!
//! Variant 1 — NoCache: every query hits the corpus scan. Baseline.
//! Variant 2 — SemanticCacheCoarse: cosine threshold = 0.90. Aggressive cache.
//!             High hit rate; some cached results may differ from fresh results.
//! Variant 3 — SemanticCacheFine: cosine threshold = 0.97. Conservative cache.
//!             Lower hit rate; cached results are nearly indistinguishable from fresh.
//!
//! Cache key matching: flat cosine scan over cached query vectors, O(|cache|·d).
//! Cache capacity: LRU eviction at max_entries. Invalidate-all on corpus mutation.

use crate::{cosine_similarity, SearchResult, SemanticCacheLayer};

// ─── NoCache ────────────────────────────────────────────────────────────────

/// Baseline: no caching. Every query is resolved by the caller against corpus.
pub struct NoCache;

impl SemanticCacheLayer for NoCache {
    fn lookup(&mut self, _query: &[f32]) -> Option<Vec<SearchResult>> {
        None
    }
    fn insert(&mut self, _query: Vec<f32>, _results: Vec<SearchResult>) {}
    fn invalidate_all(&mut self) {}
    fn len(&self) -> usize {
        0
    }
    fn name(&self) -> &str {
        "NoCache"
    }
}

// ─── Shared inner cache ──────────────────────────────────────────────────────

/// An entry in the semantic cache.
struct CacheEntry {
    query: Vec<f32>,
    results: Vec<SearchResult>,
    /// Logical access counter — updated on every hit for LRU ordering.
    last_used: u64,
}

/// Flat-scan semantic cache with configurable similarity threshold and LRU eviction.
pub struct FlatSemanticCache {
    entries: Vec<CacheEntry>,
    max_entries: usize,
    /// Minimum cosine similarity required to declare a cache hit.
    threshold: f32,
    access_counter: u64,
    /// Counts how many cache hits occurred.
    pub hits: u64,
    /// Counts how many cache misses occurred.
    pub misses: u64,
    variant_name: &'static str,
}

impl FlatSemanticCache {
    pub fn new(max_entries: usize, threshold: f32, variant_name: &'static str) -> Self {
        Self {
            entries: Vec::with_capacity(max_entries),
            max_entries,
            threshold,
            access_counter: 0,
            hits: 0,
            misses: 0,
            variant_name,
        }
    }

    fn evict_lru(&mut self) {
        // Remove the entry with the smallest last_used value.
        let oldest = self
            .entries
            .iter()
            .enumerate()
            .min_by_key(|(_, e)| e.last_used)
            .map(|(i, _)| i)
            .unwrap();
        self.entries.swap_remove(oldest);
    }

    /// Memory used by stored query vectors and result sets (approximate).
    pub fn memory_bytes(&self) -> usize {
        self.entries
            .iter()
            .fold(0, |acc, e| acc + e.query.len() * 4 + e.results.len() * 8)
    }
}

impl SemanticCacheLayer for FlatSemanticCache {
    fn lookup(&mut self, query: &[f32]) -> Option<Vec<SearchResult>> {
        self.access_counter += 1;
        let tick = self.access_counter;

        // Linear scan over cached queries: find the most similar one.
        let mut best_sim = -1.0_f32;
        let mut best_idx = usize::MAX;
        for (i, entry) in self.entries.iter().enumerate() {
            let sim = cosine_similarity(query, &entry.query);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        if best_idx < self.entries.len() && best_sim >= self.threshold {
            self.entries[best_idx].last_used = tick;
            self.hits += 1;
            Some(self.entries[best_idx].results.clone())
        } else {
            self.misses += 1;
            None
        }
    }

    fn insert(&mut self, query: Vec<f32>, results: Vec<SearchResult>) {
        self.access_counter += 1;
        let tick = self.access_counter;

        if self.entries.len() >= self.max_entries {
            self.evict_lru();
        }
        self.entries.push(CacheEntry {
            query,
            results,
            last_used: tick,
        });
    }

    fn invalidate_all(&mut self) {
        self.entries.clear();
        self.hits = 0;
        self.misses = 0;
    }

    fn len(&self) -> usize {
        self.entries.len()
    }

    fn name(&self) -> &str {
        self.variant_name
    }
}

/// Coarse semantic cache: threshold = 0.90. Maximises hit rate.
pub fn coarse(max_entries: usize) -> FlatSemanticCache {
    FlatSemanticCache::new(max_entries, 0.90, "SemanticCacheCoarse(t=0.90)")
}

/// Fine semantic cache: threshold = 0.97. Maximises result fidelity.
pub fn fine(max_entries: usize) -> FlatSemanticCache {
    FlatSemanticCache::new(max_entries, 0.97, "SemanticCacheFine(t=0.97)")
}

// ─── Recall helper ───────────────────────────────────────────────────────────

/// Overlap recall: |cached_ids ∩ fresh_ids| / k.
pub fn overlap_recall(cached: &[SearchResult], fresh: &[SearchResult]) -> f64 {
    let k = fresh.len();
    if k == 0 {
        return 1.0;
    }
    let hits = cached
        .iter()
        .filter(|c| fresh.iter().any(|f| f.id == c.id))
        .count();
    hits as f64 / k as f64
}

#[cfg(test)]
mod tests {
    use super::*;

    fn make_results(ids: &[u32]) -> Vec<SearchResult> {
        ids.iter()
            .enumerate()
            .map(|(i, &id)| SearchResult {
                id,
                distance: i as f32 * 0.1,
            })
            .collect()
    }

    #[test]
    fn no_cache_always_misses() {
        let mut c = NoCache;
        assert!(c.lookup(&[1.0, 0.0]).is_none());
        c.insert(vec![1.0, 0.0], make_results(&[0, 1, 2]));
        assert!(c.lookup(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn coarse_cache_hits_on_similar_query() {
        let mut cache = coarse(100);
        let q1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let q2 = vec![0.999f32, 0.01, 0.01, 0.01]; // very close to q1
                                                   // Normalise q2
        let norm: f32 = q2.iter().map(|x| x * x).sum::<f32>().sqrt();
        let q2n: Vec<f32> = q2.iter().map(|x| x / norm).collect();

        cache.insert(q1.clone(), make_results(&[5, 3, 7]));
        let hit = cache.lookup(&q2n);
        assert!(
            hit.is_some(),
            "expected a cache hit for near-identical query"
        );
    }

    #[test]
    fn fine_cache_misses_on_dissimilar_query() {
        let mut cache = fine(100);
        let q1 = vec![1.0f32, 0.0, 0.0, 0.0];
        let q2 = vec![0.0f32, 1.0, 0.0, 0.0]; // orthogonal → sim = 0
        cache.insert(q1, make_results(&[1, 2, 3]));
        let hit = cache.lookup(&q2);
        assert!(hit.is_none(), "orthogonal query must miss even fine cache");
    }

    #[test]
    fn lru_eviction_respects_capacity() {
        let mut cache = coarse(3);
        for i in 0..5u32 {
            // Each entry is a distinct direction.
            let mut q = vec![0.0f32; 8];
            q[(i as usize) % 8] = 1.0;
            cache.insert(q, make_results(&[i]));
        }
        assert_eq!(cache.len(), 3, "capacity=3 after 5 inserts");
    }

    #[test]
    fn invalidate_clears_cache() {
        let mut cache = fine(100);
        cache.insert(vec![1.0, 0.0], make_results(&[0]));
        cache.invalidate_all();
        assert_eq!(cache.len(), 0);
        assert!(cache.lookup(&[1.0, 0.0]).is_none());
    }

    #[test]
    fn overlap_recall_exact_match_returns_one() {
        let r = make_results(&[1, 2, 3, 4, 5]);
        assert!((overlap_recall(&r, &r) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn overlap_recall_no_match_returns_zero() {
        let a = make_results(&[1, 2, 3]);
        let b = make_results(&[4, 5, 6]);
        assert!((overlap_recall(&a, &b)).abs() < 1e-6);
    }
}
