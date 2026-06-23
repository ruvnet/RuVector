//! Semantic query result cache for RAG and agent memory.
//!
//! Three variants that trade cache hit rate against result quality:
//!
//! | Variant               | Cache key search | Threshold    | Adaptive |
//! |-----------------------|-----------------|--------------|---------|
//! | `NoCache`             | none            | —            | —       |
//! | `FixedSemanticCache`  | HNSW cosine     | fixed const  | no      |
//! | `AdaptiveSemanticCache` | HNSW cosine   | percentile   | yes     |
//!
//! **Semantic caching** stores (query_vector → search_results) pairs and serves
//! cached results when a new query is sufficiently similar to a prior query.
//! The inner key index is a lightweight HNSW built on L2-normalised query vectors.

pub mod dataset;
pub mod hnsw;

use hnsw::HnswGraph;

// ─── Core data types ─────────────────────────────────────────────────────────

/// A single cached result set.
#[derive(Clone, Debug)]
pub struct CacheEntry {
    /// The query vector (L2-normalised).
    pub query: Vec<f32>,
    /// Top-k result IDs from the underlying vector store.
    pub result_ids: Vec<u32>,
    /// Number of times this entry was served from cache.
    pub hit_count: u64,
    /// Monotonic insert serial number (used for LRU eviction).
    pub serial: u64,
}

/// Cache configuration.
#[derive(Clone, Debug)]
pub struct CacheConfig {
    pub dim: usize,
    /// Maximum number of entries before eviction.
    pub max_entries: usize,
    /// HNSW graph degree for the key index.
    pub hnsw_m: usize,
    pub hnsw_ef_construction: usize,
    /// ef used during cache key search.
    pub search_ef: usize,
    /// Fixed cosine similarity threshold for FixedSemanticCache.
    pub fixed_threshold: f32,
    /// Percentile (0–100) for adaptive threshold.
    pub adaptive_percentile: f32,
    /// Window of recent similarity observations for adaptive threshold.
    pub adaptive_window: usize,
}

impl CacheConfig {
    pub fn default_for_dim(dim: usize) -> Self {
        Self {
            dim,
            max_entries: 1024,
            hnsw_m: 16,
            hnsw_ef_construction: 100,
            search_ef: 50,
            fixed_threshold: 0.92,
            adaptive_percentile: 90.0,
            adaptive_window: 200,
        }
    }
}

/// Statistics collected over the lifetime of a cache.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub lookups: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    pub inserts: u64,
}

impl CacheStats {
    pub fn hit_rate(&self) -> f32 {
        if self.lookups == 0 {
            0.0
        } else {
            self.hits as f32 / self.lookups as f32
        }
    }
}

// ─── SemanticCache trait ──────────────────────────────────────────────────────

pub trait SemanticCache {
    /// Attempt to serve results from cache. Returns cached results if a
    /// sufficiently similar prior query exists, otherwise `None`.
    fn get(&mut self, query: &[f32]) -> Option<Vec<u32>>;

    /// Store a new (query, results) pair in the cache.
    fn put(&mut self, query: &[f32], result_ids: Vec<u32>);

    /// Invalidate the entire cache (e.g., after vector store update).
    fn invalidate(&mut self);

    fn stats(&self) -> &CacheStats;
    fn name(&self) -> &'static str;
}

// ─── Variant 1: NoCache (baseline — always miss) ─────────────────────────────

/// Baseline: no caching. Every lookup is a miss. Used to measure baseline
/// search latency without any cache infrastructure overhead.
pub struct NoCache {
    stats: CacheStats,
}

impl NoCache {
    pub fn new(_config: &CacheConfig) -> Self {
        Self {
            stats: CacheStats::default(),
        }
    }
}

impl SemanticCache for NoCache {
    fn get(&mut self, _query: &[f32]) -> Option<Vec<u32>> {
        self.stats.lookups += 1;
        self.stats.misses += 1;
        None
    }

    fn put(&mut self, _query: &[f32], _result_ids: Vec<u32>) {
        self.stats.inserts += 1;
    }

    fn invalidate(&mut self) {}

    fn stats(&self) -> &CacheStats {
        &self.stats
    }

    fn name(&self) -> &'static str {
        "NoCache"
    }
}

// ─── Shared cache storage ─────────────────────────────────────────────────────

struct CacheStore {
    config: CacheConfig,
    entries: Vec<CacheEntry>,
    /// HNSW index over normalised query vectors. Node ID = entries index.
    key_index: HnswGraph,
    serial: u64,
    stats: CacheStats,
}

impl CacheStore {
    fn new(config: CacheConfig) -> Self {
        let hcfg = hnsw::HnswConfig::new(config.dim, config.hnsw_m, config.hnsw_ef_construction);
        let key_index = HnswGraph::new(hcfg);
        Self {
            config,
            entries: Vec::new(),
            key_index,
            serial: 0,
            stats: CacheStats::default(),
        }
    }

    fn l2_normalize(v: &[f32]) -> Vec<f32> {
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        if norm < 1e-10 {
            v.to_vec()
        } else {
            v.iter().map(|x| x / norm).collect()
        }
    }

    /// Evict the least-recently-used entry. We evict the entry with the
    /// smallest `serial` (oldest insert). The HNSW does not support deletion,
    /// so we mark the slot as invalid by zeroing its query and result_ids.
    fn evict_lru(&mut self) {
        if self.entries.is_empty() {
            return;
        }
        let oldest = self
            .entries
            .iter()
            .enumerate()
            .filter(|(_, e)| !e.query.is_empty())
            .min_by_key(|(_, e)| e.serial)
            .map(|(i, _)| i);
        if let Some(idx) = oldest {
            self.entries[idx].query.clear();
            self.entries[idx].result_ids.clear();
            self.stats.evictions += 1;
        }
    }

    /// Search for the nearest cached query by cosine similarity.
    /// Returns (entry_idx, cosine_similarity) or None if cache is empty.
    fn nearest(&self, normed_query: &[f32]) -> Option<(usize, f32)> {
        if self.key_index.len() == 0 {
            return None;
        }
        let ef = self.config.search_ef;
        let results = self.key_index.search_by_cosine(normed_query, 1, ef);
        results.first().map(|&(id, sim)| (id, sim))
    }

    fn get_with_threshold(&mut self, query: &[f32], threshold: f32) -> Option<Vec<u32>> {
        self.stats.lookups += 1;
        let normed = Self::l2_normalize(query);
        match self.nearest(&normed) {
            Some((idx, sim)) if sim >= threshold => {
                if idx < self.entries.len() && !self.entries[idx].result_ids.is_empty() {
                    self.entries[idx].hit_count += 1;
                    // Update serial to mark as recently used
                    self.serial += 1;
                    self.entries[idx].serial = self.serial;
                    self.stats.hits += 1;
                    Some(self.entries[idx].result_ids.clone())
                } else {
                    self.stats.misses += 1;
                    None
                }
            }
            _ => {
                self.stats.misses += 1;
                None
            }
        }
    }

    fn put(&mut self, query: &[f32], result_ids: Vec<u32>) {
        if self.active_count() >= self.config.max_entries {
            self.evict_lru();
        }
        self.serial += 1;
        let normed = Self::l2_normalize(query);
        let entry = CacheEntry {
            query: normed.clone(),
            result_ids,
            hit_count: 0,
            serial: self.serial,
        };
        self.entries.push(entry);
        self.key_index.insert(normed);
        self.stats.inserts += 1;
    }

    fn invalidate(&mut self) {
        self.entries.clear();
        let hcfg = hnsw::HnswConfig::new(
            self.config.dim,
            self.config.hnsw_m,
            self.config.hnsw_ef_construction,
        );
        self.key_index = HnswGraph::new(hcfg);
    }

    fn active_count(&self) -> usize {
        self.entries.iter().filter(|e| !e.query.is_empty()).count()
    }
}

// ─── Variant 2: FixedSemanticCache ───────────────────────────────────────────

/// HNSW-backed semantic cache with a fixed cosine similarity threshold.
/// A cached entry is served when `cosine_sim(query, cached_query) >= threshold`.
pub struct FixedSemanticCache {
    store: CacheStore,
    threshold: f32,
}

impl FixedSemanticCache {
    pub fn new(config: CacheConfig) -> Self {
        let threshold = config.fixed_threshold;
        Self {
            store: CacheStore::new(config),
            threshold,
        }
    }
}

impl SemanticCache for FixedSemanticCache {
    fn get(&mut self, query: &[f32]) -> Option<Vec<u32>> {
        self.store.get_with_threshold(query, self.threshold)
    }

    fn put(&mut self, query: &[f32], result_ids: Vec<u32>) {
        self.store.put(query, result_ids);
    }

    fn invalidate(&mut self) {
        self.store.invalidate();
    }

    fn stats(&self) -> &CacheStats {
        &self.store.stats
    }

    fn name(&self) -> &'static str {
        "FixedSemanticCache"
    }
}

// ─── Variant 3: AdaptiveSemanticCache ────────────────────────────────────────

/// HNSW-backed semantic cache where the threshold adapts to the observed
/// distribution of cosine similarities between consecutive queries.
///
/// Strategy: maintain a sliding window of the last N similarity observations.
/// Use the Pth percentile as the threshold. This means the cache's "definition
/// of similar" tightens or loosens as the query distribution shifts.
pub struct AdaptiveSemanticCache {
    store: CacheStore,
    /// Sliding window of recent cosine similarity observations.
    observations: Vec<f32>,
    window: usize,
    percentile: f32,
    current_threshold: f32,
}

impl AdaptiveSemanticCache {
    pub fn new(config: CacheConfig) -> Self {
        let window = config.adaptive_window;
        let percentile = config.adaptive_percentile;
        // Start with a conservative threshold; adapt from first window of observations.
        let current_threshold = config.fixed_threshold;
        Self {
            store: CacheStore::new(config),
            observations: Vec::new(),
            window,
            percentile,
            current_threshold,
        }
    }

    fn update_threshold(&mut self, observed_sim: f32) {
        self.observations.push(observed_sim);
        if self.observations.len() > self.window {
            self.observations.remove(0);
        }
        if self.observations.len() >= 10 {
            let mut sorted = self.observations.clone();
            sorted.sort_by(|a, b| a.partial_cmp(b).unwrap());
            let idx = ((self.percentile / 100.0) * (sorted.len() - 1) as f32) as usize;
            self.current_threshold = sorted[idx.min(sorted.len() - 1)];
            // Clamp to reasonable range
            self.current_threshold = self.current_threshold.clamp(0.70, 0.999);
        }
    }
}

impl SemanticCache for AdaptiveSemanticCache {
    fn get(&mut self, query: &[f32]) -> Option<Vec<u32>> {
        self.store.stats.lookups += 1;
        let normed = CacheStore::l2_normalize(query);

        match self.store.nearest(&normed) {
            Some((idx, sim)) => {
                self.update_threshold(sim);
                if sim >= self.current_threshold
                    && idx < self.store.entries.len()
                    && !self.store.entries[idx].result_ids.is_empty()
                {
                    self.store.entries[idx].hit_count += 1;
                    self.store.serial += 1;
                    self.store.entries[idx].serial = self.store.serial;
                    self.store.stats.hits += 1;
                    Some(self.store.entries[idx].result_ids.clone())
                } else {
                    self.store.stats.misses += 1;
                    None
                }
            }
            None => {
                self.store.stats.misses += 1;
                None
            }
        }
    }

    fn put(&mut self, query: &[f32], result_ids: Vec<u32>) {
        self.store.put(query, result_ids);
    }

    fn invalidate(&mut self) {
        self.store.invalidate();
        self.observations.clear();
        self.current_threshold = 0.92;
    }

    fn stats(&self) -> &self::CacheStats {
        &self.store.stats
    }

    fn name(&self) -> &'static str {
        "AdaptiveSemanticCache"
    }
}

// ─── Tests ────────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn random_unit_vec(dim: usize, seed: u64) -> Vec<f32> {
        let mut v: Vec<f32> = (0..dim)
            .map(|i| {
                ((seed
                    .wrapping_mul(1664525)
                    .wrapping_add(1013904223)
                    .wrapping_add(i as u64)) as f32
                    / u64::MAX as f32)
                    - 0.5
            })
            .collect();
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt().max(1e-10);
        v.iter_mut().for_each(|x| *x /= norm);
        v
    }

    /// NoCache always returns None.
    #[test]
    fn test_no_cache_always_miss() {
        let cfg = CacheConfig::default_for_dim(64);
        let mut cache = NoCache::new(&cfg);
        let q = random_unit_vec(64, 42);
        cache.put(&q, vec![1, 2, 3]);
        assert!(cache.get(&q).is_none());
        assert_eq!(cache.stats().hit_rate(), 0.0);
    }

    /// Exact same query should be a cache hit above threshold.
    #[test]
    fn test_fixed_cache_exact_hit() {
        let cfg = CacheConfig {
            fixed_threshold: 0.95,
            ..CacheConfig::default_for_dim(64)
        };
        let mut cache = FixedSemanticCache::new(cfg);
        let q = random_unit_vec(64, 99);
        cache.put(&q, vec![10, 20, 30]);

        let result = cache.get(&q);
        assert!(result.is_some(), "identical query must be a cache hit");
        assert_eq!(result.unwrap(), vec![10, 20, 30]);
        assert_eq!(cache.stats().hits, 1);
    }

    /// Orthogonal query should miss.
    #[test]
    fn test_fixed_cache_orthogonal_miss() {
        let cfg = CacheConfig {
            fixed_threshold: 0.95,
            ..CacheConfig::default_for_dim(4)
        };
        let mut cache = FixedSemanticCache::new(cfg);
        // Store [1,0,0,0]
        cache.put(&[1.0, 0.0, 0.0, 0.0], vec![1]);
        // Query with [0,1,0,0] — cosine similarity = 0.0
        let result = cache.get(&[0.0, 1.0, 0.0, 0.0]);
        assert!(result.is_none(), "orthogonal query must miss");
    }

    /// Near-duplicate query (small perturbation) should hit at low threshold.
    #[test]
    fn test_fixed_cache_near_duplicate_hit() {
        let dim = 128;
        let cfg = CacheConfig {
            fixed_threshold: 0.90,
            ..CacheConfig::default_for_dim(dim)
        };
        let mut cache = FixedSemanticCache::new(cfg);
        let q = random_unit_vec(dim, 7);
        cache.put(&q, vec![5, 6, 7, 8, 9]);

        // Slightly perturbed version of q
        let mut q2 = q.clone();
        q2[0] += 0.01;
        let norm: f32 = q2.iter().map(|x| x * x).sum::<f32>().sqrt();
        q2.iter_mut().for_each(|x| *x /= norm);

        let result = cache.get(&q2);
        assert!(
            result.is_some(),
            "near-duplicate query should hit at threshold=0.90"
        );
    }

    /// Invalidation clears the cache.
    #[test]
    fn test_cache_invalidation() {
        let cfg = CacheConfig::default_for_dim(32);
        let mut cache = FixedSemanticCache::new(cfg);
        let q = random_unit_vec(32, 13);
        cache.put(&q, vec![1, 2]);
        cache.invalidate();
        assert!(
            cache.get(&q).is_none(),
            "after invalidation cache must be empty"
        );
    }

    /// Adaptive cache hit rate must be positive after warmup.
    #[test]
    fn test_adaptive_cache_warms_up() {
        let dim = 64;
        let cfg = CacheConfig {
            fixed_threshold: 0.85,
            adaptive_percentile: 85.0,
            adaptive_window: 20,
            ..CacheConfig::default_for_dim(dim)
        };
        let mut cache = AdaptiveSemanticCache::new(cfg);

        // Populate with 20 queries
        for seed in 0u64..20 {
            let q = random_unit_vec(dim, seed * 10);
            cache.put(&q, vec![seed as u32]);
        }

        // Re-query the same vectors — should see hits
        let mut hits = 0u32;
        for seed in 0u64..20 {
            let q = random_unit_vec(dim, seed * 10);
            if cache.get(&q).is_some() {
                hits += 1;
            }
        }
        assert!(
            hits > 0,
            "adaptive cache must have at least one hit on re-queried vectors"
        );
    }

    /// Hit rate threshold: FixedSemanticCache must achieve >= 50% hit rate when
    /// all test queries are near-duplicates of cached queries (perturbation < 1%).
    #[test]
    fn test_hit_rate_acceptance_threshold() {
        let dim = 128;
        let cfg = CacheConfig {
            fixed_threshold: 0.90,
            max_entries: 512,
            ..CacheConfig::default_for_dim(dim)
        };
        let mut cache = FixedSemanticCache::new(cfg);

        let base_queries: Vec<Vec<f32>> =
            (0u64..100).map(|s| random_unit_vec(dim, s * 37)).collect();

        for q in &base_queries {
            cache.put(q, vec![0]);
        }

        let mut hits = 0u32;
        for seed in 0u64..100 {
            let base = &base_queries[seed as usize % base_queries.len()];
            // Tiny perturbation to simulate near-duplicate query
            let mut q2 = base.clone();
            q2[0] += 0.001 * (seed as f32 / 100.0 - 0.5);
            let norm: f32 = q2.iter().map(|x| x * x).sum::<f32>().sqrt();
            q2.iter_mut().for_each(|x| *x /= norm);
            if cache.get(&q2).is_some() {
                hits += 1;
            }
        }
        let hit_rate = hits as f32 / 100.0;
        assert!(
            hit_rate >= 0.50,
            "hit rate {:.2} < 0.50 for near-duplicate queries at threshold=0.90",
            hit_rate
        );
    }
}
