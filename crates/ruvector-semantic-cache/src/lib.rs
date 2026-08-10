//! Semantic query cache for ANN search.
//!
//! Near-duplicate queries are common in agentic workloads — an agent asking
//! "recent memory about task planning" and then "past context about planning"
//! produce nearly identical query embeddings.  This crate intercepts those
//! duplicate calls before they hit the underlying ANN index.
//!
//! Three variants are provided:
//! - [`ExactCache`]    — hash-keyed, hits only on bit-identical queries (baseline)
//! - [`LinearCache`]   — cosine-similarity scan over cached queries, fixed threshold
//! - [`AdaptiveCache`] — like [`LinearCache`] but the threshold self-tunes based on
//!                       observed recall drift

pub mod adaptive;
pub mod dataset;
pub mod linear;

use std::collections::HashMap;
use std::hash::Hash;

// ─────────────────────────────────────────────
// Shared types
// ─────────────────────────────────────────────

/// A single nearest-neighbour result.
#[derive(Clone, Debug, PartialEq)]
pub struct SearchResult {
    pub id: u32,
    pub distance: f32,
}

/// Running cache statistics.
#[derive(Clone, Debug, Default)]
pub struct CacheStats {
    pub queries: u64,
    pub hits: u64,
    pub misses: u64,
    pub evictions: u64,
    /// Sum of ann_latency_ns values supplied via [`SemanticCache::record_ann_latency`].
    pub total_ann_latency_ns: u64,
    pub total_cache_lookup_ns: u64,
}

impl CacheStats {
    /// Fraction of queries served from cache.
    pub fn hit_rate(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.hits as f64 / self.queries as f64
        }
    }

    /// Mean ANN latency per miss (nanoseconds).
    pub fn mean_ann_latency_ns(&self) -> f64 {
        if self.misses == 0 {
            0.0
        } else {
            self.total_ann_latency_ns as f64 / self.misses as f64
        }
    }

    /// Mean cache lookup latency per query (nanoseconds).
    pub fn mean_lookup_ns(&self) -> f64 {
        if self.queries == 0 {
            0.0
        } else {
            self.total_cache_lookup_ns as f64 / self.queries as f64
        }
    }
}

/// Core cache trait.  Callers drive the protocol:
/// 1. Call [`SemanticCache::query`] — if `Some` is returned, a cache hit occurred.
/// 2. On `None` (miss), run your ANN search, measure `ann_latency_ns`, then call
///    [`SemanticCache::insert`] and [`SemanticCache::record_ann_latency`].
pub trait SemanticCache {
    fn query(&mut self, q: &[f32]) -> Option<Vec<SearchResult>>;
    fn insert(&mut self, q: Vec<f32>, results: Vec<SearchResult>);
    fn record_ann_latency(&mut self, ann_latency_ns: u64);
    fn stats(&self) -> &CacheStats;
    fn capacity(&self) -> usize;
    fn len(&self) -> usize;
}

// ─────────────────────────────────────────────
// Math helpers
// ─────────────────────────────────────────────

/// Dot product of two equal-length slices.
#[inline(always)]
pub fn dot(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| x * y).sum()
}

/// L2 norm of a slice.
#[inline(always)]
pub fn norm(v: &[f32]) -> f32 {
    dot(v, v).sqrt()
}

/// Normalise a vector in-place; returns `false` if the norm is near zero.
pub fn normalize(v: &mut [f32]) -> bool {
    let n = norm(v);
    if n < 1e-9 {
        return false;
    }
    for x in v.iter_mut() {
        *x /= n;
    }
    true
}

/// Cosine similarity of two pre-normalised vectors (safe for unit vectors).
#[inline(always)]
pub fn cosine_sim_unit(a: &[f32], b: &[f32]) -> f32 {
    dot(a, b).clamp(-1.0, 1.0)
}

// ─────────────────────────────────────────────
// ExactCache — baseline
// ─────────────────────────────────────────────

/// Baseline: only hits when the query bits are identical.
/// Establishes the trait-based abstraction with zero false-positive risk.
pub struct ExactCache {
    // Key: xxhash-style hash of the raw f32 bits.
    entries: HashMap<u64, (Vec<f32>, Vec<SearchResult>)>,
    capacity: usize,
    stats: CacheStats,
}

impl ExactCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: HashMap::with_capacity(capacity),
            capacity,
            stats: CacheStats::default(),
        }
    }
}

fn hash_f32_slice(v: &[f32]) -> u64 {
    let mut h = std::collections::hash_map::DefaultHasher::new();
    for &x in v {
        x.to_bits().hash(&mut h);
    }
    std::hash::Hasher::finish(&h)
}

impl SemanticCache for ExactCache {
    fn query(&mut self, q: &[f32]) -> Option<Vec<SearchResult>> {
        let t0 = std::time::Instant::now();
        self.stats.queries += 1;
        let key = hash_f32_slice(q);
        let result = self.entries.get(&key).map(|(_, results)| results.clone());
        self.stats.total_cache_lookup_ns += t0.elapsed().as_nanos() as u64;
        if result.is_some() {
            self.stats.hits += 1;
        } else {
            self.stats.misses += 1;
        }
        result
    }

    fn insert(&mut self, q: Vec<f32>, results: Vec<SearchResult>) {
        if self.entries.len() >= self.capacity {
            // Evict one arbitrary entry (simplest LRU approximation for baseline).
            if let Some(key) = self.entries.keys().next().copied() {
                self.entries.remove(&key);
                self.stats.evictions += 1;
            }
        }
        let key = hash_f32_slice(&q);
        self.entries.insert(key, (q, results));
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
        self.entries.len()
    }
}

// ─────────────────────────────────────────────
// Tests
// ─────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    fn make_result(id: u32) -> SearchResult {
        SearchResult {
            id,
            distance: 0.1 * id as f32,
        }
    }

    #[test]
    fn exact_cache_miss_then_hit() {
        let mut c = ExactCache::new(16);
        let q = vec![1.0f32, 0.0, 0.0];
        // First query — miss
        assert!(c.query(&q).is_none());
        c.insert(q.clone(), vec![make_result(0), make_result(1)]);
        c.record_ann_latency(1_000);
        // Second identical query — hit
        let res = c.query(&q);
        assert!(res.is_some());
        let res = res.unwrap();
        assert_eq!(res.len(), 2);
        assert_eq!(res[0].id, 0);
    }

    #[test]
    fn exact_cache_misses_near_duplicates() {
        let mut c = ExactCache::new(16);
        let q1 = vec![1.0f32, 0.0, 0.0];
        let q2 = vec![0.9999f32, 0.0001, 0.0]; // very close but not identical
        c.insert(q1.clone(), vec![make_result(0)]);
        // Near-duplicate must NOT hit in ExactCache
        assert!(c.query(&q2).is_none());
    }

    #[test]
    fn exact_cache_evicts_when_full() {
        let mut c = ExactCache::new(2);
        let r = vec![make_result(0)];
        c.insert(vec![1.0, 0.0], r.clone());
        c.insert(vec![0.0, 1.0], r.clone());
        c.insert(vec![0.5, 0.5], r.clone()); // triggers eviction
        assert_eq!(c.len(), 2);
        assert_eq!(c.stats().evictions, 1);
    }

    #[test]
    fn stats_track_hits_and_misses() {
        let mut c = ExactCache::new(16);
        let q = vec![1.0f32, 0.0];
        c.query(&q); // miss
        c.insert(q.clone(), vec![make_result(0)]);
        c.query(&q); // hit
        let s = c.stats();
        assert_eq!(s.queries, 2);
        assert_eq!(s.hits, 1);
        assert_eq!(s.misses, 1);
    }

    #[test]
    fn cosine_sim_unit_vectors() {
        let a = vec![1.0f32, 0.0, 0.0];
        let b = vec![1.0f32, 0.0, 0.0];
        assert!((cosine_sim_unit(&a, &b) - 1.0).abs() < 1e-6);

        let c = vec![0.0f32, 1.0, 0.0];
        assert!((cosine_sim_unit(&a, &c)).abs() < 1e-6);
    }

    #[test]
    fn normalize_produces_unit_vector() {
        let mut v = vec![3.0f32, 4.0, 0.0];
        normalize(&mut v);
        assert!((norm(&v) - 1.0).abs() < 1e-6);
    }
}
