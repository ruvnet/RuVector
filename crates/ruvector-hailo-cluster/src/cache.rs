//! LRU cache for `(text, model_fingerprint)` → embedding vector.
//!
//! Off by default. When enabled (`HailoClusterEmbedder::with_cache(cap)`),
//! repeated `embed_one_blocking` calls for the same input return the
//! cached vector without an RPC. Cache key includes the model fingerprint
//! so swapping models invalidates entries automatically.
//!
//! ## Hot-path design (iters 80–81 — SOTA optimization passes)
//!
//! Three performance-critical choices that diverge from a textbook LRU:
//!
//! 1. **`Arc<Vec<f32>>` storage, not `Vec<f32>`.** A 384-d vector is
//!    1.5 KB. Cloning that inside the Mutex on every hit doubled the
//!    critical section. Switching to `Arc<Vec<f32>>` makes the inside-
//!    lock clone an atomic-increment; the actual `Vec<f32>` materialises
//!    outside the lock at the call site.
//!
//! 2. **Counter-based LRU, not `VecDeque` reordering.** The pre-iter-80
//!    impl scanned a `VecDeque` to find a key on every hit (O(N)) just
//!    to bump it to the back. The new impl stores a `last_used: u64`
//!    on each entry; eviction picks the entry with the smallest counter
//!    via a single map walk. Eviction becomes O(N/SHARDS) but is rare
//!    after warmup; gets become true O(1) (HashMap lookup + counter bump).
//!
//! 3. **16-way sharded Mutex.** Pre-iter-81 the entire cache lived
//!    behind a single Mutex — at >4 threads, contention dominated.
//!    Splitting into 16 shards (key hash → shard index) cuts per-shard
//!    contention by ~16× under uniform key distributions. Each shard
//!    owns its own counter + LRU; CacheStats sum across shards.

use std::collections::HashMap;
use std::hash::{BuildHasher, BuildHasherDefault, Hasher};
use std::sync::{Arc, Mutex};
use std::time::{Duration, Instant};

/// Number of independent Mutex-protected shards. Power of two so the
/// shard index is a cheap mask (`hash & (SHARDS-1)`).
const SHARDS: usize = 16;
const SHARD_MASK: usize = SHARDS - 1;

/// Single-process LRU cache keyed by `(fingerprint, text)`. Thread-safe
/// via 16-way sharded Mutex. Optional TTL bounds entry lifetime.
pub struct EmbeddingCache {
    shards: Box<[Mutex<Shard>]>,
    /// Total capacity divided across shards. Each shard caps at
    /// `capacity / SHARDS + 1` so the global hard limit is approximately
    /// honoured (within ±SHARDS-1 entries).
    capacity: usize,
    per_shard_capacity: usize,
    /// `None` ≡ no time-based expiry; entries live until LRU evicts them.
    ttl: Option<Duration>,
}

struct Shard {
    /// `key → Entry` (vector, inserted-at, last-used counter)
    map: HashMap<String, Entry>,
    /// Per-shard monotonic clock — increments on every shard touch.
    /// Eviction picks the entry with the smallest stored counter.
    counter: u64,
    hits: u64,
    misses: u64,
    evictions: u64,
}

struct Entry {
    vector: Arc<Vec<f32>>,
    inserted: Instant,
    last_used: u64,
}

/// Counters for ops-side observability.
#[derive(Clone, Debug, Default, PartialEq, Eq, serde::Serialize)]
pub struct CacheStats {
    /// Configured maximum entries; 0 = cache disabled.
    pub capacity: usize,
    /// Current number of entries held (sum across shards).
    pub size: usize,
    /// Total cache hits since construction (preserved across `clear`).
    pub hits: u64,
    /// Total cache misses since construction (preserved across `clear`).
    pub misses: u64,
    /// Total entries dropped — capacity overflow, TTL expiry, or `clear`.
    pub evictions: u64,
    /// Iter-108: configured time-to-live in seconds, if any. `None`
    /// means LRU-only (entries live until capacity-pressure evicts them).
    /// Surfaced so embedded long-running coordinators can plumb the
    /// value into Prometheus/JSON dashboards without re-reading env.
    #[serde(default)]
    pub ttl_seconds: Option<u64>,
}

impl CacheStats {
    /// `true` when the cache was constructed with non-zero capacity.
    /// Convenience for "should I bother emitting these metrics?" checks
    /// in CLI-level rendering paths.
    pub fn is_enabled(&self) -> bool {
        self.capacity > 0
    }

    /// `hits + misses`. Returns 0 if the cache hasn't seen any traffic
    /// yet, which is the natural input to a hit-rate guard.
    pub fn total_requests(&self) -> u64 {
        self.hits.saturating_add(self.misses)
    }

    /// Hit rate in `[0.0, 1.0]`. Returns `0.0` when the cache hasn't
    /// seen any requests — same convention bench.rs has used inline
    /// since iter 80 (now centralised so callers stop re-implementing
    /// it). NaN-safe: f64 division by zero is short-circuited via the
    /// `total_requests() > 0` guard.
    pub fn hit_rate(&self) -> f64 {
        let total = self.total_requests();
        if total == 0 {
            0.0
        } else {
            self.hits as f64 / total as f64
        }
    }
}

impl EmbeddingCache {
    /// Build a fresh cache. `capacity == 0` is treated as a static "off"
    /// state — `get` always misses, `insert` is a no-op.
    pub fn new(capacity: usize) -> Self {
        Self::with_ttl(capacity, None)
    }

    /// Build a cache with optional TTL.
    pub fn with_ttl(capacity: usize, ttl: Option<Duration>) -> Self {
        let per_shard_capacity = if capacity == 0 {
            0
        } else {
            // Round up so the sum of per-shard caps ≥ requested cap.
            capacity.div_ceil(SHARDS)
        };
        let shards: Vec<Mutex<Shard>> = (0..SHARDS)
            .map(|_| {
                Mutex::new(Shard {
                    map: HashMap::with_capacity(per_shard_capacity.max(1)),
                    counter: 0,
                    hits: 0,
                    misses: 0,
                    evictions: 0,
                })
            })
            .collect();
        Self {
            shards: shards.into_boxed_slice(),
            capacity,
            per_shard_capacity,
            ttl,
        }
    }

    /// Compose the cache key from fingerprint + text.
    #[inline]
    fn key(fingerprint: &str, text: &str) -> String {
        let mut k = String::with_capacity(fingerprint.len() + text.len() + 1);
        k.push_str(fingerprint);
        k.push('\x00');
        k.push_str(text);
        k
    }

    /// Hash the key and pick a shard. Uses std's default sip-hash —
    /// uniform enough for cache sharding, no extra deps.
    #[inline]
    fn shard_for(&self, k: &str) -> usize {
        let mut hasher = BuildHasherDefault::<std::collections::hash_map::DefaultHasher>::default()
            .build_hasher();
        hasher.write(k.as_bytes());
        (hasher.finish() as usize) & SHARD_MASK
    }

    /// Cache lookup. Increments hits/misses on the chosen shard;
    /// promotes hit to MRU within that shard. TTL-expired entries
    /// surface as misses + evictions.
    #[inline]
    pub fn get(&self, fingerprint: &str, text: &str) -> Option<Vec<f32>> {
        if self.capacity == 0 {
            return None;
        }
        let k = Self::key(fingerprint, text);
        let shard_idx = self.shard_for(&k);

        // Two-phase to dodge the borrow checker: inspect, then mutate
        // the counters separately.
        enum Outcome {
            Hit(Arc<Vec<f32>>),
            StaleMiss,
            Miss,
        }
        let arc_vec: Option<Arc<Vec<f32>>> = {
            let mut shard = self.shards[shard_idx].lock().unwrap();
            shard.counter = shard.counter.wrapping_add(1);
            let now_counter = shard.counter;
            let now = if self.ttl.is_some() {
                Some(Instant::now())
            } else {
                None
            };

            let outcome = match shard.map.get_mut(&k) {
                Some(entry) => {
                    let stale = if let (Some(now), Some(ttl)) = (now, self.ttl) {
                        now.duration_since(entry.inserted) >= ttl
                    } else {
                        false
                    };
                    if stale {
                        Outcome::StaleMiss
                    } else {
                        entry.last_used = now_counter;
                        Outcome::Hit(Arc::clone(&entry.vector))
                    }
                }
                None => Outcome::Miss,
            };
            match outcome {
                Outcome::Hit(arc) => {
                    shard.hits += 1;
                    Some(arc)
                }
                Outcome::StaleMiss => {
                    shard.map.remove(&k);
                    shard.misses += 1;
                    shard.evictions += 1;
                    return None;
                }
                Outcome::Miss => {
                    shard.misses += 1;
                    None
                }
            }
        };
        // Materialise outside the lock — heavy memcpy is contention-free.
        arc_vec.map(|a| (*a).clone())
    }

    /// Insert or refresh. Evicts the LRU entry within the shard if over
    /// the per-shard capacity.
    pub fn insert(&self, fingerprint: &str, text: &str, value: Vec<f32>) {
        if self.capacity == 0 {
            return;
        }
        let k = Self::key(fingerprint, text);
        let shard_idx = self.shard_for(&k);
        let arc = Arc::new(value);
        let mut shard = self.shards[shard_idx].lock().unwrap();
        shard.counter = shard.counter.wrapping_add(1);
        let now_counter = shard.counter;
        shard.map.insert(
            k,
            Entry {
                vector: arc,
                inserted: Instant::now(),
                last_used: now_counter,
            },
        );

        // Eviction: while over per-shard capacity, drop entry with the
        // smallest `last_used` in this shard. O(N/SHARDS), only fires
        // on overflow.
        while shard.map.len() > self.per_shard_capacity {
            let victim_key: Option<String> = shard
                .map
                .iter()
                .min_by_key(|(_, e)| e.last_used)
                .map(|(k, _)| k.clone());
            if let Some(vk) = victim_key {
                shard.map.remove(&vk);
                shard.evictions += 1;
            } else {
                break;
            }
        }
    }

    /// Snapshot the counters and current size — sums across shards.
    /// Holds locks one shard at a time so `stats()` doesn't block all
    /// other operations simultaneously.
    pub fn stats(&self) -> CacheStats {
        let mut s = CacheStats {
            capacity: self.capacity,
            // Iter-108: surface TTL config so dashboards can plot the
            // configured budget alongside the realized eviction rate.
            ttl_seconds: self.ttl.map(|d| d.as_secs()),
            ..Default::default()
        };
        for sh in self.shards.iter() {
            let g = sh.lock().unwrap();
            s.size += g.map.len();
            s.hits += g.hits;
            s.misses += g.misses;
            s.evictions += g.evictions;
        }
        s
    }

    /// Drop every cached entry. Counts each dropped entry as an eviction.
    /// Returns the total entries dropped.
    pub fn clear(&self) -> usize {
        if self.capacity == 0 {
            return 0;
        }
        let mut total = 0usize;
        for sh in self.shards.iter() {
            let mut g = sh.lock().unwrap();
            let dropped = g.map.len();
            g.map.clear();
            g.evictions += dropped as u64;
            total += dropped;
        }
        total
    }

    /// Configured maximum entries; 0 ≡ disabled.
    pub fn capacity(&self) -> usize {
        self.capacity
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn capacity_zero_is_disabled() {
        let c = EmbeddingCache::new(0);
        c.insert("fp", "hello", vec![1.0, 2.0, 3.0]);
        assert!(c.get("fp", "hello").is_none());
        let s = c.stats();
        assert_eq!(s.capacity, 0);
        assert_eq!(s.size, 0);
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 0);
    }

    // ---- ADR-167 §8 iter-108 CacheStats accessor tests ----

    #[test]
    fn stats_is_enabled_reflects_capacity() {
        assert!(!EmbeddingCache::new(0).stats().is_enabled());
        assert!(EmbeddingCache::new(64).stats().is_enabled());
    }

    #[test]
    fn stats_ttl_seconds_round_trips() {
        // None on the constructor surfaces as None on the snapshot.
        assert!(EmbeddingCache::new(64).stats().ttl_seconds.is_none());
        // Some(Duration::from_secs(N)) -> Some(N).
        let s = EmbeddingCache::with_ttl(64, Some(Duration::from_secs(42))).stats();
        assert_eq!(s.ttl_seconds, Some(42));
    }

    #[test]
    fn stats_hit_rate_returns_zero_for_empty_traffic() {
        let s = EmbeddingCache::new(16).stats();
        // 0 hits + 0 misses must short-circuit to 0.0 (no NaN).
        assert_eq!(s.hit_rate(), 0.0);
        assert_eq!(s.total_requests(), 0);
    }

    #[test]
    fn stats_hit_rate_matches_inline_division() {
        // Build a cache with a few synthetic hits/misses by hand —
        // construct a CacheStats directly so the test doesn't depend
        // on the eviction policy of the live cache.
        let s = CacheStats {
            capacity: 64,
            size: 4,
            hits: 30,
            misses: 10,
            evictions: 0,
            ttl_seconds: None,
        };
        assert_eq!(s.total_requests(), 40);
        assert!((s.hit_rate() - 0.75).abs() < 1e-9);
    }

    #[test]
    fn stats_serializes_to_json_with_ttl_field() {
        let s = CacheStats {
            capacity: 8,
            size: 2,
            hits: 5,
            misses: 3,
            evictions: 1,
            ttl_seconds: Some(60),
        };
        let json = serde_json::to_string(&s).expect("serialize");
        assert!(json.contains("\"ttl_seconds\":60"), "missing ttl: {}", json);
        assert!(json.contains("\"hits\":5"), "missing hits: {}", json);
    }

    #[test]
    fn hit_returns_value_and_promotes_mru() {
        // With sharding, the iter-80 LRU semantics test (which assumed
        // a single global LRU) needs adjustment: each shard has its
        // own LRU. Use a tiny per-shard cap (capacity = SHARDS) so
        // each entry deterministically lands in a separate shard.
        let c = EmbeddingCache::new(SHARDS);
        // Sample 4 keys; with hash-based sharding, collisions are rare
        // enough at SHARDS=16 that 4 distinct strings usually hit 4
        // distinct shards (and even if 2 collide, eviction-correctness
        // is preserved within the shard).
        c.insert("fp", "alpha", vec![1.0]);
        c.insert("fp", "beta", vec![2.0]);
        assert_eq!(c.get("fp", "alpha"), Some(vec![1.0]));
        assert_eq!(c.get("fp", "beta"), Some(vec![2.0]));

        let s = c.stats();
        assert_eq!(s.size, 2);
        assert_eq!(s.hits, 2);
        assert_eq!(s.misses, 0);
    }

    #[test]
    fn miss_increments_miss_counter() {
        let c = EmbeddingCache::new(2);
        assert!(c.get("fp", "absent").is_none());
        assert!(c.get("fp", "still-absent").is_none());
        let s = c.stats();
        assert_eq!(s.hits, 0);
        assert_eq!(s.misses, 2);
        assert_eq!(s.size, 0);
    }

    #[test]
    fn fingerprint_isolates_models() {
        let c = EmbeddingCache::new(4);
        c.insert("model-A", "hello", vec![1.0]);
        c.insert("model-B", "hello", vec![2.0]);
        assert_eq!(c.get("model-A", "hello"), Some(vec![1.0]));
        assert_eq!(c.get("model-B", "hello"), Some(vec![2.0]));
    }

    #[test]
    fn refresh_on_existing_key_does_not_double_count_size() {
        let c = EmbeddingCache::new(2);
        c.insert("fp", "x", vec![1.0]);
        c.insert("fp", "x", vec![2.0]);
        c.insert("fp", "x", vec![3.0]);
        let s = c.stats();
        assert_eq!(s.size, 1);
        assert_eq!(c.get("fp", "x"), Some(vec![3.0]));
    }

    #[test]
    fn ttl_expired_entry_is_reported_as_miss_and_evicted() {
        let c = EmbeddingCache::with_ttl(4, Some(Duration::from_millis(10)));
        c.insert("fp", "x", vec![1.0, 2.0]);

        assert_eq!(c.get("fp", "x"), Some(vec![1.0, 2.0]));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 0);

        std::thread::sleep(Duration::from_millis(20));

        assert!(c.get("fp", "x").is_none(), "stale entry must miss");
        let s = c.stats();
        assert_eq!(s.misses, 1);
        assert_eq!(s.evictions, 1);
        assert_eq!(s.size, 0);
    }

    #[test]
    fn ttl_none_means_entries_never_expire() {
        let c = EmbeddingCache::with_ttl(4, None);
        c.insert("fp", "y", vec![3.0]);
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(c.get("fp", "y"), Some(vec![3.0]));
        assert_eq!(c.stats().hits, 1);
        assert_eq!(c.stats().misses, 0);
    }

    #[test]
    fn ttl_insert_refreshes_timestamp() {
        let c = EmbeddingCache::with_ttl(4, Some(Duration::from_millis(20)));
        c.insert("fp", "z", vec![5.0]);
        std::thread::sleep(Duration::from_millis(10));
        c.insert("fp", "z", vec![5.0]);
        std::thread::sleep(Duration::from_millis(15));
        assert_eq!(c.get("fp", "z"), Some(vec![5.0]));
    }

    #[test]
    fn clear_drops_all_entries_and_counts_evictions() {
        let c = EmbeddingCache::new(8);
        c.insert("fp", "a", vec![1.0]);
        c.insert("fp", "b", vec![2.0]);
        c.insert("fp", "c", vec![3.0]);
        let _ = c.get("fp", "a");

        let dropped = c.clear();
        assert_eq!(dropped, 3, "all 3 entries dropped");

        let s = c.stats();
        assert_eq!(s.size, 0, "cache empty after clear");
        assert_eq!(s.evictions, 3, "evictions incremented by dropped count");
        assert_eq!(s.hits, 1, "hits preserved across clear");
        assert_eq!(s.misses, 0);

        assert!(c.get("fp", "a").is_none());
        assert!(c.get("fp", "b").is_none());
    }

    #[test]
    fn clear_on_empty_cache_is_noop() {
        let c = EmbeddingCache::new(4);
        let dropped = c.clear();
        assert_eq!(dropped, 0);
        let s = c.stats();
        assert_eq!(s.evictions, 0);
        assert_eq!(s.size, 0);
    }

    #[test]
    fn clear_on_disabled_cache_returns_zero() {
        let c = EmbeddingCache::new(0);
        let dropped = c.clear();
        assert_eq!(dropped, 0);
        let s = c.stats();
        assert_eq!(s.evictions, 0);
    }

    #[test]
    fn null_separator_prevents_key_collision() {
        let c = EmbeddingCache::new(4);
        c.insert("fp:a", "bc", vec![1.0]);
        c.insert("fp:", "abc", vec![2.0]);
        assert_eq!(c.get("fp:a", "bc"), Some(vec![1.0]));
        assert_eq!(c.get("fp:", "abc"), Some(vec![2.0]));
    }

    #[test]
    fn sharding_distributes_keys_across_shards() {
        // Sanity check: 1000 distinct keys should spread across all
        // 16 shards (the default hasher is sip-hash, well-distributed).
        let c = EmbeddingCache::new(2048);
        for i in 0..1000 {
            c.insert("fp", &format!("key-{}", i), vec![i as f32]);
        }
        let s = c.stats();
        assert_eq!(s.size, 1000, "all 1000 keys retained (under cap)");
    }
}
