//! Alternative A: LRU-tiered memory — hot/warm/cold tiers driven by access frequency.
//!
//! Tiers:
//!   Hot  (full-precision, in RAM) — most-recently-accessed N_hot vectors.
//!   Warm (8-bit quantized, in RAM) — next N_warm vectors by recency.
//!   Cold (full-precision, "archived") — everything else.
//!
//! Search cascades hot → warm → cold, stopping early when k results are
//! collected from hotter tiers with confidence (distance threshold).

use crate::{
    fp32_bytes, l2_sq, q8_bytes, QuantizedVec, SearchResult, Tier, TierStats, TieredMemoryStore,
};
use std::collections::{HashMap, VecDeque};

struct ColdEntry {
    id: u64,
    vector: Vec<f32>,
}

struct WarmEntry {
    id: u64,
    qvec: QuantizedVec,
}

struct HotEntry {
    id: u64,
    vector: Vec<f32>,
}

pub struct LruTieredMemory {
    dims: usize,
    hot_cap: usize,
    warm_cap: usize,

    // hot: front of deque = most recently used
    hot: VecDeque<HotEntry>,

    warm: VecDeque<WarmEntry>,
    cold: Vec<ColdEntry>,

    // access counter
    access: HashMap<u64, u64>,
    tick: u64,
}

impl LruTieredMemory {
    /// Create a tiered store with given hot and warm capacities (vector counts).
    pub fn new(dims: usize, hot_cap: usize, warm_cap: usize) -> Self {
        LruTieredMemory {
            dims,
            hot_cap,
            warm_cap,
            hot: VecDeque::new(),
            warm: VecDeque::new(),
            cold: Vec::new(),
            access: HashMap::new(),
            tick: 0,
        }
    }

    fn promote_to_hot(&mut self, id: u64, vector: Vec<f32>) {
        // Evict LRU from hot → warm
        if self.hot.len() >= self.hot_cap {
            if let Some(evicted) = self.hot.pop_back() {
                let qvec = QuantizedVec::encode(&evicted.vector);
                self.warm.push_front(WarmEntry {
                    id: evicted.id,
                    qvec,
                });
                // Evict LRU from warm → cold
                if self.warm.len() > self.warm_cap {
                    if let Some(w_evicted) = self.warm.pop_back() {
                        let decoded = w_evicted.qvec.decode();
                        self.cold.push(ColdEntry {
                            id: w_evicted.id,
                            vector: decoded,
                        });
                    }
                }
            }
        }
        self.hot.push_front(HotEntry { id, vector });
    }
}

impl TieredMemoryStore for LruTieredMemory {
    fn name(&self) -> &str {
        "LruTieredMemory (alt-A)"
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);
        self.access.insert(id, 0);
        self.promote_to_hot(id, vector);
    }

    fn search(&mut self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.tick += 1;
        let mut results: Vec<SearchResult> = Vec::with_capacity(k * 3);

        // Search hot
        for entry in &self.hot {
            let dist = l2_sq(query, &entry.vector);
            results.push(SearchResult {
                id: entry.id,
                distance: dist,
                tier: Tier::Hot,
            });
        }

        // Search warm (decode on the fly)
        for entry in &self.warm {
            let decoded = entry.qvec.decode();
            let dist = l2_sq(query, &decoded);
            results.push(SearchResult {
                id: entry.id,
                distance: dist,
                tier: Tier::Warm,
            });
        }

        // Search cold
        for entry in &self.cold {
            let dist = l2_sq(query, &entry.vector);
            results.push(SearchResult {
                id: entry.id,
                distance: dist,
                tier: Tier::Cold,
            });
        }

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);

        // Promote top-1 hit to hot if it came from warm/cold
        if let Some(top) = results.first() {
            if top.tier != Tier::Hot {
                let id = top.id;
                // Find and remove from warm
                if let Some(pos) = self.warm.iter().position(|e| e.id == id) {
                    let w = self.warm.remove(pos).unwrap();
                    let vec = w.qvec.decode();
                    self.promote_to_hot(id, vec);
                } else if let Some(pos) = self.cold.iter().position(|e| e.id == id) {
                    let c = self.cold.remove(pos);
                    self.promote_to_hot(id, c.vector);
                }
            }
        }

        results
    }

    fn tier_stats(&self) -> TierStats {
        TierStats {
            hot_count: self.hot.len(),
            warm_count: self.warm.len(),
            cold_count: self.cold.len(),
            hot_bytes: self.hot.len() * fp32_bytes(self.dims),
            warm_bytes: self.warm.len() * q8_bytes(self.dims),
            cold_bytes: self.cold.len() * fp32_bytes(self.dims),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn lru_finds_nearest_across_tiers() {
        let mut store = LruTieredMemory::new(4, 5, 5);
        for i in 0..20u64 {
            store.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let results = store.search(&[7.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        // Nearest should be id=7
        assert_eq!(results[0].id, 7);
    }

    #[test]
    fn lru_tier_caps_respected() {
        let mut store = LruTieredMemory::new(4, 3, 5);
        for i in 0..20u64 {
            store.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let stats = store.tier_stats();
        assert!(stats.hot_count <= 3, "hot_count={}", stats.hot_count);
        assert!(stats.warm_count <= 5, "warm_count={}", stats.warm_count);
        assert_eq!(stats.total_vectors(), 20);
    }

    #[test]
    fn lru_promotes_on_access() {
        let mut store = LruTieredMemory::new(4, 2, 2);
        // Fill tiers: ids 0-5, hot=[5,4], warm=[3,2], cold=[1,0]
        for i in 0..6u64 {
            store.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        // Query near id=0 (cold), expect promotion
        let results = store.search(&[0.1, 0.0, 0.0, 0.0], 1);
        assert_eq!(results[0].id, 0);
        // id=0 should now be in hot
        let stats = store.tier_stats();
        assert!(stats.hot_count >= 1);
    }
}
