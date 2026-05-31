//! Alternative B: Coherence-tiered memory — hot/warm/cold driven by cosine
//! similarity to a running query centroid.
//!
//! After each search the query centroid is updated with exponential smoothing:
//!   centroid ← α * centroid + (1-α) * query   (α = 0.9)
//!
//! Vectors whose cosine similarity to the centroid exceeds `hot_threshold`
//! live in the hot tier; those above `warm_threshold` live in warm; the rest
//! are cold.  Re-tiering runs every `rebalance_every` inserts/searches.

use crate::{
    cosine_sim, fp32_bytes, l2_sq, q8_bytes, QuantizedVec, SearchResult, Tier, TierStats,
    TieredMemoryStore,
};

struct Entry {
    id: u64,
    vector: Vec<f32>,
    coherence: f32,
}

pub struct CoherenceTieredMemory {
    dims: usize,
    hot_threshold: f32,
    warm_threshold: f32,
    rebalance_every: usize,
    ops_since_rebalance: usize,
    alpha: f32,

    centroid: Vec<f32>,
    centroid_initialized: bool,

    hot: Vec<Entry>,
    warm: Vec<(u64, QuantizedVec)>,
    cold: Vec<Entry>,
}

impl CoherenceTieredMemory {
    /// Create a coherence-tiered store.
    ///
    /// * `hot_threshold`   — cosine sim to centroid above which a vector is hot (e.g. 0.7).
    /// * `warm_threshold`  — cosine sim above which a vector is warm (e.g. 0.3).
    /// * `rebalance_every` — re-tier all vectors after this many operations.
    pub fn new(
        dims: usize,
        hot_threshold: f32,
        warm_threshold: f32,
        rebalance_every: usize,
    ) -> Self {
        CoherenceTieredMemory {
            dims,
            hot_threshold,
            warm_threshold,
            rebalance_every,
            ops_since_rebalance: 0,
            alpha: 0.9,
            centroid: vec![0.0; dims],
            centroid_initialized: false,
            hot: Vec::new(),
            warm: Vec::new(),
            cold: Vec::new(),
        }
    }

    fn update_centroid(&mut self, query: &[f32]) {
        if !self.centroid_initialized {
            self.centroid.copy_from_slice(query);
            self.centroid_initialized = true;
        } else {
            for (c, q) in self.centroid.iter_mut().zip(query.iter()) {
                *c = self.alpha * *c + (1.0 - self.alpha) * q;
            }
        }
    }

    fn coherence_of(&self, v: &[f32]) -> f32 {
        if !self.centroid_initialized {
            return 0.0;
        }
        cosine_sim(v, &self.centroid)
    }

    fn rebalance(&mut self) {
        // Gather all vectors
        let mut all: Vec<Entry> = Vec::new();
        all.append(&mut self.hot);
        all.append(&mut self.cold);
        let warm_vec: Vec<(u64, QuantizedVec)> = self.warm.drain(..).collect();
        for (id, qvec) in warm_vec {
            let vector = qvec.decode();
            all.push(Entry {
                id,
                vector,
                coherence: 0.0,
            });
        }

        // Re-score and sort into tiers
        for e in all.iter_mut() {
            e.coherence = self.coherence_of(&e.vector);
        }

        for e in all {
            if e.coherence >= self.hot_threshold {
                self.hot.push(e);
            } else if e.coherence >= self.warm_threshold {
                let qvec = QuantizedVec::encode(&e.vector);
                self.warm.push((e.id, qvec));
            } else {
                self.cold.push(e);
            }
        }

        self.ops_since_rebalance = 0;
    }

    fn maybe_rebalance(&mut self) {
        self.ops_since_rebalance += 1;
        if self.ops_since_rebalance >= self.rebalance_every {
            self.rebalance();
        }
    }
}

impl TieredMemoryStore for CoherenceTieredMemory {
    fn name(&self) -> &str {
        "CoherenceTieredMemory (alt-B)"
    }

    fn insert(&mut self, id: u64, vector: Vec<f32>) {
        assert_eq!(vector.len(), self.dims);
        let coherence = self.coherence_of(&vector);
        if coherence >= self.hot_threshold {
            self.hot.push(Entry {
                id,
                vector,
                coherence,
            });
        } else if coherence >= self.warm_threshold {
            let qvec = QuantizedVec::encode(&vector);
            self.warm.push((id, qvec));
        } else {
            self.cold.push(Entry {
                id,
                vector,
                coherence,
            });
        }
        self.maybe_rebalance();
    }

    fn search(&mut self, query: &[f32], k: usize) -> Vec<SearchResult> {
        self.update_centroid(query);
        self.maybe_rebalance();

        let mut results: Vec<SearchResult> = Vec::with_capacity(k * 3);

        // Hot tier — exact L2 on full-precision
        for e in &self.hot {
            results.push(SearchResult {
                id: e.id,
                distance: l2_sq(query, &e.vector),
                tier: Tier::Hot,
            });
        }

        // Warm tier — approximate L2 on decoded quantized
        for (id, qvec) in &self.warm {
            let decoded = qvec.decode();
            results.push(SearchResult {
                id: *id,
                distance: l2_sq(query, &decoded),
                tier: Tier::Warm,
            });
        }

        // Cold tier — exact L2 but incurs simulated page-load cost
        for e in &self.cold {
            results.push(SearchResult {
                id: e.id,
                distance: l2_sq(query, &e.vector),
                tier: Tier::Cold,
            });
        }

        results.sort_by(|a, b| a.distance.partial_cmp(&b.distance).unwrap());
        results.truncate(k);
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
    fn coherence_finds_nearest() {
        let dims = 4;
        let mut store = CoherenceTieredMemory::new(dims, 0.7, 0.3, 100);
        for i in 0..20u64 {
            store.insert(i, vec![i as f32, 0.0, 0.0, 0.0]);
        }
        let results = store.search(&[7.0, 0.0, 0.0, 0.0], 3);
        assert_eq!(results.len(), 3);
        assert_eq!(results[0].id, 7);
    }

    #[test]
    fn coherence_rebalance_distributes_tiers() {
        let dims = 8;
        let mut store = CoherenceTieredMemory::new(dims, 0.8, 0.3, 5);

        // Seed query centroid with a "hot" direction
        let hot_dir: Vec<f32> = vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0];
        store.update_centroid(&hot_dir);

        // Insert vectors in hot direction
        for i in 0..10u64 {
            let scale = 1.0 + i as f32 * 0.01;
            store.insert(i, hot_dir.iter().map(|x| x * scale).collect());
        }
        // Insert cold vectors (orthogonal direction)
        for i in 10..20u64 {
            store.insert(i, vec![0.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0]);
        }
        // Force rebalance
        store.rebalance();

        let stats = store.tier_stats();
        // Hot vectors should dominate hot tier
        assert!(stats.hot_count >= 5, "hot={}", stats.hot_count);
        // Cold vectors should exist
        assert!(stats.cold_count >= 5, "cold={}", stats.cold_count);
        assert_eq!(stats.total_vectors(), 20);
    }

    #[test]
    fn centroid_converges_toward_queries() {
        let dims = 4;
        let mut store = CoherenceTieredMemory::new(dims, 0.7, 0.3, 100);
        store.insert(0, vec![1.0, 0.0, 0.0, 0.0]);
        // Repeatedly query in the same direction
        for _ in 0..20 {
            store.search(&[1.0, 0.0, 0.0, 0.0], 1);
        }
        // Centroid should be close to query direction
        let sim = cosine_sim(&store.centroid, &[1.0, 0.0, 0.0, 0.0]);
        assert!(sim > 0.99, "centroid_sim={sim}");
    }
}
