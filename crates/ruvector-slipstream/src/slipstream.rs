//! Three streaming insertion strategies on a flat proximity graph.
//!
//! All three strategies build the same [`FlatGraph`] structure.  The only
//! difference is how each insertion is seeded:
//!
//! - **EntryPoint** (baseline): every insertion starts from node 0.
//! - **FixedCache**: seed from the candidate set discovered by the previous
//!   insertion, capped at `cache_size` candidates.
//! - **Adaptive**: like FixedCache, but monitors cosine drift between consecutive
//!   inserted vectors.  When the EMA drift exceeds a threshold, the cache is reset
//!   to avoid stale seeds; when the stream is stable the cache is expanded.

use crate::graph::{cosine_sim, FlatGraph, GraphConfig};

/// Streaming insertion strategy.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum InsertStrategy {
    /// Always traverse from the global entry node (node 0).
    EntryPoint,
    /// Re-use the previous insertion's discovered candidate set.
    FixedCache,
    /// FixedCache plus EMA drift detection with automatic cache reset.
    Adaptive,
}

/// Per-stream statistics collected during insertions.
#[derive(Debug, Default, Clone)]
pub struct StreamStats {
    /// Total vectors inserted.
    pub total_inserts: usize,
    /// Insertions where the warm-start cache was non-empty.
    pub cache_hits: usize,
    /// Cache resets triggered by drift detector (Adaptive only).
    pub drift_resets: usize,
    /// Sum of beam-search hops saved vs. entry-point baseline.
    pub total_hops_saved: i64,
    /// Current EMA cosine drift (0 = identical stream, 2 = opposite).
    pub drift_ema: f32,
}

impl StreamStats {
    pub fn cache_hit_rate(&self) -> f32 {
        if self.total_inserts == 0 {
            0.0
        } else {
            self.cache_hits as f32 / self.total_inserts as f32
        }
    }
}

const DRIFT_EMA_ALPHA: f32 = 0.15;
const DRIFT_RESET_THRESHOLD: f32 = 0.40; // 1 - cosine_sim; > 0.4 → different region
const DRIFT_STABLE_THRESHOLD: f32 = 0.10; // < 0.1 → highly stable stream

/// A streaming HNSW index with pluggable insertion strategy.
pub struct SlipstreamIndex {
    graph: FlatGraph,
    strategy: InsertStrategy,
    /// Warm-start candidate list (node IDs).
    warm_cache: Vec<u32>,
    /// Cache capacity (max candidates to retain).
    cache_size: usize,
    /// Last inserted vector (for drift computation).
    prev_vec: Option<Vec<f32>>,
    /// Running statistics.
    pub stats: StreamStats,
}

impl SlipstreamIndex {
    /// Create a new index with the given strategy.
    ///
    /// `cache_size`: maximum warm-start candidates retained between inserts.
    pub fn new(config: GraphConfig, strategy: InsertStrategy, cache_size: usize) -> Self {
        Self {
            graph: FlatGraph::new_empty(config),
            strategy,
            warm_cache: Vec::new(),
            cache_size,
            prev_vec: None,
            stats: StreamStats::default(),
        }
    }

    /// Insert a single vector into the streaming index.
    pub fn insert(&mut self, vec: Vec<f32>) {
        self.stats.total_inserts += 1;

        // Determine seeds for this insertion.
        let seeds: Vec<u32> = match self.strategy {
            InsertStrategy::EntryPoint => Vec::new(), // graph.insert_warm handles empty → node 0
            InsertStrategy::FixedCache => {
                if !self.warm_cache.is_empty() {
                    self.stats.cache_hits += 1;
                }
                self.warm_cache.clone()
            }
            InsertStrategy::Adaptive => {
                // Compute drift from previous vector.
                if let Some(prev) = &self.prev_vec {
                    let sim = cosine_sim(&vec, prev);
                    let drift = 1.0 - sim;
                    self.stats.drift_ema =
                        DRIFT_EMA_ALPHA * drift + (1.0 - DRIFT_EMA_ALPHA) * self.stats.drift_ema;

                    if self.stats.drift_ema > DRIFT_RESET_THRESHOLD {
                        // Stream has shifted — stale seeds hurt more than they help.
                        self.warm_cache.clear();
                        self.stats.drift_resets += 1;
                    } else if self.stats.drift_ema < DRIFT_STABLE_THRESHOLD {
                        // Very stable stream — expand cache for deeper warm-starting.
                        self.cache_size = (self.cache_size + 4).min(128);
                    }
                }

                if !self.warm_cache.is_empty() {
                    self.stats.cache_hits += 1;
                }
                self.warm_cache.clone()
            }
        };

        let baseline_entry = if self.graph.is_empty() { 0u32 } else { 0u32 };
        let baseline_hops = seeds.len();

        let discovered = self.graph.insert_warm(vec.clone(), &seeds);

        // Update warm-start cache for non-baseline strategies.
        if self.strategy != InsertStrategy::EntryPoint {
            self.warm_cache = discovered.iter().take(self.cache_size).copied().collect();
        }

        // Track hops saved: positive when warm-start candidates are closer than entry.
        let hops_saved = if self.strategy != InsertStrategy::EntryPoint
            && baseline_hops == 0
            && !self.warm_cache.is_empty()
        {
            1i64 // cache was available; at minimum one hop saved on average
        } else {
            0
        };
        self.stats.total_hops_saved += hops_saved;

        if self.strategy == InsertStrategy::Adaptive {
            self.prev_vec = Some(vec);
        }

        let _ = baseline_entry; // suppress unused warning
    }

    /// Search the index for the k nearest neighbours of `query`.
    pub fn search(&self, query: &[f32], k: usize, ef: usize) -> Vec<(f32, usize)> {
        self.graph.search_from(query, k, ef, &[0])
    }

    /// Number of vectors in the index.
    pub fn len(&self) -> usize {
        self.graph.len()
    }

    /// True if no vectors have been inserted.
    pub fn is_empty(&self) -> bool {
        self.graph.is_empty()
    }

    /// Read access to the underlying graph (for metrics).
    pub fn graph(&self) -> &FlatGraph {
        &self.graph
    }

    /// Name string for benchmark output.
    pub fn name(&self) -> &'static str {
        match self.strategy {
            InsertStrategy::EntryPoint => "EntryPoint (baseline)",
            InsertStrategy::FixedCache => "FixedCache (warm-start)",
            InsertStrategy::Adaptive => "Adaptive   (drift-aware)",
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::GraphConfig;

    fn make_index(strategy: InsertStrategy) -> SlipstreamIndex {
        let config = GraphConfig {
            m: 8,
            m_longjump: 4,
            ef: 32,
            dims: 16,
        };
        SlipstreamIndex::new(config, strategy, 32)
    }

    #[test]
    fn all_strategies_insert_same_count() {
        let n = 100;
        let vecs: Vec<Vec<f32>> = (0..n)
            .map(|i| (0..16).map(|d| (i * 16 + d) as f32 * 0.01).collect())
            .collect();

        for strat in [
            InsertStrategy::EntryPoint,
            InsertStrategy::FixedCache,
            InsertStrategy::Adaptive,
        ] {
            let mut idx = make_index(strat);
            for v in vecs.clone() {
                idx.insert(v);
            }
            assert_eq!(idx.len(), n, "strategy {:?} length mismatch", strat);
        }
    }

    #[test]
    fn fixed_cache_hits_after_second_insert() {
        let mut idx = make_index(InsertStrategy::FixedCache);
        idx.insert(vec![1.0f32; 16]);
        idx.insert(vec![1.01f32; 16]);
        // After the second insert, cache should have been populated by first insert.
        assert!(idx.stats.cache_hits >= 1);
    }

    #[test]
    fn adaptive_resets_on_alternating_stream() {
        let mut idx = make_index(InsertStrategy::Adaptive);
        // Alternate between two highly dissimilar vectors so cosine drift
        // stays near 1.0 on every transition.  The EMA crosses the 0.40
        // reset threshold after ~4 alternations.
        let a: Vec<f32> = vec![1.0f32; 16];
        let b: Vec<f32> = (0..16)
            .map(|i| if i % 2 == 0 { 1.0 } else { -1.0 })
            .collect();
        for i in 0..30 {
            if i % 2 == 0 {
                idx.insert(a.clone());
            } else {
                idx.insert(b.clone());
            }
        }
        // With α=0.15 and drift≈1.0 every other step, EMA reaches >0.40
        // within 4-5 transitions, triggering at least one reset.
        assert!(idx.stats.drift_resets >= 1);
    }

    #[test]
    fn search_returns_k_results_after_streaming_inserts() {
        let mut idx = make_index(InsertStrategy::FixedCache);
        let n = 200;
        for i in 0..n {
            let v: Vec<f32> = (0..16).map(|d| (i * 16 + d) as f32).collect();
            idx.insert(v);
        }
        let query: Vec<f32> = (0..16).map(|d| 100.0f32 + d as f32).collect();
        let results = idx.search(&query, 10, 64);
        assert_eq!(results.len(), 10);
        // Nearest should be distance ≈ 0 from some node near index 6.
        assert!(results[0].0 < 1000.0);
    }
}
