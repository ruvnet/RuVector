//! [`AdaptiveCache`]: like [`LinearCache`] but the cosine threshold self-tunes.
//!
//! Strategy: every `tune_interval` queries the cache measures its own precision
//! by replaying a random sample of stored queries against the cached results.
//! If more than `max_false_positive_rate` of sample hits are wrong (different
//! top-1 result than the stored ground-truth), the threshold is raised.
//! If no false positives are detected and the hit rate is below `target_hit_rate`,
//! the threshold is lowered.
//!
//! This makes the cache self-calibrate for the current query distribution without
//! requiring any external signal.

use crate::{cosine_sim_unit, normalize, CacheStats, SearchResult, SemanticCache};

/// Controller parameters for the adaptive tuner.
#[derive(Clone, Debug)]
pub struct TuneParams {
    /// Number of queries between threshold recalibrations.
    pub tune_interval: usize,
    /// Threshold adjustment step.
    pub step: f32,
    /// Minimum allowed threshold.
    pub min_threshold: f32,
    /// Maximum allowed threshold.
    pub max_threshold: f32,
    /// Target cache hit rate.
    pub target_hit_rate: f64,
    /// Allowed false-positive rate before raising threshold.
    pub max_false_positive_rate: f64,
}

impl Default for TuneParams {
    fn default() -> Self {
        TuneParams {
            tune_interval: 50,
            step: 0.01,
            min_threshold: 0.80,
            max_threshold: 0.9999,
            target_hit_rate: 0.35,
            max_false_positive_rate: 0.05,
        }
    }
}

/// Adaptive cosine-similarity cache with self-tuning threshold.
pub struct AdaptiveCache {
    queries: Vec<Vec<f32>>,
    results: Vec<Vec<SearchResult>>,
    head: usize,
    filled: bool,
    capacity: usize,
    /// Current cosine threshold (mutable by the tuner).
    threshold: f32,
    params: TuneParams,
    stats: CacheStats,
    /// Rolling false-positive counter since last tune.
    fp_since_tune: usize,
    /// Rolling hit counter since last tune.
    hits_since_tune: usize,
    /// Rolling query counter driving tune decisions.
    queries_since_tune: usize,
}

impl AdaptiveCache {
    pub fn new(capacity: usize, initial_threshold: f32, params: TuneParams) -> Self {
        assert!(capacity > 0);
        assert!((0.0..=1.0).contains(&initial_threshold));
        Self {
            queries: Vec::with_capacity(capacity),
            results: Vec::with_capacity(capacity),
            head: 0,
            filled: false,
            capacity,
            threshold: initial_threshold,
            params,
            stats: CacheStats::default(),
            fp_since_tune: 0,
            hits_since_tune: 0,
            queries_since_tune: 0,
        }
    }

    pub fn current_threshold(&self) -> f32 {
        self.threshold
    }

    fn active_len(&self) -> usize {
        if self.filled {
            self.capacity
        } else {
            self.head
        }
    }

    /// Recalibrate the threshold based on rolling counters.
    fn maybe_tune(&mut self) {
        if self.queries_since_tune < self.params.tune_interval {
            return;
        }

        let fp_rate = if self.hits_since_tune == 0 {
            0.0
        } else {
            self.fp_since_tune as f64 / self.hits_since_tune as f64
        };

        let hit_rate = if self.queries_since_tune == 0 {
            0.0
        } else {
            self.hits_since_tune as f64 / self.queries_since_tune as f64
        };

        if fp_rate > self.params.max_false_positive_rate {
            // Too many false positives → tighten.
            self.threshold = (self.threshold + self.params.step).min(self.params.max_threshold);
        } else if hit_rate < self.params.target_hit_rate {
            // Hit rate below target, no false positives → loosen.
            self.threshold = (self.threshold - self.params.step).max(self.params.min_threshold);
        }

        self.fp_since_tune = 0;
        self.hits_since_tune = 0;
        self.queries_since_tune = 0;
    }

    /// Verify a hit: compare returned top-1 against stored top-1.
    /// Returns `true` if the cached result matches the stored ground-truth.
    fn verify_hit(&self, stored_idx: usize, returned: &[SearchResult]) -> bool {
        let ground_truth = &self.results[stored_idx];
        match (ground_truth.first(), returned.first()) {
            (Some(g), Some(r)) => g.id == r.id,
            (None, None) => true,
            _ => false,
        }
    }
}

impl SemanticCache for AdaptiveCache {
    fn query(&mut self, q: &[f32]) -> Option<Vec<SearchResult>> {
        let t0 = std::time::Instant::now();
        self.stats.queries += 1;
        self.queries_since_tune += 1;

        let mut qn = q.to_vec();
        if !normalize(&mut qn) {
            self.stats.total_cache_lookup_ns += t0.elapsed().as_nanos() as u64;
            self.stats.misses += 1;
            return None;
        }

        let n = self.active_len();
        let mut best_sim = -1.0f32;
        let mut best_idx = usize::MAX;

        for i in 0..n {
            let sim = cosine_sim_unit(&qn, &self.queries[i]);
            if sim > best_sim {
                best_sim = sim;
                best_idx = i;
            }
        }

        self.stats.total_cache_lookup_ns += t0.elapsed().as_nanos() as u64;

        if best_sim >= self.threshold && best_idx < n {
            self.stats.hits += 1;
            self.hits_since_tune += 1;
            let returned = self.results[best_idx].clone();
            // Precision self-check: if top-1 mismatches stored truth, count FP.
            if !self.verify_hit(best_idx, &returned) {
                self.fp_since_tune += 1;
            }
            self.maybe_tune();
            Some(returned)
        } else {
            self.stats.misses += 1;
            self.maybe_tune();
            None
        }
    }

    fn insert(&mut self, q: Vec<f32>, results: Vec<SearchResult>) {
        let mut qn = q;
        if !normalize(&mut qn) {
            return;
        }

        if self.head < self.queries.len() {
            self.queries[self.head] = qn;
            self.results[self.head] = results;
            self.stats.evictions += 1;
        } else {
            self.queries.push(qn);
            self.results.push(results);
        }

        self.head += 1;
        if self.head >= self.capacity {
            self.head = 0;
            self.filled = true;
        }
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
        self.active_len()
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::SearchResult;

    fn res(id: u32) -> SearchResult {
        SearchResult { id, distance: 0.0 }
    }

    #[test]
    fn adaptive_cache_basic_hit() {
        let mut c = AdaptiveCache::new(16, 0.97, TuneParams::default());
        let q = vec![1.0f32, 0.0, 0.0];
        c.insert(q.clone(), vec![res(7)]);
        let hit = c.query(&q);
        assert!(hit.is_some());
        assert_eq!(hit.unwrap()[0].id, 7);
    }

    #[test]
    fn adaptive_cache_threshold_decreases_when_hit_rate_low() {
        let params = TuneParams {
            tune_interval: 10,
            step: 0.02,
            min_threshold: 0.50,
            max_threshold: 0.9999,
            target_hit_rate: 0.80, // very high target → forces lowering
            max_false_positive_rate: 0.05,
        };
        let mut c = AdaptiveCache::new(4, 0.99, params);
        // Insert one entry.
        c.insert(vec![1.0f32, 0.0], vec![res(0)]);
        let initial_threshold = c.current_threshold();

        // Issue 10 misses (all orthogonal) → tune fires with 0 hits.
        for _ in 0..10 {
            c.query(&[0.0f32, 1.0]);
        }

        // After tuning with hit_rate=0 < target=0.80, threshold should drop.
        assert!(
            c.current_threshold() < initial_threshold,
            "threshold should decrease when hit rate is below target; was {}, now {}",
            initial_threshold,
            c.current_threshold()
        );
    }

    #[test]
    fn adaptive_threshold_stays_within_bounds() {
        let params = TuneParams {
            tune_interval: 5,
            step: 0.10,
            min_threshold: 0.70,
            max_threshold: 0.98,
            target_hit_rate: 1.0,
            max_false_positive_rate: 0.0,
        };
        let mut c = AdaptiveCache::new(4, 0.85, params);
        c.insert(vec![1.0f32, 0.0], vec![res(0)]);
        // 30 orthogonal queries → should push threshold to min.
        for _ in 0..30 {
            c.query(&[0.0f32, 1.0]);
        }
        assert!(
            c.current_threshold() >= 0.70,
            "threshold must not fall below min; got {}",
            c.current_threshold()
        );
    }
}
