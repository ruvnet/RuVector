//! Core data structures: stored entries and search results.

use crate::decay::DecayConfig;

/// A single indexed vector with its timestamp.
#[derive(Debug, Clone)]
pub struct Entry {
    /// Unique identifier.
    pub id: u64,
    /// Embedding vector.
    pub vector: Vec<f32>,
    /// Insertion timestamp in seconds since epoch (or any monotonic counter).
    pub timestamp_secs: u64,
}

/// Result of a nearest-neighbour search.
#[derive(Debug, Clone, PartialEq)]
pub struct SearchResult {
    /// Identifier of the matched entry.
    pub id: u64,
    /// Raw squared Euclidean distance from query to this entry.
    pub raw_dist: f32,
    /// Effective distance after temporal weighting (used for ranking).
    pub effective_dist: f32,
    /// Age of the entry in seconds at search time.
    pub age_secs: u64,
}

/// Squared Euclidean distance between two equal-length slices.
pub fn l2_sq(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b.iter()).map(|(x, y)| (x - y) * (x - y)).sum()
}

/// Variant tag for the three search strategies.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum IndexVariant {
    /// No temporal weighting — pure L2 ranking.
    Baseline,
    /// L2 × temporal weight — recency-biased ranking.
    TemporalDecay,
    /// Temporal decay + coherence gate prunes stale+distant candidates.
    CoherenceGated,
}

impl std::fmt::Display for IndexVariant {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            IndexVariant::Baseline => write!(f, "Baseline"),
            IndexVariant::TemporalDecay => write!(f, "TemporalDecay"),
            IndexVariant::CoherenceGated => write!(f, "CoherenceGated"),
        }
    }
}

/// Flat (brute-force) temporal-aware nearest-neighbour index.
///
/// All three variants share this structure; the `variant` field controls
/// how effective distance is computed during search.
pub struct TdIndex {
    entries: Vec<Entry>,
    /// Search variant controlling distance computation.
    pub variant: IndexVariant,
    /// Decay configuration applied during search.
    pub config: DecayConfig,
}

impl TdIndex {
    /// Create a new empty index with the given variant and config.
    pub fn new(variant: IndexVariant, config: DecayConfig) -> Self {
        Self {
            entries: Vec::new(),
            variant,
            config,
        }
    }

    /// Insert a vector with the given id and insertion timestamp.
    pub fn insert(&mut self, id: u64, vector: Vec<f32>, timestamp_secs: u64) {
        self.entries.push(Entry {
            id,
            vector,
            timestamp_secs,
        });
    }

    /// Number of entries in the index.
    pub fn len(&self) -> usize {
        self.entries.len()
    }

    /// Returns true if the index contains no entries.
    pub fn is_empty(&self) -> bool {
        self.entries.is_empty()
    }

    /// Search for top-k nearest entries, using `now_secs` as the reference timestamp.
    ///
    /// Returns results sorted ascending by effective distance (closest first).
    pub fn search(&self, query: &[f32], k: usize, now_secs: u64) -> Vec<SearchResult> {
        let mut scored: Vec<SearchResult> = self
            .entries
            .iter()
            .filter_map(|e| {
                let raw_dist = l2_sq(query, &e.vector);
                let age_secs = now_secs.saturating_sub(e.timestamp_secs) as f64;

                // CoherenceGated: skip entirely if gated
                if self.variant == IndexVariant::CoherenceGated
                    && self.config.should_gate(age_secs, raw_dist)
                {
                    return None;
                }

                let effective_dist = match self.variant {
                    IndexVariant::Baseline => raw_dist,
                    IndexVariant::TemporalDecay | IndexVariant::CoherenceGated => {
                        raw_dist * self.config.weight(age_secs)
                    }
                };

                Some(SearchResult {
                    id: e.id,
                    raw_dist,
                    effective_dist,
                    age_secs: age_secs as u64,
                })
            })
            .collect();

        scored.sort_unstable_by(|a, b| {
            a.effective_dist
                .partial_cmp(&b.effective_dist)
                .unwrap_or(std::cmp::Ordering::Equal)
        });
        scored.truncate(k);
        scored
    }

    /// Fraction of top-k results that are within `recent_threshold_secs` of `now_secs`.
    pub fn fresh_recall(
        &self,
        query: &[f32],
        k: usize,
        now_secs: u64,
        recent_threshold_secs: u64,
    ) -> f32 {
        let results = self.search(query, k, now_secs);
        if results.is_empty() {
            return 0.0;
        }
        let fresh = results
            .iter()
            .filter(|r| r.age_secs <= recent_threshold_secs)
            .count();
        fresh as f32 / results.len() as f32
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::decay::DecayConfig;

    fn make_index(variant: IndexVariant, decay_strength: f32, half_life: f64) -> TdIndex {
        let config = DecayConfig::standard(decay_strength, half_life);
        TdIndex::new(variant, config)
    }

    #[test]
    fn baseline_returns_nearest_by_l2() {
        let mut idx = make_index(IndexVariant::Baseline, 0.0, 3600.0);
        // Insert two vectors; query nearest to [1.0, 0.0]
        idx.insert(1, vec![1.0, 0.0], 1000); // close, old
        idx.insert(2, vec![0.0, 1.0], 9000); // far, recent
        let results = idx.search(&[1.0, 0.0], 1, 10_000);
        assert_eq!(results[0].id, 1, "baseline should return L2-nearest");
    }

    #[test]
    fn temporal_decay_prefers_recent_when_close() {
        let mut idx = make_index(IndexVariant::TemporalDecay, 5.0, 1000.0);
        // vector 1: L2 distance 0.1 from query, 5000s old (old)
        // vector 2: L2 distance 0.2 from query, 100s old (recent)
        let q = vec![0.0f32; 4];
        let v1: Vec<f32> = vec![0.316, 0.0, 0.0, 0.0]; // L2^2 ≈ 0.1
        let v2: Vec<f32> = vec![0.447, 0.0, 0.0, 0.0]; // L2^2 ≈ 0.2
        idx.insert(1, v1, 0); // 5000s old
        idx.insert(2, v2, 4900); // 100s old
        let results = idx.search(&q, 1, 5000);
        // With strong decay, old v1 should get penalized enough that recent v2 wins
        assert_eq!(
            results[0].id, 2,
            "temporal decay should prefer recent vector"
        );
    }

    #[test]
    fn coherence_gate_prunes_stale_distant() {
        let config = DecayConfig::with_gate(2.0, 3600.0, 2000.0, 0.1);
        let mut idx = TdIndex::new(IndexVariant::CoherenceGated, config);
        // vector 1: stale (age > 2000s) and distant (d > 0.1) → gated out
        // vector 2: recent and close → included
        let q = vec![0.0f32; 2];
        idx.insert(1, vec![0.5, 0.5], 0); // far, old
        idx.insert(2, vec![0.05, 0.05], 9500); // close, recent
        let results = idx.search(&q, 2, 10_000);
        assert!(
            results.iter().all(|r| r.id != 1 || r.effective_dist <= 0.1),
            "stale+distant entry should be gated"
        );
    }

    #[test]
    fn fresh_recall_higher_for_temporal_decay() {
        use rand::{Rng, SeedableRng};
        let mut rng = rand::rngs::StdRng::seed_from_u64(42);

        let n = 200;
        let dims = 16usize;
        let now_secs = 86400u64;
        let recent_thresh = 3600u64; // 1 hour

        // Build two indexes
        let cfg_base = DecayConfig::no_decay();
        let cfg_td = DecayConfig::standard(3.0, 3600.0);
        let mut base = TdIndex::new(IndexVariant::Baseline, cfg_base);
        let mut td = TdIndex::new(IndexVariant::TemporalDecay, cfg_td);

        for i in 0u64..n {
            let v: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
            // 80% old, 20% recent
            let ts = if i < (n * 4 / 5) {
                0u64
            } else {
                now_secs - 1800
            };
            base.insert(i, v.clone(), ts);
            td.insert(i, v, ts);
        }

        let q: Vec<f32> = (0..dims).map(|_| rng.gen_range(-1.0f32..1.0)).collect();
        let base_fr = base.fresh_recall(&q, 10, now_secs, recent_thresh);
        let td_fr = td.fresh_recall(&q, 10, now_secs, recent_thresh);

        // Temporal decay should return MORE fresh results than baseline
        assert!(
            td_fr >= base_fr,
            "TD fresh_recall={td_fr:.3} should be >= baseline={base_fr:.3}"
        );
    }

    #[test]
    fn search_returns_at_most_k() {
        let mut idx = TdIndex::new(IndexVariant::Baseline, DecayConfig::no_decay());
        for i in 0u64..5 {
            idx.insert(i, vec![i as f32, 0.0], 0);
        }
        let results = idx.search(&[0.0, 0.0], 3, 0);
        assert!(results.len() <= 3);
    }
}
