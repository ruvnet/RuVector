//! Predictive anticipation and pre-fetching

use crate::causal::CausalGraph;
use crate::long_term::LongTermStore;
use crate::types::{Pattern, PatternId, Query, SearchResult};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;

/// Anticipation hint types
#[derive(Debug, Clone)]
pub enum AnticipationHint {
    /// Sequential pattern: if A then B
    SequentialPattern {
        /// Recent query patterns
        recent: Vec<PatternId>,
    },
    /// Temporal cycle (time-of-day patterns)
    TemporalCycle {
        /// Current temporal phase
        phase: TemporalPhase,
    },
    /// Causal chain prediction
    CausalChain {
        /// Current context pattern
        context: PatternId,
    },
}

/// Temporal phase for cyclic patterns
#[derive(Debug, Clone, Copy)]
pub enum TemporalPhase {
    /// Hour of day (0-23)
    HourOfDay(u8),
    /// Day of week (0-6)
    DayOfWeek(u8),
    /// Custom phase
    Custom(u32),
}

/// Prefetch cache for anticipated queries
pub struct PrefetchCache {
    /// Cached query results
    cache: DashMap<u64, Vec<SearchResult>>,
    /// Cache capacity
    capacity: usize,
    /// LRU tracking
    lru: Arc<RwLock<VecDeque<u64>>>,
}

impl PrefetchCache {
    /// Create new prefetch cache
    pub fn new(capacity: usize) -> Self {
        Self {
            cache: DashMap::new(),
            capacity,
            lru: Arc::new(RwLock::new(VecDeque::with_capacity(capacity))),
        }
    }

    /// Insert into cache
    pub fn insert(&self, query_hash: u64, results: Vec<SearchResult>) {
        // Check capacity
        if self.cache.len() >= self.capacity {
            self.evict_lru();
        }

        // Insert
        self.cache.insert(query_hash, results);

        // Update LRU
        let mut lru = self.lru.write();
        lru.push_back(query_hash);
    }

    /// Get from cache
    pub fn get(&self, query_hash: u64) -> Option<Vec<SearchResult>> {
        self.cache.get(&query_hash).map(|v| v.clone())
    }

    /// Evict least recently used entry
    fn evict_lru(&self) {
        let mut lru = self.lru.write();
        if let Some(key) = lru.pop_front() {
            self.cache.remove(&key);
        }
    }

    /// Clear cache
    pub fn clear(&self) {
        self.cache.clear();
        self.lru.write().clear();
    }

    /// Get cache size
    pub fn len(&self) -> usize {
        self.cache.len()
    }

    /// Check if cache is empty
    pub fn is_empty(&self) -> bool {
        self.cache.is_empty()
    }
}

impl Default for PrefetchCache {
    fn default() -> Self {
        Self::new(1000)
    }
}

/// Sequential pattern tracker
pub struct SequentialPatternTracker {
    /// Pattern sequences (A -> B)
    sequences: DashMap<PatternId, Vec<PatternId>>,
    /// Pattern counts
    counts: DashMap<(PatternId, PatternId), usize>,
}

impl SequentialPatternTracker {
    /// Create new tracker
    pub fn new() -> Self {
        Self {
            sequences: DashMap::new(),
            counts: DashMap::new(),
        }
    }

    /// Record sequence: A followed by B
    pub fn record_sequence(&self, from: PatternId, to: PatternId) {
        // Add to sequences
        self.sequences
            .entry(from)
            .or_insert_with(Vec::new)
            .push(to);

        // Increment count
        *self.counts.entry((from, to)).or_insert(0) += 1;
    }

    /// Predict next pattern given current
    pub fn predict_next(&self, current: PatternId, top_k: usize) -> Vec<PatternId> {
        if let Some(nexts) = self.sequences.get(&current) {
            // Count frequencies
            let mut freq_map: std::collections::HashMap<PatternId, usize> =
                std::collections::HashMap::new();

            for &next in nexts.iter() {
                *freq_map.entry(next).or_insert(0) += 1;
            }

            // Sort by frequency
            let mut sorted: Vec<_> = freq_map.into_iter().collect();
            sorted.sort_by_key(|(_, count)| std::cmp::Reverse(*count));

            // Return top k
            sorted.into_iter().take(top_k).map(|(id, _)| id).collect()
        } else {
            Vec::new()
        }
    }
}

impl Default for SequentialPatternTracker {
    fn default() -> Self {
        Self::new()
    }
}

/// Anticipate future queries and pre-fetch
pub fn anticipate(
    hints: &[AnticipationHint],
    long_term: &LongTermStore,
    causal_graph: &CausalGraph,
    prefetch_cache: &PrefetchCache,
    sequential_tracker: &SequentialPatternTracker,
) -> usize {
    let mut num_prefetched = 0;

    for hint in hints {
        match hint {
            AnticipationHint::SequentialPattern { recent } => {
                // Predict next based on recent patterns
                if let Some(&last) = recent.last() {
                    let predicted = sequential_tracker.predict_next(last, 5);

                    for pattern_id in predicted {
                        if let Some(pattern) = long_term.get(&pattern_id) {
                            // Create query from pattern
                            let query = Query::from_embedding(pattern.embedding.clone());
                            let query_hash = query.hash();

                            // Pre-fetch if not cached
                            if prefetch_cache.get(query_hash).is_none() {
                                let results = long_term.search(&query);
                                prefetch_cache.insert(query_hash, results);
                                num_prefetched += 1;
                            }
                        }
                    }
                }
            }

            AnticipationHint::TemporalCycle { phase: _ } => {
                // TODO: Implement temporal cycle prediction
                // Would track queries by time-of-day/day-of-week
                // and pre-fetch commonly accessed patterns for current phase
            }

            AnticipationHint::CausalChain { context } => {
                // Predict downstream patterns in causal graph
                let downstream = causal_graph.causal_future(*context);

                for pattern_id in downstream.into_iter().take(5) {
                    if let Some(pattern) = long_term.get(&pattern_id) {
                        let query = Query::from_embedding(pattern.embedding.clone());
                        let query_hash = query.hash();

                        // Pre-fetch if not cached
                        if prefetch_cache.get(query_hash).is_none() {
                            let results = long_term.search(&query);
                            prefetch_cache.insert(query_hash, results);
                            num_prefetched += 1;
                        }
                    }
                }
            }
        }
    }

    num_prefetched
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_prefetch_cache() {
        let cache = PrefetchCache::new(2);

        let results1 = vec![];
        let results2 = vec![];

        cache.insert(1, results1);
        cache.insert(2, results2);

        assert_eq!(cache.len(), 2);
        assert!(cache.get(1).is_some());

        // Insert third should evict first (LRU)
        cache.insert(3, vec![]);
        assert_eq!(cache.len(), 2);
        assert!(cache.get(1).is_none());
    }

    #[test]
    fn test_sequential_tracker() {
        let tracker = SequentialPatternTracker::new();

        let p1 = PatternId::new();
        let p2 = PatternId::new();
        let p3 = PatternId::new();

        // p1 -> p2 (twice)
        tracker.record_sequence(p1, p2);
        tracker.record_sequence(p1, p2);

        // p1 -> p3 (once)
        tracker.record_sequence(p1, p3);

        let predicted = tracker.predict_next(p1, 2);

        // p2 should be first (more frequent)
        assert_eq!(predicted.len(), 2);
        assert_eq!(predicted[0], p2);
    }
}
