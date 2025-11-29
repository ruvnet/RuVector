//! Long-term consolidated memory store

use crate::types::{Pattern, PatternId, Query, SearchResult, SubstrateTime, TimeRange};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::sync::Arc;

/// Configuration for long-term store
#[derive(Debug, Clone)]
pub struct LongTermConfig {
    /// Decay rate for low-salience patterns
    pub decay_rate: f32,
    /// Minimum salience threshold
    pub min_salience: f32,
}

impl Default for LongTermConfig {
    fn default() -> Self {
        Self {
            decay_rate: 0.01,
            min_salience: 0.1,
        }
    }
}

/// Long-term consolidated memory store
pub struct LongTermStore {
    /// Pattern storage
    patterns: DashMap<PatternId, Pattern>,
    /// Temporal index (sorted by timestamp)
    temporal_index: Arc<RwLock<Vec<(SubstrateTime, PatternId)>>>,
    /// Configuration
    config: LongTermConfig,
}

impl LongTermStore {
    /// Create new long-term store
    pub fn new(config: LongTermConfig) -> Self {
        Self {
            patterns: DashMap::new(),
            temporal_index: Arc::new(RwLock::new(Vec::new())),
            config,
        }
    }

    /// Integrate pattern from consolidation
    pub fn integrate(&self, pattern: Pattern) {
        let id = pattern.id;
        let timestamp = pattern.timestamp;

        // Store pattern
        self.patterns.insert(id, pattern);

        // Update temporal index
        let mut index = self.temporal_index.write();
        index.push((timestamp, id));
        index.sort_by_key(|(t, _)| *t);
    }

    /// Get pattern by ID
    pub fn get(&self, id: &PatternId) -> Option<Pattern> {
        self.patterns.get(id).map(|p| p.clone())
    }

    /// Update pattern
    pub fn update(&self, pattern: Pattern) -> bool {
        let id = pattern.id;
        self.patterns.insert(id, pattern).is_some()
    }

    /// Search by embedding similarity
    pub fn search(&self, query: &Query) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for entry in self.patterns.iter() {
            let pattern = entry.value();
            let score = cosine_similarity(&query.embedding, &pattern.embedding);

            results.push(SearchResult {
                id: pattern.id,
                pattern: pattern.clone(),
                score,
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Take top k
        results.truncate(query.k);

        results
    }

    /// Search with time range filter
    pub fn search_with_time_range(&self, query: &Query, time_range: TimeRange) -> Vec<SearchResult> {
        let mut results = Vec::new();

        for entry in self.patterns.iter() {
            let pattern = entry.value();

            // Filter by time range
            if !time_range.contains(&pattern.timestamp) {
                continue;
            }

            let score = cosine_similarity(&query.embedding, &pattern.embedding);

            results.push(SearchResult {
                id: pattern.id,
                pattern: pattern.clone(),
                score,
            });
        }

        // Sort by score descending
        results.sort_by(|a, b| b.score.partial_cmp(&a.score).unwrap());

        // Take top k
        results.truncate(query.k);

        results
    }

    /// Filter patterns by time range
    pub fn filter_by_time(&self, time_range: TimeRange) -> Vec<Pattern> {
        let index = self.temporal_index.read();

        // Binary search for start
        let start_idx = index
            .binary_search_by_key(&time_range.start, |(t, _)| *t)
            .unwrap_or_else(|i| i);

        // Binary search for end
        let end_idx = index
            .binary_search_by_key(&time_range.end, |(t, _)| *t)
            .unwrap_or_else(|i| i);

        // Collect patterns in range
        index[start_idx..=end_idx.min(index.len().saturating_sub(1))]
            .iter()
            .filter_map(|(_, id)| self.patterns.get(id).map(|p| p.clone()))
            .collect()
    }

    /// Strategic forgetting: decay low-salience patterns
    pub fn decay_low_salience(&self, decay_rate: f32) {
        let mut to_remove = Vec::new();

        for mut entry in self.patterns.iter_mut() {
            let pattern = entry.value_mut();

            // Decay salience
            pattern.salience *= 1.0 - decay_rate;

            // Mark for removal if below threshold
            if pattern.salience < self.config.min_salience {
                to_remove.push(pattern.id);
            }
        }

        // Remove low-salience patterns
        for id in to_remove {
            self.remove(&id);
        }
    }

    /// Remove pattern
    pub fn remove(&self, id: &PatternId) -> Option<Pattern> {
        // Remove from storage
        let pattern = self.patterns.remove(id).map(|(_, p)| p)?;

        // Remove from temporal index
        let mut index = self.temporal_index.write();
        index.retain(|(_, pid)| pid != id);

        Some(pattern)
    }

    /// Get total number of patterns
    pub fn len(&self) -> usize {
        self.patterns.len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.patterns.is_empty()
    }

    /// Clear all patterns
    pub fn clear(&self) {
        self.patterns.clear();
        self.temporal_index.write().clear();
    }

    /// Get all patterns
    pub fn all(&self) -> Vec<Pattern> {
        self.patterns.iter().map(|e| e.value().clone()).collect()
    }

    /// Get statistics
    pub fn stats(&self) -> LongTermStats {
        let size = self.patterns.len();

        // Compute average salience
        let total_salience: f32 = self.patterns.iter().map(|e| e.value().salience).sum();
        let avg_salience = if size > 0 {
            total_salience / size as f32
        } else {
            0.0
        };

        // Find min/max salience
        let mut min_salience = f32::MAX;
        let mut max_salience = f32::MIN;

        for entry in self.patterns.iter() {
            let salience = entry.value().salience;
            min_salience = min_salience.min(salience);
            max_salience = max_salience.max(salience);
        }

        if size == 0 {
            min_salience = 0.0;
            max_salience = 0.0;
        }

        LongTermStats {
            size,
            avg_salience,
            min_salience,
            max_salience,
        }
    }
}

impl Default for LongTermStore {
    fn default() -> Self {
        Self::new(LongTermConfig::default())
    }
}

/// Long-term store statistics
#[derive(Debug, Clone)]
pub struct LongTermStats {
    /// Number of patterns
    pub size: usize,
    /// Average salience
    pub avg_salience: f32,
    /// Minimum salience
    pub min_salience: f32,
    /// Maximum salience
    pub max_salience: f32,
}

/// Compute cosine similarity between two vectors
fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() {
        return 0.0;
    }

    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let mag_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let mag_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if mag_a == 0.0 || mag_b == 0.0 {
        return 0.0;
    }

    dot / (mag_a * mag_b)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Metadata;

    #[test]
    fn test_long_term_store() {
        let store = LongTermStore::default();

        let pattern = Pattern::new(vec![1.0, 2.0, 3.0], Metadata::new());
        let id = pattern.id;

        store.integrate(pattern);

        assert_eq!(store.len(), 1);
        assert!(store.get(&id).is_some());
    }

    #[test]
    fn test_search() {
        let store = LongTermStore::default();

        // Add patterns
        let p1 = Pattern::new(vec![1.0, 0.0, 0.0], Metadata::new());
        let p2 = Pattern::new(vec![0.0, 1.0, 0.0], Metadata::new());

        store.integrate(p1);
        store.integrate(p2);

        // Query similar to p1
        let query = Query::from_embedding(vec![0.9, 0.1, 0.0]).with_k(1);
        let results = store.search(&query);

        assert_eq!(results.len(), 1);
        assert!(results[0].score > 0.5);
    }

    #[test]
    fn test_decay() {
        let store = LongTermStore::default();

        let mut pattern = Pattern::new(vec![1.0, 2.0, 3.0], Metadata::new());
        pattern.salience = 0.15; // Just above minimum
        let id = pattern.id;

        store.integrate(pattern);
        assert_eq!(store.len(), 1);

        // Decay should remove it
        store.decay_low_salience(0.5);
        assert_eq!(store.len(), 0);
    }
}
