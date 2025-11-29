//! Short-term volatile memory buffer

use crate::types::{Pattern, PatternId, SubstrateTime};
use dashmap::DashMap;
use parking_lot::RwLock;
use std::collections::VecDeque;
use std::sync::Arc;

/// Configuration for short-term buffer
#[derive(Debug, Clone)]
pub struct ShortTermConfig {
    /// Maximum number of patterns before consolidation
    pub max_capacity: usize,
    /// Consolidation threshold (trigger when this full)
    pub consolidation_threshold: f32,
}

impl Default for ShortTermConfig {
    fn default() -> Self {
        Self {
            max_capacity: 10_000,
            consolidation_threshold: 0.8,
        }
    }
}

/// Short-term volatile memory buffer
pub struct ShortTermBuffer {
    /// Pattern storage (FIFO queue)
    patterns: Arc<RwLock<VecDeque<Pattern>>>,
    /// Index for fast lookup by ID
    index: DashMap<PatternId, usize>,
    /// Configuration
    config: ShortTermConfig,
}

impl ShortTermBuffer {
    /// Create new short-term buffer
    pub fn new(config: ShortTermConfig) -> Self {
        Self {
            patterns: Arc::new(RwLock::new(VecDeque::with_capacity(config.max_capacity))),
            index: DashMap::new(),
            config,
        }
    }

    /// Insert pattern into buffer
    pub fn insert(&self, pattern: Pattern) -> PatternId {
        let id = pattern.id;
        let mut patterns = self.patterns.write();

        // Add to queue
        let position = patterns.len();
        patterns.push_back(pattern);

        // Update index
        self.index.insert(id, position);

        id
    }

    /// Get pattern by ID
    pub fn get(&self, id: &PatternId) -> Option<Pattern> {
        let index = self.index.get(id)?;
        let patterns = self.patterns.read();
        patterns.get(*index).cloned()
    }

    /// Get mutable pattern by ID
    pub fn get_mut<F, R>(&self, id: &PatternId, f: F) -> Option<R>
    where
        F: FnOnce(&mut Pattern) -> R,
    {
        let index = *self.index.get(id)?;
        let mut patterns = self.patterns.write();
        patterns.get_mut(index).map(f)
    }

    /// Update pattern
    pub fn update(&self, pattern: Pattern) -> bool {
        let id = pattern.id;
        if let Some(index) = self.index.get(&id) {
            let mut patterns = self.patterns.write();
            if let Some(p) = patterns.get_mut(*index) {
                *p = pattern;
                return true;
            }
        }
        false
    }

    /// Check if should trigger consolidation
    pub fn should_consolidate(&self) -> bool {
        let patterns = self.patterns.read();
        let usage = patterns.len() as f32 / self.config.max_capacity as f32;
        usage >= self.config.consolidation_threshold
    }

    /// Get current size
    pub fn len(&self) -> usize {
        self.patterns.read().len()
    }

    /// Check if empty
    pub fn is_empty(&self) -> bool {
        self.patterns.read().is_empty()
    }

    /// Drain all patterns (for consolidation)
    pub fn drain(&self) -> Vec<Pattern> {
        let mut patterns = self.patterns.write();
        self.index.clear();
        patterns.drain(..).collect()
    }

    /// Drain patterns matching predicate
    pub fn drain_filter<F>(&self, mut predicate: F) -> Vec<Pattern>
    where
        F: FnMut(&Pattern) -> bool,
    {
        let mut patterns = self.patterns.write();
        let mut result = Vec::new();
        let mut i = 0;

        while i < patterns.len() {
            if predicate(&patterns[i]) {
                let pattern = patterns.remove(i).unwrap();
                self.index.remove(&pattern.id);
                result.push(pattern);
                // Don't increment i, as we removed an element
            } else {
                // Update index since positions shifted
                self.index.insert(patterns[i].id, i);
                i += 1;
            }
        }

        result
    }

    /// Get all patterns (for iteration)
    pub fn all(&self) -> Vec<Pattern> {
        self.patterns.read().iter().cloned().collect()
    }

    /// Clear all patterns
    pub fn clear(&self) {
        self.patterns.write().clear();
        self.index.clear();
    }

    /// Get statistics
    pub fn stats(&self) -> ShortTermStats {
        let patterns = self.patterns.read();
        let size = patterns.len();
        let capacity = self.config.max_capacity;
        let usage = size as f32 / capacity as f32;

        // Compute average salience
        let total_salience: f32 = patterns.iter().map(|p| p.salience).sum();
        let avg_salience = if size > 0 {
            total_salience / size as f32
        } else {
            0.0
        };

        ShortTermStats {
            size,
            capacity,
            usage,
            avg_salience,
        }
    }
}

impl Default for ShortTermBuffer {
    fn default() -> Self {
        Self::new(ShortTermConfig::default())
    }
}

/// Short-term buffer statistics
#[derive(Debug, Clone)]
pub struct ShortTermStats {
    /// Current number of patterns
    pub size: usize,
    /// Maximum capacity
    pub capacity: usize,
    /// Usage ratio (0.0 to 1.0)
    pub usage: f32,
    /// Average salience
    pub avg_salience: f32,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::types::Metadata;

    #[test]
    fn test_short_term_buffer() {
        let buffer = ShortTermBuffer::default();

        let pattern = Pattern::new(vec![1.0, 2.0, 3.0], Metadata::new());
        let id = pattern.id;

        buffer.insert(pattern);

        assert_eq!(buffer.len(), 1);
        assert!(buffer.get(&id).is_some());

        let patterns = buffer.drain();
        assert_eq!(patterns.len(), 1);
        assert!(buffer.is_empty());
    }

    #[test]
    fn test_consolidation_threshold() {
        let config = ShortTermConfig {
            max_capacity: 10,
            consolidation_threshold: 0.8,
        };
        let buffer = ShortTermBuffer::new(config);

        // Add 7 patterns (70% full)
        for i in 0..7 {
            let pattern = Pattern::new(vec![i as f32], Metadata::new());
            buffer.insert(pattern);
        }

        assert!(!buffer.should_consolidate());

        // Add 1 more (80% full)
        let pattern = Pattern::new(vec![8.0], Metadata::new());
        buffer.insert(pattern);

        assert!(buffer.should_consolidate());
    }
}
