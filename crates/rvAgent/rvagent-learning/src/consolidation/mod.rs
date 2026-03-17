//! SONA consolidation for learned patterns
//!
//! This module provides:
//! - SonaConsolidator for pattern consolidation
//! - EWC++ integration for preventing catastrophic forgetting
//! - Memory pressure management

use crate::discovery::DiscoveryLog;
use anyhow::Result;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// SONA pattern consolidator
pub struct SonaConsolidator {
    /// Consolidated patterns
    patterns: HashMap<String, ConsolidatedPattern>,

    /// EWC++ parameters
    ewc_lambda: f64,

    /// Maximum patterns to retain
    max_patterns: usize,

    /// Minimum quality for retention
    min_quality: f64,

    /// Current memory pressure
    memory_pressure: f64,
}

/// A consolidated pattern
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidatedPattern {
    /// Pattern identifier
    pub id: String,

    /// Pattern type/category
    pub pattern_type: String,

    /// Pattern description
    pub description: String,

    /// Quality score
    pub quality: f64,

    /// Number of times seen
    pub occurrence_count: usize,

    /// Embedding vector (simplified)
    pub embedding: Vec<f32>,

    /// Fisher information for EWC++
    pub fisher_importance: f64,
}

/// Result of consolidation
#[derive(Debug, Clone)]
pub struct ConsolidationReport {
    /// Patterns added
    pub patterns_added: usize,

    /// Patterns updated
    pub patterns_updated: usize,

    /// Patterns pruned
    pub patterns_pruned: usize,

    /// Memory pressure after consolidation
    pub memory_pressure: f64,

    /// Whether EWC++ was applied
    pub ewc_applied: bool,
}

impl SonaConsolidator {
    /// Create a new consolidator
    pub fn new() -> Self {
        Self {
            patterns: HashMap::new(),
            ewc_lambda: 2000.0,
            max_patterns: 10000,
            min_quality: 0.3,
            memory_pressure: 0.0,
        }
    }

    /// Set EWC++ lambda
    pub fn with_ewc_lambda(mut self, lambda: f64) -> Self {
        self.ewc_lambda = lambda;
        self
    }

    /// Set maximum patterns
    pub fn with_max_patterns(mut self, max: usize) -> Self {
        self.max_patterns = max;
        self
    }

    /// Set minimum quality
    pub fn with_min_quality(mut self, min: f64) -> Self {
        self.min_quality = min;
        self
    }

    /// Consolidate discoveries into patterns
    pub fn consolidate(&mut self, discoveries: &[DiscoveryLog]) -> Result<usize> {
        let mut added = 0;
        let mut updated = 0;

        for discovery in discoveries {
            // Skip low-quality discoveries
            if discovery.quality.composite < self.min_quality {
                continue;
            }

            let pattern_key = self.generate_pattern_key(discovery);

            if let Some(existing) = self.patterns.get_mut(&pattern_key) {
                // Update existing pattern
                existing.occurrence_count += 1;
                existing.quality = (existing.quality + discovery.quality.composite) / 2.0;
                updated += 1;
            } else {
                // Add new pattern
                let pattern = ConsolidatedPattern {
                    id: pattern_key.clone(),
                    pattern_type: format!("{:?}", discovery.category),
                    description: discovery.description.clone(),
                    quality: discovery.quality.composite,
                    occurrence_count: 1,
                    embedding: self.generate_embedding(&discovery.description),
                    fisher_importance: 1.0,
                };
                self.patterns.insert(pattern_key, pattern);
                added += 1;
            }
        }

        // Apply EWC++ if we have enough patterns
        if self.patterns.len() > 100 {
            self.apply_ewc_plus_plus();
        }

        // Prune if over capacity
        if self.patterns.len() > self.max_patterns {
            self.prune_patterns();
        }

        // Update memory pressure
        self.update_memory_pressure();

        Ok(added + updated)
    }

    /// Generate a key for a pattern
    fn generate_pattern_key(&self, discovery: &DiscoveryLog) -> String {
        // Simple key based on category and title hash
        format!("{:?}-{}", discovery.category, &discovery.title[..discovery.title.len().min(32)])
    }

    /// Generate a simple embedding (placeholder)
    fn generate_embedding(&self, text: &str) -> Vec<f32> {
        // Placeholder - would use actual embedding model
        let mut embedding = vec![0.0f32; 128];
        for (i, c) in text.chars().take(128).enumerate() {
            embedding[i] = (c as u32 as f32) / 255.0;
        }
        embedding
    }

    /// Apply EWC++ consolidation
    fn apply_ewc_plus_plus(&mut self) {
        // EWC++ (Elastic Weight Consolidation++)
        // Prevents catastrophic forgetting by protecting important patterns

        // Calculate Fisher information based on occurrence and quality
        for pattern in self.patterns.values_mut() {
            let importance = pattern.occurrence_count as f64 * pattern.quality;
            pattern.fisher_importance = importance.min(10.0); // Cap at 10
        }

        tracing::debug!("Applied EWC++ to {} patterns", self.patterns.len());
    }

    /// Prune low-quality patterns
    fn prune_patterns(&mut self) {
        let target_size = (self.max_patterns as f64 * 0.8) as usize;
        let current_size = self.patterns.len();

        if current_size <= target_size {
            return;
        }

        // Sort patterns by importance (quality * occurrence * fisher)
        let mut pattern_scores: Vec<(String, f64)> = self
            .patterns
            .iter()
            .map(|(k, p)| {
                let score = p.quality * p.occurrence_count as f64 * p.fisher_importance;
                (k.clone(), score)
            })
            .collect();

        pattern_scores.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap());

        // Remove lowest scoring patterns
        let to_remove = current_size - target_size;
        for (key, _) in pattern_scores.iter().take(to_remove) {
            self.patterns.remove(key);
        }

        tracing::info!("Pruned {} patterns", to_remove);
    }

    /// Update memory pressure estimate
    fn update_memory_pressure(&mut self) {
        // Simple estimate based on pattern count
        self.memory_pressure = self.patterns.len() as f64 / self.max_patterns as f64;
    }

    /// Get current memory pressure (0.0-1.0)
    pub fn memory_pressure(&self) -> f64 {
        self.memory_pressure
    }

    /// Get pattern count
    pub fn pattern_count(&self) -> usize {
        self.patterns.len()
    }

    /// Search for similar patterns
    pub fn search_similar(&self, query: &str, limit: usize) -> Vec<&ConsolidatedPattern> {
        let query_embedding = self.generate_embedding(query);

        let mut similarities: Vec<(&ConsolidatedPattern, f64)> = self
            .patterns
            .values()
            .map(|p| {
                let sim = self.cosine_similarity(&query_embedding, &p.embedding);
                (p, sim)
            })
            .collect();

        similarities.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap());

        similarities
            .into_iter()
            .take(limit)
            .map(|(p, _)| p)
            .collect()
    }

    /// Compute cosine similarity
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> f64 {
        if a.len() != b.len() {
            return 0.0;
        }

        let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
        let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
        let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

        if norm_a == 0.0 || norm_b == 0.0 {
            0.0
        } else {
            (dot / (norm_a * norm_b)) as f64
        }
    }

    /// Export patterns for inspection
    pub fn export_patterns(&self) -> Vec<ConsolidatedPattern> {
        self.patterns.values().cloned().collect()
    }

    /// Get consolidation statistics
    pub fn stats(&self) -> ConsolidationStats {
        let total = self.patterns.len();
        let avg_quality = if total > 0 {
            self.patterns.values().map(|p| p.quality).sum::<f64>() / total as f64
        } else {
            0.0
        };
        let avg_occurrences = if total > 0 {
            self.patterns.values().map(|p| p.occurrence_count).sum::<usize>() as f64 / total as f64
        } else {
            0.0
        };

        ConsolidationStats {
            total_patterns: total,
            average_quality: avg_quality,
            average_occurrences: avg_occurrences,
            memory_pressure: self.memory_pressure,
            ewc_lambda: self.ewc_lambda,
        }
    }
}

impl Default for SonaConsolidator {
    fn default() -> Self {
        Self::new()
    }
}

/// Consolidation statistics
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ConsolidationStats {
    /// Total patterns stored
    pub total_patterns: usize,

    /// Average quality score
    pub average_quality: f64,

    /// Average occurrence count
    pub average_occurrences: f64,

    /// Current memory pressure
    pub memory_pressure: f64,

    /// EWC++ lambda value
    pub ewc_lambda: f64,
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::discovery::DiscoveryCategory;

    fn make_discovery(title: &str, quality: f64) -> DiscoveryLog {
        let mut d = DiscoveryLog::new(
            DiscoveryCategory::Optimization,
            title,
            "Test description",
        );
        d.quality.composite = quality;
        d
    }

    #[test]
    fn test_consolidator_creation() {
        let consolidator = SonaConsolidator::new()
            .with_ewc_lambda(1000.0)
            .with_max_patterns(5000);

        assert_eq!(consolidator.ewc_lambda, 1000.0);
        assert_eq!(consolidator.max_patterns, 5000);
    }

    #[test]
    fn test_consolidate_discoveries() {
        let mut consolidator = SonaConsolidator::new();

        let discoveries = vec![
            make_discovery("Pattern A", 0.8),
            make_discovery("Pattern B", 0.9),
            make_discovery("Low Quality", 0.2), // Should be skipped
        ];

        let count = consolidator.consolidate(&discoveries).unwrap();
        assert_eq!(count, 2); // Only high-quality patterns
        assert_eq!(consolidator.pattern_count(), 2);
    }

    #[test]
    fn test_search_similar() {
        let mut consolidator = SonaConsolidator::new();

        let discoveries = vec![
            make_discovery("Error handling pattern", 0.8),
            make_discovery("Security validation", 0.9),
        ];

        consolidator.consolidate(&discoveries).unwrap();

        let results = consolidator.search_similar("error", 1);
        assert!(!results.is_empty());
    }

    #[test]
    fn test_consolidation_stats() {
        let mut consolidator = SonaConsolidator::new();

        let discoveries = vec![
            make_discovery("A", 0.8),
            make_discovery("B", 0.6),
        ];

        consolidator.consolidate(&discoveries).unwrap();

        let stats = consolidator.stats();
        assert_eq!(stats.total_patterns, 2);
        assert!((stats.average_quality - 0.7).abs() < 0.01);
    }
}
