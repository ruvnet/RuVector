//! Memory consolidation: short-term -> long-term

use crate::causal::CausalGraph;
use crate::long_term::LongTermStore;
use crate::short_term::ShortTermBuffer;
use crate::types::{Pattern, SubstrateTime};

/// Consolidation configuration
#[derive(Debug, Clone)]
pub struct ConsolidationConfig {
    /// Salience threshold for consolidation
    pub salience_threshold: f32,
    /// Weight for access frequency
    pub w_frequency: f32,
    /// Weight for recency
    pub w_recency: f32,
    /// Weight for causal importance
    pub w_causal: f32,
    /// Weight for surprise
    pub w_surprise: f32,
}

impl Default for ConsolidationConfig {
    fn default() -> Self {
        Self {
            salience_threshold: 0.5,
            w_frequency: 0.3,
            w_recency: 0.2,
            w_causal: 0.3,
            w_surprise: 0.2,
        }
    }
}

/// Compute salience score for a pattern
pub fn compute_salience(
    pattern: &Pattern,
    causal_graph: &CausalGraph,
    long_term: &LongTermStore,
    config: &ConsolidationConfig,
) -> f32 {
    let now = SubstrateTime::now();

    // 1. Access frequency (normalized)
    let access_freq = (pattern.access_count as f32).ln_1p() / 10.0;

    // 2. Recency (exponential decay)
    let time_since_access = now.duration_since(&pattern.last_accessed);
    let seconds_since = time_since_access.num_seconds().max(1) as f32;
    let recency = 1.0 / (1.0 + seconds_since / 3600.0); // Decay over hours

    // 3. Causal importance (out-degree in causal graph)
    let causal_importance = causal_graph.out_degree(pattern.id) as f32;
    let causal_score = (causal_importance.ln_1p()) / 5.0;

    // 4. Surprise (deviation from expected)
    let surprise = compute_surprise(pattern, long_term);

    // Weighted combination
    let salience = config.w_frequency * access_freq
        + config.w_recency * recency
        + config.w_causal * causal_score
        + config.w_surprise * surprise;

    // Clamp to [0, 1]
    salience.max(0.0).min(1.0)
}

/// Compute surprise score (how unexpected this pattern is)
fn compute_surprise(pattern: &Pattern, long_term: &LongTermStore) -> f32 {
    // Simple surprise metric: inverse of similarity to nearest neighbor
    if long_term.is_empty() {
        return 1.0; // Everything is surprising if long-term is empty
    }

    // Find most similar pattern in long-term
    let mut max_similarity = 0.0f32;

    for existing in long_term.all() {
        let sim = cosine_similarity(&pattern.embedding, &existing.embedding);
        max_similarity = max_similarity.max(sim);
    }

    // Surprise = 1 - max_similarity
    (1.0 - max_similarity).max(0.0)
}

/// Consolidate short-term memory to long-term
pub fn consolidate(
    short_term: &ShortTermBuffer,
    long_term: &LongTermStore,
    causal_graph: &CausalGraph,
    config: &ConsolidationConfig,
) -> ConsolidationResult {
    let mut num_consolidated = 0;
    let mut num_forgotten = 0;

    // Drain all patterns from short-term
    let patterns = short_term.drain();

    for mut pattern in patterns {
        // Compute salience
        let salience = compute_salience(&pattern, causal_graph, long_term, config);
        pattern.salience = salience;

        // Consolidate if above threshold
        if salience >= config.salience_threshold {
            long_term.integrate(pattern);
            num_consolidated += 1;
        } else {
            // Forget (don't integrate)
            num_forgotten += 1;
        }
    }

    ConsolidationResult {
        num_consolidated,
        num_forgotten,
    }
}

/// Result of consolidation operation
#[derive(Debug, Clone)]
pub struct ConsolidationResult {
    /// Number of patterns consolidated to long-term
    pub num_consolidated: usize,
    /// Number of patterns forgotten
    pub num_forgotten: usize,
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
    fn test_compute_salience() {
        let causal_graph = CausalGraph::new();
        let long_term = LongTermStore::default();
        let config = ConsolidationConfig::default();

        let mut pattern = Pattern::new(vec![1.0, 2.0, 3.0], Metadata::new());
        pattern.access_count = 10;

        let salience = compute_salience(&pattern, &causal_graph, &long_term, &config);

        assert!(salience >= 0.0 && salience <= 1.0);
    }

    #[test]
    fn test_consolidation() {
        let short_term = ShortTermBuffer::default();
        let long_term = LongTermStore::default();
        let causal_graph = CausalGraph::new();
        let config = ConsolidationConfig::default();

        // Add high-salience pattern
        let mut p1 = Pattern::new(vec![1.0, 0.0, 0.0], Metadata::new());
        p1.access_count = 100; // High access count
        short_term.insert(p1);

        // Add low-salience pattern
        let p2 = Pattern::new(vec![0.0, 1.0, 0.0], Metadata::new());
        short_term.insert(p2);

        let result = consolidate(&short_term, &long_term, &causal_graph, &config);

        // At least one should be consolidated
        assert!(result.num_consolidated > 0);
        assert!(short_term.is_empty());
    }
}
