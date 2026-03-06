//! Maximal Marginal Relevance (MMR) for Diversity-Aware Search
//!
//! Implements MMR algorithm to balance relevance and diversity in search results:
//! MMR = λ × Similarity(query, doc) - (1-λ) × max Similarity(doc, selected_docs)

use crate::error::{Result, RuvectorError};
use crate::types::{DistanceMetric, QuantumVector, SearchResult};

// ... (MMRConfig stays same for now, lambda is f32)

/// Configuration for MMR search
#[derive(Debug, Clone)]
pub struct MMRConfig {
    /// Diversity weight (0.0 to 1.0)
    /// Higher lambda = more weight on relevance
    /// Lower lambda = more weight on diversity
    pub lambda: f32,
    /// Distance metric to use for diversity calculation
    pub metric: DistanceMetric,
    /// Fetch multiplier: fetch (k * fetch_multiplier) candidates before reranking
    pub fetch_multiplier: f32,
}

impl Default for MMRConfig {
    fn default() -> Self {
        Self {
            lambda: 0.5,
            metric: DistanceMetric::Cosine,
            fetch_multiplier: 2.0,
        }
    }
}

/// MMR Reranker
pub struct MMRSearch {
    config: MMRConfig,
}

impl MMRSearch {
    pub fn new(config: MMRConfig) -> Result<Self> {
        if config.lambda < 0.0 || config.lambda > 1.0 {
            return Err(RuvectorError::InvalidParameter(
                "MMR lambda must be between 0.0 and 1.0".to_string(),
            ));
        }
        Ok(Self { config })
    }
    // ... (new stays same)

    /// Perform MMR-based reranking of search results
    pub fn rerank(
        &self,
        query: &QuantumVector,
        candidates: Vec<SearchResult>,
        k: usize,
    ) -> Result<Vec<SearchResult>> {
        if candidates.is_empty() {
            return Ok(Vec::new());
        }

        if k == 0 {
            return Ok(Vec::new());
        }

        if k >= candidates.len() {
            return Ok(candidates);
        }

        let mut selected: Vec<SearchResult> = Vec::with_capacity(k);
        let mut remaining = candidates;

        // Iteratively select documents maximizing MMR
        for _ in 0..k {
            if remaining.is_empty() {
                break;
            }

            // Compute MMR score for each remaining candidate
            let mut best_idx = 0;
            let mut best_mmr = f32::NEG_INFINITY;

            for (idx, candidate) in remaining.iter().enumerate() {
                let mmr_score = self.compute_mmr_score(query, candidate, &selected)?;

                if mmr_score > best_mmr {
                    best_mmr = mmr_score;
                    best_idx = idx;
                }
            }

            // Move best candidate to selected set
            let best = remaining.remove(best_idx);
            selected.push(best);
        }

        Ok(selected)
    }

    /// Compute MMR score for a candidate
    fn compute_mmr_score(
        &self,
        _query: &QuantumVector,
        candidate: &SearchResult,
        selected: &[SearchResult],
    ) -> Result<f32> {
        let candidate_vec = candidate.vector.as_ref().ok_or_else(|| {
            RuvectorError::InvalidParameter("Candidate vector not available".to_string())
        })?;

        // Relevance: similarity to query (convert distance to similarity)
        let relevance = self.distance_to_similarity(candidate.score);

        // Diversity: max similarity to already selected documents
        let max_similarity = if selected.is_empty() {
            0.0
        } else {
            selected
                .iter()
                .filter_map(|s| s.vector.as_ref())
                .map(|selected_vec| {
                    let a_f32 = candidate_vec.reconstruct();
                    let b_f32 = selected_vec.reconstruct();
                    let dist = compute_distance(&a_f32, &b_f32, self.config.metric);
                    self.distance_to_similarity(dist)
                })
                .max_by(|a, b| a.partial_cmp(b).unwrap())
                .unwrap_or(0.0)
        };

        // MMR = λ × relevance - (1-λ) × max_similarity
        let mmr = self.config.lambda * relevance - (1.0 - self.config.lambda) * max_similarity;

        Ok(mmr)
    }

    /// Convert distance to similarity (higher is better)
    fn distance_to_similarity(&self, distance: f32) -> f32 {
        match self.config.metric {
            DistanceMetric::Cosine => 1.0 - distance,
            DistanceMetric::Euclidean => 1.0 / (1.0 + distance),
            DistanceMetric::Manhattan => 1.0 / (1.0 + distance),
            DistanceMetric::DotProduct => -distance, // Dot product is already similarity-like
        }
    }

    /// Perform end-to-end MMR search
    pub fn search<F>(
        &self,
        query: &QuantumVector,
        k: usize,
        search_fn: F,
    ) -> Result<Vec<SearchResult>>
    where
        F: Fn(&QuantumVector, usize) -> Result<Vec<SearchResult>>,
    {
        // Fetch more candidates than needed
        let fetch_k = (k as f32 * self.config.fetch_multiplier).ceil() as usize;
        let candidates = search_fn(query, fetch_k)?;

        // Rerank using MMR
        self.rerank(query, candidates, k)
    }
}

// Helper function
fn compute_distance(a: &[f32], b: &[f32], metric: DistanceMetric) -> f32 {
    match metric {
        DistanceMetric::Euclidean => euclidean_distance(a, b),
        DistanceMetric::Cosine => cosine_distance(a, b),
        DistanceMetric::Manhattan => manhattan_distance(a, b),
        DistanceMetric::DotProduct => dot_product_distance(a, b),
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter()
        .zip(b)
        .map(|(x, y)| {
            let diff = x - y;
            diff * diff
        })
        .sum::<f32>()
        .sqrt()
}

fn cosine_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();

    if norm_a == 0.0 || norm_b == 0.0 {
        1.0
    } else {
        1.0 - (dot / (norm_a * norm_b))
    }
}

fn manhattan_distance(a: &[f32], b: &[f32]) -> f32 {
    a.iter().zip(b).map(|(x, y)| (x - y).abs()).sum()
}

fn dot_product_distance(a: &[f32], b: &[f32]) -> f32 {
    let dot: f32 = a.iter().zip(b).map(|(x, y)| x * y).sum();
    -dot
}

#[cfg(test)]
mod tests {
    use super::*;

    fn create_search_result(id: &str, score: f32, vector: Vec<f32>) -> SearchResult {
        SearchResult {
            id: id.to_string(),
            score,
            vector: Some(QuantumVector::F32(vector)),
            metadata: None,
        }
    }

    #[test]
    fn test_mmr_config_validation() {
        let config = MMRConfig {
            lambda: 0.5,
            ..Default::default()
        };
        assert!(MMRSearch::new(config).is_ok());

        let invalid_config = MMRConfig {
            lambda: 1.5,
            ..Default::default()
        };
        assert!(MMRSearch::new(invalid_config).is_err());
    }

    #[test]
    fn test_mmr_reranking() {
        let config = MMRConfig {
            lambda: 0.5,
            metric: DistanceMetric::Euclidean,
            fetch_multiplier: 2.0,
        };

        let mmr = MMRSearch::new(config).unwrap();
        let query = QuantumVector::F32(vec![1.0, 0.0, 0.0]);

        // Create candidates with varying similarity
        let candidates = vec![
            create_search_result("doc1", 0.1, vec![0.9, 0.1, 0.0]), // Very similar to query
            create_search_result("doc2", 0.15, vec![0.9, 0.0, 0.1]), // Similar to doc1 and query
            create_search_result("doc3", 0.5, vec![0.5, 0.5, 0.5]), // Different from doc1
            create_search_result("doc4", 0.6, vec![0.0, 1.0, 0.0]), // Very different
        ];

        let results = mmr.rerank(&query, candidates, 3).unwrap();

        assert_eq!(results.len(), 3);
        // First result should be most relevant
        assert_eq!(results[0].id, "doc1");
        // MMR should promote diversity, so doc3 or doc4 should appear
        assert!(results.iter().any(|r| r.id == "doc3" || r.id == "doc4"));
    }

    #[test]
    fn test_mmr_pure_relevance() {
        let config = MMRConfig {
            lambda: 1.0, // Pure relevance
            metric: DistanceMetric::Euclidean,
            fetch_multiplier: 2.0,
        };

        let mmr = MMRSearch::new(config).unwrap();
        let query = QuantumVector::F32(vec![1.0, 0.0, 0.0]);

        let candidates = vec![
            create_search_result("doc1", 0.1, vec![0.9, 0.1, 0.0]),
            create_search_result("doc2", 0.15, vec![0.85, 0.1, 0.05]),
            create_search_result("doc3", 0.5, vec![0.5, 0.5, 0.0]),
        ];

        let results = mmr.rerank(&query, candidates, 2).unwrap();

        // With lambda=1.0, should just preserve relevance order
        assert_eq!(results[0].id, "doc1");
        assert_eq!(results[1].id, "doc2");
    }

    #[test]
    fn test_mmr_pure_diversity() {
        let config = MMRConfig {
            lambda: 0.0, // Pure diversity
            metric: DistanceMetric::Euclidean,
            fetch_multiplier: 2.0,
        };

        let mmr = MMRSearch::new(config).unwrap();
        let query = QuantumVector::F32(vec![1.0, 0.0, 0.0]);

        let candidates = vec![
            create_search_result("doc1", 0.1, vec![0.9, 0.1, 0.0]),
            create_search_result("doc2", 0.15, vec![0.9, 0.0, 0.1]), // Very similar to doc1
            create_search_result("doc3", 0.5, vec![0.0, 1.0, 0.0]),  // Very different
        ];

        let results = mmr.rerank(&query, candidates, 2).unwrap();

        // With lambda=0.0, should maximize diversity
        assert_eq!(results.len(), 2);
        // Should not select both doc1 and doc2 (they're too similar)
        let has_both_similar =
            results.iter().any(|r| r.id == "doc1") && results.iter().any(|r| r.id == "doc2");
        assert!(!has_both_similar);
    }

    #[test]
    fn test_mmr_empty_candidates() {
        let config = MMRConfig::default();
        let mmr = MMRSearch::new(config).unwrap();
        let query = QuantumVector::F32(vec![1.0, 0.0, 0.0]);

        let results = mmr.rerank(&query, Vec::new(), 5).unwrap();
        assert!(results.is_empty());
    }
}
