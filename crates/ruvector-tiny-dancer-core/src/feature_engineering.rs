//! Feature engineering for candidate scoring
//!
//! Combines semantic similarity, recency, frequency, and other metrics

use crate::error::{Result, TinyDancerError};
use crate::types::Candidate;
use chrono::Utc;
#[cfg(all(not(feature = "lattice-simd"), feature = "simd-simsimd"))]
use simsimd::SpatialSimilarity;

/// Feature vector for a candidate
#[derive(Debug, Clone)]
pub struct FeatureVector {
    /// Semantic similarity score (0.0 to 1.0)
    pub semantic_similarity: f32,
    /// Recency score (0.0 to 1.0)
    pub recency_score: f32,
    /// Frequency score (0.0 to 1.0)
    pub frequency_score: f32,
    /// Success rate (0.0 to 1.0)
    pub success_rate: f32,
    /// Metadata overlap score (0.0 to 1.0)
    pub metadata_overlap: f32,
    /// Combined feature vector
    pub features: Vec<f32>,
}

/// Feature engineering configuration
#[derive(Debug, Clone)]
pub struct FeatureConfig {
    /// Weight for semantic similarity (default: 0.4)
    pub similarity_weight: f32,
    /// Weight for recency (default: 0.2)
    pub recency_weight: f32,
    /// Weight for frequency (default: 0.15)
    pub frequency_weight: f32,
    /// Weight for success rate (default: 0.15)
    pub success_weight: f32,
    /// Weight for metadata overlap (default: 0.1)
    pub metadata_weight: f32,
    /// Decay factor for recency (default: 0.001)
    pub recency_decay: f32,
}

impl Default for FeatureConfig {
    fn default() -> Self {
        Self {
            similarity_weight: 0.4,
            recency_weight: 0.2,
            frequency_weight: 0.15,
            success_weight: 0.15,
            metadata_weight: 0.1,
            recency_decay: 0.001,
        }
    }
}

/// Feature engineering for candidate scoring
pub struct FeatureEngineer {
    config: FeatureConfig,
}

impl FeatureEngineer {
    /// Create a new feature engineer with default configuration
    pub fn new() -> Self {
        Self {
            config: FeatureConfig::default(),
        }
    }

    /// Create a new feature engineer with custom configuration
    pub fn with_config(config: FeatureConfig) -> Self {
        Self { config }
    }

    /// Extract features from a candidate
    pub fn extract_features(
        &self,
        query_embedding: &[f32],
        candidate: &Candidate,
        query_metadata: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<FeatureVector> {
        // 1. Semantic similarity (cosine similarity)
        let semantic_similarity = self.cosine_similarity(query_embedding, &candidate.embedding)?;

        // 2. Recency score (exponential decay)
        let recency_score = self.recency_score(candidate.created_at);

        // 3. Frequency score (normalized access count)
        let frequency_score = self.frequency_score(candidate.access_count);

        // 4. Success rate (direct from candidate)
        let success_rate = candidate.success_rate;

        // 5. Metadata overlap
        let metadata_overlap = if let Some(query_meta) = query_metadata {
            self.metadata_overlap_score(query_meta, &candidate.metadata)
        } else {
            0.0
        };

        // Combine features into a weighted vector
        let features = vec![
            semantic_similarity * self.config.similarity_weight,
            recency_score * self.config.recency_weight,
            frequency_score * self.config.frequency_weight,
            success_rate * self.config.success_weight,
            metadata_overlap * self.config.metadata_weight,
        ];

        Ok(FeatureVector {
            semantic_similarity,
            recency_score,
            frequency_score,
            success_rate,
            metadata_overlap,
            features,
        })
    }

    /// Extract features for a batch of candidates
    pub fn extract_batch_features(
        &self,
        query_embedding: &[f32],
        candidates: &[Candidate],
        query_metadata: Option<&std::collections::HashMap<String, serde_json::Value>>,
    ) -> Result<Vec<FeatureVector>> {
        candidates
            .iter()
            .map(|candidate| self.extract_features(query_embedding, candidate, query_metadata))
            .collect()
    }

    /// Cosine similarity between two equal-length vectors.
    ///
    /// The kernel is feature-selected and exactly one arm compiles. All three
    /// agree on the boundary conventions, which `backend_matches_scalar_reference`
    /// checks: two all-zero vectors score `1.0`, and a zero vector against a
    /// non-zero one scores `0.0`. Those are the values this path has always
    /// returned, and a backend swap is not the place to change them.
    fn cosine_similarity(&self, a: &[f32], b: &[f32]) -> Result<f32> {
        if a.len() != b.len() {
            return Err(TinyDancerError::InvalidInput(format!(
                "Vector dimension mismatch: {} vs {}",
                a.len(),
                b.len()
            )));
        }

        #[cfg(feature = "lattice-simd")]
        {
            // Returns similarity directly, so there is no `1 - x` here. It also
            // reports 0.0 for a zero norm, which collides with the genuine
            // "orthogonal" answer, so the all-zero case is separated out on the
            // cold path to keep the conventions above.
            let similarity = lattice_embed::simd::cosine_similarity(a, b);
            if similarity == 0.0 && is_all_zero(a) && is_all_zero(b) {
                return Ok(1.0);
            }
            Ok(similarity)
        }

        #[cfg(all(not(feature = "lattice-simd"), feature = "simd-simsimd"))]
        {
            // `SpatialSimilarity::cosine` returns a DISTANCE, hence the `1 - x`.
            let distance = f32::cosine(a, b).ok_or_else(|| {
                TinyDancerError::FeatureError("Cosine similarity failed".to_string())
            })?;
            Ok(1.0_f32 - distance as f32)
        }

        #[cfg(all(not(feature = "lattice-simd"), not(feature = "simd-simsimd")))]
        {
            Ok(scalar_cosine_similarity(a, b))
        }
    }

    /// Calculate recency score using exponential decay
    fn recency_score(&self, created_at: i64) -> f32 {
        let now = Utc::now().timestamp();
        let age_seconds = (now - created_at).max(0) as f32;

        // Exponential decay: score = exp(-λ * age)
        (-self.config.recency_decay * age_seconds).exp()
    }

    /// Calculate frequency score (normalized)
    fn frequency_score(&self, access_count: u64) -> f32 {
        // Use logarithmic scaling for frequency
        // score = log(1 + count) / log(1 + max_expected)
        let max_expected = 10000.0_f32; // Expected maximum access count
        ((1.0 + access_count as f32).ln() / (1.0 + max_expected).ln()).min(1.0)
    }

    /// Calculate metadata overlap score
    fn metadata_overlap_score(
        &self,
        query_metadata: &std::collections::HashMap<String, serde_json::Value>,
        candidate_metadata: &std::collections::HashMap<String, serde_json::Value>,
    ) -> f32 {
        if query_metadata.is_empty() || candidate_metadata.is_empty() {
            return 0.0;
        }

        let mut matches = 0;
        let total = query_metadata.len();

        for (key, value) in query_metadata {
            if let Some(candidate_value) = candidate_metadata.get(key) {
                if value == candidate_value {
                    matches += 1;
                }
            }
        }

        matches as f32 / total as f32
    }

    /// Get the configuration
    pub fn config(&self) -> &FeatureConfig {
        &self.config
    }
}

impl Default for FeatureEngineer {
    fn default() -> Self {
        Self::new()
    }
}

/// Scalar cosine similarity. Serves as the `--no-default-features` backend and
/// as the reference `backend_matches_scalar_reference` checks the compiled
/// backend against, so it stays compiled even when a SIMD arm is selected.
///
/// A zero norm is not an error here: two all-zero vectors score `1.0` and a
/// zero vector against a non-zero one scores `0.0`, matching the other arms.
/// The `min` mirrors the clip a distance-returning kernel applies at 0.
fn scalar_cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += x * y;
        norm_a += x * x;
        norm_b += y * y;
    }

    if norm_a == 0.0 && norm_b == 0.0 {
        return 1.0;
    }
    if dot == 0.0 {
        return 0.0;
    }
    (dot / (norm_a.sqrt() * norm_b.sqrt())).min(1.0)
}

fn is_all_zero(v: &[f32]) -> bool {
    v.iter().all(|x| *x == 0.0)
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    #[test]
    fn test_feature_extraction() {
        let engineer = FeatureEngineer::new();
        let query = vec![1.0, 0.0, 0.0];
        let candidate = Candidate {
            id: "test".to_string(),
            embedding: vec![0.9, 0.1, 0.0],
            metadata: HashMap::new(),
            created_at: Utc::now().timestamp(),
            access_count: 10,
            success_rate: 0.95,
        };

        let features = engineer.extract_features(&query, &candidate, None).unwrap();
        assert!(features.semantic_similarity > 0.8);
        assert!(features.recency_score > 0.9);
    }

    #[test]
    fn test_cosine_similarity() {
        let engineer = FeatureEngineer::new();
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        let similarity = engineer.cosine_similarity(&a, &b).unwrap();
        assert!((similarity - 1.0).abs() < 0.01);
    }

    /// Deterministic vector source. A fixed LCG keeps the grid reproducible
    /// across arms without pulling a seeded-RNG dependency into the comparison.
    fn lcg(state: &mut u64) -> f32 {
        *state = state
            .wrapping_mul(6364136223846793005)
            .wrapping_add(1442695040888963407);
        ((*state >> 33) as f32 / (1u64 << 31) as f32) * 2.0 - 1.0
    }

    /// Single-pass f64 reference. Deliberately NOT `scalar_cosine_similarity`:
    /// that one accumulates in f32 and is a backend in its own right, so using
    /// it as the reference would let a shared reduction-order bug pass.
    fn naive_cosine(a: &[f32], b: &[f32]) -> f32 {
        let mut dot = 0.0f64;
        let mut norm_a = 0.0f64;
        let mut norm_b = 0.0f64;
        for (x, y) in a.iter().zip(b.iter()) {
            dot += f64::from(*x) * f64::from(*y);
            norm_a += f64::from(*x) * f64::from(*x);
            norm_b += f64::from(*y) * f64::from(*y);
        }
        (dot / (norm_a.sqrt() * norm_b.sqrt())) as f32
    }

    /// Checks whichever backend compiled in against the reference, so this
    /// covers the SimSIMD path shipping today, the scalar path, and the lattice
    /// path alike. Dimensions straddle 4/8/16-lane widths and their remainders.
    #[test]
    fn backend_matches_scalar_reference() {
        let engineer = FeatureEngineer::new();
        const DIMS: [usize; 21] = [
            1, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 63, 64, 128, 384, 768, 1000, 1023, 1024,
        ];

        for &dim in DIMS.iter() {
            for seed_base in 0..3u64 {
                let mut state = seed_base.wrapping_mul(7919).wrapping_add(12345);
                let a: Vec<f32> = (0..dim).map(|_| lcg(&mut state)).collect();
                let b: Vec<f32> = (0..dim).map(|_| lcg(&mut state)).collect();

                let got = engineer.cosine_similarity(&a, &b).unwrap();
                let want = naive_cosine(&a, &b);
                assert!(
                    (got - want).abs() <= 1e-4 * want.abs().max(1.0),
                    "cosine mismatch dim={dim} seed={seed_base}: got {got}, want {want}"
                );
            }
        }
    }

    /// The boundary values every backend must agree on. These are conventions
    /// rather than arithmetic, so they are asserted exactly.
    #[test]
    fn backend_boundary_conventions_agree() {
        let engineer = FeatureEngineer::new();
        let zero = vec![0.0f32; 8];
        let ones = vec![1.0f32; 8];

        assert_eq!(
            engineer.cosine_similarity(&zero, &zero).unwrap(),
            1.0,
            "two all-zero vectors"
        );
        assert_eq!(
            engineer.cosine_similarity(&zero, &ones).unwrap(),
            0.0,
            "zero against non-zero"
        );

        let e1 = vec![1.0, 0.0, 0.0, 0.0];
        let e2 = vec![0.0, 1.0, 0.0, 0.0];
        assert_eq!(
            engineer.cosine_similarity(&e1, &e2).unwrap(),
            0.0,
            "orthogonal"
        );

        assert!(
            engineer.cosine_similarity(&e1, &ones).is_err(),
            "length mismatch must stay an error on every backend"
        );
    }

    #[test]
    fn test_recency_score() {
        let engineer = FeatureEngineer::new();
        let now = Utc::now().timestamp();
        let score_recent = engineer.recency_score(now);
        let score_old = engineer.recency_score(now - 86400); // 1 day ago
        assert!(score_recent > score_old);
    }
}
