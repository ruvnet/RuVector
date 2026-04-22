//! Cognitive engine integrating Hopfield networks, dentate gyrus, and HDC

use ruvector_nervous_system::hdc::{HdcMemory, Hypervector};
use ruvector_nervous_system::hopfield::ModernHopfield;
use ruvector_nervous_system::DentateGyrus;
use std::collections::hash_map::DefaultHasher;
use std::hash::{Hash, Hasher};
use std::panic;

/// Cognitive engine for knowledge cluster quality assessment
pub struct CognitiveEngine {
    /// Content-addressable memory recall
    hopfield: ModernHopfield,
    /// Pattern separation for collision detection
    dentate: DentateGyrus,
    /// Fast binary similarity filtering
    hdc: HdcMemory,
    /// Anomaly detection threshold
    anomaly_threshold: f32,
}

impl CognitiveEngine {
    /// Create a new cognitive engine with the given embedding dimension
    pub fn new(embedding_dim: usize) -> Self {
        let dim = embedding_dim.max(1);
        // DentateGyrus panics if k == 0 or k > output_dim.
        // Use output_dim = dim * 10 and k = output_dim / 50, clamped safely.
        let output_dim = dim * 10;
        let k = (output_dim / 50).max(1).min(output_dim);

        let dentate = panic::catch_unwind(|| DentateGyrus::new(dim, output_dim, k, 42))
            .unwrap_or_else(|_| {
                // Absolute fallback: minimal safe params
                DentateGyrus::new(dim, dim.max(2), 1, 42)
            });

        Self {
            hopfield: ModernHopfield::new(dim, 1.0),
            dentate,
            hdc: HdcMemory::new(),
            anomaly_threshold: 0.3,
        }
    }

    /// Store a pattern in all cognitive subsystems
    pub fn store_pattern(&mut self, id: &str, embedding: &[f32]) {
        // Store in Hopfield network
        let _ = self.hopfield.store(embedding.to_vec());

        // Store in HDC memory
        let mut hasher = DefaultHasher::new();
        id.hash(&mut hasher);
        let seed = hasher.finish();
        let hv = Hypervector::from_seed(seed);
        self.hdc.store(id, hv);

        // Encode via DentateGyrus for collision detection (result unused here
        // but exercises the pathway)
        let _ = panic::catch_unwind(panic::AssertUnwindSafe(|| self.dentate.encode(embedding)));
    }

    /// Recall the closest stored pattern via Hopfield retrieval
    pub fn recall(&self, query: &[f32]) -> Option<Vec<f32>> {
        self.hopfield.retrieve(query).ok()
    }

    /// Recall the top-k stored patterns by attention weight
    pub fn recall_k(&self, query: &[f32], k: usize) -> Vec<(usize, Vec<f32>, f32)> {
        self.hopfield.retrieve_k(query, k).unwrap_or_default()
    }

    /// Check if a new embedding is anomalous relative to its cluster
    pub fn is_anomalous(&self, embedding: &[f32], centroid: &[f32]) -> bool {
        // Try Hopfield retrieval distance first
        let hopfield_dist = match self.hopfield.retrieve(embedding) {
            Ok(retrieved) => euclidean_distance(&retrieved, embedding),
            Err(_) => euclidean_distance(embedding, centroid),
        };

        // Also check DentateGyrus sparsity: encode both and compare
        let separation = panic::catch_unwind(panic::AssertUnwindSafe(|| {
            let enc_emb = self.dentate.encode(embedding);
            let enc_cent = self.dentate.encode(centroid);
            1.0 - enc_emb.jaccard_similarity(&enc_cent) as f64
        }))
        .unwrap_or(0.0);

        // Anomalous if Hopfield distance exceeds threshold OR pattern
        // separation is very high (indicating highly dissimilar patterns)
        hopfield_dist > self.anomaly_threshold as f64 || separation > 0.95
    }

    /// Assess cluster quality (coherence)
    pub fn cluster_coherence(&self, embeddings: &[Vec<f32>]) -> f64 {
        if embeddings.len() < 2 {
            return 1.0;
        }
        let dim = embeddings[0].len();
        let n = embeddings.len() as f64;
        let mut centroid = vec![0.0f64; dim];
        for emb in embeddings {
            for (i, &v) in emb.iter().enumerate() {
                centroid[i] += v as f64;
            }
        }
        for c in &mut centroid {
            *c /= n;
        }

        let avg_dist: f64 = embeddings
            .iter()
            .map(|emb| {
                emb.iter()
                    .enumerate()
                    .map(|(i, &v)| (v as f64 - centroid[i]).powi(2))
                    .sum::<f64>()
                    .sqrt()
            })
            .sum::<f64>()
            / n;

        let base_coherence = 1.0 / (1.0 + avg_dist);

        // Additionally check Hopfield retrieval quality as coherence signal
        let centroid_f32: Vec<f32> = centroid.iter().map(|&c| c as f32).collect();
        let hopfield_signal = match self.hopfield.retrieve(&centroid_f32) {
            Ok(retrieved) => {
                // High cosine similarity with centroid => good coherence
                let dot: f64 = retrieved
                    .iter()
                    .zip(centroid_f32.iter())
                    .map(|(a, b)| (*a as f64) * (*b as f64))
                    .sum();
                let norm_r: f64 = retrieved
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                let norm_c: f64 = centroid_f32
                    .iter()
                    .map(|x| (*x as f64).powi(2))
                    .sum::<f64>()
                    .sqrt();
                if norm_r > 1e-10 && norm_c > 1e-10 {
                    (dot / (norm_r * norm_c)).max(0.0)
                } else {
                    0.5
                }
            }
            Err(_) => 0.5, // No patterns stored yet
        };

        // Blend base coherence with Hopfield signal
        base_coherence * 0.7 + hopfield_signal * 0.3
    }

    /// Separate a pattern using DentateGyrus sparse encoding.
    ///
    /// Returns a high-dimensional sparse representation that maximizes
    /// separation between similar inputs (collision-resistant encoding).
    pub fn pattern_separate(&self, embedding: &[f32]) -> Vec<f32> {
        panic::catch_unwind(panic::AssertUnwindSafe(|| {
            self.dentate.encode_dense(embedding)
        }))
        .unwrap_or_else(|_| embedding.to_vec())
    }
}

impl Default for CognitiveEngine {
    fn default() -> Self {
        Self::new(128)
    }
}

fn euclidean_distance(a: &[f32], b: &[f32]) -> f64 {
    a.iter()
        .zip(b.iter())
        .map(|(x, y)| ((*x as f64) - (*y as f64)).powi(2))
        .sum::<f64>()
        .sqrt()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_cognitive_engine_creation() {
        let engine = CognitiveEngine::new(64);
        assert!(engine.anomaly_threshold > 0.0);
    }

    #[test]
    fn test_store_and_recall() {
        let mut engine = CognitiveEngine::new(4);
        engine.store_pattern("test-1", &[1.0, 0.0, 0.0, 0.0]);
        let recalled = engine.recall(&[0.9, 0.1, 0.0, 0.0]);
        assert!(recalled.is_some());
    }

    #[test]
    fn test_pattern_separate() {
        let engine = CognitiveEngine::new(8);
        let sep = engine.pattern_separate(&[1.0, 0.5, 0.3, 0.1, 0.0, 0.0, 0.0, 0.0]);
        // Should return a vector (either sparse encoding or fallback)
        assert!(!sep.is_empty());
    }

    #[test]
    fn test_cluster_coherence_single() {
        let engine = CognitiveEngine::new(4);
        // Single embedding => perfect coherence
        assert_eq!(engine.cluster_coherence(&[vec![1.0, 2.0, 3.0, 4.0]]), 1.0);
    }

    #[test]
    fn test_anomaly_detection() {
        let mut engine = CognitiveEngine::new(4);
        engine.store_pattern("normal", &[1.0, 1.0, 1.0, 1.0]);
        // Very different from centroid should be anomalous
        let is_anom = engine.is_anomalous(&[100.0, 100.0, 100.0, 100.0], &[1.0, 1.0, 1.0, 1.0]);
        assert!(is_anom);
    }

    #[test]
    fn test_default_engine() {
        let engine = CognitiveEngine::default();
        assert!(engine.anomaly_threshold > 0.0);
    }
}
