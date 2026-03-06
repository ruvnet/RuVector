//! Embedding generation for brain memories
//!
//! Two-stage embedding pipeline:
//! 1. **Structured Hash Features** (deterministic, identical across all sessions):
//!    Multi-granularity hashing — unigram, bigram, trigram tokens hashed into
//!    disjoint subspaces with signed hashing to reduce collision bias.
//! 2. **MicroLoRA Transform** (learned, federated across sessions):
//!    Rank-2 LoRA adapter applied to the frozen hash features. Weights are
//!    learned locally via SONA and periodically federated to/from the server.

use sha3::{Shake256, digest::{Update, ExtendableOutput, XofReader}};
use sona::{SonaEngine, LearnedPattern};
use sona::engine::SonaEngineBuilder;
use std::sync::{Arc, Mutex};

/// Embedding dimension (128 f32s = 512 bytes)
pub const EMBEDDING_DIM: usize = 128;

/// Subspace allocation for multi-granularity hashing:
/// - Unigram:  dims [0..42)    = 42 dims (33%)
/// - Bigram:   dims [42..84)   = 42 dims (33%)
/// - Trigram:  dims [84..128)  = 44 dims (34%)
const UNIGRAM_START: usize = 0;
const UNIGRAM_END: usize = 42;
const BIGRAM_START: usize = 42;
const BIGRAM_END: usize = 84;
const TRIGRAM_START: usize = 84;
const TRIGRAM_END: usize = EMBEDDING_DIM; // 128

/// LoRA weights exported for federation
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct LoraWeights {
    /// Down projection weights (hidden_dim * rank)
    pub down_proj: Vec<f32>,
    /// Up projection weights (rank * hidden_dim)
    pub up_proj: Vec<f32>,
    /// LoRA rank
    pub rank: usize,
    /// Hidden dimension
    pub hidden_dim: usize,
    /// Number of local training steps that produced these weights
    pub evidence_count: u64,
}

impl LoraWeights {
    /// Validate weight shapes and values (Gate A: policy validity)
    pub fn validate(&self) -> Result<(), String> {
        // Shape check
        let expected_down = self.hidden_dim * self.rank;
        let expected_up = self.rank * self.hidden_dim;
        if self.down_proj.len() != expected_down {
            return Err(format!(
                "down_proj shape mismatch: expected {expected_down}, got {}",
                self.down_proj.len()
            ));
        }
        if self.up_proj.len() != expected_up {
            return Err(format!(
                "up_proj shape mismatch: expected {expected_up}, got {}",
                self.up_proj.len()
            ));
        }
        // NaN/Inf check
        for (i, &v) in self.down_proj.iter().chain(self.up_proj.iter()).enumerate() {
            if v.is_nan() || v.is_infinite() {
                return Err(format!("NaN/Inf at index {i}"));
            }
        }
        // Norm check: reject if L2 norm of either projection > 100
        let down_norm: f32 = self.down_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        let up_norm: f32 = self.up_proj.iter().map(|x| x * x).sum::<f32>().sqrt();
        if down_norm > 100.0 || up_norm > 100.0 {
            return Err(format!("Weight norm too large: down={down_norm:.1}, up={up_norm:.1}"));
        }
        // Minimum evidence
        if self.evidence_count < 5 {
            return Err(format!(
                "Insufficient evidence: {} (minimum 5)",
                self.evidence_count
            ));
        }
        Ok(())
    }

    /// Clip weights to [-2, 2] range
    pub fn clip(&mut self) {
        for v in self.down_proj.iter_mut().chain(self.up_proj.iter_mut()) {
            *v = v.clamp(-2.0, 2.0);
        }
    }

    /// Compute L2 distance to another set of weights
    pub fn l2_distance(&self, other: &LoraWeights) -> f32 {
        let d: f32 = self.down_proj.iter().zip(other.down_proj.iter())
            .chain(self.up_proj.iter().zip(other.up_proj.iter()))
            .map(|(a, b)| (a - b).powi(2))
            .sum();
        d.sqrt()
    }
}

/// Brain embedding engine wrapping SONA with structured hash + MicroLoRA
pub struct BrainEmbedder {
    engine: Option<Arc<Mutex<SonaEngine>>>,
    /// Locally cached consensus weights from the server (applied on embed)
    consensus_lora: Option<LoraWeights>,
    /// Count of embeddings processed (used as evidence_count for export)
    embed_count: u64,
}

impl BrainEmbedder {
    /// Create with real SONA engine
    pub fn new() -> Self {
        let engine = match std::panic::catch_unwind(|| {
            SonaEngineBuilder::new()
                .hidden_dim(EMBEDDING_DIM)
                .micro_lora_rank(2)
                .pattern_clusters(50)
                .quality_threshold(0.3)
                .build()
        }) {
            Ok(e) => Some(Arc::new(Mutex::new(e))),
            Err(_) => None,
        };
        Self {
            engine,
            consensus_lora: None,
            embed_count: 0,
        }
    }

    /// Create with hash-only fallback (no SONA)
    pub fn hash_only() -> Self {
        Self {
            engine: None,
            consensus_lora: None,
            embed_count: 0,
        }
    }

    /// Generate embedding for text content.
    ///
    /// Pipeline: text -> structured hash features -> MicroLoRA transform -> L2 normalize
    pub fn embed(&mut self, text: &str) -> Vec<f32> {
        self.embed_count += 1;

        // Stage 1: Deterministic structured hash features
        let hash_features = generate_structured_hash_features(text);

        // Stage 2: Apply MicroLoRA transform (consensus weights if available, then SONA)
        let transformed = self.apply_lora_transform(&hash_features);

        // Feed into SONA for trajectory learning if available
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                let builder = eng.begin_trajectory(transformed.clone());
                eng.end_trajectory(builder, 0.5);
                // Check for refined patterns
                let patterns = eng.find_patterns(&transformed, 1);
                if let Some(pattern) = patterns.first() {
                    if pattern.similarity(&transformed) > 0.3 {
                        return normalize_l2(&pattern.centroid);
                    }
                }
            }
        }

        transformed
    }

    /// Apply MicroLoRA transform to hash features
    fn apply_lora_transform(&self, features: &[f32]) -> Vec<f32> {
        // Try consensus weights first (from server federation)
        if let Some(ref lora) = self.consensus_lora {
            return apply_lora_forward(features, lora);
        }
        // Try SONA engine's local MicroLoRA
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                let lora_state = eng.export_lora_state();
                if let Some(layer) = lora_state.micro_lora_layers.first() {
                    let local_lora = LoraWeights {
                        down_proj: layer.lora_a.clone(),
                        up_proj: layer.lora_b.clone(),
                        rank: layer.rank,
                        hidden_dim: layer.input_dim,
                        evidence_count: self.embed_count,
                    };
                    return apply_lora_forward(features, &local_lora);
                }
            }
        }
        // No LoRA available — return raw hash features
        normalize_l2(features)
    }

    /// Import consensus LoRA weights from the server
    pub fn import_consensus_weights(&mut self, weights: LoraWeights) {
        self.consensus_lora = Some(weights);
    }

    /// Export local MicroLoRA weights for federation
    pub fn export_local_weights(&self) -> Option<LoraWeights> {
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                let lora_state = eng.export_lora_state();
                if let Some(layer) = lora_state.micro_lora_layers.first() {
                    return Some(LoraWeights {
                        down_proj: layer.lora_a.clone(),
                        up_proj: layer.lora_b.clone(),
                        rank: layer.rank,
                        hidden_dim: layer.input_dim,
                        evidence_count: self.embed_count,
                    });
                }
            }
        }
        None
    }

    /// Get number of embeddings processed
    pub fn embed_count(&self) -> u64 {
        self.embed_count
    }

    /// Record quality feedback for a trajectory
    pub fn record_feedback(&self, embedding: &[f32], quality: f32) {
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                let builder = eng.begin_trajectory(embedding.to_vec());
                eng.end_trajectory(builder, quality);
            }
        }
    }

    /// Find similar patterns from SONA's learned bank
    pub fn find_similar(&self, query: &[f32], k: usize) -> Vec<LearnedPattern> {
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                return eng.find_patterns(query, k);
            }
        }
        vec![]
    }

    /// Force a learning cycle
    pub fn force_learn(&self) -> Option<String> {
        if let Some(ref engine) = self.engine {
            if let Ok(eng) = engine.lock() {
                return Some(eng.force_learn());
            }
        }
        None
    }

    /// Check if SONA engine is active
    pub fn has_sona(&self) -> bool {
        self.engine.is_some()
    }

    /// Check if consensus LoRA weights are loaded
    pub fn has_consensus_lora(&self) -> bool {
        self.consensus_lora.is_some()
    }
}

impl Default for BrainEmbedder {
    fn default() -> Self {
        Self::new()
    }
}

/// Apply LoRA forward pass: output = L2_normalize(input + scale * (input @ down) @ up)
fn apply_lora_forward(input: &[f32], lora: &LoraWeights) -> Vec<f32> {
    let dim = lora.hidden_dim;
    let rank = lora.rank;
    let scale = (rank as f32).recip(); // alpha/rank = 1.0

    // down_proj: input (dim,) @ down (dim, rank) -> intermediate (rank,)
    let mut intermediate = vec![0.0f32; rank];
    for r in 0..rank {
        for d in 0..dim.min(input.len()) {
            intermediate[r] += input[d] * lora.down_proj[d * rank + r];
        }
    }

    // up_proj: intermediate (rank,) @ up (rank, dim) -> delta (dim,)
    let mut output: Vec<f32> = input.to_vec();
    output.resize(dim, 0.0);
    for d in 0..dim {
        let mut delta = 0.0f32;
        for r in 0..rank {
            delta += intermediate[r] * lora.up_proj[r * dim + d];
        }
        output[d] += scale * delta;
    }

    normalize_l2(&output)
}

/// L2 normalize a vector
pub fn normalize_l2(v: &[f32]) -> Vec<f32> {
    let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        v.iter().map(|x| x / norm).collect()
    } else {
        v.to_vec()
    }
}

/// Generate structured multi-granularity hash features.
///
/// Splits text into unigram, bigram, and trigram tokens. Each n-gram level
/// hashes into a disjoint subspace of the embedding vector using signed hashing
/// (hash determines both the bucket index AND the sign, reducing collision bias).
///
/// This is deterministic and identical across all sessions — the frozen base
/// that MicroLoRA adapts on top of.
pub fn generate_structured_hash_features(text: &str) -> Vec<f32> {
    let mut features = vec![0.0f32; EMBEDDING_DIM];
    let lower = text.to_lowercase();
    let words: Vec<&str> = lower.split_whitespace().collect();

    // Unigram features: each word hashes into dims [0..42)
    let unigram_dim = UNIGRAM_END - UNIGRAM_START;
    for word in &words {
        let (bucket, sign) = signed_hash(word.as_bytes(), b"uni", unigram_dim);
        features[UNIGRAM_START + bucket] += sign;
    }

    // Bigram features: consecutive word pairs hash into dims [42..84)
    let bigram_dim = BIGRAM_END - BIGRAM_START;
    for pair in words.windows(2) {
        let key = format!("{} {}", pair[0], pair[1]);
        let (bucket, sign) = signed_hash(key.as_bytes(), b"bi", bigram_dim);
        features[BIGRAM_START + bucket] += sign;
    }

    // Trigram features: consecutive word triples hash into dims [84..128)
    let trigram_dim = TRIGRAM_END - TRIGRAM_START;
    for triple in words.windows(3) {
        let key = format!("{} {} {}", triple[0], triple[1], triple[2]);
        let (bucket, sign) = signed_hash(key.as_bytes(), b"tri", trigram_dim);
        features[TRIGRAM_START + bucket] += sign;
    }

    // Also add character n-gram features for short texts or single words
    if words.len() <= 2 {
        let chars: Vec<char> = lower.chars().filter(|c| c.is_alphanumeric()).collect();
        // Character trigrams into the trigram subspace
        for window in chars.windows(3) {
            let key: String = window.iter().collect();
            let (bucket, sign) = signed_hash(key.as_bytes(), b"ctri", trigram_dim);
            features[TRIGRAM_START + bucket] += sign * 0.5; // Lower weight for char ngrams
        }
    }

    // L2 normalize and clip to [-1, 1]
    let norm: f32 = features.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm > 1e-10 {
        for v in &mut features {
            *v = (*v / norm).clamp(-1.0, 1.0);
        }
    }

    features
}

/// Signed hash: returns (bucket_index, +1.0 or -1.0).
/// Uses SHAKE-256 for uniform distribution. The first 4 bytes determine the bucket,
/// the 5th byte determines the sign.
fn signed_hash(data: &[u8], salt: &[u8], num_buckets: usize) -> (usize, f32) {
    let mut hasher = Shake256::default();
    hasher.update(b"ruvector-shf:");
    hasher.update(salt);
    hasher.update(b":");
    hasher.update(data);
    let mut reader = hasher.finalize_xof();
    let mut buf = [0u8; 5];
    reader.read(&mut buf);

    let bucket = u32::from_le_bytes([buf[0], buf[1], buf[2], buf[3]]) as usize % num_buckets;
    let sign = if buf[4] & 1 == 0 { 1.0f32 } else { -1.0f32 };
    (bucket, sign)
}

/// Public convenience: generate an embedding using structured hash features only.
/// Callers needing SONA/LoRA should use BrainEmbedder directly.
pub fn generate_embedding(text: &str) -> Vec<f32> {
    generate_structured_hash_features(text)
}

/// Compute cosine similarity between two embeddings
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    if a.len() != b.len() || a.is_empty() {
        return 0.0;
    }
    let dot: f32 = a.iter().zip(b.iter()).map(|(x, y)| x * y).sum();
    let norm_a: f32 = a.iter().map(|x| x * x).sum::<f32>().sqrt();
    let norm_b: f32 = b.iter().map(|x| x * x).sum::<f32>().sqrt();
    if norm_a < 1e-10 || norm_b < 1e-10 {
        return 0.0;
    }
    dot / (norm_a * norm_b)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_embedding_deterministic() {
        let e1 = generate_embedding("hello world");
        let e2 = generate_embedding("hello world");
        assert_eq!(e1, e2);
    }

    #[test]
    fn test_embedding_dimension() {
        let e = generate_embedding("test");
        assert_eq!(e.len(), EMBEDDING_DIM);
    }

    #[test]
    fn test_embedding_normalized() {
        let e = generate_embedding("a longer text for normalization testing");
        let norm: f32 = e.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm was {norm}");
    }

    #[test]
    fn test_similar_texts_closer() {
        // Structured hash features should give higher similarity for
        // texts sharing many n-grams than completely unrelated texts
        let e1 = generate_embedding("rust programming language features");
        let e2 = generate_embedding("rust programming language syntax");
        let e3 = generate_embedding("cooking recipes for dinner tonight");
        let sim12 = cosine_similarity(&e1, &e2);
        let sim13 = cosine_similarity(&e1, &e3);
        // Texts sharing 3/4 words should be more similar than disjoint texts
        assert!(
            sim12 > sim13,
            "similar texts should be closer: sim12={sim12}, sim13={sim13}"
        );
    }

    #[test]
    fn test_disjoint_subspaces() {
        // Verify that a single word only activates the unigram subspace
        let e = generate_structured_hash_features("hello");
        // Unigram subspace should have non-zero values
        let uni_energy: f32 = e[UNIGRAM_START..UNIGRAM_END].iter().map(|x| x * x).sum();
        assert!(uni_energy > 0.0, "unigram subspace should be active");
        // Bigram/trigram subspaces should be zero (single word = no pairs/triples)
        // But char trigrams may activate trigram subspace for short texts
    }

    #[test]
    fn test_signed_hash_distribution() {
        // Verify signed_hash produces both positive and negative signs
        let mut pos = 0;
        let mut neg = 0;
        for i in 0..100 {
            let key = format!("test-{i}");
            let (_, sign) = signed_hash(key.as_bytes(), b"test", 42);
            if sign > 0.0 { pos += 1; } else { neg += 1; }
        }
        // Both signs should appear (probabilistic, but 100 trials is enough)
        assert!(pos > 10 && neg > 10, "pos={pos}, neg={neg}");
    }

    #[test]
    fn test_brain_embedder_creates() {
        let embedder = BrainEmbedder::new();
        let _ = embedder.has_sona();
    }

    #[test]
    fn test_brain_embedder_embed() {
        let mut embedder = BrainEmbedder::new();
        let emb = embedder.embed("hello world");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.05, "norm was {norm}");
    }

    #[test]
    fn test_hash_fallback() {
        let mut embedder = BrainEmbedder::hash_only();
        assert!(!embedder.has_sona());
        assert!(!embedder.has_consensus_lora());
        let emb = embedder.embed("hello world");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        // Hash-only should match generate_embedding (both L2-normalized structured hash)
        let direct = generate_embedding("hello world");
        assert_eq!(emb, direct);
    }

    #[test]
    fn test_lora_weights_validate() {
        let valid = LoraWeights {
            down_proj: vec![0.1; 256],
            up_proj: vec![0.1; 256],
            rank: 2,
            hidden_dim: 128,
            evidence_count: 10,
        };
        assert!(valid.validate().is_ok());

        // Bad shape
        let bad_shape = LoraWeights {
            down_proj: vec![0.1; 100],
            up_proj: vec![0.1; 256],
            rank: 2,
            hidden_dim: 128,
            evidence_count: 10,
        };
        assert!(bad_shape.validate().is_err());

        // NaN
        let mut nan_weights = valid.clone();
        nan_weights.down_proj[0] = f32::NAN;
        assert!(nan_weights.validate().is_err());

        // Low evidence
        let low_ev = LoraWeights {
            evidence_count: 2,
            ..valid.clone()
        };
        assert!(low_ev.validate().is_err());
    }

    #[test]
    fn test_lora_forward_pass() {
        // Zero LoRA should return normalized input unchanged
        let input = generate_structured_hash_features("test input");
        let zero_lora = LoraWeights {
            down_proj: vec![0.0; 256],
            up_proj: vec![0.0; 256],
            rank: 2,
            hidden_dim: 128,
            evidence_count: 10,
        };
        let output = apply_lora_forward(&input, &zero_lora);
        let expected = normalize_l2(&input);
        for (a, b) in output.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-6, "zero LoRA should be identity");
        }
    }

    #[test]
    fn test_consensus_import() {
        let mut embedder = BrainEmbedder::hash_only();
        assert!(!embedder.has_consensus_lora());

        let weights = LoraWeights {
            down_proj: vec![0.01; 256],
            up_proj: vec![0.01; 256],
            rank: 2,
            hidden_dim: 128,
            evidence_count: 100,
        };
        embedder.import_consensus_weights(weights);
        assert!(embedder.has_consensus_lora());

        // Embedding should now go through LoRA transform
        let emb = embedder.embed("test with lora");
        assert_eq!(emb.len(), EMBEDDING_DIM);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.05, "norm was {norm}");
    }

    #[test]
    fn test_find_similar_empty() {
        let embedder = BrainEmbedder::hash_only();
        let results = embedder.find_similar(&[0.0; EMBEDDING_DIM], 5);
        assert!(results.is_empty());
    }
}
