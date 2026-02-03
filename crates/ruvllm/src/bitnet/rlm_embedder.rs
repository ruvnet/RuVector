//! RLM-Style Recursive Sentence Transformer Embedder (AD-24)
//!
//! An inference strategy that wraps a base embedding model in a short iterative
//! loop: embed → retrieve neighbors → contextualize → re-embed → merge.
//!
//! This produces embeddings that are:
//! - Structurally aware (conditioned on RuVector neighborhood)
//! - Contradiction-sensitive (twin embeddings at low-cut boundaries)
//! - Domain-adaptive (without full fine-tuning)
//!
//! Three variants:
//! - **A: Query-Conditioned** — optimized for retrieval under a specific query
//! - **B: Corpus-Conditioned** — stable over time, less phrasing-sensitive
//! - **C: Contradiction-Aware Twin** — bimodal for disputed claims

use crate::error::{Result, RuvLLMError};

// ============================================================================
// Configuration
// ============================================================================

/// Configuration for the RLM recursive embedder.
#[derive(Debug, Clone)]
pub struct RlmEmbedderConfig {
    /// Embedding dimension of the base model
    pub embed_dim: usize,
    /// Maximum iterations in the recursive loop
    pub max_iterations: usize,
    /// Convergence threshold: stop if cosine(iter_n, iter_n-1) > this value
    pub convergence_threshold: f32,
    /// Number of neighbors to retrieve per iteration
    pub num_neighbors: usize,
    /// Merge weight for base embedding
    pub w_base: f32,
    /// Merge weight for contextualized embedding
    pub w_context: f32,
    /// Merge weight for anti-cluster embedding
    pub w_anti: f32,
    /// Contradiction detection threshold (cosine similarity below this = contested)
    pub contradiction_threshold: f32,
    /// Embedding variant to use
    pub variant: EmbeddingVariant,
}

impl Default for RlmEmbedderConfig {
    fn default() -> Self {
        Self {
            embed_dim: 384,
            max_iterations: 2,
            convergence_threshold: 0.98,
            num_neighbors: 5,
            w_base: 0.6,
            w_context: 0.3,
            w_anti: 0.1,
            contradiction_threshold: 0.3,
            variant: EmbeddingVariant::CorpusConditioned,
        }
    }
}

/// Embedding variant (AD-24).
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum EmbeddingVariant {
    /// Variant A: query-conditioned, optimized for retrieval under specific query
    QueryConditioned,
    /// Variant B: corpus-conditioned, stable over time
    CorpusConditioned,
    /// Variant C: contradiction-aware twin embeddings at low-cut boundaries
    ContradictionAwareTwin,
}

// ============================================================================
// Output Schema
// ============================================================================

/// Stop reason for the recursive loop.
#[derive(Debug, Clone, PartialEq)]
pub enum EmbedStopReason {
    /// Cosine similarity between iterations exceeded convergence threshold
    Converged,
    /// Maximum iterations reached
    MaxIterations,
    /// Contradiction detected — produced twin embeddings (Variant C only)
    Contested,
}

/// Neighbor context used during embedding.
#[derive(Debug, Clone)]
pub struct NeighborContext {
    /// Chunk ID in the evidence corpus
    pub chunk_id: String,
    /// Pre-computed embedding of this neighbor
    pub embedding: Vec<f32>,
    /// Whether this neighbor is in an opposing cluster
    pub is_contradicting: bool,
    /// Cosine similarity to the base embedding of the target chunk
    pub similarity: f32,
}

/// Result of the RLM embedding process.
#[derive(Debug, Clone)]
pub struct RlmEmbeddingResult {
    /// Primary embedding vector (normalized)
    pub embedding: Vec<f32>,
    /// Secondary embedding for Variant C (contradiction-aware twin)
    /// None for Variants A and B.
    pub twin_embedding: Option<Vec<f32>>,
    /// Confidence: cosine similarity between final and penultimate iteration
    pub confidence: f32,
    /// IDs of neighbors used as context
    pub evidence_neighbor_ids: Vec<String>,
    /// Per-neighbor contradiction flag
    pub contradiction_flags: Vec<bool>,
    /// Primary cluster assignment (if available)
    pub cluster_id: Option<usize>,
    /// Why the loop terminated
    pub stop_reason: EmbedStopReason,
    /// Number of iterations actually executed
    pub iterations_used: usize,
}

// ============================================================================
// Base Embedder Trait
// ============================================================================

/// Trait for the base embedding model. Implementations can wrap any sentence
/// transformer (MiniLM, BGE, nomic-embed, or even a ternary-quantized model).
pub trait BaseEmbedder {
    /// Embed a single text chunk into a fixed-dimension vector.
    fn embed(&self, text: &str) -> Result<Vec<f32>>;

    /// Embedding dimension.
    fn embed_dim(&self) -> usize;
}

/// Trait for retrieving neighbors from the evidence store (e.g., RuVector).
pub trait NeighborRetriever {
    /// Retrieve the k nearest neighbors for a given embedding.
    fn retrieve(&self, embedding: &[f32], k: usize) -> Result<Vec<NeighborContext>>;
}

// ============================================================================
// RLM Embedder
// ============================================================================

/// RLM-style recursive embedder.
///
/// Wraps a `BaseEmbedder` and `NeighborRetriever` to produce context-aware,
/// contradiction-sensitive embeddings via a bounded iterative loop.
pub struct RlmEmbedder<E: BaseEmbedder, R: NeighborRetriever> {
    embedder: E,
    retriever: R,
    config: RlmEmbedderConfig,
}

impl<E: BaseEmbedder, R: NeighborRetriever> RlmEmbedder<E, R> {
    /// Create a new RLM embedder with the given base embedder and retriever.
    pub fn new(embedder: E, retriever: R, config: RlmEmbedderConfig) -> Self {
        Self {
            embedder,
            retriever,
            config,
        }
    }

    /// Embed a text chunk using the RLM recursive strategy.
    ///
    /// For Variant A (query-conditioned), pass the query as `query_context`.
    /// For Variants B and C, `query_context` can be None.
    pub fn embed(
        &self,
        text: &str,
        query_context: Option<&str>,
    ) -> Result<RlmEmbeddingResult> {
        let dim = self.config.embed_dim;

        // Step 1: Base embedding
        let base_embedding = self.embedder.embed(text)?;
        if base_embedding.len() != dim {
            return Err(RuvLLMError::Model(format!(
                "Base embedder returned {} dims, expected {}",
                base_embedding.len(),
                dim
            )));
        }

        let mut current = base_embedding.clone();
        let mut prev = base_embedding.clone();
        let mut all_neighbors: Vec<NeighborContext> = Vec::new();
        let mut iterations_used = 0;
        let mut stop_reason = EmbedStopReason::MaxIterations;

        // Recursive loop (bounded)
        for iter in 0..self.config.max_iterations {
            iterations_used = iter + 1;

            // Step 2: Retrieve neighbors
            let neighbors = self.retriever.retrieve(&current, self.config.num_neighbors)?;

            // Store neighbor info
            for n in &neighbors {
                if !all_neighbors.iter().any(|existing| existing.chunk_id == n.chunk_id) {
                    all_neighbors.push(n.clone());
                }
            }

            // Step 3: Contextualize — compute context embedding from neighbors
            let ctx_embedding = self.compute_context_embedding(&current, &neighbors, query_context)?;

            // Step 4: Check for contradiction (Variant C)
            if self.config.variant == EmbeddingVariant::ContradictionAwareTwin {
                let contradicting: Vec<&NeighborContext> = neighbors
                    .iter()
                    .filter(|n| n.is_contradicting)
                    .collect();

                if !contradicting.is_empty() {
                    // Produce twin embeddings
                    let anti_embedding = self.compute_anti_embedding(&contradicting)?;
                    let twin_a = self.merge_embedding(&current, &ctx_embedding, &anti_embedding, 1.0);
                    let twin_b = self.merge_embedding(&current, &ctx_embedding, &anti_embedding, -1.0);

                    return Ok(RlmEmbeddingResult {
                        embedding: twin_a,
                        twin_embedding: Some(twin_b),
                        confidence: cosine_similarity(&current, &prev),
                        evidence_neighbor_ids: all_neighbors.iter().map(|n| n.chunk_id.clone()).collect(),
                        contradiction_flags: all_neighbors.iter().map(|n| n.is_contradicting).collect(),
                        cluster_id: None,
                        stop_reason: EmbedStopReason::Contested,
                        iterations_used,
                    });
                }
            }

            // Step 5: Merge
            let zero_anti = vec![0.0f32; dim];
            let anti_embedding = if self.config.w_anti > 0.0 {
                let contradicting: Vec<&NeighborContext> = neighbors
                    .iter()
                    .filter(|n| n.is_contradicting)
                    .collect();
                if contradicting.is_empty() {
                    zero_anti.clone()
                } else {
                    self.compute_anti_embedding(&contradicting)?
                }
            } else {
                zero_anti.clone()
            };

            prev = current.clone();
            current = self.merge_embedding(&current, &ctx_embedding, &anti_embedding, 1.0);

            // Step 6: Check convergence
            let sim = cosine_similarity(&current, &prev);
            if sim > self.config.convergence_threshold {
                stop_reason = EmbedStopReason::Converged;
                break;
            }
        }

        let confidence = cosine_similarity(&current, &prev);

        Ok(RlmEmbeddingResult {
            embedding: current,
            twin_embedding: None,
            confidence,
            evidence_neighbor_ids: all_neighbors.iter().map(|n| n.chunk_id.clone()).collect(),
            contradiction_flags: all_neighbors.iter().map(|n| n.is_contradicting).collect(),
            cluster_id: None,
            stop_reason,
            iterations_used,
        })
    }

    /// Compute context embedding by averaging neighbor embeddings,
    /// optionally weighted by similarity. For Variant A, also factor
    /// in the query embedding.
    fn compute_context_embedding(
        &self,
        _base: &[f32],
        neighbors: &[NeighborContext],
        query_context: Option<&str>,
    ) -> Result<Vec<f32>> {
        let dim = self.config.embed_dim;

        if neighbors.is_empty() {
            return Ok(vec![0.0f32; dim]);
        }

        // Weighted average of neighbor embeddings (weight = similarity)
        let mut ctx = vec![0.0f32; dim];
        let mut total_weight = 0.0f32;

        for n in neighbors {
            if n.is_contradicting {
                continue; // Skip contradicting neighbors for context
            }
            let w = n.similarity.max(0.0);
            for (i, &val) in n.embedding.iter().enumerate() {
                if i < dim {
                    ctx[i] += val * w;
                }
            }
            total_weight += w;
        }

        if total_weight > 0.0 {
            for v in ctx.iter_mut() {
                *v /= total_weight;
            }
        }

        // Variant A: blend with query embedding
        if let (EmbeddingVariant::QueryConditioned, Some(query)) =
            (self.config.variant, query_context)
        {
            let query_emb = self.embedder.embed(query)?;
            let query_weight = 0.3;
            for (i, v) in ctx.iter_mut().enumerate() {
                if i < query_emb.len() {
                    *v = *v * (1.0 - query_weight) + query_emb[i] * query_weight;
                }
            }
        }

        Ok(ctx)
    }

    /// Compute anti-cluster embedding from contradicting neighbors.
    fn compute_anti_embedding(&self, contradicting: &[&NeighborContext]) -> Result<Vec<f32>> {
        let dim = self.config.embed_dim;
        let mut anti = vec![0.0f32; dim];
        let count = contradicting.len() as f32;

        if count == 0.0 {
            return Ok(anti);
        }

        for n in contradicting {
            for (i, &val) in n.embedding.iter().enumerate() {
                if i < dim {
                    anti[i] += val;
                }
            }
        }

        for v in anti.iter_mut() {
            *v /= count;
        }

        Ok(anti)
    }

    /// Merge base, context, and anti-cluster embeddings using the auditable merge rule.
    ///
    /// `anti_sign` controls whether anti pushes away (+1.0) or toward (-1.0).
    /// For twin embedding Variant C, the second twin uses anti_sign = -1.0.
    fn merge_embedding(
        &self,
        base: &[f32],
        ctx: &[f32],
        anti: &[f32],
        anti_sign: f32,
    ) -> Vec<f32> {
        let dim = self.config.embed_dim;
        let mut merged = vec![0.0f32; dim];

        for i in 0..dim {
            let b = if i < base.len() { base[i] } else { 0.0 };
            let c = if i < ctx.len() { ctx[i] } else { 0.0 };
            let a = if i < anti.len() { anti[i] } else { 0.0 };
            merged[i] = self.config.w_base * b
                + self.config.w_context * c
                + self.config.w_anti * anti_sign * a;
        }

        l2_normalize(&mut merged);
        merged
    }

    /// Get the current configuration.
    pub fn config(&self) -> &RlmEmbedderConfig {
        &self.config
    }
}

// ============================================================================
// Math Helpers
// ============================================================================

/// Cosine similarity between two vectors.
pub fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let len = a.len().min(b.len());
    if len == 0 {
        return 0.0;
    }

    let mut dot = 0.0f32;
    let mut norm_a = 0.0f32;
    let mut norm_b = 0.0f32;

    for i in 0..len {
        dot += a[i] * b[i];
        norm_a += a[i] * a[i];
        norm_b += b[i] * b[i];
    }

    let denom = (norm_a.sqrt() * norm_b.sqrt()).max(1e-10);
    dot / denom
}

/// L2 normalize a vector in-place.
pub fn l2_normalize(v: &mut [f32]) {
    let mut norm = 0.0f32;
    for &x in v.iter() {
        norm += x * x;
    }
    norm = norm.sqrt().max(1e-10);
    for x in v.iter_mut() {
        *x /= norm;
    }
}

/// Compute the mean of a set of embeddings.
pub fn mean_embedding(embeddings: &[&[f32]], dim: usize) -> Vec<f32> {
    let mut result = vec![0.0f32; dim];
    if embeddings.is_empty() {
        return result;
    }
    let count = embeddings.len() as f32;
    for emb in embeddings {
        for (i, &v) in emb.iter().enumerate() {
            if i < dim {
                result[i] += v;
            }
        }
    }
    for v in result.iter_mut() {
        *v /= count;
    }
    result
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;

    // -- Test implementations of traits --

    struct MockEmbedder {
        dim: usize,
    }

    impl BaseEmbedder for MockEmbedder {
        fn embed(&self, text: &str) -> Result<Vec<f32>> {
            // Deterministic embedding: hash text bytes into a vector
            let mut emb = vec![0.0f32; self.dim];
            for (i, byte) in text.bytes().enumerate() {
                emb[i % self.dim] += (byte as f32 - 128.0) / 128.0;
            }
            l2_normalize(&mut emb);
            Ok(emb)
        }

        fn embed_dim(&self) -> usize {
            self.dim
        }
    }

    struct MockRetriever {
        neighbors: Vec<NeighborContext>,
    }

    impl NeighborRetriever for MockRetriever {
        fn retrieve(&self, _embedding: &[f32], k: usize) -> Result<Vec<NeighborContext>> {
            Ok(self.neighbors.iter().take(k).cloned().collect())
        }
    }

    fn make_neighbor(id: &str, dim: usize, is_contradicting: bool, sim: f32) -> NeighborContext {
        let mut emb = vec![0.0f32; dim];
        // Deterministic based on id
        for (i, byte) in id.bytes().enumerate() {
            emb[i % dim] = (byte as f32 - 100.0) / 100.0;
        }
        l2_normalize(&mut emb);
        NeighborContext {
            chunk_id: id.to_string(),
            embedding: emb,
            is_contradicting,
            similarity: sim,
        }
    }

    #[test]
    fn test_cosine_similarity_identical() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) - 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_orthogonal() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![0.0, 1.0, 0.0];
        assert!(cosine_similarity(&a, &b).abs() < 1e-6);
    }

    #[test]
    fn test_cosine_similarity_opposite() {
        let a = vec![1.0, 0.0, 0.0];
        let b = vec![-1.0, 0.0, 0.0];
        assert!((cosine_similarity(&a, &b) + 1.0).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize() {
        let mut v = vec![3.0, 4.0];
        l2_normalize(&mut v);
        let norm: f32 = v.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 1e-6);
        assert!((v[0] - 0.6).abs() < 1e-6);
        assert!((v[1] - 0.8).abs() < 1e-6);
    }

    #[test]
    fn test_l2_normalize_zero_vector() {
        let mut v = vec![0.0, 0.0, 0.0];
        l2_normalize(&mut v);
        // Should not panic, values stay near zero
        assert!(v.iter().all(|&x| x.abs() < 1e-5));
    }

    #[test]
    fn test_mean_embedding() {
        let a = vec![1.0, 0.0];
        let b = vec![0.0, 1.0];
        let mean = mean_embedding(&[&a, &b], 2);
        assert!((mean[0] - 0.5).abs() < 1e-6);
        assert!((mean[1] - 0.5).abs() < 1e-6);
    }

    #[test]
    fn test_embed_corpus_conditioned() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![
                make_neighbor("doc-1", dim, false, 0.9),
                make_neighbor("doc-2", dim, false, 0.8),
            ],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 2,
            variant: EmbeddingVariant::CorpusConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("test chunk text", None).unwrap();

        assert_eq!(result.embedding.len(), dim);
        assert!(result.confidence > 0.0);
        assert_eq!(result.evidence_neighbor_ids.len(), 2);
        assert!(result.twin_embedding.is_none());
        assert!(result.iterations_used <= 2);
    }

    #[test]
    fn test_embed_query_conditioned() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![make_neighbor("doc-1", dim, false, 0.9)],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 2,
            variant: EmbeddingVariant::QueryConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("chunk", Some("what is X?")).unwrap();

        assert_eq!(result.embedding.len(), dim);
        assert!(result.twin_embedding.is_none());
    }

    #[test]
    fn test_embed_contradiction_aware_twin() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![
                make_neighbor("agree-1", dim, false, 0.9),
                make_neighbor("contra-1", dim, true, 0.7),
            ],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 2,
            variant: EmbeddingVariant::ContradictionAwareTwin,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("contested claim", None).unwrap();

        assert_eq!(result.embedding.len(), dim);
        assert!(result.twin_embedding.is_some());
        assert_eq!(result.stop_reason, EmbedStopReason::Contested);

        // Twin embeddings should differ
        let twin = result.twin_embedding.as_ref().unwrap();
        let sim = cosine_similarity(&result.embedding, twin);
        assert!(sim < 0.99, "Twin embeddings should differ, got cosine={}", sim);
    }

    #[test]
    fn test_embed_no_neighbors() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 2,
            variant: EmbeddingVariant::CorpusConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("isolated chunk", None).unwrap();

        assert_eq!(result.embedding.len(), dim);
        assert!(result.evidence_neighbor_ids.is_empty());
    }

    #[test]
    fn test_embed_convergence_stops_early() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        // Same neighbor every time → should converge quickly
        let retriever = MockRetriever {
            neighbors: vec![make_neighbor("stable-1", dim, false, 0.95)],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 10, // High max, but should converge before
            convergence_threshold: 0.95,
            variant: EmbeddingVariant::CorpusConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("converging chunk", None).unwrap();

        // Should stop before 10 iterations
        assert!(result.iterations_used < 10);
        assert_eq!(result.stop_reason, EmbedStopReason::Converged);
    }

    #[test]
    fn test_embed_output_is_normalized() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![make_neighbor("doc-1", dim, false, 0.8)],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("test", None).unwrap();

        let norm: f32 = result.embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!(
            (norm - 1.0).abs() < 1e-4,
            "Output embedding should be L2-normalized, got norm={}",
            norm
        );
    }

    #[test]
    fn test_contradiction_flags_populated() {
        let dim = 8;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![
                make_neighbor("agree", dim, false, 0.9),
                make_neighbor("contra", dim, true, 0.7),
                make_neighbor("agree2", dim, false, 0.6),
            ],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 1,
            variant: EmbeddingVariant::CorpusConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("chunk", None).unwrap();

        assert_eq!(result.contradiction_flags.len(), 3);
        assert!(!result.contradiction_flags[0]); // agree
        assert!(result.contradiction_flags[1]); // contra
        assert!(!result.contradiction_flags[2]); // agree2
    }

    #[test]
    fn test_embedding_result_metadata() {
        let dim = 4;
        let embedder = MockEmbedder { dim };
        let retriever = MockRetriever {
            neighbors: vec![make_neighbor("n1", dim, false, 0.5)],
        };
        let config = RlmEmbedderConfig {
            embed_dim: dim,
            max_iterations: 2,
            variant: EmbeddingVariant::CorpusConditioned,
            ..Default::default()
        };

        let rlm = RlmEmbedder::new(embedder, retriever, config);
        let result = rlm.embed("meta test", None).unwrap();

        assert!(!result.evidence_neighbor_ids.is_empty());
        assert!(result.confidence >= -1.0 && result.confidence <= 1.0);
        assert!(result.iterations_used >= 1);
    }
}
