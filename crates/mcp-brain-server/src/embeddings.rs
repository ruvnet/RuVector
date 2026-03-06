//! Neural embedding engine using ruvllm's full RLM recursive embedder pipeline.
//!
//! Three-phase architecture:
//! - Phase 1: HashEmbedder base (FNV-1a + bigrams, L2-normalized) — fallback for empty corpus
//! - Phase 2: RlmEmbedder recursive context-aware embeddings (active when corpus ≥ 3)
//! - Phase 3: candle sentence transformer (future — requires `candle` feature)
//!
//! The RlmEmbedder produces embeddings conditioned on the existing knowledge corpus:
//! - Storage uses CorpusConditioned variant (stable over time)
//! - Search uses QueryConditioned variant (optimized for retrieval relevance)

use ruvllm::bitnet::rlm_embedder::{
    BaseEmbedder, EmbeddingVariant, FlatNeighborStore, HashEmbedder, RlmEmbedder,
    RlmEmbedderConfig,
};

/// Embedding dimension used across the brain.
pub const EMBED_DIM: usize = 128;

/// Minimum corpus size before RlmEmbedder contextualization kicks in.
/// At 50+ documents the corpus is diverse enough for RLM's contextual
/// neighbor-weighted embeddings to outperform hash-based embeddings.
/// RLM uses QueryConditioned mode for search (optimized retrieval) and
/// CorpusConditioned for storage (stable over time).
const RLM_MIN_CORPUS: usize = 50;

/// Wraps the ruvllm embedding pipeline for the brain server.
///
/// On startup the corpus is empty so embeddings use pure `HashEmbedder`.
/// As memories are shared, the `FlatNeighborStore` grows and recursive
/// context-aware re-embedding kicks in — each new embedding is conditioned
/// on its neighbors in the knowledge corpus.
pub struct EmbeddingEngine {
    embedder: HashEmbedder,
    store: FlatNeighborStore,
    /// Config for query-conditioned embeddings (search)
    query_config: RlmEmbedderConfig,
    /// Config for corpus-conditioned embeddings (storage)
    corpus_config: RlmEmbedderConfig,
}

impl EmbeddingEngine {
    pub fn new() -> Self {
        Self {
            embedder: HashEmbedder::new(EMBED_DIM),
            store: FlatNeighborStore::new(EMBED_DIM),
            query_config: RlmEmbedderConfig {
                embed_dim: EMBED_DIM,
                max_iterations: 2,
                convergence_threshold: 0.98,
                num_neighbors: 5,
                w_base: 0.6,
                w_context: 0.3,
                w_anti: 0.1,
                contradiction_threshold: 0.3,
                variant: EmbeddingVariant::QueryConditioned,
            },
            corpus_config: RlmEmbedderConfig {
                embed_dim: EMBED_DIM,
                max_iterations: 2,
                convergence_threshold: 0.98,
                num_neighbors: 5,
                w_base: 0.7,
                w_context: 0.25,
                w_anti: 0.05,
                contradiction_threshold: 0.3,
                variant: EmbeddingVariant::CorpusConditioned,
            },
        }
    }

    /// Embed text for query (search). Uses QueryConditioned variant when corpus
    /// is large enough — optimizes for retrieval relevance.
    pub fn embed(&self, text: &str) -> Vec<f32> {
        if self.store.len() < RLM_MIN_CORPUS {
            return self.hash_embed(text);
        }
        let rlm = RlmEmbedder::new(
            self.embedder.clone(),
            self.store.clone(),
            self.query_config.clone(),
        );
        match rlm.embed(text, None) {
            Ok(result) => result.embedding,
            Err(_) => self.hash_embed(text),
        }
    }

    /// Embed text for storage. Uses CorpusConditioned variant when corpus
    /// is large enough — produces stable embeddings less sensitive to phrasing.
    pub fn embed_for_storage(&self, text: &str) -> Vec<f32> {
        if self.store.len() < RLM_MIN_CORPUS {
            return self.hash_embed(text);
        }
        let rlm = RlmEmbedder::new(
            self.embedder.clone(),
            self.store.clone(),
            self.corpus_config.clone(),
        );
        match rlm.embed(text, None) {
            Ok(result) => result.embedding,
            Err(_) => self.hash_embed(text),
        }
    }

    /// Pure HashEmbedder fallback.
    fn hash_embed(&self, text: &str) -> Vec<f32> {
        self.embedder
            .embed(text)
            .unwrap_or_else(|_| vec![0.0; EMBED_DIM])
    }

    /// Add a document to the neighbor store so future embeddings are contextualized.
    pub fn add_to_corpus(&mut self, id: &str, embedding: Vec<f32>, cluster_id: Option<usize>) {
        self.store.add(id, embedding, cluster_id);
    }

    /// Number of documents in the corpus.
    pub fn corpus_size(&self) -> usize {
        self.store.len()
    }

    /// Whether RlmEmbedder is active (corpus large enough).
    pub fn is_rlm_active(&self) -> bool {
        self.store.len() >= RLM_MIN_CORPUS
    }

    /// Embedding dimension.
    pub fn dim(&self) -> usize {
        EMBED_DIM
    }

    /// Engine name for status reporting.
    pub fn engine_name(&self) -> &'static str {
        if self.is_rlm_active() {
            "ruvllm::RlmEmbedder"
        } else {
            "ruvllm::HashEmbedder"
        }
    }

    /// Combine title + content + tags into embeddable text.
    pub fn prepare_text(title: &str, content: &str, tags: &[String]) -> String {
        let tag_str = tags.join(" ");
        format!("{} {} {}", title, content, tag_str)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ruvllm::bitnet::rlm_embedder::cosine_similarity;

    #[test]
    fn test_embed_basic() {
        let engine = EmbeddingEngine::new();
        let emb = engine.embed("hello world");
        assert_eq!(emb.len(), EMBED_DIM);
        let norm: f32 = emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.01, "norm={}", norm);
    }

    #[test]
    fn test_deterministic() {
        let engine = EmbeddingEngine::new();
        let a = engine.embed("test input");
        let b = engine.embed("test input");
        assert_eq!(a, b);
    }

    #[test]
    fn test_similar_texts_closer() {
        let engine = EmbeddingEngine::new();
        let a = engine.embed("rust programming language");
        let b = engine.embed("rust code development");
        let c = engine.embed("banana fruit recipe");

        let sim_ab = cosine_similarity(&a, &b);
        let sim_ac = cosine_similarity(&a, &c);
        assert!(sim_ab > sim_ac, "sim_ab={} sim_ac={}", sim_ab, sim_ac);
    }

    #[test]
    fn test_rlm_activation() {
        let mut engine = EmbeddingEngine::new();
        assert!(!engine.is_rlm_active());
        assert_eq!(engine.engine_name(), "ruvllm::HashEmbedder");

        // Add corpus entries — RlmEmbedder activates at RLM_MIN_CORPUS (50)
        for i in 0..RLM_MIN_CORPUS {
            let text = format!("document about topic {} in domain {}", i, i % 10);
            let emb = engine.hash_embed(&text);
            engine.add_to_corpus(&format!("doc-{}", i), emb, None);
        }
        assert!(engine.is_rlm_active());
        assert_eq!(engine.engine_name(), "ruvllm::RlmEmbedder");
    }

    #[test]
    fn test_rlm_embed_produces_valid_output() {
        let mut engine = EmbeddingEngine::new();
        // Build corpus large enough to activate RlmEmbedder
        for i in 0..RLM_MIN_CORPUS {
            let text = format!("document about topic {} in domain {}", i, i % 10);
            let emb = engine.hash_embed(&text);
            engine.add_to_corpus(&format!("doc-{}", i), emb, None);
        }
        assert!(engine.is_rlm_active());

        // Query embedding (context-aware)
        let q_emb = engine.embed("rust programming systems");
        assert_eq!(q_emb.len(), EMBED_DIM);
        let norm: f32 = q_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "query norm={}", norm);

        // Storage embedding (corpus-conditioned)
        let s_emb = engine.embed_for_storage("rust programming systems");
        assert_eq!(s_emb.len(), EMBED_DIM);
        let norm: f32 = s_emb.iter().map(|x| x * x).sum::<f32>().sqrt();
        assert!((norm - 1.0).abs() < 0.1, "storage norm={}", norm);
    }

    #[test]
    fn test_rlm_contextual_shift() {
        let mut engine = EmbeddingEngine::new();
        // Build corpus large enough to activate RlmEmbedder
        for i in 0..RLM_MIN_CORPUS {
            let text = format!("document {} covering area {}", i, i % 5);
            let emb = engine.hash_embed(&text);
            engine.add_to_corpus(&format!("doc-{}", i), emb, None);
        }
        assert!(engine.is_rlm_active());

        // RlmEmbedder should produce different embeddings than raw HashEmbedder
        let text = "neural embedding graph knowledge";
        let hash_emb = engine.hash_embed(text);
        let rlm_emb = engine.embed(text);

        // They should differ (contextual shift applied)
        let sim = cosine_similarity(&hash_emb, &rlm_emb);
        assert!(
            sim < 0.999,
            "RlmEmbedder should shift from raw hash; similarity={:.4}",
            sim
        );
        // But still be somewhat similar (same base text)
        assert!(
            sim > 0.1,
            "RlmEmbedder shift too extreme; similarity={:.4}",
            sim
        );
    }

    #[test]
    fn test_prepare_text() {
        let text = EmbeddingEngine::prepare_text(
            "Title",
            "Content here",
            &["tag1".into(), "tag2".into()],
        );
        assert_eq!(text, "Title Content here tag1 tag2");
    }
}
