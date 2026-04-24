//! Advanced Features for Ruvector
//!
//! This module provides advanced vector database capabilities:
//! - Enhanced Product Quantization with precomputed lookup tables
//! - Filtered Search with automatic strategy selection
//! - MMR (Maximal Marginal Relevance) for diversity
//! - Hybrid Search combining vector and keyword matching
//! - Conformal Prediction for uncertainty quantification
//! - Multi-Vector Retrieval (ColBERT-style late interaction)
//! - Matryoshka Representation Learning (adaptive-dimension search)
//! - Optimized Product Quantization (OPQ) with learned rotation matrix

pub mod compaction;
pub mod conformal_prediction;
pub mod diskann;
pub mod filtered_search;
pub mod graph_rag;
pub use graph_rag::{
    Community, CommunityDetection, Entity, GraphRAGConfig, GraphRAGPipeline, KnowledgeGraph,
    Relation, RetrievalResult,
};
pub mod hybrid_search;
pub mod matryoshka;
pub mod mmr;
pub mod multi_vector;
pub mod opq;
pub mod product_quantization;
pub mod sparse_vector;

// Re-exports
pub use compaction::{BloomFilter, CompactionConfig, LSMIndex, LSMStats, MemTable, Segment};
pub use conformal_prediction::{
    ConformalConfig, ConformalPredictor, NonconformityMeasure, PredictionSet,
};
pub use diskann::{
    DiskIndex, DiskNode, IOStats, MedoidFinder, PageCache, VamanaConfig, VamanaGraph,
};
pub use filtered_search::{FilterExpression, FilterStrategy, FilteredSearch};
pub use hybrid_search::{HybridConfig, HybridSearch, NormalizationStrategy, BM25};
pub use matryoshka::{FunnelConfig, MatryoshkaConfig, MatryoshkaIndex};
pub use mmr::{MMRConfig, MMRSearch};
pub use multi_vector::{MultiVectorConfig, MultiVectorIndex, ScoringVariant};
pub use opq::{OPQConfig, OPQIndex, RotationMatrix};
pub use product_quantization::{EnhancedPQ, LookupTable, PQConfig};
pub use sparse_vector::{
    fuse_rankings, FusionConfig, FusionStrategy, ScoredDoc, SparseIndex, SparseVector,
};
