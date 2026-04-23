//! # compact-graph-embed
//!
//! Anchor-based compact graph embeddings inspired by NodePiece.
//!
//! Each graph node is represented as a short sequence of `(anchor_id, distance)` tokens.
//! Node embeddings are reconstructed as the weighted mean of token embeddings stored in
//! a compact table — giving 10x+ RAM savings vs dense per-node embedding matrices.
//!
//! ## Quick Start
//!
//! ```rust
//! use compact_graph_embed::{
//!     CsrGraph, AnchorTokenizer, EmbeddingTableF32, MeanEmbedder, NodeEmbedder, NodeTokenizer,
//! };
//!
//! // Build a small graph
//! let edges = vec![(0,1),(1,2),(2,3),(3,0),(0,2)];
//! let graph = CsrGraph::new(4, &edges);
//!
//! // Tokenize with 2 anchors, max BFS distance 2
//! let tokenizer = AnchorTokenizer::new(&graph, 2, 2, 42);
//!
//! // Random f32 embedding table
//! let storage = EmbeddingTableF32::new_random(
//!     tokenizer.num_anchors(), tokenizer.max_dist(), 16, 99
//! );
//!
//! // Compose into an embedder
//! let embedder = MeanEmbedder::new(tokenizer, storage);
//!
//! // Embed node 0
//! let vec = embedder.embed(0).unwrap();
//! assert_eq!(vec.len(), 16);
//! ```

pub mod anchor;
pub mod embedder;
pub mod error;
pub mod graph;
pub mod storage;
pub mod traits;

pub use anchor::AnchorTokenizer;
pub use embedder::{GenericMeanEmbedder, MeanEmbedder};
pub use error::EmbedError;
pub use graph::CsrGraph;
pub use storage::{EmbeddingTableF32, EmbeddingTableI8};
pub use traits::{NodeEmbedder, NodeTokenizer, Token, TokenStorage};
