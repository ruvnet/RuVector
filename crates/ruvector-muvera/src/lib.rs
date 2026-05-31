//! `ruvector-muvera` — Fixed Dimensional Encodings for scalable multi-vector retrieval.
//!
//! Implements the MUVERA algorithm (arXiv:2405.19504, NeurIPS 2024, Google Research).
//! Converts ColBERT-style multi-token embedding sets into fixed-dimension single vectors
//! that approximate Chamfer / MaxSim similarity, enabling standard HNSW or IVF indexing
//! for multi-vector workloads with formally bounded approximation error.
//!
//! ## Quick start
//!
//! ```rust
//! use ruvector_muvera::{FdeConfig, FdeEncoder, MuveraIndex};
//! use rand::SeedableRng;
//! use rand::rngs::StdRng;
//!
//! let cfg = FdeConfig { dim: 8, buckets: 8, d_proj: 4, reps: 2 };
//! let mut rng = StdRng::seed_from_u64(42);
//! let encoder = FdeEncoder::new(cfg, &mut rng).unwrap();
//! let mut index = MuveraIndex::new(encoder);
//!
//! let doc_tokens = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
//! index.insert("doc1".into(), &doc_tokens).unwrap();
//!
//! let query_tokens = vec![vec![1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]];
//! let results = index.search(&query_tokens, 1).unwrap();
//! assert_eq!(results[0].id, "doc1");
//! ```

pub mod encoder;
pub mod error;
pub mod index;

pub use encoder::{FdeConfig, FdeEncoder};
pub use error::MuveraError;
pub use index::{FlatBackend, MuveraIndex, MuveraResult, VectorBackend};
