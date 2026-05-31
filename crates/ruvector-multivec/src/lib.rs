//! Multi-vector late-interaction search for ruvector.
//!
//! Motivated by: *"MUVERA: Multi-Vector Retrieval via Fixed Dimensional Encodings"*,
//! Karpukhin et al., NeurIPS 2024, arXiv:2405.19504.
//!
//! ## The Problem
//!
//! ColBERT-style retrieval represents each document as T token embeddings.
//! Scoring a single query against n documents with T tokens each at dimension D
//! costs O(n × T_q × T_d × D) — unusable at scale. The existing
//! `ruvector-core::advanced_features::multi_vector::MultiVectorIndex` is a
//! correct brute-force implementation; at 100 K documents × 32 tokens it is
//! ~25× slower than single-vector HNSW.
//!
//! ## MUVERA's Solution — Fixed Dimensional Encoding (FDE)
//!
//! FDE converts a variable-length set of token vectors into a single dense
//! vector of fixed dimension `R × K × D` by:
//!
//! 1. Sample R sets of K random unit vectors (hyperplanes) from a seeded RNG.
//! 2. For each token, assign it to the closest hyperplane (soft argmax) within
//!    each repetition.
//! 3. Sum-aggregate all token vectors that fall in the same bucket.
//!
//! The resulting flat vector approximates the Chamfer / MaxSim score when dotted
//! with a query FDE vector. Standard ANN (HNSW) then applies directly.
//!
//! ## Crate Contents
//!
//! | Module | Contents |
//! |--------|----------|
//! | `scoring` | `maxsim_exact`, `chamfer_score`, `centroid_dot`, `FdeEncoder` |
//! | `index`   | `MultiVecIndex` trait + `CentroidIndex`, `MaxSimIndex`, `MuveraFdeIndex` |
//! | `error`   | `MultivecError` |
//!
//! ## Variants
//!
//! | Variant | Score | Mem/doc | QPS (n=10K, T=32, D=128) | Recall@10 |
//! |---------|-------|---------|--------------------------|-----------|
//! | `CentroidIndex` | centroid dot | 1×D×4B | highest | lowest |
//! | `MaxSimIndex (MaxSim)` | exact ColBERT | T×D×4B | baseline | 100% (oracle) |
//! | `MaxSimIndex (Chamfer)` | Chamfer | T×D×4B | ~same as MaxSim | ~oracle |
//! | `MuveraFdeIndex` | FDE approx | R×K×D×4B | 3-8× faster | ~95% |
//!
//! (Exact numbers from `cargo run --release -p ruvector-multivec`.)

pub mod error;
pub mod index;
pub mod scoring;

pub use error::MultivecError;
pub use index::{
    CentroidIndex, MaxSimIndex, MultiVecIndex, MuveraFdeIndex, MuveraFdeRerankIndex, SearchResult,
};
pub use scoring::{centroid_dot, chamfer_score, dot, l2_normalize, maxsim_exact, FdeEncoder};
