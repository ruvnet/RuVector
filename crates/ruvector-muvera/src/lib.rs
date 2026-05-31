//! # ruvector-muvera
//!
//! Multi-Vector Retrieval via Fixed Dimensional Encodings (MUVERA).
//!
//! Based on: Karpukhin et al., "MUVERA: Multi-Vector Retrieval via Fixed
//! Dimensional Encodings", NeurIPS 2024, arXiv:2405.19504.
//!
//! ## The problem
//!
//! ColBERT and similar dense retrieval models produce one vector per token
//! (e.g., 32 query tokens × 128D = 4,096 floats per query). Scoring a single
//! document requires computing MaxSim(Q, D): for every query token, find the
//! most similar doc token, then average those maxima. Against a corpus of 1M
//! documents this is **O(|Q|×|D|×n×d)** — intractable without approximation.
//!
//! ## The MUVERA solution
//!
//! Convert each multi-vector set into a single Fixed Dimensional Encoding
//! (FDE) whose **inner product** approximates MaxSim. The conversion:
//!
//! 1. Sample R random unit vectors ("reps") from N(0,I_D) and fix them.
//! 2. For each token vector v: find the rep r* = argmax_r ⟨v,r⟩.
//! 3. Accumulate v into slot r* of the FDE (R×D output vector).
//!
//! Once encoded, every multi-vector doc is a single (R×D)-dim float vector.
//! Standard single-vector MIPS (HNSW, flat scan, IVF) apply directly.
//!
//! ## Variants in this crate
//!
//! | Struct | Complexity | Use when |
//! |--------|------------|----------|
//! | `BruteForceMaxSim` | O(n·|Q|·|D|·d) | Ground truth / small corpus |
//! | `FlatFdeIndex` | O(n·R·D) | Medium corpus, exact FDE |
//! | `HnswFdeIndex` | O(R·D·log n) | Large corpus, approximate |

pub mod encoder;
pub mod error;
pub mod index;

pub use encoder::FdeEncoder;
pub use error::MuveraError;
pub use index::{
    recall_at_k, BruteForceMaxSim, FlatFdeIndex, HnswFdeIndex, MultiVecIndex, SearchResult,
};
