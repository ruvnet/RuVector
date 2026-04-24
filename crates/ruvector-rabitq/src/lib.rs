#![allow(clippy::manual_div_ceil)]
#![allow(clippy::needless_range_loop)]
#![allow(clippy::doc_overindented_list_items)]

//! RaBitQ: Rotation-Based 1-bit Quantization for Approximate Nearest-Neighbor Search
//!
//! Motivated by the SIGMOD 2024 algorithm by Jianyang Gao & Cheng Long:
//! *"RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//! for Approximate Nearest Neighbor Search"*. This crate ships two estimators:
//!
//! 1. **Symmetric (Charikar-style)** — both query and database are 1-bit.
//!    `est_cos = cos(π · (1 − B/D))` where B = padding-safe XNOR-popcount.
//!    Cheapest per candidate (O(D/64) popcount + 1 cos) — use when memory
//!    dominates throughput.
//!
//! 2. **Asymmetric (RaBitQ-2024-style)** — query is f32, database is 1-bit.
//!    `est_ip = ‖q‖ · ‖x‖ · (1/√D) · Σᵢ sign(x_rot,i) · q_rot,i`. Unbiased on
//!    Haar-uniform rotations; tighter variance than (1). O(D) per candidate.
//!
//! ## Indexes
//!
//! | Variant | Storage | Estimator | Rerank |
//! |---|---|---|---|
//! | `FlatF32Index` | f32 originals | exact L2 | N/A |
//! | `RabitqIndex` | rotation + codes | symmetric | no |
//! | `RabitqPlusIndex` | rotation + codes + originals | symmetric + exact rerank | yes |
//! | `RabitqAsymIndex` | rotation + codes [+ originals] | asymmetric + optional rerank | opt |
//!
//! All satisfy [`index::AnnIndex`]. Search is top-k via a bounded max-heap
//! (O(n log k)), and scoring uses `f32::total_cmp` so NaN never panics.
//!
//! ## Guarantees
//!
//! - Padding-safe popcount at any D (handles `D % 64 != 0` correctly).
//! - Deterministic: `(dim, seed, data)` triple → bit-identical rotation +
//!   index build + search output across runs.
//! - No `unsafe`, no external BLAS/LAPACK, no C/C++ deps.
//!
//! ## Benchmarks
//!
//! See `benches/rabitq_bench.rs` for distance-kernel micro-benchmarks and
//! `src/main.rs` (`rabitq-demo`) for end-to-end recall + throughput across
//! n ∈ {1 k, 5 k, 50 k, 100 k}.

pub mod error;
pub mod index;
pub mod kernel;
pub mod quantize;
pub mod rotation;

pub use error::RabitqError;
pub use index::{
    AnnIndex, FlatF32Index, RabitqAsymIndex, RabitqIndex, RabitqPlusIndex, SearchResult,
};
pub use kernel::{CpuKernel, KernelCaps, ScanRequest, ScanResponse, VectorKernel};
pub use quantize::{pack_bits, unpack_bits, BinaryCode};
pub use rotation::RandomRotation;
