//! # ruvector-turboquant
//!
//! **Turbo4**: a 4-bit Lloyd-Max quantized vector *datatype* (ADR-296) —
//! Qdrant-style primary-storage quantization, not a search-time cache:
//!
//! * deterministic randomized Hadamard rotation (sign ⊙ permute ⊙ block-FWHT
//!   rounds, seeded SplitMix64 — bit-stable across platforms and versions,
//!   no `rand` dependency);
//! * precomputed 16-level Lloyd-Max tables for the rotated (≈ Gaussian)
//!   coordinates — no training pass, online ingest;
//! * packed nibble codes: `D/2 + 8` bytes per vector (≈ 7.9× vs f32 at
//!   1536-D) — **the original float vector is never stored**;
//! * direct scoring on packed codes: symmetric (code×code, for graph
//!   construction), asymmetric (int8 query×code, for traversal), and exact
//!   f32 rescoring — with runtime-dispatched AVX2 kernels and a scalar
//!   oracle they are tested bit-exact against.
//!
//! ```
//! use ruvector_turboquant::{Metric, Turbo4Codec, score};
//!
//! let dim = 128;
//! let codec = Turbo4Codec::new(dim, 42).unwrap();
//! let a: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.37).sin()).collect();
//! let b: Vec<f32> = (0..dim).map(|i| (i as f32 * 0.11).cos()).collect();
//!
//! let code_a = codec.encode(&a).unwrap();   // 64 + 8 bytes, floats discarded
//! let code_b = codec.encode(&b).unwrap();
//! let query = codec.encode_query(&a).unwrap();
//!
//! let d_sym = score::symmetric_distance(Metric::Euclidean, &code_a, &code_b, dim);
//! let d_asym = score::asymmetric_distance(Metric::Euclidean, &query.blob, &code_b, dim);
//! let d_exact = score::rescore(Metric::Euclidean, &query, &code_b, dim);
//! assert!((d_asym - d_sym).abs() < 0.15 * d_sym.max(1.0));
//! assert!((d_exact - d_asym).abs() < 0.1 * d_asym.max(1.0));
//! ```

pub mod bits1;
pub mod codec;
pub mod rotation;
pub mod score;
pub mod simd;
pub mod tables;

pub use bits1::{encode_bits, Bits1Query};
pub use codec::{Turbo4Codec, Turbo4Query, META_BYTES};
pub use rotation::Rotation;
pub use score::{asymmetric_distance, rescore, symmetric_distance, Metric};

/// Errors from the Turbo4 codec.
#[derive(Debug, thiserror::Error)]
pub enum TurboQuantError {
    /// Dimension must be even and ≥ 2 (the two-run nibble layout splits D in half).
    #[error("Turbo4 requires an even dimension >= 2, got {0}")]
    InvalidDimension(usize),
    /// Input vector length differs from the codec dimension.
    #[error("dimension mismatch: codec expects {expected}, got {actual}")]
    DimensionMismatch { expected: usize, actual: usize },
}
