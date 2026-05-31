#![allow(clippy::needless_range_loop)]

//! PDX: Columnar Vector Layout with Dimension-Pruning Search.
//!
//! Traditional vector stores use **row-major** layout: each vector occupies a
//! contiguous row of D floats.  When computing distances for N vectors, the
//! inner loop jumps between rows → poor cache utilisation and no SIMD.
//!
//! PDX (Kuffo, Krippner, Boncz — SIGMOD 2025, arXiv:2503.04422) flips the
//! layout **within each partition block**: dimension d is stored as a
//! contiguous column of N floats.  The distance loop becomes:
//!
//! ```text
//! for dim d:
//!     col = block.col(d)           // N contiguous f32s → SIMD-friendly
//!     for vec i in 0..N:
//!         partial[i] += (query[d] - col[i])^2
//! ```
//!
//! LLVM auto-vectorises the inner loop with no hand-written intrinsics.
//!
//! ## Dimension pruning (BOND / ADSampling variant)
//!
//! Because partial distances accumulate left-to-right across dimensions,
//! we can exploit a simple lower-bound: once `partial_l2[i] > τ` (where τ is
//! the current k-th nearest distance), vector `i` **cannot** be in the top-k
//! and can be skipped for all remaining dimensions.
//!
//! Using an exponential dimension schedule (8 → 16 → 32 → … → D) we
//! process the first few cheap dimensions, prune obvious losers early, and
//! spend full-dim work only on genuine candidates.
//!
//! ## Backends in this crate
//!
//! | Struct | Layout | Pruning | Notes |
//! |--------|--------|---------|-------|
//! | [`RowMajorIndex`] | row-major | none | baseline |
//! | [`PdxFlatIndex`] | columnar (PDX) | none | shows layout gain alone |
//! | [`PdxPruneIndex`] | columnar (PDX) | exponential lower-bound | full PDX |
//!
//! All three implement [`AnnIndex`].
//!
//! ## Citation
//!
//! ```text
//! @article{kuffo2025pdx,
//!   title  = {PDX: A Data Layout for Vector Similarity Search},
//!   author = {Kuffo, Manuel and Krippner, Till and Boncz, Peter},
//!   journal= {SIGMOD 2025},
//!   year   = {2025},
//!   url    = {https://arxiv.org/abs/2503.04422}
//! }
//! ```

pub mod error;
pub mod index;
pub mod layout;
mod tests;

pub use error::PdxError;
pub use index::{AnnIndex, PdxFlatIndex, PdxPruneIndex, RowMajorIndex, SearchResult};
pub use layout::PdxBlock;
