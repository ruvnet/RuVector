//! RaBitQ: Rotation-Based 1-bit Quantization for Approximate Nearest-Neighbor Search
//!
//! Implements the SIGMOD 2024 algorithm by Jianyang Gao & Cheng Long:
//! "RaBitQ: Quantizing High-Dimensional Vectors with a Theoretical Error Bound
//!  for Approximate Nearest Neighbor Search"
//!
//! ## Algorithm overview
//!
//! 1. Normalize all database vectors to the unit sphere.
//! 2. Apply a random orthogonal rotation P (drawn from the Haar distribution)
//!    so that quantisation error becomes isotropic across dimensions.
//! 3. Store each rotated vector as a single bit per dimension (sign bit → ±1/√D).
//! 4. At query time compute the angular distance estimator:
//!    `est_cos = cos(π · (1 − B/D))` where B = XNOR-popcount of the two binary codes.
//!    `est_sq_dist = ‖q‖² + ‖x‖² − 2·‖q‖·‖x‖·est_cos`
//!
//! The estimator error decreases as O(1/√D) and gives provably good recall on structured data.

pub mod error;
pub mod index;
pub mod quantize;
pub mod rotation;

pub use error::RabitqError;
pub use index::{RabitqIndex, SearchResult};
pub use quantize::{pack_bits, unpack_bits, BinaryCode};
pub use rotation::RandomRotation;
