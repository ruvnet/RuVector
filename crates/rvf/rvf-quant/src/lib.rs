//! Temperature-tiered vector quantization for the RuVector Format (RVF).
//!
//! Provides three quantization levels mapped to temperature tiers:
//!
//! | Tier | Quantization | Compression |
//! |------|-------------|-------------|
//! | Hot  | Scalar (int8) | 4x |
//! | Warm | Product (PQ)  | 8-16x |
//! | Cold | Binary (1-bit)| 32x |
//!
//! The [`rabitq`] module provides a RaBitQ-style 1-bit codec (centroid
//! centering + seeded random rotation + correction scalars) with an
//! asymmetric distance estimator, intended for two-stage
//! (scan-then-rescore) search at ~32x code compression.
//!
//! A Count-Min Sketch tracks per-block access frequency to drive
//! promotion/demotion decisions.

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

pub mod binary;
pub mod codec;
pub mod product;
pub mod rabitq;
pub mod scalar;
pub mod sketch;
pub mod tier;
pub mod traits;

pub use binary::{decode_binary, encode_binary, hamming_distance};
pub use product::ProductQuantizer;
pub use rabitq::{RabitqCode, RabitqQuantizer, RabitqQuery};
pub use scalar::ScalarQuantizer;
pub use sketch::CountMinSketch;
pub use tier::TemperatureTier;
pub use traits::Quantizer;
