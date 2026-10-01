//! Progressive HNSW indexing for the RuVector Format (RVF).
//!
//! This crate implements the three-layer progressive indexing model:
//!
//! - **Layer A**: Entry points + coarse routing (< 5ms load, ~0.70 recall)
//! - **Layer B**: Partial adjacency for hot region (100ms-1s load, ~0.85 recall)
//! - **Layer C**: Full HNSW adjacency (seconds load, >= 0.95 recall)

#![cfg_attr(not(feature = "std"), no_std)]

extern crate alloc;

#[inline]
pub(crate) fn sqrt_f32(value: f32) -> f32 {
    #[cfg(feature = "std")]
    {
        value.sqrt()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::sqrtf(value)
    }
}

#[inline]
pub(crate) fn ln_f64(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.ln()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::log(value)
    }
}

#[inline]
pub(crate) fn floor_f64(value: f64) -> f64 {
    #[cfg(feature = "std")]
    {
        value.floor()
    }
    #[cfg(not(feature = "std"))]
    {
        libm::floor(value)
    }
}

pub mod builder;
pub mod codec;
pub mod distance;
pub mod hnsw;
pub mod layers;
pub mod progressive;
pub mod traits;

pub use builder::{build_full_index, build_layer_a, build_layer_b, build_layer_c};
pub use codec::{decode_index_seg, encode_index_seg, CodecError, IndexSegData, IndexSegHeader};
pub use distance::{cosine_distance, dot_product, l2_distance};
pub use hnsw::{HnswConfig, HnswGraph, HnswLayer};
pub use layers::{IndexLayer, IndexState, LayerA, LayerB, LayerC, PartitionEntry};
pub use progressive::ProgressiveIndex;
pub use traits::VectorStore;

#[cfg(feature = "std")]
pub use traits::InMemoryVectorStore;
