//! Burn-based GAT implementation with autodiff support.
//!
//! Phase 1: ✅ Skeleton complete. Phase 2 onwards: implement `todo!()` markers.

pub mod gat_encoder;
pub mod gat_layer;
pub mod gradcheck;
pub mod tensor_conversion;

pub use gat_encoder::GATEncoderBurn;
pub use gat_layer::{GATLayerBurn, GATLayerBurnConfig};
