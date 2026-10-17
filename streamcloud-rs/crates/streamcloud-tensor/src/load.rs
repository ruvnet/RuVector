//! candle weight materialization (feature `candle`).
//!
//! Loads safetensors weights into `candle_core::Tensor`s. Use
//! [`crate::WeightIndex`] first to validate a checkpoint's inventory without
//! paying the full RAM cost; use these helpers once you intend to materialize.

use crate::TensorError;
use candle_core::{Device, Tensor};
use std::collections::HashMap;
use std::path::Path;

/// Load every tensor from a safetensors file onto `device`.
pub fn load_safetensors(
    path: impl AsRef<Path>,
    device: &Device,
) -> Result<HashMap<String, Tensor>, TensorError> {
    candle_core::safetensors::load(path, device)
        .map_err(|e| TensorError::SafeTensors(e.to_string()))
}

/// Load a single named tensor from a safetensors file onto `device`.
pub fn load_tensor(
    path: impl AsRef<Path>,
    name: &str,
    device: &Device,
) -> Result<Tensor, TensorError> {
    let mut map = load_safetensors(path, device)?;
    map.remove(name)
        .ok_or_else(|| TensorError::SafeTensors(format!("tensor not found: {name}")))
}
