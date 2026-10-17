//! # streamcloud-tensor
//!
//! Model configuration and **safetensors weight loading** for StreamCloud (the
//! Rust port of LingBot-Map).
//!
//! The upstream checkpoint (`robbyant/lingbot-map`, ~4.63 GB, Apache-2.0) ships
//! as safetensors. This crate provides:
//!
//! * [`ModelConfig`] — the Geometric Context Transformer hyperparameters,
//!   loadable from the upstream `config.json`.
//! * [`WeightIndex`] — a header-only reader that maps tensor names to shapes /
//!   dtypes **without** loading the multi-GB payload, so we can validate a
//!   checkpoint and plan loading on machines that cannot hold it in RAM.
//!
//! Actual tensor materialization into `candle_core::Tensor` lives behind the
//! `candle` feature (see [`load`]).

use std::collections::BTreeMap;

pub mod config;
pub use config::ModelConfig;

#[cfg(feature = "candle")]
pub mod load;

/// Errors from config / weight handling.
#[derive(Debug, thiserror::Error)]
pub enum TensorError {
    /// I/O error reading a file.
    #[error("io error: {0}")]
    Io(#[from] std::io::Error),
    /// JSON (config) parse error.
    #[error("config parse error: {0}")]
    Config(#[from] serde_json::Error),
    /// safetensors parse error.
    #[error("safetensors error: {0}")]
    SafeTensors(String),
}

/// Metadata for a single tensor in a safetensors file (no payload).
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct TensorInfo {
    /// Element dtype as reported by safetensors (e.g. `F32`, `BF16`).
    pub dtype: String,
    /// Tensor shape.
    pub shape: Vec<usize>,
}

impl TensorInfo {
    /// Number of elements.
    pub fn numel(&self) -> usize {
        self.shape.iter().product()
    }
}

/// Header-only view of a safetensors checkpoint: tensor name -> [`TensorInfo`].
#[derive(Debug, Clone, Default)]
pub struct WeightIndex {
    /// Tensor name -> metadata, ordered for stable iteration.
    pub tensors: BTreeMap<String, TensorInfo>,
}

impl WeightIndex {
    /// Parse a safetensors file's header without reading tensor data.
    pub fn from_file(path: impl AsRef<std::path::Path>) -> Result<Self, TensorError> {
        let bytes = std::fs::read(path)?;
        Self::from_bytes(&bytes)
    }

    /// Parse a safetensors header from an in-memory buffer.
    pub fn from_bytes(bytes: &[u8]) -> Result<Self, TensorError> {
        let st = safetensors::SafeTensors::deserialize(bytes)
            .map_err(|e| TensorError::SafeTensors(e.to_string()))?;
        let mut tensors = BTreeMap::new();
        for (name, view) in st.tensors() {
            tensors.insert(
                name.to_string(),
                TensorInfo {
                    dtype: format!("{:?}", view.dtype()),
                    shape: view.shape().to_vec(),
                },
            );
        }
        Ok(Self { tensors })
    }

    /// Total parameter count across all tensors.
    pub fn total_params(&self) -> usize {
        self.tensors.values().map(TensorInfo::numel).sum()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn tensorinfo_numel() {
        let t = TensorInfo {
            dtype: "F32".into(),
            shape: vec![2, 3, 4],
        };
        assert_eq!(t.numel(), 24);
    }

    #[test]
    fn empty_index_has_zero_params() {
        assert_eq!(WeightIndex::default().total_params(), 0);
    }
}
