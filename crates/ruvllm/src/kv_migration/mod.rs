//! Cross-Model KV-Cache Migration (PIR WP13, ADR-314)
//!
//! Implements the first shippable slice of cross-model KV-cache migration
//! per arXiv:2608.03893, "Cross-Model KV Cache Transfer in LLM Families:
//! A Closed-Form Linear Mapping for Prefill Reuse":
//!
//! - [`mapper::LinearKvMapper`]: a closed-form least-squares linear mapping
//!   (per-layer `W_k`, `W_v`) fitted from paired calibration activations of
//!   the source and target models, applied via matmul.
//! - [`quality::TransferQualityGate`]: a transfer-quality gate that predicts
//!   migration quality on a held-out validation set **before** any migration
//!   happens. The paper found 2 of 6 tested model pairs degrade sharply under
//!   the linear mapper, so we NEVER migrate blind — the gate can veto a
//!   migration into a full re-prefill.
//! - [`MlpKvMapper`]: a documented stub for the paper's nonlinear MLP
//!   fallback (recovers up to +37pp HellaSwag on the degrading pairs).
//!   The [`mapper::KvMapper`] trait already accommodates it; the
//!   implementation is deliberately out of scope for this slice.
//!
//! Feature-gated behind `kv-migration` (default off) so it cannot affect
//! existing builds.
//!
//! ## Data model
//!
//! A cache snapshot is a per-layer list of `[tokens, dim]` K/V matrices where
//! `dim = num_kv_heads * head_dim`, i.e. the flattened per-token KV vector.
//! This matches the flat layout produced by
//! `TwoTierKvCache::get_all_kv()` (stride = `num_kv_heads * head_dim`);
//! use [`LayerKvTensor::from_flat`] to adapt it.

pub mod mapper;
pub mod quality;

#[cfg(test)]
mod tests;

pub use mapper::{
    CalibrationLayerPair, KvMapper, LinearKvMapper, DEFAULT_RIDGE_LAMBDA, MAX_CALIBRATION_LAYERS,
    MAX_CALIBRATION_TOKENS, MAX_FIT_DIM,
};
pub use quality::{
    GateDecision, LayerQuality, QualityGateConfig, QualityReport, TransferQualityGate,
};

use crate::error::{Result, RuvLLMError};
use ndarray::Array2;

/// A single layer's K/V tensors in `[tokens, dim]` layout, where
/// `dim = num_kv_heads * head_dim` (the flattened per-token KV vector).
#[derive(Debug, Clone)]
pub struct LayerKvTensor {
    /// Key matrix, shape `[tokens, dim]`.
    pub keys: Array2<f32>,
    /// Value matrix, shape `[tokens, dim]`.
    pub values: Array2<f32>,
}

impl LayerKvTensor {
    /// Create from key/value matrices, validating that shapes match.
    pub fn new(keys: Array2<f32>, values: Array2<f32>) -> Result<Self> {
        if keys.shape() != values.shape() {
            return Err(RuvLLMError::KvCache(format!(
                "key shape {:?} != value shape {:?}",
                keys.shape(),
                values.shape()
            )));
        }
        Ok(Self { keys, values })
    }

    /// Build from the flat `(keys, values)` layout returned by
    /// `TwoTierKvCache::get_all_kv()`, where each token occupies a
    /// contiguous stride of `num_kv_heads * head_dim` floats.
    pub fn from_flat(
        keys: &[f32],
        values: &[f32],
        num_kv_heads: usize,
        head_dim: usize,
    ) -> Result<Self> {
        let dim = num_kv_heads.checked_mul(head_dim).ok_or_else(|| {
            RuvLLMError::KvCache(format!(
                "KV dimension overflow: num_kv_heads ({num_kv_heads}) * head_dim ({head_dim}) \
                 exceeds usize"
            ))
        })?;
        if dim == 0 {
            return Err(RuvLLMError::KvCache("zero KV dimension".to_string()));
        }
        if keys.len() != values.len() || keys.len() % dim != 0 {
            return Err(RuvLLMError::KvCache(format!(
                "flat KV length mismatch: keys={}, values={}, stride={}",
                keys.len(),
                values.len(),
                dim
            )));
        }
        let tokens = keys.len() / dim;
        let keys = Array2::from_shape_vec((tokens, dim), keys.to_vec())
            .map_err(|e| RuvLLMError::KvCache(e.to_string()))?;
        let values = Array2::from_shape_vec((tokens, dim), values.to_vec())
            .map_err(|e| RuvLLMError::KvCache(e.to_string()))?;
        Self::new(keys, values)
    }

    /// Number of tokens in this layer tensor.
    pub fn tokens(&self) -> usize {
        self.keys.nrows()
    }

    /// Flattened per-token KV dimension (`num_kv_heads * head_dim`).
    pub fn dim(&self) -> usize {
        self.keys.ncols()
    }
}

/// A full multi-layer KV-cache snapshot for one sequence.
#[derive(Debug, Clone, Default)]
pub struct KvCacheSnapshot {
    /// Per-layer K/V tensors, ordered by layer index.
    pub layers: Vec<LayerKvTensor>,
}

impl KvCacheSnapshot {
    /// Create a snapshot from per-layer tensors.
    pub fn new(layers: Vec<LayerKvTensor>) -> Self {
        Self { layers }
    }

    /// Number of layers.
    pub fn num_layers(&self) -> usize {
        self.layers.len()
    }
}

/// Outcome of a gated migration attempt.
#[derive(Debug, Clone)]
pub enum MigrationResult {
    /// The gate approved: the cache was mapped to the target geometry.
    Migrated {
        /// The migrated cache in target-model geometry.
        cache: KvCacheSnapshot,
        /// The quality report that justified the migration.
        report: QualityReport,
    },
    /// The gate vetoed: the caller must run a full re-prefill on the
    /// target model instead of reusing the source cache.
    Reprefill {
        /// The quality report explaining the veto.
        report: QualityReport,
    },
}

/// Migrate a source-model KV cache into target-model geometry, gated by a
/// predicted transfer-quality check — never migrate blind.
///
/// The gate assesses the fitted `mapper` against its held-out validation
/// set first. Only if the gate's decision is [`GateDecision::Migrate`] is
/// the source cache actually mapped; otherwise the caller gets
/// [`MigrationResult::Reprefill`] and must prefill from scratch.
pub fn migrate_kv_cache(
    source_cache: &KvCacheSnapshot,
    mapper: &dyn KvMapper,
    gate: &TransferQualityGate,
) -> Result<MigrationResult> {
    if source_cache.num_layers() != mapper.num_layers() {
        return Err(RuvLLMError::KvCache(format!(
            "cache has {} layers but mapper was fitted for {}",
            source_cache.num_layers(),
            mapper.num_layers()
        )));
    }

    let report = gate.assess(mapper)?;
    if report.decision == GateDecision::Reprefill {
        return Ok(MigrationResult::Reprefill { report });
    }

    let mut layers = Vec::with_capacity(source_cache.num_layers());
    for (idx, layer) in source_cache.layers.iter().enumerate() {
        let (keys, values) = mapper.map_layer(idx, &layer.keys, &layer.values)?;
        layers.push(LayerKvTensor::new(keys, values)?);
    }

    Ok(MigrationResult::Migrated {
        cache: KvCacheSnapshot::new(layers),
        report,
    })
}

/// Stub for the paper's nonlinear MLP fallback mapper (arXiv:2608.03893 §5).
///
/// The paper found that on the two model pairs where the closed-form linear
/// mapper degrades sharply, a small per-layer MLP recovers up to +37pp
/// HellaSwag accuracy. The [`KvMapper`] trait already accommodates this
/// variant; this slice deliberately ships only the linear mapper, so every
/// method of this type returns [`RuvLLMError::NotImplemented`]. Tracked as
/// follow-up work under PIR WP13 / ADR-314.
#[derive(Debug, Clone, Copy, Default)]
pub struct MlpKvMapper;

impl KvMapper for MlpKvMapper {
    fn num_layers(&self) -> usize {
        0
    }

    fn source_dim(&self) -> usize {
        0
    }

    fn target_dim(&self) -> usize {
        0
    }

    fn map_layer(
        &self,
        _layer: usize,
        _keys: &Array2<f32>,
        _values: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        Err(RuvLLMError::NotImplemented(
            "MlpKvMapper: nonlinear MLP fallback is a documented stub in this slice (ADR-314)"
                .to_string(),
        ))
    }
}
