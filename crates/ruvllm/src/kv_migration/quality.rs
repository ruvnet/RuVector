//! Transfer-quality gate: predict migration quality BEFORE migrating.
//!
//! arXiv:2608.03893 found that 2 of 6 tested model pairs degrade sharply
//! under the closed-form linear mapper. This module implements the routing
//! gate ADR-314 mandates: given a fitted mapper and a small held-out
//! validation set of paired activations, compute a predicted-quality score
//! (per-layer reconstruction cosine similarity and relative Frobenius
//! error) and decide [`GateDecision::Migrate`] vs [`GateDecision::Reprefill`].
//!
//! The decision is deliberately conservative: it gates on the *worst* layer,
//! not the mean, because a single badly-mapped layer is enough to corrupt
//! downstream attention. A vetoed migration costs only the re-prefill we
//! would have paid anyway; an approved bad migration silently degrades
//! generation quality.

use crate::error::{Result, RuvLLMError};
use ndarray::Array2;

use super::mapper::{CalibrationLayerPair, KvMapper};

/// Squared-Frobenius-norm floor below which an activation matrix is treated
/// as degenerate (all-zero or vanishingly small).
///
/// A degenerate operand carries no directional information, so a cosine or
/// relative-error score computed from it is meaningless. Crucially, an
/// all-zero validation pair must NEVER score as a perfect pass: since
/// `pred = source · W` is zero whenever the source is zero, a zeroed
/// validation set would otherwise approve *any* mapper unconditionally —
/// defeating the gate's "never migrate blind" contract.
const DEGENERATE_NORM_EPS: f64 = 1e-12;

/// Gate decision: migrate the cache, or veto into a full re-prefill.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GateDecision {
    /// Predicted quality clears the thresholds — safe to migrate.
    Migrate,
    /// Predicted quality is below threshold — re-prefill on the target
    /// model instead of reusing the mapped cache.
    Reprefill,
}

/// Thresholds for the transfer-quality gate.
///
/// Defaults are conservative (see module docs): every layer's key AND value
/// reconstruction must clear both thresholds or the gate vetoes.
#[derive(Debug, Clone, Copy)]
pub struct QualityGateConfig {
    /// Minimum acceptable per-layer reconstruction cosine similarity
    /// (worst layer, min of keys/values). Default: `0.92`.
    pub min_cosine: f32,
    /// Maximum acceptable per-layer relative Frobenius error
    /// `‖pred − target‖_F / ‖target‖_F` (worst layer, max of keys/values).
    /// Default: `0.25`.
    pub max_relative_error: f32,
}

impl Default for QualityGateConfig {
    fn default() -> Self {
        Self {
            min_cosine: 0.92,
            max_relative_error: 0.25,
        }
    }
}

/// Reconstruction quality for one layer of the validation set.
#[derive(Debug, Clone, Copy)]
pub struct LayerQuality {
    /// Layer index.
    pub layer: usize,
    /// Cosine similarity between mapped and true target keys.
    pub key_cosine: f32,
    /// Cosine similarity between mapped and true target values.
    pub value_cosine: f32,
    /// Relative Frobenius error for keys.
    pub key_relative_error: f32,
    /// Relative Frobenius error for values.
    pub value_relative_error: f32,
}

impl LayerQuality {
    /// Worst (lowest) cosine across keys and values.
    pub fn worst_cosine(&self) -> f32 {
        self.key_cosine.min(self.value_cosine)
    }

    /// Worst (highest) relative error across keys and values.
    pub fn worst_relative_error(&self) -> f32 {
        self.key_relative_error.max(self.value_relative_error)
    }
}

/// Aggregated predicted-quality report produced by the gate.
#[derive(Debug, Clone)]
pub struct QualityReport {
    /// Per-layer reconstruction quality.
    pub layers: Vec<LayerQuality>,
    /// Mean of per-layer worst cosines.
    pub mean_cosine: f32,
    /// Minimum per-layer worst cosine (the layer that decides the veto).
    pub worst_cosine: f32,
    /// Mean of per-layer worst relative errors.
    pub mean_relative_error: f32,
    /// Maximum per-layer worst relative error.
    pub worst_relative_error: f32,
    /// The gate's decision under its configured thresholds.
    pub decision: GateDecision,
}

/// Transfer-quality gate: owns a held-out validation set of paired
/// activations plus thresholds, and assesses a fitted mapper against them.
///
/// The validation set must be disjoint from the calibration set the mapper
/// was fitted on, or the score is an overfit self-assessment.
#[derive(Debug, Clone)]
pub struct TransferQualityGate {
    config: QualityGateConfig,
    validation: Vec<CalibrationLayerPair>,
}

impl TransferQualityGate {
    /// Create a gate from thresholds and a held-out validation set
    /// (one [`CalibrationLayerPair`] per layer, in layer order).
    ///
    /// Rejects a validation set in which the majority of pairs (including
    /// all of them) are degenerate — i.e. have near-zero-norm activations on
    /// either side. Such a set cannot meaningfully assess a mapper, and an
    /// all-zero set would otherwise approve any mapper unconditionally.
    /// Individual degenerate pairs that slip past this check still score as
    /// hard failures in [`Self::assess`], never as passes.
    pub fn new(config: QualityGateConfig, validation: Vec<CalibrationLayerPair>) -> Result<Self> {
        if validation.is_empty() {
            return Err(RuvLLMError::KvCache(
                "TransferQualityGate requires a non-empty validation set".to_string(),
            ));
        }
        let degenerate = validation.iter().filter(|p| pair_is_degenerate(p)).count();
        if degenerate * 2 > validation.len() {
            return Err(RuvLLMError::KvCache(format!(
                "validation set is degenerate: {degenerate} of {} pairs have \
                 near-zero-norm activations and cannot assess a mapper",
                validation.len()
            )));
        }
        Ok(Self { config, validation })
    }

    /// Create a gate with the conservative default thresholds.
    pub fn with_defaults(validation: Vec<CalibrationLayerPair>) -> Result<Self> {
        Self::new(QualityGateConfig::default(), validation)
    }

    /// The gate's threshold configuration.
    pub fn config(&self) -> &QualityGateConfig {
        &self.config
    }

    /// Assess a fitted mapper against the held-out validation set and
    /// return the predicted-quality report with a migrate/re-prefill
    /// decision.
    pub fn assess(&self, mapper: &dyn KvMapper) -> Result<QualityReport> {
        if mapper.num_layers() != self.validation.len() {
            return Err(RuvLLMError::KvCache(format!(
                "mapper has {} layers but validation set has {}",
                mapper.num_layers(),
                self.validation.len()
            )));
        }
        // Real runtime dimension validation (NOT a debug_assert): the
        // comparison helpers iterate `zip`, which silently truncates to the
        // shorter operand in release builds — a mismatched validation set
        // would be scored on a tiny prefix of the data. Reject it instead.
        for (idx, pair) in self.validation.iter().enumerate() {
            if pair.source.dim() != mapper.source_dim() {
                return Err(RuvLLMError::KvCache(format!(
                    "validation pair {idx}: source dim {} does not match mapper source dim {}",
                    pair.source.dim(),
                    mapper.source_dim()
                )));
            }
            if pair.target.dim() != mapper.target_dim() {
                return Err(RuvLLMError::KvCache(format!(
                    "validation pair {idx}: target dim {} does not match mapper target dim {}",
                    pair.target.dim(),
                    mapper.target_dim()
                )));
            }
        }

        let mut layers = Vec::with_capacity(self.validation.len());
        for (idx, pair) in self.validation.iter().enumerate() {
            let (pred_k, pred_v) = mapper.map_layer(idx, &pair.source.keys, &pair.source.values)?;
            layers.push(LayerQuality {
                layer: idx,
                key_cosine: cosine_similarity(&pred_k, &pair.target.keys),
                value_cosine: cosine_similarity(&pred_v, &pair.target.values),
                key_relative_error: relative_error(&pred_k, &pair.target.keys),
                value_relative_error: relative_error(&pred_v, &pair.target.values),
            });
        }

        let n = layers.len() as f32;
        let mean_cosine = layers.iter().map(LayerQuality::worst_cosine).sum::<f32>() / n;
        let worst_cosine = layers
            .iter()
            .map(LayerQuality::worst_cosine)
            .fold(f32::INFINITY, f32::min);
        let mean_relative_error = layers
            .iter()
            .map(LayerQuality::worst_relative_error)
            .sum::<f32>()
            / n;
        let worst_relative_error = layers
            .iter()
            .map(LayerQuality::worst_relative_error)
            .fold(f32::NEG_INFINITY, f32::max);

        // Conservative rule: every layer must clear both thresholds, and any
        // non-finite score (NaN/Inf from degenerate inputs) is a veto.
        let clean = worst_cosine.is_finite() && worst_relative_error.is_finite();
        let decision = if clean
            && worst_cosine >= self.config.min_cosine
            && worst_relative_error <= self.config.max_relative_error
        {
            GateDecision::Migrate
        } else {
            GateDecision::Reprefill
        };

        Ok(QualityReport {
            layers,
            mean_cosine,
            worst_cosine,
            mean_relative_error,
            worst_relative_error,
            decision,
        })
    }
}

/// Squared Frobenius norm of a matrix, accumulated in f64.
fn squared_norm(m: &Array2<f32>) -> f64 {
    m.iter().map(|v| (*v as f64) * (*v as f64)).sum()
}

/// True when any side of a validation pair has a near-zero-norm activation
/// matrix, i.e. carries no signal a mapper could be assessed against.
fn pair_is_degenerate(pair: &CalibrationLayerPair) -> bool {
    squared_norm(&pair.source.keys) <= DEGENERATE_NORM_EPS
        || squared_norm(&pair.source.values) <= DEGENERATE_NORM_EPS
        || squared_norm(&pair.target.keys) <= DEGENERATE_NORM_EPS
        || squared_norm(&pair.target.values) <= DEGENERATE_NORM_EPS
}

/// Flattened cosine similarity between two equally-shaped matrices.
///
/// A degenerate (near-zero-norm) operand on either side scores `-1.0` — the
/// worst possible similarity — so a zeroed pair is a veto, never a pass.
/// Shape equality is enforced by [`TransferQualityGate::assess`] before this
/// is called; the assert is defense-in-depth against truncating `zip`.
fn cosine_similarity(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape(), "cosine_similarity shape mismatch");
    let mut dot = 0.0f64;
    let mut na = 0.0f64;
    let mut nb = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        dot += (*x as f64) * (*y as f64);
        na += (*x as f64) * (*x as f64);
        nb += (*y as f64) * (*y as f64);
    }
    if na <= DEGENERATE_NORM_EPS || nb <= DEGENERATE_NORM_EPS {
        return -1.0;
    }
    (dot / (na.sqrt() * nb.sqrt())) as f32
}

/// Relative Frobenius error `‖a − b‖_F / ‖b‖_F`.
///
/// A degenerate (near-zero-norm) target scores `f32::INFINITY` — even when
/// the prediction is also zero — so a zeroed pair is a veto, never a pass.
/// Shape equality is enforced by [`TransferQualityGate::assess`] before this
/// is called; the assert is defense-in-depth against truncating `zip`.
fn relative_error(a: &Array2<f32>, b: &Array2<f32>) -> f32 {
    assert_eq!(a.shape(), b.shape(), "relative_error shape mismatch");
    let mut diff = 0.0f64;
    let mut norm = 0.0f64;
    for (x, y) in a.iter().zip(b.iter()) {
        let d = (*x as f64) - (*y as f64);
        diff += d * d;
        norm += (*y as f64) * (*y as f64);
    }
    if norm <= DEGENERATE_NORM_EPS {
        return f32::INFINITY;
    }
    (diff.sqrt() / norm.sqrt()) as f32
}
