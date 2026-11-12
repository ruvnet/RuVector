//! KV mappers: the [`KvMapper`] trait and the closed-form
//! [`LinearKvMapper`] from arXiv:2608.03893.
//!
//! ## Approach
//!
//! For each transformer layer `l`, the linear mapper learns two matrices
//! `W_k[l]` and `W_v[l]` of shape `[src_dim, tgt_dim]` such that
//!
//! ```text
//! K_tgt ≈ K_src · W_k[l]      V_tgt ≈ V_src · W_v[l]
//! ```
//!
//! where `K_src` is `[tokens, src_dim]` and `K_tgt` is `[tokens, tgt_dim]`,
//! with `dim = num_kv_heads * head_dim` (the flattened per-token KV vector).
//!
//! The fit is the closed-form ridge-regularized least-squares solution
//! (normal equations):
//!
//! ```text
//! W = (Xᵀ X + λ I)⁻¹ Xᵀ Y
//! ```
//!
//! solved by Gaussian elimination with partial pivoting — no external
//! linear-algebra dependency beyond `ndarray`, which the crate already uses.
//!
//! ## Head-count / head-dim mismatch
//!
//! Following the paper, geometry mismatches between source and target
//! (different `num_kv_heads`, different `head_dim`, or both) are handled by
//! learning the projection over the *flattened* per-token KV vector:
//! `W` is `[src_heads * src_head_dim, tgt_heads * tgt_head_dim]`, so any
//! rectangular geometry change is expressible. When the head grids align,
//! the learned `W` naturally converges to the paper's block-diagonal
//! per-head projection; when they do not, the full matrix subsumes the
//! paper's block/linear projection without special-casing.
//!
//! ## Assumptions (documented per ADR-314)
//!
//! 1. Calibration pairs are *position-aligned*: row `t` of the source
//!    activations and row `t` of the target activations correspond to the
//!    same token at the same position (same tokenizer, or a shared prefix
//!    tokenized identically by both models).
//! 2. RoPE/positional encoding conventions are compatible between the two
//!    models (true within a model family, the paper's scope).
//! 3. Per-layer correspondence is 1:1 — source layer `l` maps to target
//!    layer `l`. Depth-mismatched families are out of scope for this slice.

use crate::error::{Result, RuvLLMError};
use ndarray::Array2;

use super::LayerKvTensor;

/// Default ridge regularization strength for [`LinearKvMapper::fit`].
///
/// Small enough not to bias a well-conditioned fit, large enough to keep the
/// normal equations solvable when calibration tokens barely cover `src_dim`.
pub const DEFAULT_RIDGE_LAMBDA: f32 = 1e-4;

/// Maximum flattened per-token KV dimension (`num_kv_heads * head_dim`)
/// accepted by [`LinearKvMapper::fit`], on both the source and target side.
///
/// The closed-form solve is O(n³) in the source dimension and allocates an
/// `[n, n]` Gram matrix, so calibration input of size O(tokens · n) buys
/// O(n³) work — an unbounded `n` is a denial-of-service hazard (`n = 65_536`
/// would request a ~17 GB Gram matrix, and an allocation failure aborts the
/// process). 8192 comfortably covers realistic flattened KV dimensions while
/// keeping the solve tractable.
pub const MAX_FIT_DIM: usize = 8192;

/// Maximum number of calibration layers accepted by [`LinearKvMapper::fit`].
/// Each layer costs two independent O(n³) solves (keys and values).
pub const MAX_CALIBRATION_LAYERS: usize = 512;

/// Maximum calibration tokens per layer accepted by [`LinearKvMapper::fit`].
/// Bounds the O(tokens · n²) Gram-matrix accumulation.
pub const MAX_CALIBRATION_TOKENS: usize = 1 << 20;

/// Paired calibration activations for one layer: the same token sequence's
/// K/V tensors as produced by the source model and by the target model.
#[derive(Debug, Clone)]
pub struct CalibrationLayerPair {
    /// Source-model K/V for the calibration tokens, `[tokens, src_dim]`.
    pub source: LayerKvTensor,
    /// Target-model K/V for the same tokens, `[tokens, tgt_dim]`.
    pub target: LayerKvTensor,
}

impl CalibrationLayerPair {
    /// Create a pair, validating token alignment.
    pub fn new(source: LayerKvTensor, target: LayerKvTensor) -> Result<Self> {
        if source.tokens() != target.tokens() {
            return Err(RuvLLMError::KvCache(format!(
                "calibration pair token mismatch: source={}, target={}",
                source.tokens(),
                target.tokens()
            )));
        }
        Ok(Self { source, target })
    }
}

/// Maps a per-layer K/V tensor pair from source-model geometry to
/// target-model geometry.
///
/// Implemented by [`LinearKvMapper`] (this slice) and, in a future slice,
/// by the nonlinear [`super::MlpKvMapper`] fallback the paper prescribes
/// for the pairs where the linear map degrades.
pub trait KvMapper: Send + Sync {
    /// Number of layers this mapper was fitted for.
    fn num_layers(&self) -> usize;

    /// Source flattened per-token KV dimension.
    fn source_dim(&self) -> usize;

    /// Target flattened per-token KV dimension.
    fn target_dim(&self) -> usize;

    /// Map one layer's `[tokens, src_dim]` K/V matrices into
    /// `[tokens, tgt_dim]` target geometry.
    fn map_layer(
        &self,
        layer: usize,
        keys: &Array2<f32>,
        values: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>)>;
}

/// Per-layer fitted projection matrices.
#[derive(Debug, Clone)]
struct LayerMaps {
    /// Key projection, `[src_dim, tgt_dim]`.
    w_k: Array2<f32>,
    /// Value projection, `[src_dim, tgt_dim]`.
    w_v: Array2<f32>,
}

/// Closed-form linear KV mapper (arXiv:2608.03893).
///
/// Fit once per (source model, target model) pair from a small paired
/// calibration set, then applied to any number of cached sequences via two
/// matmuls per layer.
#[derive(Debug, Clone)]
pub struct LinearKvMapper {
    layers: Vec<LayerMaps>,
    source_dim: usize,
    target_dim: usize,
}

impl LinearKvMapper {
    /// Fit per-layer `W_k`, `W_v` from paired calibration activations using
    /// the closed-form ridge least-squares solution
    /// `W = (Xᵀ X + λ I)⁻¹ Xᵀ Y`.
    ///
    /// `calibration` holds one [`CalibrationLayerPair`] per layer, in layer
    /// order. All layers must share the same source and target dimensions.
    /// `ridge_lambda` must be positive; see [`DEFAULT_RIDGE_LAMBDA`].
    ///
    /// ## Cost
    ///
    /// Per layer, the closed-form solve costs O(tokens · n² + n³ + n² · m)
    /// time and allocates an `[n, n]` Gram matrix, where `n` is the source
    /// dimension and `m` the target dimension — the O(n³) term dominates for
    /// realistic shapes. Inputs are bounded by [`MAX_FIT_DIM`],
    /// [`MAX_CALIBRATION_LAYERS`], and [`MAX_CALIBRATION_TOKENS`]; anything
    /// larger is rejected up front, before any allocation or solve.
    pub fn fit(calibration: &[CalibrationLayerPair], ridge_lambda: f32) -> Result<Self> {
        if calibration.is_empty() {
            return Err(RuvLLMError::KvCache(
                "cannot fit LinearKvMapper from empty calibration set".to_string(),
            ));
        }
        if calibration.len() > MAX_CALIBRATION_LAYERS {
            return Err(RuvLLMError::KvCache(format!(
                "calibration set has {} layers, exceeding MAX_CALIBRATION_LAYERS = {}",
                calibration.len(),
                MAX_CALIBRATION_LAYERS
            )));
        }
        if !(ridge_lambda > 0.0) || !ridge_lambda.is_finite() {
            return Err(RuvLLMError::KvCache(format!(
                "ridge_lambda must be a positive finite value, got {ridge_lambda}"
            )));
        }

        let source_dim = calibration[0].source.dim();
        let target_dim = calibration[0].target.dim();
        if source_dim == 0 || target_dim == 0 {
            return Err(RuvLLMError::KvCache(format!(
                "calibration dims must be non-zero, got {source_dim}->{target_dim}"
            )));
        }
        if source_dim > MAX_FIT_DIM || target_dim > MAX_FIT_DIM {
            return Err(RuvLLMError::KvCache(format!(
                "calibration dims {source_dim}->{target_dim} exceed MAX_FIT_DIM = {MAX_FIT_DIM} \
                 (the solve is O(n³) in the source dimension)"
            )));
        }

        // Validate every layer BEFORE the first O(n³) solve, so a malformed
        // set is rejected without paying for any partial fit.
        for (idx, pair) in calibration.iter().enumerate() {
            if pair.source.dim() != source_dim || pair.target.dim() != target_dim {
                return Err(RuvLLMError::KvCache(format!(
                    "layer {idx} dims ({}->{}) differ from layer 0 ({source_dim}->{target_dim})",
                    pair.source.dim(),
                    pair.target.dim()
                )));
            }
            if pair.source.tokens() > MAX_CALIBRATION_TOKENS {
                return Err(RuvLLMError::KvCache(format!(
                    "layer {idx} has {} calibration tokens, exceeding MAX_CALIBRATION_TOKENS = {}",
                    pair.source.tokens(),
                    MAX_CALIBRATION_TOKENS
                )));
            }
        }

        let mut layers = Vec::with_capacity(calibration.len());
        for pair in calibration.iter() {
            let w_k = ridge_least_squares(&pair.source.keys, &pair.target.keys, ridge_lambda)?;
            let w_v = ridge_least_squares(&pair.source.values, &pair.target.values, ridge_lambda)?;
            layers.push(LayerMaps { w_k, w_v });
        }

        Ok(Self {
            layers,
            source_dim,
            target_dim,
        })
    }
}

impl KvMapper for LinearKvMapper {
    fn num_layers(&self) -> usize {
        self.layers.len()
    }

    fn source_dim(&self) -> usize {
        self.source_dim
    }

    fn target_dim(&self) -> usize {
        self.target_dim
    }

    fn map_layer(
        &self,
        layer: usize,
        keys: &Array2<f32>,
        values: &Array2<f32>,
    ) -> Result<(Array2<f32>, Array2<f32>)> {
        let maps = self.layers.get(layer).ok_or_else(|| {
            RuvLLMError::KvCache(format!(
                "layer {layer} out of range (mapper has {} layers)",
                self.layers.len()
            ))
        })?;
        if keys.ncols() != self.source_dim || values.ncols() != self.source_dim {
            return Err(RuvLLMError::KvCache(format!(
                "layer {layer}: input dim {} does not match mapper source dim {}",
                keys.ncols(),
                self.source_dim
            )));
        }
        Ok((keys.dot(&maps.w_k), values.dot(&maps.w_v)))
    }
}

/// Closed-form ridge least squares: `W = (Xᵀ X + λ I)⁻¹ Xᵀ Y`.
///
/// `x` is `[tokens, n]`, `y` is `[tokens, m]`; returns `W` as `[n, m]`.
fn ridge_least_squares(x: &Array2<f32>, y: &Array2<f32>, lambda: f32) -> Result<Array2<f32>> {
    if x.nrows() != y.nrows() {
        return Err(RuvLLMError::KvCache(format!(
            "least squares row mismatch: x has {}, y has {}",
            x.nrows(),
            y.nrows()
        )));
    }
    if x.nrows() == 0 {
        return Err(RuvLLMError::KvCache(
            "least squares requires at least one calibration token".to_string(),
        ));
    }

    let n = x.ncols();
    let mut a = x.t().dot(x); // [n, n]
    for i in 0..n {
        a[[i, i]] += lambda;
    }
    let b = x.t().dot(y); // [n, m]
    solve_linear_system(a, b)
}

/// Solve `A · W = B` for `W` by Gaussian elimination with partial pivoting.
///
/// `a` is `[n, n]` (consumed), `b` is `[n, m]`; returns `W` as `[n, m]`.
/// `A` here is the ridge-regularized Gram matrix, so it is symmetric
/// positive definite and the elimination cannot hit a zero pivot unless the
/// inputs are degenerate (NaN/Inf), which is reported as an error.
fn solve_linear_system(mut a: Array2<f32>, mut b: Array2<f32>) -> Result<Array2<f32>> {
    let n = a.nrows();
    let m = b.ncols();

    // Forward elimination with partial pivoting.
    for col in 0..n {
        // Find pivot row.
        let mut pivot = col;
        let mut best = a[[col, col]].abs();
        for row in (col + 1)..n {
            let mag = a[[row, col]].abs();
            if mag > best {
                best = mag;
                pivot = row;
            }
        }
        if !best.is_finite() || best < f32::EPSILON {
            return Err(RuvLLMError::KvCache(format!(
                "singular or degenerate system at column {col} (pivot magnitude {best})"
            )));
        }
        if pivot != col {
            for j in 0..n {
                a.swap([col, j], [pivot, j]);
            }
            for j in 0..m {
                b.swap([col, j], [pivot, j]);
            }
        }

        let diag = a[[col, col]];
        for row in (col + 1)..n {
            let factor = a[[row, col]] / diag;
            if factor == 0.0 {
                continue;
            }
            for j in col..n {
                a[[row, j]] -= factor * a[[col, j]];
            }
            for j in 0..m {
                b[[row, j]] -= factor * b[[col, j]];
            }
        }
    }

    // Back substitution.
    let mut w = Array2::<f32>::zeros((n, m));
    for row in (0..n).rev() {
        for j in 0..m {
            let mut acc = b[[row, j]];
            for k in (row + 1)..n {
                acc -= a[[row, k]] * w[[k, j]];
            }
            w[[row, j]] = acc / a[[row, row]];
        }
    }

    Ok(w)
}
