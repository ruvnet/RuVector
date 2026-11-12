//! NAPI-RS bindings for attention mechanisms
//!
//! Provides Node.js bindings for all attention variants:
//! - Scaled dot-product attention
//! - Multi-head attention
//! - Hyperbolic attention
//! - Flash attention
//! - Linear attention
//! - Local-global attention
//! - Mixture of Experts attention
//!
//! Every constructor validates its configuration and every `compute` validates
//! the lengths of its inputs (see [`crate::validate`]). Sync bindings are also
//! marked `catch_unwind`, so a Rust panic surfaces as a JavaScript `Error`
//! instead of aborting the process.

use crate::validate::{self, F32Rows};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_attention::{
    attention::{MultiHeadAttention as RustMultiHead, ScaledDotProductAttention},
    hyperbolic::{HyperbolicAttention as RustHyperbolic, HyperbolicAttentionConfig},
    moe::{MoEAttention as RustMoE, MoEConfig as RustMoEConfig},
    sparse::{
        FlashAttention as RustFlash, LinearAttention as RustLinear,
        LocalGlobalAttention as RustLocalGlobal,
    },
    traits::Attention,
};

/// Validates `query`/`keys`/`values` against `dim` and runs `attention`.
pub(crate) fn run<A: Attention>(
    attention: &A,
    dim: usize,
    value_dim: Option<usize>,
    query: Float32Array,
    keys: F32Rows,
    values: F32Rows,
) -> Result<Float32Array> {
    run_slices(
        attention,
        dim,
        value_dim,
        query.as_ref(),
        &keys.slices(),
        &values.slices(),
    )
}

/// Flat-input variant of [`run`]: `keys` is `n * dim` floats row-major and
/// `values` is `n * value_dim` floats (`value_dim` defaults to
/// `values.length / n`). Three typed arrays cross the boundary instead of
/// `2n + 1`, and nothing is copied.
pub(crate) fn run_flat<A: Attention>(
    attention: &A,
    dim: usize,
    value_dim: Option<usize>,
    query: Float32Array,
    keys: Float32Array,
    values: Float32Array,
) -> Result<Float32Array> {
    let (k, v) = flat_kv(dim, value_dim, keys.as_ref(), values.as_ref())?;
    run_slices(attention, dim, value_dim, query.as_ref(), &k, &v)
}

/// Splits flat keys/values into row slices (see [`run_flat`]).
pub(crate) fn flat_kv<'a>(
    dim: usize,
    value_dim: Option<usize>,
    keys: &'a [f32],
    values: &'a [f32],
) -> Result<(Vec<&'a [f32]>, Vec<&'a [f32]>)> {
    let k = validate::flat_rows("keys", keys, dim)?;
    let n = k.len();
    let vd = match value_dim {
        Some(d) => d,
        None if values.len() % n == 0 => values.len() / n,
        None => {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "values length {} is not a multiple of the key count {n}",
                    values.len()
                ),
            ))
        }
    };
    let v = validate::flat_rows("values", values, vd)?;
    Ok((k, v))
}

fn run_slices<A: Attention>(
    attention: &A,
    dim: usize,
    value_dim: Option<usize>,
    query: &[f32],
    keys: &[&[f32]],
    values: &[&[f32]],
) -> Result<Float32Array> {
    validate::qkv(dim, query, keys, values, value_dim)?;
    let result = attention
        .compute(query, keys, values)
        .map_err(|e| Error::from_reason(e.to_string()))?;
    Ok(Float32Array::new(result))
}

/// Attention configuration object
#[napi(object)]
pub struct AttentionConfig {
    pub dim: u32,
    pub num_heads: Option<u32>,
    pub dropout: Option<f64>,
    pub scale: Option<f64>,
    pub causal: Option<bool>,
}

/// Scaled dot-product attention
#[napi]
pub struct DotProductAttention {
    inner: ScaledDotProductAttention,
    dim_value: usize,
}

#[napi]
impl DotProductAttention {
    /// Create a new scaled dot-product attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        Ok(Self {
            inner: ScaledDotProductAttention::new(d),
            dim_value: d,
        })
    }

    /// Compute attention output
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `keys` - Array of key vectors
    /// * `values` - Array of value vectors
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        run_flat(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Compute attention with mask
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `keys` - Array of key vectors
    /// * `values` - Array of value vectors
    /// * `mask` - Boolean mask array (true = attend, false = mask)
    #[napi(catch_unwind)]
    pub fn compute_with_mask(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
        mask: Vec<bool>,
    ) -> Result<Float32Array> {
        let keys_vec = keys.slices();
        let values_vec = values.slices();
        validate::qkv(self.dim_value, query.as_ref(), &keys_vec, &values_vec, None)?;
        if mask.len() != keys_vec.len() {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "mask must have one entry per key ({}), got {}",
                    keys_vec.len(),
                    mask.len()
                ),
            ));
        }

        let result = self
            .inner
            .compute_with_mask(
                query.as_ref(),
                &keys_vec,
                &values_vec,
                Some(mask.as_slice()),
            )
            .map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Float32Array::new(result))
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.inner.dim() as u32
    }
}

/// Multi-head attention mechanism
#[napi]
pub struct MultiHeadAttention {
    inner: RustMultiHead,
    dim_value: usize,
    num_heads_value: usize,
}

#[napi]
impl MultiHeadAttention {
    /// Create a new multi-head attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension (must be divisible by num_heads)
    /// * `num_heads` - Number of attention heads
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32, num_heads: u32) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let h = validate::heads(d, num_heads)?;
        let inner = RustMultiHead::try_new(d, h).map_err(|e| Error::from_reason(e.to_string()))?;

        Ok(Self {
            inner,
            dim_value: d,
            num_heads_value: h,
        })
    }

    /// Compute multi-head attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        let d = self.dim_value;
        run(&self.inner, d, Some(d), query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        let d = self.dim_value;
        run_flat(&self.inner, d, Some(d), query, keys, values)
    }

    /// Get the number of heads
    #[napi(getter)]
    pub fn num_heads(&self) -> u32 {
        self.num_heads_value as u32
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim_value as u32
    }

    /// Get the head dimension
    #[napi(getter)]
    pub fn head_dim(&self) -> u32 {
        (self.dim_value / self.num_heads_value) as u32
    }
}

/// Hyperbolic attention in Poincaré ball model
#[napi]
pub struct HyperbolicAttention {
    inner: RustHyperbolic,
    curvature_value: f32,
    dim_value: usize,
}

#[napi]
impl HyperbolicAttention {
    /// Create a new hyperbolic attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `curvature` - Hyperbolic curvature (typically 1.0)
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32, curvature: f64) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let c = validate::finite("curvature", curvature, false)?;
        let config = HyperbolicAttentionConfig {
            dim: d,
            curvature: c,
            ..Default::default()
        };
        Ok(Self {
            inner: RustHyperbolic::new(config),
            curvature_value: c,
            dim_value: d,
        })
    }

    /// Create with full configuration
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `curvature` - Hyperbolic curvature
    /// * `adaptive_curvature` - Whether to use adaptive curvature
    /// * `temperature` - Temperature for softmax
    #[napi(factory, catch_unwind)]
    pub fn with_config(
        dim: u32,
        curvature: f64,
        adaptive_curvature: bool,
        temperature: f64,
    ) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let c = validate::finite("curvature", curvature, false)?;
        let t = validate::finite("temperature", temperature, true)?;
        let config = HyperbolicAttentionConfig {
            dim: d,
            curvature: c,
            adaptive_curvature,
            temperature: t,
            frechet_max_iter: 100,
            frechet_tol: 1e-6,
        };
        Ok(Self {
            inner: RustHyperbolic::new(config),
            curvature_value: c,
            dim_value: d,
        })
    }

    /// Compute hyperbolic attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        run_flat(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Get the curvature
    #[napi(getter)]
    pub fn curvature(&self) -> f64 {
        self.curvature_value as f64
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim_value as u32
    }
}

/// Flash attention with tiled computation
#[napi]
pub struct FlashAttention {
    inner: RustFlash,
    dim_value: usize,
    block_size_value: usize,
}

#[napi]
impl FlashAttention {
    /// Create a new flash attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `block_size` - Block size for tiled computation
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32, block_size: u32) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let b = validate::count("blockSize", block_size)?;
        Ok(Self {
            inner: RustFlash::new(d, b),
            dim_value: d,
            block_size_value: b,
        })
    }

    /// Compute flash attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        run_flat(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim_value as u32
    }

    /// Get the block size
    #[napi(getter)]
    pub fn block_size(&self) -> u32 {
        self.block_size_value as u32
    }
}

/// Linear attention (Performer-style) with O(n) complexity
#[napi]
pub struct LinearAttention {
    inner: RustLinear,
    dim_value: usize,
    num_features_value: usize,
}

#[napi]
impl LinearAttention {
    /// Create a new linear attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `num_features` - Number of random features
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32, num_features: u32) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let f = validate::count("numFeatures", num_features)?;
        validate::elements("LinearAttention feature matrix", &[d, f])?;
        Ok(Self {
            inner: RustLinear::new(d, f),
            dim_value: d,
            num_features_value: f,
        })
    }

    /// Compute linear attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        if let Some(v) = values.first_len() {
            validate::elements(
                "LinearAttention key-value summary",
                &[self.num_features_value, v],
            )?;
        }
        run(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        let n = keys.len() / self.dim_value;
        if n > 0 {
            validate::elements(
                "LinearAttention key-value summary",
                &[self.num_features_value, values.len() / n],
            )?;
        }
        run_flat(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim_value as u32
    }

    /// Get the number of random features
    #[napi(getter)]
    pub fn num_features(&self) -> u32 {
        self.num_features_value as u32
    }
}

/// Local-global attention (Longformer-style)
#[napi]
pub struct LocalGlobalAttention {
    inner: RustLocalGlobal,
    dim_value: usize,
    local_window_value: usize,
    global_tokens_value: usize,
}

#[napi]
impl LocalGlobalAttention {
    /// Create a new local-global attention instance
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `local_window` - Size of local attention window
    /// * `global_tokens` - Number of global attention tokens
    #[napi(constructor, catch_unwind)]
    pub fn new(dim: u32, local_window: u32, global_tokens: u32) -> Result<Self> {
        let d = validate::dim("dim", dim)?;
        let w = validate::count("localWindow", local_window)?;
        if global_tokens > validate::MAX_COUNT {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "globalTokens must be at most {}, got {global_tokens}",
                    validate::MAX_COUNT
                ),
            ));
        }
        let g = global_tokens as usize;
        Ok(Self {
            inner: RustLocalGlobal::new(d, w, g),
            dim_value: d,
            local_window_value: w,
            global_tokens_value: g,
        })
    }

    /// Compute local-global attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        run_flat(&self.inner, self.dim_value, None, query, keys, values)
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.dim_value as u32
    }

    /// Get the local window size
    #[napi(getter)]
    pub fn local_window(&self) -> u32 {
        self.local_window_value as u32
    }

    /// Get the number of global tokens
    #[napi(getter)]
    pub fn global_tokens(&self) -> u32 {
        self.global_tokens_value as u32
    }
}

/// MoE attention configuration
#[napi(object)]
pub struct MoEConfig {
    pub dim: u32,
    pub num_experts: u32,
    pub top_k: u32,
    pub expert_capacity: Option<f64>,
}

/// Largest accepted expert count.
const MAX_EXPERTS: u32 = 1024;

/// Mixture of Experts attention
#[napi]
pub struct MoEAttention {
    inner: RustMoE,
    config: MoEConfig,
}

#[napi]
impl MoEAttention {
    /// Create a new MoE attention instance
    ///
    /// # Arguments
    /// * `config` - MoE configuration object
    #[napi(constructor, catch_unwind)]
    pub fn new(config: MoEConfig) -> Result<Self> {
        let d = validate::dim("dim", config.dim)?;
        if config.num_experts == 0 || config.num_experts > MAX_EXPERTS {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "numExperts must be between 1 and {MAX_EXPERTS}, got {}",
                    config.num_experts
                ),
            ));
        }
        if config.top_k == 0 || config.top_k > config.num_experts {
            return Err(Error::new(
                Status::InvalidArg,
                format!(
                    "topK must be between 1 and numExperts ({}), got {}",
                    config.num_experts, config.top_k
                ),
            ));
        }
        let e = config.num_experts as usize;
        // Router gate is [num_experts x dim]; each linear expert holds a
        // [dim/4 x dim] feature matrix.
        validate::elements("MoE router", &[e, d])?;
        validate::elements("MoE experts", &[e, d, d.div_ceil(4)])?;
        let capacity = validate::finite(
            "expertCapacity",
            config.expert_capacity.unwrap_or(1.25),
            true,
        )?;

        let rust_config = RustMoEConfig::builder()
            .dim(d)
            .num_experts(e)
            .top_k(config.top_k as usize)
            .expert_capacity(capacity)
            .build();

        Ok(Self {
            inner: RustMoE::new(rust_config),
            config,
        })
    }

    /// Create with simple parameters
    ///
    /// # Arguments
    /// * `dim` - Embedding dimension
    /// * `num_experts` - Number of expert networks
    /// * `top_k` - Number of experts to route to
    #[napi(factory, catch_unwind)]
    pub fn simple(dim: u32, num_experts: u32, top_k: u32) -> Result<Self> {
        let config = MoEConfig {
            dim,
            num_experts,
            top_k,
            expert_capacity: Some(1.25),
        };
        Self::new(config)
    }

    /// Compute MoE attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        let d = self.config.dim as usize;
        run(&self.inner, d, None, query, keys, values)
    }

    /// Same as `compute`, with keys and values passed as flat row-major
    /// `Float32Array`s (`n * dim` floats each). Faster for many keys: no
    /// per-key typed arrays cross the native boundary.
    #[napi(catch_unwind)]
    pub fn compute_flat(
        &self,
        query: Float32Array,
        keys: Float32Array,
        values: Float32Array,
    ) -> Result<Float32Array> {
        let d = self.config.dim as usize;
        run_flat(&self.inner, d, None, query, keys, values)
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.config.dim
    }

    /// Get the number of experts
    #[napi(getter)]
    pub fn num_experts(&self) -> u32 {
        self.config.num_experts
    }

    /// Get the top-k value
    #[napi(getter)]
    pub fn top_k(&self) -> u32 {
        self.config.top_k
    }
}

// Utility functions

fn same_len(a: &[f32], b: &[f32]) -> Result<()> {
    if a.len() != b.len() {
        return Err(Error::new(
            Status::InvalidArg,
            format!(
                "vectors must have the same length, got {} and {}",
                a.len(),
                b.len()
            ),
        ));
    }
    Ok(())
}

/// Project a vector into the Poincaré ball
#[napi(catch_unwind)]
pub fn project_to_poincare_ball(vector: Float32Array, curvature: f64) -> Result<Float32Array> {
    let c = validate::finite("curvature", curvature, false)?;
    let v = vector.to_vec();
    let projected = ruvector_attention::hyperbolic::project_to_ball(&v, c, 1e-5);
    Ok(Float32Array::new(projected))
}

/// Compute hyperbolic (Poincaré) distance between two points
#[napi(catch_unwind)]
pub fn poincare_distance(a: Float32Array, b: Float32Array, curvature: f64) -> Result<f64> {
    let c = validate::finite("curvature", curvature, false)?;
    same_len(a.as_ref(), b.as_ref())?;
    Ok(ruvector_attention::hyperbolic::poincare_distance(a.as_ref(), b.as_ref(), c) as f64)
}

/// Möbius addition in hyperbolic space
#[napi(catch_unwind)]
pub fn mobius_addition(a: Float32Array, b: Float32Array, curvature: f64) -> Result<Float32Array> {
    let c = validate::finite("curvature", curvature, false)?;
    same_len(a.as_ref(), b.as_ref())?;
    let result = ruvector_attention::hyperbolic::mobius_add(a.as_ref(), b.as_ref(), c);
    Ok(Float32Array::new(result))
}

/// Exponential map from tangent space to hyperbolic space
#[napi(catch_unwind)]
pub fn exp_map(base: Float32Array, tangent: Float32Array, curvature: f64) -> Result<Float32Array> {
    let c = validate::finite("curvature", curvature, false)?;
    same_len(base.as_ref(), tangent.as_ref())?;
    let result = ruvector_attention::hyperbolic::exp_map(base.as_ref(), tangent.as_ref(), c);
    Ok(Float32Array::new(result))
}

/// Logarithmic map from hyperbolic space to tangent space
#[napi(catch_unwind)]
pub fn log_map(base: Float32Array, point: Float32Array, curvature: f64) -> Result<Float32Array> {
    let c = validate::finite("curvature", curvature, false)?;
    same_len(base.as_ref(), point.as_ref())?;
    let result = ruvector_attention::hyperbolic::log_map(base.as_ref(), point.as_ref(), c);
    Ok(Float32Array::new(result))
}
