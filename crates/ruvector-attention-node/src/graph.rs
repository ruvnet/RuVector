//! NAPI-RS bindings for graph attention mechanisms
//!
//! Provides Node.js bindings for:
//! - Edge-featured attention (GATv2-style)
//! - Graph RoPE (Rotary Position Embeddings for graphs)
//! - Dual-space attention (Euclidean + Hyperbolic)
//!
//! Configuration and input lengths are validated before they reach the Rust
//! code, and sync bindings are marked `catch_unwind`.

use crate::attention::run;
use crate::validate::{self, F32Rows};
use napi::bindgen_prelude::*;
use napi_derive::napi;
use ruvector_attention::graph::{
    DualSpaceAttention as RustDualSpace, DualSpaceConfig as RustDualConfig,
    EdgeFeaturedAttention as RustEdgeFeatured, EdgeFeaturedConfig as RustEdgeConfig,
    GraphRoPE as RustGraphRoPE, RoPEConfig as RustRoPEConfig,
};

fn invalid(message: String) -> Error {
    Error::new(Status::InvalidArg, message)
}

fn rust_err(e: ruvector_attention::error::AttentionError) -> Error {
    Error::from_reason(e.to_string())
}

// ============================================================================
// Edge-Featured Attention
// ============================================================================

/// Configuration for edge-featured attention
#[napi(object)]
pub struct EdgeFeaturedConfig {
    pub node_dim: u32,
    pub edge_dim: u32,
    pub num_heads: u32,
    pub concat_heads: Option<bool>,
    pub add_self_loops: Option<bool>,
    pub negative_slope: Option<f64>,
}

/// Edge-featured attention (GATv2-style)
#[napi]
pub struct EdgeFeaturedAttention {
    inner: RustEdgeFeatured,
    config: EdgeFeaturedConfig,
}

#[napi]
impl EdgeFeaturedAttention {
    /// Create a new edge-featured attention instance
    ///
    /// # Arguments
    /// * `config` - Edge-featured attention configuration
    #[napi(constructor, catch_unwind)]
    pub fn new(config: EdgeFeaturedConfig) -> Result<Self> {
        let node_dim = validate::dim("nodeDim", config.node_dim)?;
        let edge_dim = validate::dim("edgeDim", config.edge_dim)?;
        let num_heads = validate::count("numHeads", config.num_heads)?;
        if num_heads > node_dim {
            return Err(invalid(format!(
                "numHeads ({num_heads}) must not exceed nodeDim ({node_dim})"
            )));
        }
        // Node and edge projections are [num_heads x head_dim x {node,edge}_dim].
        let head_dim = node_dim / num_heads;
        validate::elements(
            "EdgeFeaturedAttention node weights",
            &[num_heads, head_dim, node_dim],
        )?;
        validate::elements(
            "EdgeFeaturedAttention edge weights",
            &[num_heads, head_dim, edge_dim],
        )?;
        let negative_slope =
            validate::finite("negativeSlope", config.negative_slope.unwrap_or(0.2), false)?;

        let rust_config = RustEdgeConfig {
            node_dim,
            edge_dim,
            num_heads,
            concat_heads: config.concat_heads.unwrap_or(true),
            add_self_loops: config.add_self_loops.unwrap_or(true),
            negative_slope,
            dropout: 0.0,
        };
        Ok(Self {
            inner: RustEdgeFeatured::try_new(rust_config).map_err(rust_err)?,
            config,
        })
    }

    /// Create with simple parameters
    #[napi(factory, catch_unwind)]
    pub fn simple(node_dim: u32, edge_dim: u32, num_heads: u32) -> Result<Self> {
        Self::new(EdgeFeaturedConfig {
            node_dim,
            edge_dim,
            num_heads,
            concat_heads: Some(true),
            add_self_loops: Some(true),
            negative_slope: Some(0.2),
        })
    }

    /// Compute attention without edge features (standard attention)
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        let d = self.config.node_dim as usize;
        run(&self.inner, d, Some(d), query, keys, values)
    }

    /// Compute attention with edge features
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `keys` - Array of key vectors
    /// * `values` - Array of value vectors
    /// * `edge_features` - Array of edge feature vectors (same length as keys)
    #[napi(catch_unwind)]
    pub fn compute_with_edges(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] edge_features: F32Rows,
    ) -> Result<Float32Array> {
        let d = self.config.node_dim as usize;
        let edge_dim = self.config.edge_dim as usize;
        let keys_vec = keys.slices();
        let values_vec = values.slices();
        let edges_vec = edge_features.slices();
        validate::qkv(d, query.as_ref(), &keys_vec, &values_vec, Some(d))?;
        if edges_vec.len() != keys_vec.len() {
            return Err(invalid(format!(
                "edgeFeatures must have one entry per key ({}), got {}",
                keys_vec.len(),
                edges_vec.len()
            )));
        }
        for (i, edge) in edges_vec.iter().enumerate() {
            validate::len(&format!("edgeFeatures[{i}]"), edge, edge_dim)?;
        }

        let result = self
            .inner
            .compute_with_edges(query.as_ref(), &keys_vec, &values_vec, &edges_vec)
            .map_err(rust_err)?;

        Ok(Float32Array::new(result))
    }

    /// Get the node dimension
    #[napi(getter)]
    pub fn node_dim(&self) -> u32 {
        self.config.node_dim
    }

    /// Get the edge dimension
    #[napi(getter)]
    pub fn edge_dim(&self) -> u32 {
        self.config.edge_dim
    }

    /// Get the number of heads
    #[napi(getter)]
    pub fn num_heads(&self) -> u32 {
        self.config.num_heads
    }
}

// ============================================================================
// Graph RoPE Attention
// ============================================================================

/// Configuration for Graph RoPE attention
#[napi(object)]
pub struct RoPEConfig {
    pub dim: u32,
    pub max_position: u32,
    pub base: Option<f64>,
    pub scaling_factor: Option<f64>,
}

/// Graph RoPE attention (Rotary Position Embeddings for graphs)
#[napi]
pub struct GraphRoPEAttention {
    inner: RustGraphRoPE,
    config: RoPEConfig,
}

#[napi]
impl GraphRoPEAttention {
    /// Create a new Graph RoPE attention instance
    ///
    /// # Arguments
    /// * `config` - RoPE configuration
    #[napi(constructor, catch_unwind)]
    pub fn new(config: RoPEConfig) -> Result<Self> {
        let dim = validate::dim("dim", config.dim)?;
        if dim % 2 != 0 {
            return Err(invalid(format!("dim must be even for RoPE, got {dim}")));
        }
        let max_position = validate::count("maxPosition", config.max_position)?;
        // cos/sin caches are [max_position x dim] each.
        validate::elements("GraphRoPE position cache", &[max_position, dim])?;
        let base = validate::finite("base", config.base.unwrap_or(10000.0), true)?;
        let scaling_factor =
            validate::finite("scalingFactor", config.scaling_factor.unwrap_or(1.0), true)?;

        let rust_config = RustRoPEConfig {
            dim,
            max_position,
            base,
            scaling_factor,
        };
        Ok(Self {
            inner: RustGraphRoPE::try_new(rust_config).map_err(rust_err)?,
            config,
        })
    }

    /// Create with simple parameters
    #[napi(factory, catch_unwind)]
    pub fn simple(dim: u32, max_position: u32) -> Result<Self> {
        Self::new(RoPEConfig {
            dim,
            max_position,
            base: Some(10000.0),
            scaling_factor: Some(1.0),
        })
    }

    /// Compute attention without positional encoding
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(
            &self.inner,
            self.config.dim as usize,
            None,
            query,
            keys,
            values,
        )
    }

    /// Compute attention with graph positions
    ///
    /// # Arguments
    /// * `query` - Query vector
    /// * `keys` - Array of key vectors
    /// * `values` - Array of value vectors
    /// * `query_position` - Position of query node
    /// * `key_positions` - Positions of key nodes (e.g., hop distances)
    #[napi(catch_unwind)]
    pub fn compute_with_positions(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
        query_position: u32,
        key_positions: Vec<u32>,
    ) -> Result<Float32Array> {
        let keys_vec = keys.slices();
        let values_vec = values.slices();
        validate::qkv(
            self.config.dim as usize,
            query.as_ref(),
            &keys_vec,
            &values_vec,
            None,
        )?;
        if key_positions.len() != keys_vec.len() {
            return Err(invalid(format!(
                "keyPositions must have one entry per key ({}), got {}",
                keys_vec.len(),
                key_positions.len()
            )));
        }
        let positions_usize: Vec<usize> = key_positions.into_iter().map(|p| p as usize).collect();

        let result = self
            .inner
            .compute_with_positions(
                query.as_ref(),
                &keys_vec,
                &values_vec,
                query_position as usize,
                &positions_usize,
            )
            .map_err(rust_err)?;

        Ok(Float32Array::new(result))
    }

    /// Apply rotary embedding to a vector
    #[napi(catch_unwind)]
    pub fn apply_rotary(&self, vector: Float32Array, position: u32) -> Result<Float32Array> {
        let result = self
            .inner
            .try_apply_rotary(vector.as_ref(), position as usize)
            .map_err(rust_err)?;
        Ok(Float32Array::new(result))
    }

    /// Convert graph distance to position bucket
    #[napi(catch_unwind)]
    pub fn distance_to_position(distance: u32, max_distance: u32) -> u32 {
        RustGraphRoPE::distance_to_position(distance as usize, max_distance as usize) as u32
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.config.dim
    }

    /// Get the max position
    #[napi(getter)]
    pub fn max_position(&self) -> u32 {
        self.config.max_position
    }
}

// ============================================================================
// Dual-Space Attention
// ============================================================================

/// Configuration for dual-space attention
#[napi(object)]
pub struct DualSpaceConfig {
    pub dim: u32,
    pub curvature: f64,
    pub euclidean_weight: f64,
    pub hyperbolic_weight: f64,
    pub temperature: Option<f64>,
}

/// Dual-space attention (Euclidean + Hyperbolic)
#[napi]
pub struct DualSpaceAttention {
    inner: RustDualSpace,
    config: DualSpaceConfig,
}

#[napi]
impl DualSpaceAttention {
    /// Create a new dual-space attention instance
    ///
    /// # Arguments
    /// * `config` - Dual-space configuration
    #[napi(constructor, catch_unwind)]
    pub fn new(config: DualSpaceConfig) -> Result<Self> {
        let dim = validate::dim("dim", config.dim)?;
        // Three [dim x dim] projection matrices.
        validate::elements("DualSpaceAttention projections", &[3, dim, dim])?;
        let rust_config = RustDualConfig {
            dim,
            curvature: validate::finite("curvature", config.curvature, false)?,
            euclidean_weight: validate::finite("euclideanWeight", config.euclidean_weight, false)?,
            hyperbolic_weight: validate::finite(
                "hyperbolicWeight",
                config.hyperbolic_weight,
                false,
            )?,
            learn_weights: false,
            temperature: validate::finite("temperature", config.temperature.unwrap_or(1.0), true)?,
        };
        Ok(Self {
            inner: RustDualSpace::new(rust_config),
            config,
        })
    }

    /// Create with simple parameters (equal weights)
    #[napi(factory, catch_unwind)]
    pub fn simple(dim: u32, curvature: f64) -> Result<Self> {
        Self::new(DualSpaceConfig {
            dim,
            curvature,
            euclidean_weight: 0.5,
            hyperbolic_weight: 0.5,
            temperature: Some(1.0),
        })
    }

    /// Create with custom weights
    #[napi(factory, catch_unwind)]
    pub fn with_weights(
        dim: u32,
        curvature: f64,
        euclidean_weight: f64,
        hyperbolic_weight: f64,
    ) -> Result<Self> {
        Self::new(DualSpaceConfig {
            dim,
            curvature,
            euclidean_weight,
            hyperbolic_weight,
            temperature: Some(1.0),
        })
    }

    /// Compute dual-space attention
    #[napi(catch_unwind)]
    pub fn compute(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
        #[napi(ts_arg_type = "Float32Array[]")] values: F32Rows,
    ) -> Result<Float32Array> {
        run(
            &self.inner,
            self.config.dim as usize,
            None,
            query,
            keys,
            values,
        )
    }

    /// Get space contributions (Euclidean and Hyperbolic scores separately)
    #[napi(catch_unwind)]
    pub fn get_space_contributions(
        &self,
        query: Float32Array,
        #[napi(ts_arg_type = "Float32Array[]")] keys: F32Rows,
    ) -> Result<SpaceContributions> {
        let keys_vec = keys.slices();
        validate::qk(self.config.dim as usize, query.as_ref(), &keys_vec)?;

        let (euc_scores, hyp_scores) = self
            .inner
            .get_space_contributions(query.as_ref(), &keys_vec);

        Ok(SpaceContributions {
            euclidean_scores: Float32Array::new(euc_scores),
            hyperbolic_scores: Float32Array::new(hyp_scores),
        })
    }

    /// Get the dimension
    #[napi(getter)]
    pub fn dim(&self) -> u32 {
        self.config.dim
    }

    /// Get the curvature
    #[napi(getter)]
    pub fn curvature(&self) -> f64 {
        self.config.curvature
    }

    /// Get the Euclidean weight
    #[napi(getter)]
    pub fn euclidean_weight(&self) -> f64 {
        self.config.euclidean_weight
    }

    /// Get the Hyperbolic weight
    #[napi(getter)]
    pub fn hyperbolic_weight(&self) -> f64 {
        self.config.hyperbolic_weight
    }
}

/// Space contribution scores
#[napi(object)]
pub struct SpaceContributions {
    pub euclidean_scores: Float32Array,
    pub hyperbolic_scores: Float32Array,
}
