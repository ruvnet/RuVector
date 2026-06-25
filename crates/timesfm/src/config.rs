//! [`TimesfmConfig`] — the exact TimesFM 1.0 200M hyperparameters.
//!
//! Mirrors the `TimesFMConfig` dataclass defaults from
//! google-research/timesfm (`v1/src/timesfm/pytorch_patched_decoder.py`)
//! and the HF model card `google/timesfm-1.0-200m`.

/// The nine forecast quantiles produced alongside the point (mean) channel.
pub const QUANTILES: [f64; 9] = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9];

/// Hyperparameter bundle for the patched decoder.
///
/// Defaults reproduce TimesFM 1.0 200M exactly. Note the non-obvious bits:
/// `intermediate_size == hidden_size` (NOT 4×), MHA (`num_kv_heads ==
/// num_heads`), and `num_outputs = 1 + len(quantiles)`.
#[derive(Debug, Clone, PartialEq)]
pub struct TimesfmConfig {
    /// Number of stacked decoder layers. Default 20.
    pub num_layers: usize,
    /// Model / residual stream width `D`. Default 1280.
    pub hidden_size: usize,
    /// MLP feed-forward hidden width. Default 1280 (equal to `hidden_size`).
    pub intermediate_size: usize,
    /// Number of attention (query) heads. Default 16.
    pub num_heads: usize,
    /// Number of key/value heads. Default 16 (MHA, not GQA at this size).
    pub num_kv_heads: usize,
    /// Per-head dimension. Default 80 (16 × 80 = 1280).
    pub head_dim: usize,
    /// RMSNorm / LayerNorm epsilon. Default 1e-6.
    pub rms_norm_eps: f64,
    /// Input patch length `P`. Default 32.
    pub patch_len: usize,
    /// Output patch (horizon) length. Default 128.
    pub horizon_len: usize,
    /// Number of frequency buckets for the additive frequency embedding. 3.
    pub num_freq: usize,
    /// Forecast quantiles (9 of them).
    pub quantiles: Vec<f64>,
    /// Padding sentinel value (`pad_val`). Default 1123581321.0.
    pub pad_val: f64,
    /// Numerical tolerance for masked-mean/std. Default 1e-6.
    pub tolerance: f64,
    /// Whether to add the sinusoidal positional embedding. Default true.
    pub use_positional_embedding: bool,
    /// Maximum supported context length. Default 512 (= 16 patches of 32).
    pub max_context_len: usize,
}

impl Default for TimesfmConfig {
    fn default() -> Self {
        Self::timesfm_1p0_200m()
    }
}

impl TimesfmConfig {
    /// The canonical TimesFM 1.0 200M configuration.
    #[must_use]
    pub fn timesfm_1p0_200m() -> Self {
        Self {
            num_layers: 20,
            hidden_size: 1280,
            intermediate_size: 1280,
            num_heads: 16,
            num_kv_heads: 16,
            head_dim: 80,
            rms_norm_eps: 1e-6,
            patch_len: 32,
            horizon_len: 128,
            num_freq: 3,
            quantiles: QUANTILES.to_vec(),
            pad_val: 1_123_581_321.0,
            tolerance: 1e-6,
            use_positional_embedding: true,
            max_context_len: 512,
        }
    }

    /// A small config for fast shape tests (2 layers, narrow width). Keeps the
    /// `head_dim * num_heads == hidden_size` invariant and all structural
    /// relationships of the real model so shape tests stay faithful.
    #[must_use]
    pub fn tiny() -> Self {
        Self {
            num_layers: 2,
            hidden_size: 32,
            intermediate_size: 32,
            num_heads: 4,
            num_kv_heads: 4,
            head_dim: 8,
            rms_norm_eps: 1e-6,
            patch_len: 4,
            horizon_len: 8,
            num_freq: 3,
            quantiles: QUANTILES.to_vec(),
            pad_val: 1_123_581_321.0,
            tolerance: 1e-6,
            use_positional_embedding: true,
            max_context_len: 32,
        }
    }

    /// `num_outputs = 1 + len(quantiles)` (index 0 = mean, 1..=9 = quantiles).
    #[must_use]
    pub fn num_outputs(&self) -> usize {
        1 + self.quantiles.len()
    }

    /// Number of query heads per kv head (`num_queries_per_kv`). 1 for MHA.
    #[must_use]
    pub fn num_queries_per_kv(&self) -> usize {
        self.num_heads / self.num_kv_heads.max(1)
    }

    /// Fused QKV projection output width: `(num_heads + 2*num_kv_heads) * head_dim`.
    #[must_use]
    pub fn qkv_dim(&self) -> usize {
        (self.num_heads + 2 * self.num_kv_heads) * self.head_dim
    }

    /// Input width of `input_ff_layer`: `2 * patch_len` (value + pad mask).
    #[must_use]
    pub fn input_ff_in_dim(&self) -> usize {
        2 * self.patch_len
    }

    /// Output width of `horizon_ff_layer`: `horizon_len * num_outputs`.
    #[must_use]
    pub fn horizon_ff_out_dim(&self) -> usize {
        self.horizon_len * self.num_outputs()
    }

    /// Maximum number of patches in a full context (`max_context_len / patch_len`).
    #[must_use]
    pub fn max_patches(&self) -> usize {
        self.max_context_len / self.patch_len
    }

    /// Validate the structural invariants TimesFM relies on.
    pub fn validate(&self) -> Result<(), String> {
        if self.num_heads * self.head_dim != self.hidden_size {
            return Err(format!(
                "num_heads ({}) * head_dim ({}) must equal hidden_size ({})",
                self.num_heads, self.head_dim, self.hidden_size
            ));
        }
        if self.num_heads % self.num_kv_heads != 0 {
            return Err(format!(
                "num_heads ({}) must be divisible by num_kv_heads ({})",
                self.num_heads, self.num_kv_heads
            ));
        }
        if self.max_context_len % self.patch_len != 0 {
            return Err(format!(
                "max_context_len ({}) must be divisible by patch_len ({})",
                self.max_context_len, self.patch_len
            ));
        }
        Ok(())
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn defaults_match_spec() {
        let c = TimesfmConfig::default();
        assert_eq!(c.num_layers, 20);
        assert_eq!(c.hidden_size, 1280);
        assert_eq!(c.intermediate_size, 1280);
        assert_eq!(c.num_heads, 16);
        assert_eq!(c.num_kv_heads, 16);
        assert_eq!(c.head_dim, 80);
        assert_eq!(c.patch_len, 32);
        assert_eq!(c.horizon_len, 128);
        assert_eq!(c.num_outputs(), 10);
        assert_eq!(c.num_queries_per_kv(), 1);
        assert_eq!(c.qkv_dim(), (16 + 2 * 16) * 80);
        assert_eq!(c.qkv_dim(), 3840);
        assert_eq!(c.input_ff_in_dim(), 64);
        assert_eq!(c.horizon_ff_out_dim(), 128 * 10);
        assert_eq!(c.horizon_ff_out_dim(), 1280);
        assert_eq!(c.max_patches(), 16);
        assert_eq!(c.quantiles.len(), 9);
        c.validate().unwrap();
    }

    #[test]
    fn tiny_is_consistent() {
        TimesfmConfig::tiny().validate().unwrap();
    }
}
