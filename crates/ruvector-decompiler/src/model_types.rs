//! Domain types for LLM model weight decompilation.
//!
//! These types represent the reconstructed architecture, layer topology,
//! tokenizer info, and quantization details extracted from model weight files.

use std::collections::HashMap;

use serde::{Deserialize, Serialize};

use crate::types::WitnessChainData;

/// Result of decompiling an LLM model weight file.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDecompileResult {
    /// Detected file format.
    pub format: ModelFormat,
    /// Reconstructed model architecture.
    pub architecture: ModelArchitecture,
    /// Per-layer breakdown.
    pub layers: Vec<LayerInfo>,
    /// Tokenizer info (if extractable from metadata).
    pub tokenizer: Option<TokenizerInfo>,
    /// Quantization details.
    pub quantization: Option<QuantizationInfo>,
    /// Witness chain for provenance.
    pub witness: WitnessChainData,
    /// Raw metadata key-value pairs.
    pub metadata: HashMap<String, String>,
}

/// Detected model file format.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum ModelFormat {
    /// GGUF format with version number.
    Gguf { version: u32 },
    /// HuggingFace Safetensors format.
    Safetensors,
    /// ONNX format (future).
    Onnx,
}

impl std::fmt::Display for ModelFormat {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Gguf { version } => write!(f, "GGUF v{}", version),
            Self::Safetensors => write!(f, "Safetensors"),
            Self::Onnx => write!(f, "ONNX"),
        }
    }
}

/// Reconstructed model architecture parameters.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelArchitecture {
    /// Architecture family name (e.g., "llama", "mistral", "phi").
    pub name: String,
    /// Hidden dimension size.
    pub hidden_size: usize,
    /// Number of transformer layers.
    pub num_layers: usize,
    /// Number of attention heads.
    pub num_heads: usize,
    /// Number of key-value heads (for GQA/MQA).
    pub num_kv_heads: usize,
    /// FFN intermediate size.
    pub intermediate_size: usize,
    /// Vocabulary size.
    pub vocab_size: usize,
    /// Maximum sequence length.
    pub max_sequence_length: usize,
    /// Total parameter count.
    pub total_params: usize,
    /// Estimated size in MB at FP16.
    pub estimated_size_mb: f64,
}

/// Information about a single layer in the model.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerInfo {
    /// Layer index (0-based).
    pub index: usize,
    /// Layer type classification.
    pub layer_type: LayerType,
    /// Tensor names belonging to this layer.
    pub tensor_names: Vec<String>,
    /// Total parameter count in this layer.
    pub param_count: usize,
    /// Quantization type string (e.g., "Q4_K_M").
    pub quantization: Option<String>,
}

/// Classification of a model layer.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub enum LayerType {
    /// Token embedding layer.
    Embedding,
    /// Self-attention layer with head configuration.
    Attention {
        heads: usize,
        kv_heads: usize,
        head_dim: usize,
    },
    /// Feed-forward / MLP layer.
    Mlp { up_size: usize, down_size: usize },
    /// Layer normalization.
    LayerNorm,
    /// RMS normalization.
    RmsNorm,
    /// Output / language model head.
    Output,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Embedding => write!(f, "Embedding"),
            Self::Attention {
                heads,
                kv_heads,
                head_dim,
            } => {
                if heads == kv_heads {
                    write!(f, "Attention ({} heads, dim {})", heads, head_dim)
                } else {
                    write!(
                        f,
                        "Attention ({} heads, {} KV heads, dim {})",
                        heads, kv_heads, head_dim
                    )
                }
            }
            Self::Mlp { up_size, down_size } => {
                write!(f, "MLP (up {}, down {})", up_size, down_size)
            }
            Self::LayerNorm => write!(f, "LayerNorm"),
            Self::RmsNorm => write!(f, "RMSNorm"),
            Self::Output => write!(f, "Output"),
        }
    }
}

/// Tokenizer information extracted from model metadata.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TokenizerInfo {
    /// Total vocabulary size.
    pub vocab_size: usize,
    /// Special tokens (name, token ID).
    pub special_tokens: Vec<(String, u32)>,
    /// Sample of first N tokens (ID, text).
    pub sample_tokens: Vec<(u32, String)>,
}

/// Quantization method details.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct QuantizationInfo {
    /// Quantization method name (e.g., "Q4_K_M", "Q8_0").
    pub method: String,
    /// Average bits per weight.
    pub bits_per_weight: f32,
    /// Estimated original size in MB (FP16).
    pub original_size_mb: f64,
    /// Quantized file size in MB.
    pub quantized_size_mb: f64,
    /// Compression ratio (original / quantized).
    pub compression_ratio: f32,
}

/// Internal tensor info used during parsing.
#[derive(Debug, Clone)]
pub(crate) struct ModelTensorInfo {
    /// Tensor name.
    pub name: String,
    /// Tensor shape dimensions.
    pub shape: Vec<usize>,
    /// Quantization type ID.
    pub quant_type: u32,
    /// Quantization type name.
    pub quant_name: String,
    /// Bits per weight for this tensor.
    pub bits_per_weight: f32,
    /// Data offset in the file.
    pub offset: u64,
}

impl ModelTensorInfo {
    /// Total number of elements.
    pub fn num_elements(&self) -> usize {
        self.shape.iter().product()
    }
}
