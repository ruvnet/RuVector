//! BitNet b1.58 Inference Backend
//!
//! This module implements the `BitNetBackend` inference pipeline for BitNet b1.58
//! MoE models (e.g., GLM-4.7-Flash). It wires together the quantizer, TL1 kernel,
//! and MoE routing into a working inference pipeline.
//!
//! ## Phase 0 Scope
//!
//! - Attention is a placeholder (pass-through) for smoke testing
//! - MoE routing is fully functional (FP16 gate + softmax + top-K)
//! - Expert FFN uses real TL1 GEMV on ternary weights
//! - Embedding lookup and LM head are FP16 matmul
//!
//! ## Architecture
//!
//! ```text
//! Embedding (FP16) -> [Transformer Layers] -> RMSNorm -> LM Head (FP16) -> Logits
//!
//! Each Transformer Layer:
//!   RMSNorm -> Attention (placeholder) -> Residual
//!   -> RMSNorm -> MoE Gate (FP16) -> Top-K Expert Selection
//!   -> Expert FFN (TL1 GEMV on ternary) -> Weighted Sum -> Residual
//! ```

use std::path::Path;

use crate::backends::{
    GenerateParams, GeneratedToken, LlmBackend, ModelArchitecture, ModelConfig,
    ModelInfo, Quantization, StreamEvent, TokenStream, Tokenizer,
};
use crate::error::{Result, RuvLLMError};
use crate::gguf::{GgufFile, GgufQuantType};

use super::ternary_tensor::TernaryTensor;

// ============================================================================
// Configuration
// ============================================================================

/// Model configuration for BitNet MoE inference.
///
/// Describes the architecture dimensions extracted from GGUF metadata
/// or supplied manually for testing.
#[derive(Debug, Clone)]
pub struct BitNetModelConfig {
    /// Number of transformer layers
    pub num_layers: usize,
    /// Hidden state dimension
    pub hidden_size: usize,
    /// Number of MoE experts per layer
    pub num_experts: usize,
    /// Number of active experts per token (top-K)
    pub active_experts: usize,
    /// FFN intermediate dimension per expert
    pub intermediate_size: usize,
    /// Number of attention query heads
    pub num_attention_heads: usize,
    /// Number of attention key-value heads (GQA)
    pub num_kv_heads: usize,
    /// Vocabulary size
    pub vocab_size: usize,
    /// Maximum context length
    pub max_context: usize,
    /// RoPE frequency base
    pub rope_theta: f32,
}

impl Default for BitNetModelConfig {
    fn default() -> Self {
        // Default values loosely based on GLM-4.7-Flash architecture
        Self {
            num_layers: 28,
            hidden_size: 4096,
            num_experts: 8,
            active_experts: 2,
            intermediate_size: 11008,
            num_attention_heads: 32,
            num_kv_heads: 8,
            vocab_size: 151552,
            max_context: 8192,
            rope_theta: 10000.0,
        }
    }
}

// ============================================================================
// TL1 Lookup Table
// ============================================================================

/// Pre-computed lookup table for packed 2-bit ternary bytes.
///
/// For each of the 256 possible byte values, stores the four decoded
/// ternary values {-1, 0, +1}. This avoids per-element bit manipulation
/// during the hot GEMV inner loop.
type Tl1Lut = [[i8; 4]; 256];

/// Build the TL1 lookup table at load time.
///
/// Encoding per the ternary_tensor module:
/// - 00 = -1, 01 = 0, 10 = +1, 11 = 0 (reserved)
fn build_tl1_lut() -> Tl1Lut {
    let mut lut = [[0i8; 4]; 256];
    for byte_val in 0u16..256 {
        for pos in 0..4 {
            let bits = ((byte_val as u8) >> (pos * 2)) & 0b11;
            lut[byte_val as usize][pos] = match bits {
                0b00 => -1,
                0b01 => 0,
                0b10 => 1,
                0b11 => 0, // reserved
                _ => unreachable!(),
            };
        }
    }
    lut
}

// ============================================================================
// Per-Layer and Per-Expert Weight Storage
// ============================================================================

/// Ternary weights for a single MoE expert (gate, up, down projections).
#[derive(Debug, Clone)]
struct ExpertWeights {
    /// gate_proj: [intermediate_size, hidden_size]
    gate_proj: TernaryTensor,
    /// up_proj: [intermediate_size, hidden_size]
    up_proj: TernaryTensor,
    /// down_proj: [hidden_size, intermediate_size]
    down_proj: TernaryTensor,
}

/// Weights for a single transformer layer.
#[derive(Debug, Clone)]
struct TransformerLayer {
    /// Input RMSNorm weight [hidden_size]
    input_norm_weight: Vec<f32>,
    /// Post-attention RMSNorm weight [hidden_size]
    post_attn_norm_weight: Vec<f32>,
    /// MoE router gate weight [num_experts, hidden_size] (FP32, stored row-major)
    gate_weight: Vec<f32>,
    /// Per-expert FFN weights (ternary)
    experts: Vec<ExpertWeights>,
}

// ============================================================================
// BitNetBackend
// ============================================================================

/// BitNet b1.58 MoE inference backend.
///
/// Provides model loading from GGUF and forward pass inference using
/// ternary TL1 GEMV kernels for expert FFN layers and FP32 for shared
/// layers (embeddings, norms, router, LM head).
///
/// # Example
///
/// ```rust,ignore
/// use ruvllm::bitnet::backend::BitNetBackend;
/// use ruvllm::backends::{LlmBackend, ModelConfig, GenerateParams};
///
/// let mut backend = BitNetBackend::new();
/// backend.load_model("model.gguf", ModelConfig::default())?;
///
/// let logits = backend.forward(&[1, 2, 3])?;
/// ```
pub struct BitNetBackend {
    /// Model configuration (set after load)
    config: Option<BitNetModelConfig>,
    /// Embedding table [vocab_size * hidden_size], row-major FP32
    embedding: Vec<f32>,
    /// LM head weight [vocab_size * hidden_size], row-major FP32
    lm_head: Vec<f32>,
    /// Final RMSNorm weight [hidden_size]
    final_norm_weight: Vec<f32>,
    /// Transformer layers
    layers: Vec<TransformerLayer>,
    /// Pre-computed TL1 lookup table
    tl1_lut: Tl1Lut,
    /// Whether a model is loaded
    loaded: bool,
    /// Model path (for info)
    model_path: String,
}

impl BitNetBackend {
    /// Create a new unloaded BitNetBackend.
    pub fn new() -> Self {
        Self {
            config: None,
            embedding: Vec::new(),
            lm_head: Vec::new(),
            final_norm_weight: Vec::new(),
            layers: Vec::new(),
            tl1_lut: build_tl1_lut(),
            loaded: false,
            model_path: String::new(),
        }
    }

    // ========================================================================
    // Model Loading
    // ========================================================================

    /// Load a BitNet MoE model from a GGUF file.
    ///
    /// Parses the GGUF file, extracts model configuration from metadata,
    /// separates FP16 shared tensors from ternary expert tensors, and
    /// pre-builds the TL1 lookup table.
    fn load_gguf(&mut self, path: &str) -> Result<()> {
        let gguf = GgufFile::open_mmap(Path::new(path))?;

        // Extract model config from GGUF metadata
        let config = self.extract_config(&gguf)?;

        // Load embedding table (FP16/FP32)
        self.embedding = self.load_fp_tensor(&gguf, "model.embed_tokens.weight", &config)?;

        // Load LM head (may share weights with embedding in some architectures)
        self.lm_head = if gguf.get_tensor("lm_head.weight").is_some() {
            self.load_fp_tensor(&gguf, "lm_head.weight", &config)?
        } else if gguf.get_tensor("output.weight").is_some() {
            self.load_fp_tensor(&gguf, "output.weight", &config)?
        } else {
            // Tied embeddings: copy embedding table
            self.embedding.clone()
        };

        // Load final norm
        self.final_norm_weight =
            self.load_fp_tensor(&gguf, "model.norm.weight", &config)?;

        // Load transformer layers
        self.layers = Vec::with_capacity(config.num_layers);
        for layer_idx in 0..config.num_layers {
            let layer = self.load_layer(&gguf, layer_idx, &config)?;
            self.layers.push(layer);
        }

        self.config = Some(config);
        self.loaded = true;
        self.model_path = path.to_string();

        Ok(())
    }

    /// Extract BitNetModelConfig from GGUF metadata.
    fn extract_config(&self, gguf: &GgufFile) -> Result<BitNetModelConfig> {
        let num_layers = gguf.layer_count().unwrap_or(28);
        let hidden_size = gguf.embedding_length().unwrap_or(4096);
        let num_attention_heads = gguf.head_count().unwrap_or(32);
        let num_kv_heads = gguf.head_count_kv().unwrap_or(8);
        let vocab_size = gguf.vocab_size().unwrap_or(151552);
        let max_context = gguf.context_length().unwrap_or(8192);
        let rope_theta = gguf.rope_freq_base().unwrap_or(10000.0);
        let intermediate_size = gguf.feed_forward_length().unwrap_or(11008);

        // Detect expert count from tensor names
        let num_experts = self.detect_expert_count(gguf).unwrap_or(8);

        // Detect active experts from metadata or default to 2
        let active_experts = gguf
            .metadata
            .get("model.expert_count_active")
            .or_else(|| gguf.metadata.get("llm.expert_used_count"))
            .and_then(|v| v.as_u64())
            .map(|v| v as usize)
            .unwrap_or(2);

        Ok(BitNetModelConfig {
            num_layers,
            hidden_size,
            num_experts,
            active_experts,
            intermediate_size,
            num_attention_heads,
            num_kv_heads,
            vocab_size,
            max_context,
            rope_theta,
        })
    }

    /// Detect the number of MoE experts by scanning tensor names.
    fn detect_expert_count(&self, gguf: &GgufFile) -> Option<usize> {
        let mut max_expert_idx = 0usize;
        let mut found_any = false;

        for tensor in &gguf.tensors {
            // Look for patterns like "experts.0.", "experts.7.", etc.
            if let Some(pos) = tensor.name.find("experts.") {
                let after = &tensor.name[pos + 8..];
                if let Some(dot) = after.find('.') {
                    if let Ok(idx) = after[..dot].parse::<usize>() {
                        max_expert_idx = max_expert_idx.max(idx);
                        found_any = true;
                    }
                }
            }
        }

        if found_any {
            Some(max_expert_idx + 1)
        } else {
            None
        }
    }

    /// Load an FP16/FP32 tensor from GGUF, returning FP32 data.
    fn load_fp_tensor(
        &self,
        gguf: &GgufFile,
        name: &str,
        _config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        match gguf.get_tensor(name) {
            Some(_) => gguf.load_tensor_f32(name),
            None => Err(RuvLLMError::NotFound(format!(
                "Required tensor not found: {}",
                name
            ))),
        }
    }

    /// Load a ternary tensor from GGUF (BitnetT158 or dequant + re-quantize).
    fn load_ternary_tensor(
        &self,
        gguf: &GgufFile,
        name: &str,
    ) -> Result<TernaryTensor> {
        let info = gguf
            .get_tensor(name)
            .ok_or_else(|| RuvLLMError::NotFound(format!("Tensor not found: {}", name)))?;

        if info.dtype == GgufQuantType::BitnetT158 {
            // Native ternary format: extract packed data and scales directly
            let raw = gguf.load_tensor_quantized(name)?;
            let num_elements = info.num_elements();
            let block_size = 256usize;
            let num_blocks = (num_elements + block_size - 1) / block_size;
            let type_size = 66usize; // 64 packed + 2 FP16 scale

            let mut packed_data = Vec::with_capacity(num_blocks * 64);
            let mut scales = Vec::with_capacity(num_blocks);

            for blk in 0..num_blocks {
                let offset = blk * type_size;
                if offset + type_size > raw.data.len() {
                    break;
                }
                packed_data.extend_from_slice(&raw.data[offset..offset + 64]);
                let scale_bits =
                    u16::from_le_bytes([raw.data[offset + 64], raw.data[offset + 65]]);
                scales.push(f16_to_f32(scale_bits));
            }

            let shape = if info.shape.len() == 2 {
                (info.shape[0], info.shape[1])
            } else {
                (1, num_elements)
            };

            Ok(TernaryTensor {
                packed_data,
                scales,
                shape,
                block_size,
            })
        } else {
            // Non-native format: dequantize to FP32, then quantize to ternary
            let fp32 = gguf.load_tensor_f32(name)?;
            let num_elements = fp32.len();
            let shape = if info.shape.len() == 2 {
                (info.shape[0], info.shape[1])
            } else {
                (1, num_elements)
            };

            let ptconfig = super::quantizer::PtBitnetConfig::default();
            super::quantizer::quantize_tensor(&fp32, shape, &ptconfig)
        }
    }

    /// Load a single transformer layer.
    fn load_layer(
        &self,
        gguf: &GgufFile,
        idx: usize,
        config: &BitNetModelConfig,
    ) -> Result<TransformerLayer> {
        let prefix = format!("model.layers.{}", idx);

        // Norm weights (FP16/FP32)
        let input_norm_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.input_layernorm.weight", prefix),
            config,
        )?;
        let post_attn_norm_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.post_attention_layernorm.weight", prefix),
            config,
        )?;

        // MoE router gate (FP16/FP32): [num_experts, hidden_size]
        let gate_weight = self.load_fp_tensor(
            gguf,
            &format!("{}.mlp.gate.weight", prefix),
            config,
        )?;

        // Expert FFN weights (ternary)
        let mut experts = Vec::with_capacity(config.num_experts);
        for expert_idx in 0..config.num_experts {
            let expert_prefix =
                format!("{}.mlp.experts.{}", prefix, expert_idx);

            let gate_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.gate_proj.weight", expert_prefix),
            )?;
            let up_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.up_proj.weight", expert_prefix),
            )?;
            let down_proj = self.load_ternary_tensor(
                gguf,
                &format!("{}.down_proj.weight", expert_prefix),
            )?;

            experts.push(ExpertWeights {
                gate_proj,
                up_proj,
                down_proj,
            });
        }

        Ok(TransformerLayer {
            input_norm_weight,
            post_attn_norm_weight,
            gate_weight,
            experts,
        })
    }

    // ========================================================================
    // Forward Pass
    // ========================================================================

    /// Run the full forward pass, returning logits for the last token.
    ///
    /// # Arguments
    ///
    /// * `token_ids` - Input token ID sequence
    ///
    /// # Returns
    ///
    /// Logits vector of length `vocab_size`
    pub fn forward(&self, token_ids: &[u32]) -> Result<Vec<f32>> {
        let config = self.config.as_ref().ok_or_else(|| {
            RuvLLMError::Model("No model loaded".to_string())
        })?;

        if token_ids.is_empty() {
            return Err(RuvLLMError::Model("Empty token sequence".to_string()));
        }

        let hidden = config.hidden_size;

        // Embedding lookup: take last token for single-token generation
        let last_token = *token_ids.last().unwrap() as usize;
        if last_token >= config.vocab_size {
            return Err(RuvLLMError::Model(format!(
                "Token ID {} exceeds vocab size {}",
                last_token, config.vocab_size
            )));
        }
        let mut hidden_states: Vec<f32> =
            self.embedding[last_token * hidden..(last_token + 1) * hidden].to_vec();

        // Transformer layers
        for (layer_idx, layer) in self.layers.iter().enumerate() {
            hidden_states = self.forward_layer(
                &hidden_states,
                layer,
                layer_idx,
                config,
            )?;
        }

        // Final RMSNorm
        rms_norm_inplace(&mut hidden_states, &self.final_norm_weight, 1e-6);

        // LM head: logits = hidden_states @ lm_head^T
        let logits = fp32_matvec_transposed(
            &self.lm_head,
            &hidden_states,
            config.vocab_size,
            hidden,
        );

        Ok(logits)
    }

    /// Forward pass through a single transformer layer.
    fn forward_layer(
        &self,
        input: &[f32],
        layer: &TransformerLayer,
        _layer_idx: usize,
        config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        let hidden = config.hidden_size;

        // --- Pre-attention norm ---
        let mut normed = input.to_vec();
        rms_norm_inplace(&mut normed, &layer.input_norm_weight, 1e-6);

        // --- Attention (Phase 0 placeholder: pass-through) ---
        // In Phase 1 this would compute Q/K/V projections, RoPE, and causal attention.
        let attn_out = normed;

        // --- Residual after attention ---
        let mut residual: Vec<f32> = input
            .iter()
            .zip(attn_out.iter())
            .map(|(r, a)| r + a)
            .collect();

        // --- Post-attention norm ---
        let mut normed_ffn = residual.clone();
        rms_norm_inplace(&mut normed_ffn, &layer.post_attn_norm_weight, 1e-6);

        // --- MoE routing ---
        let (expert_indices, expert_weights) =
            self.route_experts(&normed_ffn, &layer.gate_weight, config)?;

        // --- Expert forward + weighted sum ---
        let mut moe_output = vec![0.0f32; hidden];
        for (&eidx, &eweight) in expert_indices.iter().zip(expert_weights.iter()) {
            if eidx >= layer.experts.len() {
                return Err(RuvLLMError::Model(format!(
                    "Expert index {} out of bounds (layer has {} experts)",
                    eidx,
                    layer.experts.len()
                )));
            }
            let expert_out =
                self.expert_forward(&normed_ffn, &layer.experts[eidx], config)?;
            for (o, &e) in moe_output.iter_mut().zip(expert_out.iter()) {
                *o += eweight * e;
            }
        }

        // --- Residual after MoE ---
        for (r, &m) in residual.iter_mut().zip(moe_output.iter()) {
            *r += m;
        }

        Ok(residual)
    }

    // ========================================================================
    // MoE Router
    // ========================================================================

    /// Route hidden states to the top-K experts.
    ///
    /// Computes `scores = hidden_states @ gate_weight^T`, applies softmax,
    /// then selects the top-K experts with highest scores.
    ///
    /// # Returns
    ///
    /// Tuple of (expert_indices, expert_weights) both of length active_experts.
    fn route_experts(
        &self,
        hidden_states: &[f32],
        gate_weight: &[f32],
        config: &BitNetModelConfig,
    ) -> Result<(Vec<usize>, Vec<f32>)> {
        let num_experts = config.num_experts;
        let hidden = config.hidden_size;
        // Clamp top_k to num_experts to prevent selecting more experts than exist
        let top_k = config.active_experts.min(num_experts);

        if num_experts == 0 {
            return Ok((vec![], vec![]));
        }

        // Gate: scores[e] = dot(hidden_states, gate_weight[e])
        let mut scores = vec![0.0f32; num_experts];
        for e in 0..num_experts {
            let row_start = e * hidden;
            if row_start + hidden > gate_weight.len() {
                break;
            }
            let mut dot = 0.0f32;
            for j in 0..hidden {
                dot += hidden_states[j] * gate_weight[row_start + j];
            }
            scores[e] = dot;
        }

        // Softmax over expert scores
        softmax_inplace(&mut scores);

        // Top-K selection
        let mut indexed: Vec<(usize, f32)> =
            scores.iter().copied().enumerate().collect();
        indexed.sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

        let selected: Vec<(usize, f32)> = indexed.into_iter().take(top_k).collect();

        // Renormalize selected weights so they sum to 1
        let weight_sum: f32 = selected.iter().map(|(_, w)| w).sum();
        let norm_factor = if weight_sum > 1e-12 { 1.0 / weight_sum } else { 1.0 };

        let expert_indices: Vec<usize> = selected.iter().map(|(i, _)| *i).collect();
        let expert_weights: Vec<f32> =
            selected.iter().map(|(_, w)| w * norm_factor).collect();

        Ok((expert_indices, expert_weights))
    }

    // ========================================================================
    // Expert FFN (TL1 GEMV)
    // ========================================================================

    /// Forward pass through a single expert's SwiGLU FFN.
    ///
    /// Computes:
    /// ```text
    /// gate = TL1_GEMV(gate_proj, input)
    /// up   = TL1_GEMV(up_proj, input)
    /// hidden = silu(gate) * up
    /// output = TL1_GEMV(down_proj, hidden)
    /// ```
    fn expert_forward(
        &self,
        input: &[f32],
        expert: &ExpertWeights,
        config: &BitNetModelConfig,
    ) -> Result<Vec<f32>> {
        let intermediate = config.intermediate_size;
        let hidden = config.hidden_size;

        // gate_proj: [intermediate_size, hidden_size] @ input[hidden_size] -> [intermediate_size]
        let gate_out = self.tl1_gemv(&expert.gate_proj, input, intermediate, hidden);

        // up_proj: [intermediate_size, hidden_size] @ input[hidden_size] -> [intermediate_size]
        let up_out = self.tl1_gemv(&expert.up_proj, input, intermediate, hidden);

        // SiLU(gate) * up (element-wise)
        let mut fused = vec![0.0f32; intermediate];
        for i in 0..intermediate {
            let silu_val = gate_out[i] * sigmoid(gate_out[i]);
            fused[i] = silu_val * up_out[i];
        }

        // down_proj: [hidden_size, intermediate_size] @ fused[intermediate_size] -> [hidden_size]
        let output = self.tl1_gemv(&expert.down_proj, &fused, hidden, intermediate);

        Ok(output)
    }

    /// TL1 GEMV: ternary matrix-vector product using the pre-built lookup table.
    ///
    /// Computes `output[i] = sum_j(ternary_weight[i,j] * input[j]) * scale[block]`
    /// using addition/subtraction only (multiplication-free for the ternary part).
    ///
    /// The lookup table maps each packed byte to its four ternary values,
    /// eliminating per-element bit extraction from the inner loop.
    fn tl1_gemv(
        &self,
        weight: &TernaryTensor,
        input: &[f32],
        out_rows: usize,
        in_cols: usize,
    ) -> Vec<f32> {
        let block_size = weight.block_size;
        let mut output = vec![0.0f32; out_rows];

        // Each row of the weight matrix is a contiguous sequence of packed bytes.
        // packed bytes per row = ceil(in_cols / 4)
        let bytes_per_row = (in_cols + 3) / 4;
        // Number of scale entries per row
        let blocks_per_row = (in_cols + block_size - 1) / block_size;

        for row in 0..out_rows {
            let row_byte_offset = row * bytes_per_row;
            let row_scale_offset = row * blocks_per_row;
            let mut accum = 0.0f32;

            for blk in 0..blocks_per_row {
                let scale = weight
                    .scales
                    .get(row_scale_offset + blk)
                    .copied()
                    .unwrap_or(1.0);

                let blk_start_col = blk * block_size;
                let blk_end_col = (blk_start_col + block_size).min(in_cols);
                let mut block_accum = 0.0f32;

                // Process 4 elements at a time via LUT
                let mut c = blk_start_col;

                while c + 4 <= blk_end_col {
                    let byte_idx = row_byte_offset + c / 4;
                    if byte_idx >= weight.packed_data.len() {
                        break;
                    }
                    let packed_byte = weight.packed_data[byte_idx];
                    let ternary = &self.tl1_lut[packed_byte as usize];

                    // Accumulate: ternary[k] * input[c+k] for k=0..3
                    // Since ternary is {-1, 0, +1}, this is add/sub/skip
                    for k in 0..4 {
                        let t = ternary[k];
                        if t == 1 {
                            block_accum += input[c + k];
                        } else if t == -1 {
                            block_accum -= input[c + k];
                        }
                        // t == 0: skip (multiplication-free)
                    }
                    c += 4;
                }

                // Handle tail elements (< 4 remaining in block)
                while c < blk_end_col {
                    let byte_idx = row_byte_offset + c / 4;
                    let bit_pos = c % 4;
                    if byte_idx < weight.packed_data.len() {
                        let t = self.tl1_lut[weight.packed_data[byte_idx] as usize][bit_pos];
                        if t == 1 {
                            block_accum += input[c];
                        } else if t == -1 {
                            block_accum -= input[c];
                        }
                    }
                    c += 1;
                }

                accum += block_accum * scale;
            }

            output[row] = accum;
        }

        output
    }

    /// Greedy-decode a single next token from logits.
    fn argmax(logits: &[f32]) -> u32 {
        let mut best_idx = 0u32;
        let mut best_val = f32::NEG_INFINITY;
        for (i, &v) in logits.iter().enumerate() {
            if v > best_val {
                best_val = v;
                best_idx = i as u32;
            }
        }
        best_idx
    }
}

// ============================================================================
// LlmBackend Trait Implementation
// ============================================================================

impl LlmBackend for BitNetBackend {
    fn load_model(&mut self, model_id: &str, _config: ModelConfig) -> Result<()> {
        self.load_gguf(model_id)
    }

    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        if !self.loaded {
            return Err(RuvLLMError::Model("No model loaded".to_string()));
        }

        // Phase 0: simple greedy decode with hardcoded token IDs
        // A real implementation would use the tokenizer to encode the prompt.
        // For smoke testing, treat prompt bytes as token IDs.
        let mut tokens: Vec<u32> = prompt.bytes().map(|b| b as u32).collect();
        let mut generated = Vec::new();

        for _ in 0..params.max_tokens {
            let logits = self.forward(&tokens)?;
            let next_token = Self::argmax(&logits);

            // Simple EOS check (token 0 or 2 are common EOS)
            if next_token == 0 || next_token == 2 {
                break;
            }

            generated.push(next_token);
            tokens.push(next_token);
        }

        // Phase 0: return token IDs as string (no real tokenizer)
        let text: String = generated
            .iter()
            .map(|&t| format!("[{}]", t))
            .collect::<Vec<_>>()
            .join("");

        Ok(text)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
        // Delegate to non-streaming generate for Phase 0
        let result = self.generate(prompt, params)?;
        let tokens: Vec<Result<GeneratedToken>> = result
            .chars()
            .enumerate()
            .map(|(i, c)| {
                Ok(GeneratedToken {
                    id: i as u32,
                    text: c.to_string(),
                    logprob: None,
                    is_special: false,
                })
            })
            .collect();
        Ok(Box::new(tokens.into_iter()))
    }

    fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream> {
        let (tx, stream) = TokenStream::channel();
        let result = self.generate(prompt, params.clone());

        match result {
            Ok(text) => {
                let _ = tx.send(StreamEvent::Token(GeneratedToken {
                    id: 0,
                    text,
                    logprob: None,
                    is_special: false,
                }));
                let _ = tx.send(StreamEvent::Done {
                    total_tokens: 1,
                    duration_ms: 0,
                    tokens_per_second: 0.0,
                });
            }
            Err(e) => {
                let _ = tx.send(StreamEvent::Error(e.to_string()));
            }
        }

        Ok(stream)
    }

    fn get_embeddings(&self, _text: &str) -> Result<Vec<f32>> {
        Err(RuvLLMError::NotImplemented(
            "BitNetBackend embeddings not yet supported".to_string(),
        ))
    }

    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        None // Phase 0: no tokenizer
    }

    fn is_model_loaded(&self) -> bool {
        self.loaded
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let config = self.config.as_ref()?;
        Some(ModelInfo {
            name: self.model_path.clone(),
            architecture: ModelArchitecture::Qwen, // Closest match for GLM-style MoE
            num_parameters: config.num_layers
                * config.num_experts
                * config.intermediate_size
                * config.hidden_size
                * 3, // rough estimate
            vocab_size: config.vocab_size,
            hidden_size: config.hidden_size,
            num_layers: config.num_layers,
            max_context_length: config.max_context,
            quantization: Some(Quantization::Q2K), // ~2 bits/weight
            memory_usage: self.embedding.len() * 4
                + self.lm_head.len() * 4
                + self
                    .layers
                    .iter()
                    .map(|l| {
                        l.gate_weight.len() * 4
                            + l.input_norm_weight.len() * 4
                            + l.post_attn_norm_weight.len() * 4
                            + l.experts
                                .iter()
                                .map(|e| {
                                    e.gate_proj.memory_bytes()
                                        + e.up_proj.memory_bytes()
                                        + e.down_proj.memory_bytes()
                                })
                                .sum::<usize>()
                    })
                    .sum::<usize>(),
        })
    }

    fn unload_model(&mut self) {
        self.config = None;
        self.embedding.clear();
        self.lm_head.clear();
        self.final_norm_weight.clear();
        self.layers.clear();
        self.loaded = false;
        self.model_path.clear();
    }
}

// ============================================================================
// Math Helpers (standalone functions used by the backend)
// ============================================================================

/// In-place RMSNorm: x = x / rms(x) * weight
fn rms_norm_inplace(x: &mut [f32], weight: &[f32], eps: f32) {
    let n = x.len();
    let mut sum_sq = 0.0f32;
    for &v in x.iter() {
        sum_sq += v * v;
    }
    let rms = (sum_sq / n as f32 + eps).sqrt();
    let inv_rms = 1.0 / rms;

    for i in 0..n {
        x[i] = x[i] * inv_rms * weight.get(i).copied().unwrap_or(1.0);
    }
}

/// In-place softmax.
///
/// Guards against NaN propagation: if all inputs are -inf or NaN,
/// the result is a uniform distribution (1/n for each element).
fn softmax_inplace(x: &mut [f32]) {
    if x.is_empty() {
        return;
    }

    let max_val = x.iter().cloned().fold(f32::NEG_INFINITY, f32::max);

    // Guard: if max_val is -inf or NaN, no valid scores exist.
    // Fall back to uniform distribution.
    if max_val.is_nan() || max_val.is_infinite() && max_val.is_sign_negative() {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }

    let mut sum_exp = 0.0f32;
    for v in x.iter_mut() {
        *v = (*v - max_val).exp();
        sum_exp += *v;
    }
    // Guard: if sum_exp is zero, NaN, or subnormal, fall back to uniform
    if !sum_exp.is_normal() || sum_exp <= 0.0 {
        let uniform = 1.0 / x.len() as f32;
        for v in x.iter_mut() {
            *v = uniform;
        }
        return;
    }
    for v in x.iter_mut() {
        *v /= sum_exp;
    }
}

/// Sigmoid activation.
#[inline(always)]
fn sigmoid(x: f32) -> f32 {
    1.0 / (1.0 + (-x).exp())
}

/// FP16 bits to FP32 conversion (same as in gguf/quantization.rs).
#[inline(always)]
fn f16_to_f32(bits: u16) -> f32 {
    let sign = ((bits & 0x8000) as u32) << 16;
    let exp = ((bits >> 10) & 0x1F) as u32;
    let frac = (bits & 0x03FF) as u32;

    if exp == 0 {
        if frac == 0 {
            return f32::from_bits(sign);
        }
        let mut e = 1u32;
        let mut f = frac;
        while (f & 0x0400) == 0 {
            f <<= 1;
            e += 1;
        }
        f &= 0x03FF;
        return f32::from_bits(sign | ((127 - 15 + 1 - e) << 23) | (f << 13));
    }

    if exp == 31 {
        return f32::from_bits(sign | 0x7F80_0000 | (frac << 13));
    }

    f32::from_bits(sign | ((exp + 127 - 15) << 23) | (frac << 13))
}

/// FP32 matrix-vector product (transposed): out[i] = dot(mat[i*cols..], vec)
///
/// mat is [rows, cols] row-major, vec is [cols], out is [rows].
fn fp32_matvec_transposed(mat: &[f32], vec: &[f32], rows: usize, cols: usize) -> Vec<f32> {
    let mut output = vec![0.0f32; rows];
    for i in 0..rows {
        let row_start = i * cols;
        if row_start + cols > mat.len() {
            break;
        }
        let mut dot = 0.0f32;
        for j in 0..cols {
            dot += mat[row_start + j] * vec[j];
        }
        output[i] = dot;
    }
    output
}

// ============================================================================
// Tests
// ============================================================================

#[cfg(test)]
mod tests {
    use super::*;
    use crate::bitnet::{pack_ternary, TernaryTensor};

    #[test]
    fn test_build_tl1_lut() {
        let lut = build_tl1_lut();

        // Byte 0x00 = all bits 00 = all -1
        assert_eq!(lut[0x00], [-1, -1, -1, -1]);

        // Byte 0x55 = 01_01_01_01 = all 0
        assert_eq!(lut[0x55], [0, 0, 0, 0]);

        // Byte 0xAA = 10_10_10_10 = all +1
        assert_eq!(lut[0xAA], [1, 1, 1, 1]);

        // Byte 0x24 = 00_10_01_00 => positions: [00, 01, 10, 00] => [-1, 0, 1, -1]
        // bit layout LSB first: bits[0:1]=00, bits[2:3]=01, bits[4:5]=10, bits[6:7]=00
        // 0x24 = 0b00_10_01_00
        assert_eq!(lut[0x24], [-1, 0, 1, -1]);
    }

    #[test]
    fn test_rms_norm_inplace() {
        let mut x = vec![1.0, 2.0, 3.0, 4.0];
        let w = vec![1.0; 4];
        rms_norm_inplace(&mut x, &w, 1e-6);

        // RMS of [1,2,3,4] = sqrt((1+4+9+16)/4) = sqrt(7.5) ≈ 2.7386
        let rms = (30.0f32 / 4.0).sqrt();
        let expected: Vec<f32> = [1.0, 2.0, 3.0, 4.0]
            .iter()
            .map(|v| v / rms)
            .collect();

        for (a, b) in x.iter().zip(expected.iter()) {
            assert!((a - b).abs() < 1e-4, "got {} expected {}", a, b);
        }
    }

    #[test]
    fn test_softmax_inplace() {
        let mut x = vec![1.0, 2.0, 3.0];
        softmax_inplace(&mut x);

        // Sum should be 1.0
        let sum: f32 = x.iter().sum();
        assert!((sum - 1.0).abs() < 1e-6);

        // Values should be ordered
        assert!(x[0] < x[1]);
        assert!(x[1] < x[2]);
    }

    #[test]
    fn test_sigmoid() {
        assert!((sigmoid(0.0) - 0.5).abs() < 1e-6);
        assert!(sigmoid(10.0) > 0.999);
        assert!(sigmoid(-10.0) < 0.001);
    }

    #[test]
    fn test_fp32_matvec_transposed() {
        // Identity matrix 3x3
        let mat = vec![
            1.0, 0.0, 0.0,
            0.0, 1.0, 0.0,
            0.0, 0.0, 1.0,
        ];
        let vec_in = vec![2.0, 3.0, 4.0];
        let out = fp32_matvec_transposed(&mat, &vec_in, 3, 3);
        assert_eq!(out, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_tl1_gemv_simple() {
        let backend = BitNetBackend::new();

        // Create a 2x4 ternary weight matrix:
        // Row 0: [+1, +1, +1, +1]
        // Row 1: [-1, -1, -1, -1]
        let row0 = vec![1i8, 1, 1, 1];
        let row1 = vec![-1i8, -1, -1, -1];
        let mut all = row0.clone();
        all.extend_from_slice(&row1);
        let packed = pack_ternary(&all);

        let weight = TernaryTensor {
            packed_data: packed,
            scales: vec![1.0, 1.0], // one scale per block (each row < 256, so 1 block per row)
            shape: (2, 4),
            block_size: 256,
        };

        let input = vec![1.0, 2.0, 3.0, 4.0];
        let output = backend.tl1_gemv(&weight, &input, 2, 4);

        // Row 0: 1+2+3+4 = 10, scale=1.0
        assert!((output[0] - 10.0).abs() < 1e-6);
        // Row 1: -(1+2+3+4) = -10, scale=1.0
        assert!((output[1] - (-10.0)).abs() < 1e-6);
    }

    #[test]
    fn test_tl1_gemv_with_zeros() {
        let backend = BitNetBackend::new();

        // Row: [+1, 0, -1, 0]
        let vals = vec![1i8, 0, -1, 0];
        let packed = pack_ternary(&vals);

        let weight = TernaryTensor {
            packed_data: packed,
            scales: vec![2.0],
            shape: (1, 4),
            block_size: 256,
        };

        let input = vec![5.0, 3.0, 7.0, 9.0];
        let output = backend.tl1_gemv(&weight, &input, 1, 4);

        // Result: (5.0 + 0 - 7.0 + 0) * 2.0 = -2.0 * 2.0 = -4.0
        assert!((output[0] - (-4.0)).abs() < 1e-6);
    }

    #[test]
    fn test_bitnet_model_config_default() {
        let config = BitNetModelConfig::default();
        assert_eq!(config.num_layers, 28);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.num_experts, 8);
        assert_eq!(config.active_experts, 2);
    }

    #[test]
    fn test_route_experts_topk() {
        let backend = BitNetBackend::new();
        let config = BitNetModelConfig {
            num_experts: 4,
            active_experts: 2,
            hidden_size: 4,
            ..Default::default()
        };

        // Gate weight [4 experts, 4 hidden]: identity-like so expert scores = hidden_states
        let gate_weight = vec![
            1.0, 0.0, 0.0, 0.0, // Expert 0 looks at dim 0
            0.0, 1.0, 0.0, 0.0, // Expert 1 looks at dim 1
            0.0, 0.0, 1.0, 0.0, // Expert 2 looks at dim 2
            0.0, 0.0, 0.0, 1.0, // Expert 3 looks at dim 3
        ];

        // Hidden states: dim 2 is highest, dim 3 is second
        let hidden = vec![0.1, 0.2, 0.9, 0.5];

        let (indices, weights) = backend
            .route_experts(&hidden, &gate_weight, &config)
            .unwrap();

        assert_eq!(indices.len(), 2);
        assert_eq!(weights.len(), 2);

        // Expert 2 should be first (score 0.9), Expert 3 second (score 0.5)
        assert_eq!(indices[0], 2);
        assert_eq!(indices[1], 3);

        // Weights should sum to ~1.0
        let wsum: f32 = weights.iter().sum();
        assert!((wsum - 1.0).abs() < 1e-4);
    }

    #[test]
    fn test_backend_new_unloaded() {
        let backend = BitNetBackend::new();
        assert!(!backend.is_model_loaded());
        assert!(backend.model_info().is_none());
    }
}
