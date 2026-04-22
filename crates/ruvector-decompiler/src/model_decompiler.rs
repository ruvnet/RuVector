//! LLM model weight decompiler.
//!
//! Analyzes GGUF and Safetensors weight files to reconstruct architecture,
//! tokenizer, layer topology, quantization, and witness chain provenance.
//!
//! See ADR-138 for design rationale.

use std::collections::HashMap;
use std::path::Path;

use sha3::{Digest, Sha3_256};

use crate::error::{DecompilerError, Result};
use crate::model_gguf::{self, GgufValue};
use crate::model_safetensors;
use crate::model_types::*;
use crate::types::WitnessChainData;

/// Decompile a model weight file (auto-detect format from extension).
pub fn decompile_model(path: &Path) -> Result<ModelDecompileResult> {
    let ext = path
        .extension()
        .and_then(|e| e.to_str())
        .unwrap_or("")
        .to_lowercase();

    match ext.as_str() {
        "gguf" => decompile_gguf(path),
        "safetensors" => decompile_safetensors(path),
        _ => Err(DecompilerError::ModelError(format!(
            "unsupported model format: .{} (expected .gguf or .safetensors)",
            ext
        ))),
    }
}

/// Decompile a GGUF model file.
pub fn decompile_gguf(path: &Path) -> Result<ModelDecompileResult> {
    let (version, metadata, tensors) = model_gguf::parse_gguf_file(path)?;

    let architecture = infer_architecture_from_gguf(&metadata, &tensors);
    let layers = extract_layers(&tensors, &architecture);
    let tokenizer = extract_tokenizer_from_gguf(&metadata);
    let quantization = detect_quantization(&tensors, &architecture);
    let witness = build_model_witness(path, &tensors);

    // Flatten metadata to string map for output.
    let metadata_strings = flatten_gguf_metadata(&metadata);

    Ok(ModelDecompileResult {
        format: ModelFormat::Gguf { version },
        architecture,
        layers,
        tokenizer,
        quantization,
        witness,
        metadata: metadata_strings,
    })
}

/// Decompile a Safetensors model file.
pub fn decompile_safetensors(path: &Path) -> Result<ModelDecompileResult> {
    let (raw_metadata, tensors) = model_safetensors::parse_safetensors_file(path)?;

    let architecture = infer_architecture_from_tensors(&raw_metadata, &tensors);
    let layers = extract_layers(&tensors, &architecture);
    let quantization = detect_quantization(&tensors, &architecture);
    let witness = build_model_witness(path, &tensors);

    Ok(ModelDecompileResult {
        format: ModelFormat::Safetensors,
        architecture,
        layers,
        tokenizer: None, // Safetensors rarely contains tokenizer
        quantization,
        witness,
        metadata: raw_metadata,
    })
}

// ── Architecture inference ───────────────────────────────────────────────

fn infer_architecture_from_gguf(
    metadata: &HashMap<String, GgufValue>,
    tensors: &[ModelTensorInfo],
) -> ModelArchitecture {
    // GGUF metadata keys for architecture.
    let name = metadata
        .get("general.architecture")
        .and_then(|v| v.as_str())
        .unwrap_or("unknown")
        .to_string();

    let arch_prefix = format!("{}.", name);

    let hidden_size = get_meta_u64(metadata, &format!("{}embedding_length", arch_prefix))
        .or_else(|| infer_hidden_size(tensors))
        .unwrap_or(0) as usize;

    let num_layers = get_meta_u64(metadata, &format!("{}block_count", arch_prefix))
        .or_else(|| Some(infer_num_layers(tensors) as u64))
        .unwrap_or(0) as usize;

    let num_heads = get_meta_u64(metadata, &format!("{}attention.head_count", arch_prefix))
        .unwrap_or(0) as usize;

    let num_kv_heads = get_meta_u64(metadata, &format!("{}attention.head_count_kv", arch_prefix))
        .or_else(|| infer_kv_heads(tensors, hidden_size, num_heads))
        .unwrap_or(num_heads as u64) as usize;

    let intermediate_size = get_meta_u64(metadata, &format!("{}feed_forward_length", arch_prefix))
        .or_else(|| infer_ffn_size(tensors))
        .unwrap_or(0) as usize;

    let vocab_size = infer_vocab_size(tensors).unwrap_or(0);

    let max_seq_len =
        get_meta_u64(metadata, &format!("{}context_length", arch_prefix)).unwrap_or(0) as usize;

    let total_params: usize = tensors.iter().map(|t| t.num_elements()).sum();
    let estimated_size_mb = (total_params as f64 * 2.0) / (1024.0 * 1024.0);

    ModelArchitecture {
        name,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        vocab_size,
        max_sequence_length: max_seq_len,
        total_params,
        estimated_size_mb,
    }
}

fn infer_architecture_from_tensors(
    metadata: &HashMap<String, String>,
    tensors: &[ModelTensorInfo],
) -> ModelArchitecture {
    let name = metadata
        .get("architecture")
        .or_else(|| metadata.get("model_type"))
        .cloned()
        .unwrap_or_else(|| infer_arch_name_from_tensor_names(tensors));

    let hidden_size = infer_hidden_size(tensors).unwrap_or(0) as usize;
    let num_layers = infer_num_layers(tensors);
    let num_heads = infer_num_heads(tensors, hidden_size);
    let num_kv_heads =
        infer_kv_heads(tensors, hidden_size, num_heads).unwrap_or(num_heads as u64) as usize;
    let intermediate_size = infer_ffn_size(tensors).unwrap_or(0) as usize;
    let vocab_size = infer_vocab_size(tensors).unwrap_or(0);
    let total_params: usize = tensors.iter().map(|t| t.num_elements()).sum();
    let estimated_size_mb = (total_params as f64 * 2.0) / (1024.0 * 1024.0);

    ModelArchitecture {
        name,
        hidden_size,
        num_layers,
        num_heads,
        num_kv_heads,
        intermediate_size,
        vocab_size,
        max_sequence_length: 0,
        total_params,
        estimated_size_mb,
    }
}

// ── Tensor shape analysis helpers ────────────────────────────────────────

fn get_meta_u64(metadata: &HashMap<String, GgufValue>, key: &str) -> Option<u64> {
    metadata.get(key).and_then(|v| v.as_u64())
}

fn infer_hidden_size(tensors: &[ModelTensorInfo]) -> Option<u64> {
    // Look for embedding tensor: shape [vocab_size, hidden_size]
    for t in tensors {
        if (t.name.contains("embed") || t.name.contains("token_embd")) && t.shape.len() == 2 {
            return Some(t.shape[1] as u64);
        }
    }
    // Fall back to attention Q projection: shape [hidden, hidden]
    for t in tensors {
        if (t.name.contains("attn_q") || t.name.contains(".q_proj")) && t.shape.len() == 2 {
            return Some(t.shape[1] as u64);
        }
    }
    None
}

fn infer_num_layers(tensors: &[ModelTensorInfo]) -> usize {
    let mut max_idx: i64 = -1;
    for t in tensors {
        // Match "blk.N.", "layers.N.", "h.N.", "model.layers.N."
        if let Some(idx) = extract_layer_index(&t.name) {
            if idx as i64 > max_idx {
                max_idx = idx as i64;
            }
        }
    }
    if max_idx >= 0 {
        (max_idx + 1) as usize
    } else {
        0
    }
}

fn extract_layer_index(name: &str) -> Option<usize> {
    // Common patterns: "blk.0.", "layers.0.", "h.0.", "model.layers.0."
    for prefix in &["blk.", "layers.", "h.", "model.layers."] {
        if let Some(rest) = name
            .strip_prefix(prefix)
            .or_else(|| name.find(prefix).map(|i| &name[i + prefix.len()..]))
        {
            if let Some(dot) = rest.find('.') {
                if let Ok(idx) = rest[..dot].parse::<usize>() {
                    return Some(idx);
                }
            }
        }
    }
    None
}

fn infer_num_heads(tensors: &[ModelTensorInfo], hidden_size: usize) -> usize {
    if hidden_size == 0 {
        return 0;
    }
    // Standard head_dim = 128 for most modern models
    for head_dim in &[128, 64, 96, 256] {
        if hidden_size % head_dim == 0 {
            return hidden_size / head_dim;
        }
    }
    0
}

fn infer_kv_heads(
    tensors: &[ModelTensorInfo],
    hidden_size: usize,
    num_heads: usize,
) -> Option<u64> {
    if hidden_size == 0 || num_heads == 0 {
        return None;
    }
    let head_dim = hidden_size / num_heads;
    // Look for K projection tensor shape: [kv_heads * head_dim, hidden_size]
    for t in tensors {
        if (t.name.contains("attn_k") || t.name.contains(".k_proj")) && t.shape.len() == 2 {
            let k_dim = t.shape[0];
            if head_dim > 0 && k_dim % head_dim == 0 {
                return Some((k_dim / head_dim) as u64);
            }
        }
    }
    None
}

fn infer_ffn_size(tensors: &[ModelTensorInfo]) -> Option<u64> {
    for t in tensors {
        if (t.name.contains("ffn_up")
            || t.name.contains(".up_proj")
            || t.name.contains("ffn_gate")
            || t.name.contains(".gate_proj"))
            && t.shape.len() == 2
        {
            return Some(t.shape[0] as u64);
        }
    }
    None
}

fn infer_vocab_size(tensors: &[ModelTensorInfo]) -> Option<usize> {
    for t in tensors {
        if (t.name.contains("embed") || t.name.contains("token_embd")) && t.shape.len() == 2 {
            return Some(t.shape[0]);
        }
    }
    None
}

fn infer_arch_name_from_tensor_names(tensors: &[ModelTensorInfo]) -> String {
    let any_name = tensors.first().map(|t| t.name.as_str()).unwrap_or("");
    if any_name.contains("model.layers") {
        if tensors.iter().any(|t| t.name.contains("gate_proj")) {
            "llama".to_string()
        } else {
            "transformer".to_string()
        }
    } else if any_name.contains("h.") {
        "gpt2".to_string()
    } else {
        "unknown".to_string()
    }
}

// ── Layer extraction ─────────────────────────────────────────────────────

fn extract_layers(tensors: &[ModelTensorInfo], arch: &ModelArchitecture) -> Vec<LayerInfo> {
    let mut layers = Vec::new();
    let head_dim = if arch.num_heads > 0 {
        arch.hidden_size / arch.num_heads
    } else {
        0
    };

    // Embedding layer
    let embed_tensors: Vec<&ModelTensorInfo> = tensors
        .iter()
        .filter(|t| {
            (t.name.contains("embed") || t.name.contains("token_embd"))
                && !t.name.contains("position")
                && extract_layer_index(&t.name).is_none()
        })
        .collect();
    if !embed_tensors.is_empty() {
        layers.push(LayerInfo {
            index: 0,
            layer_type: LayerType::Embedding,
            tensor_names: embed_tensors.iter().map(|t| t.name.clone()).collect(),
            param_count: embed_tensors.iter().map(|t| t.num_elements()).sum(),
            quantization: embed_tensors.first().map(|t| t.quant_name.clone()),
        });
    }

    // Per-block layers
    for blk in 0..arch.num_layers {
        let block_tensors: Vec<&ModelTensorInfo> = tensors
            .iter()
            .filter(|t| extract_layer_index(&t.name) == Some(blk))
            .collect();
        if block_tensors.is_empty() {
            continue;
        }

        // Attention tensors
        let attn: Vec<&ModelTensorInfo> = block_tensors
            .iter()
            .filter(|t| {
                t.name.contains("attn")
                    || t.name.contains("self_attn")
                    || t.name.contains("q_proj")
                    || t.name.contains("k_proj")
                    || t.name.contains("v_proj")
                    || t.name.contains("o_proj")
            })
            .copied()
            .collect();
        if !attn.is_empty() {
            layers.push(LayerInfo {
                index: blk,
                layer_type: LayerType::Attention {
                    heads: arch.num_heads,
                    kv_heads: arch.num_kv_heads,
                    head_dim,
                },
                tensor_names: attn.iter().map(|t| t.name.clone()).collect(),
                param_count: attn.iter().map(|t| t.num_elements()).sum(),
                quantization: attn.first().map(|t| t.quant_name.clone()),
            });
        }

        // MLP tensors
        let mlp: Vec<&ModelTensorInfo> = block_tensors
            .iter()
            .filter(|t| {
                t.name.contains("ffn")
                    || t.name.contains("mlp")
                    || t.name.contains("up_proj")
                    || t.name.contains("down_proj")
                    || t.name.contains("gate_proj")
            })
            .copied()
            .collect();
        if !mlp.is_empty() {
            layers.push(LayerInfo {
                index: blk,
                layer_type: LayerType::Mlp {
                    up_size: arch.intermediate_size,
                    down_size: arch.hidden_size,
                },
                tensor_names: mlp.iter().map(|t| t.name.clone()).collect(),
                param_count: mlp.iter().map(|t| t.num_elements()).sum(),
                quantization: mlp.first().map(|t| t.quant_name.clone()),
            });
        }

        // Norm tensors
        let norm: Vec<&ModelTensorInfo> = block_tensors
            .iter()
            .filter(|t| t.name.contains("norm"))
            .copied()
            .collect();
        for nt in &norm {
            let lt = if nt.name.contains("rms") {
                LayerType::RmsNorm
            } else {
                // Default: GGUF LLaMA uses "attn_norm" / "ffn_norm" which are RMSNorm
                LayerType::RmsNorm
            };
            layers.push(LayerInfo {
                index: blk,
                layer_type: lt,
                tensor_names: vec![nt.name.clone()],
                param_count: nt.num_elements(),
                quantization: Some(nt.quant_name.clone()),
            });
        }
    }

    // Output layer
    let output_tensors: Vec<&ModelTensorInfo> = tensors
        .iter()
        .filter(|t| {
            t.name.contains("output")
                && extract_layer_index(&t.name).is_none()
                && !t.name.contains("norm")
        })
        .collect();
    if !output_tensors.is_empty() {
        layers.push(LayerInfo {
            index: arch.num_layers,
            layer_type: LayerType::Output,
            tensor_names: output_tensors.iter().map(|t| t.name.clone()).collect(),
            param_count: output_tensors.iter().map(|t| t.num_elements()).sum(),
            quantization: output_tensors.first().map(|t| t.quant_name.clone()),
        });
    }

    layers
}

// ── Tokenizer extraction ─────────────────────────────────────────────────

fn extract_tokenizer_from_gguf(metadata: &HashMap<String, GgufValue>) -> Option<TokenizerInfo> {
    let vocab_size = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| arr.len())?;

    let mut special_tokens = Vec::new();
    // Common special token keys in GGUF.
    for key in &[
        "tokenizer.ggml.bos_token_id",
        "tokenizer.ggml.eos_token_id",
        "tokenizer.ggml.padding_token_id",
        "tokenizer.ggml.unknown_token_id",
    ] {
        if let Some(id) = metadata.get(*key).and_then(|v| v.as_u64()) {
            let name = key
                .strip_prefix("tokenizer.ggml.")
                .unwrap_or(key)
                .to_string();
            special_tokens.push((name, id as u32));
        }
    }

    // Sample first 100 tokens.
    let sample_tokens = metadata
        .get("tokenizer.ggml.tokens")
        .and_then(|v| v.as_array())
        .map(|arr| {
            arr.iter()
                .take(100)
                .enumerate()
                .filter_map(|(i, v)| v.as_str().map(|s| (i as u32, s.to_string())))
                .collect::<Vec<_>>()
        })
        .unwrap_or_default();

    Some(TokenizerInfo {
        vocab_size,
        special_tokens,
        sample_tokens,
    })
}

// ── Quantization detection ───────────────────────────────────────────────

fn detect_quantization(
    tensors: &[ModelTensorInfo],
    arch: &ModelArchitecture,
) -> Option<QuantizationInfo> {
    if tensors.is_empty() {
        return None;
    }

    // Find the most common quantization type (excluding embeddings/norms).
    let mut type_counts: HashMap<String, usize> = HashMap::new();
    let mut type_bits: HashMap<String, f32> = HashMap::new();
    for t in tensors {
        if t.name.contains("norm") || t.name.contains("embed") || t.name.contains("embd") {
            continue;
        }
        *type_counts.entry(t.quant_name.clone()).or_insert(0) += 1;
        type_bits.insert(t.quant_name.clone(), t.bits_per_weight);
    }

    let method = type_counts
        .iter()
        .max_by_key(|(_, count)| *count)
        .map(|(name, _)| name.clone())
        .unwrap_or_else(|| "Unknown".to_string());

    let bits_per_weight = type_bits.get(&method).copied().unwrap_or(0.0);
    let original_size_mb = arch.estimated_size_mb;

    // Calculate quantized size from actual tensor data.
    let total_bits: f64 = tensors
        .iter()
        .map(|t| t.num_elements() as f64 * t.bits_per_weight as f64)
        .sum();
    let quantized_size_mb = total_bits / 8.0 / (1024.0 * 1024.0);

    let compression_ratio = if quantized_size_mb > 0.0 {
        (original_size_mb / quantized_size_mb) as f32
    } else {
        1.0
    };

    Some(QuantizationInfo {
        method,
        bits_per_weight,
        original_size_mb,
        quantized_size_mb,
        compression_ratio,
    })
}

// ── Witness chain ────────────────────────────────────────────────────────

fn build_model_witness(path: &Path, tensors: &[ModelTensorInfo]) -> WitnessChainData {
    let path_str = path.display().to_string();
    let source_hash = sha3_hex(path_str.as_bytes());

    let module_witnesses = tensors
        .iter()
        .take(32) // Limit witness entries to avoid bloat.
        .map(|t| {
            let content = format!("{}:{:?}:{}", t.name, t.shape, t.quant_name);
            crate::types::ModuleWitnessData {
                module_name: t.name.clone(),
                byte_range: (t.offset as usize, 0),
                content_hash: sha3_hex(content.as_bytes()),
                inferred_names_hash: sha3_hex(t.quant_name.as_bytes()),
            }
        })
        .collect::<Vec<_>>();

    // Merkle root over witness entries.
    let chain_root = if module_witnesses.is_empty() {
        sha3_hex(b"empty")
    } else {
        let combined: String = module_witnesses
            .iter()
            .map(|w| w.content_hash.clone())
            .collect::<Vec<_>>()
            .join("");
        sha3_hex(combined.as_bytes())
    };

    WitnessChainData {
        source_hash,
        module_witnesses,
        chain_root,
    }
}

fn sha3_hex(data: &[u8]) -> String {
    let mut hasher = Sha3_256::new();
    hasher.update(data);
    let result = hasher.finalize();
    result.iter().map(|b| format!("{:02x}", b)).collect()
}

// ── Metadata flattening ──────────────────────────────────────────────────

fn flatten_gguf_metadata(metadata: &HashMap<String, GgufValue>) -> HashMap<String, String> {
    let mut out = HashMap::new();
    for (key, value) in metadata {
        let s = match value {
            GgufValue::String(s) => s.clone(),
            GgufValue::U32(v) => v.to_string(),
            GgufValue::U64(v) => v.to_string(),
            GgufValue::I32(v) => v.to_string(),
            GgufValue::I64(v) => v.to_string(),
            GgufValue::F32(v) => v.to_string(),
            GgufValue::F64(v) => v.to_string(),
            GgufValue::Bool(v) => v.to_string(),
            GgufValue::U8(v) => v.to_string(),
            GgufValue::I8(v) => v.to_string(),
            GgufValue::U16(v) => v.to_string(),
            GgufValue::I16(v) => v.to_string(),
            GgufValue::Array(arr) => format!("[{} elements]", arr.len()),
        };
        out.insert(key.clone(), s);
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_extract_layer_index() {
        assert_eq!(extract_layer_index("blk.0.attn_q.weight"), Some(0));
        assert_eq!(extract_layer_index("blk.31.ffn_up.weight"), Some(31));
        assert_eq!(
            extract_layer_index("model.layers.5.self_attn.q_proj"),
            Some(5)
        );
        assert_eq!(extract_layer_index("token_embd.weight"), None);
        assert_eq!(extract_layer_index("output.weight"), None);
    }

    #[test]
    fn test_infer_num_heads() {
        assert_eq!(infer_num_heads(&[], 4096), 32);
        assert_eq!(infer_num_heads(&[], 2048), 16);
        assert_eq!(infer_num_heads(&[], 0), 0);
    }

    #[test]
    fn test_sha3_hex() {
        let hash = sha3_hex(b"hello");
        assert_eq!(hash.len(), 64); // 32 bytes = 64 hex chars
    }

    #[test]
    fn test_infer_arch_name() {
        let tensors = vec![ModelTensorInfo {
            name: "model.layers.0.gate_proj.weight".to_string(),
            shape: vec![4096, 4096],
            quant_type: 0,
            quant_name: "F32".to_string(),
            bits_per_weight: 32.0,
            offset: 0,
        }];
        assert_eq!(infer_arch_name_from_tensor_names(&tensors), "llama");
    }

    #[test]
    fn test_detect_quantization_empty() {
        let arch = ModelArchitecture {
            name: "test".to_string(),
            hidden_size: 4096,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            intermediate_size: 14336,
            vocab_size: 128256,
            max_sequence_length: 8192,
            total_params: 8_000_000_000,
            estimated_size_mb: 15360.0,
        };
        assert!(detect_quantization(&[], &arch).is_none());
    }
}
