//! Integration tests for LLM model weight decompilation.
//!
//! These tests create minimal valid GGUF and Safetensors files inline
//! to verify the full decompilation pipeline without external models.

#![cfg(feature = "model")]

use std::collections::HashMap;
use std::io::Write;
use std::path::PathBuf;

use ruvector_decompiler::model_decompiler;
use ruvector_decompiler::model_types::*;

// ── Helpers for building test files ──────────────────────────────────────

/// Build a minimal valid GGUF v3 file with LLaMA-like structure.
fn build_test_gguf() -> Vec<u8> {
    let mut buf = Vec::new();

    // Header: magic, version, tensor_count, metadata_count
    buf.extend_from_slice(&0x46554747u32.to_le_bytes()); // GGUF magic
    buf.extend_from_slice(&3u32.to_le_bytes()); // version 3
    buf.extend_from_slice(&5u64.to_le_bytes()); // 5 tensors
    buf.extend_from_slice(&4u64.to_le_bytes()); // 4 metadata entries

    // Metadata 1: general.architecture = "llama"
    write_kv_string(&mut buf, "general.architecture", "llama");
    // Metadata 2: llama.embedding_length = 256
    write_kv_u32(&mut buf, "llama.embedding_length", 256);
    // Metadata 3: llama.block_count = 2
    write_kv_u32(&mut buf, "llama.block_count", 2);
    // Metadata 4: llama.attention.head_count = 4
    write_kv_u32(&mut buf, "llama.attention.head_count", 4);

    // Tensor 1: token_embd.weight [512, 256] F32
    write_tensor_info(&mut buf, "token_embd.weight", &[512, 256], 0, 0);
    // Tensor 2: blk.0.attn_q.weight [256, 256] F32
    write_tensor_info(&mut buf, "blk.0.attn_q.weight", &[256, 256], 0, 524288);
    // Tensor 3: blk.0.attn_k.weight [64, 256] F32
    write_tensor_info(&mut buf, "blk.0.attn_k.weight", &[64, 256], 0, 786432);
    // Tensor 4: blk.0.ffn_up.weight [1024, 256] F32
    write_tensor_info(&mut buf, "blk.0.ffn_up.weight", &[1024, 256], 0, 851968);
    // Tensor 5: blk.1.attn_q.weight [256, 256] F32
    write_tensor_info(&mut buf, "blk.1.attn_q.weight", &[256, 256], 0, 1900544);

    buf
}

fn write_kv_string(buf: &mut Vec<u8>, key: &str, value: &str) {
    // Key string: length (u64) + bytes
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    // Value type: 8 = String
    buf.extend_from_slice(&8u32.to_le_bytes());
    // Value string: length (u64) + bytes
    buf.extend_from_slice(&(value.len() as u64).to_le_bytes());
    buf.extend_from_slice(value.as_bytes());
}

fn write_kv_u32(buf: &mut Vec<u8>, key: &str, value: u32) {
    buf.extend_from_slice(&(key.len() as u64).to_le_bytes());
    buf.extend_from_slice(key.as_bytes());
    // Value type: 4 = U32
    buf.extend_from_slice(&4u32.to_le_bytes());
    buf.extend_from_slice(&value.to_le_bytes());
}

fn write_tensor_info(buf: &mut Vec<u8>, name: &str, shape: &[u64], dtype: u32, offset: u64) {
    // Name string
    buf.extend_from_slice(&(name.len() as u64).to_le_bytes());
    buf.extend_from_slice(name.as_bytes());
    // n_dims
    buf.extend_from_slice(&(shape.len() as u32).to_le_bytes());
    // Shape dimensions
    for dim in shape {
        buf.extend_from_slice(&dim.to_le_bytes());
    }
    // dtype
    buf.extend_from_slice(&dtype.to_le_bytes());
    // offset
    buf.extend_from_slice(&offset.to_le_bytes());
}

fn write_test_file(name: &str, data: &[u8]) -> PathBuf {
    let path = std::env::temp_dir().join(name);
    let mut f = std::fs::File::create(&path).unwrap();
    f.write_all(data).unwrap();
    path
}

// ── GGUF tests ───────────────────────────────────────────────────────────

#[test]
fn test_decompile_gguf_basic() {
    let data = build_test_gguf();
    let path = write_test_file("test_decompile.gguf", &data);

    let result = model_decompiler::decompile_gguf(&path).unwrap();

    // Format check
    match &result.format {
        ModelFormat::Gguf { version } => assert_eq!(*version, 3),
        _ => panic!("expected GGUF format"),
    }

    // Architecture
    assert_eq!(result.architecture.name, "llama");
    assert_eq!(result.architecture.hidden_size, 256);
    assert_eq!(result.architecture.num_layers, 2);
    assert_eq!(result.architecture.num_heads, 4);
    assert_eq!(result.architecture.vocab_size, 512);

    // KV heads: K proj is [64, 256], head_dim = 256/4 = 64, so kv_heads = 64/64 = 1
    assert_eq!(result.architecture.num_kv_heads, 1);

    // FFN size from ffn_up: [1024, 256] -> intermediate = 1024
    assert_eq!(result.architecture.intermediate_size, 1024);

    // Total params
    let expected_params = 512 * 256 + 256 * 256 + 64 * 256 + 1024 * 256 + 256 * 256;
    assert_eq!(result.architecture.total_params, expected_params);

    // Layers should include embedding, attention, MLP
    assert!(!result.layers.is_empty());
    assert!(result
        .layers
        .iter()
        .any(|l| matches!(l.layer_type, LayerType::Embedding)));
    assert!(result
        .layers
        .iter()
        .any(|l| matches!(l.layer_type, LayerType::Attention { .. })));
    assert!(result
        .layers
        .iter()
        .any(|l| matches!(l.layer_type, LayerType::Mlp { .. })));

    // Witness chain
    assert!(!result.witness.source_hash.is_empty());
    assert!(!result.witness.chain_root.is_empty());
    assert!(!result.witness.module_witnesses.is_empty());

    // Metadata
    assert_eq!(
        result
            .metadata
            .get("general.architecture")
            .map(|s| s.as_str()),
        Some("llama")
    );

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_decompile_gguf_invalid_magic() {
    let mut data = vec![0u8; 24];
    // Wrong magic
    data[..4].copy_from_slice(&0xDEADBEEFu32.to_le_bytes());
    let path = write_test_file("test_bad_magic.gguf", &data);

    let result = model_decompiler::decompile_gguf(&path);
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("not a GGUF file"));

    let _ = std::fs::remove_file(&path);
}

// ── Safetensors tests ────────────────────────────────────────────────────

#[test]
fn test_decompile_safetensors_basic() {
    let header = serde_json::json!({
        "model.embed_tokens.weight": {
            "dtype": "F16",
            "shape": [32000, 4096],
            "data_offsets": [0, 262144000]
        },
        "model.layers.0.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [4096, 4096],
            "data_offsets": [262144000, 295698432]
        },
        "model.layers.0.self_attn.k_proj.weight": {
            "dtype": "F16",
            "shape": [1024, 4096],
            "data_offsets": [295698432, 304087040]
        },
        "model.layers.0.mlp.gate_proj.weight": {
            "dtype": "F16",
            "shape": [14336, 4096],
            "data_offsets": [304087040, 421527552]
        },
        "model.layers.1.self_attn.q_proj.weight": {
            "dtype": "F16",
            "shape": [4096, 4096],
            "data_offsets": [421527552, 455081984]
        },
        "__metadata__": {
            "format": "pt",
            "architecture": "llama"
        }
    });

    let header_bytes = serde_json::to_vec(&header).unwrap();
    let header_len = header_bytes.len() as u64;

    let mut file_bytes = Vec::new();
    file_bytes.extend_from_slice(&header_len.to_le_bytes());
    file_bytes.extend_from_slice(&header_bytes);

    let path = write_test_file("test_decompile.safetensors", &file_bytes);

    let result = model_decompiler::decompile_safetensors(&path).unwrap();

    match &result.format {
        ModelFormat::Safetensors => {}
        _ => panic!("expected Safetensors format"),
    }

    assert_eq!(result.architecture.name, "llama");
    assert_eq!(result.architecture.hidden_size, 4096);
    assert_eq!(result.architecture.num_layers, 2);
    assert_eq!(result.architecture.vocab_size, 32000);
    assert_eq!(result.architecture.num_heads, 32);
    // K proj [1024, 4096], head_dim=128, kv_heads = 1024/128 = 8
    assert_eq!(result.architecture.num_kv_heads, 8);
    assert_eq!(result.architecture.intermediate_size, 14336);

    // No tokenizer in safetensors
    assert!(result.tokenizer.is_none());

    // Witness chain
    assert!(!result.witness.source_hash.is_empty());

    let _ = std::fs::remove_file(&path);
}

// ── Auto-detect tests ────────────────────────────────────────────────────

#[test]
fn test_decompile_model_auto_detect_gguf() {
    let data = build_test_gguf();
    let path = write_test_file("test_auto.gguf", &data);

    let result = model_decompiler::decompile_model(&path).unwrap();
    assert!(matches!(result.format, ModelFormat::Gguf { .. }));

    let _ = std::fs::remove_file(&path);
}

#[test]
fn test_decompile_model_unsupported_ext() {
    let path = std::env::temp_dir().join("test_bad.xyz");
    std::fs::write(&path, b"garbage").unwrap();

    let result = model_decompiler::decompile_model(&path);
    assert!(result.is_err());
    assert!(format!("{}", result.unwrap_err()).contains("unsupported model format"));

    let _ = std::fs::remove_file(&path);
}

// ── Quantization tests ───────────────────────────────────────────────────

#[test]
fn test_quantization_detection() {
    // Build a GGUF with Q4_K tensors (type 12)
    let mut buf = Vec::new();
    buf.extend_from_slice(&0x46554747u32.to_le_bytes());
    buf.extend_from_slice(&3u32.to_le_bytes());
    buf.extend_from_slice(&2u64.to_le_bytes()); // 2 tensors
    buf.extend_from_slice(&1u64.to_le_bytes()); // 1 metadata

    write_kv_string(&mut buf, "general.architecture", "llama");

    // Tensor 1: token_embd.weight [100, 64] F16 (type 1)
    write_tensor_info(&mut buf, "token_embd.weight", &[100, 64], 1, 0);
    // Tensor 2: blk.0.attn_q.weight [64, 64] Q4_K (type 12)
    write_tensor_info(&mut buf, "blk.0.attn_q.weight", &[64, 64], 12, 12800);

    let path = write_test_file("test_quant.gguf", &buf);
    let result = model_decompiler::decompile_gguf(&path).unwrap();

    let quant = result.quantization.unwrap();
    assert_eq!(quant.method, "Q4_K");
    assert!(quant.bits_per_weight > 0.0);
    assert!(quant.compression_ratio > 1.0);

    let _ = std::fs::remove_file(&path);
}
