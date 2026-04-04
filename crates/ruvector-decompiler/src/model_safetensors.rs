//! Safetensors format parser for model decompilation.
//!
//! Safetensors is a simple format: 8-byte header length (LE u64) followed by
//! a JSON header mapping tensor names to `{ dtype, shape, data_offsets }`,
//! then raw tensor data. We only parse the JSON header.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::{DecompilerError, Result};
use crate::model_types::ModelTensorInfo;

/// Safetensors magic: first 8 bytes are the header length as u64 LE.
/// We validate the header length is reasonable (< 100MB).
const MAX_HEADER_SIZE: u64 = 100 * 1024 * 1024;

/// Parse a safetensors file and return tensor info from the JSON header.
pub(crate) fn parse_safetensors_file(
    path: &Path,
) -> Result<(HashMap<String, String>, Vec<ModelTensorInfo>)> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        DecompilerError::ModelError(format!("failed to open {}: {}", path.display(), e))
    })?;

    // Read 8-byte header length
    let mut len_bytes = [0u8; 8];
    file.read_exact(&mut len_bytes).map_err(|e| {
        DecompilerError::ModelError(format!("failed to read header length: {}", e))
    })?;
    let header_len = u64::from_le_bytes(len_bytes);

    if header_len > MAX_HEADER_SIZE {
        return Err(DecompilerError::ModelError(format!(
            "safetensors header too large: {} bytes",
            header_len
        )));
    }

    // Read JSON header
    let mut header_bytes = vec![0u8; header_len as usize];
    file.read_exact(&mut header_bytes).map_err(|e| {
        DecompilerError::ModelError(format!("failed to read header JSON: {}", e))
    })?;

    let header: HashMap<String, serde_json::Value> =
        serde_json::from_slice(&header_bytes).map_err(|e| {
            DecompilerError::ModelError(format!("invalid safetensors header JSON: {}", e))
        })?;

    let mut tensors = Vec::new();
    let mut metadata = HashMap::new();

    for (name, value) in &header {
        // The "__metadata__" key holds string metadata, not tensor info.
        if name == "__metadata__" {
            if let Some(obj) = value.as_object() {
                for (k, v) in obj {
                    if let Some(s) = v.as_str() {
                        metadata.insert(k.clone(), s.to_string());
                    }
                }
            }
            continue;
        }

        let obj = match value.as_object() {
            Some(o) => o,
            None => continue,
        };

        let dtype = obj
            .get("dtype")
            .and_then(|v| v.as_str())
            .unwrap_or("F32")
            .to_string();

        let shape: Vec<usize> = obj
            .get("shape")
            .and_then(|v| v.as_array())
            .map(|arr| {
                arr.iter()
                    .filter_map(|v| v.as_u64().map(|n| n as usize))
                    .collect()
            })
            .unwrap_or_default();

        let offset = obj
            .get("data_offsets")
            .and_then(|v| v.as_array())
            .and_then(|arr| arr.first())
            .and_then(|v| v.as_u64())
            .unwrap_or(0);

        let (quant_type, bits) = safetensors_dtype_info(&dtype);

        tensors.push(ModelTensorInfo {
            name: name.clone(),
            shape,
            quant_type,
            quant_name: dtype,
            bits_per_weight: bits,
            offset,
        });
    }

    // Sort tensors by offset for consistent ordering.
    tensors.sort_by_key(|t| t.offset);

    Ok((metadata, tensors))
}

/// Map safetensors dtype string to (quant_type_id, bits_per_weight).
fn safetensors_dtype_info(dtype: &str) -> (u32, f32) {
    match dtype {
        "F32" => (0, 32.0),
        "F16" => (1, 16.0),
        "BF16" => (29, 16.0),
        "F64" => (28, 64.0),
        "I8" => (24, 8.0),
        "I16" => (25, 16.0),
        "I32" => (26, 32.0),
        "I64" => (27, 64.0),
        "U8" => (24, 8.0),
        "BOOL" => (24, 8.0),
        _ => (0, 32.0),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_safetensors_dtype_info() {
        assert_eq!(safetensors_dtype_info("F16"), (1, 16.0));
        assert_eq!(safetensors_dtype_info("BF16"), (29, 16.0));
        assert_eq!(safetensors_dtype_info("F32"), (0, 32.0));
    }

    #[test]
    fn test_parse_minimal_safetensors() {
        // Build a minimal valid safetensors file in memory.
        let header = serde_json::json!({
            "weight": {
                "dtype": "F16",
                "shape": [4, 4],
                "data_offsets": [0, 32]
            },
            "__metadata__": {
                "format": "pt"
            }
        });
        let header_bytes = serde_json::to_vec(&header).unwrap();
        let header_len = header_bytes.len() as u64;

        let mut file_bytes = Vec::new();
        file_bytes.extend_from_slice(&header_len.to_le_bytes());
        file_bytes.extend_from_slice(&header_bytes);
        // Add some fake tensor data (32 bytes for 4x4 F16)
        file_bytes.extend_from_slice(&[0u8; 32]);

        // Write to temp file and parse
        let tmp = std::env::temp_dir().join("test_minimal.safetensors");
        std::fs::write(&tmp, &file_bytes).unwrap();

        let (metadata, tensors) = parse_safetensors_file(&tmp).unwrap();

        assert_eq!(metadata.get("format").map(|s| s.as_str()), Some("pt"));
        assert_eq!(tensors.len(), 1);
        assert_eq!(tensors[0].name, "weight");
        assert_eq!(tensors[0].shape, vec![4, 4]);
        assert_eq!(tensors[0].quant_name, "F16");

        let _ = std::fs::remove_file(&tmp);
    }
}
