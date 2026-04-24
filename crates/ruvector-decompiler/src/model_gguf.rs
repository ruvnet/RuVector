//! Self-contained GGUF v2/v3 parser for model decompilation.
//!
//! This is a minimal copy of the GGUF parsing logic from `ruvllm` to avoid
//! pulling in heavy transitive dependencies (candle, tokenizers, etc.).
//! See ADR-138 for the rationale.

use std::collections::HashMap;
use std::io::Read;
use std::path::Path;

use crate::error::{DecompilerError, Result};
use crate::model_types::ModelTensorInfo;

/// GGUF magic number: "GGUF" in little-endian.
pub(crate) const GGUF_MAGIC: u32 = 0x46554747;

// ── Metadata value type ──────────────────────────────────────────────────

/// GGUF metadata value (simplified for decompilation -- we only need
/// strings, integers, floats, bools, and arrays).
#[derive(Debug, Clone)]
pub(crate) enum GgufValue {
    U8(u8),
    I8(i8),
    U16(u16),
    I16(i16),
    U32(u32),
    I32(i32),
    U64(u64),
    I64(i64),
    F32(f32),
    F64(f64),
    Bool(bool),
    String(String),
    Array(Vec<GgufValue>),
}

impl GgufValue {
    pub fn as_str(&self) -> Option<&str> {
        match self {
            GgufValue::String(s) => Some(s),
            _ => None,
        }
    }

    pub fn as_u64(&self) -> Option<u64> {
        match self {
            GgufValue::U8(v) => Some(*v as u64),
            GgufValue::U16(v) => Some(*v as u64),
            GgufValue::U32(v) => Some(*v as u64),
            GgufValue::U64(v) => Some(*v),
            GgufValue::I8(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I16(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I32(v) if *v >= 0 => Some(*v as u64),
            GgufValue::I64(v) if *v >= 0 => Some(*v as u64),
            _ => None,
        }
    }

    pub fn as_array(&self) -> Option<&[GgufValue]> {
        match self {
            GgufValue::Array(arr) => Some(arr),
            _ => None,
        }
    }
}

// ── Quantization type names ──────────────────────────────────────────────

/// Map GGUF quant type ID to (name, bits_per_weight).
pub(crate) fn quant_type_info(id: u32) -> (&'static str, f32) {
    match id {
        0 => ("F32", 32.0),
        1 => ("F16", 16.0),
        2 => ("Q4_0", 4.5),
        3 => ("Q4_1", 5.0),
        4 => ("Q4_2", 4.5),
        5 => ("Q4_3", 5.0),
        6 => ("Q5_0", 5.5),
        7 => ("Q5_1", 6.0),
        8 => ("Q8_0", 8.5),
        9 => ("Q8_1", 9.0),
        10 => ("Q2_K", 2.56),
        11 => ("Q3_K", 3.44),
        12 => ("Q4_K", 4.5),
        13 => ("Q5_K", 5.5),
        14 => ("Q6_K", 6.56),
        15 => ("Q8_K", 8.5),
        16 => ("IQ2_XXS", 2.06),
        17 => ("IQ2_XS", 2.31),
        18 => ("IQ3_XXS", 3.06),
        19 => ("IQ1_S", 1.56),
        20 => ("IQ4_NL", 4.5),
        21 => ("IQ3_S", 3.44),
        22 => ("IQ2_S", 2.56),
        23 => ("IQ4_XS", 4.25),
        24 => ("I8", 8.0),
        25 => ("I16", 16.0),
        26 => ("I32", 32.0),
        27 => ("I64", 64.0),
        28 => ("F64", 64.0),
        29 => ("BF16", 16.0),
        30 => ("BitNet", 2.0),
        _ => ("Unknown", 0.0),
    }
}

// ── Parsing ──────────────────────────────────────────────────────────────

/// Parse a GGUF file and return metadata + tensor info.
pub(crate) fn parse_gguf_file(
    path: &Path,
) -> Result<(u32, HashMap<String, GgufValue>, Vec<ModelTensorInfo>)> {
    let mut file = std::fs::File::open(path).map_err(|e| {
        DecompilerError::ModelError(format!("failed to open {}: {}", path.display(), e))
    })?;

    // Header: magic (4) + version (4) + tensor_count (8) + metadata_count (8)
    let magic = read_u32(&mut file)?;
    if magic != GGUF_MAGIC {
        return Err(DecompilerError::ModelError(format!(
            "not a GGUF file (magic: 0x{:08x}, expected 0x{:08x})",
            magic, GGUF_MAGIC
        )));
    }

    let version = read_u32(&mut file)?;
    if version < 2 || version > 3 {
        return Err(DecompilerError::ModelError(format!(
            "unsupported GGUF version: {} (expected 2 or 3)",
            version
        )));
    }

    let tensor_count = read_u64(&mut file)?;
    let metadata_count = read_u64(&mut file)?;

    // Parse metadata
    let mut metadata = HashMap::with_capacity(metadata_count as usize);
    for _ in 0..metadata_count {
        let key = read_string(&mut file)?;
        let value = read_value(&mut file)?;
        metadata.insert(key, value);
    }

    // Parse tensor infos
    let mut tensors = Vec::with_capacity(tensor_count as usize);
    for _ in 0..tensor_count {
        let name = read_string(&mut file)?;
        let n_dims = read_u32(&mut file)? as usize;
        let mut shape = Vec::with_capacity(n_dims);
        for _ in 0..n_dims {
            shape.push(read_u64(&mut file)? as usize);
        }
        let quant_type_id = read_u32(&mut file)?;
        let offset = read_u64(&mut file)?;
        let (quant_name, bits) = quant_type_info(quant_type_id);

        tensors.push(ModelTensorInfo {
            name,
            shape,
            quant_type: quant_type_id,
            quant_name: quant_name.to_string(),
            bits_per_weight: bits,
            offset,
        });
    }

    Ok((version, metadata, tensors))
}

// ── Value reading ────────────────────────────────────────────────────────

const MAX_STRING_SIZE: usize = 65536;
const MAX_ARRAY_SIZE: usize = 10_000_000;

fn read_value<R: Read>(reader: &mut R) -> Result<GgufValue> {
    let type_id = read_u32(reader)?;
    match type_id {
        0 => Ok(GgufValue::U8(read_u8(reader)?)),
        1 => Ok(GgufValue::I8(read_u8(reader)? as i8)),
        2 => Ok(GgufValue::U16(read_u16(reader)?)),
        3 => Ok(GgufValue::I16(read_u16(reader)? as i16)),
        4 => Ok(GgufValue::U32(read_u32(reader)?)),
        5 => Ok(GgufValue::I32(read_u32(reader)? as i32)),
        6 => Ok(GgufValue::F32(read_f32(reader)?)),
        7 => Ok(GgufValue::Bool(read_u8(reader)? != 0)),
        8 => Ok(GgufValue::String(read_string(reader)?)),
        9 => read_array(reader),
        10 => Ok(GgufValue::U64(read_u64(reader)?)),
        11 => Ok(GgufValue::I64(read_u64(reader)? as i64)),
        12 => Ok(GgufValue::F64(read_f64(reader)?)),
        _ => Err(DecompilerError::ModelError(format!(
            "unknown GGUF value type: {}",
            type_id
        ))),
    }
}

fn read_array<R: Read>(reader: &mut R) -> Result<GgufValue> {
    let elem_type = read_u32(reader)?;
    let count = read_u64(reader)?;
    if count > MAX_ARRAY_SIZE as u64 {
        return Err(DecompilerError::ModelError(format!(
            "array too large: {} elements",
            count
        )));
    }
    let count = count as usize;
    let mut values = Vec::with_capacity(count);
    for _ in 0..count {
        let v = match elem_type {
            0 => GgufValue::U8(read_u8(reader)?),
            1 => GgufValue::I8(read_u8(reader)? as i8),
            2 => GgufValue::U16(read_u16(reader)?),
            3 => GgufValue::I16(read_u16(reader)? as i16),
            4 => GgufValue::U32(read_u32(reader)?),
            5 => GgufValue::I32(read_u32(reader)? as i32),
            6 => GgufValue::F32(read_f32(reader)?),
            7 => GgufValue::Bool(read_u8(reader)? != 0),
            8 => GgufValue::String(read_string(reader)?),
            9 => read_array(reader)?,
            10 => GgufValue::U64(read_u64(reader)?),
            11 => GgufValue::I64(read_u64(reader)? as i64),
            12 => GgufValue::F64(read_f64(reader)?),
            _ => {
                return Err(DecompilerError::ModelError(format!(
                    "unknown array element type: {}",
                    elem_type
                )))
            }
        };
        values.push(v);
    }
    Ok(GgufValue::Array(values))
}

// ── Primitive readers ────────────────────────────────────────────────────

fn read_err(e: std::io::Error) -> DecompilerError {
    DecompilerError::ModelError(format!("read error: {}", e))
}

fn read_u8<R: Read>(r: &mut R) -> Result<u8> {
    let mut buf = [0u8; 1];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(buf[0])
}

fn read_u16<R: Read>(r: &mut R) -> Result<u16> {
    let mut buf = [0u8; 2];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(u16::from_le_bytes(buf))
}

fn read_u32<R: Read>(r: &mut R) -> Result<u32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(u32::from_le_bytes(buf))
}

fn read_u64<R: Read>(r: &mut R) -> Result<u64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(u64::from_le_bytes(buf))
}

fn read_f32<R: Read>(r: &mut R) -> Result<f32> {
    let mut buf = [0u8; 4];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(f32::from_le_bytes(buf))
}

fn read_f64<R: Read>(r: &mut R) -> Result<f64> {
    let mut buf = [0u8; 8];
    r.read_exact(&mut buf).map_err(read_err)?;
    Ok(f64::from_le_bytes(buf))
}

fn read_string<R: Read>(r: &mut R) -> Result<String> {
    let len = read_u64(r)? as usize;
    if len > MAX_STRING_SIZE {
        return Err(DecompilerError::ModelError(format!(
            "string too long: {} bytes",
            len
        )));
    }
    let mut buf = vec![0u8; len];
    r.read_exact(&mut buf).map_err(read_err)?;
    String::from_utf8(buf).map_err(|e| DecompilerError::ModelError(format!("invalid UTF-8: {}", e)))
}

// ── Tests ────────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use std::io::Cursor;

    #[test]
    fn test_quant_type_info() {
        assert_eq!(quant_type_info(0).0, "F32");
        assert_eq!(quant_type_info(1).0, "F16");
        assert_eq!(quant_type_info(12).0, "Q4_K");
        assert_eq!(quant_type_info(999).0, "Unknown");
    }

    #[test]
    fn test_read_primitives() {
        let data = 42u32.to_le_bytes();
        let mut cursor = Cursor::new(data);
        assert_eq!(read_u32(&mut cursor).unwrap(), 42);
    }

    #[test]
    fn test_read_string() {
        let mut data = Vec::new();
        data.extend_from_slice(&5u64.to_le_bytes());
        data.extend_from_slice(b"hello");
        let mut cursor = Cursor::new(data);
        assert_eq!(read_string(&mut cursor).unwrap(), "hello");
    }

    #[test]
    fn test_read_string_too_long() {
        let mut data = Vec::new();
        data.extend_from_slice(&(MAX_STRING_SIZE as u64 + 1).to_le_bytes());
        let mut cursor = Cursor::new(data);
        assert!(read_string(&mut cursor).is_err());
    }

    #[test]
    fn test_gguf_value_as_str() {
        let v = GgufValue::String("llama".to_string());
        assert_eq!(v.as_str(), Some("llama"));
        let v = GgufValue::U32(42);
        assert_eq!(v.as_str(), None);
    }

    #[test]
    fn test_gguf_value_as_u64() {
        let v = GgufValue::U32(42);
        assert_eq!(v.as_u64(), Some(42));
        let v = GgufValue::I32(-1);
        assert_eq!(v.as_u64(), None);
    }
}
