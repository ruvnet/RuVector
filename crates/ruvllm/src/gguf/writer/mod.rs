//! SafeTensors conversion to GGUF for supported llama and qwen2 checkpoints.
//!
//! Exact NFC normalization is accepted without changing BPE metadata, matching
//! llama.cpp's converter and runtime. Other normalizers remain unsupported.
//! Llama3 rotary scaling emits an F32 `rope_freqs.weight` tensor after model tensors.
//! Added tokens use the converter's CONTROL-token predicate. Q8_0 embeddings are
//! quantized by default; explicit F16 retention is available through [`WriterOptions`].

mod input;
mod metadata;

use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{self, BufWriter, Read, Write};
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};

use half::{bf16, f16};

use super::{GgufQuantType, GgufValue, DEFAULT_ALIGNMENT, GGUF_MAGIC, GGUF_VERSION};
use crate::error::{Result, RuvLLMError};
use input::{Checkpoint, SourceDtype, SourceTensor};
use metadata::ModelMetadata;

/// Supported conversion output types. Norms and one-dimensional tensors remain F32.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum OutputType {
    /// Half-precision floating-point matrices.
    F16,
    /// GGML Q8_0 blocks, with 32 values and a half-precision scale.
    Q8_0,
}

/// Output storage policy for SafeTensors conversion.
#[derive(Debug, Clone, Copy)]
pub struct WriterOptions {
    /// Matrix storage type.
    pub format: OutputType,
    /// Retain F16 token embeddings when writing Q8_0 (default: false).
    pub keep_embedding_f16: bool,
    /// Retain the F16 output projection when writing Q8_0.
    pub keep_output_f16: bool,
}

impl Default for WriterOptions {
    fn default() -> Self {
        Self {
            format: OutputType::F16,
            keep_embedding_f16: false,
            keep_output_f16: true,
        }
    }
}

/// Counts for a successfully published conversion.
#[derive(Debug, Clone)]
pub struct ConversionSummary {
    /// Number of tensors in the GGUF table.
    pub tensor_count: usize,
    /// Number of stored tensor elements.
    pub element_count: u64,
    /// Final GGUF byte length, including alignment padding.
    pub output_bytes: u64,
}

/// Convert a single SafeTensors file or indexed HuggingFace directory atomically.
///
/// Unsupported configuration, tokenizer profiles, tensor names and shapes are refused.
/// Existing output paths are refused. Failed writes remove the temporary file; abrupt
/// process termination before publication can leave a private `.part` file, never a
/// partially written file at the requested output path.
pub fn convert_safetensors(
    source: &Path,
    output: &Path,
    options: &WriterOptions,
) -> Result<ConversionSummary> {
    require_absent(output)?;
    let checkpoint = input::open(source)?;
    let mut metadata = metadata::load(&checkpoint.root)?;
    metadata.values.insert(
        "general.file_type".into(),
        GgufValue::U32(match options.format {
            OutputType::F16 => 1,
            OutputType::Q8_0 => 7,
        }),
    );
    metadata
        .values
        .insert("general.quantization_version".into(), GgufValue::U32(2));
    metadata.values.insert(
        "general.alignment".into(),
        GgufValue::U32(DEFAULT_ALIGNMENT as u32),
    );
    let tensors = plan_tensors(&checkpoint, &metadata, options)?;
    let header = make_header(&metadata.values, &tensors)?;
    let data_bytes = tensors.iter().try_fold(0_u64, |_, tensor| {
        align(
            tensor
                .offset
                .checked_add(tensor.byte_len)
                .ok_or_else(|| invalid("GGUF tensor data length overflow"))?,
        )
    })?;
    let element_count = tensors.iter().try_fold(0_u64, |sum, tensor| {
        sum.checked_add(tensor.elements)
            .ok_or_else(|| invalid("GGUF element count overflow"))
    })?;
    let output_bytes = (header.len() as u64)
        .checked_add(data_bytes)
        .ok_or_else(|| invalid("GGUF output length overflow"))?;
    publish(output, |file| {
        let mut sink = BufWriter::new(file);
        write_gguf(&mut sink, &header, &tensors, &checkpoint)?;
        sink.flush().map_err(io_error)
    })?;
    Ok(ConversionSummary {
        tensor_count: tensors.len(),
        element_count,
        output_bytes,
    })
}

enum TensorSource {
    Checkpoint(SourceTensor),
    Generated(Vec<f32>),
}

struct TensorPlan {
    source: TensorSource,
    shape: Vec<usize>,
    name: String,
    dtype: GgufQuantType,
    permutation_heads: Option<usize>,
    elements: u64,
    offset: u64,
    byte_len: u64,
}

fn invalid(message: impl Into<String>) -> RuvLLMError {
    RuvLLMError::Model(message.into())
}

fn io_error(error: io::Error) -> RuvLLMError {
    invalid(format!("GGUF output: {error}"))
}

fn align(value: u64) -> Result<u64> {
    let alignment = DEFAULT_ALIGNMENT as u64;
    value
        .checked_add(alignment - 1)
        .map(|n| n / alignment * alignment)
        .ok_or_else(|| invalid("GGUF alignment overflow"))
}

fn plan_tensors(
    checkpoint: &Checkpoint,
    model: &ModelMetadata,
    options: &WriterOptions,
) -> Result<Vec<TensorPlan>> {
    let mut mapped = BTreeMap::new();
    for (name, source) in &checkpoint.tensors {
        let (target, shape, permutation_heads) = map_tensor(name, model)?;
        if source.shape != shape {
            return Err(invalid(format!(
                "tensor {name} has shape {:?}, expected {shape:?}",
                source.shape
            )));
        }
        let elements = source.shape.iter().try_fold(1_u64, |n, &dimension| {
            n.checked_mul(dimension as u64)
                .ok_or_else(|| invalid(format!("tensor {name} element count overflow")))
        })?;
        let keep_f16 = (target == "token_embd.weight" && options.keep_embedding_f16)
            || (target == "output.weight" && options.keep_output_f16);
        let dtype = if source.shape.len() == 1 || target.ends_with("_norm.weight") {
            GgufQuantType::F32
        } else if options.format == OutputType::F16 || keep_f16 {
            GgufQuantType::F16
        } else {
            if source.shape.last().copied().unwrap_or(0) % 32 != 0 {
                return Err(invalid(format!(
                    "Q8_0 requires the innermost dimension of {name} to be divisible by 32"
                )));
            }
            GgufQuantType::Q8_0
        };
        let byte_len = match dtype {
            GgufQuantType::F32 => elements.checked_mul(4),
            GgufQuantType::F16 => elements.checked_mul(2),
            GgufQuantType::Q8_0 => (elements / 32).checked_mul(34),
            _ => None,
        }
        .ok_or_else(|| invalid(format!("tensor {name} output size overflow")))?;
        let plan = TensorPlan {
            source: TensorSource::Checkpoint(source.clone()),
            shape: source.shape.clone(),
            name: target.clone(),
            dtype,
            permutation_heads,
            elements,
            offset: 0,
            byte_len,
        };
        if mapped.insert(target.clone(), plan).is_some() {
            return Err(invalid(format!("duplicate GGUF tensor mapping: {target}")));
        }
    }
    let present: BTreeSet<_> = mapped.keys().cloned().collect();
    for required in required_tensors(model) {
        if !present.contains(&required) {
            return Err(invalid(format!(
                "missing required model tensor: {required}"
            )));
        }
    }
    let mut planned: Vec<_> = mapped.into_values().collect();
    // Extra rotary factors follow the source tensors, as in the converter.
    if let Some(factors) = &model.rope_factors {
        let elements = factors.len() as u64;
        planned.push(TensorPlan {
            source: TensorSource::Generated(factors.clone()),
            shape: vec![factors.len()],
            name: "rope_freqs.weight".into(),
            dtype: GgufQuantType::F32,
            permutation_heads: None,
            elements,
            offset: 0,
            byte_len: elements
                .checked_mul(4)
                .ok_or_else(|| invalid("rope factor size overflow"))?,
        });
    }
    let mut offset = 0_u64;
    let mut result = Vec::with_capacity(planned.len());
    for mut tensor in planned {
        tensor.offset = offset;
        offset = align(
            offset
                .checked_add(tensor.byte_len)
                .ok_or_else(|| invalid("GGUF tensor offset overflow"))?,
        )?;
        result.push(tensor);
    }
    Ok(result)
}

fn required_tensors(model: &ModelMetadata) -> BTreeSet<String> {
    let mut names = BTreeSet::from(["token_embd.weight".into(), "output_norm.weight".into()]);
    if !model.tie_embeddings {
        names.insert("output.weight".into());
    }
    for layer in 0..model.blocks {
        for suffix in [
            "attn_norm.weight",
            "ffn_norm.weight",
            "attn_q.weight",
            "attn_k.weight",
            "attn_v.weight",
            "attn_output.weight",
            "ffn_gate.weight",
            "ffn_up.weight",
            "ffn_down.weight",
        ] {
            names.insert(format!("blk.{layer}.{suffix}"));
        }
        if model.attention_bias {
            for suffix in ["attn_q.bias", "attn_k.bias", "attn_v.bias"] {
                names.insert(format!("blk.{layer}.{suffix}"));
            }
            if model.architecture == "llama" {
                names.insert(format!("blk.{layer}.attn_output.bias"));
            }
        }
        if model.mlp_bias {
            for suffix in ["ffn_gate.bias", "ffn_up.bias", "ffn_down.bias"] {
                names.insert(format!("blk.{layer}.{suffix}"));
            }
        }
    }
    names
}

fn map_tensor(name: &str, model: &ModelMetadata) -> Result<(String, Vec<usize>, Option<usize>)> {
    let hidden = model.hidden;
    match name {
        "model.embed_tokens.weight" => {
            return Ok(("token_embd.weight".into(), vec![model.vocab, hidden], None))
        }
        "model.norm.weight" => return Ok(("output_norm.weight".into(), vec![hidden], None)),
        "lm_head.weight" => return Ok(("output.weight".into(), vec![model.vocab, hidden], None)),
        _ => {}
    }
    let (layer, suffix) = name
        .strip_prefix("model.layers.")
        .and_then(|rest| rest.split_once('.'))
        .ok_or_else(|| invalid(format!("unsupported tensor name: {name}")))?;
    let layer: usize = layer
        .parse()
        .map_err(|_| invalid(format!("invalid layer in tensor {name}")))?;
    if layer >= model.blocks {
        return Err(invalid(format!(
            "tensor layer outside configured block count: {name}"
        )));
    }
    let kv_width = (hidden / model.heads)
        .checked_mul(model.kv_heads)
        .ok_or_else(|| invalid("key/value width overflow"))?;
    let (target, shape, heads) = match suffix {
        "input_layernorm.weight" => ("attn_norm.weight", vec![hidden], None),
        "post_attention_layernorm.weight" => ("ffn_norm.weight", vec![hidden], None),
        "self_attn.q_proj.weight" => ("attn_q.weight", vec![hidden, hidden], Some(model.heads)),
        "self_attn.k_proj.weight" => (
            "attn_k.weight",
            vec![kv_width, hidden],
            Some(model.kv_heads),
        ),
        "self_attn.v_proj.weight" => ("attn_v.weight", vec![kv_width, hidden], None),
        "self_attn.o_proj.weight" => ("attn_output.weight", vec![hidden, hidden], None),
        "mlp.gate_proj.weight" => ("ffn_gate.weight", vec![model.intermediate, hidden], None),
        "mlp.up_proj.weight" => ("ffn_up.weight", vec![model.intermediate, hidden], None),
        "mlp.down_proj.weight" => ("ffn_down.weight", vec![hidden, model.intermediate], None),
        "self_attn.q_proj.bias" if model.attention_bias => {
            ("attn_q.bias", vec![hidden], Some(model.heads))
        }
        "self_attn.k_proj.bias" if model.attention_bias => {
            ("attn_k.bias", vec![kv_width], Some(model.kv_heads))
        }
        "self_attn.v_proj.bias" if model.attention_bias => ("attn_v.bias", vec![kv_width], None),
        "self_attn.o_proj.bias" if model.attention_bias && model.architecture == "llama" => {
            ("attn_output.bias", vec![hidden], None)
        }
        "mlp.gate_proj.bias" if model.mlp_bias => ("ffn_gate.bias", vec![model.intermediate], None),
        "mlp.up_proj.bias" if model.mlp_bias => ("ffn_up.bias", vec![model.intermediate], None),
        "mlp.down_proj.bias" if model.mlp_bias => ("ffn_down.bias", vec![hidden], None),
        _ => return Err(invalid(format!("unsupported tensor name: {name}"))),
    };
    Ok((
        format!("blk.{layer}.{target}"),
        shape,
        if model.architecture == "llama" {
            heads
        } else {
            None
        },
    ))
}

fn write_u32<W: Write + ?Sized>(sink: &mut W, value: u32) -> Result<()> {
    sink.write_all(&value.to_le_bytes()).map_err(io_error)
}

fn write_u64<W: Write + ?Sized>(sink: &mut W, value: u64) -> Result<()> {
    sink.write_all(&value.to_le_bytes()).map_err(io_error)
}

fn write_string<W: Write + ?Sized>(sink: &mut W, value: &str) -> Result<()> {
    write_u64(sink, value.len() as u64)?;
    sink.write_all(value.as_bytes()).map_err(io_error)
}

fn value_type(value: &GgufValue) -> u32 {
    match value {
        GgufValue::U8(_) => 0,
        GgufValue::I8(_) => 1,
        GgufValue::U16(_) => 2,
        GgufValue::I16(_) => 3,
        GgufValue::U32(_) => 4,
        GgufValue::I32(_) => 5,
        GgufValue::F32(_) => 6,
        GgufValue::Bool(_) => 7,
        GgufValue::String(_) => 8,
        GgufValue::Array(_) => 9,
        GgufValue::U64(_) => 10,
        GgufValue::I64(_) => 11,
        GgufValue::F64(_) => 12,
    }
}

fn write_value<W: Write + ?Sized>(sink: &mut W, value: &GgufValue) -> Result<()> {
    match value {
        GgufValue::U8(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::I8(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::U16(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::I16(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::U32(v) => write_u32(sink, *v),
        GgufValue::I32(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::F32(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::Bool(v) => sink.write_all(&[u8::from(*v)]).map_err(io_error),
        GgufValue::String(v) => write_string(sink, v),
        GgufValue::U64(v) => write_u64(sink, *v),
        GgufValue::I64(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::F64(v) => sink.write_all(&v.to_le_bytes()).map_err(io_error),
        GgufValue::Array(values) => {
            // An empty array is only emitted for tokenizer merges, whose type is string.
            let kind = values.first().map(value_type).unwrap_or(8);
            if kind == 9 || values.iter().any(|value| value_type(value) != kind) {
                return Err(invalid("GGUF metadata arrays must contain one scalar type"));
            }
            write_u32(sink, kind)?;
            write_u64(sink, values.len() as u64)?;
            for value in values {
                write_value(sink, value)?;
            }
            Ok(())
        }
    }
}

fn make_header(metadata: &BTreeMap<String, GgufValue>, tensors: &[TensorPlan]) -> Result<Vec<u8>> {
    if tensors.is_empty() {
        return Err(invalid("cannot write GGUF without tensors"));
    }
    let mut bytes = Vec::new();
    write_u32(&mut bytes, GGUF_MAGIC)?;
    write_u32(&mut bytes, GGUF_VERSION)?;
    write_u64(&mut bytes, tensors.len() as u64)?;
    write_u64(&mut bytes, metadata.len() as u64)?;
    for (name, value) in metadata {
        write_string(&mut bytes, name)?;
        write_u32(&mut bytes, value_type(value))?;
        write_value(&mut bytes, value)?;
    }
    for tensor in tensors {
        write_string(&mut bytes, &tensor.name)?;
        write_u32(&mut bytes, tensor.shape.len() as u32)?;
        for &dimension in tensor.shape.iter().rev() {
            write_u64(&mut bytes, dimension as u64)?;
        }
        write_u32(&mut bytes, tensor.dtype as u32)?;
        write_u64(&mut bytes, tensor.offset)?;
    }
    let length = usize::try_from(align(bytes.len() as u64)?)
        .map_err(|_| invalid("GGUF header exceeds addressable memory"))?;
    bytes.resize(length, 0);
    Ok(bytes)
}

fn source_value(bytes: &[u8], dtype: SourceDtype, index: usize) -> Result<f32> {
    let start = index
        .checked_mul(dtype.byte_width())
        .ok_or_else(|| invalid("source tensor byte index overflow"))?;
    let end = start
        .checked_add(dtype.byte_width())
        .ok_or_else(|| invalid("source tensor byte index overflow"))?;
    let value = bytes
        .get(start..end)
        .ok_or_else(|| invalid("source tensor data is truncated"))?;
    let value = match dtype {
        SourceDtype::F32 => f32::from_le_bytes([value[0], value[1], value[2], value[3]]),
        SourceDtype::F16 => f16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32(),
        SourceDtype::BF16 => bf16::from_bits(u16::from_le_bytes([value[0], value[1]])).to_f32(),
    };
    if !value.is_finite() {
        return Err(invalid("source tensor contains a non-finite value"));
    }
    Ok(value)
}

fn input_row(output_row: usize, rows: usize, heads: Option<usize>) -> Result<usize> {
    let Some(heads) = heads else {
        return Ok(output_row);
    };
    if heads == 0 || rows % heads != 0 || (rows / heads) % 2 != 0 {
        return Err(invalid(
            "llama rotary projection requires an even head dimension",
        ));
    }
    let per_head = rows / heads;
    let head = output_row / per_head;
    let within = output_row % per_head;
    Ok(head * per_head + (within % 2) * (per_head / 2) + within / 2)
}

fn encode_q8_block(values: &[f32; 32]) -> Result<[u8; 34]> {
    if values.iter().any(|v| !v.is_finite()) {
        return Err(invalid("Q8_0 input contains a non-finite value"));
    }
    let maximum = values.iter().fold(0_f32, |acc, value| acc.max(value.abs()));
    let scale = maximum / 127.0;
    let reciprocal = if scale == 0.0 { 0.0 } else { 1.0 / scale };
    if !reciprocal.is_finite() {
        return Err(invalid(
            "Q8_0 scale is too small for finite f32 quantization",
        ));
    }
    let stored_scale = f16::from_f32(scale);
    if !stored_scale.is_finite() {
        return Err(invalid("Q8_0 scale exceeds finite F16 range"));
    }
    let mut block = [0_u8; 34];
    block[..2].copy_from_slice(&stored_scale.to_bits().to_le_bytes());
    for (index, value) in values.iter().enumerate() {
        // GGML rounds using the original f32 scale, before storing it as f16.
        block[index + 2] = (value * reciprocal).round().clamp(-127.0, 127.0) as i8 as u8;
    }
    Ok(block)
}

fn write_gguf<W: Write + ?Sized>(
    sink: &mut W,
    header: &[u8],
    tensors: &[TensorPlan],
    checkpoint: &Checkpoint,
) -> Result<()> {
    sink.write_all(header).map_err(io_error)?;
    let mut position = 0_u64;
    for tensor in tensors {
        if position != tensor.offset {
            return Err(invalid("GGUF planned tensor offset mismatch"));
        }
        match &tensor.source {
            TensorSource::Generated(values) => {
                for value in values {
                    sink.write_all(&value.to_le_bytes()).map_err(io_error)?;
                }
            }
            TensorSource::Checkpoint(source) => {
                let length = usize::try_from(source.byte_len)
                    .map_err(|_| invalid("source tensor exceeds addressable memory"))?;
                let mut bytes = Vec::new();
                bytes
                    .try_reserve_exact(length)
                    .map_err(|_| invalid("cannot allocate source tensor buffer"))?;
                bytes.resize(length, 0);
                let mut reader = checkpoint.tensor_reader(source)?;
                reader.read_exact(&mut bytes).map_err(io_error)?;
                reader.finish()?;
                let rows = source.shape[0];
                let columns = source.shape.get(1).copied().unwrap_or(1);
                for row in 0..rows {
                    let start = input_row(row, rows, tensor.permutation_heads)?
                        .checked_mul(columns)
                        .ok_or_else(|| invalid("tensor row offset overflow"))?;
                    if tensor.dtype == GgufQuantType::Q8_0 {
                        for column in (0..columns).step_by(32) {
                            let mut values = [0_f32; 32];
                            for (lane, value) in values.iter_mut().enumerate() {
                                *value = source_value(&bytes, source.dtype, start + column + lane)?;
                            }
                            sink.write_all(&encode_q8_block(&values)?)
                                .map_err(io_error)?;
                        }
                    } else {
                        for column in 0..columns {
                            let value = source_value(&bytes, source.dtype, start + column)?;
                            match tensor.dtype {
                                GgufQuantType::F32 => {
                                    sink.write_all(&value.to_le_bytes()).map_err(io_error)?
                                }
                                GgufQuantType::F16 => {
                                    let value = f16::from_f32(value);
                                    if !value.is_finite() {
                                        return Err(invalid(format!(
                                            "tensor {} overflows F16",
                                            tensor.name
                                        )));
                                    }
                                    sink.write_all(&value.to_bits().to_le_bytes())
                                        .map_err(io_error)?;
                                }
                                _ => return Err(invalid("unsupported GGUF output tensor type")),
                            }
                        }
                    }
                }
            }
        }
        position = position
            .checked_add(tensor.byte_len)
            .ok_or_else(|| invalid("GGUF data position overflow"))?;
        let aligned = align(position)?;
        let padding = (aligned - position) as usize;
        sink.write_all(&[0_u8; DEFAULT_ALIGNMENT][..padding])
            .map_err(io_error)?;
        position = aligned;
    }
    Ok(())
}

fn require_absent(path: &Path) -> Result<()> {
    match fs::symlink_metadata(path) {
        Ok(_) => Err(invalid(format!(
            "output already exists: {}",
            path.display()
        ))),
        Err(error) if error.kind() == io::ErrorKind::NotFound => Ok(()),
        Err(error) => Err(io_error(error)),
    }
}

struct TemporaryOutput {
    path: PathBuf,
}

impl Drop for TemporaryOutput {
    fn drop(&mut self) {
        let _ = fs::remove_file(&self.path);
    }
}

fn publish<F>(output: &Path, write: F) -> Result<()>
where
    F: FnOnce(&mut File) -> Result<()>,
{
    require_absent(output)?;
    let parent = output
        .parent()
        .filter(|p| !p.as_os_str().is_empty())
        .unwrap_or(Path::new("."));
    let name = output
        .file_name()
        .ok_or_else(|| invalid("output must name a GGUF file"))?;
    static NEXT_TEMP: AtomicU64 = AtomicU64::new(0);
    let mut opened = None;
    for _ in 0..128 {
        let mut temporary = std::ffi::OsString::from(".");
        temporary.push(name);
        temporary.push(format!(
            ".part.{}.{}",
            std::process::id(),
            NEXT_TEMP.fetch_add(1, Ordering::Relaxed)
        ));
        let path = parent.join(temporary);
        let mut options = OpenOptions::new();
        options.write(true).create_new(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        match options.open(&path) {
            Ok(file) => {
                opened = Some((TemporaryOutput { path }, file));
                break;
            }
            Err(error) if error.kind() == io::ErrorKind::AlreadyExists => continue,
            Err(error) => return Err(io_error(error)),
        }
    }
    let (temporary, mut file) =
        opened.ok_or_else(|| invalid("cannot reserve temporary GGUF output"))?;
    write(&mut file)?;
    file.sync_all().map_err(io_error)?;
    drop(file);
    // Linking the synced inode publishes complete data without replacing a racing output.
    fs::hard_link(&temporary.path, output).map_err(io_error)?;
    Ok(())
}

#[cfg(test)]
mod tests;
