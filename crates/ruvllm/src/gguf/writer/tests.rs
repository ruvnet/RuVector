use super::*;
use crate::gguf::parser::{parse_header, parse_metadata, parse_tensor_infos};
use crate::gguf::quantization::dequantize_tensor;
use crate::gguf::tensors::TensorInfo;
use serde_json::{json, Map, Value};
use std::collections::HashMap;
use std::io::Cursor;
use tempfile::TempDir;

#[derive(Clone, Copy)]
enum FixtureDtype {
    F32,
    F16,
    BF16,
}

impl FixtureDtype {
    fn name(self) -> &'static str {
        match self {
            Self::F32 => "F32",
            Self::F16 => "F16",
            Self::BF16 => "BF16",
        }
    }

    fn bytes(self, values: &[f32]) -> Vec<u8> {
        let mut bytes = Vec::new();
        for &value in values {
            match self {
                Self::F32 => bytes.extend_from_slice(&value.to_le_bytes()),
                Self::F16 => bytes.extend_from_slice(&f16::from_f32(value).to_bits().to_le_bytes()),
                Self::BF16 => {
                    bytes.extend_from_slice(&bf16::from_f32(value).to_bits().to_le_bytes())
                }
            }
        }
        bytes
    }

    fn decoded(self, value: f32) -> f32 {
        match self {
            Self::F32 => value,
            Self::F16 => f16::from_f32(value).to_f32(),
            Self::BF16 => bf16::from_f32(value).to_f32(),
        }
    }
}

#[derive(Clone)]
struct FixtureTensor {
    source_name: String,
    output_name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
    dtype: FixtureDtype,
}

struct Fixture {
    config: Value,
    tensors: Vec<FixtureTensor>,
}

impl Fixture {
    fn new(architecture: &str, dtype: FixtureDtype, hidden: usize, attention_bias: bool) -> Self {
        let mut fixture = Self {
            config: json!({
                "model_type": architecture,
                "hidden_size": hidden,
                "num_attention_heads": 4,
                "num_key_value_heads": 2,
                "intermediate_size": 64,
                "num_hidden_layers": 1,
                "vocab_size": 32,
                "max_position_embeddings": 128,
                "rope_theta": 10000.0,
                "rms_norm_eps": 0.00001,
                "hidden_act": "silu",
                "attention_bias": attention_bias,
                "mlp_bias": false,
                "tie_word_embeddings": false,
                "bos_token_id": 0,
                "eos_token_id": 1,
                "pad_token_id": 2
            }),
            tensors: Vec::new(),
        };
        let kv = hidden / 2;
        for (source, output, shape) in [
            (
                "model.embed_tokens.weight",
                "token_embd.weight",
                vec![32, hidden],
            ),
            ("model.norm.weight", "output_norm.weight", vec![hidden]),
            ("lm_head.weight", "output.weight", vec![32, hidden]),
            (
                "model.layers.0.input_layernorm.weight",
                "blk.0.attn_norm.weight",
                vec![hidden],
            ),
            (
                "model.layers.0.post_attention_layernorm.weight",
                "blk.0.ffn_norm.weight",
                vec![hidden],
            ),
            (
                "model.layers.0.self_attn.q_proj.weight",
                "blk.0.attn_q.weight",
                vec![hidden, hidden],
            ),
            (
                "model.layers.0.self_attn.k_proj.weight",
                "blk.0.attn_k.weight",
                vec![kv, hidden],
            ),
            (
                "model.layers.0.self_attn.v_proj.weight",
                "blk.0.attn_v.weight",
                vec![kv, hidden],
            ),
            (
                "model.layers.0.self_attn.o_proj.weight",
                "blk.0.attn_output.weight",
                vec![hidden, hidden],
            ),
            (
                "model.layers.0.mlp.gate_proj.weight",
                "blk.0.ffn_gate.weight",
                vec![64, hidden],
            ),
            (
                "model.layers.0.mlp.up_proj.weight",
                "blk.0.ffn_up.weight",
                vec![64, hidden],
            ),
            (
                "model.layers.0.mlp.down_proj.weight",
                "blk.0.ffn_down.weight",
                vec![hidden, 64],
            ),
        ] {
            fixture.add_tensor(source, output, shape, dtype);
        }
        if attention_bias {
            for (source, output, width) in [
                ("q_proj", "attn_q", hidden),
                ("k_proj", "attn_k", kv),
                ("v_proj", "attn_v", kv),
            ] {
                fixture.add_tensor(
                    &format!("model.layers.0.self_attn.{source}.bias"),
                    &format!("blk.0.{output}.bias"),
                    vec![width],
                    dtype,
                );
            }
            if architecture == "llama" {
                fixture.add_tensor(
                    "model.layers.0.self_attn.o_proj.bias",
                    "blk.0.attn_output.bias",
                    vec![hidden],
                    dtype,
                );
            }
        }
        fixture
    }

    fn add_tensor(&mut self, source: &str, output: &str, shape: Vec<usize>, dtype: FixtureDtype) {
        let ordinal = self.tensors.len();
        let values = (0..shape.iter().product::<usize>())
            .map(|index| index as f32 / 128.0 - 2.0 + ordinal as f32 / 16.0)
            .collect();
        self.tensors.push(FixtureTensor {
            source_name: source.into(),
            output_name: output.into(),
            shape,
            values,
            dtype,
        });
    }

    fn write(&self, directory: &Path, sharded: bool) -> PathBuf {
        fs::create_dir_all(directory).unwrap();
        fs::write(
            directory.join("config.json"),
            serde_json::to_vec(&self.config).unwrap(),
        )
        .unwrap();
        let tokens = fixture_tokens();
        let vocab: Map<String, Value> = tokens
            .iter()
            .enumerate()
            .map(|(id, token)| (token.clone(), json!(id)))
            .collect();
        let tokenizer = json!({
            "version": "1.0",
            "normalizer": null,
            "post_processor": null,
            "pre_tokenizer": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "decoder": {"type": "ByteLevel", "add_prefix_space": false, "trim_offsets": true, "use_regex": true},
            "added_tokens": [],
            "model": {"type": "BPE", "vocab": vocab, "merges": ["a b"]}
        });
        fs::write(
            directory.join("tokenizer.json"),
            serde_json::to_vec(&tokenizer).unwrap(),
        )
        .unwrap();
        fs::write(
            directory.join("tokenizer_config.json"),
            serde_json::to_vec(&json!({
                "bos_token": "a", "eos_token": "b", "pad_token": "ab",
                "add_bos_token": false, "add_eos_token": false
            }))
            .unwrap(),
        )
        .unwrap();
        if !sharded {
            let path = directory.join("model.safetensors");
            write_safetensors(&path, &self.tensors);
            return path;
        }
        let midpoint = self.tensors.len() / 2;
        let mut weight_map = Map::new();
        let mut total_size = 0_u64;
        for (filename, tensors) in [
            (
                "model-00001-of-00002.safetensors",
                &self.tensors[..midpoint],
            ),
            (
                "model-00002-of-00002.safetensors",
                &self.tensors[midpoint..],
            ),
        ] {
            write_safetensors(&directory.join(filename), tensors);
            for tensor in tensors {
                weight_map.insert(tensor.source_name.clone(), json!(filename));
                total_size += tensor.dtype.bytes(&tensor.values).len() as u64;
            }
        }
        fs::write(
            directory.join("model.safetensors.index.json"),
            serde_json::to_vec(&json!({
                "metadata": {"total_size": total_size}, "weight_map": weight_map
            }))
            .unwrap(),
        )
        .unwrap();
        directory.to_path_buf()
    }
}

fn fixture_tokens() -> Vec<String> {
    let mut tokens = vec!["a".into(), "b".into(), "ab".into()];
    tokens.extend((3..32).map(|id| format!("token{id}")));
    tokens
}

fn write_safetensors(path: &Path, tensors: &[FixtureTensor]) {
    let mut header = Map::new();
    let mut payload = Vec::new();
    for tensor in tensors {
        assert_eq!(tensor.values.len(), tensor.shape.iter().product::<usize>());
        let start = payload.len();
        payload.extend_from_slice(&tensor.dtype.bytes(&tensor.values));
        header.insert(
            tensor.source_name.clone(),
            json!({
                "dtype": tensor.dtype.name(), "shape": tensor.shape,
                "data_offsets": [start, payload.len()]
            }),
        );
    }
    let mut json = serde_json::to_vec(&header).unwrap();
    while json.len() % 8 != 0 {
        json.push(b' ');
    }
    let mut bytes = Vec::new();
    bytes.extend_from_slice(&(json.len() as u64).to_le_bytes());
    bytes.extend_from_slice(&json);
    bytes.extend_from_slice(&payload);
    fs::write(path, bytes).unwrap();
}

struct Parsed {
    bytes: Vec<u8>,
    metadata: HashMap<String, GgufValue>,
    tensors: Vec<TensorInfo>,
    data_start: usize,
}

impl Parsed {
    fn read(path: &Path) -> Self {
        let bytes = fs::read(path).unwrap();
        assert_eq!(&bytes[..4], b"GGUF");
        let mut cursor = Cursor::new(bytes.as_slice());
        let header = parse_header(&mut cursor).unwrap();
        assert_eq!(header.magic, GGUF_MAGIC);
        assert_eq!(header.version, 3);
        assert!(header.tensor_count > 0);
        let metadata = parse_metadata(&mut cursor, header.metadata_kv_count).unwrap();
        let tensors = parse_tensor_infos(&mut cursor, header.tensor_count).unwrap();
        assert_eq!(metadata.len() as u64, header.metadata_kv_count);
        assert_eq!(tensors.len() as u64, header.tensor_count);
        let data_start = (cursor.position() as usize + 31) / 32 * 32;
        assert!(data_start < bytes.len());
        Self {
            bytes,
            metadata,
            tensors,
            data_start,
        }
    }

    fn tensor(&self, name: &str) -> &TensorInfo {
        self.tensors
            .iter()
            .find(|tensor| tensor.name == name)
            .unwrap()
    }

    fn data(&self, tensor: &TensorInfo) -> &[u8] {
        let elements = tensor.shape.iter().product::<usize>();
        let count = match tensor.dtype {
            GgufQuantType::F32 => elements * 4,
            GgufQuantType::F16 => elements * 2,
            GgufQuantType::Q8_0 => {
                assert_eq!(elements % 32, 0);
                elements / 32 * 34
            }
            other => panic!("unexpected tensor encoding: {other:?}"),
        };
        let start = self.data_start + tensor.offset as usize;
        assert_eq!(start % 32, 0);
        assert!(start + count <= self.bytes.len());
        &self.bytes[start..start + count]
    }

    fn values(&self, name: &str) -> Vec<f32> {
        let tensor = self.tensor(name);
        dequantize_tensor(
            self.data(tensor),
            tensor.dtype,
            tensor.shape.iter().product(),
        )
        .unwrap()
    }
}

fn expected_values(tensor: &FixtureTensor, architecture: &str) -> Vec<f32> {
    let values: Vec<f32> = tensor
        .values
        .iter()
        .map(|&v| tensor.dtype.decoded(v))
        .collect();
    let rotary = tensor.output_name.contains(".attn_q.") || tensor.output_name.contains(".attn_k.");
    if architecture != "llama" || !rotary {
        return values;
    }
    let columns = tensor.shape.get(1).copied().unwrap_or(1);
    let mut expected = Vec::new();
    for base in (0..tensor.shape[0]).step_by(8) {
        for row in [0, 4, 1, 5, 2, 6, 3, 7] {
            expected.extend_from_slice(&values[(base + row) * columns..(base + row + 1) * columns]);
        }
    }
    expected
}

fn assert_metadata(parsed: &Parsed, architecture: &str, format: OutputType) {
    assert_eq!(
        parsed.metadata["general.architecture"].as_str(),
        Some(architecture)
    );
    assert_eq!(
        parsed.metadata["general.file_type"].as_u64(),
        Some(if format == OutputType::F16 { 1 } else { 7 })
    );
    assert_eq!(
        parsed.metadata["general.quantization_version"].as_u64(),
        Some(2)
    );
    assert_eq!(parsed.metadata["general.alignment"].as_u64(), Some(32));
    for (key, value) in [
        ("context_length", 128),
        ("embedding_length", 32),
        ("block_count", 1),
        ("feed_forward_length", 64),
        ("attention.head_count", 4),
        ("attention.head_count_kv", 2),
        ("vocab_size", 32),
        ("rope.dimension_count", 8),
    ] {
        assert_eq!(
            parsed.metadata[&format!("{architecture}.{key}")].as_u64(),
            Some(value)
        );
    }
    assert_eq!(
        parsed.metadata[&format!("{architecture}.rope.freq_base")].as_f32(),
        Some(10000.0)
    );
    assert_eq!(
        parsed.metadata[&format!("{architecture}.attention.layer_norm_rms_epsilon")].as_f32(),
        Some(0.00001)
    );
    assert_eq!(
        parsed.metadata["tokenizer.ggml.model"].as_str(),
        Some("gpt2")
    );
    assert_eq!(
        parsed.metadata["tokenizer.ggml.pre"].as_str(),
        Some("gpt-2")
    );
    for (key, value) in [
        ("bos_token_id", 0),
        ("eos_token_id", 1),
        ("padding_token_id", 2),
    ] {
        assert_eq!(
            parsed.metadata[&format!("tokenizer.ggml.{key}")].as_u64(),
            Some(value)
        );
    }
    assert_eq!(
        parsed.metadata["tokenizer.ggml.add_bos_token"].as_bool(),
        Some(false)
    );
    assert_eq!(
        parsed.metadata["tokenizer.ggml.add_eos_token"].as_bool(),
        Some(false)
    );
    let tokens: Vec<&str> = parsed.metadata["tokenizer.ggml.tokens"]
        .as_array()
        .unwrap()
        .iter()
        .map(|value| value.as_str().unwrap())
        .collect();
    assert_eq!(
        tokens,
        fixture_tokens()
            .iter()
            .map(String::as_str)
            .collect::<Vec<_>>()
    );
    let types = parsed.metadata["tokenizer.ggml.token_type"]
        .as_array()
        .unwrap();
    assert_eq!(types.len(), 32);
    for value in types {
        assert_eq!(value.as_u64(), Some(1));
    }
    let merges = parsed.metadata["tokenizer.ggml.merges"].as_array().unwrap();
    assert_eq!(merges.len(), 1);
    assert_eq!(merges[0].as_str(), Some("a b"));
}

fn assert_tensor_values(parsed: &Parsed, source: &FixtureTensor, architecture: &str) {
    let tensor = parsed.tensor(&source.output_name);
    let expected = expected_values(source, architecture);
    let actual = parsed.values(&source.output_name);
    assert_eq!(actual.len(), expected.len());
    if tensor.dtype == GgufQuantType::Q8_0 {
        for (source_block, output_block) in expected.chunks_exact(32).zip(actual.chunks_exact(32)) {
            let maximum = source_block.iter().fold(0.0_f32, |a, &b| a.max(b.abs()));
            let scale = maximum / 127.0;
            let stored_scale = f16::from_f32(scale).to_f32();
            assert!(stored_scale == 0.0 || stored_scale >= 0.00006103515625);
            let bound = scale * 0.5 + 127.0 * (stored_scale - scale).abs() + 0.000001;
            for (&source_value, &output_value) in source_block.iter().zip(output_block) {
                assert!(
                    (source_value - output_value).abs() <= bound,
                    "{}: {source_value} -> {output_value}, bound {bound}",
                    source.output_name
                );
            }
        }
    } else {
        let expected: Vec<f32> = expected
            .into_iter()
            .map(|value| {
                if tensor.dtype == GgufQuantType::F16 {
                    f16::from_f32(value).to_f32()
                } else {
                    value
                }
            })
            .collect();
        assert_eq!(actual, expected, "{}", source.output_name);
    }
}

#[test]
fn tiny_models_preserve_tensors_metadata_and_values() {
    for architecture in ["llama", "qwen2"] {
        for format in [OutputType::F16, OutputType::Q8_0] {
            let directory = TempDir::new().unwrap();
            let fixture =
                Fixture::new(architecture, FixtureDtype::F32, 32, architecture == "qwen2");
            let source = fixture.write(&directory.path().join("source"), false);
            let output = directory.path().join("model.gguf");
            let options = WriterOptions {
                format,
                keep_embedding_f16: false,
                keep_output_f16: false,
            };
            let summary = convert_safetensors(&source, &output, &options).unwrap();
            let parsed = Parsed::read(&output);
            let count = if architecture == "llama" { 12 } else { 15 };
            assert_eq!(summary.tensor_count, count);
            assert_eq!(parsed.tensors.len(), count);
            assert_eq!(summary.output_bytes, parsed.bytes.len() as u64);
            assert_eq!(
                summary.element_count,
                fixture
                    .tensors
                    .iter()
                    .map(|t| t.values.len() as u64)
                    .sum::<u64>()
            );
            let names: BTreeSet<_> = parsed.tensors.iter().map(|t| t.name.as_str()).collect();
            let expected_names: BTreeSet<_> = fixture
                .tensors
                .iter()
                .map(|t| t.output_name.as_str())
                .collect();
            assert_eq!(names, expected_names);
            assert_metadata(&parsed, architecture, format);
            let mut previous_end = 0;
            for tensor in &parsed.tensors {
                assert!(tensor.offset >= previous_end);
                previous_end = tensor.offset + parsed.data(tensor).len() as u64;
            }
            for source in &fixture.tensors {
                let tensor = parsed.tensor(&source.output_name);
                assert_eq!(
                    tensor.shape,
                    source.shape.iter().rev().copied().collect::<Vec<_>>()
                );
                let expected_type = if source.shape.len() == 1 {
                    GgufQuantType::F32
                } else if format == OutputType::F16 {
                    GgufQuantType::F16
                } else {
                    GgufQuantType::Q8_0
                };
                assert_eq!(tensor.dtype, expected_type, "{}", tensor.name);
                assert_tensor_values(&parsed, source, architecture);
            }
        }
    }
}

#[test]
fn q8_embedding_and_output_retention_are_independent() {
    for keep_embedding_f16 in [false, true] {
        for keep_output_f16 in [false, true] {
            let directory = TempDir::new().unwrap();
            let fixture = Fixture::new("qwen2", FixtureDtype::F32, 32, true);
            let source = fixture.write(&directory.path().join("source"), false);
            let output = directory.path().join("model.gguf");
            let options = WriterOptions {
                format: OutputType::Q8_0,
                keep_embedding_f16,
                keep_output_f16,
            };
            convert_safetensors(&source, &output, &options).unwrap();
            let parsed = Parsed::read(&output);
            for (name, keep) in [
                ("token_embd.weight", keep_embedding_f16),
                ("output.weight", keep_output_f16),
            ] {
                assert_eq!(
                    parsed.tensor(name).dtype,
                    if keep {
                        GgufQuantType::F16
                    } else {
                        GgufQuantType::Q8_0
                    }
                );
            }
            assert_eq!(
                parsed.tensor("blk.0.attn_q.weight").dtype,
                GgufQuantType::Q8_0
            );
            assert_eq!(parsed.tensor("blk.0.attn_q.bias").dtype, GgufQuantType::F32);
            assert_eq!(
                parsed.tensor("output_norm.weight").dtype,
                GgufQuantType::F32
            );
        }
    }
}

#[test]
fn q8_blocks_match_hand_computed_reference() {
    assert_eq!(encode_q8_block(&[0.0; 32]).unwrap(), [0; 34]);
    let mut ties = [0.0; 32];
    ties[..8].copy_from_slice(&[127.0, -127.0, 0.5, -0.5, 1.5, -1.5, 126.5, -126.5]);
    let mut expected = [0_u8; 34];
    expected[..10].copy_from_slice(&[0x00, 0x3c, 127, 129, 1, 255, 2, 254, 127, 129]);
    assert_eq!(encode_q8_block(&ties).unwrap(), expected);

    let mut values = [0.0; 32];
    values[..6].copy_from_slice(&[1.0, -1.0, 0.0039369, -0.0039369, 0.25, -0.25]);
    let original_scale = 1.0_f32 / 127.0;
    let stored_scale = f16::from_bits(0x2008).to_f32();
    assert_ne!(original_scale, stored_scale);
    assert!(values[2] / original_scale < 0.5);
    assert!(values[2] / stored_scale > 0.5);
    let mut expected = [0_u8; 34];
    expected[..8].copy_from_slice(&[0x08, 0x20, 127, 129, 0, 0, 32, 224]);
    assert_eq!(encode_q8_block(&values).unwrap(), expected);
}

#[test]
fn llama_rotary_rows_are_permuted_and_qwen_rows_are_unchanged() {
    for architecture in ["llama", "qwen2"] {
        let directory = TempDir::new().unwrap();
        let mut fixture = Fixture::new(architecture, FixtureDtype::F32, 32, true);
        for tensor in &mut fixture.tensors {
            let columns = tensor.shape.get(1).copied().unwrap_or(1);
            for (index, value) in tensor.values.iter_mut().enumerate() {
                *value = (index / columns) as f32;
            }
        }
        let source = fixture.write(&directory.path().join("source"), false);
        let output = directory.path().join("model.gguf");
        convert_safetensors(&source, &output, &WriterOptions::default()).unwrap();
        let parsed = Parsed::read(&output);
        for source in &fixture.tensors {
            assert_tensor_values(&parsed, source, architecture);
        }
        let q = parsed.values("blk.0.attn_q.bias");
        let k = parsed.values("blk.0.attn_k.bias");
        let first_head = if architecture == "llama" {
            vec![0.0, 4.0, 1.0, 5.0, 2.0, 6.0, 3.0, 7.0]
        } else {
            vec![0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        };
        assert_eq!(&q[..8], first_head.as_slice());
        assert_eq!(&k[..8], first_head.as_slice());
        assert_eq!(
            &parsed.values("blk.0.attn_v.bias")[..8],
            &[0.0, 1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0]
        );
    }
}

#[test]
fn sharded_and_single_file_checkpoints_produce_identical_output() {
    for format in [OutputType::F16, OutputType::Q8_0] {
        let directory = TempDir::new().unwrap();
        let fixture = Fixture::new("qwen2", FixtureDtype::BF16, 32, true);
        let single = fixture.write(&directory.path().join("single"), false);
        let sharded = fixture.write(&directory.path().join("sharded"), true);
        let first = directory.path().join("single.gguf");
        let second = directory.path().join("sharded.gguf");
        let options = WriterOptions {
            format,
            ..WriterOptions::default()
        };
        convert_safetensors(&single, &first, &options).unwrap();
        convert_safetensors(&sharded, &second, &options).unwrap();
        assert_eq!(fs::read(&first).unwrap(), fs::read(&second).unwrap());
        assert_eq!(Parsed::read(&second).tensors.len(), 15);
    }
}

#[test]
fn all_source_dtypes_decode_with_their_own_encoding() {
    let probe = [1.0, -2.5, 1.00390625, 0.125];
    assert_ne!(
        f16::from_f32(probe[2]).to_bits(),
        bf16::from_f32(probe[2]).to_bits()
    );
    assert_ne!(
        f16::from_f32(probe[2]).to_f32(),
        bf16::from_f32(probe[2]).to_f32()
    );
    for dtype in [FixtureDtype::F32, FixtureDtype::F16, FixtureDtype::BF16] {
        let directory = TempDir::new().unwrap();
        let mut fixture = Fixture::new("qwen2", dtype, 32, true);
        for tensor in &mut fixture.tensors {
            for (index, value) in tensor.values.iter_mut().enumerate() {
                *value = probe[index % probe.len()];
            }
        }
        let source = fixture.write(&directory.path().join("source"), false);
        let output = directory.path().join("model.gguf");
        convert_safetensors(&source, &output, &WriterOptions::default()).unwrap();
        let parsed = Parsed::read(&output);
        assert_eq!(
            parsed.values("output_norm.weight")[..4],
            probe.map(|value| dtype.decoded(value))
        );
        assert_eq!(
            parsed.values("token_embd.weight")[..4],
            probe.map(|value| f16::from_f32(dtype.decoded(value)).to_f32())
        );
        for tensor in &fixture.tensors {
            assert_tensor_values(&parsed, tensor, "qwen2");
        }
    }
}

fn directory_entries(path: &Path) -> BTreeSet<String> {
    fs::read_dir(path)
        .unwrap()
        .map(|entry| entry.unwrap().file_name().to_string_lossy().into_owned())
        .collect()
}

fn assert_rejected(source: &Path, output: &Path, options: &WriterOptions, expected_error: &str) {
    let parent = output.parent().unwrap();
    let before = directory_entries(parent);
    let error = convert_safetensors(source, output, options)
        .unwrap_err()
        .to_string();
    assert!(
        error.contains(expected_error),
        "expected {expected_error:?}, received {error:?}"
    );
    assert!(!output.exists());
    assert_eq!(
        directory_entries(parent),
        before,
        "failed conversion left a temporary file"
    );
}

#[test]
fn invalid_inputs_leave_no_output() {
    for case in [
        "architecture",
        "missing_config",
        "truncated",
        "missing_tensor",
        "unknown_tensor",
        "wrong_shape",
        "nan",
        "infinity",
        "negative_infinity",
    ] {
        let directory = TempDir::new().unwrap();
        let mut fixture = Fixture::new("qwen2", FixtureDtype::F32, 32, true);
        let expected_error = match case {
            "architecture" => {
                fixture.config["model_type"] = json!("unsupported_architecture");
                "unsupported_architecture"
            }
            "missing_config" => {
                fixture
                    .config
                    .as_object_mut()
                    .unwrap()
                    .remove("hidden_size");
                "hidden_size"
            }
            "missing_tensor" => {
                fixture
                    .tensors
                    .retain(|t| t.output_name != "blk.0.attn_k.weight");
                "missing required model tensor"
            }
            "unknown_tensor" => {
                fixture.tensors[0].source_name = "model.unrecognized.weight".into();
                "unsupported tensor name"
            }
            "wrong_shape" => {
                fixture.tensors[0].shape = vec![16, 64];
                "expected"
            }
            "nan" => {
                fixture.tensors[0].values[0] = f32::NAN;
                "non-finite"
            }
            "infinity" => {
                fixture.tensors[0].values[0] = f32::INFINITY;
                "non-finite"
            }
            "negative_infinity" => {
                fixture.tensors[0].values[0] = f32::NEG_INFINITY;
                "non-finite"
            }
            "truncated" => "truncated",
            _ => unreachable!(),
        };
        let source = fixture.write(&directory.path().join("source"), false);
        if case == "truncated" {
            let length = fs::metadata(&source).unwrap().len();
            OpenOptions::new()
                .write(true)
                .open(&source)
                .unwrap()
                .set_len(length - 1)
                .unwrap();
        }
        for format in [OutputType::F16, OutputType::Q8_0] {
            let options = WriterOptions {
                format,
                ..WriterOptions::default()
            };
            assert_rejected(
                &source,
                &directory.path().join("model.gguf"),
                &options,
                expected_error,
            );
        }
    }
}

#[test]
fn q8_rejects_non_block_aligned_rows() {
    let directory = TempDir::new().unwrap();
    let fixture = Fixture::new("qwen2", FixtureDtype::F32, 24, true);
    let source = fixture.write(&directory.path().join("source"), false);
    let options = WriterOptions {
        format: OutputType::Q8_0,
        keep_embedding_f16: false,
        keep_output_f16: false,
    };
    assert_rejected(
        &source,
        &directory.path().join("model.gguf"),
        &options,
        "divisible by 32",
    );
}

#[test]
fn existing_output_is_preserved() {
    let directory = TempDir::new().unwrap();
    let fixture = Fixture::new("llama", FixtureDtype::F32, 32, false);
    let source = fixture.write(&directory.path().join("source"), false);
    let output = directory.path().join("model.gguf");
    let sentinel = b"existing model must survive";
    fs::write(&output, sentinel).unwrap();
    let before = directory_entries(directory.path());
    let error = convert_safetensors(&source, &output, &WriterOptions::default())
        .unwrap_err()
        .to_string();
    assert!(error.contains("output already exists"), "{error}");
    assert_eq!(fs::read(&output).unwrap(), sentinel);
    assert_eq!(directory_entries(directory.path()), before);
}

#[test]
fn publication_does_not_replace_a_destination_created_during_writing() {
    let directory = TempDir::new().unwrap();
    let fixture = Fixture::new("qwen2", FixtureDtype::F32, 32, true);
    let source = fixture.write(&directory.path().join("source"), false);
    let checkpoint = input::open(&source).unwrap();
    let model = metadata::load(&checkpoint.root).unwrap();
    let tensors = plan_tensors(&checkpoint, &model, &WriterOptions::default()).unwrap();
    let header = make_header(&model.values, &tensors).unwrap();
    let output = directory.path().join("model.gguf");
    let sentinel = b"another completed model";
    let error = publish(&output, |file| {
        write_gguf(file, &header, &tensors, &checkpoint)?;
        assert!(file.metadata().unwrap().len() > header.len() as u64);
        fs::write(&output, sentinel).map_err(io_error)
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("exists"), "{error}");
    assert_eq!(fs::read(&output).unwrap(), sentinel);
    assert_eq!(
        directory_entries(directory.path()),
        BTreeSet::from(["model.gguf".into(), "source".into()])
    );
}

struct FailAfterSink<'a> {
    file: &'a mut File,
    remaining: usize,
    written: &'a mut usize,
}

impl Write for FailAfterSink<'_> {
    fn write(&mut self, bytes: &[u8]) -> io::Result<usize> {
        if self.remaining == 0 {
            return Err(io::Error::other("injected tensor write failure"));
        }
        let count = bytes.len().min(self.remaining);
        let count = self.file.write(&bytes[..count])?;
        self.remaining -= count;
        *self.written += count;
        Ok(count)
    }

    fn flush(&mut self) -> io::Result<()> {
        self.file.flush()
    }
}

#[test]
fn write_failures_do_not_publish_or_leave_temporary_files() {
    let directory = TempDir::new().unwrap();
    let fixture = Fixture::new("qwen2", FixtureDtype::F32, 32, true);
    let source = fixture.write(&directory.path().join("source"), false);
    let checkpoint = input::open(&source).unwrap();
    let model = metadata::load(&checkpoint.root).unwrap();
    let tensors = plan_tensors(&checkpoint, &model, &WriterOptions::default()).unwrap();
    let header = make_header(&model.values, &tensors).unwrap();
    let output = directory.path().join("model.gguf");
    let before = directory_entries(directory.path());
    let mut written = 0;
    let limit = header.len() + 7;
    let error = publish(&output, |file| {
        let mut sink = FailAfterSink {
            file,
            remaining: limit,
            written: &mut written,
        };
        write_gguf(&mut sink, &header, &tensors, &checkpoint)
    })
    .unwrap_err()
    .to_string();
    assert!(error.contains("injected tensor write failure"), "{error}");
    assert_eq!(written, limit);
    assert!(written > header.len());
    assert!(!output.exists());
    assert_eq!(directory_entries(directory.path()), before);
}

fn edit_fixture_json(path: &Path, edit: impl FnOnce(&mut Value)) {
    let mut value: Value = serde_json::from_slice(&fs::read(path).unwrap()).unwrap();
    edit(&mut value);
    fs::write(path, serde_json::to_vec(&value).unwrap()).unwrap();
}

#[test]
fn nfc_normalizer_produces_identical_gguf() {
    for format in [OutputType::F16, OutputType::Q8_0] {
        let directory = TempDir::new().unwrap();
        let source_dir = directory.path().join("source");
        let source = Fixture::new("qwen2", FixtureDtype::F32, 32, true).write(&source_dir, false);
        let options = WriterOptions {
            format,
            ..WriterOptions::default()
        };
        let plain = directory.path().join("plain.gguf");
        convert_safetensors(&source, &plain, &options).unwrap();
        edit_fixture_json(&source_dir.join("tokenizer.json"), |value| {
            value["normalizer"] = json!({"type":"NFC"});
        });
        let nfc = directory.path().join("nfc.gguf");
        convert_safetensors(&source, &nfc, &options).unwrap();
        assert_eq!(fs::read(nfc).unwrap(), fs::read(plain).unwrap());
    }
}

#[test]
fn unsupported_normalizers_leave_no_output() {
    for normalizer in [
        json!({"type":"Lowercase"}),
        json!({"type":"NFC","extra":false}),
        json!({"type":"Sequence","normalizers":[{"type":"NFC"}]}),
        json!(false),
    ] {
        let directory = TempDir::new().unwrap();
        let source_dir = directory.path().join("source");
        let source = Fixture::new("qwen2", FixtureDtype::F32, 32, true).write(&source_dir, false);
        edit_fixture_json(&source_dir.join("tokenizer.json"), |value| {
            value["normalizer"] = normalizer
        });
        let before = directory_entries(directory.path());
        let output = directory.path().join("refused.gguf");
        let error = convert_safetensors(&source, &output, &WriterOptions::default()).unwrap_err();
        assert!(error.to_string().contains("normalizer"), "{error}");
        assert!(!output.exists());
        assert_eq!(directory_entries(directory.path()), before);
    }
}

#[test]
fn llama3_rope_factors_match_official_bytes() {
    // llama.cpp f2cc7282, Llama-3.2-1B: independent F32 tensor bytes.
    let golden = include_bytes!("fixtures/llama3_rope_factors_f32_le.bin");
    assert_eq!(golden.len(), 32 * 4);
    for format in [OutputType::F16, OutputType::Q8_0] {
        let directory = TempDir::new().unwrap();
        let mut fixture = Fixture::new("llama", FixtureDtype::F32, 256, false);
        fixture.config["head_dim"] = json!(64);
        fixture.config["rope_theta"] = json!(500000.0);
        fixture.config["rope_scaling"] = json!({
            "factor":32.0,"high_freq_factor":4.0,"low_freq_factor":1.0,
            "original_max_position_embeddings":8192,"rope_type":"llama3"
        });
        let source_dir = directory.path().join("source");
        let source = fixture.write(&source_dir, false);
        let options = WriterOptions {
            format,
            ..WriterOptions::default()
        };
        let output = directory.path().join("scaled.gguf");
        let summary = convert_safetensors(&source, &output, &options).unwrap();
        let parsed = Parsed::read(&output);
        let tensor = parsed.tensor("rope_freqs.weight");
        assert_eq!(tensor.dtype, GgufQuantType::F32);
        assert_eq!(tensor.shape, [32]);
        assert_eq!(parsed.data(tensor), golden.as_slice());
        assert_eq!(parsed.tensors.last().unwrap().name, "rope_freqs.weight");
        assert_eq!(summary.tensor_count, fixture.tensors.len() + 1);
        assert_eq!(
            summary.element_count,
            fixture
                .tensors
                .iter()
                .map(|t| t.values.len() as u64)
                .sum::<u64>()
                + 32
        );
        assert_eq!(summary.output_bytes, fs::metadata(&output).unwrap().len());
        edit_fixture_json(&source_dir.join("config.json"), |value| {
            value["rope_scaling"] = Value::Null
        });
        let plain = directory.path().join("plain.gguf");
        convert_safetensors(&source, &plain, &options).unwrap();
        let unscaled = Parsed::read(&plain);
        assert!(!unscaled
            .tensors
            .iter()
            .any(|t| t.name == "rope_freqs.weight"));
        assert_eq!(parsed.metadata.len(), unscaled.metadata.len());
        for (key, value) in &parsed.metadata {
            assert_eq!(
                format!("{value:?}"),
                format!("{:?}", unscaled.metadata[key]),
                "{key}"
            );
        }
        for source_tensor in &fixture.tensors {
            let name = &source_tensor.output_name;
            assert_eq!(
                parsed.data(parsed.tensor(name)),
                unscaled.data(unscaled.tensor(name)),
                "{name}"
            );
        }
    }
}

#[test]
fn llama3_rope_defaults_and_legacy_type_are_equivalent() {
    let directory = TempDir::new().unwrap();
    let mut fixture = Fixture::new("llama", FixtureDtype::F32, 256, false);
    fixture.config["rope_scaling"] = json!({"type":"llama3"});
    let source_dir = directory.path().join("source");
    let source = fixture.write(&source_dir, false);
    let legacy = directory.path().join("legacy.gguf");
    convert_safetensors(&source, &legacy, &WriterOptions::default()).unwrap();
    edit_fixture_json(&source_dir.join("config.json"), |value| {
        value["head_dim"] = json!(64);
        value["rope_scaling"] = json!({"rope_type":"llama3","factor":8.0,
            "low_freq_factor":1.0,"high_freq_factor":4.0,"original_max_position_embeddings":8192});
    });
    let explicit = directory.path().join("explicit.gguf");
    convert_safetensors(&source, &explicit, &WriterOptions::default()).unwrap();
    assert_eq!(fs::read(legacy).unwrap(), fs::read(explicit).unwrap());
}

#[test]
fn unsupported_rope_scaling_leaves_no_output() {
    for scaling in [
        json!({"rope_type":"yarn","factor":2.0}),
        json!({"rope_type":"linear","factor":2.0}),
        json!({"rope_type":"dynamic","factor":2.0}),
        json!({"rope_type":"longrope","factor":2.0}),
        json!({"rope_type":"llama3","unknown_extension":false}),
        json!({"rope_type":"llama3","type":"yarn"}),
        json!({"rope_type":"llama3","factor":0.0}),
        json!({"rope_type":"llama3","factor":"32"}),
        json!({"rope_type":"llama3","low_freq_factor":4.0,"high_freq_factor":1.0}),
        json!({"rope_type":"llama3","original_max_position_embeddings":0}),
        json!([]),
    ] {
        let directory = TempDir::new().unwrap();
        let mut fixture = Fixture::new("llama", FixtureDtype::F32, 32, false);
        fixture.config["rope_scaling"] = scaling;
        let source = fixture.write(&directory.path().join("source"), false);
        let before = directory_entries(directory.path());
        let output = directory.path().join("refused.gguf");
        let error = convert_safetensors(&source, &output, &WriterOptions::default()).unwrap_err();
        assert!(error.to_string().contains("rope_scaling"), "{error}");
        assert!(!output.exists());
        assert_eq!(directory_entries(directory.path()), before);
    }
    let directory = TempDir::new().unwrap();
    let mut fixture = Fixture::new("qwen2", FixtureDtype::F32, 32, true);
    fixture.config["rope_scaling"] = json!({"rope_type":"llama3"});
    let source = fixture.write(&directory.path().join("source"), false);
    let output = directory.path().join("refused.gguf");
    assert!(convert_safetensors(&source, &output, &WriterOptions::default()).is_err());
    assert!(!output.exists());
}

#[test]
fn added_token_types_match_official_predicate() {
    let directory = TempDir::new().unwrap();
    let source_dir = directory.path().join("source");
    let source = Fixture::new("qwen2", FixtureDtype::F32, 32, true).write(&source_dir, false);
    let cases = [
        ("<|fim_prefix|>", false, 3),
        ("<|fim_middle|>", false, 3),
        ("<|fim_suffix|>", false, 3),
        ("<|fim_pad|>", false, 3),
        ("<|repo_name|>", false, 3),
        ("<|file_sep|>", false, 3),
        ("<tool_call>", false, 4),
        ("</tool_call>", false, 4),
        ("<pad>", false, 3),
        ("<mask>", false, 3),
        ("<2mass>", false, 3),
        ("[@BOS@]", false, 3),
        ("<｜control｜>", false, 3),
        ("<unused0>", false, 3),
        ("ordinary", true, 3),
        ("ordinary-user", false, 4),
    ];
    let mut added = Vec::new();
    edit_fixture_json(&source_dir.join("tokenizer.json"), |value| {
        let vocab = value["model"]["vocab"].as_object_mut().unwrap();
        for (index, (content, special, _)) in cases.iter().enumerate() {
            let id = index + 10;
            vocab.remove(&format!("token{id}"));
            vocab.insert((*content).into(), json!(id));
            added.push(json!({"id":id,"content":content,"special":special}));
        }
        value["added_tokens"] = json!(added);
    });
    edit_fixture_json(&source_dir.join("tokenizer_config.json"), |value| {
        let decoder: Map<String, Value> = added
            .iter()
            .map(|token| (token["id"].as_u64().unwrap().to_string(), token.clone()))
            .collect();
        value["added_tokens_decoder"] = Value::Object(decoder);
    });
    let output = directory.path().join("tokens.gguf");
    convert_safetensors(&source, &output, &WriterOptions::default()).unwrap();
    let parsed = Parsed::read(&output);
    let types = parsed.metadata["tokenizer.ggml.token_type"]
        .as_array()
        .unwrap();
    let tokens = parsed.metadata["tokenizer.ggml.tokens"].as_array().unwrap();
    for (index, (content, _, kind)) in cases.iter().enumerate() {
        assert_eq!(
            types[index + 10].as_i64(),
            Some(i64::from(*kind)),
            "{content}"
        );
        assert_eq!(tokens[index + 10].as_str(), Some(*content));
    }
}

#[test]
fn q8_default_quantizes_token_embeddings() {
    assert!(!WriterOptions::default().keep_embedding_f16);
    let directory = TempDir::new().unwrap();
    let source = Fixture::new("llama", FixtureDtype::F32, 32, false)
        .write(&directory.path().join("source"), false);
    let output = directory.path().join("default-q8.gguf");
    let options = WriterOptions {
        format: OutputType::Q8_0,
        ..WriterOptions::default()
    };
    convert_safetensors(&source, &output, &options).unwrap();
    let parsed = Parsed::read(&output);
    assert_eq!(
        parsed.tensor("token_embd.weight").dtype,
        GgufQuantType::Q8_0
    );
    assert_eq!(parsed.tensor("output.weight").dtype, GgufQuantType::F16);
}

fn write_smollm_fixture(directory: &Path) -> PathBuf {
    let metadata: Value =
        serde_json::from_str(include_str!("fixtures/smollm2_metadata.json")).unwrap();
    let mut fixture = Fixture::new("llama", FixtureDtype::BF16, 96, false);
    fixture.config = metadata["config"].clone();
    for (key, value) in [
        ("hidden_size", 96),
        ("intermediate_size", 64),
        ("num_hidden_layers", 1),
        ("num_attention_heads", 3),
        ("num_key_value_heads", 1),
        ("vocab_size", 32),
    ] {
        fixture.config[key] = json!(value);
    }
    fixture
        .tensors
        .retain(|tensor| tensor.output_name != "output.weight");
    for tensor in &mut fixture.tensors {
        if ["blk.0.attn_k.weight", "blk.0.attn_v.weight"].contains(&tensor.output_name.as_str()) {
            tensor.shape[0] = 32;
            tensor.values.truncate(32 * 96);
        }
    }
    let source = fixture.write(directory, false);
    edit_fixture_json(&directory.join("tokenizer.json"), |value| {
        let mut vocab = Map::new();
        for token in metadata["tokenizer"]["added_tokens"].as_array().unwrap() {
            vocab.insert(
                token["content"].as_str().unwrap().into(),
                token["id"].clone(),
            );
        }
        for (content, id) in [("a", 17), ("b", 18), ("ab", 19)] {
            vocab.insert(content.into(), json!(id));
        }
        for id in 20..32 {
            vocab.insert(format!("token{id}"), json!(id));
        }
        *value = metadata["tokenizer"].clone();
        value["model"] = metadata["bpe_options"].clone();
        value["model"]["vocab"] = Value::Object(vocab);
        value["model"]["merges"] = json!(["a b"]);
    });
    for name in [
        "tokenizer_config",
        "special_tokens_map",
        "generation_config",
    ] {
        fs::write(
            directory.join(format!("{name}.json")),
            serde_json::to_vec(&metadata[name]).unwrap(),
        )
        .unwrap();
    }
    source
}

#[test]
fn smollm_pretokenizer_profile_is_supported() {
    let directory = TempDir::new().unwrap();
    write_smollm_fixture(directory.path());
    let metadata = metadata::load(directory.path()).unwrap();
    assert_eq!(
        metadata.values["tokenizer.ggml.pre"].as_str(),
        Some("smollm")
    );
    assert_eq!(
        metadata.values["tokenizer.ggml.model"].as_str(),
        Some("gpt2")
    );
    assert_eq!(
        metadata.values["tokenizer.ggml.add_space_prefix"].as_bool(),
        Some(false)
    );
}

#[test]
fn unsupported_digits_profiles_leave_no_output() {
    let byte =
        json!({"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true});
    let digits = json!({"type":"Digits","individual_digits":true});
    for parts in [
        json!([{"type":"Digits","individual_digits":false}, byte]),
        json!([{"type":"Digits"}, byte]),
        json!([{"type":"Digits","individual_digits":null}, byte]),
        json!([byte, digits]),
        json!([digits, byte, byte]),
        json!([{"type":"Digits","individual_digits":true,"unknown":false}, byte]),
        json!([digits, {"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":false}]),
        json!([digits, {"type":"ByteLevel","add_prefix_space":true,"trim_offsets":true,"use_regex":true}]),
    ] {
        let directory = TempDir::new().unwrap();
        let source_dir = directory.path().join("source");
        let source = write_smollm_fixture(&source_dir);
        edit_fixture_json(&source_dir.join("tokenizer.json"), |value| {
            value["pre_tokenizer"] = json!({"type":"Sequence","pretokenizers":parts});
        });
        let before = directory_entries(directory.path());
        let output = directory.path().join("refused.gguf");
        let error = convert_safetensors(&source, &output, &WriterOptions::default()).unwrap_err();
        let message = error.to_string();
        assert!(
            message.contains("unsupported") || message.contains("individual_digits"),
            "{message}"
        );
        assert!(!output.exists());
        assert_eq!(directory_entries(directory.path()), before);
    }
}

#[test]
fn smollm_full_conversion_preserves_configuration_and_token_types() {
    for format in [OutputType::F16, OutputType::Q8_0] {
        let directory = TempDir::new().unwrap();
        let source_dir = directory.path().join("source");
        let source = write_smollm_fixture(&source_dir);
        let output = directory.path().join("smollm.gguf");
        let summary = convert_safetensors(
            &source,
            &output,
            &WriterOptions {
                format,
                ..WriterOptions::default()
            },
        )
        .unwrap();
        let parsed = Parsed::read(&output);
        assert_eq!(summary.tensor_count, 11);
        assert_eq!(summary.output_bytes, fs::metadata(&output).unwrap().len());
        assert!(!parsed
            .tensors
            .iter()
            .any(|t| t.name == "output.weight" || t.name == "rope_freqs.weight"));
        assert_eq!(parsed.tensor("blk.0.attn_q.weight").shape, [96, 96]);
        assert_eq!(parsed.tensor("blk.0.attn_k.weight").shape, [96, 32]);
        assert_eq!(
            parsed.metadata["general.architecture"].as_str(),
            Some("llama")
        );
        assert_eq!(parsed.metadata["llama.context_length"].as_u64(), Some(8192));
        assert_eq!(
            parsed.metadata["llama.rope.freq_base"].as_f32(),
            Some(100000.0)
        );
        assert_eq!(
            parsed.metadata["llama.attention.head_count"].as_u64(),
            Some(3)
        );
        assert_eq!(
            parsed.metadata["llama.attention.head_count_kv"].as_u64(),
            Some(1)
        );
        assert_eq!(
            parsed.metadata["tokenizer.ggml.pre"].as_str(),
            Some("smollm")
        );
        for (role, id) in [("bos", 1), ("eos", 2), ("unknown", 0), ("padding", 2)] {
            assert_eq!(
                parsed.metadata[&format!("tokenizer.ggml.{role}_token_id")].as_u64(),
                Some(id)
            );
        }
        for role in ["bos", "eos"] {
            assert_eq!(
                parsed.metadata[&format!("tokenizer.ggml.add_{role}_token")].as_bool(),
                Some(false)
            );
        }
        let types = parsed.metadata["tokenizer.ggml.token_type"]
            .as_array()
            .unwrap();
        for (id, kind) in types.iter().enumerate() {
            assert_eq!(kind.as_i64(), Some(if id < 17 { 3 } else { 1 }));
        }
        let reference: Value =
            serde_json::from_str(include_str!("fixtures/smollm2_metadata.json")).unwrap();
        assert_eq!(
            parsed.metadata["tokenizer.chat_template"].as_str(),
            reference["tokenizer_config"]["chat_template"].as_str()
        );
        assert_eq!(
            parsed.tensor("token_embd.weight").dtype,
            if format == OutputType::F16 {
                GgufQuantType::F16
            } else {
                GgufQuantType::Q8_0
            }
        );
        assert_eq!(
            parsed.tensor("output_norm.weight").dtype,
            GgufQuantType::F32
        );
        for tensor in &parsed.tensors {
            assert!(parsed
                .values(&tensor.name)
                .iter()
                .all(|value| value.is_finite()));
        }
    }
}
