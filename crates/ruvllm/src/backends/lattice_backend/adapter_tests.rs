use super::*;
use crate::backends::AdapterInfo;
use crate::error::ModelAdapterError;
use lattice_inference::measurement::gpu_test_lock;
use serde_json::{json, Map, Value};
use std::fs::{self, File};
use std::io::{BufWriter, Write};
use tempfile::TempDir;

const HIDDEN: usize = 512;
const INTERMEDIATE: usize = 64;
const VOCAB: usize = 64;
const RANK: usize = 4;
const LAYER: &str = "model.language_model.layers.0";
const DOWN: &str = "base_model.model.model.layers.0.mlp.down_proj";
const PROMPT: &str = "aa";

struct Tensor {
    name: String,
    shape: Vec<usize>,
    values: Vec<f32>,
}

impl Tensor {
    fn sparse(name: impl Into<String>, shape: &[usize], entries: &[(usize, f32)]) -> Self {
        let mut values = vec![0.0; shape.iter().product()];
        for &(index, value) in entries {
            values[index] = value;
        }
        Self {
            name: name.into(),
            shape: shape.to_vec(),
            values,
        }
    }
}

fn write_safetensors(path: &Path, tensors: &[Tensor], alpha: Option<f32>) {
    let mut header = Map::new();
    if let Some(alpha) = alpha {
        header.insert("__metadata__".into(), json!({"alpha":alpha.to_string()}));
    }
    let mut offset = 0;
    for tensor in tensors {
        assert_eq!(tensor.values.len(), tensor.shape.iter().product::<usize>());
        let end = offset + tensor.values.len() * 4;
        assert!(header
            .insert(
                tensor.name.clone(),
                json!({"dtype":"F32","shape":tensor.shape,"data_offsets":[offset,end]}),
            )
            .is_none());
        offset = end;
    }
    let mut header = serde_json::to_vec(&Value::Object(header)).unwrap();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut file = BufWriter::new(File::create(path).unwrap());
    file.write_all(&(header.len() as u64).to_le_bytes())
        .unwrap();
    file.write_all(&header).unwrap();
    for tensor in tensors {
        for value in &tensor.values {
            file.write_all(&value.to_le_bytes()).unwrap();
        }
    }
    file.flush().unwrap();
}

fn model_tensors() -> Vec<Tensor> {
    let embeddings = [(HIDDEN, 1.0), (2 * HIDDEN + 1, 1.0)];
    let mut tensors = vec![
        Tensor::sparse(
            "model.language_model.embed_tokens.weight",
            &[VOCAB, HIDDEN],
            &embeddings,
        ),
        Tensor::sparse("model.language_model.norm.weight", &[HIDDEN], &[]),
    ];
    for name in ["input_layernorm", "post_attention_layernorm"] {
        tensors.push(Tensor::sparse(
            format!("{LAYER}.{name}.weight"),
            &[HIDDEN],
            &[],
        ));
    }
    for (name, shape) in [
        ("q_proj", vec![1024, HIDDEN]),
        ("k_proj", vec![256, HIDDEN]),
        ("v_proj", vec![256, HIDDEN]),
        ("o_proj", vec![HIDDEN, 512]),
        ("q_norm", vec![256]),
        ("k_norm", vec![256]),
    ] {
        tensors.push(Tensor::sparse(
            format!("{LAYER}.self_attn.{name}.weight"),
            &shape,
            &[],
        ));
    }
    for name in ["gate_proj", "up_proj"] {
        tensors.push(Tensor::sparse(
            format!("{LAYER}.mlp.{name}.weight"),
            &[INTERMEDIATE, HIDDEN],
            &[(0, 1.0 / 16.0)],
        ));
    }
    tensors.push(Tensor::sparse(
        format!("{LAYER}.mlp.down_proj.weight"),
        &[HIDDEN, INTERMEDIATE],
        &[],
    ));
    assert_eq!(tensors.len(), 13);
    tensors
}

struct Fixture {
    directory: TempDir,
    model: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("model");
        fs::create_dir(&model).unwrap();
        let model = fs::canonicalize(model).unwrap();
        let config = json!({
            "model_type":"qwen3_5",
            "torch_dtype":"float32",
            "tie_word_embeddings":true,
            "text_config":{
                "model_type":"qwen3_5_text",
                "hidden_size":HIDDEN,
                "intermediate_size":INTERMEDIATE,
                "num_hidden_layers":1,
                "vocab_size":VOCAB,
                "rms_norm_eps":0.000001,
                "num_attention_heads":2,
                "num_key_value_heads":1,
                "head_dim":256,
                "rope_theta":10000000.0,
                "partial_rotary_factor":0.25,
                "rope_parameters":null,
                "linear_num_key_heads":1,
                "linear_num_value_heads":1,
                "linear_key_head_dim":16,
                "linear_value_head_dim":16,
                "linear_conv_kernel_dim":4,
                "num_experts":null,
                "num_experts_per_tok":null,
                "moe_intermediate_size":null,
                "shared_expert_intermediate_size":null,
                "output_router_logits":false,
                "router_aux_loss_coef":null,
                "tie_word_embeddings":true,
                "full_attention_interval":1,
                "layer_types":["full_attention"],
                "layer_mask":[true],
                "eos_token_id":63,
                "max_position_embeddings":128,
                "mtp_num_hidden_layers":0,
                "mtp_use_dedicated_embeddings":false,
                "quarot_rotation_seed":null,
                "vision_config":null,
                "image_token_id":null,
                "video_token_id":null,
                "vision_start_token_id":null,
                "vision_end_token_id":null
            }
        });
        fs::write(model.join("config.json"), config.to_string()).unwrap();
        let mut vocabulary = Map::new();
        vocabulary.insert("<unk>".into(), json!(0));
        vocabulary.insert("a".into(), json!(1));
        vocabulary.insert("b".into(), json!(2));
        for id in 3..63 {
            vocabulary.insert(format!("token{id}"), json!(id));
        }
        vocabulary.insert("<|endoftext|>".into(), json!(63));
        let tokenizer = json!({
            "version":"1.0",
            "truncation":null,
            "padding":null,
            "normalizer":null,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "post_processor":null,
            "decoder":{"type":"ByteLevel","add_prefix_space":false,"trim_offsets":true,"use_regex":true},
            "added_tokens":[{"id":63,"content":"<|endoftext|>","special":true,"single_word":false,"lstrip":false,"rstrip":false,"normalized":false}],
            "model":{
                "type":"BPE","dropout":null,"unk_token":"<unk>",
                "continuing_subword_prefix":null,"end_of_word_suffix":null,
                "fuse_unk":false,"byte_fallback":false,"ignore_merges":false,
                "vocab":vocabulary,"merges":[]
            }
        });
        fs::write(model.join("tokenizer.json"), tokenizer.to_string()).unwrap();
        write_safetensors(&model.join("model.safetensors"), &model_tensors(), None);
        Self { directory, model }
    }

    fn backend(&self) -> LatticeBackend {
        self.backend_named(self.model.to_str().unwrap())
    }

    fn backend_named(&self, model_id: &str) -> LatticeBackend {
        let mut backend = LatticeBackend::new().unwrap();
        backend
            .load_model(
                model_id,
                ModelConfig {
                    architecture: ModelArchitecture::Qwen,
                    max_sequence_length: 32,
                    ..ModelConfig::default()
                },
            )
            .expect("the native adapter test requires a working Metal device and model load");
        assert!(backend.is_model_loaded());
        let info = backend.model_info().unwrap();
        assert_eq!(
            (info.hidden_size, info.vocab_size, info.num_layers),
            (HIDDEN, VOCAB, 1)
        );
        assert_eq!(backend.tokenizer().unwrap().encode(PROMPT).unwrap(), [1, 1]);
        assert_eq!(backend.active_lora_adapter().unwrap(), None);
        backend
    }

    fn replace_weights(&self, entries: &[(&str, usize, f32)]) {
        let mut tensors = model_tensors();
        for &(projection, index, value) in entries {
            let name = format!("{LAYER}.{projection}.weight");
            let tensor = tensors
                .iter_mut()
                .find(|tensor| tensor.name == name)
                .unwrap();
            tensor.values[index] = value;
        }
        write_safetensors(&self.model.join("model.safetensors"), &tensors, None);
    }

    fn projection_adapter(&self, name: &str, module: &str, row: usize, b: f32) -> PathBuf {
        let (block, output) = match module {
            "q_proj" => ("self_attn", 2 * HIDDEN),
            "gate_proj" => ("mlp", INTERMEDIATE),
            _ => panic!("unsupported fixture projection {module}"),
        };
        assert!(row < output);
        let path = self.adapter(name, 0.0, RANK as f32);
        let config_path = path.join("adapter_config.json");
        let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
        config["target_modules"] = json!([module]);
        fs::write(config_path, config.to_string()).unwrap();
        let prefix = format!("base_model.model.model.layers.0.{block}.{module}");
        write_safetensors(
            &path.join("adapter_model.safetensors"),
            &[
                Tensor::sparse(
                    format!("{prefix}.lora_A.weight"),
                    &[RANK, HIDDEN],
                    &[(0, 1.0)],
                ),
                Tensor::sparse(
                    format!("{prefix}.lora_B.weight"),
                    &[output, RANK],
                    &[(row * RANK, b)],
                ),
            ],
            None,
        );
        path
    }

    fn adapter(&self, name: &str, b: f32, alpha: f32) -> PathBuf {
        let path = self.directory.path().join(name);
        fs::create_dir(&path).unwrap();
        let config = json!({
            "peft_type":"LORA",
            "task_type":"CAUSAL_LM",
            "inference_mode":true,
            "base_model_name_or_path":self.model.to_str().unwrap(),
            "r":RANK,
            "lora_alpha":alpha,
            "target_modules":["down_proj"],
            "bias":"none",
            "fan_in_fan_out":false,
            "use_rslora":false,
            "use_dora":false,
            "modules_to_save":null
        });
        fs::write(path.join("adapter_config.json"), config.to_string()).unwrap();
        write_safetensors(
            &path.join("adapter_model.safetensors"),
            &adapter_tensors(b),
            Some(123.0),
        );
        path
    }
}

fn adapter_tensors(b: f32) -> Vec<Tensor> {
    vec![
        Tensor::sparse(
            format!("{DOWN}.lora_A.weight"),
            &[RANK, INTERMEDIATE],
            &[(0, 1.0)],
        ),
        Tensor::sparse(
            format!("{DOWN}.lora_B.weight"),
            &[HIDDEN, RANK],
            &[(RANK, b)],
        ),
    ]
}

fn greedy_params() -> GenerateParams {
    GenerateParams {
        max_tokens: 2,
        temperature: 0.0,
        top_p: 1.0,
        top_k: 0,
        repetition_penalty: 1.0,
        frequency_penalty: 0.0,
        presence_penalty: 0.0,
        stop_sequences: Vec::new(),
        seed: Some(7),
    }
}

fn probe(backend: &LatticeBackend, winner: usize) -> Vec<f32> {
    let logits = backend.adapter_logits(vec![1, 1]).unwrap();
    assert_eq!(logits.len(), VOCAB);
    assert!(logits.iter().all(|value| value.is_finite()));
    for (index, &value) in logits.iter().enumerate() {
        if index != winner {
            assert!(
                logits[winner] > value + 1.0,
                "token {winner} must beat {index}: {logits:?}"
            );
        }
    }
    logits
}

fn bits(logits: &[f32]) -> Vec<u32> {
    logits.iter().map(|value| value.to_bits()).collect()
}

fn assert_projection_effect(
    backend: &mut LatticeBackend,
    zero: &Path,
    adapter: &Path,
    module: &str,
) {
    let base = probe(backend, 1);
    assert_generation(backend, 1, "aa");
    backend.load_lora_adapter(zero, 1.0).unwrap();
    assert_eq!(
        backend
            .active_lora_adapter()
            .unwrap()
            .unwrap()
            .target_modules,
        [module]
    );
    assert_eq!(bits(&probe(backend, 1)), bits(&base), "zero B control");
    backend.load_lora_adapter(adapter, 1.0).unwrap();
    let active = backend.active_lora_adapter().unwrap().unwrap();
    assert_eq!(active.target_modules, [module]);
    assert_eq!(active.rank, RANK);
    assert_eq!(
        active.path,
        fs::canonicalize(adapter.join("adapter_model.safetensors")).unwrap()
    );
    let adapted = probe(backend, 1);
    assert_ne!(
        bits(&adapted),
        bits(&base),
        "{module} must change native logits"
    );
    let effect = adapted
        .iter()
        .zip(&base)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        effect > 1.0,
        "{module} effect must exceed rounding noise: {effect}"
    );
    assert_eq!(bits(&probe(backend, 1)), bits(&adapted));
    assert_generation(backend, 1, "aa");
    backend.unload_lora_adapter().unwrap();
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(
        bits(&probe(backend, 1)),
        bits(&base),
        "{module} unload must restore every logit bit"
    );
    assert_generation(backend, 1, "aa");
}

#[test]
fn q_projection_adapter_changes_native_logits_and_unload_restores_bits() {
    let _gpu_guard = gpu_test_lock();
    let fixture = Fixture::new();
    fixture.replace_weights(&[
        ("self_attn.v_proj", 0, 1.0 / 32.0),
        ("self_attn.o_proj", HIDDEN, 1.0),
    ]);
    // Each full-attention head interleaves Q then its gate; row 256 is head zero's first gate.
    let adapter = fixture.projection_adapter("q-adapter", "q_proj", 256, 1.0);
    let zero = fixture.projection_adapter("q-zero", "q_proj", 256, 0.0);
    let mut backend = fixture.backend();
    assert_projection_effect(&mut backend, &zero, &adapter, "q_proj");
}

#[test]
fn gate_projection_adapter_changes_native_logits_and_unload_restores_bits() {
    let _gpu_guard = gpu_test_lock();
    let fixture = Fixture::new();
    fixture.replace_weights(&[("mlp.down_proj", INTERMEDIATE, 1.0 / 16.0)]);
    let adapter = fixture.projection_adapter("gate-adapter", "gate_proj", 0, 1.0 / 16.0);
    let zero = fixture.projection_adapter("gate-zero", "gate_proj", 0, 0.0);
    let mut backend = fixture.backend();
    assert_projection_effect(&mut backend, &zero, &adapter, "gate_proj");
}

#[test]
fn adapter_context_refusal_does_not_prevent_base_model_load() {
    let _gpu_guard = gpu_test_lock();
    let fixture = Fixture::new();
    let config_path = fixture.model.join("config.json");
    let original = fs::read_to_string(&config_path).unwrap();
    let duplicated = format!(
        "{{\"adapter_probe\":0,\"adapter_probe\":1,{}",
        original.strip_prefix('{').unwrap()
    );
    fs::write(&config_path, duplicated).unwrap();
    assert!(Qwen35Config::from_config_json(&config_path).is_ok());
    let adapter = fixture.adapter("cached-refusal", 2.0, 4.0);
    let mut backend = fixture.backend();
    let base = probe(&backend, 1);
    assert_generation(&backend, 1, "aa");
    fs::write(&config_path, original).unwrap();
    let error = backend
        .load_lora_adapter(&adapter, 1.0)
        .expect_err("the captured context error must survive config repair");
    match error {
        ModelAdapterError::InvalidFormat { message } => {
            assert!(message.contains("duplicate key adapter_probe"), "{message}")
        }
        other => panic!("expected the captured duplicate-key error, got {other:?}"),
    }
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));
    assert_generation(&backend, 1, "aa");
}

#[test]
fn model_info_preserves_caller_path_spelling() {
    let _gpu_guard = gpu_test_lock();
    let fixture = Fixture::new();
    let caller = format!("{}/.", fixture.model.display());
    let backend = fixture.backend_named(&caller);
    assert_eq!(backend.model_info().unwrap().name, caller);
    assert_eq!(backend.model_id, fixture.model.display().to_string());
    assert_generation(&backend, 1, "aa");
}

fn assert_generation(backend: &LatticeBackend, expected_id: u32, expected_text: &str) {
    assert_eq!(
        backend.generate(PROMPT, greedy_params()).unwrap(),
        expected_text
    );
    let stream = backend.generate_stream_v2(PROMPT, greedy_params()).unwrap();
    let mut ids = Vec::new();
    let mut text = String::new();
    let mut completions = 0;
    for event in stream {
        match event.unwrap() {
            StreamEvent::Token(token) => {
                assert!(!token.is_special);
                ids.push(token.id);
                text.push_str(&token.text);
            }
            StreamEvent::Done { total_tokens, .. } => {
                assert_eq!(total_tokens, 2);
                completions += 1;
            }
            StreamEvent::Error(message) => panic!("native generation failed: {message}"),
        }
    }
    assert_eq!(completions, 1);
    assert_eq!(ids, [expected_id, expected_id]);
    assert_eq!(text, expected_text);
}

fn assert_active(
    backend: &LatticeBackend,
    directory: &Path,
    scale: f32,
    alpha: f32,
) -> AdapterInfo {
    let info = backend
        .active_lora_adapter()
        .unwrap()
        .expect("adapter must be active");
    assert_eq!(
        info.path,
        fs::canonicalize(directory.join("adapter_model.safetensors")).unwrap()
    );
    assert_eq!(info.scale.to_bits(), scale.to_bits());
    assert_eq!(info.rank, RANK);
    assert_eq!(info.alpha.to_bits(), alpha.to_bits());
    assert_eq!(info.target_modules, ["down_proj"]);
    info
}

fn assert_unloaded_adapter_operations(backend: &mut LatticeBackend) {
    assert!(!backend.is_model_loaded());
    for result in [
        backend.load_lora_adapter(Path::new("unused-adapter.safetensors"), 1.0),
        backend.unload_lora_adapter(),
        backend.active_lora_adapter().map(|_| ()),
    ] {
        assert!(
            matches!(result, Err(ModelAdapterError::ModelNotLoaded)),
            "unloaded adapter operation must return ModelNotLoaded: {result:?}"
        );
    }
}

fn assert_invalid_scales_preserve_state(
    backend: &mut LatticeBackend,
    adapter: &Path,
    active: Option<AdapterInfo>,
    logits: &[f32],
    winner: usize,
    text: &str,
) {
    for scale in [
        f32::from_bits(0x7fc0_1234),
        f32::INFINITY,
        f32::NEG_INFINITY,
    ] {
        let error = backend
            .load_lora_adapter(adapter, scale)
            .expect_err("nonfinite adapter scale must fail");
        match error {
            ModelAdapterError::InvalidScale { scale: actual } => {
                assert_eq!(actual.to_bits(), scale.to_bits());
            }
            other => panic!("expected InvalidScale for {scale:?}, got {other:?}"),
        }
        assert_eq!(backend.active_lora_adapter().unwrap(), active);
        assert_eq!(bits(&probe(backend, winner)), bits(logits));
        assert_generation(backend, winner as u32, text);
    }
}

fn load_with_changed_base_config(backend: &mut LatticeBackend, fixture: &Fixture, adapter: &Path) {
    let path = fixture.model.join("config.json");
    let original = fs::read(&path).unwrap();
    let mut changed: Value = serde_json::from_slice(&original).unwrap();
    changed["text_config"]["hidden_size"] = json!(2 * HIDDEN);
    fs::write(&path, changed.to_string()).unwrap();
    let result = backend.load_lora_adapter(adapter, 1.0);
    fs::write(&path, &original).unwrap();
    assert_eq!(fs::read(&path).unwrap(), original);
    result.expect("adapter validation must use the loaded model's cached geometry");
    assert_active(backend, adapter, 1.0, 4.0);
    assert_generation(backend, 2, "bb");
}

#[test]
fn adapter_changes_native_logits_and_generation_and_unload_restores_bits() {
    let _gpu_guard = gpu_test_lock();
    let fixture = Fixture::new();
    let adapter = fixture.adapter("active", 2.0, 4.0);
    let zero = fixture.adapter("zero", 0.0, 4.0);
    let scaled = fixture.adapter("scaled", 2.0, 8.0);
    let mut backend = fixture.backend();

    let base = probe(&backend, 1);
    assert_generation(&backend, 1, "aa");
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));

    backend.load_lora_adapter(&zero, 1.0).unwrap();
    assert_active(&backend, &zero, 1.0, 4.0);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base), "zero B control");
    assert_generation(&backend, 1, "aa");

    backend.load_lora_adapter(&adapter, 0.0).unwrap();
    assert_active(&backend, &adapter, 0.0, 4.0);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base), "zero scale control");
    assert_generation(&backend, 1, "aa");

    backend
        .load_lora_adapter(&adapter.join("adapter_model.safetensors"), 1.0)
        .unwrap();
    assert_active(&backend, &adapter, 1.0, 4.0);
    let adapted = probe(&backend, 2);
    let effect = adapted
        .iter()
        .zip(&base)
        .map(|(a, b)| (a - b).abs())
        .fold(0.0_f32, f32::max);
    assert!(
        effect > 1.0,
        "adapter effect must exceed rounding noise: {effect}"
    );
    assert_generation(&backend, 2, "bb");
    assert_eq!(bits(&probe(&backend, 2)), bits(&adapted));

    backend.load_lora_adapter(&scaled, 0.5).unwrap();
    assert_active(&backend, &scaled, 0.5, 8.0);
    assert_eq!(
        bits(&probe(&backend, 2)),
        bits(&adapted),
        "alpha/rank and caller scale compose once"
    );
    assert_generation(&backend, 2, "bb");

    backend.unload_lora_adapter().unwrap();
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(
        bits(&probe(&backend, 1)),
        bits(&base),
        "unload must restore every logit bit"
    );
    assert_generation(&backend, 1, "aa");
    backend.unload_lora_adapter().unwrap();
    assert_eq!(
        bits(&probe(&backend, 1)),
        bits(&base),
        "unload is idempotent"
    );
    backend.unload_model();
    assert_unloaded_adapter_operations(&mut backend);
}

#[derive(Clone, Copy, Debug)]
enum Refusal {
    Missing,
    Shape,
    Rank,
    Configuration,
    Target,
    Format,
}

fn refused_candidates(fixture: &Fixture) -> Vec<(PathBuf, Refusal)> {
    let missing = fixture.directory.path().join("missing.safetensors");
    let shape = fixture.adapter("wrong-shape", 2.0, 4.0);
    let mut tensors = adapter_tensors(2.0);
    tensors[0] = Tensor::sparse(
        format!("{DOWN}.lora_A.weight"),
        &[RANK, INTERMEDIATE - 1],
        &[(0, 1.0)],
    );
    write_safetensors(
        &shape.join("adapter_model.safetensors"),
        &tensors,
        Some(4.0),
    );

    let rank = fixture.adapter("wrong-rank", 2.0, 4.0);
    let mut tensors = adapter_tensors(2.0);
    tensors[1] = Tensor::sparse(
        format!("{DOWN}.lora_B.weight"),
        &[HIDDEN, RANK - 1],
        &[(RANK - 1, 2.0)],
    );
    write_safetensors(&rank.join("adapter_model.safetensors"), &tensors, Some(4.0));

    let configuration = fixture.adapter("unsupported-semantics", 2.0, 4.0);
    let config_path = configuration.join("adapter_config.json");
    let mut config: Value = serde_json::from_slice(&fs::read(&config_path).unwrap()).unwrap();
    config["use_rslora"] = json!(true);
    fs::write(config_path, config.to_string()).unwrap();

    let target = fixture.adapter("unsupported-target", 2.0, 4.0);
    let mut tensors = adapter_tensors(2.0);
    for (suffix, shape) in [("A", vec![RANK, HIDDEN]), ("B", vec![HIDDEN, RANK])] {
        tensors.push(Tensor::sparse(
            format!("base_model.model.model.layers.0.mlp.unknown_proj.lora_{suffix}.weight"),
            &shape,
            &[(0, 1.0)],
        ));
    }
    write_safetensors(
        &target.join("adapter_model.safetensors"),
        &tensors,
        Some(4.0),
    );

    let corrupt = fixture.adapter("corrupt", 2.0, 4.0);
    fs::OpenOptions::new()
        .write(true)
        .open(corrupt.join("adapter_model.safetensors"))
        .unwrap()
        .set_len(11)
        .unwrap();
    vec![
        (missing, Refusal::Missing),
        (shape, Refusal::Shape),
        (rank, Refusal::Rank),
        (configuration, Refusal::Configuration),
        (target, Refusal::Target),
        (corrupt, Refusal::Format),
    ]
}

fn assert_refusal(error: ModelAdapterError, expected: Refusal) {
    match (expected, error) {
        (Refusal::Missing, ModelAdapterError::NotFound { path }) => assert!(!path.exists()),
        (
            Refusal::Shape,
            ModelAdapterError::ShapeMismatch {
                tensor,
                expected,
                actual,
            },
        ) => {
            assert!(!tensor.is_empty());
            assert_ne!(expected, actual);
        }
        (Refusal::Rank, ModelAdapterError::RankMismatch { message }) => {
            assert!(!message.is_empty())
        }
        (Refusal::Configuration, ModelAdapterError::InvalidFormat { message }) => {
            assert!(message.contains("use_rslora"))
        }
        (Refusal::Target, ModelAdapterError::UnsupportedTarget { target }) => {
            assert!(!target.is_empty())
        }
        (Refusal::Format, ModelAdapterError::InvalidFormat { message }) => {
            assert!(!message.is_empty())
        }
        (expected, error) => panic!("wrong refusal category for {expected:?}: {error:?}"),
    }
}

#[test]
fn failed_adapter_load_preserves_base_and_active_native_state() {
    let _gpu_guard = gpu_test_lock();
    let mut unloaded = LatticeBackend::new().unwrap();
    assert_unloaded_adapter_operations(&mut unloaded);
    let fixture = Fixture::new();
    let valid = fixture.adapter("valid", 2.0, 4.0);
    let replacement = fixture.adapter("replacement", 2.0, 8.0);
    let candidates = refused_candidates(&fixture);
    let mut backend = fixture.backend();
    let base = probe(&backend, 1);
    assert_invalid_scales_preserve_state(&mut backend, &valid, None, &base, 1, "aa");
    backend.adapter_fail_next_installs(1).unwrap();
    let error = backend
        .load_lora_adapter(&valid, 1.0)
        .expect_err("native first-install failure must be reported");
    match error {
        ModelAdapterError::Backend { message } => {
            assert!(!message.is_empty());
        }
        other => panic!("expected Backend after native first-install failure, got {other:?}"),
    }
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));
    assert_generation(&backend, 1, "aa");
    load_with_changed_base_config(&mut backend, &fixture, &valid);
    let cached_context_adapted = probe(&backend, 2);
    backend.unload_lora_adapter().unwrap();
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));
    for (candidate, category) in &candidates {
        let error = backend
            .load_lora_adapter(candidate, 1.0)
            .expect_err("invalid first adapter must fail");
        assert_refusal(error, *category);
        assert_eq!(backend.active_lora_adapter().unwrap(), None);
        assert_eq!(
            bits(&probe(&backend, 1)),
            bits(&base),
            "failed first load: {category:?}"
        );
        assert_generation(&backend, 1, "aa");
    }

    backend.load_lora_adapter(&valid, 1.0).unwrap();
    let active = assert_active(&backend, &valid, 1.0, 4.0);
    let adapted = probe(&backend, 2);
    assert_ne!(bits(&adapted), bits(&base));
    assert_invalid_scales_preserve_state(
        &mut backend,
        &valid,
        Some(active.clone()),
        &adapted,
        2,
        "bb",
    );
    assert_eq!(bits(&cached_context_adapted), bits(&adapted));
    load_with_changed_base_config(&mut backend, &fixture, &valid);
    assert_eq!(backend.active_lora_adapter().unwrap(), Some(active.clone()));
    assert_eq!(bits(&probe(&backend, 2)), bits(&adapted));
    for (candidate, category) in &candidates {
        let error = backend
            .load_lora_adapter(candidate, 1.0)
            .expect_err("invalid replacement must fail");
        assert_refusal(error, *category);
        assert_eq!(backend.active_lora_adapter().unwrap(), Some(active.clone()));
        assert_eq!(
            bits(&probe(&backend, 2)),
            bits(&adapted),
            "failed replacement: {category:?}"
        );
        assert_generation(&backend, 2, "bb");
    }
    backend.adapter_fail_next_installs(1).unwrap();
    let error = backend
        .load_lora_adapter(&replacement, 0.5)
        .expect_err("candidate install failure must be reported after successful rollback");
    match error {
        ModelAdapterError::Backend { message } => {
            assert!(!message.is_empty());
        }
        other => panic!("expected Backend after native install failure, got {other:?}"),
    }
    assert_eq!(backend.active_lora_adapter().unwrap(), Some(active.clone()));
    assert_eq!(bits(&probe(&backend, 2)), bits(&adapted));
    assert_generation(&backend, 2, "bb");

    backend.adapter_fail_next_installs(2).unwrap();
    let error = backend
        .load_lora_adapter(&replacement, 0.5)
        .expect_err("candidate and rollback failures must be reported");
    match error {
        ModelAdapterError::Backend { message } => {
            assert!(
                message.contains("no active adapter"),
                "rollback failure must describe the resulting state: {message}"
            );
        }
        other => panic!("expected Backend after rollback failure, got {other:?}"),
    }
    assert_eq!(backend.active_lora_adapter().unwrap(), None);
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));
    assert_generation(&backend, 1, "aa");

    backend.load_lora_adapter(&replacement, 0.5).unwrap();
    assert_active(&backend, &replacement, 0.5, 8.0);
    assert_eq!(bits(&probe(&backend, 2)), bits(&adapted));
    assert_generation(&backend, 2, "bb");
    backend.unload_lora_adapter().unwrap();
    assert_eq!(bits(&probe(&backend, 1)), bits(&base));
    backend.unload_model();
    assert_unloaded_adapter_operations(&mut backend);
}
