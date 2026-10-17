use super::*;
use serde_json::json as value;
use std::os::unix::fs::PermissionsExt;
use tempfile::TempDir;

const A: &str = "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight";
const B: &str = "base_model.model.model.layers.0.self_attn.q_proj.lora_B.weight";

fn model_json() -> Value {
    value!({
        "_name_or_path": "fixture/base",
        "_commit_hash": "fixture-revision",
        "model_type": "qwen3_5",
        "text_config": {
            "hidden_size": 8,
            "intermediate_size": 12,
            "num_hidden_layers": 1,
            "vocab_size": 16,
            "num_attention_heads": 2,
            "num_key_value_heads": 1,
            "head_dim": 4,
            "rms_norm_eps": 0.000001,
            "rope_theta": 10000.0,
            "partial_rotary_factor": 0.5,
            "linear_num_key_heads": 1,
            "linear_num_value_heads": 1,
            "linear_key_head_dim": 2,
            "linear_value_head_dim": 2,
            "linear_conv_kernel_dim": 4,
            "num_experts": null,
            "num_experts_per_tok": null,
            "moe_intermediate_size": null,
            "shared_expert_intermediate_size": null,
            "full_attention_interval": 1,
            "layer_types": ["full_attention"],
            "layer_mask": [true],
            "eos_token_id": 15,
            "max_position_embeddings": 32,
            "mtp_num_hidden_layers": 0,
            "mtp_use_dedicated_embeddings": false,
            "quarot_rotation_seed": null
        }
    })
}

fn peft_json() -> Value {
    value!({
        "peft_type": "LORA",
        "task_type": "CAUSAL_LM",
        "base_model_name_or_path": "fixture/base",
        "r": 2,
        "lora_alpha": 4,
        "target_modules": ["q_proj"],
        "bias": "none",
        "inference_mode": true,
        "init_lora_weights": true,
        "lora_dropout": 0.0
    })
}

fn tensor_header(dtype: &str) -> Value {
    let width = if dtype == "F32" { 4 } else { 2 };
    let a_end = 16 * width;
    let b_end = a_end + 32 * width;
    let mut header = serde_json::Map::new();
    header.insert(
        A.into(),
        value!({"dtype":dtype,"shape":[2,8],"data_offsets":[0,a_end]}),
    );
    header.insert(
        B.into(),
        value!({"dtype":dtype,"shape":[16,2],"data_offsets":[a_end,b_end]}),
    );
    Value::Object(header)
}

fn safetensors(header: &Value, payload: &[u8]) -> Vec<u8> {
    raw_safetensors(&serde_json::to_vec(header).unwrap(), payload)
}

fn raw_safetensors(header: &[u8], payload: &[u8]) -> Vec<u8> {
    let mut header = header.to_vec();
    while header.len() % 8 != 0 {
        header.push(b' ');
    }
    let mut bytes = (header.len() as u64).to_le_bytes().to_vec();
    bytes.extend(header);
    bytes.extend_from_slice(payload);
    bytes
}

fn payload(dtype: &str) -> Vec<u8> {
    match dtype {
        "F32" => 0.25f32.to_le_bytes().repeat(48),
        "F16" => half::f16::from_f32(0.25).to_bits().to_le_bytes().repeat(48),
        "BF16" => half::bf16::from_f32(0.25)
            .to_bits()
            .to_le_bytes()
            .repeat(48),
        _ => panic!("fixture dtype"),
    }
}

struct Fixture {
    _directory: TempDir,
    model: PathBuf,
    adapter: PathBuf,
}

impl Fixture {
    fn new() -> Self {
        let directory = tempfile::tempdir().unwrap();
        let model = directory.path().join("base");
        let adapter = directory.path().join("adapter");
        fs::create_dir(&model).unwrap();
        fs::create_dir(&adapter).unwrap();
        let fixture = Self {
            _directory: directory,
            model,
            adapter,
        };
        fixture.write_model(&model_json());
        fixture.write_config(&peft_json());
        fixture.write_tensors(&tensor_header("F32"), &payload("F32"));
        fixture
    }

    fn write_model(&self, config: &Value) {
        fs::write(self.model.join("config.json"), config.to_string()).unwrap();
    }

    fn write_config(&self, config: &Value) {
        fs::write(self.adapter.join("adapter_config.json"), config.to_string()).unwrap();
    }

    fn write_tensors(&self, header: &Value, payload: &[u8]) {
        fs::write(self.file(), safetensors(header, payload)).unwrap();
    }

    fn file(&self) -> PathBuf {
        self.adapter.join("adapter_model.safetensors")
    }

    fn error(&self, scale: f32) -> ModelAdapterError {
        prepare(
            &ModelContext::read(&self.model).unwrap(),
            &self.adapter,
            scale,
        )
        .err()
        .expect("preflight must reject")
    }

    fn valid(&self, scale: f32) {
        assert!(
            prepare(
                &ModelContext::read(&self.model).unwrap(),
                &self.adapter,
                scale
            )
            .is_ok(),
            "valid fixture failed"
        );
    }
}

fn assert_invalid(error: ModelAdapterError) {
    assert!(
        matches!(error, ModelAdapterError::InvalidFormat { .. }),
        "{error:?}"
    );
}

#[test]
fn preflight_accepts_each_supported_dtype_and_finite_multiplier() {
    let fixture = Fixture::new();
    for dtype in ["F32", "F16", "BF16"] {
        fixture.write_tensors(&tensor_header(dtype), &payload(dtype));
        for scale in [0.0, -0.5, 1.0, 2.0] {
            fixture.valid(scale);
        }
        assert!(prepare(
            &ModelContext::read(&fixture.model).unwrap(),
            &fixture.file(),
            1.0
        )
        .is_ok());
    }
}

#[test]
fn preflight_ignores_string_sidecar_alpha_and_refuses_numeric_metadata() {
    let fixture = Fixture::new();
    let context = ModelContext::read(&fixture.model).unwrap();
    fixture.valid(1.0);
    for alpha in ["123456", "NaN", "Infinity", "not-a-number"] {
        let mut header = tensor_header("F32");
        header["__metadata__"] = value!({"format":"pt", "alpha":alpha});
        fixture.write_tensors(&header, &payload("F32"));
        let prepared = prepare(&context, &fixture.adapter, 0.5).unwrap();
        assert_eq!(prepared.info.alpha.to_bits(), 4.0f32.to_bits());
        assert_eq!(prepared.info.scale.to_bits(), 0.5f32.to_bits());
        assert_eq!(prepared.info.rank, 2);
        assert_eq!(prepared.info.target_modules, ["q_proj"]);
    }
    let mut header = tensor_header("F32");
    header["__metadata__"] = value!({"alpha": 4.0});
    fixture.write_tensors(&header, &payload("F32"));
    assert_invalid(fixture.error(1.0));
}

#[test]
fn preflight_uses_captured_model_context_after_disk_config_changes() {
    let fixture = Fixture::new();
    let context = ModelContext::read(&fixture.model).unwrap();
    assert!(prepare(&context, &fixture.adapter, 1.0).is_ok());
    let mut model = model_json();
    model["_name_or_path"] = value!("different/base");
    model["text_config"]["hidden_size"] = value!(16);
    fixture.write_model(&model);
    let prepared = prepare(&context, &fixture.adapter, 1.0).unwrap();
    assert_eq!(prepared.info.rank, 2);
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::ShapeMismatch { .. }
    ));
    model["_name_or_path"] = value!("fixture/base");
    fixture.write_model(&model);
    assert!(prepare(&context, &fixture.adapter, 1.0).is_ok());
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::ShapeMismatch { .. }
    ));
}

#[test]
fn duplicate_json_keys_are_rejected_at_every_depth() {
    for text in [
        r#"{"r":2,"r":4}"#,
        r#"{"outer":{"r":2,"r":4}}"#,
        r#"[{"r":2,"r":4}]"#,
        r#"{"r":2} trailing"#,
        r#"{"r":NaN}"#,
    ] {
        assert!(json(text.as_bytes()).is_err(), "accepted {text}");
    }
    assert_eq!(
        json(br#"{"r":2,"items":[null,true,"x"]}"#).unwrap(),
        value!({"r":2,"items":[null,true,"x"]})
    );
}

#[test]
fn preflight_rejects_invalid_header_lengths_and_json() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for bytes in [
        vec![],
        vec![0; 7],
        (MAX_JSON_BYTES + 1).to_le_bytes().to_vec(),
        16u64.to_le_bytes().to_vec(),
        raw_safetensors(b"[]", &[]),
        raw_safetensors(b"{", &[]),
        raw_safetensors(br#"{"__metadata__":{},"__metadata__":{}}"#, &[]),
    ] {
        fs::write(fixture.file(), bytes).unwrap();
        assert_invalid(fixture.error(1.0));
    }
}

#[test]
fn preflight_rejects_keys_the_upstream_loader_would_ignore_or_alias() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for bad in [
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A",
        "base_model.model.model.layers.0.self_attn.q_proj.lora_A.weight.extra",
        "base_model.model.model.layers.00.self_attn.q_proj.lora_A.weight",
        "base_model.model.model.layers.0.mlp.q_proj.lora_A.weight",
        "model.layers.0.self_attn.q_proj.lora_a",
        "unrelated.weight",
    ] {
        let mut header = tensor_header("F32");
        let tensor = header.as_object_mut().unwrap().remove(A).unwrap();
        header.as_object_mut().unwrap().insert(bad.into(), tensor);
        fixture.write_tensors(&header, &payload("F32"));
        assert!(
            matches!(
                fixture.error(1.0),
                ModelAdapterError::UnsupportedTarget { .. }
            ),
            "accepted {bad}"
        );
    }
}

#[test]
fn preflight_requires_complete_pairs_and_matching_declared_modules() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    let mut header = tensor_header("F32");
    header.as_object_mut().unwrap().remove(B);
    fixture.write_tensors(&header, &payload("F32")[..64]);
    assert_invalid(fixture.error(1.0));
    fixture.write_tensors(&tensor_header("F32"), &payload("F32"));
    let mut config = peft_json();
    config["target_modules"] = value!(["q_proj", "k_proj"]);
    fixture.write_config(&config);
    assert_invalid(fixture.error(1.0));
}

#[test]
fn preflight_rejects_a_missing_whole_layer_pair() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    let mut model = model_json();
    model["text_config"]["num_hidden_layers"] = value!(2);
    model["text_config"]["layer_types"] = value!(["full_attention", "full_attention"]);
    model["text_config"]["layer_mask"] = value!([true, true]);
    fixture.write_model(&model);
    assert_invalid(fixture.error(1.0));
    let mut header = tensor_header("F32");
    let mut second_a = header[A].clone();
    second_a["data_offsets"] = value!([192, 256]);
    let mut second_b = header[B].clone();
    second_b["data_offsets"] = value!([256, 384]);
    header
        .as_object_mut()
        .unwrap()
        .insert(A.replace("layers.0", "layers.1"), second_a);
    header
        .as_object_mut()
        .unwrap()
        .insert(B.replace("layers.0", "layers.1"), second_b);
    fixture.write_tensors(&header, &payload("F32").repeat(2));
    fixture.valid(1.0);
}

#[test]
fn preflight_rejects_out_of_range_gdn_and_masked_targets() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    let mut header = serde_json::Map::new();
    for (name, tensor) in tensor_header("F32").as_object().unwrap() {
        header.insert(name.replace("layers.0", "layers.1"), tensor.clone());
    }
    fixture.write_tensors(&Value::Object(header), &payload("F32"));
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::UnsupportedTarget { .. }
    ));
    fixture.write_tensors(&tensor_header("F32"), &payload("F32"));
    let mut model = model_json();
    model["text_config"]["layer_types"] = value!(["linear_attention"]);
    fixture.write_model(&model);
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::UnsupportedTarget { .. }
    ));
    model = model_json();
    model["text_config"]["layer_mask"] = value!([false]);
    fixture.write_model(&model);
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::UnsupportedTarget { .. }
    ));
}

#[test]
fn preflight_rejects_moe_mlp_targets_accepted_by_geometry_helper() {
    let fixture = Fixture::new();
    let mut config = peft_json();
    config["target_modules"] = value!(["down_proj"]);
    fixture.write_config(&config);
    let mut header = serde_json::Map::new();
    header.insert(
        "base_model.model.model.layers.0.mlp.down_proj.lora_A.weight".into(),
        value!({"dtype":"F32","shape":[2,12],"data_offsets":[0,96]}),
    );
    header.insert(
        "base_model.model.model.layers.0.mlp.down_proj.lora_B.weight".into(),
        value!({"dtype":"F32","shape":[8,2],"data_offsets":[96,160]}),
    );
    fixture.write_tensors(&Value::Object(header), &0.25f32.to_le_bytes().repeat(40));
    fixture.valid(1.0);
    let mut model = model_json();
    model["text_config"]["num_experts"] = value!(2);
    model["text_config"]["num_experts_per_tok"] = value!(1);
    model["text_config"]["moe_intermediate_size"] = value!(12);
    model["text_config"]["shared_expert_intermediate_size"] = value!(12);
    fixture.write_model(&model);
    assert!(matches!(
        fixture.error(1.0),
        ModelAdapterError::UnsupportedTarget { .. }
    ));
}

#[test]
fn preflight_returns_rank_and_shape_errors_before_materialization() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for shape in [value!([3, 8]), value!([0, 8])] {
        let mut header = tensor_header("F32");
        header[A]["shape"] = shape;
        fixture.write_tensors(&header, &payload("F32"));
        assert!(matches!(
            fixture.error(1.0),
            ModelAdapterError::RankMismatch { .. }
        ));
    }
    for shape in [
        value!([2, 7]),
        value!([2, 0]),
        value!([2, usize::MAX]),
        value!([16]),
    ] {
        let mut header = tensor_header("F32");
        header[A]["shape"] = shape;
        fixture.write_tensors(&header, &payload("F32"));
        assert!(matches!(
            fixture.error(1.0),
            ModelAdapterError::ShapeMismatch { .. }
        ));
    }
}

#[test]
fn preflight_rejects_invalid_ranges_and_unclaimed_data() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for offsets in [
        value!([1, 65]),
        value!([65, 64]),
        value!([0, 63]),
        value!([0, 193]),
    ] {
        let mut header = tensor_header("F32");
        header[A]["data_offsets"] = offsets;
        fixture.write_tensors(&header, &payload("F32"));
        assert_invalid(fixture.error(1.0));
    }
    let mut overlap = tensor_header("F32");
    overlap[B]["data_offsets"] = value!([0, 128]);
    fixture.write_tensors(&overlap, &payload("F32"));
    assert_invalid(fixture.error(1.0));
    let mut data = payload("F32");
    data.push(0);
    fixture.write_tensors(&tensor_header("F32"), &data);
    assert_invalid(fixture.error(1.0));
    data.truncate(191);
    fixture.write_tensors(&tensor_header("F32"), &data);
    assert_invalid(fixture.error(1.0));
}

#[test]
fn preflight_rejects_unsupported_dtype_and_extra_tensor_metadata() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for dtype in ["F64", "I8", "U8", "BOOL"] {
        let mut header = tensor_header("F32");
        header[A]["dtype"] = value!(dtype);
        fixture.write_tensors(&header, &payload("F32"));
        assert_invalid(fixture.error(1.0));
    }
    let mut header = tensor_header("F32");
    header[A]["extra"] = value!(true);
    fixture.write_tensors(&header, &payload("F32"));
    assert_invalid(fixture.error(1.0));
}

#[test]
fn preflight_rejects_nonfinite_elements_in_each_supported_dtype() {
    let fixture = Fixture::new();
    for dtype in ["F32", "F16", "BF16"] {
        fixture.write_tensors(&tensor_header(dtype), &payload(dtype));
        fixture.valid(1.0);
        for value in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY] {
            let bad = match dtype {
                "F32" => value.to_le_bytes().to_vec(),
                "F16" => half::f16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                "BF16" => half::bf16::from_f32(value).to_bits().to_le_bytes().to_vec(),
                _ => unreachable!(),
            };
            for element in [0, 47] {
                let mut data = payload(dtype);
                let start = element * bad.len();
                data[start..start + bad.len()].copy_from_slice(&bad);
                fixture.write_tensors(&tensor_header(dtype), &data);
                assert_invalid(fixture.error(1.0));
            }
        }
    }
}

#[test]
fn preflight_rejects_invalid_scale_rank_and_alpha() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for scale in [f32::NAN, f32::INFINITY, f32::NEG_INFINITY, f32::MAX] {
        assert!(matches!(
            fixture.error(scale),
            ModelAdapterError::InvalidScale { .. }
        ));
    }
    for rank in [value!(0), value!(-1), value!(1.5), value!(u64::MAX)] {
        let mut config = peft_json();
        config["r"] = rank;
        fixture.write_config(&config);
        assert!(matches!(
            fixture.error(1.0),
            ModelAdapterError::RankMismatch { .. }
        ));
    }
    for alpha in [value!(1e100), value!("NaN"), Value::Null] {
        let mut config = peft_json();
        config["lora_alpha"] = alpha;
        fixture.write_config(&config);
        assert_invalid(fixture.error(1.0));
    }
}

#[test]
fn preflight_preserves_declared_identity_metadata() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    let mut config = peft_json();
    config["base_model_name_or_path"] = value!("other/base");
    fixture.write_config(&config);
    let context = ModelContext::read(&fixture.model).unwrap();
    let prepared = prepare(&context, &fixture.adapter, 1.0).unwrap();
    assert_eq!(
        prepared.info.declared_base_model.as_deref(),
        Some("other/base")
    );
    config["base_model_name_or_path"] = value!(fs::canonicalize(&fixture.model).unwrap());
    fixture.write_config(&config);
    fixture.valid(1.0);
    config["revision"] = value!("other-revision");
    fixture.write_config(&config);
    let prepared = prepare(&context, &fixture.adapter, 1.0).unwrap();
    assert_eq!(
        prepared.info.declared_revision.as_deref(),
        Some("other-revision")
    );
    config["revision"] = value!("fixture-revision");
    fixture.write_config(&config);
    fixture.valid(1.0);
    config
        .as_object_mut()
        .unwrap()
        .remove("base_model_name_or_path");
    fixture.write_config(&config);
    let prepared = prepare(&context, &fixture.adapter, 1.0).unwrap();
    assert_eq!(prepared.info.declared_base_model, None);
    assert_eq!(
        prepared.info.declared_revision.as_deref(),
        Some("fixture-revision")
    );
    for key in ["base_model_name_or_path", "revision"] {
        config[key] = Value::Null;
    }
    fixture.write_config(&config);
    let prepared = prepare(&context, &fixture.adapter, 1.0).unwrap();
    assert_eq!(prepared.info.declared_base_model, None);
    assert_eq!(prepared.info.declared_revision, None);
    for key in ["base_model_name_or_path", "revision"] {
        let mut malformed = config.clone();
        malformed[key] = value!(42);
        fixture.write_config(&malformed);
        assert_invalid(fixture.error(1.0));
    }
}

fn write_peft_0_21_2_fixture(fixture: &Fixture) -> Value {
    let bytes = include_bytes!("fixtures/peft_0_21_2_adapter_config.json");
    let config = json(bytes).unwrap();
    assert_eq!(config.as_object().unwrap().len(), 41);
    fs::write(fixture.adapter.join("adapter_config.json"), bytes).unwrap();
    let context = ModelContext::read(&fixture.model).unwrap();
    let rank = config["r"].as_u64().unwrap() as usize;
    let mut header = serde_json::Map::new();
    let mut data = Vec::new();
    for target in config["target_modules"].as_array().unwrap() {
        let target = target.as_str().unwrap();
        let geometry = qwen35_projection_shape(&context.config, 0, target).unwrap();
        let block = if ["gate_proj", "up_proj", "down_proj"].contains(&target) {
            "mlp"
        } else {
            "self_attn"
        };
        for (side, shape) in [("A", [rank, geometry.d_in]), ("B", [geometry.d_out, rank])] {
            let begin = data.len();
            data.extend(0.25f32.to_le_bytes().repeat(shape.iter().product()));
            let name =
                format!("base_model.model.model.layers.0.{block}.{target}.lora_{side}.weight");
            assert!(header
                .insert(
                    name,
                    value!({"dtype":"F32","shape":shape,"data_offsets":[begin,data.len()]})
                )
                .is_none());
        }
    }
    assert_eq!(header.len(), 14);
    fixture.write_tensors(&Value::Object(header), &data);
    config
}

#[test]
fn preflight_accepts_exact_peft_0_21_2_configuration() {
    let fixture = Fixture::new();
    write_peft_0_21_2_fixture(&fixture);
    let prepared = prepare(
        &ModelContext::read(&fixture.model).unwrap(),
        &fixture.adapter,
        1.0,
    )
    .unwrap();
    assert_eq!(prepared.info.rank, 8);
    assert_eq!(prepared.info.alpha.to_bits(), 16.0f32.to_bits());
    assert_eq!(
        prepared.info.target_modules,
        [
            "down_proj",
            "gate_proj",
            "k_proj",
            "o_proj",
            "q_proj",
            "up_proj",
            "v_proj"
        ]
    );
    assert_eq!(
        prepared.info.declared_base_model.as_deref(),
        Some("Qwen/Qwen3.5-0.8B")
    );
    assert_eq!(prepared.info.declared_revision, None);
}

#[test]
fn preflight_refuses_active_peft_extension_configuration() {
    let fixture = Fixture::new();
    let mut config = write_peft_0_21_2_fixture(&fixture);
    fixture.valid(1.0);
    for setting in [
        value!({"x": 1}),
        value!(true),
        value!(0),
        value!(""),
        value!([null]),
    ] {
        config["kasa_config"] = setting;
        fixture.write_config(&config);
        let ModelAdapterError::InvalidFormat { message } = fixture.error(1.0) else {
            panic!("active extension must be refused as an invalid configuration");
        };
        assert!(message.contains("kasa_config"));
        assert!(message.contains("null, false, an empty object or an empty array"));
    }
    for setting in [Value::Null, value!(false), value!({}), value!([])] {
        config["kasa_config"] = setting;
        fixture.write_config(&config);
        fixture.valid(1.0);
    }
    for version in ["", "0.21.2", "future-version"] {
        config["peft_version"] = value!(version);
        fixture.write_config(&config);
        fixture.valid(1.0);
    }
}

#[test]
fn preflight_accepts_null_task_type() {
    let fixture = Fixture::new();
    let mut config = peft_json();
    for task in [Value::Null, value!("CAUSAL_LM")] {
        config["task_type"] = task;
        fixture.write_config(&config);
        fixture.valid(1.0);
    }
    config.as_object_mut().unwrap().remove("task_type");
    fixture.write_config(&config);
    fixture.valid(1.0);
    for task in [value!("SEQ_CLS"), value!(false), value!({})] {
        config["task_type"] = task;
        fixture.write_config(&config);
        assert_invalid(fixture.error(1.0));
    }
}

#[test]
fn preflight_reports_hub_identity_without_enforcing_it() {
    let fixture = Fixture::new();
    let mut model = model_json();
    model.as_object_mut().unwrap().remove("_name_or_path");
    model.as_object_mut().unwrap().remove("_commit_hash");
    fixture.write_model(&model);
    let mut config = peft_json();
    config["base_model_name_or_path"] = value!("Qwen/Qwen3.5-0.8B");
    config["revision"] = value!("declared-training-revision");
    fixture.write_config(&config);
    let prepared = prepare(
        &ModelContext::read(&fixture.model).unwrap(),
        &fixture.adapter,
        1.0,
    )
    .unwrap();
    assert_eq!(
        prepared.info.declared_base_model.as_deref(),
        Some("Qwen/Qwen3.5-0.8B")
    );
    assert_eq!(
        prepared.info.declared_revision.as_deref(),
        Some("declared-training-revision")
    );
}

#[test]
fn preflight_rejects_unsupported_peft_semantics() {
    let fixture = Fixture::new();
    fixture.valid(1.0);
    for (key, setting) in [
        ("use_rslora", value!(true)),
        ("use_dora", value!(true)),
        ("fan_in_fan_out", value!(true)),
        ("lora_bias", value!(true)),
        ("use_qalora", value!(true)),
        ("bias", value!("all")),
        ("modules_to_save", value!(["lm_head"])),
        ("rank_pattern", value!({"q_proj":4})),
        ("alpha_pattern", value!({"q_proj":8})),
        ("layer_replication", value!([[0, 1]])),
        ("trainable_token_indices", value!([1])),
        ("target_parameters", value!(["weight"])),
        ("alora_invocation_tokens", value!([1, 2])),
        ("init_lora_weights", value!("pissa")),
        ("init_lora_weights", value!("olora")),
        ("init_lora_weights", value!("corda")),
        ("layers_to_transform", value!([0])),
        ("layers_pattern", value!("layers")),
        ("exclude_modules", value!(["q_proj"])),
        ("unknown_future_semantics", value!(true)),
    ] {
        let mut config = peft_json();
        config[key] = setting;
        fixture.write_config(&config);
        assert_invalid(fixture.error(1.0));
    }
    for initializer in [value!(true), value!(false), value!("gaussian")] {
        let mut config = peft_json();
        config["init_lora_weights"] = initializer;
        fixture.write_config(&config);
        fixture.valid(1.0);
    }
}

#[test]
fn bounded_reads_report_missing_nonregular_and_oversized_files() {
    let directory = tempfile::tempdir().unwrap();
    assert!(matches!(
        read_bounded(&directory.path().join("missing"), 4).unwrap_err(),
        ModelAdapterError::NotFound { .. }
    ));
    assert!(matches!(
        read_bounded(directory.path(), 4).unwrap_err(),
        ModelAdapterError::InvalidFormat { .. }
    ));
    let path = directory.path().join("data");
    fs::write(&path, b"1234").unwrap();
    assert_eq!(read_bounded(&path, 4).unwrap(), b"1234");
    fs::write(&path, b"12345").unwrap();
    assert!(matches!(
        read_bounded(&path, 4).unwrap_err(),
        ModelAdapterError::InvalidFormat { .. }
    ));
}

#[test]
fn snapshot_is_private_exact_and_removed_on_drop() {
    let bytes = b"validated bytes";
    let snapshot = Snapshot::create(bytes).unwrap();
    let path = snapshot.path();
    let directory = path.parent().unwrap().to_owned();
    assert_eq!(fs::read(&path).unwrap(), bytes);
    assert_eq!(
        fs::metadata(&directory).unwrap().permissions().mode() & 0o777,
        0o700
    );
    assert_eq!(
        fs::metadata(&path).unwrap().permissions().mode() & 0o777,
        0o600
    );
    drop(snapshot);
    assert!(!path.exists());
    assert!(!directory.exists());
}
