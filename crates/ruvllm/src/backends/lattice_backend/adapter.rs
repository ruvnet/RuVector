use super::{AdapterInfo, MetalQwen35State, Qwen35Config};
use crate::error::ModelAdapterError;
use lattice_inference::lora_hook::qwen35_projection_shape;
use serde::Deserialize;
use serde_json::Value;
use std::collections::{BTreeMap, BTreeSet};
use std::fs::{self, File, OpenOptions};
use std::io::{Read, Write};
use std::os::unix::fs::{DirBuilderExt, OpenOptionsExt};
use std::path::{Path, PathBuf};

type Result<T> = std::result::Result<T, ModelAdapterError>;

const MAX_ADAPTER_BYTES: u64 = 1 << 30;
const MAX_JSON_BYTES: u64 = 16 << 20;
const TARGETS: &[&str] = &[
    "q_proj",
    "k_proj",
    "v_proj",
    "o_proj",
    "gate_proj",
    "up_proj",
    "down_proj",
];

fn invalid(message: impl Into<String>) -> ModelAdapterError {
    ModelAdapterError::InvalidFormat {
        message: message.into(),
    }
}

fn io_error(path: &Path, source: std::io::Error) -> ModelAdapterError {
    if source.kind() == std::io::ErrorKind::NotFound {
        ModelAdapterError::NotFound {
            path: path.to_owned(),
        }
    } else {
        ModelAdapterError::Io {
            path: path.to_owned(),
            source,
        }
    }
}

fn read_bounded(path: &Path, limit: u64) -> Result<Vec<u8>> {
    if !fs::metadata(path).map_err(|e| io_error(path, e))?.is_file() {
        return Err(invalid(format!("{} is not a regular file", path.display())));
    }
    let file = File::open(path).map_err(|e| io_error(path, e))?;
    if !file.metadata().map_err(|e| io_error(path, e))?.is_file() {
        return Err(invalid(format!("{} is not a regular file", path.display())));
    }
    let mut bytes = Vec::new();
    file.take(limit + 1)
        .read_to_end(&mut bytes)
        .map_err(|e| io_error(path, e))?;
    if bytes.len() as u64 > limit {
        return Err(invalid(format!(
            "{} exceeds the {limit}-byte limit",
            path.display()
        )));
    }
    Ok(bytes)
}

// Reject duplicate keys before either JSON consumer can silently choose a value.
struct UniqueJson(Value);
impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: serde::Deserializer<'de>>(
        deserializer: D,
    ) -> std::result::Result<Self, D::Error> {
        struct Visitor;
        impl<'de> serde::de::Visitor<'de> for Visitor {
            type Value = UniqueJson;
            fn expecting(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                f.write_str("JSON without duplicate object keys")
            }
            fn visit_bool<E: serde::de::Error>(
                self,
                v: bool,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_i64<E: serde::de::Error>(self, v: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_u64<E: serde::de::Error>(self, v: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_f64<E: serde::de::Error>(self, v: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(v)
                    .map(|n| UniqueJson(Value::Number(n)))
                    .ok_or_else(|| E::custom("nonfinite number"))
            }
            fn visit_str<E: serde::de::Error>(
                self,
                v: &str,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(v.into()))
            }
            fn visit_unit<E: serde::de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }
            fn visit_seq<A: serde::de::SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(UniqueJson(value)) = seq.next_element()? {
                    values.push(value);
                }
                Ok(UniqueJson(Value::Array(values)))
            }
            fn visit_map<A: serde::de::MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = serde_json::Map::new();
                while let Some((key, UniqueJson(value))) = map.next_entry::<String, UniqueJson>()? {
                    if values.insert(key.clone(), value).is_some() {
                        return Err(serde::de::Error::custom(format!("duplicate key {key}")));
                    }
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(Visitor)
    }
}

fn json(bytes: &[u8]) -> Result<Value> {
    serde_json::from_slice::<UniqueJson>(bytes)
        .map(|v| v.0)
        .map_err(|e| invalid(e.to_string()))
}

struct PeftConfig {
    rank: usize,
    alpha: f32,
    targets: BTreeSet<String>,
    declared_base_model: Option<String>,
    declared_revision: Option<String>,
}

fn inactive_configuration(value: &Value) -> bool {
    value.is_null()
        || value == &Value::Bool(false)
        || value.as_object().is_some_and(|value| value.is_empty())
        || value.as_array().is_some_and(|value| value.is_empty())
}

fn declared_string(config: &Value, field: &str) -> Result<Option<String>> {
    match config.get(field) {
        None | Some(Value::Null) => Ok(None),
        Some(Value::String(value)) => Ok(Some(value.clone())),
        Some(_) => Err(invalid(format!("{field} must be a string or null"))),
    }
}

fn configuration(path: &Path, scale: f32) -> Result<PeftConfig> {
    if !scale.is_finite() {
        return Err(ModelAdapterError::InvalidScale { scale });
    }
    let config = json(&read_bounded(path, MAX_JSON_BYTES)?)?;
    let object = config
        .as_object()
        .ok_or_else(|| invalid("adapter_config.json must be an object"))?;
    const CONFIG_FIELDS: &[&str] = &[
        "r",
        "lora_alpha",
        "target_modules",
        "base_model_name_or_path",
        "revision",
        "peft_type",
        "task_type",
        "bias",
        "use_rslora",
        "use_dora",
        "fan_in_fan_out",
        "lora_bias",
        "use_qalora",
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "layer_replication",
        "target_parameters",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "megatron_config",
        "arrow_config",
        "init_lora_weights",
        "inference_mode",
        "lora_dropout",
        "auto_mapping",
        "layers_to_transform",
        "layers_pattern",
        "exclude_modules",
        "megatron_core",
        "loftq_config",
        "eva_config",
        "corda_config",
        "qalora_group_size",
        "ensure_weight_tying",
    ];
    for (key, value) in object {
        if !(CONFIG_FIELDS.contains(&key.as_str())
            || inactive_configuration(value)
            || key == "peft_version" && value.is_string())
        {
            return Err(invalid(format!(
                "unsupported adapter configuration field {key}: unknown fields must be null, false, an empty object or an empty array; peft_version also accepts strings"
            )));
        }
    }
    let rank = config
        .get("r")
        .and_then(Value::as_u64)
        .and_then(|r| usize::try_from(r).ok())
        .filter(|r| *r > 0 && *r <= u32::MAX as usize)
        .ok_or_else(|| ModelAdapterError::RankMismatch {
            message: "r must be a positive u32 rank".into(),
        })?;
    let alpha = config
        .get("lora_alpha")
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid("lora_alpha must be a finite number"))? as f32;
    if !alpha.is_finite() {
        return Err(invalid("lora_alpha must fit finite f32"));
    }
    if !(alpha * scale).is_finite() {
        return Err(ModelAdapterError::InvalidScale { scale });
    }
    for (key, expected) in [("peft_type", "LORA"), ("bias", "none")] {
        if let Some(value) = object.get(key) {
            if value.as_str() != Some(expected) {
                return Err(invalid(format!("unsupported {key}: {value}")));
            }
        }
    }
    if let Some(value) = object.get("task_type") {
        if !value.is_null() && value.as_str() != Some("CAUSAL_LM") {
            return Err(invalid(format!("unsupported task_type: {value}")));
        }
    }
    for key in [
        "use_rslora",
        "use_dora",
        "fan_in_fan_out",
        "lora_bias",
        "use_qalora",
    ] {
        if let Some(value) = object.get(key) {
            if value != &Value::Bool(false) {
                return Err(invalid(format!("unsupported {key}: {value}")));
            }
        }
    }
    for key in [
        "rank_pattern",
        "alpha_pattern",
        "modules_to_save",
        "layer_replication",
        "target_parameters",
        "trainable_token_indices",
        "alora_invocation_tokens",
        "megatron_config",
        "arrow_config",
        "layers_to_transform",
        "layers_pattern",
        "exclude_modules",
        "loftq_config",
        "eva_config",
        "corda_config",
    ] {
        if let Some(value) = object.get(key) {
            if !(value.is_null()
                || value.as_object().is_some_and(|m| m.is_empty())
                || value.as_array().is_some_and(|a| a.is_empty()))
            {
                return Err(invalid(format!("unsupported {key}: {value}")));
            }
        }
    }
    if let Some(initializer) = object.get("init_lora_weights") {
        if !(initializer.is_boolean() || initializer.as_str() == Some("gaussian")) {
            return Err(invalid(format!(
                "unsupported init_lora_weights: {initializer}"
            )));
        }
    }
    let declared = config
        .get("target_modules")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("target_modules must be an array of projection names"))?;
    let mut targets = BTreeSet::new();
    for value in declared {
        let target = value
            .as_str()
            .ok_or_else(|| invalid("target module must be a string"))?;
        if !TARGETS.contains(&target) {
            return Err(ModelAdapterError::UnsupportedTarget {
                target: target.into(),
            });
        }
        if !targets.insert(target.to_owned()) {
            return Err(invalid(format!("duplicate target module {target}")));
        }
    }
    if targets.is_empty() {
        return Err(invalid("target_modules is empty"));
    }
    let declared_base_model = declared_string(&config, "base_model_name_or_path")?;
    let declared_revision = declared_string(&config, "revision")?;
    Ok(PeftConfig {
        rank,
        alpha,
        targets,
        declared_base_model,
        declared_revision,
    })
}

pub(super) struct ModelContext {
    config: Qwen35Config,
}

impl ModelContext {
    pub(super) fn read(model_dir: &Path) -> Result<Self> {
        let directory = model_dir
            .canonicalize()
            .map_err(|e| io_error(model_dir, e))?;
        let bytes = read_bounded(&directory.join("config.json"), MAX_JSON_BYTES)?;
        json(&bytes)?;
        let text = std::str::from_utf8(&bytes).map_err(|e| invalid(e.to_string()))?;
        let config =
            Qwen35Config::from_config_json_str(text).map_err(|e| invalid(e.to_string()))?;
        Ok(Self { config })
    }
}

type InstallAdapter = dyn Fn(&mut MetalQwen35State) -> Result<()>;

pub(super) struct PreparedAdapter {
    pub(super) info: AdapterInfo,
    install: Box<InstallAdapter>,
}

impl PreparedAdapter {
    pub(super) fn install(&self, state: &mut MetalQwen35State) -> Result<()> {
        (self.install)(state)
    }
}

#[derive(Deserialize)]
#[serde(deny_unknown_fields)]
struct Tensor {
    dtype: String,
    shape: Vec<usize>,
    data_offsets: [usize; 2],
}

fn validate_tensors(bytes: &[u8], peft: &PeftConfig, model: &Qwen35Config) -> Result<usize> {
    let prefix: [u8; 8] = bytes
        .get(..8)
        .ok_or_else(|| invalid("missing safetensors length"))?
        .try_into()
        .map_err(|_| invalid("missing safetensors length"))?;
    let header_len = usize::try_from(u64::from_le_bytes(prefix))
        .map_err(|_| invalid("header length overflow"))?;
    if header_len as u64 > MAX_JSON_BYTES {
        return Err(invalid("safetensors header exceeds 16 MiB"));
    }
    let data_start = 8usize
        .checked_add(header_len)
        .ok_or_else(|| invalid("header length overflow"))?;
    let header = json(
        bytes
            .get(8..data_start)
            .ok_or_else(|| invalid("truncated safetensors header"))?,
    )?;
    let entries = header
        .as_object()
        .ok_or_else(|| invalid("safetensors header must be an object"))?;
    let data_len = bytes.len() - data_start;
    let mut pairs = BTreeMap::<(usize, String), u8>::new();
    let mut modules = BTreeSet::new();
    let mut ranges = Vec::new();
    for (name, value) in entries {
        if name == "__metadata__" {
            if !value
                .as_object()
                .is_some_and(|m| m.values().all(Value::is_string))
            {
                return Err(invalid("safetensors metadata must contain strings"));
            }
            continue;
        }
        let unsupported = || ModelAdapterError::UnsupportedTarget {
            target: name.clone(),
        };
        let rest = name
            .strip_prefix("base_model.model.model.layers.")
            .ok_or_else(unsupported)?;
        let parts: Vec<_> = rest.split('.').collect();
        if parts.len() != 5 || parts[4] != "weight" {
            return Err(unsupported());
        }
        let layer = parts[0].parse::<usize>().map_err(|_| unsupported())?;
        if parts[0] != layer.to_string() {
            return Err(unsupported());
        }
        let module = parts[2];
        let block = if ["gate_proj", "up_proj", "down_proj"].contains(&module) {
            "mlp"
        } else {
            "self_attn"
        };
        if parts[1] != block || !peft.targets.contains(module) || !TARGETS.contains(&module) {
            return Err(unsupported());
        }
        let side = match parts[3] {
            "lora_A" => 1,
            "lora_B" => 2,
            _ => return Err(unsupported()),
        };
        if block == "mlp" && model.is_moe() {
            return Err(unsupported());
        }
        if !model.is_layer_active(layer) {
            return Err(unsupported());
        }
        let geometry = qwen35_projection_shape(model, layer, module).map_err(|_| unsupported())?;
        let tensor: Tensor =
            serde_json::from_value(value.clone()).map_err(|e| invalid(e.to_string()))?;
        if tensor.shape.len() != 2 {
            return Err(ModelAdapterError::ShapeMismatch {
                tensor: name.clone(),
                expected: if side == 1 {
                    vec![peft.rank, geometry.d_in]
                } else {
                    vec![geometry.d_out, peft.rank]
                },
                actual: tensor.shape,
            });
        }
        let actual_rank = tensor.shape[if side == 1 { 0 } else { 1 }];
        if actual_rank != peft.rank {
            return Err(ModelAdapterError::RankMismatch {
                message: format!(
                    "{name} has rank {actual_rank}, config declares {}",
                    peft.rank
                ),
            });
        }
        let expected = if side == 1 {
            vec![peft.rank, geometry.d_in]
        } else {
            vec![geometry.d_out, peft.rank]
        };
        if tensor.shape != expected {
            return Err(ModelAdapterError::ShapeMismatch {
                tensor: name.clone(),
                expected,
                actual: tensor.shape,
            });
        }
        let width = match tensor.dtype.as_str() {
            "F32" => 4,
            "F16" | "BF16" => 2,
            _ => {
                return Err(invalid(format!(
                    "{name}: unsupported dtype {}",
                    tensor.dtype
                )))
            }
        };
        let count = tensor
            .shape
            .iter()
            .try_fold(1usize, |n, d| n.checked_mul(*d))
            .and_then(|n| n.checked_mul(width))
            .ok_or_else(|| invalid("tensor byte size overflow"))?;
        let [start, end] = tensor.data_offsets;
        if end > data_len || end.checked_sub(start) != Some(count) {
            return Err(invalid(format!("{name}: invalid tensor range")));
        }
        ranges.push((start, end));
        let present = pairs.entry((layer, module.to_owned())).or_default();
        if *present & side != 0 {
            return Err(invalid(format!("duplicate projection side {name}")));
        }
        *present |= side;
        modules.insert(module.to_owned());
    }
    if pairs.is_empty() || pairs.values().any(|sides| *sides != 3) {
        return Err(invalid("adapter requires complete nonempty A/B pairs"));
    }
    if modules != peft.targets {
        return Err(invalid("tensor targets differ from target_modules"));
    }
    let mut expected_pairs = BTreeSet::new();
    for layer in 0..model.num_hidden_layers {
        if !model.is_layer_active(layer) {
            continue;
        }
        for module in &peft.targets {
            if qwen35_projection_shape(model, layer, module).is_ok() {
                expected_pairs.insert((layer, module.clone()));
            }
        }
    }
    if pairs.keys().cloned().collect::<BTreeSet<_>>() != expected_pairs {
        return Err(invalid(
            "adapter pairs do not cover every compatible active layer",
        ));
    }
    ranges.sort_unstable();
    let mut consumed = 0;
    for (start, end) in ranges {
        if start != consumed {
            return Err(invalid("tensor ranges overlap or leave holes"));
        }
        consumed = end;
    }
    if consumed != data_len {
        return Err(invalid("unclaimed safetensors data"));
    }
    Ok(pairs.len())
}

struct Snapshot(PathBuf);
impl Snapshot {
    fn create(bytes: &[u8]) -> Result<Self> {
        let dir = std::env::temp_dir().join(format!("ruvllm-adapter-{}", uuid::Uuid::new_v4()));
        fs::DirBuilder::new()
            .mode(0o700)
            .create(&dir)
            .map_err(|e| io_error(&dir, e))?;
        let snapshot = Self(dir);
        let path = snapshot.path();
        let mut file = OpenOptions::new()
            .write(true)
            .create_new(true)
            .mode(0o600)
            .open(&path)
            .map_err(|e| io_error(&path, e))?;
        file.write_all(bytes).map_err(|e| io_error(&path, e))?;
        file.sync_all().map_err(|e| io_error(&path, e))?;
        Ok(snapshot)
    }
    fn path(&self) -> PathBuf {
        self.0.join("adapter_model.safetensors")
    }
}
impl Drop for Snapshot {
    fn drop(&mut self) {
        let _ = fs::remove_file(self.path());
        let _ = fs::remove_dir(&self.0);
    }
}

pub(super) fn prepare_captured(
    context: &std::result::Result<ModelContext, String>,
    path: &Path,
    scale: f32,
) -> Result<PreparedAdapter> {
    let context = context
        .as_ref()
        .map_err(|message| invalid(message.clone()))?;
    prepare(context, path, scale)
}

// Keep the inferred upstream descriptor type private without adding a lattice-fann dependency.
pub(super) fn prepare(context: &ModelContext, path: &Path, scale: f32) -> Result<PreparedAdapter> {
    let file = if path.is_dir() {
        path.join("adapter_model.safetensors")
    } else {
        path.to_owned()
    };
    let file = file.canonicalize().map_err(|e| io_error(&file, e))?;
    let directory = file
        .parent()
        .ok_or_else(|| invalid("adapter file has no parent directory"))?;
    let peft = configuration(&directory.join("adapter_config.json"), scale)?;
    let bytes = read_bounded(&file, MAX_ADAPTER_BYTES)?;
    let pair_count = validate_tensors(&bytes, &peft, &context.config)?;
    // The upstream loader reopens its path; give it only the bytes validated above.
    let snapshot = Snapshot::create(&bytes)?;
    drop(bytes);
    let (layers, mut descriptor) =
        lattice_inference::lora_file::load_lora_safetensors(&snapshot.path())
            .map_err(|e| invalid(e.to_string()))?;
    if layers.len() != pair_count
        || descriptor.rank != peft.rank
        || descriptor
            .target_modules
            .iter()
            .cloned()
            .collect::<BTreeSet<_>>()
            != peft.targets
    {
        return Err(invalid(
            "Lattice loader projection inventory differs from validated adapter",
        ));
    }
    for layer in &layers {
        if layer.a.iter().chain(&layer.b).any(|v| !v.is_finite()) {
            return Err(invalid("adapter tensor contains a nonfinite value"));
        }
    }
    descriptor.alpha = peft.alpha * scale;
    let info = AdapterInfo {
        path: file,
        scale,
        rank: peft.rank,
        alpha: peft.alpha,
        target_modules: peft.targets.into_iter().collect(),
        declared_base_model: peft.declared_base_model,
        declared_revision: peft.declared_revision,
    };
    Ok(PreparedAdapter {
        info,
        install: Box::new(move |state: &mut MetalQwen35State| {
            // Installation may rotate its owned factors; preserve pristine host data for rollback.
            let copy = layers
                .iter()
                .map(
                    |layer| lattice_inference::forward::metal_qwen35::LoraLayerData {
                        layer_idx: layer.layer_idx,
                        module: layer.module.clone(),
                        a: layer.a.clone(),
                        b: layer.b.clone(),
                        rank: layer.rank,
                        d_in: layer.d_in,
                        d_out: layer.d_out,
                    },
                )
                .collect();
            state
                .load_lora_adapter_with_descriptor(copy, &descriptor, None)
                .map_err(|e| ModelAdapterError::Backend {
                    message: e.to_string(),
                })?;
            Ok(())
        }),
    })
}

#[cfg(test)]
#[path = "adapter_validation_tests.rs"]
mod validation_tests;
