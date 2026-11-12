use std::collections::{BTreeMap, BTreeSet};
use std::fs::File;
use std::io::Read;
use std::path::Path;

use serde::de::{self, MapAccess, SeqAccess, Visitor};
use serde::{Deserialize, Deserializer};
use serde_json::{Map, Value};

use crate::error::{Result, RuvLLMError};
use crate::gguf::GgufValue;

const MAX_JSON_BYTES: u64 = 64 * 1024 * 1024;
const MAX_VOCAB: usize = 1_000_000;
const QWEN_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";
const LLAMA_REGEX: &str = r"(?i:'s|'t|'re|'ve|'m|'ll|'d)|[^\r\n\p{L}\p{N}]?\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]+[\r\n]*|\s*[\r\n]+|\s+(?!\S)|\s+";

pub(super) struct ModelMetadata {
    pub(super) architecture: String,
    pub(super) hidden: usize,
    pub(super) intermediate: usize,
    pub(super) blocks: usize,
    pub(super) heads: usize,
    pub(super) kv_heads: usize,
    pub(super) vocab: usize,
    pub(super) tie_embeddings: bool,
    pub(super) attention_bias: bool,
    pub(super) mlp_bias: bool,
    pub(super) values: BTreeMap<String, GgufValue>,
    pub(super) rope_factors: Option<Vec<f32>>,
}

pub(super) fn load(root: &Path) -> Result<ModelMetadata> {
    let config = read_json(&root.join("config.json"))?;
    object(&config, "config.json")?;
    let architecture = string(&config, "model_type")?.to_owned();
    if architecture != "llama" && architecture != "qwen2" {
        return Err(invalid(format!("unsupported architecture: {architecture}")));
    }
    let hidden = positive(&config, "hidden_size")?;
    let intermediate = positive(&config, "intermediate_size")?;
    let blocks = positive(&config, "num_hidden_layers")?;
    let heads = positive(&config, "num_attention_heads")?;
    let kv_heads = if config.get("num_key_value_heads").is_some() {
        positive(&config, "num_key_value_heads")?
    } else {
        heads
    };
    let vocab = positive(&config, "vocab_size")?;
    let context = positive(&config, "max_position_embeddings")?;
    if vocab > MAX_VOCAB || blocks > 65_536 {
        return Err(invalid(
            "model vocabulary or layer count exceeds supported limits",
        ));
    }
    if hidden % heads != 0 || heads % kv_heads != 0 || (hidden / heads) % 2 != 0 {
        return Err(invalid("invalid attention geometry: hidden/head and query/KV head ratios must be integral, and head dimension must be even"));
    }
    if config.get("head_dim").is_some() && positive(&config, "head_dim")? != hidden / heads {
        return Err(invalid(
            "unsupported head_dim: expected hidden_size / num_attention_heads",
        ));
    }
    if optional_string(&config, "hidden_act")?.unwrap_or("silu") != "silu" {
        return Err(invalid("unsupported hidden_act: only silu is supported"));
    }
    for field in ["rope_parameters", "quantization_config"] {
        if non_null(&config, field).is_some() {
            return Err(invalid(format!("unsupported config field: {field}")));
        }
    }
    if boolean(&config, "use_sliding_window", false)? {
        return Err(invalid("unsupported use_sliding_window=true"));
    }
    if architecture == "llama" && non_null(&config, "sliding_window").is_some() {
        return Err(invalid("unsupported llama sliding_window"));
    }
    if let Some(window) = non_null(&config, "sliding_window") {
        positive_value(window, "sliding_window")?;
    }
    if boolean(&config, "use_mrope", false)? {
        return Err(invalid("unsupported use_mrope=true"));
    }
    if let Some(rotary) = non_null(&config, "partial_rotary_factor") {
        if rotary.as_f64() != Some(1.0) {
            return Err(invalid("unsupported partial_rotary_factor"));
        }
    }
    if let Some(architectures) = non_null(&config, "architectures") {
        let expected = if architecture == "llama" {
            "LlamaForCausalLM"
        } else {
            "Qwen2ForCausalLM"
        };
        let names = architectures
            .as_array()
            .ok_or_else(|| invalid("architectures must be an array"))?;
        if names.len() != 1 || names[0].as_str() != Some(expected) {
            return Err(invalid(format!(
                "unsupported architectures for {architecture}"
            )));
        }
    }
    let rope = positive_float(&config, "rope_theta")?;
    let rope_factors = llama3_rope_factors(&config, &architecture, hidden / heads, rope)
        .map_err(|error| invalid(format!("invalid rope_scaling: {error}")))?;
    let epsilon = positive_float(&config, "rms_norm_eps")?;
    let tie_embeddings = boolean(&config, "tie_word_embeddings", false)?;
    let attention_bias = boolean(&config, "attention_bias", architecture == "qwen2")?;
    let mlp_bias = boolean(&config, "mlp_bias", false)?;
    if architecture == "qwen2" && (!attention_bias || mlp_bias) {
        return Err(invalid("unsupported qwen2 bias configuration"));
    }

    let mut values = BTreeMap::new();
    values.insert(
        "general.architecture".into(),
        GgufValue::String(architecture.clone()),
    );
    for (key, value) in [
        ("context_length", context),
        ("embedding_length", hidden),
        ("block_count", blocks),
        ("feed_forward_length", intermediate),
        ("attention.head_count", heads),
        ("attention.head_count_kv", kv_heads),
        ("vocab_size", vocab),
        ("rope.dimension_count", hidden / heads),
    ] {
        values.insert(
            format!("{architecture}.{key}"),
            GgufValue::U32(value as u32),
        );
    }
    values.insert(
        format!("{architecture}.rope.freq_base"),
        GgufValue::F32(rope),
    );
    values.insert(
        format!("{architecture}.attention.layer_norm_rms_epsilon"),
        GgufValue::F32(epsilon),
    );
    load_tokenizer(root, &config, vocab, &mut values)?;
    Ok(ModelMetadata {
        architecture,
        hidden,
        intermediate,
        blocks,
        heads,
        kv_heads,
        vocab,
        tie_embeddings,
        attention_bias,
        mlp_bias,
        values,
        rope_factors,
    })
}

fn llama3_rope_factors(
    config: &Value,
    architecture: &str,
    dimension: usize,
    base: f32,
) -> Result<Option<Vec<f32>>> {
    let Some(scaling) = non_null(config, "rope_scaling") else {
        return Ok(None);
    };
    known_fields(
        scaling,
        &[
            "rope_type",
            "type",
            "factor",
            "low_freq_factor",
            "high_freq_factor",
            "original_max_position_embeddings",
        ],
        "rope_scaling",
    )?;
    let kind = optional_string(scaling, "rope_type")?.or(optional_string(scaling, "type")?);
    if architecture != "llama" || kind != Some("llama3") {
        return Err(invalid(
            "unsupported rope_scaling: only llama3 on llama is supported",
        ));
    }
    if let (Some(current), Some(legacy)) = (
        optional_string(scaling, "rope_type")?,
        optional_string(scaling, "type")?,
    ) {
        if current != legacy {
            return Err(invalid("conflicting rope_scaling type and rope_type"));
        }
    }
    let number = |key, default| {
        if scaling.get(key).is_some() {
            positive_float(scaling, key)
        } else {
            Ok(default)
        }
    };
    let factor = number("factor", 8.0)?;
    let low = number("low_freq_factor", 1.0)?;
    let high = number("high_freq_factor", 4.0)?;
    if high <= low {
        return Err(invalid(
            "rope_scaling high_freq_factor must exceed low_freq_factor",
        ));
    }
    let context = if scaling.get("original_max_position_embeddings").is_some() {
        positive(scaling, "original_max_position_embeddings")? as f32
    } else {
        8192.0
    };
    let mut factors = Vec::new();
    factors
        .try_reserve_exact(dimension / 2)
        .map_err(|_| invalid("cannot allocate llama3 rope factors"))?;
    for index in (0..dimension).step_by(2) {
        // The converter's float32 tensor rounds the power before its reciprocal.
        let exponent = index as f32 / dimension as f32;
        let power = f64::from(base).powf(f64::from(exponent)) as f32;
        let frequency = 1.0_f32 / power;
        let wavelength = (2.0_f32 * std::f32::consts::PI) / frequency;
        let value = if wavelength < context / high {
            1.0
        } else if wavelength > context / low {
            factor
        } else {
            let smooth = (context / wavelength - low) / (high - low);
            1.0 / ((1.0 - smooth) / factor + smooth)
        };
        if !value.is_finite() || value <= 0.0 {
            return Err(invalid("rope_scaling produced an invalid frequency factor"));
        }
        factors.push(value);
    }
    Ok(Some(factors))
}

fn added_token_type(token: &Value, content: &str) -> Result<i32> {
    let looks_special = matches!(content, "<pad>" | "<mask>" | "<2mass>" | "[@BOS@]")
        || (content.starts_with("<|") && content.ends_with("|>"))
        || (content.starts_with("<｜") && content.ends_with("｜>"))
        || (content.starts_with("<unused") && content.ends_with('>'));
    Ok(if boolean(token, "special", false)? || looks_special {
        3
    } else {
        4
    })
}

fn load_tokenizer(
    root: &Path,
    config: &Value,
    vocab_size: usize,
    values: &mut BTreeMap<String, GgufValue>,
) -> Result<()> {
    let tokenizer = read_json(&root.join("tokenizer.json"))?;
    object(&tokenizer, "tokenizer.json")?;
    for key in ["truncation", "padding"] {
        if non_null(&tokenizer, key).is_some() {
            return Err(invalid(format!("unsupported tokenizer {key}")));
        }
    }
    let profile = tokenizer_profile(&tokenizer)?;
    let model = tokenizer
        .get("model")
        .ok_or_else(|| invalid("missing tokenizer model"))?;
    if string(model, "type")? != "BPE" {
        return Err(invalid("unsupported tokenizer model: expected BPE"));
    }
    if let Some(dropout) = non_null(model, "dropout") {
        if dropout.as_f64() != Some(0.0) {
            return Err(invalid(
                "unsupported BPE dropout: expected null or numeric zero",
            ));
        }
    }
    for key in ["continuing_subword_prefix", "end_of_word_suffix"] {
        if let Some(value) = non_null(model, key) {
            if value.as_str() != Some("") {
                return Err(invalid(format!("unsupported BPE {key}")));
            }
        }
    }
    if boolean(model, "byte_fallback", false)? || boolean(model, "fuse_unk", false)? {
        return Err(invalid("unsupported BPE byte_fallback or fuse_unk"));
    }
    let ignore_merges = boolean(model, "ignore_merges", false)?;
    if ignore_merges != (profile == "llama-bpe") {
        return Err(invalid(
            "BPE ignore_merges disagrees with the target tokenizer profile",
        ));
    }

    let vocab = model
        .get("vocab")
        .and_then(Value::as_object)
        .ok_or_else(|| invalid("BPE vocab must be an object"))?;
    if vocab.is_empty() {
        return Err(invalid("BPE vocab is empty"));
    }
    let mut tokens = vec![None::<String>; vocab_size];
    let mut types = vec![5_i32; vocab_size];
    let mut by_content = BTreeMap::<String, usize>::new();
    for (token, index) in vocab {
        let id = token_id(index, vocab_size, "vocabulary token")?;
        if token.is_empty() || tokens[id].is_some() {
            return Err(invalid("empty token or duplicate vocabulary ID"));
        }
        tokens[id] = Some(token.clone());
        types[id] = 1;
        by_content.insert(token.clone(), id);
    }
    let mut added_ids = BTreeSet::new();
    if let Some(added) = non_null(&tokenizer, "added_tokens") {
        for token in added
            .as_array()
            .ok_or_else(|| invalid("added_tokens must be an array"))?
        {
            let id = token_id(
                token
                    .get("id")
                    .ok_or_else(|| invalid("missing added token ID"))?,
                vocab_size,
                "added token",
            )?;
            let content = string(token, "content")?;
            if content.is_empty() || !added_ids.insert(id) {
                return Err(invalid("empty or duplicate added token"));
            }
            check_added_token_flags(token)?;
            if tokens[id].as_deref().is_some_and(|old| old != content)
                || by_content.get(content).is_some_and(|old| *old != id)
            {
                return Err(invalid("conflicting added token content or ID"));
            }
            tokens[id] = Some(content.to_owned());
            by_content.insert(content.to_owned(), id);
            types[id] = added_token_type(token, content)?;
        }
    }
    let merges = load_merges(model, vocab)?;

    let tokenizer_config_path = root.join("tokenizer_config.json");
    let tokenizer_config = match std::fs::symlink_metadata(&tokenizer_config_path) {
        Ok(_) => read_json(&tokenizer_config_path)?,
        Err(error) if error.kind() == std::io::ErrorKind::NotFound => Value::Object(Map::new()),
        Err(error) => {
            return Err(invalid(format!(
                "cannot inspect tokenizer_config.json: {error}"
            )))
        }
    };
    object(&tokenizer_config, "tokenizer_config.json")?;
    if boolean(&tokenizer_config, "add_prefix_space", false)? {
        return Err(invalid("unsupported tokenizer add_prefix_space=true"));
    }
    if boolean(&tokenizer_config, "split_special_tokens", false)? {
        return Err(invalid("unsupported split_special_tokens=true"));
    }
    if let Some(decoder) = non_null(&tokenizer_config, "added_tokens_decoder") {
        for (id_text, token) in object(decoder, "added_tokens_decoder")? {
            let id = id_text
                .parse::<usize>()
                .map_err(|_| invalid("invalid added_tokens_decoder ID"))?;
            if id >= vocab_size || !added_ids.contains(&id) {
                return Err(invalid(
                    "added_tokens_decoder ID is absent from added_tokens",
                ));
            }
            if tokens[id].as_deref() != Some(string(token, "content")?) {
                return Err(invalid(
                    "added_tokens_decoder content conflicts with tokenizer.json",
                ));
            }
            check_added_token_flags(token)?;
            let expected = added_token_type(token, string(token, "content")?)?;
            if types[id] != expected {
                return Err(invalid(
                    "added_tokens_decoder special flag conflicts with tokenizer.json",
                ));
            }
        }
    }

    let mut special_ids = BTreeMap::<&str, usize>::new();
    for role in ["bos", "eos", "pad", "unk"] {
        let key = format!("{role}_token_id");
        for source in [config, &tokenizer_config] {
            if let Some(value) = non_null(source, &key) {
                let id = token_id(value, vocab_size, &key)?;
                insert_special(&mut special_ids, role, id)?;
            }
        }
        if let Some(token) = non_null(&tokenizer_config, &format!("{role}_token")) {
            if token.is_object() {
                check_added_token_flags(token)?;
            }
            let content = token
                .as_str()
                .or_else(|| token.get("content").and_then(Value::as_str))
                .ok_or_else(|| invalid(format!("invalid {role}_token content")))?;
            let id = by_content
                .get(content)
                .copied()
                .ok_or_else(|| invalid(format!("{role}_token is absent from vocabulary")))?;
            insert_special(&mut special_ids, role, id)?;
        }
    }
    if let Some(unknown) = non_null(model, "unk_token") {
        let content = unknown
            .as_str()
            .ok_or_else(|| invalid("BPE unk_token must be a string"))?;
        let id = by_content
            .get(content)
            .copied()
            .ok_or_else(|| invalid("BPE unk_token is absent from vocabulary"))?;
        insert_special(&mut special_ids, "unk", id)?;
    }

    let (prefix, suffix) = postprocessor_tokens(tokenizer.get("post_processor"), &by_content)?;
    if let Some(id) = prefix {
        insert_special(&mut special_ids, "bos", id)?;
    }
    if let Some(id) = suffix {
        insert_special(&mut special_ids, "eos", id)?;
    }
    for (role, enabled) in [("bos", prefix.is_some()), ("eos", suffix.is_some())] {
        let key = format!("add_{role}_token");
        if tokenizer_config.get(&key).is_some()
            && boolean(&tokenizer_config, &key, false)? != enabled
        {
            return Err(invalid(format!(
                "{key} conflicts with tokenizer post_processor"
            )));
        }
        values.insert(format!("tokenizer.ggml.{key}"), GgufValue::Bool(enabled));
    }
    for (role, id) in special_ids {
        if tokens[id].is_none() {
            return Err(invalid(format!(
                "{role}_token_id refers to an unused vocabulary slot"
            )));
        }
        let key = match role {
            "pad" => "padding_token_id",
            "unk" => "unknown_token_id",
            "bos" => "bos_token_id",
            "eos" => "eos_token_id",
            _ => return Err(invalid("unsupported special token role")),
        };
        values.insert(format!("tokenizer.ggml.{key}"), GgufValue::U32(id as u32));
    }
    for (id, token) in tokens.iter().enumerate() {
        if token.is_none() && by_content.contains_key(&format!("[PAD{id}]")) {
            return Err(invalid(
                "unused vocabulary placeholder conflicts with an existing token",
            ));
        }
    }
    let tokens = tokens
        .into_iter()
        .enumerate()
        .map(|(id, token)| GgufValue::String(token.unwrap_or_else(|| format!("[PAD{id}]"))))
        .collect();
    values.insert(
        "tokenizer.ggml.model".into(),
        GgufValue::String("gpt2".into()),
    );
    values.insert(
        "tokenizer.ggml.pre".into(),
        GgufValue::String(profile.into()),
    );
    values.insert("tokenizer.ggml.tokens".into(), GgufValue::Array(tokens));
    values.insert(
        "tokenizer.ggml.token_type".into(),
        GgufValue::Array(types.into_iter().map(GgufValue::I32).collect()),
    );
    values.insert("tokenizer.ggml.merges".into(), GgufValue::Array(merges));
    values.insert(
        "tokenizer.ggml.add_space_prefix".into(),
        GgufValue::Bool(false),
    );
    if let Some(template) = non_null(&tokenizer_config, "chat_template") {
        values.insert(
            "tokenizer.chat_template".into(),
            GgufValue::String(
                template
                    .as_str()
                    .ok_or_else(|| invalid("chat_template must be a string"))?
                    .into(),
            ),
        );
    }
    Ok(())
}

fn tokenizer_profile(tokenizer: &Value) -> Result<&'static str> {
    let pre = tokenizer
        .get("pre_tokenizer")
        .ok_or_else(|| invalid("missing tokenizer pre_tokenizer"))?;
    let profile = match string(pre, "type")? {
        "ByteLevel" => {
            byte_level(pre, true)?;
            "gpt-2"
        }
        "Sequence" => {
            known_fields(pre, &["type", "pretokenizers"], "pre_tokenizer Sequence")?;
            let parts = pre
                .get("pretokenizers")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("pretokenizers must be an array"))?;
            if parts.len() != 2 {
                return Err(invalid("unsupported pre_tokenizer sequence"));
            }
            let split = &parts[0];
            if string(split, "type")? == "Digits" {
                known_fields(split, &["type", "individual_digits"], "Digits")?;
                if split.get("individual_digits").and_then(Value::as_bool) != Some(true) {
                    return Err(invalid("Digits requires individual_digits=true"));
                }
                // Digits splits numbers before ByteLevel applies its GPT-2 regex.
                byte_level(&parts[1], true)?;
                "smollm"
            } else {
                known_fields(split, &["type", "pattern", "behavior", "invert"], "Split")?;
                if string(split, "type")? != "Split"
                    || string(split, "behavior")? != "Isolated"
                    || boolean(split, "invert", false)?
                {
                    return Err(invalid("unsupported BPE Split behavior"));
                }
                let pattern = split
                    .get("pattern")
                    .ok_or_else(|| invalid("missing Split pattern"))?;
                known_fields(pattern, &["Regex"], "Split pattern")?;
                let regex = string(pattern, "Regex")?;
                byte_level(&parts[1], false)?;
                match regex {
                    QWEN_REGEX => "qwen2",
                    LLAMA_REGEX => "llama-bpe",
                    _ => return Err(invalid("unsupported BPE Split regex")),
                }
            }
        }
        _ => return Err(invalid("unsupported tokenizer pre_tokenizer")),
    };
    // llama.cpp's BPE runtime omits NFC, matching the converter's no-normalizer output.
    if let Some(normalizer) = non_null(tokenizer, "normalizer") {
        let exact_nfc = normalizer.as_object().is_some_and(|fields| {
            fields.len() == 1 && fields.get("type").and_then(Value::as_str) == Some("NFC")
        });
        if !exact_nfc {
            return Err(invalid(
                "unsupported tokenizer normalizer: only null or exactly {type: NFC} is supported",
            ));
        }
    }
    let decoder = tokenizer
        .get("decoder")
        .ok_or_else(|| invalid("missing tokenizer decoder"))?;
    byte_level_component(decoder)?;
    Ok(profile)
}

fn byte_level(value: &Value, use_regex: bool) -> Result<()> {
    byte_level_component(value)?;
    if boolean(value, "add_prefix_space", true)? || boolean(value, "use_regex", true)? != use_regex
    {
        return Err(invalid(
            "unsupported ByteLevel add_prefix_space or use_regex",
        ));
    }
    Ok(())
}

fn byte_level_component(value: &Value) -> Result<()> {
    known_fields(
        value,
        &["type", "add_prefix_space", "trim_offsets", "use_regex"],
        "ByteLevel",
    )?;
    if string(value, "type")? != "ByteLevel" {
        return Err(invalid(
            "unsupported tokenizer component: expected ByteLevel",
        ));
    }
    for key in ["add_prefix_space", "trim_offsets", "use_regex"] {
        boolean(value, key, false)?;
    }
    Ok(())
}

fn check_added_token_flags(token: &Value) -> Result<()> {
    for key in ["single_word", "lstrip", "rstrip"] {
        if boolean(token, key, false)? {
            return Err(invalid(format!("unsupported added token {key}=true")));
        }
    }
    boolean(token, "normalized", false)?;
    Ok(())
}

fn load_merges(model: &Value, vocab: &Map<String, Value>) -> Result<Vec<GgufValue>> {
    let merges = model
        .get("merges")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("BPE merges must be an array"))?;
    if merges.is_empty() {
        return Err(invalid("BPE merges are empty"));
    }
    let mut seen = BTreeSet::new();
    let mut result = Vec::with_capacity(merges.len());
    for merge in merges {
        let (left, right) = if let Some(pair) = merge.as_str() {
            let mut parts = pair.split(' ');
            let left = parts.next().unwrap_or("");
            let right = parts
                .next()
                .ok_or_else(|| invalid("invalid BPE merge pair"))?;
            if parts.next().is_some() {
                return Err(invalid("unsupported spaces in BPE merge components"));
            }
            (left, right)
        } else {
            let pair = merge
                .as_array()
                .ok_or_else(|| invalid("invalid BPE merge pair"))?;
            if pair.len() != 2 {
                return Err(invalid("invalid BPE merge pair length"));
            }
            (
                pair[0]
                    .as_str()
                    .ok_or_else(|| invalid("invalid BPE merge component"))?,
                pair[1]
                    .as_str()
                    .ok_or_else(|| invalid("invalid BPE merge component"))?,
            )
        };
        if left.is_empty() || right.is_empty() || left.contains(' ') || right.contains(' ') {
            return Err(invalid(
                "unsupported empty or space-containing BPE merge component",
            ));
        }
        if !vocab.contains_key(left)
            || !vocab.contains_key(right)
            || !vocab.contains_key(&format!("{left}{right}"))
        {
            return Err(invalid(
                "BPE merge references tokens absent from vocabulary",
            ));
        }
        let pair = format!("{left} {right}");
        if !seen.insert(pair.clone()) {
            return Err(invalid("duplicate BPE merge"));
        }
        result.push(GgufValue::String(pair));
    }
    Ok(result)
}

fn postprocessor_tokens(
    processor: Option<&Value>,
    vocab: &BTreeMap<String, usize>,
) -> Result<(Option<usize>, Option<usize>)> {
    let Some(processor) = processor.filter(|value| !value.is_null()) else {
        return Ok((None, None));
    };
    match string(processor, "type")? {
        "ByteLevel" => {
            byte_level_component(processor)?;
            Ok((None, None))
        }
        "TemplateProcessing" => template_tokens(processor, vocab),
        "Sequence" => {
            known_fields(
                processor,
                &["type", "processors"],
                "post_processor Sequence",
            )?;
            let processors = processor
                .get("processors")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("post_processor processors must be an array"))?;
            if processors.len() > 2 || processors.is_empty() {
                return Err(invalid("unsupported post_processor sequence"));
            }
            let mut template = None;
            for part in processors {
                match string(part, "type")? {
                    "ByteLevel" => byte_level_component(part)?,
                    "TemplateProcessing" if template.is_none() => {
                        template = Some(template_tokens(part, vocab)?);
                    }
                    _ => return Err(invalid("unsupported post_processor sequence component")),
                }
            }
            Ok(template.unwrap_or((None, None)))
        }
        _ => Err(invalid("unsupported tokenizer post_processor")),
    }
}

fn template_tokens(
    processor: &Value,
    vocab: &BTreeMap<String, usize>,
) -> Result<(Option<usize>, Option<usize>)> {
    known_fields(
        processor,
        &["type", "single", "pair", "special_tokens"],
        "TemplateProcessing",
    )?;
    let single = processor
        .get("single")
        .and_then(Value::as_array)
        .ok_or_else(|| invalid("TemplateProcessing.single must be an array"))?;
    if single.is_empty() || single.len() > 3 {
        return Err(invalid("unsupported single token template"));
    }
    let specials = processor
        .get("special_tokens")
        .ok_or_else(|| invalid("missing TemplateProcessing.special_tokens"))?;
    let mut prefix = None;
    let mut suffix = None;
    let mut has_sequence = false;
    for part in single {
        let fields = object(part, "token template entry")?;
        if fields.len() != 1 {
            return Err(invalid("invalid token template entry"));
        }
        if let Some(sequence) = part.get("Sequence") {
            known_fields(sequence, &["id", "type_id"], "template Sequence")?;
            if has_sequence
                || string(sequence, "id")? != "A"
                || sequence.get("type_id").and_then(Value::as_u64) != Some(0)
            {
                return Err(invalid("unsupported template sequence"));
            }
            has_sequence = true;
        } else if let Some(token) = part.get("SpecialToken") {
            known_fields(token, &["id", "type_id"], "template SpecialToken")?;
            if token.get("type_id").and_then(Value::as_u64) != Some(0) {
                return Err(invalid("unsupported special token type_id"));
            }
            let name = string(token, "id")?;
            let entry = specials
                .get(name)
                .ok_or_else(|| invalid("missing template special token"))?;
            known_fields(entry, &["id", "ids", "tokens"], "template special token")?;
            let ids = entry
                .get("ids")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("invalid template special token IDs"))?;
            let tokens = entry
                .get("tokens")
                .and_then(Value::as_array)
                .ok_or_else(|| invalid("invalid template special token strings"))?;
            if ids.len() != 1 || tokens.len() != 1 || string(entry, "id")? != name {
                return Err(invalid("unsupported multi-token template special"));
            }
            let content = tokens[0]
                .as_str()
                .ok_or_else(|| invalid("invalid template special content"))?;
            let id = vocab
                .get(content)
                .copied()
                .ok_or_else(|| invalid("template token is absent from vocabulary"))?;
            if ids[0].as_u64() != Some(id as u64) {
                return Err(invalid("template token ID conflicts with vocabulary"));
            }
            let target = if has_sequence {
                &mut suffix
            } else {
                &mut prefix
            };
            if target.replace(id).is_some() {
                return Err(invalid("unsupported multiple prefix or suffix tokens"));
            }
        } else {
            return Err(invalid("unknown token template entry"));
        }
    }
    if !has_sequence {
        return Err(invalid("token template has no input sequence"));
    }
    if let Some(pair) = processor.get("pair") {
        let mut expected = single.clone();
        let mut second = single.clone();
        for part in &mut second {
            if let Some(sequence) = part.get_mut("Sequence") {
                sequence["id"] = Value::String("B".into());
                sequence["type_id"] = Value::from(1);
            }
            if let Some(token) = part.get_mut("SpecialToken") {
                token["type_id"] = Value::from(1);
            }
        }
        expected.extend(second);
        let plain = serde_json::json!([
            {"Sequence":{"id":"A","type_id":0}},
            {"Sequence":{"id":"B","type_id":1}}
        ]);
        let mut zero_types = expected.clone();
        for part in &mut zero_types {
            if let Some(sequence) = part.get_mut("Sequence") {
                sequence["type_id"] = Value::from(0);
            }
            if let Some(token) = part.get_mut("SpecialToken") {
                token["type_id"] = Value::from(0);
            }
        }
        let plain_zero = serde_json::json!([
            {"Sequence":{"id":"A","type_id":0}},
            {"Sequence":{"id":"B","type_id":0}}
        ]);
        if pair != &Value::Array(expected)
            && pair != &Value::Array(zero_types)
            && pair != &plain
            && pair != &plain_zero
        {
            return Err(invalid("unsupported pair token template"));
        }
    }
    Ok((prefix, suffix))
}

fn insert_special<'a>(ids: &mut BTreeMap<&'a str, usize>, role: &'a str, id: usize) -> Result<()> {
    if ids.get(role).is_some_and(|previous| *previous != id) {
        return Err(invalid(format!("conflicting {role} token IDs")));
    }
    ids.insert(role, id);
    Ok(())
}

fn non_null<'a>(value: &'a Value, key: &str) -> Option<&'a Value> {
    value.get(key).filter(|value| !value.is_null())
}

fn object<'a>(value: &'a Value, label: &str) -> Result<&'a Map<String, Value>> {
    value
        .as_object()
        .ok_or_else(|| invalid(format!("{label} must be an object")))
}

fn known_fields(value: &Value, keys: &[&str], label: &str) -> Result<()> {
    if let Some(key) = object(value, label)?
        .keys()
        .find(|key| !keys.contains(&key.as_str()))
    {
        return Err(invalid(format!("unsupported {label} field: {key}")));
    }
    Ok(())
}

fn string<'a>(value: &'a Value, key: &str) -> Result<&'a str> {
    value
        .get(key)
        .and_then(Value::as_str)
        .ok_or_else(|| invalid(format!("missing or invalid {key}")))
}

fn optional_string<'a>(value: &'a Value, key: &str) -> Result<Option<&'a str>> {
    value
        .get(key)
        .map(|value| {
            value
                .as_str()
                .ok_or_else(|| invalid(format!("{key} must be a string")))
        })
        .transpose()
}

fn boolean(value: &Value, key: &str, default: bool) -> Result<bool> {
    match value.get(key) {
        None => Ok(default),
        Some(value) => value
            .as_bool()
            .ok_or_else(|| invalid(format!("{key} must be a boolean"))),
    }
}

fn positive(config: &Value, key: &str) -> Result<usize> {
    positive_value(
        config
            .get(key)
            .ok_or_else(|| invalid(format!("missing {key}")))?,
        key,
    )
}

fn positive_value(value: &Value, key: &str) -> Result<usize> {
    let integer = value
        .as_u64()
        .filter(|value| *value > 0 && *value <= u32::MAX as u64)
        .ok_or_else(|| invalid(format!("{key} must be a positive U32")))?;
    usize::try_from(integer).map_err(|_| invalid(format!("{key} exceeds platform limits")))
}

fn positive_float(config: &Value, key: &str) -> Result<f32> {
    let value = config
        .get(key)
        .and_then(Value::as_f64)
        .ok_or_else(|| invalid(format!("missing or invalid {key}")))? as f32;
    if !value.is_finite() || value <= 0.0 {
        return Err(invalid(format!("{key} must be finite and positive")));
    }
    Ok(value)
}

fn token_id(value: &Value, vocab: usize, label: &str) -> Result<usize> {
    let id = value
        .as_u64()
        .filter(|id| *id < vocab as u64)
        .ok_or_else(|| invalid(format!("{label} ID must be an integer below vocab_size")))?;
    Ok(id as usize)
}

fn invalid(message: impl Into<String>) -> RuvLLMError {
    RuvLLMError::Model(message.into())
}

fn read_json(path: &Path) -> Result<Value> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|error| invalid(format!("cannot inspect {}: {error}", path.display())))?;
    if !metadata.file_type().is_file() || metadata.len() > MAX_JSON_BYTES {
        return Err(invalid(format!(
            "{} must be a regular JSON file no larger than 64 MiB",
            path.display()
        )));
    }
    let mut bytes = Vec::new();
    File::open(path)
        .map_err(|error| invalid(format!("cannot open {}: {error}", path.display())))?
        .take(MAX_JSON_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|error| invalid(format!("cannot read {}: {error}", path.display())))?;
    if bytes.len() as u64 > MAX_JSON_BYTES {
        return Err(invalid(format!(
            "{} exceeds JSON size limit",
            path.display()
        )));
    }
    serde_json::from_slice::<UniqueJson>(&bytes)
        .map(|json| json.0)
        .map_err(|error| invalid(format!("invalid {}: {error}", path.display())))
}

struct UniqueJson(Value);

impl<'de> Deserialize<'de> for UniqueJson {
    fn deserialize<D: Deserializer<'de>>(deserializer: D) -> std::result::Result<Self, D::Error> {
        struct JsonVisitor;
        impl<'de> Visitor<'de> for JsonVisitor {
            type Value = UniqueJson;

            fn expecting(&self, formatter: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
                formatter.write_str("JSON with unique object keys")
            }

            fn visit_bool<E: de::Error>(self, value: bool) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Bool(value)))
            }

            fn visit_i64<E: de::Error>(self, value: i64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }

            fn visit_u64<E: de::Error>(self, value: u64) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::from(value)))
            }

            fn visit_f64<E: de::Error>(self, value: f64) -> std::result::Result<Self::Value, E> {
                serde_json::Number::from_f64(value)
                    .map(|number| UniqueJson(Value::Number(number)))
                    .ok_or_else(|| E::custom("non-finite JSON number"))
            }

            fn visit_str<E: de::Error>(self, value: &str) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value.into())))
            }

            fn visit_string<E: de::Error>(
                self,
                value: String,
            ) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::String(value)))
            }

            fn visit_unit<E: de::Error>(self) -> std::result::Result<Self::Value, E> {
                Ok(UniqueJson(Value::Null))
            }

            fn visit_seq<A: SeqAccess<'de>>(
                self,
                mut seq: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Vec::new();
                while let Some(value) = seq.next_element::<UniqueJson>()? {
                    values.push(value.0);
                }
                Ok(UniqueJson(Value::Array(values)))
            }

            fn visit_map<A: MapAccess<'de>>(
                self,
                mut map: A,
            ) -> std::result::Result<Self::Value, A::Error> {
                let mut values = Map::new();
                while let Some(key) = map.next_key::<String>()? {
                    if values.contains_key(&key) {
                        return Err(de::Error::custom(format!("duplicate JSON key: {key}")));
                    }
                    values.insert(key, map.next_value::<UniqueJson>()?.0);
                }
                Ok(UniqueJson(Value::Object(values)))
            }
        }
        deserializer.deserialize_any(JsonVisitor)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config() -> Value {
        json!({
            "model_type":"llama",
            "hidden_size":32,
            "intermediate_size":64,
            "num_hidden_layers":1,
            "num_attention_heads":4,
            "num_key_value_heads":2,
            "vocab_size":8,
            "max_position_embeddings":128,
            "rope_theta":10000,
            "rms_norm_eps":0.00001,
            "bos_token_id":3,
            "eos_token_id":4
        })
    }

    fn tokenizer() -> Value {
        json!({
            "normalizer":null,
            "pre_tokenizer":{"type":"ByteLevel","add_prefix_space":false,"use_regex":true},
            "decoder":{"type":"ByteLevel"},
            "post_processor":null,
            "model":{"type":"BPE","vocab":{"a":0,"b":1,"ab":2,"<bos>":3,"<eos>":4},"merges":["a b"]},
            "added_tokens":[{"id":5,"content":"<extra>","special":false}]
        })
    }

    fn fixture(config: &Value, tokenizer: &Value) -> tempfile::TempDir {
        let directory = tempfile::tempdir().unwrap();
        std::fs::write(directory.path().join("config.json"), config.to_string()).unwrap();
        std::fs::write(
            directory.path().join("tokenizer.json"),
            tokenizer.to_string(),
        )
        .unwrap();
        directory
    }

    fn split_profile(regex: &str) -> Value {
        json!({
            "type":"Sequence",
            "pretokenizers":[
                {"type":"Split","pattern":{"Regex":regex},"behavior":"Isolated","invert":false},
                {"type":"ByteLevel","add_prefix_space":false,"use_regex":false}
            ]
        })
    }

    #[test]
    fn token_metadata_preserves_indices_added_tokens_and_unused_slots() {
        let directory = fixture(&config(), &tokenizer());
        let model = load(directory.path()).unwrap();
        assert_eq!(model.architecture, "llama");
        assert_eq!(model.kv_heads, 2);
        assert!(!model.attention_bias);
        assert!(!model.tie_embeddings);
        assert_eq!(model.values["tokenizer.ggml.pre"].as_str(), Some("gpt-2"));
        let GgufValue::Array(tokens) = &model.values["tokenizer.ggml.tokens"] else {
            panic!("expected token array");
        };
        assert_eq!(tokens.len(), 8);
        assert_eq!(tokens[5].as_str(), Some("<extra>"));
        assert_eq!(tokens[6].as_str(), Some("[PAD6]"));
        let GgufValue::Array(types) = &model.values["tokenizer.ggml.token_type"] else {
            panic!("expected token type array");
        };
        assert_eq!(types[0].as_i64(), Some(1));
        assert_eq!(types[3].as_i64(), Some(1));
        assert_eq!(types[5].as_i64(), Some(4));
        assert_eq!(types[6].as_i64(), Some(5));
        assert!(matches!(
            model.values["tokenizer.ggml.add_bos_token"],
            GgufValue::Bool(false)
        ));
    }

    #[test]
    fn qwen_defaults_allow_inactive_sliding_window_without_normalization() {
        let mut config = config();
        config["model_type"] = json!("qwen2");
        config["sliding_window"] = json!(32768);
        config["use_sliding_window"] = json!(false);
        config["tie_word_embeddings"] = json!(true);
        let mut tokenizer = tokenizer();
        tokenizer["pre_tokenizer"] = split_profile(QWEN_REGEX);
        let directory = fixture(&config, &tokenizer);
        let model = load(directory.path()).unwrap();
        assert!(model.attention_bias);
        assert!(model.tie_embeddings);
        assert_eq!(model.values["tokenizer.ggml.pre"].as_str(), Some("qwen2"));
    }

    #[test]
    fn accepts_qwen_nfc_without_changing_token_metadata() {
        let mut config = config();
        config["model_type"] = json!("qwen2");
        let mut tokenizer = tokenizer();
        tokenizer["pre_tokenizer"] = split_profile(QWEN_REGEX);
        tokenizer["normalizer"] = json!({"type":"NFC"});
        let directory = fixture(&config, &tokenizer);
        let nfc = load(directory.path()).unwrap();
        tokenizer["normalizer"] = Value::Null;
        let directory = fixture(&config, &tokenizer);
        let plain = load(directory.path()).unwrap();
        assert_eq!(format!("{:?}", nfc.values), format!("{:?}", plain.values));
    }

    #[test]
    fn requires_ignore_merges_to_match_the_selected_profile() {
        let mut llama = tokenizer();
        llama["pre_tokenizer"] = split_profile(LLAMA_REGEX);
        for ignored in [None, Some(false)] {
            if let Some(value) = ignored {
                llama["model"]["ignore_merges"] = json!(value);
            }
            let directory = fixture(&config(), &llama);
            let error = load(directory.path()).err().unwrap();
            assert!(error.to_string().contains("ignore_merges"));
        }
        llama["model"]["ignore_merges"] = json!(true);
        let directory = fixture(&config(), &llama);
        assert!(load(directory.path()).is_ok());

        let gpt2 = tokenizer();
        let mut qwen = tokenizer();
        qwen["pre_tokenizer"] = split_profile(QWEN_REGEX);
        for mut tokenizer in [gpt2, qwen] {
            tokenizer["model"]["ignore_merges"] = json!(true);
            let directory = fixture(&config(), &tokenizer);
            let error = load(directory.path()).err().unwrap();
            assert!(error.to_string().contains("ignore_merges"));
            tokenizer["model"]["ignore_merges"] = json!(false);
            let directory = fixture(&config(), &tokenizer);
            assert!(load(directory.path()).is_ok());
        }
    }

    #[test]
    fn refuses_missing_or_unsupported_model_metadata() {
        for (field, value) in [
            ("model_type", json!("gpt2")),
            ("hidden_size", json!(0)),
            ("num_attention_heads", json!(3)),
            ("num_key_value_heads", json!(3)),
            ("hidden_act", json!("gelu")),
            ("rope_scaling", json!({"type":"linear","factor":2})),
            ("use_sliding_window", json!(true)),
            ("head_dim", json!(16)),
            ("rms_norm_eps", json!(-0.001)),
            ("bos_token_id", json!(8)),
        ] {
            let mut config = config();
            config[field] = value;
            let directory = fixture(&config, &tokenizer());
            assert!(load(directory.path()).is_err(), "{field}");
        }
        let mut config = config();
        config.as_object_mut().unwrap().remove("rope_theta");
        let directory = fixture(&config, &tokenizer());
        assert!(load(directory.path()).is_err());
    }

    #[test]
    fn refuses_unknown_tokenizer_transformations() {
        for (field, value) in [
            ("normalizer", json!({"type":"Lowercase"})),
            ("decoder", json!({"type":"WordPiece","prefix":"##"})),
            ("pre_tokenizer", split_profile(r"\p{L}+")),
            ("post_processor", json!({"type":"RobertaProcessing"})),
            ("padding", json!({"strategy":"BatchLongest"})),
        ] {
            let mut tokenizer = tokenizer();
            tokenizer[field] = value;
            let directory = fixture(&config(), &tokenizer);
            assert!(load(directory.path()).is_err(), "{field}");
        }
    }

    #[test]
    fn sidecar_special_token_objects_preserve_supported_flags_only() {
        for role in ["bos", "eos", "pad", "unk"] {
            let content = if role == "eos" { "<eos>" } else { "<bos>" };
            for (flag, value) in [
                ("single_word", json!(true)),
                ("lstrip", json!(true)),
                ("rstrip", json!(true)),
                ("lstrip", json!("false")),
            ] {
                let directory = fixture(&config(), &tokenizer());
                let mut token = json!({"content":content});
                token[flag] = value;
                let mut sidecar = json!({});
                sidecar[format!("{role}_token")] = token;
                std::fs::write(
                    directory.path().join("tokenizer_config.json"),
                    sidecar.to_string(),
                )
                .unwrap();
                assert!(load(directory.path()).is_err(), "{role}: {flag}");
            }
            let directory = fixture(&config(), &tokenizer());
            let mut sidecar = json!({});
            sidecar[format!("{role}_token")] =
                json!({"content":content,"lstrip":false,"rstrip":false,"single_word":false});
            std::fs::write(
                directory.path().join("tokenizer_config.json"),
                sidecar.to_string(),
            )
            .unwrap();
            assert!(load(directory.path()).is_ok(), "{role}");
        }
    }

    #[test]
    fn dropout_must_be_null_or_numeric_zero() {
        for dropout in [json!(""), json!("0"), json!(false), json!(0.1), json!(-1)] {
            let mut tokenizer = tokenizer();
            tokenizer["model"]["dropout"] = dropout;
            let directory = fixture(&config(), &tokenizer);
            assert!(load(directory.path()).is_err());
        }
        for dropout in [Value::Null, json!(0), json!(0.0)] {
            let mut tokenizer = tokenizer();
            tokenizer["model"]["dropout"] = dropout;
            let directory = fixture(&config(), &tokenizer);
            assert!(load(directory.path()).is_ok());
        }
    }

    #[test]
    fn refuses_vocabulary_collisions_bad_added_flags_and_invalid_merges() {
        let mut collision = tokenizer();
        collision["model"]["vocab"]["b"] = json!(0);
        let mut added = tokenizer();
        added["added_tokens"][0]["lstrip"] = json!(true);
        let mut merges = tokenizer();
        merges["model"]["merges"] = json!([["a", "missing"]]);
        for tokenizer in [collision, added, merges] {
            let directory = fixture(&config(), &tokenizer);
            assert!(load(directory.path()).is_err());
        }
        let mut tokenizer = tokenizer();
        tokenizer["model"]["merges"] = json!([["a", "b"]]);
        let directory = fixture(&config(), &tokenizer);
        assert!(load(directory.path()).is_ok());
    }

    #[test]
    fn preserves_template_bos_and_refuses_conflicting_sidecar_flags() {
        let mut tokenizer = tokenizer();
        tokenizer["pre_tokenizer"] = split_profile(LLAMA_REGEX);
        tokenizer["model"]["ignore_merges"] = json!(true);
        tokenizer["post_processor"] = json!({
            "type":"TemplateProcessing",
            "single":[
                {"SpecialToken":{"id":"<bos>","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}}
            ],
            "pair":[
                {"SpecialToken":{"id":"<bos>","type_id":0}},
                {"Sequence":{"id":"A","type_id":0}},
                {"SpecialToken":{"id":"<bos>","type_id":0}},
                {"Sequence":{"id":"B","type_id":0}}
            ],
            "special_tokens":{"<bos>":{"id":"<bos>","ids":[3],"tokens":["<bos>"]}}
        });
        let directory = fixture(&config(), &tokenizer);
        let model = load(directory.path()).unwrap();
        assert!(matches!(
            model.values["tokenizer.ggml.add_bos_token"],
            GgufValue::Bool(true)
        ));
        assert_eq!(
            model.values["tokenizer.ggml.pre"].as_str(),
            Some("llama-bpe")
        );
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            json!({"add_bos_token":false}).to_string(),
        )
        .unwrap();
        assert!(load(directory.path()).is_err());
    }

    #[test]
    fn rejects_duplicate_keys_in_nested_json() {
        let bytes = br#"{"model":{"vocab":{"a":0,"a":1}}}"#;
        assert!(serde_json::from_slice::<UniqueJson>(bytes).is_err());
    }

    #[test]
    fn qwen_added_special_unknown_token_remains_control() {
        let mut config = config();
        config["model_type"] = json!("qwen2");
        config["unk_token_id"] = json!(0);
        let mut tokenizer = tokenizer();
        tokenizer["pre_tokenizer"] = split_profile(QWEN_REGEX);
        tokenizer["added_tokens"]
            .as_array_mut()
            .unwrap()
            .push(json!({"id":0,"content":"a","special":true}));
        tokenizer["model"]["unk_token"] = json!("a");
        let directory = fixture(&config, &tokenizer);
        std::fs::write(
            directory.path().join("tokenizer_config.json"),
            json!({
                "unk_token":"a", "added_tokens_decoder":{"0":{"content":"a","special":true}}
            })
            .to_string(),
        )
        .unwrap();
        let model = load(directory.path()).unwrap();
        let types = model.values["tokenizer.ggml.token_type"]
            .as_array()
            .unwrap();
        assert_eq!(model.values["tokenizer.ggml.pre"].as_str(), Some("qwen2"));
        assert_eq!(
            model.values["tokenizer.ggml.unknown_token_id"].as_u64(),
            Some(0)
        );
        assert_eq!(types[0].as_i64(), Some(3));
        assert_eq!(types[3].as_i64(), Some(1));
    }
}
