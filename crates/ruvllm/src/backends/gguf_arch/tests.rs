//! GGUF architecture dispatch tests, including tiny Qwen2/Qwen3 GGUF
//! fixtures run end to end through `CandleBackend`.

use super::*;
use crate::backends::{
    CandleBackend, GenerateParams, GenerationOutput, LlmBackend, ModelConfig, StreamEvent,
};
use crate::serving::FinishReason;
use candle_core::quantized::{gguf_file, GgmlDType, QTensor};
use candle_core::{Device, Tensor};
use std::path::{Path, PathBuf};

const HIDDEN: usize = 8;
const FFN: usize = 16;
/// Toy byte-level vocabulary (GGUF token_type 1 = normal, 3 = control).
/// The last token is the GGUF-declared EOS, named so that only the file's
/// `eos_token_id` can make it a stop token.
const VOCAB: &[(&str, i32)] = &[
    ("a", 1),
    ("b", 1),
    ("c", 1),
    ("ab", 1),
    ("<|im_start|>", 3),
    ("<|im_end|>", 3),
    ("<|my_eos|>", 3),
];
const EOS: usize = 6;

fn md(entries: &[(&str, Value)]) -> HashMap<String, Value> {
    entries
        .iter()
        .map(|(k, v)| (k.to_string(), v.clone()))
        .collect()
}

fn write_gguf(path: &Path, metadata: &[(String, Value)], tensors: &[(String, QTensor)]) {
    let metadata: Vec<(&str, &Value)> = metadata.iter().map(|(k, v)| (k.as_str(), v)).collect();
    let tensors: Vec<(&str, &QTensor)> = tensors.iter().map(|(k, t)| (k.as_str(), t)).collect();
    let mut file = std::fs::File::create(path).unwrap();
    gguf_file::write(&mut file, &metadata, &tensors).unwrap();
}

fn filled(shape: &[usize], value: f32) -> QTensor {
    let t = Tensor::full(value, shape, &Device::Cpu).unwrap();
    QTensor::quantize(&t, GgmlDType::F32).unwrap()
}

/// A one-layer Qwen2/Qwen3 GGUF with all-zero layer weights, so the
/// logits are `output.weight · 1`: with `eos_wins` only the EOS row is
/// non-zero, otherwise all logits tie and greedy decoding picks token 0.
fn tiny_qwen(dir: &Path, arch: &str, eos_wins: bool) -> PathBuf {
    tiny_qwen_with(dir, arch, eos_wins, &[])
}

/// [`tiny_qwen`] with extra metadata (written to a file of its own).
fn tiny_qwen_with(dir: &Path, arch: &str, eos_wins: bool, extra: &[(&str, Value)]) -> PathBuf {
    let (heads, kv_heads) = (2usize, 1usize);
    // Qwen3 carries an explicit head dim that is not HIDDEN / heads.
    let head_dim = if arch == "qwen3" {
        HIDDEN
    } else {
        HIDDEN / heads
    };
    let s = |v: &str| Value::String(v.to_string());
    let mut metadata = vec![
        ("general.architecture".to_string(), s(arch)),
        (
            format!("{arch}.attention.head_count"),
            Value::U32(heads as u32),
        ),
        (
            format!("{arch}.attention.head_count_kv"),
            Value::U32(kv_heads as u32),
        ),
        (
            format!("{arch}.embedding_length"),
            Value::U32(HIDDEN as u32),
        ),
        (format!("{arch}.context_length"), Value::U32(64)),
        (format!("{arch}.block_count"), Value::U32(1)),
        (
            format!("{arch}.attention.layer_norm_rms_epsilon"),
            Value::F32(1e-6),
        ),
        (format!("{arch}.rope.freq_base"), Value::F32(10_000.0)),
        ("tokenizer.ggml.model".to_string(), s("gpt2")),
        ("tokenizer.ggml.pre".to_string(), s("qwen2")),
        (
            "tokenizer.ggml.tokens".to_string(),
            Value::Array(VOCAB.iter().map(|(t, _)| s(t)).collect()),
        ),
        (
            "tokenizer.ggml.token_type".to_string(),
            Value::Array(VOCAB.iter().map(|&(_, t)| Value::I32(t)).collect()),
        ),
        (
            "tokenizer.ggml.merges".to_string(),
            Value::Array(vec![s("a b")]),
        ),
        (
            "tokenizer.ggml.eos_token_id".to_string(),
            Value::U32(EOS as u32),
        ),
    ];
    if arch == "qwen3" {
        metadata.push((
            "qwen3.attention.key_length".to_string(),
            Value::U32(head_dim as u32),
        ));
    }
    metadata.extend(extra.iter().map(|(k, v)| (k.to_string(), v.clone())));

    let vocab = VOCAB.len();
    let mut output = vec![0f32; vocab * HIDDEN];
    if eos_wins {
        output[EOS * HIDDEN..(EOS + 1) * HIDDEN].fill(1.0);
    }
    let output = Tensor::from_vec(output, (vocab, HIDDEN), &Device::Cpu).unwrap();
    let (q_out, kv_out) = (heads * head_dim, kv_heads * head_dim);
    let mut tensors = vec![
        ("token_embd.weight", filled(&[vocab, HIDDEN], 1.0)),
        ("output_norm.weight", filled(&[HIDDEN], 1.0)),
        (
            "output.weight",
            QTensor::quantize(&output, GgmlDType::F32).unwrap(),
        ),
        ("blk.0.attn_norm.weight", filled(&[HIDDEN], 0.0)),
        ("blk.0.ffn_norm.weight", filled(&[HIDDEN], 0.0)),
        ("blk.0.attn_q.weight", filled(&[q_out, HIDDEN], 0.0)),
        ("blk.0.attn_k.weight", filled(&[kv_out, HIDDEN], 0.0)),
        ("blk.0.attn_v.weight", filled(&[kv_out, HIDDEN], 0.0)),
        ("blk.0.attn_output.weight", filled(&[HIDDEN, q_out], 0.0)),
        ("blk.0.ffn_gate.weight", filled(&[FFN, HIDDEN], 0.0)),
        ("blk.0.ffn_up.weight", filled(&[FFN, HIDDEN], 0.0)),
        ("blk.0.ffn_down.weight", filled(&[HIDDEN, FFN], 0.0)),
    ];
    if arch == "qwen2" {
        tensors.push(("blk.0.attn_q.bias", filled(&[q_out], 0.0)));
        tensors.push(("blk.0.attn_k.bias", filled(&[kv_out], 0.0)));
        tensors.push(("blk.0.attn_v.bias", filled(&[kv_out], 0.0)));
    } else {
        tensors.push(("blk.0.attn_q_norm.weight", filled(&[head_dim], 1.0)));
        tensors.push(("blk.0.attn_k_norm.weight", filled(&[head_dim], 1.0)));
    }
    let tensors: Vec<(String, QTensor)> = tensors
        .into_iter()
        .map(|(k, t)| (k.to_string(), t))
        .collect();

    let path = dir.join(format!("tiny-{arch}-{eos_wins}-{}.gguf", extra.len()));
    write_gguf(&path, &metadata, &tensors);
    path
}

fn load(path: &Path) -> CandleBackend {
    let mut backend = CandleBackend::default();
    backend
        .load_model(path.to_str().unwrap(), ModelConfig::default())
        .unwrap();
    backend
}

fn greedy(max_tokens: usize) -> GenerateParams {
    GenerateParams::default()
        .with_max_tokens(max_tokens)
        .with_temperature(0.0)
}

/// Generate, returning the output and the concatenated streamed deltas.
fn run(backend: &CandleBackend, params: GenerateParams) -> (GenerationOutput, String) {
    let mut streamed = String::new();
    let out = backend
        .generate_detailed("abc", params, &mut |t| {
            streamed.push_str(&t.text);
            true
        })
        .unwrap();
    (out, streamed)
}

#[test]
fn architecture_dispatch_is_exact() {
    let arch = |a: &str| {
        GgufArch::from_metadata(&md(&[("general.architecture", Value::String(a.into()))]))
    };
    assert_eq!(
        GgufArch::from_metadata(&HashMap::new()).unwrap(),
        GgufArch::Llama
    );
    assert_eq!(arch("LLaMA").unwrap(), GgufArch::Llama);
    assert_eq!(arch("mistral").unwrap(), GgufArch::Llama);
    assert_eq!(arch("qwen2").unwrap(), GgufArch::Qwen2);
    assert_eq!(arch("qwen3").unwrap(), GgufArch::Qwen3);
    for unsupported in [
        "qwen35",
        "qwen3moe",
        "qwen3next",
        "phi3",
        "gemma2",
        "llama4",
    ] {
        let err = arch(unsupported).unwrap_err().to_string();
        assert!(err.contains(&format!("'{unsupported}'")), "{err}");
        assert!(err.contains(SUPPORTED_GGUF_ARCHITECTURES), "{err}");
    }
}

#[test]
fn split_files_are_rejected_before_loading() {
    let split = md(&[
        ("general.architecture", Value::String("qwen2".into())),
        ("split.count", Value::U16(3)),
    ]);
    let err = GgufArch::from_metadata(&split).unwrap_err().to_string();
    assert!(err.contains("split GGUF (3 parts)"), "{err}");
    let single = md(&[("split.count", Value::U16(1))]);
    assert_eq!(GgufArch::from_metadata(&single).unwrap(), GgufArch::Llama);
}

#[test]
fn special_ids_come_from_metadata_within_the_vocabulary() {
    let ids = GgufSpecialIds::from_metadata(
        &md(&[
            ("tokenizer.ggml.eos_token_id", Value::U32(5)),
            ("tokenizer.ggml.bos_token_id", Value::U32(4)),
            ("tokenizer.ggml.eot_token_id", Value::U32(6)),
            ("tokenizer.ggml.eom_token_id", Value::U32(99)),
        ]),
        8,
    );
    assert_eq!(ids.eos, Some(5));
    assert_eq!(ids.bos, Some(4));
    assert_eq!(ids.stop, vec![5, 6], "out-of-vocabulary eom is dropped");
    assert_eq!(
        GgufSpecialIds::from_metadata(&HashMap::new(), 8),
        GgufSpecialIds::default()
    );
}

#[test]
fn streamable_len_holds_back_partial_utf8_and_stop_prefixes() {
    assert_eq!(streamable_len("ab\u{FFFD}", 0), 2);
    assert_eq!(streamable_len("abc", 2), 1);
    // 'é' is two bytes: never split it.
    assert_eq!(streamable_len("aé", 1), 1);
    assert_eq!(streamable_len("", 3), 0);
    // Pending bytes do not count toward the holdback: with stop "b€"
    // (holdback 3), "ab" + a partial '€' may still become "ab€".
    assert_eq!(streamable_len("ab\u{FFFD}", 3), 0);
    assert_eq!(streamable_len("abcd\u{FFFD}", 3), 1);
    let stops = ["".to_string(), "lo".to_string(), "l".to_string()];
    assert_eq!(find_stop("hello", &stops), Some(2));
    assert_eq!(find_stop("hey", &stops), None);
}

#[test]
fn unsupported_gguf_fails_by_name_and_unloads_the_previous_model() {
    let dir = tempfile::tempdir().unwrap();
    let mut backend = load(&tiny_qwen(dir.path(), "qwen2", false));
    assert!(backend.is_model_loaded());

    let path = dir.path().join("hybrid.gguf");
    let arch = Value::String("qwen35".into());
    write_gguf(&path, &[("general.architecture".to_string(), arch)], &[]);
    let err = backend
        .load_model(path.to_str().unwrap(), ModelConfig::default())
        .unwrap_err()
        .to_string();
    assert!(err.contains("'qwen35' is not supported"), "{err}");
    assert!(!backend.is_model_loaded());
    assert!(backend.tokenizer().is_none());
}

#[test]
fn qwen_ggufs_load_with_their_own_loader_and_template() {
    let dir = tempfile::tempdir().unwrap();
    for arch in ["qwen2", "qwen3"] {
        let backend = load(&tiny_qwen(dir.path(), arch, false));
        let info = backend.model_info().unwrap();
        assert_eq!(info.architecture, super::ModelArchitecture::Qwen);
        assert_eq!(info.vocab_size, VOCAB.len());
        assert_eq!(
            LlmBackend::chat_template(&backend),
            Some(ChatTemplate::Qwen)
        );
        let eos = backend.tokenizer().unwrap().special_tokens().eos_token_id;
        assert_eq!(eos, Some(EOS as u32), "{arch}: EOS from GGUF metadata");
    }
}

#[test]
fn max_tokens_ends_with_length_and_exact_counts_on_every_request() {
    let dir = tempfile::tempdir().unwrap();
    for arch in ["qwen2", "qwen3"] {
        let backend = load(&tiny_qwen(dir.path(), arch, false));
        // Twice: Qwen3's KV cache must be cleared between requests.
        for _ in 0..2 {
            let (out, streamed) = run(&backend, greedy(3));
            assert_eq!(
                out,
                GenerationOutput {
                    text: "aaa".into(),
                    prompt_tokens: 2, // "ab" + "c"
                    completion_tokens: 3,
                    finish_reason: FinishReason::Length,
                },
                "{arch}"
            );
            assert_eq!(streamed, out.text);
        }
    }
}

#[test]
fn gguf_eos_token_ends_generation() {
    let dir = tempfile::tempdir().unwrap();
    for arch in ["qwen2", "qwen3"] {
        let backend = load(&tiny_qwen(dir.path(), arch, true));
        let (out, streamed) = run(&backend, greedy(8));
        assert_eq!(out.finish_reason, FinishReason::EndOfSequence, "{arch}");
        assert_eq!((out.text.as_str(), out.completion_tokens), ("", 1));
        assert!(streamed.is_empty());
    }
}

#[test]
fn stop_sequences_are_cut_and_never_streamed() {
    let dir = tempfile::tempdir().unwrap();
    let backend = load(&tiny_qwen(dir.path(), "qwen2", false));

    let (out, streamed) = run(&backend, greedy(5).with_stop_sequence("aa"));
    assert_eq!(out.finish_reason, FinishReason::Stop);
    assert_eq!((out.text.as_str(), out.completion_tokens), ("", 2));
    assert_eq!(streamed, "");

    // A held-back possible stop prefix is flushed when it never completes.
    let (out, streamed) = run(&backend, greedy(3).with_stop_sequence("ab"));
    assert_eq!(out.finish_reason, FinishReason::Length);
    assert_eq!((out.text.as_str(), streamed.as_str()), ("aaa", "aaa"));
}

#[test]
fn callback_can_cancel_and_stream_replays_tokens() {
    let dir = tempfile::tempdir().unwrap();
    let backend = load(&tiny_qwen(dir.path(), "qwen2", false));
    let out = backend
        .generate_detailed("abc", greedy(5), &mut |_| false)
        .unwrap();
    assert_eq!(out.finish_reason, FinishReason::Cancelled);
    assert_eq!(out.completion_tokens, 1);

    let events: Vec<StreamEvent> = backend
        .generate_stream_v2("abc", greedy(3))
        .unwrap()
        .map(|e| e.unwrap())
        .collect();
    let text: String = events
        .iter()
        .filter_map(|e| match e {
            StreamEvent::Token(t) => Some(t.text.as_str()),
            _ => None,
        })
        .collect();
    assert_eq!(text, "aaa");
    assert!(matches!(
        events.last(),
        Some(StreamEvent::Done {
            total_tokens: 3,
            ..
        })
    ));
}

#[test]
fn chat_template_comes_from_the_file_for_every_architecture() {
    let tpl = |t: &str| md(&[("tokenizer.chat_template", Value::String(t.into()))]);
    let chatml = "{{'<|im_start|>' + m['role'] + '\\n'}}";
    let deepseek = "{{'<｜User｜>' + m['content'] + '<｜Assistant｜>'}}";
    for qwen in [GgufArch::Qwen2, GgufArch::Qwen3] {
        let ok = |m: &HashMap<String, Value>, tokens| qwen.chat_template(m, tokens).unwrap();
        assert_eq!(ok(&tpl(chatml), true), Some(ChatTemplate::Qwen));
        assert_eq!(ok(&HashMap::new(), true), Some(ChatTemplate::Qwen));
        let err = qwen.chat_template(&tpl(deepseek), true).unwrap_err();
        assert!(err.to_string().contains("not ChatML"), "{err}");
        let err = qwen.chat_template(&HashMap::new(), false).unwrap_err();
        assert!(err.to_string().contains("no <|im_start|>"), "{err}");
    }
    let llama = |t: &str| GgufArch::Llama.chat_template(&tpl(t), false).unwrap();
    assert_eq!(llama(chatml), Some(ChatTemplate::ChatML));
    assert_eq!(llama("<|start_header_id|>"), Some(ChatTemplate::Llama3));
    assert_eq!(llama("[INST] <<SYS>>"), Some(ChatTemplate::Llama2));
    assert_eq!(llama("[INST] {{ m }} [/INST]"), Some(ChatTemplate::Mistral));
    assert_eq!(llama(deepseek), None, "unknown: the caller guesses by name");
    // Only the last path component names the model.
    assert_eq!(
        template_from_name("/home/phil/models/llama-3-8b.gguf"),
        ChatTemplate::Llama3
    );
    assert_eq!(
        template_from_name("/home/phil/tinyllama/"),
        ChatTemplate::ChatML
    );
    assert_eq!(
        template_from_name("Qwen/Qwen2.5-0.5B-Instruct-GGUF"),
        ChatTemplate::Qwen
    );
}

#[test]
fn non_chatml_qwen_gguf_fails_to_load_naming_the_format() {
    let dir = tempfile::tempdir().unwrap();
    let template = Value::String("{{'<｜User｜>' + m['content']}}".into());
    let path = tiny_qwen_with(
        dir.path(),
        "qwen2",
        false,
        &[("tokenizer.chat_template", template)],
    );
    let mut backend = CandleBackend::default();
    let err = backend
        .load_model(path.to_str().unwrap(), ModelConfig::default())
        .unwrap_err()
        .to_string();
    assert!(
        err.contains("'qwen2'") && err.contains("not ChatML"),
        "{err}"
    );
    assert!(!backend.is_model_loaded());
}

#[test]
fn quantization_is_read_from_the_file() {
    let ft = |n: u32| md(&[("general.file_type", Value::U32(n))]);
    let hint = Some(Quantization::Q4K);
    assert_eq!(file_quantization(&ft(7), hint), Some(Quantization::Q8));
    assert_eq!(file_quantization(&ft(15), None), Some(Quantization::Q4K));
    assert_eq!(file_quantization(&ft(18), hint), None, "Q6_K: no variant");
    assert_eq!(file_quantization(&HashMap::new(), hint), hint);

    let dir = tempfile::tempdir().unwrap();
    let path = tiny_qwen_with(
        dir.path(),
        "qwen3",
        false,
        &[("general.file_type", Value::U32(0))],
    );
    let config = ModelConfig {
        quantization: Some(Quantization::Q4K),
        ..ModelConfig::default()
    };
    let mut backend = CandleBackend::default();
    backend.load_model(path.to_str().unwrap(), config).unwrap();
    let info = backend.model_info().unwrap();
    assert_eq!(info.quantization, Some(Quantization::None));
}

#[test]
fn a_seeded_sampler_draws_a_fresh_number_per_token() {
    // Uniform logits: a sampler re-created per token with the same seed
    // would return the same index every time.
    let params = GenerateParams::default().with_temperature(0.7).with_seed(7);
    let mut sampler = CandleBackend::logits_processor(&params);
    let uniform = Tensor::zeros(16, candle_core::DType::F32, &Device::Cpu).unwrap();
    let draws: Vec<u32> = (0..12).map(|_| sampler.sample(&uniform).unwrap()).collect();
    assert!(draws.iter().any(|&d| d != draws[0]), "{draws:?}");

    // ...and a fixed seed still reproduces a generation exactly.
    let dir = tempfile::tempdir().unwrap();
    let backend = load(&tiny_qwen(dir.path(), "qwen2", false));
    let sampled = || run(&backend, params.clone().with_max_tokens(6)).0;
    assert_eq!(sampled(), sampled());
}

#[test]
fn streaming_without_a_model_is_an_error_not_a_mock() {
    let backend = CandleBackend::default();
    assert!(backend.generate_stream_v2("hello", greedy(3)).is_err());
    assert!(backend.generate("hello", greedy(3)).is_err());
}
