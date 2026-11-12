//! Lattice-based LLM inference backend
//!
//! This module provides a pluggable [`LlmBackend`] on top of
//! [`lattice-inference`](https://crates.io/crates/lattice-inference), a
//! pure-Rust Qwen3.5 inference engine with a hand-written Metal GPU forward
//! pass. macOS-only today: `lattice-inference`'s Metal path FFI-binds
//! `metal`/`objc`, which don't exist off Apple platforms, so both the
//! dependency (`Cargo.toml`, `[target.'cfg(target_os = "macos")'.dependencies]`)
//! and this module (`backends/mod.rs`, `#[cfg(all(feature = "lattice",
//! target_os = "macos"))]`) are target-gated.
//!
//! ## Concurrency
//!
//! `lattice_inference::forward::metal_qwen35::MetalQwen35State` owns raw
//! `metal::*` objects and is `!Send`, so it lives on one dedicated worker
//! thread for the lifetime of the loaded model: the same shape
//! `lattice_serve.rs` (lattice's own OpenAI-compatible server binary) already
//! uses for the identical problem, just riding plain `std::sync::mpsc`
//! instead of tokio's, since ruvllm's [`TokenStream`](super::TokenStream) is
//! std-mpsc-backed. [`LatticeBackend`] holds the command sender and worker
//! handle; the `!Send` state never crosses a thread boundary. Unload and
//! drop join the worker after closing the command channel.
//!
//! ## Lattice Backend Example
//!
//! ```rust,ignore
//! use ruvllm::backends::{LatticeBackend, ModelConfig, GenerateParams, LlmBackend};
//!
//! let mut backend = LatticeBackend::new()?;
//!
//! // A local directory containing tokenizer.json + config.json, plus either
//! // a lattice Q4 weight set (*.q4 files) or a Qwen3.5 safetensors checkpoint.
//! backend.load_model("/path/to/qwen3.5-0.8b", ModelConfig::default())?;
//!
//! let params = GenerateParams::default()
//!     .with_max_tokens(256)
//!     .with_temperature(0.7);
//!
//! let response = backend.generate("Hello, world!", params)?;
//! ```

use super::{
    AdapterInfo, GenerateParams, GeneratedToken, LlmBackend, ModelArchitecture, ModelConfig,
    ModelInfo, Quantization, SpecialTokens, StreamEvent, TokenStream, Tokenizer,
};
use crate::error::{ModelAdapterError, Result, RuvLLMError};

use lattice_inference::forward::metal_qwen35::MetalQwen35State;
use lattice_inference::model::qwen35::Qwen35Model;
use lattice_inference::model::qwen35_config::Qwen35Config;
use lattice_inference::tokenizer::bpe::BpeTokenizer;
use lattice_inference::GenerateConfig;
// Brought into scope for method-resolution only (`.tokenize()`/`.decode()` on
// `BpeTokenizer` are trait-provided, not inherent), aliased because ruvllm
// already has its own `Tokenizer` trait (`super::Tokenizer`) in this file.
use lattice_inference::tokenizer::Tokenizer as LatticeTokenizerTrait;

use std::collections::HashSet;
use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::sync::Arc;
use std::time::Instant;

mod adapter;
#[cfg(test)]
mod adapter_tests;

type AdapterResult<T> = std::result::Result<T, ModelAdapterError>;

/// How a `generate`/`generate_stream_v2` job wants its result delivered.
enum Reply {
    /// `generate`: block for the whole accumulated string.
    Once(mpsc::Sender<Result<String>>),
    /// `generate_stream_v2`: one `StreamEvent` per token, then `Done`.
    Stream(mpsc::Sender<StreamEvent>),
}

/// Operations serialized on the thread that owns the Metal state.
///
/// Mirrors `lattice_serve.rs`'s `Job`/`spawn_worker`/worker-loop shape (the
/// existing, working answer to `MetalQwen35State` being `!Send`), but rides
/// plain `std::sync::mpsc` instead of tokio's mpsc, because `TokenStream` is
/// std-mpsc-backed.
enum Job {
    Generate {
        prompt: String,
        params: GenerateParams,
        reply: Reply,
    },
    LoadAdapter {
        path: PathBuf,
        scale: f32,
        reply: mpsc::Sender<AdapterResult<()>>,
    },
    UnloadAdapter {
        reply: mpsc::Sender<AdapterResult<()>>,
    },
    ActiveAdapter {
        reply: mpsc::Sender<AdapterResult<Option<AdapterInfo>>>,
    },
    #[cfg(test)]
    AdapterLogits {
        tokens: Vec<u32>,
        reply: mpsc::Sender<AdapterResult<Vec<f32>>>,
    },
    #[cfg(test)]
    FailNextAdapterInstalls {
        count: usize,
        reply: mpsc::Sender<AdapterResult<()>>,
    },
}

/// Everything the backend needs once the worker thread has finished loading,
/// without ever handing back the `!Send` Metal state itself.
struct WorkerReady {
    info: ModelInfo,
    tokenizer: BpeTokenizer,
}

/// Map ruvllm's `GenerateParams` onto lattice's `GenerateConfig`.
///
/// `frequency_penalty`/`presence_penalty` have no lattice equivalent
/// (lattice's sampler exposes `repetition_penalty` only), so nonzero values
/// are rejected up front in `generate`/`generate_stream_v2` rather than
/// silently ignored here; see [`reject_unsupported_params`].
///
/// String stops are passed to Lattice and the existing local scanner also
/// protects the backend's streamed-text boundary.
fn to_generate_config(params: &GenerateParams) -> GenerateConfig {
    // `GenerateConfig` is `#[non_exhaustive]`, so it is built from its default
    // and the mapped fields are assigned one by one.
    let mut cfg = GenerateConfig::default();
    cfg.max_new_tokens = params.max_tokens;
    cfg.temperature = params.temperature;
    cfg.top_k = params.top_k;
    cfg.top_p = params.top_p;
    cfg.repetition_penalty = params.repetition_penalty;
    cfg.seed = params.seed;
    cfg.stop_strings = params.stop_sequences.clone();
    cfg
}

/// Reject `GenerateParams` values this backend cannot honor, instead of
/// silently changing public API behavior (the serving engine and the mistral
/// backend both treat these penalties as live: serving/engine.rs:547,
/// mistral_backend.rs:907).
fn reject_unsupported_params(params: &GenerateParams) -> Result<()> {
    if params.frequency_penalty != 0.0 || params.presence_penalty != 0.0 {
        return Err(RuvLLMError::NotImplemented(
            "LatticeBackend does not support frequency_penalty/presence_penalty \
             (lattice's sampler exposes repetition_penalty only); set them to 0.0"
                .to_string(),
        ));
    }
    Ok(())
}

/// Incremental stop-string scanner over streamed deltas.
///
/// The backend retains its own streamed-text boundary check: scan the
/// accumulated text, hold back the longest possible stop-string prefix so a
/// partial match never leaks to the caller, and cut generation through the
/// token callback's `false` return as soon as a stop matches. The matched
/// stop string itself is excluded from the output, matching
/// `GenerateConfig::stop_strings` semantics.
struct StopScan {
    stops: Vec<String>,
    /// Bytes to hold back: the longest stop is `hold + 1` bytes long, so any
    /// suffix that could still grow into a match stays in `buf`.
    hold: usize,
    buf: String,
    stopped: bool,
}

impl StopScan {
    /// Returns `None` when there is nothing to scan for (no non-empty stops),
    /// letting callers skip the buffering entirely.
    fn new(stops: &[String]) -> Option<Self> {
        let stops: Vec<String> = stops.iter().filter(|s| !s.is_empty()).cloned().collect();
        let hold = stops.iter().map(|s| s.len()).max()?.saturating_sub(1);
        Some(Self {
            stops,
            hold,
            buf: String::new(),
            stopped: false,
        })
    }

    /// Feed one delta; returns the text that is now safe to emit. Sets
    /// `self.stopped` (and drops the match plus everything after it) when a
    /// stop string is found.
    fn push(&mut self, delta: &str) -> String {
        self.buf.push_str(delta);
        if let Some(pos) = self
            .stops
            .iter()
            .filter_map(|s| self.buf.find(s.as_str()))
            .min()
        {
            self.stopped = true;
            let head = self.buf[..pos].to_string();
            self.buf.clear();
            return head;
        }
        if self.buf.len() <= self.hold {
            return String::new();
        }
        // Emit all but the trailing `hold` bytes, snapped down to a char
        // boundary so multi-byte codepoints are never split.
        let mut split = self.buf.len() - self.hold;
        while !self.buf.is_char_boundary(split) {
            split -= 1;
        }
        let head = self.buf[..split].to_string();
        self.buf.drain(..split);
        head
    }

    /// Generation ended without a stop match: release the held-back tail.
    fn finish(&mut self) -> String {
        std::mem::take(&mut self.buf)
    }
}

/// Sum the on-disk byte size of every file in `dir` with extension `ext`.
/// Used to derive `ModelInfo::memory_usage`, mirroring candle_backend.rs's
/// own sum-of-weight-file-sizes approach (candle_backend.rs:929-933).
fn sum_file_sizes(dir: &Path, ext: &str) -> usize {
    let mut total = 0usize;
    if let Ok(entries) = std::fs::read_dir(dir) {
        for entry in entries.flatten() {
            if entry.path().extension().and_then(|e| e.to_str()) == Some(ext) {
                if let Ok(meta) = entry.metadata() {
                    total += meta.len() as usize;
                }
            }
        }
    }
    total
}

/// Derive the `ModelInfo` precision label for a safetensors checkpoint from
/// its own config.json `torch_dtype`, mirroring lattice_bench.rs's honesty
/// guard (O3: the label must match the actual artifact) instead of
/// hard-coding `Bf16`. Falls back to `Bf16` — the Qwen3.5 release dtype and
/// this backend's previous fixed label — when the field is missing or names
/// a dtype we don't map; a label fallback must not fail a load that
/// `from_safetensors` already accepted.
fn safetensors_precision_label(config_path: &Path) -> Quantization {
    std::fs::read_to_string(config_path)
        .ok()
        .and_then(|s| serde_json::from_str::<serde_json::Value>(&s).ok())
        .and_then(|v| {
            v.get("torch_dtype")
                .and_then(|d| d.as_str())
                .map(|d| d.to_string())
        })
        .map(|dtype| match dtype.as_str() {
            "float16" => Quantization::F16,
            "float32" => Quantization::None,
            _ => Quantization::Bf16,
        })
        .unwrap_or(Quantization::Bf16)
}

/// Probe well-known Qwen special-token strings, mirroring candle_backend.rs's
/// own probing approach for its `SpecialTokens` (candle_backend.rs:530-547).
fn probe_special_tokens(tok: &BpeTokenizer) -> SpecialTokens {
    SpecialTokens {
        bos_token_id: tok.special_token_id("<|im_start|>"),
        eos_token_id: tok
            .special_token_id("<|endoftext|>")
            .or_else(|| tok.special_token_id("<|im_end|>")),
        pad_token_id: tok.special_token_id("<|endoftext|>"),
        unk_token_id: None,
    }
}

/// Load the Metal state + tokenizer + derived `ModelInfo` for `model_dir`.
/// Mirrors `lattice_serve.rs::load_model`'s Q4-dir-vs-safetensors branch
/// (lattice_serve.rs:331-358) exactly, generalized to auto-detect the format
/// from directory contents instead of a CLI flag.
fn load_worker_state(
    model_dir: &Path,
    tokenizer_path: &Path,
    config_path: &Path,
    is_q4: bool,
    max_seq_len_hint: usize,
) -> std::result::Result<(MetalQwen35State, BpeTokenizer, ModelInfo), String> {
    let tokenizer = BpeTokenizer::from_tokenizer_json(tokenizer_path)
        .map_err(|e| format!("tokenizer load failed ({}): {e}", tokenizer_path.display()))?;

    let cfg = Qwen35Config::from_config_json(config_path)
        .map_err(|e| format!("config.json parse failed: {e}"))?;
    // Honor the caller's `max_sequence_length` (ModelConfig) as the KV-cache
    // window, clamped to what the RoPE table actually supports; exceeding
    // `max_position_embeddings` is refused inside `from_q4_dir`/`new_session`
    // anyway, so clamp here for a clean error message instead of a load failure.
    let max_cache_len = max_seq_len_hint.min(cfg.max_position_embeddings).max(1);

    let (metal, quant, memory_usage) = if is_q4 {
        let metal = MetalQwen35State::from_q4_dir(model_dir, tokenizer_path, &cfg, max_cache_len)
            .map_err(|e| format!("Q4 model load failed: {e}"))?;
        (metal, Quantization::Q4, sum_file_sizes(model_dir, "q4"))
    } else {
        let model = Qwen35Model::from_safetensors(model_dir)
            .map_err(|e| format!("safetensors load failed: {e}"))?;
        let metal = MetalQwen35State::new(model.weights(), model.config(), max_cache_len)
            .map_err(|e| format!("Metal init failed: {e}"))?;
        (
            metal,
            safetensors_precision_label(config_path),
            sum_file_sizes(model_dir, "safetensors"),
        )
    };

    // No live parameter count is available from `Qwen35Config` alone (Qwen3.5
    // mixes GDN/full-attention/MoE layers, so a generic per-layer formula like
    // candle_backend.rs's `estimate_parameters` would be architecture-wrong
    // here); derive it instead from what we already measured: on-disk bytes
    // divided by the known bytes-per-weight for the quant format we just loaded.
    let num_parameters = (memory_usage as f32 / quant.bytes_per_weight()) as usize;

    let info = ModelInfo {
        name: model_dir.display().to_string(),
        architecture: ModelArchitecture::Qwen,
        num_parameters,
        vocab_size: cfg.vocab_size,
        hidden_size: cfg.hidden_size,
        num_layers: cfg.num_hidden_layers,
        max_context_length: max_cache_len,
        quantization: Some(quant),
        memory_usage,
    };

    Ok((metal, tokenizer, info))
}

fn adapter_backend_error(message: impl Into<String>) -> ModelAdapterError {
    ModelAdapterError::Backend {
        message: message.into(),
    }
}

#[derive(Default)]
struct AdapterInstaller {
    #[cfg(test)]
    failures_remaining: usize,
}

impl AdapterInstaller {
    fn install(
        &mut self,
        state: &mut MetalQwen35State,
        prepared: &adapter::PreparedAdapter,
    ) -> AdapterResult<()> {
        #[cfg(test)]
        if self.failures_remaining > 0 {
            self.failures_remaining -= 1;
            return Err(adapter_backend_error(
                "injected adapter installation failure",
            ));
        }
        prepared.install(state)
    }
}

/// Spawn the dedicated thread that owns the `!Send` `MetalQwen35State` for
/// the lifetime of the loaded model. Reuses `lattice_serve.rs`'s worker shape
/// (module doc :29-32, `spawn_worker` :244, `run_worker_loop` :289): the
/// non-`Send` state never crosses a thread boundary, only `Job`s (plain data)
/// do. Jobs are served one at a time, matching serve, and Metal is
/// single-context anyway; concurrent `&self` calls simply queue on the channel.
fn spawn_worker(
    model_dir: PathBuf,
    tokenizer_path: PathBuf,
    config_path: PathBuf,
    is_q4: bool,
    max_seq_len_hint: usize,
    ready: mpsc::Sender<std::result::Result<WorkerReady, String>>,
) -> (mpsc::Sender<Job>, std::thread::JoinHandle<()>) {
    let (job_tx, job_rx) = mpsc::channel::<Job>();
    let worker = std::thread::spawn(move || {
        let model_context =
            adapter::ModelContext::read(&model_dir).map_err(|error| error.to_string());
        let loaded = load_worker_state(
            &model_dir,
            &tokenizer_path,
            &config_path,
            is_q4,
            max_seq_len_hint,
        );
        let (mut metal, tokenizer, info) = match loaded {
            Ok(t) => t,
            Err(e) => {
                let _ = ready.send(Err(e));
                return;
            }
        };
        let mut active_adapter: Option<adapter::PreparedAdapter> = None;
        let mut installer = AdapterInstaller::default();

        let special = probe_special_tokens(&tokenizer);
        let special_ids: HashSet<u32> = [
            special.bos_token_id,
            special.eos_token_id,
            special.pad_token_id,
            special.unk_token_id,
        ]
        .into_iter()
        .flatten()
        .collect();

        if ready
            .send(Ok(WorkerReady {
                info,
                tokenizer: tokenizer.clone(),
            }))
            .is_err()
        {
            // The caller gave up (dropped the ready receiver) before we
            // finished loading; nothing left to serve.
            return;
        }

        for job in job_rx {
            let (prompt, params, reply) = match job {
                Job::Generate {
                    prompt,
                    params,
                    reply,
                } => (prompt, params, reply),
                Job::LoadAdapter { path, scale, reply } => {
                    let result = (|| {
                        let prepared = adapter::prepare_captured(&model_context, &path, scale)?;
                        metal.unload_lora_adapter();
                        metal.reset_state();
                        if let Err(install_error) = installer.install(&mut metal, &prepared) {
                            metal.unload_lora_adapter();
                            metal.reset_state();
                            if let Some(previous) = active_adapter.as_ref() {
                                if let Err(restore_error) = installer.install(&mut metal, previous)
                                {
                                    metal.unload_lora_adapter();
                                    metal.reset_state();
                                    active_adapter = None;
                                    return Err(adapter_backend_error(format!(
                                        "adapter installation failed: {install_error}; \
                                         previous adapter restoration also failed: {restore_error}; \
                                         no active adapter; base-model generation is active"
                                    )));
                                }
                                metal.reset_state();
                                return Err(adapter_backend_error(format!(
                                    "adapter installation failed: {install_error}; \
                                     previous adapter restored"
                                )));
                            }
                            return Err(adapter_backend_error(format!(
                                "adapter installation failed: {install_error}; \
                                 no active adapter; base-model generation is active"
                            )));
                        }
                        metal.reset_state();
                        active_adapter = Some(prepared);
                        Ok(())
                    })();
                    let _ = reply.send(result);
                    continue;
                }
                Job::UnloadAdapter { reply } => {
                    metal.unload_lora_adapter();
                    metal.reset_state();
                    active_adapter = None;
                    let _ = reply.send(Ok(()));
                    continue;
                }
                Job::ActiveAdapter { reply } => {
                    let _ = reply.send(Ok(active_adapter
                        .as_ref()
                        .map(|adapter| adapter.info.clone())));
                    continue;
                }
                #[cfg(test)]
                Job::FailNextAdapterInstalls { count, reply } => {
                    installer.failures_remaining = count;
                    let _ = reply.send(Ok(()));
                    continue;
                }
                #[cfg(test)]
                Job::AdapterLogits { tokens, reply } => {
                    metal.reset_state();
                    let result = (|| {
                        if tokens.is_empty() {
                            return Err(adapter_backend_error("logit probe requires tokens"));
                        }
                        let mut logits = Vec::new();
                        for (position, token) in tokens.into_iter().enumerate() {
                            logits = metal.try_forward_step(token, position).map_err(|error| {
                                adapter_backend_error(format!(
                                    "adapter logit probe failed: {error}"
                                ))
                            })?;
                        }
                        Ok(logits)
                    })();
                    metal.reset_state();
                    let _ = reply.send(result);
                    continue;
                }
            };
            let cfg = to_generate_config(&params);
            match reply {
                Reply::Once(tx) => {
                    let result = match StopScan::new(&cfg.stop_strings) {
                        // No stop strings: the plain accumulate-everything path.
                        None => metal
                            .generate(&prompt, &tokenizer, &cfg)
                            .map(|out| out.text),
                        // Stop strings requested: drive the streaming loop so a
                        // match actually HALTS generation (StopScan doc above),
                        // instead of truncating after a full max_tokens run.
                        Some(mut scan) => {
                            let mut text = String::new();
                            metal
                                .generate_streaming_with_cancel(
                                    &prompt,
                                    &tokenizer,
                                    &cfg,
                                    |delta, _id| {
                                        text.push_str(&scan.push(delta));
                                        !scan.stopped
                                    },
                                    || false,
                                )
                                .map(|_| {
                                    if !scan.stopped {
                                        text.push_str(&scan.finish());
                                    }
                                    text
                                })
                        }
                    };
                    let _ = tx.send(result.map_err(|e| {
                        RuvLLMError::Backend(format!("lattice generation failed: {e}"))
                    }));
                }
                Reply::Stream(tx) => {
                    let start = Instant::now();
                    let mut scan = StopScan::new(&cfg.stop_strings);
                    // With a StopScan active, emitted text lags the decode by
                    // up to `hold` bytes, so each emitted chunk carries the id
                    // of the token whose delta released it (approximate but
                    // monotone); the flushed tail reuses the last seen id.
                    let mut last_id = 0u32;
                    let result = metal.generate_streaming_with_cancel(
                        &prompt,
                        &tokenizer,
                        &cfg,
                        |delta, id| {
                            last_id = id;
                            let (text, keep_going) = match scan.as_mut() {
                                None => (delta.to_string(), true),
                                Some(s) => (s.push(delta), !s.stopped),
                            };
                            let send_ok = if text.is_empty() {
                                true
                            } else {
                                tx.send(StreamEvent::Token(GeneratedToken {
                                    id,
                                    text,
                                    logprob: None,
                                    is_special: special_ids.contains(&id),
                                }))
                                .is_ok()
                            };
                            keep_going && send_ok
                        },
                        || false,
                    );
                    let out = match result {
                        Ok(out) => out,
                        Err(e) => {
                            let _ = tx.send(StreamEvent::Error(format!(
                                "lattice generation failed: {e}"
                            )));
                            continue;
                        }
                    };
                    if let Some(s) = scan.as_mut() {
                        if !s.stopped {
                            let tail = s.finish();
                            if !tail.is_empty() {
                                let _ = tx.send(StreamEvent::Token(GeneratedToken {
                                    id: last_id,
                                    text: tail,
                                    logprob: None,
                                    is_special: special_ids.contains(&last_id),
                                }));
                            }
                        }
                    }
                    let duration_ms = start.elapsed().as_millis() as u64;
                    let tokens_per_second = if duration_ms > 0 {
                        out.generated_tokens as f64 / (duration_ms as f64 / 1000.0)
                    } else {
                        0.0
                    };
                    let _ = tx.send(StreamEvent::Done {
                        total_tokens: out.generated_tokens,
                        duration_ms,
                        tokens_per_second,
                    });
                }
            }
        }
    });
    (job_tx, worker)
}

/// Wraps a lattice `BpeTokenizer` (cheaply `Clone`, `Arc`-backed internally)
/// behind ruvllm's [`Tokenizer`] trait object.
pub struct LatticeTok {
    inner: BpeTokenizer,
    special: SpecialTokens,
}

impl Tokenizer for LatticeTok {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        let input = self.inner.tokenize(text);
        Ok(input.input_ids[..input.real_length].to_vec())
    }

    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens).ok_or_else(|| {
            RuvLLMError::Tokenization("lattice BpeTokenizer::decode returned None".to_string())
        })
    }

    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }

    fn special_tokens(&self) -> SpecialTokens {
        self.special.clone()
    }
}

/// Lattice-based inference backend.
///
/// Provides Qwen3.5 inference via lattice's pure-Rust Metal GPU forward pass.
/// See the module docs for the concurrency model and a usage example.
pub struct LatticeBackend {
    /// `None` until `load_model` succeeds. The `Sender` is `Send + Sync` even
    /// though the `!Send` `MetalQwen35State` it feeds jobs to is not; the
    /// state itself never leaves the dedicated worker thread.
    jobs: Option<mpsc::Sender<Job>>,
    worker: Option<std::thread::JoinHandle<()>>,
    info: Option<ModelInfo>,
    /// Cloneable snapshot for `tokenizer()`.
    tok: Option<Arc<LatticeTok>>,
    model_id: String,
}

impl Default for LatticeBackend {
    fn default() -> Self {
        Self {
            jobs: None,
            worker: None,
            info: None,
            tok: None,
            model_id: String::new(),
        }
    }
}

impl LatticeBackend {
    /// Create an unloaded lattice backend.
    ///
    /// Mirrors `CandleBackend::new()`'s fallible shape; lattice has no
    /// device-selection step to fail on here (Metal device lookup happens
    /// lazily inside the worker thread at `load_model` time), so this
    /// particular constructor cannot itself fail.
    pub fn new() -> Result<Self> {
        Ok(Self::default())
    }

    fn adapter_request<T>(
        &self,
        command: impl FnOnce(mpsc::Sender<AdapterResult<T>>) -> Job,
    ) -> AdapterResult<T> {
        let jobs = self
            .jobs
            .as_ref()
            .ok_or(ModelAdapterError::ModelNotLoaded)?;
        let (reply, response) = mpsc::channel();
        jobs.send(command(reply))
            .map_err(|_| adapter_backend_error("lattice worker thread has exited"))?;
        response.recv().map_err(|_| {
            adapter_backend_error("lattice worker thread dropped the adapter reply channel")
        })?
    }

    #[cfg(test)]
    fn adapter_logits(&self, tokens: Vec<u32>) -> AdapterResult<Vec<f32>> {
        self.adapter_request(|reply| Job::AdapterLogits { tokens, reply })
    }

    #[cfg(test)]
    fn adapter_fail_next_installs(&self, count: usize) -> AdapterResult<()> {
        self.adapter_request(|reply| Job::FailNextAdapterInstalls { count, reply })
    }
}

impl Drop for LatticeBackend {
    fn drop(&mut self) {
        self.unload_model();
    }
}

impl LlmBackend for LatticeBackend {
    fn load_model(&mut self, model_id: &str, config: ModelConfig) -> Result<()> {
        let path = Path::new(model_id);
        if !path.is_dir() {
            return Err(RuvLLMError::NotFound(format!(
                "lattice backend requires a local model directory; '{model_id}' is not a directory"
            )));
        }
        let model_dir = std::fs::canonicalize(path).map_err(|error| {
            RuvLLMError::Model(format!("cannot resolve base model directory: {error}"))
        })?;
        let path = model_dir.as_path();
        let tokenizer_path = path.join("tokenizer.json");
        if !tokenizer_path.exists() {
            return Err(RuvLLMError::NotFound(format!(
                "tokenizer.json not found in {}",
                path.display()
            )));
        }
        let config_path = path.join("config.json");
        if !config_path.exists() {
            return Err(RuvLLMError::NotFound(format!(
                "config.json not found in {}",
                path.display()
            )));
        }

        // Directory-content sniff, mirroring candle_backend.rs's own
        // GGUF-file-scan-then-safetensors-fallback dir dispatch
        // (candle_backend.rs:1260-1297): prefer the lattice Q4 weight set
        // when both are present.
        let mut has_q4 = false;
        let mut has_safetensors = false;
        if let Ok(entries) = std::fs::read_dir(path) {
            for entry in entries.flatten() {
                match entry.path().extension().and_then(|e| e.to_str()) {
                    Some("q4") => has_q4 = true,
                    Some("safetensors") => has_safetensors = true,
                    _ => {}
                }
            }
        }
        if !has_q4 && !has_safetensors {
            return Err(RuvLLMError::NotFound(format!(
                "no .q4 or .safetensors files found in {}",
                path.display()
            )));
        }

        let (ready_tx, ready_rx) = mpsc::channel();
        let (job_tx, worker) = spawn_worker(
            path.to_path_buf(),
            tokenizer_path,
            config_path,
            has_q4,
            config.max_sequence_length,
            ready_tx,
        );

        let mut worker_ready = match ready_rx.recv() {
            Ok(Ok(ready)) => ready,
            failed => {
                drop(job_tx);
                let _ = worker.join();
                return Err(match failed {
                    Ok(Err(error)) => RuvLLMError::Model(error),
                    _ => RuvLLMError::Backend(
                        "lattice worker thread exited before signaling readiness".to_string(),
                    ),
                });
            }
        };

        worker_ready.info.name = model_id.to_owned();
        let special = probe_special_tokens(&worker_ready.tokenizer);
        self.unload_model();
        self.tok = Some(Arc::new(LatticeTok {
            inner: worker_ready.tokenizer,
            special,
        }));
        self.info = Some(worker_ready.info);
        self.jobs = Some(job_tx);
        self.worker = Some(worker);
        self.model_id = path.display().to_string();
        Ok(())
    }

    fn load_lora_adapter(
        &mut self,
        path: &Path,
        scale: f32,
    ) -> std::result::Result<(), ModelAdapterError> {
        if self.jobs.is_none() {
            return Err(ModelAdapterError::ModelNotLoaded);
        }
        if !scale.is_finite() {
            return Err(ModelAdapterError::InvalidScale { scale });
        }
        let path = path.to_path_buf();
        self.adapter_request(|reply| Job::LoadAdapter { path, scale, reply })
    }

    fn unload_lora_adapter(&mut self) -> std::result::Result<(), ModelAdapterError> {
        self.adapter_request(|reply| Job::UnloadAdapter { reply })
    }

    fn active_lora_adapter(&self) -> std::result::Result<Option<AdapterInfo>, ModelAdapterError> {
        self.adapter_request(|reply| Job::ActiveAdapter { reply })
    }

    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        reject_unsupported_params(&params)?;
        let jobs = self
            .jobs
            .as_ref()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No model loaded".to_string()))?;
        let (tx, rx) = mpsc::channel();
        jobs.send(Job::Generate {
            prompt: prompt.to_string(),
            params,
            reply: Reply::Once(tx),
        })
        .map_err(|_| RuvLLMError::Backend("lattice worker thread has exited".to_string()))?;
        rx.recv().map_err(|_| {
            RuvLLMError::Backend("lattice worker thread dropped the reply channel".to_string())
        })?
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
        // Thin adapter over `generate_stream_v2`, mirroring candle_backend.rs:1438-1455.
        let stream = self.generate_stream_v2(prompt, params)?;
        let iter = stream.filter_map(|event_result| match event_result {
            Ok(StreamEvent::Token(token)) => Some(Ok(token)),
            Ok(StreamEvent::Done { .. }) => None,
            Ok(StreamEvent::Error(msg)) => Some(Err(RuvLLMError::Generation(msg))),
            Err(e) => Some(Err(e)),
        });
        Ok(Box::new(iter))
    }

    fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream> {
        reject_unsupported_params(&params)?;
        let jobs = self
            .jobs
            .as_ref()
            .ok_or_else(|| RuvLLMError::InvalidOperation("No model loaded".to_string()))?;
        let (tx, stream) = TokenStream::channel();
        jobs.send(Job::Generate {
            prompt: prompt.to_string(),
            params,
            reply: Reply::Stream(tx),
        })
        .map_err(|_| RuvLLMError::Backend("lattice worker thread has exited".to_string()))?;
        // Unlike candle_backend.rs's `generate_stream_v2` (candle_backend.rs:1457-1600,
        // which samples exactly one token from the initial prefill logits then
        // sends `Done`), the worker drives the real autoregressive decode loop
        // (`generate_streaming_with_cancel`) and streams every decoded token.
        Ok(stream)
    }

    fn get_embeddings(&self, _text: &str) -> Result<Vec<f32>> {
        // O1 (ratified, ruvllm_design_note.md §7): lattice does not expose
        // hidden-state pooling through this seam yet. An honest error beats
        // candle's all-zero placeholder (candle_backend.rs:1602-1620).
        Err(RuvLLMError::NotImplemented(
            "LatticeBackend::get_embeddings: hidden-state pooling is not wired up yet".to_string(),
        ))
    }

    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tok.as_deref().map(|t| t as &dyn Tokenizer)
    }

    fn is_model_loaded(&self) -> bool {
        self.jobs.is_some()
    }

    fn model_info(&self) -> Option<ModelInfo> {
        self.info.clone()
    }

    fn unload_model(&mut self) {
        // Join after closing the channel so model destruction finishes before
        // the caller releases resources that protect Metal access.
        self.jobs = None;
        if let Some(worker) = self.worker.take() {
            let _ = worker.join();
        }
        self.info = None;
        self.tok = None;
        self.model_id.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn scan(stops: &[&str]) -> StopScan {
        StopScan::new(&stops.iter().map(|s| s.to_string()).collect::<Vec<_>>())
            .expect("non-empty stops")
    }

    #[test]
    fn safetensors_precision_label_follows_torch_dtype() {
        let dir = std::env::temp_dir().join(format!("lattice-dtype-test-{}", std::process::id()));
        std::fs::create_dir_all(&dir).unwrap();
        let cfg = dir.join("config.json");
        for (dtype, expected) in [
            ("bfloat16", Quantization::Bf16),
            ("float16", Quantization::F16),
            ("float32", Quantization::None),
            ("int8", Quantization::Bf16), // unmapped → fallback, not a failure
        ] {
            std::fs::write(&cfg, format!("{{\"torch_dtype\": \"{dtype}\"}}")).unwrap();
            assert_eq!(safetensors_precision_label(&cfg), expected, "dtype {dtype}");
        }
        // Missing field and unreadable path both fall back to Bf16.
        std::fs::write(&cfg, "{}").unwrap();
        assert_eq!(safetensors_precision_label(&cfg), Quantization::Bf16);
        assert_eq!(
            safetensors_precision_label(&dir.join("nope.json")),
            Quantization::Bf16
        );
        let _ = std::fs::remove_dir_all(&dir);
    }

    #[test]
    fn stop_scan_none_when_no_stops() {
        assert!(StopScan::new(&[]).is_none());
        assert!(StopScan::new(&[String::new()]).is_none());
    }

    #[test]
    fn stop_scan_cuts_at_stop_and_excludes_it() {
        let mut s = scan(&["</s>"]);
        let mut out = String::new();
        for delta in ["hello ", "world</s", ">ignored"] {
            out.push_str(&s.push(delta));
            if s.stopped {
                break;
            }
        }
        assert!(s.stopped);
        assert_eq!(out, "hello world");
    }

    #[test]
    fn stop_scan_holds_back_partial_prefix() {
        let mut s = scan(&["STOP"]);
        // hold = 3 bytes: the trailing 3 bytes always stay buffered, since
        // they could still grow into "STOP".
        assert_eq!(s.push("abST"), "a"); // "bST" retained
        assert!(!s.stopped);
        assert_eq!(s.push("x"), "b"); // "STx" retained
                                      // It never completes; finish() releases the held-back tail.
        assert_eq!(s.finish(), "STx");
        assert!(!s.stopped);
    }

    #[test]
    fn stop_scan_earliest_of_multiple_stops_wins() {
        let mut s = scan(&["<end>", "!!"]);
        let out = s.push("abc!!def<end>");
        assert!(s.stopped);
        assert_eq!(out, "abc");
    }

    #[test]
    fn stop_scan_utf8_boundary_safe() {
        let mut s = scan(&["<stop>"]); // hold = 5 bytes
        let mut out = String::new();
        // Multi-byte CJK codepoints (3 bytes each) forced through the
        // hold-back split: the boundary snap must never panic mid-codepoint.
        for delta in ["你好世界", "又一段文字"] {
            out.push_str(&s.push(delta));
        }
        out.push_str(&s.finish());
        assert!(!s.stopped);
        assert_eq!(out, "你好世界又一段文字");
    }

    #[test]
    fn nonzero_penalties_rejected_not_ignored() {
        let backend = LatticeBackend::default();
        let params = GenerateParams {
            frequency_penalty: 0.1,
            ..GenerateParams::default()
        };
        assert!(matches!(
            backend.generate("x", params),
            Err(RuvLLMError::NotImplemented(_))
        ));
        let params = GenerateParams {
            presence_penalty: 0.1,
            ..GenerateParams::default()
        };
        assert!(matches!(
            backend.generate_stream_v2("x", params),
            Err(RuvLLMError::NotImplemented(_))
        ));
    }
}
