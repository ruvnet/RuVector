//! Candle-based LLM inference backend
//!
//! This module provides a Rust-native LLM inference backend using the Candle framework
//! from HuggingFace. It supports:
//!
//! - Multiple architectures: Mistral, Llama, Phi, Qwen, Gemma
//! - Quantization: GGUF Q4/Q8 formats
//! - Metal acceleration on Apple Silicon (M1/M2/M3/M4)
//! - Memory-efficient inference with paged attention
//! - Chat templates for instruction-tuned models
//! - Streaming decode with proper UTF-8 handling
//!
//! ## Mac M4 Pro Optimizations
//!
//! This backend is optimized for Apple Silicon with:
//! - Metal Performance Shaders for matrix operations
//! - NEON SIMD for CPU fallback
//! - Memory-mapped weight loading
//! - Efficient KV cache management
//!
//! ## Chat Templates
//!
//! The backend uses RuvTokenizer for advanced chat template support:
//! - Llama 3: `<|begin_of_text|><|start_header_id|>role<|end_header_id|>`
//! - Mistral: `[INST] system\n\nuser [/INST]`
//! - Qwen/ChatML: `<|im_start|>role\ncontent<|im_end|>`
//! - Phi: `<|user|>\ncontent<|end|>`
//!
//! ## Example with Chat
//!
//! ```rust,ignore
//! use ruvllm::backends::CandleBackend;
//! use ruvllm::tokenizer::{ChatMessage, ChatTemplate};
//!
//! let mut backend = CandleBackend::new()?;
//! backend.load_model("Qwen/Qwen2.5-0.5B-Instruct", ModelConfig::default())?;
//!
//! let messages = vec![
//!     ChatMessage::system("You are a helpful assistant."),
//!     ChatMessage::user("What is Rust?"),
//! ];
//!
//! let prompt = backend.apply_chat_template(&messages)?;
//! let response = backend.generate(&prompt, GenerateParams::default())?;
//! ```

use super::{
    DType, DeviceType, GenerateParams, GeneratedToken, GenerationOutput, LlmBackend,
    ModelArchitecture, ModelConfig, ModelInfo, Quantization, SpecialTokens, StreamEvent,
    TokenStream, Tokenizer,
};
use crate::error::{Result, RuvLLMError};
use crate::sona::{SonaConfig, SonaIntegration, Trajectory};
use crate::tokenizer::{ChatMessage, ChatTemplate, RuvTokenizer};

use std::path::{Path, PathBuf};
use std::sync::mpsc;
use std::time::Instant;

#[cfg(feature = "candle")]
use candle_core::{DType as CandleDType, Device, IndexOp, Tensor};
#[cfg(feature = "candle")]
use candle_nn::VarBuilder;
#[cfg(feature = "candle")]
use candle_transformers::generation::{LogitsProcessor, Sampling};
#[cfg(feature = "candle")]
use tokenizers::Tokenizer as HfTokenizer;

/// Temperatures below this sample greedily (candle's `LogitsProcessor` threshold).
#[cfg(feature = "candle")]
const GREEDY_BELOW: f32 = 1e-7;

/// Internal model configuration
#[derive(Debug, Clone)]
struct ModelConfigInternal {
    hidden_size: usize,
    intermediate_size: usize,
    num_layers: usize,
    num_heads: usize,
    num_kv_heads: usize,
    vocab_size: usize,
    max_position_embeddings: usize,
    rope_theta: f64,
    sliding_window: Option<usize>,
    head_dim: usize,
    rms_norm_eps: f64,
}

impl Default for ModelConfigInternal {
    fn default() -> Self {
        Self {
            hidden_size: 4096,
            intermediate_size: 14336,
            num_layers: 32,
            num_heads: 32,
            num_kv_heads: 8,
            vocab_size: 32000,
            max_position_embeddings: 4096,
            rope_theta: 10000.0,
            sliding_window: None,
            head_dim: 128,
            rms_norm_eps: 1e-5,
        }
    }
}

// ============================================================================
// Candle-enabled implementation
// ============================================================================

#[cfg(feature = "candle")]
mod candle_impl {
    use super::*;
    use crate::backends::gguf_arch::{self, GgufArch, GgufSpecialIds};
    use crate::serving::FinishReason;
    use candle_core::quantized::gguf_file;
    use candle_transformers::models::{
        llama as llama_model, mistral as mistral_model, quantized_llama as qlama,
        quantized_qwen2 as qqwen2, quantized_qwen3 as qqwen3,
    };
    use std::sync::Mutex;

    /// Enum representing loaded model instances
    pub enum LoadedModelInner {
        /// Mistral model (safetensors)
        Mistral(mistral_model::Model),
        /// Llama model (safetensors) with its KV cache + the params
        /// needed to recreate the cache on `clear_kv_cache` (ADR-180
        /// iter 5: candle's Cache holds ks/vs Tensor vecs that don't
        /// reset on position-zero; we must build a fresh Cache to
        /// isolate request state).
        Llama(
            llama_model::Llama,
            llama_model::Cache,
            llama_model::Config,
            candle_core::DType,
        ),
        /// Quantized GGUF model (Llama-based architecture)
        QuantizedLlama(qlama::ModelWeights),
        /// Quantized GGUF model, `general.architecture = "qwen2"` (Qwen2, Qwen2.5)
        QuantizedQwen2(qqwen2::ModelWeights),
        /// Quantized GGUF model, `general.architecture = "qwen3"`
        QuantizedQwen3(qqwen3::ModelWeights),
    }

    /// Wrapper for loaded model state
    pub struct LoadedModel {
        /// Model inner variant (wrapped in Mutex for interior mutability)
        pub inner: Mutex<LoadedModelInner>,
        /// Model configuration
        pub config: ModelConfigInternal,
        /// Model info
        pub info: ModelInfo,
    }

    /// Candle tokenizer wrapper (legacy, kept for compatibility)
    ///
    /// For new code, prefer using `RuvTokenizer` directly via `ruvllm::tokenizer`.
    pub struct CandleTokenizer {
        pub inner: HfTokenizer,
        pub special_tokens: SpecialTokens,
    }

    impl Tokenizer for CandleTokenizer {
        fn encode(&self, text: &str) -> Result<Vec<u32>> {
            let encoding = self
                .inner
                .encode(text, false)
                .map_err(|e| RuvLLMError::Tokenization(format!("Tokenization failed: {}", e)))?;
            Ok(encoding.get_ids().to_vec())
        }

        fn decode(&self, tokens: &[u32]) -> Result<String> {
            self.inner
                .decode(tokens, true)
                .map_err(|e| RuvLLMError::Tokenization(format!("Decoding failed: {}", e)))
        }

        fn vocab_size(&self) -> usize {
            self.inner.get_vocab_size(true)
        }

        fn special_tokens(&self) -> SpecialTokens {
            self.special_tokens.clone()
        }
    }

    /// Candle-based inference backend
    ///
    /// Provides high-performance LLM inference using the Candle framework.
    /// Optimized for Apple Silicon with Metal acceleration.
    ///
    /// ## Tokenizer Support
    ///
    /// The backend maintains two tokenizer references:
    /// - `tokenizer`: Legacy `CandleTokenizer` for trait compatibility
    /// - `ruv_tokenizer`: Enhanced `RuvTokenizer` with chat templates and streaming decode
    ///
    /// For new features like chat templates, use the `ruv_tokenizer()` method.
    pub struct CandleBackend {
        /// Current device
        pub device: Device,
        /// Loaded model
        pub model: Option<LoadedModel>,
        /// Legacy tokenizer (for trait compatibility)
        pub tokenizer: Option<CandleTokenizer>,
        /// Enhanced tokenizer with chat templates and streaming decode
        pub ruv_tokenizer: Option<RuvTokenizer>,
        /// Cache directory for models
        pub cache_dir: PathBuf,
        /// Configuration
        pub config: Option<ModelConfig>,
        /// Token ids that end generation (EOS and end-of-turn markers)
        stop_token_ids: Vec<u32>,
        /// Serializes generations: the KV cache and `current_pos` are shared
        generation_lock: Mutex<()>,
        /// Model ID for chat template detection
        model_id: String,
        /// Current sequence position for KV cache
        current_pos: Mutex<usize>,
        /// SONA self-learning integration
        sona: Option<SonaIntegration>,
    }

    impl Default for CandleBackend {
        fn default() -> Self {
            Self {
                device: Device::Cpu,
                model: None,
                tokenizer: None,
                ruv_tokenizer: None,
                cache_dir: get_cache_dir(),
                config: None,
                stop_token_ids: Vec::new(),
                generation_lock: Mutex::new(()),
                model_id: String::new(),
                current_pos: Mutex::new(0),
                sona: Some(SonaIntegration::new(SonaConfig::default())),
            }
        }
    }

    impl CandleBackend {
        /// Create a new Candle backend
        pub fn new() -> Result<Self> {
            let device = Self::select_device(DeviceType::default())?;

            let cache_dir = get_cache_dir();
            std::fs::create_dir_all(&cache_dir).map_err(|e| {
                RuvLLMError::Storage(format!("Failed to create cache directory: {}", e))
            })?;

            Ok(Self {
                device,
                model: None,
                tokenizer: None,
                ruv_tokenizer: None,
                cache_dir,
                config: None,
                stop_token_ids: Vec::new(),
                generation_lock: Mutex::new(()),
                model_id: String::new(),
                current_pos: Mutex::new(0),
                sona: Some(SonaIntegration::new(SonaConfig::default())),
            })
        }

        /// Get SONA learning stats
        pub fn sona_stats(&self) -> Option<crate::sona::SonaStats> {
            self.sona.as_ref().map(|s| s.stats())
        }

        /// Enable/disable SONA learning
        pub fn set_sona_enabled(&mut self, enabled: bool) {
            if enabled && self.sona.is_none() {
                self.sona = Some(SonaIntegration::new(SonaConfig::default()));
            } else if !enabled {
                self.sona = None;
            }
        }

        /// Create a simple embedding from text (placeholder - should use real embeddings)
        fn simple_embedding(text: &str, dim: usize) -> Vec<f32> {
            let mut embedding = vec![0.0f32; dim];
            let bytes = text.as_bytes();
            for (i, &b) in bytes.iter().enumerate() {
                embedding[i % dim] += (b as f32) / 255.0;
            }
            // Normalize
            let norm: f32 = embedding.iter().map(|x| x * x).sum::<f32>().sqrt();
            if norm > 0.0 {
                for x in &mut embedding {
                    *x /= norm;
                }
            }
            embedding
        }

        /// Get the enhanced RuvTokenizer with chat template support
        ///
        /// Returns `None` if no tokenizer is loaded.
        pub fn ruv_tokenizer(&self) -> Option<&RuvTokenizer> {
            self.ruv_tokenizer.as_ref()
        }

        /// Get mutable reference to RuvTokenizer (needed for streaming decode)
        pub fn ruv_tokenizer_mut(&mut self) -> Option<&mut RuvTokenizer> {
            self.ruv_tokenizer.as_mut()
        }

        /// Apply chat template to messages
        ///
        /// Uses the model's detected chat template format to properly
        /// format multi-turn conversations for instruction-tuned models.
        pub fn apply_chat_template(&self, messages: &[ChatMessage]) -> Result<String> {
            let tokenizer = self
                .ruv_tokenizer
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer loaded".to_string()))?;

            tokenizer.apply_chat_template(messages)
        }

        /// Get the current chat template
        pub fn chat_template(&self) -> Option<&ChatTemplate> {
            self.ruv_tokenizer.as_ref().and_then(|t| t.chat_template())
        }

        /// Set a custom chat template
        pub fn set_chat_template(&mut self, template: ChatTemplate) {
            if let Some(tokenizer) = self.ruv_tokenizer.take() {
                self.ruv_tokenizer = Some(tokenizer.with_chat_template(template));
            }
        }

        /// Decode a single token for streaming output
        pub fn decode_stream(&mut self, token: u32) -> Result<Option<String>> {
            let tokenizer = self
                .ruv_tokenizer
                .as_mut()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer loaded".to_string()))?;

            tokenizer.decode_stream(token)
        }

        /// Flush any remaining bytes in the streaming buffer
        pub fn flush_stream(&mut self) -> Result<Option<String>> {
            let tokenizer = self
                .ruv_tokenizer
                .as_mut()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer loaded".to_string()))?;

            tokenizer.flush_stream()
        }

        /// Reset the streaming decode buffer
        pub fn reset_stream(&mut self) {
            if let Some(tokenizer) = self.ruv_tokenizer.as_mut() {
                tokenizer.reset_stream();
            }
        }

        /// Get the model ID
        pub fn model_id(&self) -> &str {
            &self.model_id
        }

        /// Create backend with specific device
        pub fn with_device(device_type: DeviceType) -> Result<Self> {
            let device = Self::select_device(device_type)?;
            Ok(Self {
                device,
                ..Default::default()
            })
        }

        /// Select device based on type
        pub fn select_device(device_type: DeviceType) -> Result<Device> {
            match device_type {
                DeviceType::Cpu => Ok(Device::Cpu),
                DeviceType::Metal => {
                    #[cfg(target_os = "macos")]
                    {
                        Device::new_metal(0).map_err(|e| {
                            RuvLLMError::Backend(format!(
                                "Failed to initialize Metal device: {}",
                                e
                            ))
                        })
                    }
                    #[cfg(not(target_os = "macos"))]
                    {
                        tracing::warn!("Metal requested but not available, falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
                DeviceType::Cuda(device_id) => {
                    #[cfg(feature = "cuda")]
                    {
                        Device::new_cuda(device_id).map_err(|e| {
                            RuvLLMError::Backend(format!("Failed to initialize CUDA device: {}", e))
                        })
                    }
                    #[cfg(not(feature = "cuda"))]
                    {
                        let _ = device_id;
                        tracing::warn!("CUDA requested but not available, falling back to CPU");
                        Ok(Device::Cpu)
                    }
                }
            }
        }

        /// Set cache directory for model downloads
        pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
            self.cache_dir = cache_dir.into();
            self
        }

        /// Convert our DType to Candle DType
        fn to_candle_dtype(dtype: DType) -> CandleDType {
            match dtype {
                DType::F32 => CandleDType::F32,
                DType::F16 => CandleDType::F16,
                DType::Bf16 => CandleDType::BF16,
            }
        }

        /// Load model from HuggingFace Hub.
        ///
        /// ADR-179 iter 8: gated behind `hub-download` feature. The sync
        /// `hf_hub::api::sync` API requires the `ureq` feature, which
        /// in turn pulls `native-tls` → `openssl-sys`. That doesn't
        /// cross-build to aarch64 cleanly. For Pi 5 deploys the worker
        /// rsyncs models out-of-band and uses the local-path branch of
        /// `load_model` only.
        #[cfg(feature = "hub-download")]
        pub fn load_from_hub(&mut self, model_id: &str, config: &ModelConfig) -> Result<()> {
            use hf_hub::{api::sync::Api, Repo, RepoType};

            tracing::info!("Loading model from HuggingFace Hub: {}", model_id);

            let api = Api::new().map_err(|e| {
                RuvLLMError::Storage(format!("Failed to initialize HuggingFace API: {}", e))
            })?;

            let repo = api.repo(Repo::new(model_id.to_string(), RepoType::Model));

            // Store model ID for chat template detection
            self.model_id = model_id.to_string();

            // Download tokenizer
            let tokenizer_path = repo.get("tokenizer.json").map_err(|e| {
                RuvLLMError::NotFound(format!("Tokenizer not found for {}: {}", model_id, e))
            })?;

            self.load_tokenizer(&tokenizer_path)?;

            // Also load the enhanced RuvTokenizer with chat template support
            let ruv_tokenizer = RuvTokenizer::from_file(&tokenizer_path)?;
            let chat_template = gguf_arch::template_from_name(model_id);
            self.ruv_tokenizer = Some(ruv_tokenizer.with_chat_template(chat_template));

            // Try to download GGUF file based on quantization
            let gguf_filenames = match config.quantization {
                Some(Quantization::Q4K) => vec![
                    "model-q4_k_m.gguf",
                    "model.Q4_K_M.gguf",
                    "ggml-model-q4_k_m.gguf",
                ],
                Some(Quantization::Q4) => {
                    vec!["model-q4_0.gguf", "model.Q4_0.gguf", "ggml-model-q4_0.gguf"]
                }
                Some(Quantization::Q8) => {
                    vec!["model-q8_0.gguf", "model.Q8_0.gguf", "ggml-model-q8_0.gguf"]
                }
                _ => vec![],
            };

            for filename in &gguf_filenames {
                if let Ok(gguf_path) = repo.get(filename) {
                    tracing::info!("Found GGUF file: {}", filename);
                    return self.load_gguf(&gguf_path, config);
                }
            }

            // Fall back to safetensors
            tracing::info!("No GGUF file found, loading safetensors");

            let weights_files = self.get_safetensors_files(&repo)?;
            let config_path = repo.get("config.json").map_err(|e| {
                RuvLLMError::NotFound(format!("Config not found for {}: {}", model_id, e))
            })?;

            self.load_safetensors(&weights_files, &config_path, config)
        }

        /// Get list of safetensors files from repo.
        /// ADR-179 iter 8: hub-download feature only (see load_from_hub).
        #[cfg(feature = "hub-download")]
        fn get_safetensors_files(&self, repo: &hf_hub::api::sync::ApiRepo) -> Result<Vec<PathBuf>> {
            // Try single file first
            if let Ok(path) = repo.get("model.safetensors") {
                return Ok(vec![path]);
            }

            // Try sharded files - look for the index file first
            if let Ok(index_path) = repo.get("model.safetensors.index.json") {
                let index_str = std::fs::read_to_string(&index_path).map_err(|e| {
                    RuvLLMError::Storage(format!("Failed to read safetensors index: {}", e))
                })?;
                let index: serde_json::Value = serde_json::from_str(&index_str)?;

                if let Some(weight_map) = index.get("weight_map").and_then(|w| w.as_object()) {
                    let mut shard_files: std::collections::HashSet<String> =
                        std::collections::HashSet::new();
                    for filename in weight_map.values() {
                        if let Some(f) = filename.as_str() {
                            shard_files.insert(f.to_string());
                        }
                    }

                    let mut files = Vec::new();
                    for shard in shard_files {
                        if let Ok(path) = repo.get(&shard) {
                            files.push(path);
                        }
                    }

                    if !files.is_empty() {
                        files.sort();
                        return Ok(files);
                    }
                }
            }

            Err(RuvLLMError::NotFound(
                "No safetensors files found. Try using a quantized GGUF model.".to_string(),
            ))
        }

        /// Load tokenizer from path
        pub fn load_tokenizer(&mut self, path: &Path) -> Result<()> {
            tracing::info!("Loading tokenizer from: {:?}", path);

            let tokenizer = HfTokenizer::from_file(path)
                .map_err(|e| RuvLLMError::Storage(format!("Failed to load tokenizer: {}", e)))?;
            self.install_tokenizer(tokenizer);
            Ok(())
        }

        /// Build the tokenizer from the vocabulary embedded in a GGUF file
        /// (`tokenizer.ggml.*`), when no `tokenizer.json` was loaded. Returns
        /// whether one was installed; an unsupported tokenizer model is logged,
        /// not fatal, so an explicit `load_tokenizer` afterwards still works.
        fn load_embedded_tokenizer(&mut self, content: &gguf_file::Content) -> bool {
            let md = &content.metadata;
            let Some(model) = md
                .get("tokenizer.ggml.model")
                .and_then(|v| v.to_string().ok())
                .cloned()
            else {
                return false;
            };
            // All-or-nothing: dropping one malformed entry would shift every later
            // token id, so any non-string entry yields an empty list (rejected below).
            let strings = |key: &str| -> Vec<String> {
                md.get(key)
                    .and_then(|v| v.to_vec().ok())
                    .and_then(|vs| {
                        vs.iter()
                            .map(|v| v.to_string().ok().cloned())
                            .collect::<Option<Vec<_>>>()
                    })
                    .unwrap_or_default()
            };
            let tokens = strings("tokenizer.ggml.tokens");
            let merges = strings("tokenizer.ggml.merges");
            let pre = md
                .get("tokenizer.ggml.pre")
                .and_then(|v| v.to_string().ok())
                .map(String::as_str);
            let types: Option<Vec<i32>> = md
                .get("tokenizer.ggml.token_type")
                .and_then(|v| v.to_vec().ok())
                .and_then(|vs| {
                    vs.iter()
                        .map(|v| v.to_i32().ok())
                        .collect::<Option<Vec<_>>>()
                })
                .filter(|t| t.len() == tokens.len());
            match crate::backends::gguf_tokenizer::from_gguf_parts_with_pre(
                &model,
                pre,
                &tokens,
                &merges,
                types.as_deref(),
            ) {
                Ok(tokenizer) => {
                    tracing::info!(
                        "Built the '{}' tokenizer embedded in the GGUF ({} tokens)",
                        model,
                        tokens.len()
                    );
                    self.install_tokenizer(tokenizer);
                    true
                }
                Err(e) => {
                    tracing::warn!("{}", e);
                    false
                }
            }
        }

        fn install_tokenizer(&mut self, tokenizer: HfTokenizer) {
            // Detect special tokens
            let special_tokens = SpecialTokens {
                bos_token_id: tokenizer
                    .token_to_id("<s>")
                    .or_else(|| tokenizer.token_to_id("<|begin_of_text|>"))
                    .or_else(|| tokenizer.token_to_id("<|startoftext|>")),
                eos_token_id: tokenizer
                    .token_to_id("</s>")
                    .or_else(|| tokenizer.token_to_id("<|end_of_text|>"))
                    .or_else(|| tokenizer.token_to_id("<|endoftext|>"))
                    .or_else(|| tokenizer.token_to_id("<|eot_id|>")),
                pad_token_id: tokenizer
                    .token_to_id("<pad>")
                    .or_else(|| tokenizer.token_to_id("<|pad|>"))
                    .or_else(|| tokenizer.token_to_id("[PAD]")),
                unk_token_id: tokenizer
                    .token_to_id("<unk>")
                    .or_else(|| tokenizer.token_to_id("[UNK]")),
            };

            tracing::debug!(
                "Special tokens: bos={:?}, eos={:?}",
                special_tokens.bos_token_id,
                special_tokens.eos_token_id
            );

            // Extend, not replace: a tokenizer loaded after `load_gguf` must
            // keep the end-of-turn ids the file declared. `unload_model`
            // clears the list, so ids never carry over between models.
            for id in gguf_arch::vocab_stop_ids(&tokenizer) {
                if !self.stop_token_ids.contains(&id) {
                    self.stop_token_ids.push(id);
                }
            }
            self.tokenizer = Some(CandleTokenizer {
                inner: tokenizer,
                special_tokens,
            });
        }

        /// Apply what a GGUF says about its tokenizer: the declared EOS /
        /// end-of-turn ids (authoritative over the vocabulary guess; Qwen
        /// instruct ends turns with `<|im_end|>`, not `<|endoftext|>`), and a
        /// chat template so `apply_chat_template` works for a bare `.gguf`.
        ///
        /// The file's own template wins over a guess from the model name; when
        /// it gives none, an earlier guess is kept, else one is made from
        /// `file_name`. Fails for a Qwen-architecture file that is not ChatML.
        fn apply_gguf_tokenizer_metadata(
            &mut self,
            arch: GgufArch,
            md: &std::collections::HashMap<String, gguf_file::Value>,
            file_name: &str,
        ) -> Result<()> {
            let Some(tokenizer) = self.tokenizer.as_mut() else {
                return Ok(());
            };
            let ids = GgufSpecialIds::from_metadata(md, tokenizer.inner.get_vocab_size(true));
            if ids.eos.is_some() {
                tokenizer.special_tokens.eos_token_id = ids.eos;
            }
            if ids.bos.is_some() {
                tokenizer.special_tokens.bos_token_id = ids.bos;
            }
            for id in ids.stop {
                if !self.stop_token_ids.contains(&id) {
                    self.stop_token_ids.push(id);
                }
            }

            let vocab = &tokenizer.inner;
            let has_chatml_tokens = vocab.token_to_id("<|im_start|>").is_some()
                && vocab.token_to_id("<|im_end|>").is_some();
            let from_file = arch.chat_template(md, has_chatml_tokens)?;
            let ruv = match (self.ruv_tokenizer.take(), from_file) {
                (Some(ruv), Some(template)) => ruv.with_chat_template(template),
                (Some(ruv), None) if ruv.chat_template().is_some() => ruv,
                (ruv, from_file) => ruv
                    .unwrap_or_else(|| {
                        RuvTokenizer::from_hf_tokenizer(vocab.clone(), Some(&self.model_id))
                    })
                    .with_chat_template(
                        from_file.unwrap_or_else(|| gguf_arch::template_from_name(file_name)),
                    ),
            };
            self.ruv_tokenizer = Some(match ids.eos {
                Some(eos) => ruv.with_eos_token_id(eos),
                None => ruv,
            });
            Ok(())
        }

        /// Load GGUF quantized model
        pub fn load_gguf(&mut self, path: &Path, config: &ModelConfig) -> Result<()> {
            tracing::info!("Loading GGUF model from: {:?}", path);

            let mut file = std::fs::File::open(path)
                .map_err(|e| RuvLLMError::Storage(format!("Failed to open GGUF file: {}", e)))?;

            // Read GGUF content
            let mut gguf_content = gguf_file::Content::read(&mut file)
                .map_err(|e| RuvLLMError::Storage(format!("Failed to read GGUF file: {}", e)))?;

            // `general.architecture` picks the weight loader; anything without
            // one (qwen3moe, qwen35, phi3, gemma*, ...) is rejected by name.
            let arch = GgufArch::from_metadata(&gguf_content.metadata)?;
            let quantization =
                gguf_arch::file_quantization(&gguf_content.metadata, config.quantization);

            // A bare .gguf carries its vocabulary; use it when no tokenizer.json
            // was found next to the file (load_model tries that first).
            if self.tokenizer.is_none() {
                self.load_embedded_tokenizer(&gguf_content);
            }
            // Before the weights: an unusable chat format fails the load cheaply.
            let file_name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            self.apply_gguf_tokenizer_metadata(arch, &gguf_content.metadata, file_name)?;

            // Extract config from GGUF metadata
            let hidden_size = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.embedding_length",
                        "mistral.embedding_length",
                        "phi.embedding_length",
                        "qwen2.embedding_length",
                        "qwen3.embedding_length",
                        "gemma.embedding_length",
                        "gemma3.embedding_length",
                    ],
                )
                .unwrap_or(4096) as usize;

            let num_layers = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.block_count",
                        "mistral.block_count",
                        "phi.block_count",
                        "qwen2.block_count",
                        "qwen3.block_count",
                        "gemma.block_count",
                        "gemma3.block_count",
                    ],
                )
                .unwrap_or(32) as usize;

            let num_heads = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.attention.head_count",
                        "mistral.attention.head_count",
                        "phi.attention.head_count",
                        "qwen2.attention.head_count",
                        "qwen3.attention.head_count",
                        "gemma.attention.head_count",
                        "gemma3.attention.head_count",
                    ],
                )
                .unwrap_or(32) as usize;

            let num_kv_heads = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.attention.head_count_kv",
                        "mistral.attention.head_count_kv",
                        "phi.attention.head_count_kv",
                        "qwen2.attention.head_count_kv",
                        "qwen3.attention.head_count_kv",
                        "gemma.attention.head_count_kv",
                        "gemma3.attention.head_count_kv",
                    ],
                )
                .unwrap_or(num_heads as u32) as usize;

            let vocab_size = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.vocab_size",
                        "mistral.vocab_size",
                        "phi.vocab_size",
                        "qwen2.vocab_size",
                        "qwen3.vocab_size",
                        "gemma.vocab_size",
                        "gemma3.vocab_size",
                        "tokenizer.ggml.tokens_size",
                    ],
                )
                .map(|v| v as usize)
                .or_else(|| {
                    let tokens = gguf_content.metadata.get("tokenizer.ggml.tokens")?;
                    tokens.to_vec().ok().map(Vec::len)
                })
                .unwrap_or(32000);

            let intermediate_size = self
                .get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.feed_forward_length",
                        "mistral.feed_forward_length",
                        "phi.feed_forward_length",
                        "qwen2.feed_forward_length",
                        "qwen3.feed_forward_length",
                        "gemma.feed_forward_length",
                        "gemma3.feed_forward_length",
                    ],
                )
                .unwrap_or(14336) as usize;

            let rope_theta = self
                .get_gguf_f32(
                    &gguf_content,
                    &[
                        "llama.rope.freq_base",
                        "mistral.rope.freq_base",
                        "phi.rope.freq_base",
                        "qwen2.rope.freq_base",
                        "qwen3.rope.freq_base",
                        "gemma.rope.freq_base",
                        "gemma3.rope.freq_base",
                    ],
                )
                .unwrap_or(10000.0) as f64;

            let context_length =
                self.get_gguf_u32(
                    &gguf_content,
                    &[
                        "llama.context_length",
                        "mistral.context_length",
                        "phi.context_length",
                        "qwen2.context_length",
                        "qwen3.context_length",
                        "gemma.context_length",
                        "gemma3.context_length",
                    ],
                )
                .unwrap_or(config.max_sequence_length as u32) as usize;

            let rms_norm_eps = self
                .get_gguf_f32(
                    &gguf_content,
                    &[
                        "llama.attention.layer_norm_rms_epsilon",
                        "mistral.attention.layer_norm_rms_epsilon",
                        "qwen2.attention.layer_norm_rms_epsilon",
                        "qwen3.attention.layer_norm_rms_epsilon",
                        "gemma.attention.layer_norm_rms_epsilon",
                        "gemma3.attention.layer_norm_rms_epsilon",
                    ],
                )
                .unwrap_or(1e-5) as f64;

            // Qwen3 declares a head dim that is not hidden / heads.
            let head_dim = self
                .get_gguf_u32(
                    &gguf_content,
                    &["qwen3.attention.key_length", "llama.attention.key_length"],
                )
                .map_or(hidden_size / num_heads, |v| v as usize);

            let model_config = ModelConfigInternal {
                hidden_size,
                intermediate_size,
                num_layers,
                num_heads,
                num_kv_heads,
                vocab_size,
                max_position_embeddings: context_length.min(config.max_sequence_length),
                rope_theta,
                sliding_window: config.sliding_window,
                head_dim,
                rms_norm_eps,
            };

            tracing::info!(
                "Model config: hidden={}, layers={}, heads={}, kv_heads={}, vocab={}",
                hidden_size,
                num_layers,
                num_heads,
                num_kv_heads,
                vocab_size
            );

            // Load the quantized model weights with the architecture's loader
            let weights_err = |e: candle_core::Error| {
                RuvLLMError::Model(format!("Failed to load {:?} GGUF weights: {}", arch, e))
            };
            let inner = match arch {
                GgufArch::Llama => LoadedModelInner::QuantizedLlama(
                    qlama::ModelWeights::from_gguf(gguf_content, &mut file, &self.device)
                        .map_err(weights_err)?,
                ),
                GgufArch::Qwen2 => LoadedModelInner::QuantizedQwen2(
                    qqwen2::ModelWeights::from_gguf(gguf_content, &mut file, &self.device)
                        .map_err(weights_err)?,
                ),
                GgufArch::Qwen3 => {
                    // quantized_qwen3 builds its RoPE table and mask in
                    // `general.dtype`, F16 when absent (llama.cpp never writes
                    // it); F16 is exact for integer positions only up to 2048
                    // and coarse for the angles. Ask for F32.
                    gguf_content
                        .metadata
                        .entry("general.dtype".to_string())
                        .or_insert(gguf_file::Value::U32(0));
                    LoadedModelInner::QuantizedQwen3(
                        qqwen3::ModelWeights::from_gguf(gguf_content, &mut file, &self.device)
                            .map_err(weights_err)?,
                    )
                }
            };

            let memory_usage = estimate_gguf_memory(path)?;

            let info = ModelInfo {
                name: path
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                architecture: arch.model_architecture(config.architecture),
                num_parameters: estimate_parameters(hidden_size, num_layers, vocab_size),
                vocab_size,
                hidden_size,
                num_layers,
                max_context_length: model_config.max_position_embeddings,
                quantization,
                memory_usage,
            };

            self.model = Some(LoadedModel {
                inner: Mutex::new(inner),
                config: model_config,
                info,
            });

            self.config = Some(config.clone());
            *self.current_pos.lock().expect("current_pos mutex poisoned") = 0;

            tracing::info!("GGUF model loaded successfully");
            Ok(())
        }

        /// Get u32 value from GGUF metadata with fallback keys
        fn get_gguf_u32(&self, content: &gguf_file::Content, keys: &[&str]) -> Option<u32> {
            for key in keys {
                if let Some(value) = content.metadata.get(*key) {
                    if let Ok(v) = value.to_u32() {
                        return Some(v);
                    }
                }
            }
            None
        }

        /// Get f32 value from GGUF metadata with fallback keys
        fn get_gguf_f32(&self, content: &gguf_file::Content, keys: &[&str]) -> Option<f32> {
            for key in keys {
                if let Some(value) = content.metadata.get(*key) {
                    if let Ok(v) = value.to_f32() {
                        return Some(v);
                    }
                }
            }
            None
        }

        /// Load model from safetensors files
        pub fn load_safetensors(
            &mut self,
            weights_files: &[PathBuf],
            config_path: &Path,
            config: &ModelConfig,
        ) -> Result<()> {
            tracing::info!("Loading safetensors from {} files", weights_files.len());

            // Read model config JSON
            let config_str = std::fs::read_to_string(config_path)
                .map_err(|e| RuvLLMError::Storage(format!("Failed to read config: {}", e)))?;

            let model_json: serde_json::Value = serde_json::from_str(&config_str)?;

            // Extract configuration
            let hidden_size = model_json["hidden_size"].as_u64().unwrap_or(4096) as usize;
            let num_layers = model_json["num_hidden_layers"].as_u64().unwrap_or(32) as usize;
            let num_heads = model_json["num_attention_heads"].as_u64().unwrap_or(32) as usize;
            let num_kv_heads = model_json["num_key_value_heads"]
                .as_u64()
                .unwrap_or(num_heads as u64) as usize;
            let vocab_size = model_json["vocab_size"].as_u64().unwrap_or(32000) as usize;
            let intermediate_size =
                model_json["intermediate_size"].as_u64().unwrap_or(14336) as usize;
            let rope_theta = model_json["rope_theta"].as_f64().unwrap_or(10000.0);
            let rms_norm_eps = model_json["rms_norm_eps"].as_f64().unwrap_or(1e-5);
            let head_dim = hidden_size / num_heads;

            let model_config = ModelConfigInternal {
                hidden_size,
                intermediate_size,
                num_layers,
                num_heads,
                num_kv_heads,
                vocab_size,
                max_position_embeddings: config.max_sequence_length,
                rope_theta,
                sliding_window: config.sliding_window,
                head_dim,
                rms_norm_eps,
            };

            // Determine dtype for loading
            let dtype = Self::to_candle_dtype(config.dtype);

            // Create VarBuilder from safetensors files
            let vb = unsafe {
                VarBuilder::from_mmaped_safetensors(weights_files, dtype, &self.device)
                    .map_err(|e| RuvLLMError::Model(format!("Failed to load safetensors: {}", e)))?
            };

            // Load model based on architecture
            let inner = match config.architecture {
                ModelArchitecture::Mistral => {
                    let mistral_config = mistral_model::Config {
                        vocab_size,
                        hidden_size,
                        intermediate_size,
                        num_hidden_layers: num_layers,
                        num_attention_heads: num_heads,
                        num_key_value_heads: num_kv_heads,
                        hidden_act: candle_nn::Activation::Silu,
                        max_position_embeddings: config.max_sequence_length,
                        rms_norm_eps,
                        rope_theta,
                        sliding_window: config.sliding_window,
                        use_flash_attn: config.use_flash_attention,
                        head_dim: Some(head_dim),
                    };

                    let model = mistral_model::Model::new(&mistral_config, vb).map_err(|e| {
                        RuvLLMError::Model(format!("Failed to create Mistral model: {}", e))
                    })?;

                    LoadedModelInner::Mistral(model)
                }
                ModelArchitecture::Llama => {
                    // Llama 3.2 1B/3B tie the output projection to the input
                    // embeddings (no lm_head.weight tensor in the checkpoint);
                    // detect from tensor presence instead of hardcoding.
                    let tie_word_embeddings = !vb.contains_tensor("lm_head.weight");
                    let llama_config = llama_model::Config {
                        hidden_size,
                        intermediate_size,
                        vocab_size,
                        num_hidden_layers: num_layers,
                        num_attention_heads: num_heads,
                        num_key_value_heads: num_kv_heads,
                        rms_norm_eps,
                        rope_theta: rope_theta as f32,
                        use_flash_attn: config.use_flash_attention,
                        bos_token_id: None,
                        eos_token_id: None,
                        rope_scaling: None,
                        max_position_embeddings: config.max_sequence_length,
                        tie_word_embeddings,
                    };

                    let model = llama_model::Llama::load(vb, &llama_config).map_err(|e| {
                        RuvLLMError::Model(format!("Failed to create Llama model: {}", e))
                    })?;

                    // Create KV cache for the Llama model
                    let cache = llama_model::Cache::new(true, dtype, &llama_config, &self.device)
                        .map_err(|e| {
                        RuvLLMError::Model(format!("Failed to create Llama cache: {}", e))
                    })?;

                    // ADR-180 iter 5: store config + dtype alongside so
                    // clear_kv_cache() can rebuild a fresh Cache for
                    // each request — request isolation requires this.
                    LoadedModelInner::Llama(model, cache, llama_config, dtype)
                }
                _ => {
                    return Err(RuvLLMError::Config(format!(
                        "Architecture {:?} not yet supported for safetensors loading",
                        config.architecture
                    )));
                }
            };

            let memory_usage: usize = weights_files
                .iter()
                .filter_map(|p| std::fs::metadata(p).ok())
                .map(|m| m.len() as usize)
                .sum();

            let info = ModelInfo {
                name: weights_files
                    .first()
                    .and_then(|p| p.parent())
                    .and_then(|p| p.file_name())
                    .and_then(|s| s.to_str())
                    .unwrap_or("unknown")
                    .to_string(),
                architecture: config.architecture,
                num_parameters: estimate_parameters(hidden_size, num_layers, vocab_size),
                vocab_size,
                hidden_size,
                num_layers,
                max_context_length: config.max_sequence_length,
                quantization: config.quantization,
                memory_usage,
            };

            self.model = Some(LoadedModel {
                inner: Mutex::new(inner),
                config: model_config,
                info,
            });

            self.config = Some(config.clone());
            *self.current_pos.lock().expect("current_pos mutex poisoned") = 0;

            tracing::info!("Safetensors model loaded successfully");
            Ok(())
        }

        /// Forward pass through the model
        fn forward(&self, input_ids: &Tensor, seq_len: usize) -> Result<Tensor> {
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No model loaded".to_string()))?;

            let mut pos = self.current_pos.lock().expect("current_pos mutex poisoned");
            let current_pos = *pos;

            let mut inner = model.inner.lock().map_err(|e| {
                RuvLLMError::Backend(format!("Failed to acquire model lock: {}", e))
            })?;

            let logits = match &mut *inner {
                LoadedModelInner::QuantizedLlama(m) => m
                    .forward(input_ids, current_pos)
                    .map_err(|e| RuvLLMError::Generation(format!("Forward pass failed: {}", e)))?,
                LoadedModelInner::QuantizedQwen2(m) => m
                    .forward(input_ids, current_pos)
                    .map_err(|e| RuvLLMError::Generation(format!("Forward pass failed: {}", e)))?,
                LoadedModelInner::QuantizedQwen3(m) => m
                    .forward(input_ids, current_pos)
                    .map_err(|e| RuvLLMError::Generation(format!("Forward pass failed: {}", e)))?,
                LoadedModelInner::Mistral(m) => m
                    .forward(input_ids, current_pos)
                    .map_err(|e| RuvLLMError::Generation(format!("Forward pass failed: {}", e)))?,
                LoadedModelInner::Llama(m, cache, _, _) => m
                    .forward(input_ids, current_pos, cache)
                    .map_err(|e| RuvLLMError::Generation(format!("Forward pass failed: {}", e)))?,
            };

            *pos += seq_len;
            Ok(logits)
        }

        /// Clear the KV cache and reset position
        ///
        /// Mistral and Qwen3 expose `clear_kv_cache()`; Llama gets a fresh
        /// cache. Quantized Llama and Qwen2 drop their cache on the next
        /// forward at position 0, so resetting the position is enough.
        fn clear_kv_cache(&self) {
            if let Some(model) = &self.model {
                if let Ok(mut inner) = model.inner.lock() {
                    match &mut *inner {
                        LoadedModelInner::QuantizedLlama(_m) => {
                            // quantized_llama::ModelWeights doesn't expose clear_kv_cache
                            // The cache is managed internally; resetting position is sufficient
                        }
                        LoadedModelInner::QuantizedQwen2(_m) => {
                            // Same as quantized_llama: replaced at index_pos == 0
                        }
                        LoadedModelInner::QuantizedQwen3(m) => {
                            // ConcatKvCache appends regardless of offset: must reset
                            m.clear_kv_cache();
                        }
                        LoadedModelInner::Mistral(m) => {
                            m.clear_kv_cache();
                        }
                        LoadedModelInner::Llama(_m, cache_slot, cfg, dtype) => {
                            // ADR-180 iter 5: candle's external Cache is
                            // NOT auto-reset on position-zero — its
                            // ks/vs Tensor vecs accumulate across calls
                            // and break the next request's forward pass
                            // ("cannot broadcast [N,N] to [1,H,N,X]" with
                            // X = stale total seq len). Build a fresh
                            // Cache and replace the held one so each
                            // generate() starts clean.
                            match llama_model::Cache::new(true, *dtype, cfg, &self.device) {
                                Ok(fresh) => {
                                    *cache_slot = fresh;
                                }
                                Err(e) => {
                                    tracing::warn!(
                                        error = ?e,
                                        "clear_kv_cache: failed to rebuild Llama Cache; \
                                         next generate() may panic"
                                    );
                                }
                            }
                        }
                    }
                }
            }
            if let Ok(mut pos) = self.current_pos.lock() {
                *pos = 0;
            }
        }

        /// The sampler for one generation. One per request, so a fixed seed
        /// seeds one random sequence rather than reusing its first draw for
        /// every token. Temperature is applied to the logits in
        /// `sample_token` (before top-k/top-p), so the sampler uses 1.0.
        pub(crate) fn logits_processor(params: &GenerateParams) -> LogitsProcessor {
            let seed = params.seed.unwrap_or_else(|| {
                use std::time::{SystemTime, UNIX_EPOCH};
                SystemTime::now()
                    .duration_since(UNIX_EPOCH)
                    .map(|d| d.as_nanos() as u64)
                    .unwrap_or(42)
            });
            let sampling = if params.temperature < GREEDY_BELOW {
                Sampling::ArgMax
            } else {
                Sampling::All { temperature: 1.0 }
            };
            LogitsProcessor::from_sampling(seed, sampling)
        }

        /// Sample next token from logits
        fn sample_token(
            &self,
            logits: &Tensor,
            params: &GenerateParams,
            generated_tokens: &[u32],
            sampler: &mut LogitsProcessor,
        ) -> Result<u32> {
            // Get logits shape and squeeze batch dimension if needed
            let logits = if logits.dims().len() == 3 {
                logits.squeeze(0).map_err(|e| {
                    RuvLLMError::Generation(format!("Failed to squeeze logits: {}", e))
                })?
            } else {
                logits.clone()
            };

            // Get logits for the last position
            let last_logits = if logits.dims().len() == 2 {
                let seq_len = logits.dim(0).map_err(|e| {
                    RuvLLMError::Generation(format!("Failed to get seq_len: {}", e))
                })?;
                logits.i(seq_len - 1).map_err(|e| {
                    RuvLLMError::Generation(format!("Failed to get last logits: {}", e))
                })?
            } else {
                logits
            };

            // Convert to f32 vector for processing
            let mut logits_vec: Vec<f32> = last_logits
                .to_vec1()
                .map_err(|e| RuvLLMError::Generation(format!("Failed to convert logits: {}", e)))?;

            // Apply repetition penalty
            if params.repetition_penalty != 1.0 {
                for &token_id in generated_tokens {
                    if (token_id as usize) < logits_vec.len() {
                        let logit = &mut logits_vec[token_id as usize];
                        if *logit > 0.0 {
                            *logit /= params.repetition_penalty;
                        } else {
                            *logit *= params.repetition_penalty;
                        }
                    }
                }
            }

            // Apply temperature (the only place it is applied; see `logits_processor`)
            if params.temperature >= GREEDY_BELOW && params.temperature != 1.0 {
                for logit in &mut logits_vec {
                    *logit /= params.temperature;
                }
            }

            // Create indexed logits for sorting
            let mut indexed_logits: Vec<(usize, f32)> = logits_vec
                .iter()
                .enumerate()
                .map(|(i, &v)| (i, v))
                .collect();

            indexed_logits
                .sort_by(|a, b| b.1.partial_cmp(&a.1).unwrap_or(std::cmp::Ordering::Equal));

            // Apply top-k filtering
            if params.top_k > 0 && params.top_k < indexed_logits.len() {
                indexed_logits.truncate(params.top_k);
            }

            // Apply top-p (nucleus) sampling
            if params.top_p < 1.0 {
                let max_logit = indexed_logits
                    .iter()
                    .map(|(_, v)| *v)
                    .fold(f32::NEG_INFINITY, f32::max);
                let exp_logits: Vec<f32> = indexed_logits
                    .iter()
                    .map(|(_, v)| (v - max_logit).exp())
                    .collect();
                let sum_exp: f32 = exp_logits.iter().sum();
                let probs: Vec<f32> = exp_logits.iter().map(|e| e / sum_exp).collect();

                let mut cumsum = 0.0;
                let mut cutoff_idx = probs.len();
                for (i, p) in probs.iter().enumerate() {
                    cumsum += p;
                    if cumsum > params.top_p {
                        cutoff_idx = i + 1;
                        break;
                    }
                }
                indexed_logits.truncate(cutoff_idx);
            }

            // Sample from filtered distribution
            let filtered_logits: Vec<f32> = indexed_logits.iter().map(|(_, v)| *v).collect();
            let filtered_tensor =
                Tensor::from_vec(filtered_logits, indexed_logits.len(), &self.device)
                    .map_err(|e| RuvLLMError::Generation(e.to_string()))?;

            let sampled_idx = sampler
                .sample(&filtered_tensor)
                .map_err(|e| RuvLLMError::Generation(format!("Sampling failed: {}", e)))?;

            Ok(indexed_logits[sampled_idx as usize].0 as u32)
        }
    }

    impl LlmBackend for CandleBackend {
        fn load_model(&mut self, model_id: &str, config: ModelConfig) -> Result<()> {
            // Start clean: a previous model's tokenizer must not outlive it,
            // and a failed load must not leave the old model half-replaced.
            self.unload_model();
            let path = Path::new(model_id);

            if path.exists() {
                // Local path
                if path.extension().map_or(false, |e| e == "gguf") {
                    // Direct GGUF file
                    let tokenizer_path = path
                        .parent()
                        .map(|p| p.join("tokenizer.json"))
                        .filter(|p| p.exists());

                    if let Some(tok_path) = tokenizer_path {
                        self.load_tokenizer(&tok_path)?;
                    }
                    self.model_id = model_id.to_string();
                    return self.load_gguf(path, &config);
                } else if path.is_dir() {
                    // Directory with model files
                    let tokenizer_path = path.join("tokenizer.json");
                    if tokenizer_path.exists() {
                        self.load_tokenizer(&tokenizer_path)?;
                        let ruv_tok = RuvTokenizer::from_file(&tokenizer_path)?;
                        let template = gguf_arch::template_from_name(model_id);
                        self.ruv_tokenizer = Some(ruv_tok.with_chat_template(template));
                    }

                    self.model_id = model_id.to_string();

                    // Exactly one GGUF: with several (e.g. two quantizations of
                    // one repo) `read_dir` order would pick one arbitrarily.
                    let mut ggufs: Vec<PathBuf> = std::fs::read_dir(path)
                        .map(|entries| {
                            entries
                                .flatten()
                                .map(|entry| entry.path())
                                .filter(|p| p.extension().map_or(false, |e| e == "gguf"))
                                .collect()
                        })
                        .unwrap_or_default();
                    ggufs.sort();
                    match ggufs.as_slice() {
                        [] => {}
                        [gguf] => return self.load_gguf(gguf, &config),
                        several => {
                            let names: Vec<String> = several
                                .iter()
                                .filter_map(|p| p.file_name()?.to_str().map(str::to_string))
                                .collect();
                            return Err(RuvLLMError::InvalidOperation(format!(
                                "{} holds {} GGUF files ({}); pass the path of the one to load",
                                path.display(),
                                several.len(),
                                names.join(", ")
                            )));
                        }
                    }

                    // Check for safetensors
                    let config_file = path.join("config.json");
                    if !config_file.exists() {
                        return Err(RuvLLMError::NotFound(format!(
                            "config.json not found in {:?}",
                            path
                        )));
                    }

                    // Find safetensors files
                    let mut weights_files = Vec::new();
                    if let Ok(entries) = std::fs::read_dir(path) {
                        for entry in entries.flatten() {
                            let entry_path = entry.path();
                            if entry_path.extension().map_or(false, |e| e == "safetensors") {
                                weights_files.push(entry_path);
                            }
                        }
                    }

                    if weights_files.is_empty() {
                        return Err(RuvLLMError::NotFound(
                            "No .safetensors or .gguf files found".to_string(),
                        ));
                    }

                    weights_files.sort();
                    return self.load_safetensors(&weights_files, &config_file, &config);
                }
            }

            // Treat as HuggingFace Hub model ID (gated, ADR-179 iter 8).
            #[cfg(feature = "hub-download")]
            {
                return self.load_from_hub(model_id, &config);
            }
            #[cfg(not(feature = "hub-download"))]
            {
                Err(RuvLLMError::NotFound(format!(
                    "model_id '{}' not a local path and hub-download feature is disabled (ADR-179 Pi cross-build)",
                    model_id
                )))
            }
        }

        fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
            Ok(self.generate_detailed(prompt, params, &mut |_| true)?.text)
        }

        fn generate_detailed(
            &self,
            prompt: &str,
            params: GenerateParams,
            on_token: &mut dyn FnMut(&GeneratedToken) -> bool,
        ) -> Result<GenerationOutput> {
            // One generation at a time: the KV cache and position are shared.
            // The lock guards no data, so it is safe to take even if an
            // earlier generation panicked while holding it.
            let _generation = self
                .generation_lock
                .lock()
                .unwrap_or_else(|poisoned| poisoned.into_inner());

            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer loaded".to_string()))?;
            let model = self
                .model
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No model loaded".to_string()))?;

            // Clear KV cache for new generation
            self.clear_kv_cache();

            // Encode prompt
            let tokens = tokenizer.encode(prompt)?;
            let prompt_tokens = tokens.len();
            tracing::debug!("Prompt encoded to {} tokens", prompt_tokens);
            if prompt_tokens == 0 {
                return Err(RuvLLMError::Generation(
                    "Prompt encodes to zero tokens".to_string(),
                ));
            }

            let max_ctx = model.config.max_position_embeddings;
            if prompt_tokens >= max_ctx {
                return Err(RuvLLMError::Generation(format!(
                    "Prompt too long: {} tokens exceeds max context {}",
                    prompt_tokens, max_ctx
                )));
            }

            // Process prompt through model
            let input_tensor = Tensor::new(tokens.as_slice(), &self.device)
                .map_err(|e| RuvLLMError::Generation(e.to_string()))?
                .unsqueeze(0)
                .map_err(|e| RuvLLMError::Generation(e.to_string()))?;
            let mut logits = self.forward(&input_tensor, prompt_tokens)?;

            // Text is re-decoded from all generated tokens each step and
            // streamed once settled: a trailing partial UTF-8 sequence and the
            // last (longest stop sequence - 1) bytes are held back, so a stop
            // sequence never reaches `on_token`.
            let holdback = params
                .stop_sequences
                .iter()
                .map(String::len)
                .max()
                .unwrap_or(0)
                .saturating_sub(1);
            let mut generated_tokens: Vec<u32> = Vec::new();
            let mut text = String::new();
            let mut emitted = 0usize;
            let mut completion_tokens = 0usize;
            let mut finish_reason = FinishReason::Length;
            let mut sampler = Self::logits_processor(&params);

            while completion_tokens < params.max_tokens {
                let next_token =
                    self.sample_token(&logits, &params, &generated_tokens, &mut sampler)?;
                completion_tokens += 1;

                if self.stop_token_ids.contains(&next_token) {
                    tracing::debug!(
                        "Stop token {} at position {}",
                        next_token,
                        completion_tokens
                    );
                    finish_reason = FinishReason::EndOfSequence;
                    break;
                }
                generated_tokens.push(next_token);
                text = tokenizer.decode(&generated_tokens)?;

                if let Some(cut) = gguf_arch::find_stop(&text, &params.stop_sequences) {
                    text.truncate(cut);
                    finish_reason = FinishReason::Stop;
                    break;
                }

                let ready = gguf_arch::streamable_len(&text, holdback);
                if let Some(delta) = text.get(emitted..ready).filter(|d| !d.is_empty()) {
                    let piece = GeneratedToken {
                        id: next_token,
                        text: delta.to_string(),
                        logprob: None,
                        is_special: false,
                    };
                    emitted = ready;
                    if !on_token(&piece) {
                        finish_reason = FinishReason::Cancelled;
                        break;
                    }
                }

                // Check max context
                let current_pos = *self.current_pos.lock().expect("current_pos mutex poisoned");
                if current_pos >= max_ctx - 1 {
                    tracing::warn!("Reached max context length");
                    break;
                }

                // Forward pass for next token
                let next_input = Tensor::new(&[next_token], &self.device)
                    .map_err(|e| RuvLLMError::Generation(e.to_string()))?
                    .unsqueeze(0)
                    .map_err(|e| RuvLLMError::Generation(e.to_string()))?;
                logits = self.forward(&next_input, 1)?;
            }

            // Flush what was held back (the stop sequence is already cut off).
            if finish_reason != FinishReason::Cancelled {
                if let Some(rest) = text.get(emitted..).filter(|r| !r.is_empty()) {
                    on_token(&GeneratedToken {
                        id: generated_tokens.last().copied().unwrap_or_default(),
                        text: rest.to_string(),
                        logprob: None,
                        is_special: false,
                    });
                }
            }

            // Record trajectory for SONA learning
            if let Some(ref sona) = self.sona {
                // Create simple embeddings from token statistics
                let query_embedding = Self::simple_embedding(prompt, 768);
                let response_embedding = Self::simple_embedding(&text, 768);

                let trajectory = Trajectory {
                    request_id: format!(
                        "req-{}",
                        std::time::SystemTime::now()
                            .duration_since(std::time::UNIX_EPOCH)
                            .map(|d| d.as_millis())
                            .unwrap_or(0)
                    ),
                    session_id: "default".to_string(),
                    query_embedding,
                    response_embedding,
                    quality_score: 0.8, // Default quality, can be updated with feedback
                    routing_features: vec![
                        completion_tokens as f32 / params.max_tokens.max(1) as f32,
                        params.temperature,
                        params.top_p,
                        0.5, // placeholder
                    ],
                    model_index: 0,
                    timestamp: chrono::Utc::now(),
                };

                if let Err(e) = sona.record_trajectory(trajectory) {
                    tracing::debug!("SONA trajectory recording failed: {}", e);
                } else {
                    tracing::debug!("SONA instant learning triggered");
                }
            }

            Ok(GenerationOutput {
                text,
                prompt_tokens,
                completion_tokens,
                finish_reason,
            })
        }

        fn chat_template(&self) -> Option<ChatTemplate> {
            self.ruv_tokenizer
                .as_ref()
                .and_then(|t| t.chat_template().cloned())
        }

        fn generate_stream(
            &self,
            prompt: &str,
            params: GenerateParams,
        ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
            // Use the new streaming implementation and convert to legacy iterator
            let stream = self.generate_stream_v2(prompt, params)?;

            // Create an adapter that converts StreamEvent to GeneratedToken
            let iter = stream.filter_map(|event_result| match event_result {
                Ok(StreamEvent::Token(token)) => Some(Ok(token)),
                Ok(StreamEvent::Done { .. }) => None,
                Ok(StreamEvent::Error(msg)) => Some(Err(RuvLLMError::Generation(msg))),
                Err(e) => Some(Err(e)),
            });

            Ok(Box::new(iter))
        }

        fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream> {
            // The model is borrowed from `self`, so it cannot move to a worker
            // thread: generation runs to completion here and the stream replays
            // its deltas. For tokens as they are produced, use the
            // `generate_detailed` callback.
            let (tx, stream) = TokenStream::channel();
            let start = Instant::now();
            let output = self.generate_detailed(prompt, params, &mut |token| {
                // The receiver is `stream`, alive until this returns.
                let _ = tx.send(StreamEvent::Token(token.clone()));
                true
            })?;

            let duration_ms = start.elapsed().as_millis() as u64;
            let tokens_per_second = if duration_ms > 0 {
                output.completion_tokens as f64 / (duration_ms as f64 / 1000.0)
            } else {
                0.0
            };
            let _ = tx.send(StreamEvent::Done {
                total_tokens: output.completion_tokens,
                duration_ms,
                tokens_per_second,
            });
            Ok(stream)
        }

        fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
            let tokenizer = self
                .tokenizer
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No tokenizer loaded".to_string()))?;

            let model = self
                .model
                .as_ref()
                .ok_or_else(|| RuvLLMError::InvalidOperation("No model loaded".to_string()))?;

            let _input_ids = tokenizer.encode(text)?;

            // Placeholder - full implementation would extract hidden states
            let hidden_size = model.config.hidden_size;
            let embeddings = vec![0.0f32; hidden_size];

            Ok(embeddings)
        }

        fn tokenizer(&self) -> Option<&dyn Tokenizer> {
            self.tokenizer.as_ref().map(|t| t as &dyn Tokenizer)
        }

        fn is_model_loaded(&self) -> bool {
            self.model.is_some()
        }

        fn model_info(&self) -> Option<ModelInfo> {
            self.model.as_ref().map(|m| m.info.clone())
        }

        fn unload_model(&mut self) {
            self.model = None;
            self.tokenizer = None;
            self.ruv_tokenizer = None;
            self.stop_token_ids.clear();
            self.config = None;
            self.model_id.clear();
            *self.current_pos.lock().expect("current_pos mutex poisoned") = 0;
        }
    }
}

// ============================================================================
// Non-candle stub implementation
// ============================================================================

#[cfg(not(feature = "candle"))]
mod stub_impl {
    use super::*;

    /// Stub tokenizer for when candle is disabled
    pub struct CandleTokenizer {
        vocab_size: usize,
        special_tokens: SpecialTokens,
    }

    impl Default for CandleTokenizer {
        fn default() -> Self {
            Self {
                vocab_size: 32000,
                special_tokens: SpecialTokens::default(),
            }
        }
    }

    impl Tokenizer for CandleTokenizer {
        fn encode(&self, _text: &str) -> Result<Vec<u32>> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn decode(&self, _tokens: &[u32]) -> Result<String> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn vocab_size(&self) -> usize {
            self.vocab_size
        }

        fn special_tokens(&self) -> SpecialTokens {
            self.special_tokens.clone()
        }
    }

    /// Stub backend for when candle is disabled
    pub struct CandleBackend {
        cache_dir: PathBuf,
    }

    impl Default for CandleBackend {
        fn default() -> Self {
            Self {
                cache_dir: get_cache_dir(),
            }
        }
    }

    impl CandleBackend {
        pub fn new() -> Result<Self> {
            Ok(Self::default())
        }

        pub fn with_device(_device_type: DeviceType) -> Result<Self> {
            Ok(Self::default())
        }

        pub fn with_cache_dir(mut self, cache_dir: impl Into<PathBuf>) -> Self {
            self.cache_dir = cache_dir.into();
            self
        }
    }

    impl LlmBackend for CandleBackend {
        fn load_model(&mut self, _model_id: &str, _config: ModelConfig) -> Result<()> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled. Enable with `candle` feature.".to_string(),
            ))
        }

        fn generate(&self, _prompt: &str, _params: GenerateParams) -> Result<String> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn generate_stream(
            &self,
            _prompt: &str,
            _params: GenerateParams,
        ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn generate_stream_v2(
            &self,
            _prompt: &str,
            _params: GenerateParams,
        ) -> Result<TokenStream> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn get_embeddings(&self, _text: &str) -> Result<Vec<f32>> {
            Err(RuvLLMError::Config(
                "Candle feature not enabled".to_string(),
            ))
        }

        fn tokenizer(&self) -> Option<&dyn Tokenizer> {
            None
        }

        fn is_model_loaded(&self) -> bool {
            false
        }

        fn model_info(&self) -> Option<ModelInfo> {
            None
        }

        fn unload_model(&mut self) {}
    }
}

// ============================================================================
// Public re-exports
// ============================================================================

#[cfg(feature = "candle")]
pub use candle_impl::{CandleBackend, CandleTokenizer};

#[cfg(not(feature = "candle"))]
pub use stub_impl::{CandleBackend, CandleTokenizer};

// ============================================================================
// Helper functions
// ============================================================================

/// Get cache directory for models
fn get_cache_dir() -> PathBuf {
    dirs::cache_dir()
        .unwrap_or_else(|| PathBuf::from("."))
        .join("ruvllm")
        .join("models")
}

/// Estimate GGUF model memory usage
fn estimate_gguf_memory(path: &Path) -> Result<usize> {
    let metadata = std::fs::metadata(path)
        .map_err(|e| RuvLLMError::Storage(format!("Failed to read file metadata: {}", e)))?;
    // GGUF file size plus overhead for KV cache and activations
    Ok((metadata.len() as f64 * 1.2) as usize)
}

/// Estimate number of parameters
fn estimate_parameters(hidden_size: usize, num_layers: usize, vocab_size: usize) -> usize {
    let embedding_params = vocab_size * hidden_size;
    // Attention: Q, K, V, O projections = 4 * h * h
    // FFN: For LLaMA-like models with intermediate_size ≈ 3.5 * hidden_size
    // FFN params = 3 * hidden_size * intermediate_size ≈ 10.5 * h²
    // We use 11 * h² / 2 = 5.5 * h² to be conservative
    let attention_params = 4 * hidden_size * hidden_size;
    let intermediate_size = (hidden_size * 7) / 2; // ~3.5x hidden_size
    let ffn_params = 3 * hidden_size * intermediate_size;
    let layer_params = num_layers * (attention_params + ffn_params);
    let output_params = vocab_size * hidden_size;
    embedding_params + layer_params + output_params
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_backend_creation() {
        let backend = CandleBackend::default();
        assert!(!backend.is_model_loaded());
    }

    #[test]
    fn test_model_config_default() {
        let config = ModelConfigInternal::default();
        assert_eq!(config.max_position_embeddings, 4096);
        assert_eq!(config.hidden_size, 4096);
        assert_eq!(config.head_dim, 128);
    }

    #[test]
    fn test_estimate_parameters() {
        // Mistral 7B: hidden_size=4096, layers=32, vocab=32000
        let params = estimate_parameters(4096, 32, 32000);
        // Should be roughly 7-8B (actual is ~7.2B, our estimate includes full embedding + output)
        assert!(params > 6_000_000_000, "params={} should be > 6B", params);
        assert!(params < 9_000_000_000, "params={} should be < 9B", params);
    }

    #[test]
    fn test_get_cache_dir() {
        let cache_dir = get_cache_dir();
        assert!(cache_dir.to_string_lossy().contains("ruvllm"));
    }

    #[test]
    fn test_quantization_is_gguf() {
        assert!(super::Quantization::Q4K.is_gguf());
        assert!(super::Quantization::Q8.is_gguf());
        assert!(!super::Quantization::F16.is_gguf());
    }
}
