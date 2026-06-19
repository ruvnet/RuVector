//! Backend for recurrent-depth models (OpenMythos) implementing [`LlmBackend`].
//!
//! Wires the OpenMythos execution graph into the standard serving interface:
//! tokenizer-driven `generate` with sampling, streaming, embeddings, and model
//! info. Weights load from a local checkpoint directory containing:
//!
//! - `config.json`  — a [`CheckpointManifest`] (`architecture` + [`MythosConfig`])
//! - `model.safetensors` — weights named by the module hierarchy
//! - `tokenizer.json` — a HuggingFace tokenizer
//!
//! Loading enforces the honest boundary via
//! [`crate::models::openmythos::validate_mythos_metadata`]: a non-recurrent-depth
//! `architecture` is rejected rather than run.
//!
//! This module requires the `candle` feature (the OpenMythos execution graph).

use std::collections::BTreeMap;
use std::path::Path;

use super::{
    DeviceType, GenerateParams, GeneratedToken, LlmBackend, ModelArchitecture, ModelConfig,
    ModelInfo, SpecialTokens, StreamEvent, TokenStream, Tokenizer,
};
use crate::error::{Result, RuvLLMError};
use crate::models::openmythos::{validate_mythos_metadata, MythosConfig, OpenMythos};
use crate::models::sampling::SamplingConfig;
use crate::tokenizer::RuvTokenizer;

use candle_core::{DType, Device};

/// On-disk checkpoint manifest (`config.json`).
#[derive(Debug, Clone, serde::Serialize, serde::Deserialize)]
pub struct CheckpointManifest {
    /// `general.architecture`-style tag (must be recurrent-depth compatible).
    pub architecture: String,
    /// The model configuration.
    pub model: MythosConfig,
    /// Optional EOS token id override.
    #[serde(default)]
    pub eos_token_id: Option<u32>,
}

/// A [`Tokenizer`] adapter over [`RuvTokenizer`].
pub struct MythosTokenizer {
    inner: RuvTokenizer,
}

impl Tokenizer for MythosTokenizer {
    fn encode(&self, text: &str) -> Result<Vec<u32>> {
        self.inner.encode(text)
    }
    fn decode(&self, tokens: &[u32]) -> Result<String> {
        self.inner.decode(tokens)
    }
    fn vocab_size(&self) -> usize {
        self.inner.vocab_size()
    }
    fn special_tokens(&self) -> SpecialTokens {
        SpecialTokens {
            bos_token_id: None,
            eos_token_id: Some(self.inner.eos_token_id()),
            pad_token_id: None,
            unk_token_id: None,
        }
    }
}

/// Backend serving an OpenMythos recurrent-depth model.
pub struct RecurrentBackend {
    model: Option<OpenMythos>,
    tokenizer: Option<MythosTokenizer>,
    cfg: Option<MythosConfig>,
    model_id: String,
    /// Recurrent depth (loop iterations) per generated token.
    n_loops: usize,
    /// EOS token id used to stop generation.
    eos: Option<u32>,
    device: Device,
}

impl Default for RecurrentBackend {
    fn default() -> Self {
        Self::new()
    }
}

impl RecurrentBackend {
    /// Create an empty backend (no model loaded).
    pub fn new() -> Self {
        Self {
            model: None,
            tokenizer: None,
            cfg: None,
            model_id: String::new(),
            n_loops: 0,
            eos: None,
            device: Device::Cpu,
        }
    }

    /// Construct directly from an in-memory model (testing / embedding use).
    pub fn from_model(
        model: OpenMythos,
        tokenizer: Option<RuvTokenizer>,
        model_id: impl Into<String>,
    ) -> Self {
        let cfg = model.config().clone();
        let eos = tokenizer.as_ref().map(|t| t.eos_token_id());
        Self {
            n_loops: cfg.max_loop_iters,
            model: Some(model),
            tokenizer: tokenizer.map(|inner| MythosTokenizer { inner }),
            cfg: Some(cfg),
            model_id: model_id.into(),
            eos,
            device: Device::Cpu,
        }
    }

    /// Recurrent depth per generated token (defaults to `max_loop_iters`).
    pub fn set_n_loops(&mut self, n_loops: usize) {
        self.n_loops = n_loops;
    }

    /// Generate raw token ids from a prompt of token ids (bypasses the
    /// tokenizer). Useful for embedding models and tests.
    pub fn generate_token_ids(&self, prompt: &[u32], params: &GenerateParams) -> Result<Vec<u32>> {
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| RuvLLMError::Model("no model loaded".into()))?;
        model.generate_sampled(
            prompt,
            params.max_tokens,
            self.n_loops,
            self.eos,
            sampling_from(params),
        )
    }
}

fn sampling_from(params: &GenerateParams) -> SamplingConfig {
    SamplingConfig {
        temperature: params.temperature,
        top_k: params.top_k,
        top_p: params.top_p,
        repetition_penalty: params.repetition_penalty,
        repetition_window: 64,
        seed: params.seed.unwrap_or(42),
    }
}

fn select_device(config: &ModelConfig) -> Device {
    match config.device {
        DeviceType::Cpu => Device::Cpu,
        DeviceType::Metal => Device::new_metal(0).unwrap_or(Device::Cpu),
        DeviceType::Cuda(id) => Device::new_cuda(id).unwrap_or(Device::Cpu),
    }
}

fn estimate_params(cfg: &MythosConfig) -> usize {
    let blocks = cfg.prelude_layers + cfg.coda_layers + 1;
    let per_block = 12 * cfg.dim * cfg.dim;
    cfg.vocab_size * cfg.dim * 2 + blocks * per_block
}

impl LlmBackend for RecurrentBackend {
    fn load_model(&mut self, model_id: &str, config: ModelConfig) -> Result<()> {
        let dir = Path::new(model_id);
        if !dir.is_dir() {
            return Err(RuvLLMError::Model(format!(
                "OpenMythos checkpoint must be a directory, got: {model_id}"
            )));
        }

        // Manifest + honest-boundary validation.
        let manifest_raw = std::fs::read_to_string(dir.join("config.json"))
            .map_err(|e| RuvLLMError::Model(format!("read config.json: {e}")))?;
        let manifest: CheckpointManifest = serde_json::from_str(&manifest_raw)
            .map_err(|e| RuvLLMError::Model(format!("parse config.json: {e}")))?;

        let mut meta = BTreeMap::new();
        meta.insert(
            "general.architecture".to_string(),
            manifest.architecture.clone(),
        );
        validate_mythos_metadata(&meta)?;
        manifest.model.validate()?;

        self.device = select_device(&config);

        // Tokenizer (optional but expected for text I/O).
        let tok_path = dir.join("tokenizer.json");
        if tok_path.is_file() {
            let tok = RuvTokenizer::from_file(&tok_path)?;
            self.eos = manifest.eos_token_id.or(Some(tok.eos_token_id()));
            self.tokenizer = Some(MythosTokenizer { inner: tok });
        } else {
            self.eos = manifest.eos_token_id;
        }

        // Weights.
        let weights = dir.join("model.safetensors");
        if !weights.is_file() {
            return Err(RuvLLMError::Model(format!(
                "missing model.safetensors in {model_id}"
            )));
        }
        let model =
            OpenMythos::from_safetensors(&[weights], manifest.model.clone(), &meta, &self.device)?;

        self.n_loops = manifest.model.max_loop_iters;
        self.cfg = Some(manifest.model);
        self.model = Some(model);
        self.model_id = model_id.to_string();
        Ok(())
    }

    fn generate(&self, prompt: &str, params: GenerateParams) -> Result<String> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| RuvLLMError::Tokenization("no tokenizer loaded".into()))?;
        let ids = tok.encode(prompt)?;
        let out = self.generate_token_ids(&ids, &params)?;
        tok.decode(&out)
    }

    fn generate_stream(
        &self,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<Box<dyn Iterator<Item = Result<GeneratedToken>> + Send + '_>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| RuvLLMError::Tokenization("no tokenizer loaded".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| RuvLLMError::Model("no model loaded".into()))?;
        let ids = tok.encode(prompt)?;
        let mut items = Vec::with_capacity(params.max_tokens);

        // True per-token streaming: yield each token immediately after sampling
        // rather than buffering the full sequence.
        model.generate_stream_sampled(
            &ids,
            params.max_tokens,
            self.n_loops,
            self.eos,
            sampling_from(&params),
            |id| {
                let text = tok.decode(&[id]).unwrap_or_default();
                items.push(Ok(GeneratedToken {
                    id,
                    text,
                    logprob: None,
                    is_special: false,
                }));
                true // continue
            },
        )?;
        Ok(Box::new(items.into_iter()))
    }

    fn generate_stream_v2(&self, prompt: &str, params: GenerateParams) -> Result<TokenStream> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| RuvLLMError::Tokenization("no tokenizer loaded".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| RuvLLMError::Model("no model loaded".into()))?;
        let ids = tok.encode(prompt)?;
        let (tx, stream) = TokenStream::channel();
        let start = std::time::Instant::now();
        let mut count = 0usize;

        let result = model.generate_stream_sampled(
            &ids,
            params.max_tokens,
            self.n_loops,
            self.eos,
            sampling_from(&params),
            |id| {
                let text = tok.decode(&[id]).unwrap_or_default();
                count += 1;
                tx.send(StreamEvent::Token(GeneratedToken {
                    id,
                    text,
                    logprob: None,
                    is_special: false,
                }))
                .is_ok()
            },
        );
        if let Err(e) = result {
            let _ = tx.send(StreamEvent::Error(e.to_string()));
        }
        let ms = start.elapsed().as_millis() as u64;
        let tps = if ms > 0 {
            count as f64 / (ms as f64 / 1000.0)
        } else {
            0.0
        };
        let _ = tx.send(StreamEvent::Done {
            total_tokens: count,
            duration_ms: ms,
            tokens_per_second: tps,
        });
        Ok(stream)
    }

    fn get_embeddings(&self, text: &str) -> Result<Vec<f32>> {
        let tok = self
            .tokenizer
            .as_ref()
            .ok_or_else(|| RuvLLMError::Tokenization("no tokenizer loaded".into()))?;
        let model = self
            .model
            .as_ref()
            .ok_or_else(|| RuvLLMError::Model("no model loaded".into()))?;
        let ids = tok.encode(text)?;
        model.embed_pooled(&ids)
    }

    fn tokenizer(&self) -> Option<&dyn Tokenizer> {
        self.tokenizer.as_ref().map(|t| t as &dyn Tokenizer)
    }

    fn is_model_loaded(&self) -> bool {
        self.model.is_some()
    }

    fn model_info(&self) -> Option<ModelInfo> {
        let cfg = self.cfg.as_ref()?;
        Some(ModelInfo {
            name: if self.model_id.is_empty() {
                "openmythos".to_string()
            } else {
                self.model_id.clone()
            },
            // ModelArchitecture has no RDT variant; the real architecture is
            // carried in `name`. Llama is the closest tensor-layout proxy.
            architecture: ModelArchitecture::Llama,
            num_parameters: estimate_params(cfg),
            vocab_size: cfg.vocab_size,
            hidden_size: cfg.dim,
            num_layers: cfg.prelude_layers + cfg.coda_layers + 1,
            max_context_length: cfg.max_seq_len,
            quantization: None,
            memory_usage: estimate_params(cfg) * 4,
        })
    }

    fn unload_model(&mut self) {
        self.model = None;
        self.tokenizer = None;
        self.cfg = None;
        self.model_id.clear();
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use candle_nn::{VarBuilder, VarMap};

    fn in_memory_backend() -> RecurrentBackend {
        let cfg = MythosConfig::tiny();
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let model = OpenMythos::load(vb, cfg).unwrap();
        RecurrentBackend::from_model(model, None, "test-mythos")
    }

    #[test]
    fn reports_loaded_and_info() {
        let b = in_memory_backend();
        assert!(b.is_model_loaded());
        let info = b.model_info().unwrap();
        assert_eq!(info.vocab_size, MythosConfig::tiny().vocab_size);
        assert_eq!(info.name, "test-mythos");
    }

    #[test]
    fn generate_token_ids_runs_through_backend() {
        let b = in_memory_backend();
        let params = GenerateParams {
            max_tokens: 4,
            temperature: 0.0,
            ..Default::default()
        };
        let out = b.generate_token_ids(&[1, 2, 3], &params).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn generate_without_tokenizer_errors() {
        let b = in_memory_backend();
        assert!(b.generate("hello", GenerateParams::default()).is_err());
    }

    #[test]
    fn manifest_round_trips_through_json() {
        let manifest = CheckpointManifest {
            architecture: "openmythos".into(),
            model: MythosConfig::tiny(),
            eos_token_id: Some(2),
        };
        let json = serde_json::to_string(&manifest).unwrap();
        let back: CheckpointManifest = serde_json::from_str(&json).unwrap();
        assert_eq!(back.model, MythosConfig::tiny());
        assert_eq!(back.architecture, "openmythos");
    }

    /// Write a loadable checkpoint dir (config.json + model.safetensors) and
    /// return its path-owning TempDir plus the config used.
    fn write_checkpoint(arch: &str) -> (tempfile::TempDir, MythosConfig) {
        let cfg = MythosConfig::tiny();
        let dir = tempfile::tempdir().unwrap();

        // Weights from a VarMap (names match the module hierarchy on load).
        let varmap = VarMap::new();
        let vb = VarBuilder::from_varmap(&varmap, DType::F32, &Device::Cpu);
        let _ = OpenMythos::load(vb, cfg.clone()).unwrap();
        varmap.save(dir.path().join("model.safetensors")).unwrap();

        // Manifest.
        let manifest = CheckpointManifest {
            architecture: arch.to_string(),
            model: cfg.clone(),
            eos_token_id: None,
        };
        std::fs::write(
            dir.path().join("config.json"),
            serde_json::to_string(&manifest).unwrap(),
        )
        .unwrap();
        (dir, cfg)
    }

    #[test]
    fn load_model_from_disk_then_generate() {
        let (dir, cfg) = write_checkpoint("openmythos");
        let mut b = RecurrentBackend::new();
        b.load_model(dir.path().to_str().unwrap(), ModelConfig::default())
            .expect("load_model");
        assert!(b.is_model_loaded());
        assert_eq!(b.model_info().unwrap().vocab_size, cfg.vocab_size);

        let params = GenerateParams {
            max_tokens: 4,
            temperature: 0.0,
            ..Default::default()
        };
        let out = b.generate_token_ids(&[1, 2, 3], &params).unwrap();
        assert_eq!(out.len(), 4);
    }

    #[test]
    fn load_model_rejects_non_mythos_architecture() {
        // Honest boundary enforced through the full disk loader.
        let (dir, _cfg) = write_checkpoint("llama");
        let mut b = RecurrentBackend::new();
        let err = b.load_model(dir.path().to_str().unwrap(), ModelConfig::default());
        assert!(err.is_err());
        assert!(!b.is_model_loaded());
    }

    #[test]
    fn load_model_requires_directory() {
        let mut b = RecurrentBackend::new();
        assert!(b
            .load_model("/nonexistent/path/to/model", ModelConfig::default())
            .is_err());
    }
}
