//! The language model behind `loadModel`, `generate`, `chat` and `query`.
//!
//! With the `candle` feature this holds a real [`ruvllm_lib::CandleBackend`]
//! loaded from a GGUF file. Without it, loading fails with an error saying the
//! binary cannot load models. In neither case is any text produced without a
//! loaded model: generation returns [`NO_MODEL`] instead.

use napi::bindgen_prelude::*;

use super::{
    JsChatMessage, JsGenerationConfig, JsGenerationResult, JsLoadModelOptions, JsModelInfo,
};

/// Error for `generate`/`chat`/`query` before a model is loaded. The JS
/// wrapper matches on the `RUVLLM_NO_LANGUAGE_MODEL` prefix.
pub(super) const NO_MODEL: &str = "RUVLLM_NO_LANGUAGE_MODEL: no language model is loaded; \
                                   call loadModel(path) with a GGUF file first";

fn no_model() -> Error {
    Error::from_reason(NO_MODEL)
}

#[cfg(feature = "candle")]
mod imp {
    use super::*;
    use parking_lot::RwLock;
    use ruvllm_lib::{
        CandleBackend, ChatMessage, ChatTemplate, DeviceType, FinishReason, GenerateParams,
        GenerationOutput, LlmBackend, ModelArchitecture, ModelConfig, Role,
    };
    use std::path::Path;

    /// Default context window when `loadModel` is not given `maxContext`.
    const DEFAULT_MAX_CONTEXT: usize = 4096;

    fn invalid(msg: impl Into<String>) -> Error {
        Error::new(Status::InvalidArg, msg.into())
    }

    fn backend_error(context: &str, e: ruvllm_lib::RuvLLMError) -> Error {
        Error::from_reason(format!("{context}: {e}"))
    }

    fn to_u32(n: usize) -> u32 {
        u32::try_from(n).unwrap_or(u32::MAX)
    }

    /// Holds at most one loaded model. Loading takes the write lock;
    /// generation takes a read lock (the backend serialises decoding itself).
    #[derive(Default)]
    pub struct ModelSlot(RwLock<Option<CandleBackend>>);

    impl ModelSlot {
        pub fn load(&self, path: &str, opts: &JsLoadModelOptions) -> Result<JsModelInfo> {
            let max_context = match opts.max_context {
                Some(0) => return Err(invalid("maxContext must be at least 1")),
                Some(n) => n as usize,
                None => DEFAULT_MAX_CONTEXT,
            };
            // Checked here because the backend treats a path that does not
            // exist as a HuggingFace Hub id, which this binding cannot fetch
            // (built without `hub-download`): a typo would surface as a
            // confusing hub error.
            if !Path::new(path).exists() {
                return Err(Error::from_reason(format!(
                    "RUVLLM_MODEL_NOT_FOUND: no such file or directory: {path} \
                     (loadModel takes a local GGUF file or a directory holding one)"
                )));
            }

            let mut slot = self.0.write();
            // Free the previous model before reading the next one; a failed
            // load therefore leaves no model loaded rather than a stale one.
            *slot = None;

            let device = if cfg!(feature = "metal") {
                DeviceType::Metal
            } else {
                DeviceType::Cpu
            };
            let mut backend = CandleBackend::with_device(device)
                .map_err(|e| backend_error("RUVLLM_MODEL_LOAD_FAILED", e))?;
            // A hint only (a GGUF's own `general.architecture` wins), taken
            // from the file or directory name: matching on the whole path
            // would let a parent directory such as `/home/phil/` say "phi".
            let name = Path::new(path)
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or(path);
            let config = ModelConfig {
                architecture: ModelArchitecture::detect_from_model_id(name).unwrap_or_default(),
                max_sequence_length: max_context,
                device,
                ..Default::default()
            };
            backend
                .load_model(path, config)
                .map_err(|e| backend_error(&format!("RUVLLM_MODEL_LOAD_FAILED: {path}"), e))?;
            // The backend loads weights even when it found no usable
            // tokenizer (e.g. a llama GGUF with a SentencePiece vocabulary
            // and no tokenizer.json); fail now rather than on every call.
            if backend.tokenizer().is_none() {
                return Err(Error::from_reason(format!(
                    "RUVLLM_TOKENIZER_MISSING: {path}: the weights loaded but no tokenizer is \
                     available; the tokenizer embedded in the GGUF could not be used, so put the \
                     model's tokenizer.json in the same directory as the model file"
                )));
            }
            let info = model_info(&backend).ok_or_else(|| {
                Error::from_reason(format!(
                    "RUVLLM_MODEL_LOAD_FAILED: {path}: the backend reported no model after loading"
                ))
            })?;
            *slot = Some(backend);
            Ok(info)
        }

        pub fn unload(&self) {
            *self.0.write() = None;
        }

        pub fn is_loaded(&self) -> bool {
            self.0.read().is_some()
        }

        pub fn info(&self) -> Option<JsModelInfo> {
            self.0.read().as_ref().and_then(model_info)
        }

        pub fn ensure_loaded(&self) -> Result<()> {
            if self.is_loaded() {
                Ok(())
            } else {
                Err(no_model())
            }
        }

        pub fn generate(
            &self,
            prompt: &str,
            cfg: &JsGenerationConfig,
        ) -> Result<JsGenerationResult> {
            let slot = self.0.read();
            let backend = slot.as_ref().ok_or_else(no_model)?;
            let params = to_params(cfg)?;
            run(backend, prompt, params)
        }

        pub fn chat(
            &self,
            messages: &[JsChatMessage],
            cfg: &JsGenerationConfig,
        ) -> Result<JsGenerationResult> {
            let slot = self.0.read();
            let backend = slot.as_ref().ok_or_else(no_model)?;
            if messages.is_empty() {
                return Err(invalid("chat needs at least one message"));
            }
            let messages = messages
                .iter()
                .map(to_chat_message)
                .collect::<Result<Vec<_>>>()?;
            let params = to_params(cfg)?;
            let prompt = backend
                .apply_chat_template(&messages)
                .map_err(|e| backend_error("RUVLLM_CHAT_TEMPLATE_FAILED", e))?;
            run(backend, &prompt, params)
        }
    }

    fn run(
        backend: &CandleBackend,
        prompt: &str,
        params: GenerateParams,
    ) -> Result<JsGenerationResult> {
        let out = backend
            .generate_detailed(prompt, params, &mut |_| true)
            .map_err(|e| backend_error("RUVLLM_GENERATION_FAILED", e))?;
        to_result(out)
    }

    fn to_result(out: GenerationOutput) -> Result<JsGenerationResult> {
        // OpenAI semantics: an end-of-sequence token and a stop sequence are
        // both "stop"; hitting max tokens or a full context is "length".
        let finish_reason = match out.finish_reason {
            FinishReason::EndOfSequence | FinishReason::Stop => "stop",
            FinishReason::Length => "length",
            FinishReason::Cancelled => "cancelled",
            FinishReason::Error => {
                return Err(Error::from_reason(
                    "RUVLLM_GENERATION_FAILED: the backend reported an error",
                ))
            }
        };
        Ok(JsGenerationResult {
            text: out.text,
            prompt_tokens: to_u32(out.prompt_tokens),
            completion_tokens: to_u32(out.completion_tokens),
            finish_reason: finish_reason.to_string(),
        })
    }

    fn to_params(cfg: &JsGenerationConfig) -> Result<GenerateParams> {
        let max_tokens = cfg.max_tokens.unwrap_or(256);
        let temperature = cfg.temperature.unwrap_or(0.7);
        let top_p = cfg.top_p.unwrap_or(0.9);
        let repetition_penalty = cfg.repetition_penalty.unwrap_or(1.1);
        if max_tokens == 0 {
            return Err(invalid("maxTokens must be at least 1"));
        }
        if !(temperature.is_finite() && temperature >= 0.0) {
            return Err(invalid("temperature must be a finite number >= 0"));
        }
        if !(top_p.is_finite() && top_p > 0.0 && top_p <= 1.0) {
            return Err(invalid("topP must be in (0, 1]"));
        }
        if !(repetition_penalty.is_finite() && repetition_penalty > 0.0) {
            return Err(invalid("repetitionPenalty must be a finite number > 0"));
        }
        let stop_sequences = cfg.stop_sequences.clone().unwrap_or_default();
        if stop_sequences.iter().any(String::is_empty) {
            return Err(invalid("stopSequences must not contain empty strings"));
        }
        Ok(GenerateParams {
            max_tokens: max_tokens as usize,
            temperature: temperature as f32,
            top_p: top_p as f32,
            top_k: cfg.top_k.unwrap_or(50) as usize,
            repetition_penalty: repetition_penalty as f32,
            stop_sequences,
            seed: cfg.seed.map(u64::from),
            ..GenerateParams::default()
        })
    }

    fn to_chat_message(m: &JsChatMessage) -> Result<ChatMessage> {
        let role = match m.role.as_str() {
            "system" => Role::System,
            "user" => Role::User,
            "assistant" => Role::Assistant,
            other => {
                return Err(invalid(format!(
                    "unknown chat role '{other}' (expected system, user or assistant)"
                )))
            }
        };
        Ok(ChatMessage::new(role, m.content.clone()))
    }

    fn model_info(backend: &CandleBackend) -> Option<JsModelInfo> {
        let info = backend.model_info()?;
        let chat_template = backend.chat_template().map(|t| match t {
            ChatTemplate::Custom(_) => "Custom".to_string(),
            named => format!("{named:?}"),
        });
        Some(JsModelInfo {
            name: info.name,
            architecture: format!("{:?}", info.architecture),
            num_parameters: info.num_parameters as f64,
            vocab_size: to_u32(info.vocab_size),
            hidden_size: to_u32(info.hidden_size),
            num_layers: to_u32(info.num_layers),
            max_context_length: to_u32(info.max_context_length),
            quantization: info.quantization.map(|q| format!("{q:?}")),
            memory_usage_bytes: info.memory_usage as f64,
            chat_template,
        })
    }
}

#[cfg(not(feature = "candle"))]
mod imp {
    use super::*;

    const NO_BACKEND: &str = "RUVLLM_NO_INFERENCE_BACKEND: this native binary was built \
                              without the `candle` feature and cannot load models; rebuild \
                              it with `--features napi,candle`";

    /// Stand-in for builds without an inference backend: nothing can be
    /// loaded, so every generation call fails.
    #[derive(Default)]
    pub struct ModelSlot;

    impl ModelSlot {
        pub fn load(&self, _path: &str, _opts: &JsLoadModelOptions) -> Result<JsModelInfo> {
            Err(Error::from_reason(NO_BACKEND))
        }

        pub fn unload(&self) {}

        pub fn is_loaded(&self) -> bool {
            false
        }

        pub fn info(&self) -> Option<JsModelInfo> {
            None
        }

        pub fn ensure_loaded(&self) -> Result<()> {
            Err(no_model())
        }

        pub fn generate(
            &self,
            _prompt: &str,
            _cfg: &JsGenerationConfig,
        ) -> Result<JsGenerationResult> {
            Err(no_model())
        }

        pub fn chat(
            &self,
            _messages: &[JsChatMessage],
            _cfg: &JsGenerationConfig,
        ) -> Result<JsGenerationResult> {
            Err(no_model())
        }
    }
}

pub(super) use imp::ModelSlot;
