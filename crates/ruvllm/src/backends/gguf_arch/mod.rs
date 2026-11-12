//! GGUF architecture dispatch and tokenizer metadata for the candle backend.
//!
//! `general.architecture` picks the candle-transformers weight loader. Each
//! loader reads its own tensor names and `<arch>.*` metadata keys, so a file
//! whose architecture has no loader is rejected by name instead of being
//! handed to a loader that cannot read it.

use std::collections::HashMap;

use candle_core::quantized::gguf_file::Value;
use tokenizers::Tokenizer as HfTokenizer;

use super::{ModelArchitecture, Quantization};
use crate::error::{Result, RuvLLMError};
use crate::tokenizer::ChatTemplate;

/// Architectures the candle backend can load from GGUF, as listed in errors.
pub(crate) const SUPPORTED_GGUF_ARCHITECTURES: &str = "llama, mistral, qwen2, qwen3";

/// End-of-text / end-of-turn tokens that end generation whenever the
/// vocabulary has them. Instruct models end a turn with a different token than
/// the base model's EOS (Qwen: `<|im_end|>` vs `<|endoftext|>`).
const STOP_TOKENS: &[&str] = &[
    "<|im_end|>",
    "<|endoftext|>",
    "<|eot_id|>",
    "<|end_of_text|>",
    "</s>",
];

/// Weight layout of a GGUF file, chosen from `general.architecture`.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum GgufArch {
    /// `llama`, `mistral`, or no architecture key: `quantized_llama`.
    Llama,
    /// `qwen2` (Qwen2, Qwen2.5): `quantized_qwen2` (q/k/v biases).
    Qwen2,
    /// `qwen3`: `quantized_qwen3` (q/k norms, head dim from `key_length`).
    Qwen3,
}

impl GgufArch {
    /// Select the loader for a file's metadata. Split files and architectures
    /// without a loader are errors that name what was found.
    pub(crate) fn from_metadata(md: &HashMap<String, Value>) -> Result<Self> {
        // llama.cpp writes split.count as u16; to_u64 upcasts it.
        if let Some(parts) = md
            .get("split.count")
            .and_then(|v| v.to_u64().ok())
            .filter(|&n| n > 1)
        {
            return Err(RuvLLMError::Model(format!(
                "split GGUF ({parts} parts) is not supported; merge the parts with \
                 `llama-gguf-split --merge` or pick a single-file quantization"
            )));
        }
        let arch = md
            .get("general.architecture")
            .and_then(|v| v.to_string().ok())
            .cloned()
            .unwrap_or_default();
        match arch.to_lowercase().as_str() {
            "" | "llama" | "mistral" => Ok(Self::Llama),
            "qwen2" => Ok(Self::Qwen2),
            "qwen3" => Ok(Self::Qwen3),
            _ => Err(RuvLLMError::Model(format!(
                "GGUF architecture '{arch}' is not supported by the candle backend \
                 (supported: {SUPPORTED_GGUF_ARCHITECTURES})"
            ))),
        }
    }

    /// Architecture to report in `ModelInfo`; the file wins over the caller's
    /// hint except for the llama family, which shares one loader.
    pub(crate) fn model_architecture(self, configured: ModelArchitecture) -> ModelArchitecture {
        match self {
            Self::Llama => configured,
            Self::Qwen2 | Self::Qwen3 => ModelArchitecture::Qwen,
        }
    }

    /// `general.architecture` value this loader reads, for messages.
    fn name(self) -> &'static str {
        match self {
            Self::Llama => "llama",
            Self::Qwen2 => "qwen2",
            Self::Qwen3 => "qwen3",
        }
    }

    /// Chat template the file determines, from its embedded
    /// `tokenizer.chat_template` (and, for Qwen, its vocabulary).
    ///
    /// `Ok(None)`: a llama-family file whose template is absent or not one
    /// ruvllm renders; the caller guesses from the model name. A qwen2/qwen3
    /// file is ChatML only when its template (if any) is ChatML and its
    /// vocabulary has `<|im_start|>`/`<|im_end|>`; anything else is an error,
    /// because Qwen-architecture fine-tunes such as DeepSeek-R1-Distill-Qwen
    /// use their own prompt format and produce garbage under ChatML.
    pub(crate) fn chat_template(
        self,
        md: &HashMap<String, Value>,
        has_chatml_tokens: bool,
    ) -> Result<Option<ChatTemplate>> {
        let embedded = md
            .get("tokenizer.chat_template")
            .and_then(|v| v.to_string().ok())
            .map(String::as_str);
        if self == Self::Llama {
            return Ok(embedded.and_then(classify_chat_template));
        }
        let problem = match embedded {
            Some(t) if !t.contains("<|im_start|>") => {
                "its embedded chat template (tokenizer.chat_template) is not ChatML"
            }
            _ if !has_chatml_tokens => "its vocabulary has no <|im_start|>/<|im_end|> tokens",
            _ => return Ok(Some(ChatTemplate::Qwen)),
        };
        Err(RuvLLMError::Model(format!(
            "unsupported chat format for GGUF architecture '{}': {problem}. ruvllm renders \
             {} prompts only as ChatML (<|im_start|>role ... <|im_end|>); fine-tunes with \
             their own format (e.g. DeepSeek-R1-Distill-Qwen) are not supported",
            self.name(),
            self.name()
        )))
    }
}

/// The template an embedded Jinja chat template renders, when it is one ruvllm
/// implements.
fn classify_chat_template(template: &str) -> Option<ChatTemplate> {
    if template.contains("<|im_start|>") {
        Some(ChatTemplate::ChatML)
    } else if template.contains("<|start_header_id|>") {
        Some(ChatTemplate::Llama3)
    } else if template.contains("[INST]") {
        Some(if template.contains("<<SYS>>") {
            ChatTemplate::Llama2
        } else {
            ChatTemplate::Mistral
        })
    } else {
        None
    }
}

/// Guess a chat template from the last component of a model id or path, so
/// that a parent directory (`/home/phil/...`) cannot select a template.
pub(crate) fn template_from_name(model_id: &str) -> ChatTemplate {
    let name = model_id
        .trim_end_matches(['/', '\\'])
        .rsplit(['/', '\\'])
        .next()
        .unwrap_or(model_id);
    ChatTemplate::detect_from_model_id(name)
}

/// Weight quantization a GGUF declares in `general.file_type` (llama.cpp's
/// `LLAMA_FTYPE_*`). `hint` (the caller's request) only when the file does not
/// say; `None` when it declares a type with no [`Quantization`] variant
/// (e.g. Q5_K, Q6_K, IQ*), rather than misreport it.
pub(crate) fn file_quantization(
    md: &HashMap<String, Value>,
    hint: Option<Quantization>,
) -> Option<Quantization> {
    let Some(file_type) = md.get("general.file_type").and_then(|v| v.to_u32().ok()) else {
        return hint;
    };
    match file_type {
        0 => Some(Quantization::None),
        1 => Some(Quantization::F16),
        2 | 3 => Some(Quantization::Q4),
        7 => Some(Quantization::Q8),
        10 => Some(Quantization::Q2K),
        14 | 15 => Some(Quantization::Q4K),
        32 => Some(Quantization::Bf16),
        _ => None,
    }
}

/// Stop tokens a vocabulary has, from [`STOP_TOKENS`].
pub(crate) fn vocab_stop_ids(tokenizer: &HfTokenizer) -> Vec<u32> {
    STOP_TOKENS
        .iter()
        .filter_map(|t| tokenizer.token_to_id(t))
        .collect()
}

/// Special-token ids a GGUF declares in `tokenizer.ggml.*_token_id`.
#[derive(Debug, Default, PartialEq, Eq)]
pub(crate) struct GgufSpecialIds {
    pub bos: Option<u32>,
    pub eos: Option<u32>,
    /// `eos` plus the end-of-turn and end-of-message ids, when declared
    pub stop: Vec<u32>,
}

impl GgufSpecialIds {
    /// Read the ids, keeping only those inside a `vocab_size` vocabulary.
    pub(crate) fn from_metadata(md: &HashMap<String, Value>, vocab_size: usize) -> Self {
        let id = |key: &str| {
            md.get(key)
                .and_then(|v| {
                    let signed = || v.to_i32().ok().and_then(|i| u64::try_from(i).ok());
                    v.to_u64().ok().or_else(signed)
                })
                .and_then(|id| u32::try_from(id).ok())
                .filter(|&id| (id as usize) < vocab_size)
        };
        let eos = id("tokenizer.ggml.eos_token_id");
        let stop = [
            eos,
            id("tokenizer.ggml.eot_token_id"),
            id("tokenizer.ggml.eom_token_id"),
        ]
        .into_iter()
        .flatten()
        .collect();
        Self {
            bos: id("tokenizer.ggml.bos_token_id"),
            eos,
            stop,
        }
    }
}

/// Byte length of the prefix of `text` that can be streamed: everything but
/// a trailing U+FFFD run (an incomplete UTF-8 sequence the next token may
/// complete) and the last `holdback` bytes before it (a possible
/// stop-sequence prefix; the pending bytes may complete one).
pub(crate) fn streamable_len(text: &str, holdback: usize) -> usize {
    let settled = text.trim_end_matches('\u{FFFD}').len();
    let mut end = settled.saturating_sub(holdback);
    while !text.is_char_boundary(end) {
        end -= 1;
    }
    end
}

/// Byte offset of the earliest stop sequence in `text`; empty sequences never match.
pub(crate) fn find_stop(text: &str, stop_sequences: &[String]) -> Option<usize> {
    stop_sequences
        .iter()
        .filter(|s| !s.is_empty())
        .filter_map(|s| text.find(s.as_str()))
        .min()
}

#[cfg(test)]
mod tests;
