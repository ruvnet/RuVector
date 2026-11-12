//! Model definitions and aliases for RuvLLM CLI
//!
//! This module defines the recommended models for different use cases,
//! optimized for Mac M4 Pro with 36GB unified memory.

use serde::{Deserialize, Serialize};
use std::collections::HashMap;

/// Recommended models for RuvLLM on Mac M4 Pro
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ModelDefinition {
    /// HuggingFace model ID
    pub hf_id: String,
    /// Short alias for CLI
    pub alias: String,
    /// Display name
    pub name: String,
    /// Model architecture (mistral, llama, phi, qwen)
    pub architecture: String,
    /// Parameter count in billions
    pub params_b: f32,
    /// Primary use case
    pub use_case: String,
    /// Recommended quantization
    pub recommended_quant: String,
    /// Estimated memory usage in GB (for recommended quant)
    pub memory_gb: f32,
    /// Context length
    pub context_length: usize,
    /// Notes about the model
    pub notes: String,
    /// GGUF "twin" repo for models whose `hf_id` only hosts safetensors.
    /// When a quantized (GGUF) download is requested, weights are resolved
    /// from this repo instead of `hf_id`. `None` means `hf_id` itself hosts
    /// the GGUF files (or no GGUF twin is known).
    #[serde(default)]
    pub gguf_repo: Option<String>,
}

/// Get all recommended models
pub fn get_recommended_models() -> Vec<ModelDefinition> {
    vec![
        // Primary reasoning model
        ModelDefinition {
            hf_id: "Qwen/Qwen2.5-14B-Instruct-GGUF".to_string(),
            alias: "qwen".to_string(),
            name: "Qwen2.5-14B-Instruct".to_string(),
            architecture: "qwen".to_string(),
            params_b: 14.0,
            use_case: "Primary reasoning, code generation, complex tasks".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 9.5,
            context_length: 32768,
            notes: "Best overall performance for reasoning tasks on M4 Pro".to_string(),
            gguf_repo: None,
        },
        // Fast instruction following
        ModelDefinition {
            hf_id: "mistralai/Mistral-7B-Instruct-v0.3".to_string(),
            alias: "mistral".to_string(),
            name: "Mistral-7B-Instruct-v0.3".to_string(),
            architecture: "mistral".to_string(),
            params_b: 7.0,
            use_case: "Fast instruction following, general chat".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 4.5,
            context_length: 32768,
            notes: "Excellent speed/quality tradeoff with sliding window attention".to_string(),
            gguf_repo: Some("bartowski/Mistral-7B-Instruct-v0.3-GGUF".to_string()),
        },
        // Tiny/testing model
        ModelDefinition {
            hf_id: "microsoft/Phi-4-mini-instruct".to_string(),
            alias: "phi".to_string(),
            name: "Phi-4-mini".to_string(),
            architecture: "phi".to_string(),
            params_b: 3.8,
            use_case: "Testing, quick prototyping, resource-constrained".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 2.5,
            context_length: 16384,
            notes: "Surprisingly capable for its size, fast inference".to_string(),
            gguf_repo: Some("bartowski/microsoft_Phi-4-mini-instruct-GGUF".to_string()),
        },
        // Tool use model
        ModelDefinition {
            hf_id: "meta-llama/Llama-3.2-3B-Instruct".to_string(),
            alias: "llama".to_string(),
            name: "Llama-3.2-3B-Instruct".to_string(),
            architecture: "llama".to_string(),
            params_b: 3.2,
            use_case: "Tool use, function calling, structured output".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 2.2,
            context_length: 131072,
            notes: "Optimized for tool use and function calling".to_string(),
            gguf_repo: Some("bartowski/Llama-3.2-3B-Instruct-GGUF".to_string()),
        },
        // Code-specific model
        ModelDefinition {
            hf_id: "Qwen/Qwen2.5-Coder-7B-Instruct-GGUF".to_string(),
            alias: "qwen-coder".to_string(),
            name: "Qwen2.5-Coder-7B-Instruct".to_string(),
            architecture: "qwen".to_string(),
            params_b: 7.0,
            use_case: "Code generation, code review, debugging".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 4.8,
            context_length: 32768,
            notes: "Specialized for coding tasks, excellent at code completion".to_string(),
            gguf_repo: None,
        },
        // Large reasoning model (for when you have the memory)
        ModelDefinition {
            hf_id: "Qwen/Qwen2.5-32B-Instruct-GGUF".to_string(),
            alias: "qwen-large".to_string(),
            name: "Qwen2.5-32B-Instruct".to_string(),
            architecture: "qwen".to_string(),
            params_b: 32.0,
            use_case: "Complex reasoning, research, highest quality output".to_string(),
            recommended_quant: "Q4_K_M".to_string(),
            memory_gb: 20.0,
            context_length: 32768,
            notes: "Requires significant memory, but provides best quality".to_string(),
            gguf_repo: None,
        },
    ]
}

/// Get model by alias or HF ID
pub fn get_model(identifier: &str) -> Option<ModelDefinition> {
    let models = get_recommended_models();

    // First try exact alias match
    if let Some(model) = models.iter().find(|m| m.alias == identifier) {
        return Some(model.clone());
    }

    // Try HF ID match
    if let Some(model) = models.iter().find(|m| m.hf_id == identifier) {
        return Some(model.clone());
    }

    // Try partial HF ID match
    if let Some(model) = models.iter().find(|m| m.hf_id.contains(identifier)) {
        return Some(model.clone());
    }

    None
}

/// Resolve model identifier to HuggingFace ID
pub fn resolve_model_id(identifier: &str) -> String {
    if let Some(model) = get_model(identifier) {
        model.hf_id
    } else {
        // Assume it's a direct HF model ID
        identifier.to_string()
    }
}

/// Resolve the repo that hosts the *weights* for `identifier` at `quant`.
///
/// A quantized (GGUF) request against an alias whose `hf_id` is a
/// safetensors-only repo is routed to the registry's `gguf_repo` twin when one
/// is defined. Repos that already host GGUF files (id contains "GGUF") and
/// unquantized requests resolve as before. Unknown identifiers pass through
/// unchanged — the download command then validates against the actual HF file
/// listing and fails with the available files rather than a 404 on a glob.
pub fn resolve_weights_repo(identifier: &str, quant: QuantPreset) -> String {
    let base = resolve_model_id(identifier);
    if quant == QuantPreset::None || base.to_ascii_uppercase().contains("GGUF") {
        return base;
    }
    if let Some(model) = get_model(identifier) {
        if let Some(gguf_repo) = model.gguf_repo {
            return gguf_repo;
        }
    }
    base
}

/// What to pass to `LlmBackend::load_model` for `model_id` at `quant`.
///
/// `download` stores every quantization of a repo in one directory,
/// `<cache_dir>/models/<model_id>`. When that directory holds several GGUF
/// files, this returns the one matching `quant` (the first part of a split
/// file, which the backend then rejects by name), and none matching is an
/// error listing them rather than a load of another one. A lone GGUF is
/// returned whatever its name; with no GGUF (safetensors) the directory; when
/// nothing is cached, `model_id` itself.
pub fn local_model_source(
    cache_dir: &str,
    model_id: &str,
    quant: QuantPreset,
) -> anyhow::Result<String> {
    let dir = std::path::PathBuf::from(cache_dir)
        .join("models")
        .join(model_id);
    if !dir.is_dir() {
        return Ok(if dir.exists() {
            dir.display().to_string()
        } else {
            model_id.to_string()
        });
    }
    let mut ggufs: Vec<String> = std::fs::read_dir(&dir)?
        .flatten()
        .filter_map(|entry| entry.file_name().into_string().ok())
        .filter(|name| name.to_ascii_lowercase().ends_with(".gguf"))
        .collect();
    if ggufs.is_empty() {
        return Ok(dir.display().to_string());
    }
    ggufs.sort();
    let selected = crate::commands::download::select_gguf_files(model_id, quant, &ggufs);
    // A lone GGUF is loaded whatever its name says (e.g. `model.gguf`), as
    // before; quantization only chooses between several.
    if let [only] = ggufs.as_slice() {
        if selected.is_err() {
            tracing::warn!(
                "{} is the only GGUF in {}; loading it although the requested quantization is {}",
                only,
                dir.display(),
                quant
            );
        }
        return Ok(dir.join(only).display().to_string());
    }
    let selected = selected.map_err(|e| {
        anyhow::anyhow!(
            "{e}\n(in the download cache {}; pass -q/--quantization for one of them, or \
                 the path of a .gguf file)",
            dir.display()
        )
    })?;
    Ok(dir.join(&selected[0]).display().to_string())
}

/// Get model aliases map
pub fn get_aliases() -> HashMap<String, String> {
    get_recommended_models()
        .into_iter()
        .map(|m| (m.alias, m.hf_id))
        .collect()
}

/// Quantization presets
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum QuantPreset {
    /// 4-bit K-quants (best quality/size tradeoff)
    Q4K,
    /// 8-bit quantization (higher quality, more memory)
    Q8,
    /// 16-bit floating point (high quality, most memory)
    F16,
    /// No quantization (full precision)
    None,
}

impl QuantPreset {
    /// Parse from string
    pub fn from_str(s: &str) -> Option<Self> {
        match s.to_lowercase().as_str() {
            "q4k" | "q4_k" | "q4_k_m" | "q4" => Some(Self::Q4K),
            "q8" | "q8_0" => Some(Self::Q8),
            "f16" | "fp16" => Some(Self::F16),
            "none" | "f32" | "fp32" => Some(Self::None),
            _ => None,
        }
    }

    /// Get GGUF file suffix
    pub fn gguf_suffix(&self) -> &'static str {
        match self {
            Self::Q4K => "Q4_K_M.gguf",
            Self::Q8 => "Q8_0.gguf",
            Self::F16 => "F16.gguf",
            Self::None => "F32.gguf",
        }
    }

    /// Quant tags to match against actual GGUF filenames (lowercase).
    /// Repos vary in spelling (e.g. Qwen ships `fp16`, bartowski ships `f16`),
    /// so each preset may accept several tags.
    pub fn gguf_tags(&self) -> &'static [&'static str] {
        match self {
            Self::Q4K => &["q4_k_m"],
            Self::Q8 => &["q8_0"],
            Self::F16 => &["f16", "fp16"],
            Self::None => &["f32", "fp32"],
        }
    }

    /// Get bytes per weight
    pub fn bytes_per_weight(&self) -> f32 {
        match self {
            Self::Q4K => 0.5,
            Self::Q8 => 1.0,
            Self::F16 => 2.0,
            Self::None => 4.0,
        }
    }

    /// Estimate memory usage in GB for given parameter count
    pub fn estimate_memory_gb(&self, params_b: f32) -> f32 {
        // Base memory for weights
        let weight_memory = params_b * self.bytes_per_weight();
        // Add overhead for KV cache, activations, etc. (roughly 20%)
        weight_memory * 1.2
    }
}

impl std::fmt::Display for QuantPreset {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            Self::Q4K => write!(f, "Q4_K_M"),
            Self::Q8 => write!(f, "Q8_0"),
            Self::F16 => write!(f, "F16"),
            Self::None => write!(f, "F32"),
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_get_model_by_alias() {
        let model = get_model("qwen").unwrap();
        assert!(model.hf_id.contains("Qwen2.5-14B"));
    }

    #[test]
    fn test_resolve_model_id() {
        assert!(resolve_model_id("mistral").contains("Mistral-7B"));
        assert_eq!(resolve_model_id("custom/model"), "custom/model");
    }

    #[test]
    fn test_quant_preset() {
        assert_eq!(QuantPreset::from_str("q4k"), Some(QuantPreset::Q4K));
        assert_eq!(QuantPreset::Q4K.bytes_per_weight(), 0.5);
    }

    #[test]
    fn test_weights_repo_gguf_twin_for_safetensors_alias() {
        // `phi` -> microsoft/Phi-4-mini-instruct hosts safetensors only; a
        // quantized request must route to the GGUF twin, not the base repo.
        let repo = resolve_weights_repo("phi", QuantPreset::Q4K);
        assert_eq!(repo, "bartowski/microsoft_Phi-4-mini-instruct-GGUF");
        assert!(repo.contains("GGUF"));
    }

    #[test]
    fn test_weights_repo_unquantized_stays_on_base_repo() {
        assert_eq!(
            resolve_weights_repo("phi", QuantPreset::None),
            "microsoft/Phi-4-mini-instruct"
        );
    }

    #[test]
    fn test_weights_repo_native_gguf_alias_unchanged() {
        assert_eq!(
            resolve_weights_repo("qwen", QuantPreset::Q4K),
            "Qwen/Qwen2.5-14B-Instruct-GGUF"
        );
    }

    #[test]
    fn test_weights_repo_unknown_id_passes_through() {
        assert_eq!(
            resolve_weights_repo("custom/model", QuantPreset::Q4K),
            "custom/model"
        );
    }

    #[test]
    fn test_gguf_tags_cover_repo_spelling_variants() {
        assert!(QuantPreset::F16.gguf_tags().contains(&"fp16"));
        assert_eq!(QuantPreset::Q4K.gguf_tags(), &["q4_k_m"]);
    }

    #[test]
    fn test_local_model_source_honors_the_quantization() {
        let cache = tempfile::tempdir().unwrap();
        let cache_dir = cache.path().to_str().unwrap();
        let repo = "Qwen/Qwen3-0.6B-GGUF";
        let source = |q| local_model_source(cache_dir, repo, q);
        assert_eq!(source(QuantPreset::Q8).unwrap(), repo, "nothing cached");

        let dir = cache.path().join("models").join(repo);
        std::fs::create_dir_all(&dir).unwrap();
        assert_eq!(source(QuantPreset::Q8).unwrap(), dir.display().to_string());

        // A lone GGUF loads whatever its name says, as before.
        let q8 = dir.join("Qwen3-0.6B-Q8_0.gguf").display().to_string();
        std::fs::write(&q8, b"").unwrap();
        assert_eq!(source(QuantPreset::Q4K).unwrap(), q8);
        let renamed = dir.join("model.gguf");
        std::fs::rename(&q8, &renamed).unwrap();
        assert_eq!(
            source(QuantPreset::F16).unwrap(),
            renamed.display().to_string()
        );
        std::fs::remove_file(&renamed).unwrap();

        // With several, the quantization chooses, and no match is an error.
        for name in ["Qwen3-0.6B-Q4_K_M.gguf", "Qwen3-0.6B-Q8_0.gguf"] {
            std::fs::write(dir.join(name), b"").unwrap();
        }
        let q4 = dir.join("Qwen3-0.6B-Q4_K_M.gguf").display().to_string();
        assert_eq!(source(QuantPreset::Q8).unwrap(), q8);
        assert_eq!(source(QuantPreset::Q4K).unwrap(), q4);
        let err = format!("{:#}", source(QuantPreset::F16).unwrap_err());
        assert!(err.contains("Qwen3-0.6B-Q8_0.gguf"), "{err}");
        assert!(err.contains("-q/--quantization"), "{err}");
    }
}
